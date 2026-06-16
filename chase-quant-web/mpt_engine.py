"""
MPT Portfolio Optimization Engine — Markowitz + Black-Litterman + HRP
=====================================================================
Chase量化策略 Phase 15: 从"选币打分"升级到"组合数学优化"

核心理念:
  - 单币得分高 ≠ 组合最优 — 相关性决定分散化效果
  - 马科维茨有效边界 → 给定风险下最大化收益
  - Black-Litterman → 把 AI 观点融入数学权重
  - HRP → 协方差奇异时的稳健备选

引擎能力:
  1. 最大夏普比率优化 (Riskfolio-Lib + Ledoit-Wolf 收缩)
  2. 最小风险优化 (Riskfolio-Lib)
  3. 风险平价 (Risk Parity — 等风险贡献)
  4. 层级风险平价 (HRP — 不需要逆协方差矩阵)
  5. Black-Litterman (AI观点 → P/Q矩阵 → 后验收益 → 最优权重)
  6. 组合诊断 (条件数 / VaR / 风险贡献 / 集中度 / 有效N)
  7. 再平衡建议 (考虑换手率约束)

依赖: riskfolio-lib, sklearn, scipy, numpy, pandas
与 asset_graph.py + portfolio.py 全链路集成

使用:
  from mpt_engine import MPTPortfolioOptimizer, ensemble_to_bl_views

  # 从 asset_graph 获取价格和协方差
  builder = AssetGraphBuilder()
  snapshot = builder.build(["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"])

  # MPT 优化
  opt = MPTPortfolioOptimizer(
      prices_df=builder.close_prices,
      graph_snapshot=snapshot,
      risk_profile="moderate",
      risk_free_rate=0.02,
  )

  # 最大夏普
  result = opt.optimize_max_sharpe()

  # 或用 AI 观点做 Black-Litterman
  views = {"BTC/USDT": 0.15, "ETH/USDT": 0.05, "SOL/USDT": -0.10}
  result = opt.optimize_black_litterman(views)

  # 再平衡指令
  instructions = opt.rebalance_check(
      current_holdings={"BTC/USDT": 2000, "ETH/USDT": 1500},
      optimal_weights=result["weights"],
      total_capital=10000,
  )
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf, OAS

# ── Riskfolio-Lib (主力引擎) ──
try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"
MPT_DIR = DATA_DIR / "mpt_optimizations"
MPT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════
# 风险偏好配置
# ═══════════════════════════════════════════
RISK_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "label": "🛡️ 保守型",
        "risk_aversion": 0.8,          # 高风险厌恶
        "max_single_asset": 0.30,      # 单币上限 30%
        "min_assets": 5,               # 至少持有 5 个币
        "target_vol": 0.25,            # 目标年化波动 25% (crypto)
        "max_drawdown_limit": 0.20,    # 最大回撤容忍 20%
        "rebalance_threshold": 0.05,   # 权重偏离 5% 触发再平衡
    },
    "moderate": {
        "label": "⚖️ 均衡型 (Chase哥默认)",
        "risk_aversion": 0.6,
        "max_single_asset": 0.40,
        "min_assets": 3,
        "target_vol": 0.45,
        "max_drawdown_limit": 0.35,
        "rebalance_threshold": 0.08,
    },
    "aggressive": {
        "label": "🚀 激进型",
        "risk_aversion": 0.3,
        "max_single_asset": 0.55,
        "min_assets": 2,
        "target_vol": 0.70,
        "max_drawdown_limit": 0.50,
        "rebalance_threshold": 0.12,
    },
}


# ═══════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════

@dataclass
class MPTResult:
    """MPT 优化结果"""
    method: str                          # "max_sharpe" | "min_risk" | "risk_parity" | "hrp" | "black_litterman"
    weights: Dict[str, float]            # 资产名 → 权重
    expected_return: float               # 年化期望收益
    expected_volatility: float           # 年化波动率
    sharpe_ratio: float                  # 夏普比率
    condition_number: float              # 协方差矩阵条件数
    risk_contributions: Dict[str, float] # 各资产风险贡献
    diversification_score: float         # 分散化评分 0-100
    effective_n: float                   # 有效资产数
    herfindahl: float                    # 集中度指数
    var_95: float                        # 95% VaR
    cvar_95: float                       # 95% CVaR

    # 可选
    efficient_frontier_points: Optional[List[Dict]] = None
    bl_posterior_returns: Optional[Dict[str, float]] = None
    bl_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "condition_number": self.condition_number,
            "diversification_score": self.diversification_score,
            "effective_n": self.effective_n,
            "herfindahl": self.herfindahl,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "warnings": self.warnings,
        }
        if self.efficient_frontier_points:
            d["efficient_frontier_points"] = self.efficient_frontier_points
        return d


# ═══════════════════════════════════════════
# Black-Litterman: AI观点 → 数学权重
# ═══════════════════════════════════════════

def ensemble_to_bl_views(
    ensemble_scores: Dict[str, float],
    min_confidence: float = 0.3,
    max_confidence: float = 0.8,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    将 Multi-LLM Ensemble 评分转换为 Black-Litterman 观点

    Args:
        ensemble_scores: {资产: 评分} — 来自 Multi-LLM Ensemble
        min_confidence: 最低置信度
        max_confidence: 最高置信度

    Returns:
        views: {资产: 预期超额收益 (小数)}
        confidences: {资产: 置信度 0-1}
    """
    views = {}
    confidences = {}

    scores = pd.Series(ensemble_scores)
    if len(scores) == 0:
        return views, confidences

    # 评分归一化到 [-1, 1]
    score_mean = scores.mean()
    score_std = scores.std()
    if score_std < 1e-8:
        return views, confidences

    z_scores = (scores - score_mean) / score_std

    for asset, z in z_scores.items():
        # 默认为市场均衡 (0观点)，只有强信号才表达观点
        if abs(z) < 0.5:
            continue

        # 观点强度: z=1 → ~10% 年化超额, z=3 → ~50%
        view_strength = np.clip(z * 0.15, -0.60, 0.60)
        views[asset] = float(view_strength)

        # 置信度: 基于 z-score 绝对值和 Ensemble 一致性
        confidence = min_confidence + (max_confidence - min_confidence) * min(abs(z) / 2.5, 1.0)
        confidences[asset] = float(confidence)

    return views, confidences


def _build_bl_matrices(
    assets: List[str],
    views: Dict[str, float],
    confidences: Dict[str, float],
    market_cap_weights: Optional[Dict[str, float]] = None,
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 Black-Litterman 的 P, Q, Omega 矩阵

    Returns:
        pi: 隐含均衡收益 (N,)
        P: 观点选择矩阵 (K, N)
        Q: 观点收益向量 (K,)
        Omega: 观点不确定性对角阵 (K, K)
        tau: 均衡收益不确定性标量
    """
    n = len(assets)
    asset_to_idx = {a: i for i, a in enumerate(assets)}

    # P 矩阵: 每个观点一行
    view_assets = [a for a in assets if a in views]
    k = len(view_assets)
    if k == 0:
        return None, None, None, None, None

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for row, asset in enumerate(view_assets):
        col = asset_to_idx[asset]
        P[row, col] = 1.0
        Q[row] = views[asset]
        # 置信度越高 → Omega 越小 → 观点越确定
        conf = confidences.get(asset, 0.5)
        omega_diag[row] = tau / max(conf, 0.01)

    Omega = np.diag(omega_diag)

    # 隐含均衡收益 π (简化: 等权或市值加权)
    if market_cap_weights:
        w_eq = np.array([market_cap_weights.get(a, 1.0 / n) for a in assets])
    else:
        w_eq = np.ones(n) / n

    return w_eq, P, Q, Omega


# ═══════════════════════════════════════════
# 主引擎: MPTPortfolioOptimizer
# ═══════════════════════════════════════════

class MPTPortfolioOptimizer:
    """
    MPT 组合优化引擎

    支持 5 种优化方法:
      - max_sharpe: 最大夏普比率 (Riskfolio-Lib, Classic + MV)
      - min_risk: 最小风险 (Riskfolio-Lib, Classic + MV)
      - risk_parity: 风险平价 (Riskfolio-Lib, Classic + MV)
      - hrp: 层级风险平价 (Riskfolio-Lib HCPortfolio)
      - black_litterman: Black-Litterman (scipy SLSQP + 后验收益)

    Fallback 链: Riskfolio-Lib → scipy SLSQP → 等权
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        risk_profile: str = "moderate",
        graph_snapshot: Any = None,
        risk_free_rate: float = 0.02,
        ema_decay: float = 0.94,
        lookback_window: Optional[int] = None,
    ):
        """
        Args:
            prices_df: 价格 DataFrame, columns=资产名, index=时间
            risk_profile: "conservative" | "moderate" | "aggressive"
            graph_snapshot: AssetGraphBuilder 的 GraphSnapshot (可选, 复用协方差)
            risk_free_rate: 无风险利率 (年化)
            ema_decay: EMA 衰减因子 (越近期权重越高)
            lookback_window: 回看窗口 (None=全部)
        """
        self.risk_free_rate = risk_free_rate
        self.profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])
        self.risk_profile_name = risk_profile
        self.ema_decay = ema_decay

        # 价格 & 收益
        self.prices = prices_df.copy()
        if lookback_window:
            self.prices = self.prices.iloc[-lookback_window:]

        self.returns = self.prices.pct_change().dropna()
        self.assets = list(self.returns.columns)
        self.n = len(self.assets)
        self.t = len(self.returns)

        # EMA 时间衰减权重
        t = self.t
        self._time_weights = np.array([ema_decay ** (t - i) for i in range(t)])
        self._time_weights = self._time_weights / self._time_weights.sum()

        # 协方差矩阵 — 优先从 graph_snapshot 取, 否则自行计算
        if graph_snapshot is not None and hasattr(graph_snapshot, 'adj_raw'):
            self._build_from_snapshot(graph_snapshot)
        else:
            self._build_covariance()

        # Riskfolio Portfolio 对象
        self._rp = None
        if RISKFOLIO_AVAILABLE:
            try:
                self._rp = rp.Portfolio(returns=self.returns)
                self._rp.assets = self.assets
            except Exception:
                pass

    def _build_from_snapshot(self, snapshot):
        """从 AssetGraphBuilder 的 GraphSnapshot 提取协方差"""
        try:
            # 尝试取 adj_raw 中的 Pearson 相关矩阵转协方差
            adj_raw = snapshot.adj_raw
            if "pearson" in adj_raw:
                corr = np.array(adj_raw["pearson"])
                # 对齐资产
                snap_assets = snapshot.symbols
                if set(snap_assets) == set(self.assets):
                    # 按 self.assets 顺序重排
                    idx_map = {s: i for i, s in enumerate(snap_assets)}
                    reorder = [idx_map.get(a) for a in self.assets]
                    if all(r is not None for r in reorder):
                        corr = corr[np.ix_(reorder, reorder)]
                # 相关矩阵 → 协方差矩阵
                stds = self.returns.std().values * np.sqrt(365)
                D = np.diag(stds)
                self.Sigma = pd.DataFrame(
                    D @ corr @ D,
                    index=self.assets,
                    columns=self.assets,
                )
                self._corr = pd.DataFrame(corr, index=self.assets[:len(corr)], columns=self.assets[:len(corr)])
                self._condition_number = float(np.linalg.cond(self.Sigma.values))
                return
        except Exception:
            pass

        # Fallback
        self._build_covariance()

    def _build_covariance(self):
        """Ledoit-Wolf 收缩协方差"""
        try:
            cov_shrunk, _ = ledoit_wolf(self.returns)
            # 年化
            self.Sigma = pd.DataFrame(
                cov_shrunk * 365,
                index=self.assets,
                columns=self.assets,
            )
            # 相关矩阵
            diag_sqrt = np.sqrt(np.diag(cov_shrunk))
            self._corr = pd.DataFrame(
                cov_shrunk / np.outer(diag_sqrt, diag_sqrt),
                index=self.assets,
                columns=self.assets,
            )
        except Exception:
            # 回退到样本协方差
            self.Sigma = self.returns.cov() * 365
            self._corr = self.returns.corr()

        self._condition_number = float(np.linalg.cond(self.Sigma.values))

        # 期望收益 (EMA 加权年化)
        weighted_returns = self.returns.mul(self._time_weights, axis=0)
        self.mu = weighted_returns.sum() * 365

    @property
    def condition_number(self) -> float:
        return self._condition_number

    @property
    def is_near_singular(self) -> bool:
        """协方差是否接近奇异 (条件数 > 100 建议用 HRP)"""
        return self._condition_number > 100

    # ── 最大夏普比率 ──
    def optimize_max_sharpe(self) -> MPTResult:
        """优化最大夏普比率 (Riskfolio-Lib 主力, scipy fallback)"""
        result = self._try_riskfolio("max_sharpe")
        if result is not None:
            return result
        return self._scipy_max_sharpe()

    def _try_riskfolio(self, method: str) -> Optional[MPTResult]:
        """尝试用 Riskfolio-Lib 优化"""
        if self._rp is None:
            return None

        try:
            if method == "max_sharpe":
                w = self._rp.optimization(
                    model='Classic', rm='MV', obj='Sharpe',
                    rf=self.risk_free_rate, l=self.profile['risk_aversion'],
                    hist=False,
                )
            elif method == "min_risk":
                w = self._rp.optimization(
                    model='Classic', rm='MV', obj='MinRisk',
                    rf=self.risk_free_rate, l=self.profile['risk_aversion'],
                    hist=False,
                )
            elif method == "risk_parity":
                w = self._rp.optimization(
                    model='Classic', rm='MV', obj='RiskParity',
                    rf=self.risk_free_rate, l=self.profile['risk_aversion'],
                    hist=False,
                )
            else:
                return None

            weights = w['weights'].to_dict() if hasattr(w['weights'], 'to_dict') else dict(w['weights'])
            return self._build_result(method, weights)

        except Exception:
            return None

    def _scipy_max_sharpe(self) -> MPTResult:
        """scipy SLSQP fallback: 最大夏普比率"""
        n = self.n
        mu = self.mu.values
        Sigma = self.Sigma.values
        rf = self.risk_free_rate

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            return -(port_ret - rf) / max(port_vol, 1e-10)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, self.profile['max_single_asset']) for _ in range(n)]

        # 多起点尝试
        best_w = None
        best_val = float('inf')
        for init in [np.ones(n) / n, np.random.dirichlet(np.ones(n))]:
            try:
                res = minimize(neg_sharpe, init, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_w = res.x
            except Exception:
                continue

        if best_w is None:
            best_w = np.ones(n) / n

        weights = {a: float(w) for a, w in zip(self.assets, best_w)}
        return self._build_result("max_sharpe", weights)

    # ── 最小风险 ──
    def optimize_min_risk(self) -> MPTResult:
        result = self._try_riskfolio("min_risk")
        if result is not None:
            return result

        # scipy fallback
        n = self.n
        Sigma = self.Sigma.values

        def port_vol(w):
            return np.sqrt(w @ Sigma @ w)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.profile['max_single_asset']) for _ in range(n)]

        res = minimize(port_vol, np.ones(n) / n, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        weights = {a: float(w) for a, w in zip(self.assets, res.x)}
        return self._build_result("min_risk", weights)

    # ── 风险平价 ──
    def optimize_risk_parity(self) -> MPTResult:
        result = self._try_riskfolio("risk_parity")
        if result is not None:
            return result

        # scipy fallback: 等风险贡献
        n = self.n
        Sigma = self.Sigma.values

        def risk_parity_objective(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            mrc = Sigma @ w / max(port_vol, 1e-10)
            rc = w * mrc
            target_rc = port_vol / n
            return np.sum((rc - target_rc) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.profile['max_single_asset']) for _ in range(n)]

        res = minimize(risk_parity_objective, np.ones(n) / n, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        weights = {a: float(w) for a, w in zip(self.assets, res.x)}
        return self._build_result("risk_parity", weights)

    # ── HRP (层级风险平价) ──
    def optimize_hrp(self) -> MPTResult:
        """HRP — 不需要逆协方差，协方差奇异时最稳健"""
        if RISKFOLIO_AVAILABLE:
            try:
                hrp = rp.HCPortfolio(returns=self.returns)
                w = hrp.optimization(
                    model='HRP', rm='MV', rf=self.risk_free_rate,
                    linkage='ward', max_k=min(10, self.n), hist=False,
                )
                weights = w['weights'].to_dict() if hasattr(w['weights'], 'to_dict') else dict(w['weights'])
                return self._build_result("hrp", weights)
            except Exception:
                pass

        # Fallback: 手动 HRP
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # 距离矩阵
        corr = self._corr.values
        dist = np.sqrt(2 * (1 - corr))
        condensed = squareform(dist, checks=False)

        # 层次聚类
        try:
            Z = linkage(condensed, method='ward')
            clusters = fcluster(Z, t=min(10, self.n), criterion='maxclust')
        except Exception:
            clusters = np.arange(1, self.n + 1)

        # 逆波动率加权 (cluster-level), 然后等权
        vols = self.returns.std().values * np.sqrt(365)
        inv_vols = 1.0 / np.maximum(vols, 1e-10)

        # 每个 cluster 内等波动率加权
        unique_clusters = np.unique(clusters)
        cluster_weights = {}
        for c in unique_clusters:
            mask = clusters == c
            cluster_vol = np.sqrt(inv_vols[mask] @ self.Sigma.values[np.ix_(mask, mask)] @ inv_vols[mask])
            cluster_weights[c] = 1.0 / max(cluster_vol, 1e-10)

        # 归一化
        total = sum(cluster_weights.values())
        for c in cluster_weights:
            cluster_weights[c] /= max(total, 1e-10)

        # 分配给各资产
        weights = {}
        for i, a in enumerate(self.assets):
            c = clusters[i]
            c_mask = clusters == c
            c_inv_vol_sum = inv_vols[c_mask].sum()
            weights[a] = float(cluster_weights[c] * (inv_vols[i] / max(c_inv_vol_sum, 1e-10)))

        # 再次归一化
        w_sum = sum(weights.values())
        weights = {a: w / max(w_sum, 1e-10) for a, w in weights.items()}

        return self._build_result("hrp", weights)

    # ── Black-Litterman ──
    def optimize_black_litterman(
        self,
        views: Dict[str, float],
        view_confidences: Optional[Dict[str, float]] = None,
        market_cap_weights: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
    ) -> MPTResult:
        """
        Black-Litterman 优化 — 融合 AI 观点与市场均衡

        Args:
            views: {资产: 预期年化超额收益} — 来自 LLM Ensemble
            view_confidences: {资产: 置信度 0-1} — 来自 LLM 投票一致性
            market_cap_weights: 市值权重 (None=等权)
            tau: 均衡收益不确定性
        """
        n = self.n
        Sigma = self.Sigma.values
        rf = self.risk_free_rate

        # 过滤出有效观点
        valid_views = {}
        valid_confs = {}
        for a in self.assets:
            if a in views:
                valid_views[a] = views[a]
                valid_confs[a] = view_confidences.get(a, 0.5) if view_confidences else 0.5

        if not valid_views:
            # 无有效观点 → fallback to max sharpe
            result = self.optimize_max_sharpe()
            result.method = "black_litterman"
            result.warnings.append("无有效AI观点, 退回最大夏普优化")
            return result

        # 构建 BL 矩阵
        w_eq, P, Q, Omega = _build_bl_matrices(
            self.assets, valid_views, valid_confs,
            market_cap_weights, tau,
        )

        if P is None:
            result = self.optimize_max_sharpe()
            result.method = "black_litterman"
            return result

        # 后验收益: E[R] = [(τΣ)^(-1) + P^T Ω^(-1) P]^(-1) [(τΣ)^(-1) Π + P^T Ω^(-1) Q]
        try:
            tau_Sigma_inv = np.linalg.inv(tau * Sigma)
            Omega_inv = np.linalg.inv(Omega)

            M = tau_Sigma_inv + P.T @ Omega_inv @ P
            posterior_mu = np.linalg.solve(M, tau_Sigma_inv @ w_eq + P.T @ Omega_inv @ Q)
        except np.linalg.LinAlgError:
            # 数值不稳定 → 用 HRP
            result = self.optimize_hrp()
            result.warnings.append("BL后验计算数值不稳定, 退回HRP")
            return result

        # 后验协方差: Σ_post = Σ + [(τΣ)^(-1) + P^T Ω^(-1) P]^(-1)
        try:
            posterior_Sigma = Sigma + np.linalg.inv(M)
        except np.linalg.LinAlgError:
            posterior_Sigma = Sigma

        # 用后验参数做最大夏普优化
        def neg_sharpe_post(w):
            ret = w @ posterior_mu
            vol = np.sqrt(w @ posterior_Sigma @ w)
            return -(ret - rf) / max(vol, 1e-10)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.profile['max_single_asset']) for _ in range(n)]

        best_w = None
        best_val = float('inf')
        for init in [np.ones(n) / n, w_eq]:
            try:
                res = minimize(neg_sharpe_post, init, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_w = res.x
            except Exception:
                continue

        if best_w is None:
            best_w = np.ones(n) / n

        weights = {a: float(w) for a, w in zip(self.assets, best_w)}
        result = self._build_result("black_litterman", weights)

        # 记录后验信息
        result.bl_posterior_returns = {a: float(r) for a, r in zip(self.assets, posterior_mu)}
        avg_conf = np.mean(list(valid_confs.values())) if valid_confs else 0.5
        result.bl_confidence = float(avg_conf)

        return result

    # ── 有效边界 ──
    def efficient_frontier(self, n_points: int = 50) -> List[Dict]:
        """计算有效边界点 (用于前端绘制)"""
        n = self.n
        mu = self.mu.values
        Sigma = self.Sigma.values
        rf = self.risk_free_rate

        # 先找最小风险和最大收益组合
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n)]

        # 最小风险
        res_min = minimize(lambda w: np.sqrt(w @ Sigma @ w),
                           np.ones(n) / n, method='SLSQP',
                           bounds=bounds, constraints=constraints)
        min_vol = np.sqrt(res_min.x @ Sigma @ res_min.x)
        min_ret = res_min.x @ mu

        # 最大收益 (单资产)
        max_ret = mu.max()

        # 在 [min_ret, max_ret] 范围内采样目标收益
        target_rets = np.linspace(min_ret, max_ret * 0.9, n_points)
        frontier = []

        for target_ret in target_rets:
            constraints_eff = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target_ret: w @ mu - t},
            ]

            res = minimize(lambda w: np.sqrt(w @ Sigma @ w),
                           np.ones(n) / n, method='SLSQP',
                           bounds=bounds, constraints=constraints_eff,
                           options={'maxiter': 500, 'ftol': 1e-8})

            if res.success:
                w = res.x
                port_vol = np.sqrt(w @ Sigma @ w)
                port_ret = w @ mu
                sharpe = (port_ret - rf) / max(port_vol, 1e-10)
                frontier.append({
                    "volatility": float(port_vol),
                    "return": float(port_ret),
                    "sharpe": float(sharpe),
                })

        return frontier

    # ── 组合诊断 ──
    def comprehensive_diagnosis(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        诊断组合健康度

        Returns:
            dict with:
              - condition_number: 协方差条件数
              - var_95: 95% 日 VaR
              - cvar_95: 95% 日 CVaR
              - risk_contribution: 各资产风险贡献
              - herfindahl: 集中度指数
              - effective_n: 有效资产数
              - diversification_score: 分散化评分 0-100
              - recommendation: 建议策略
        """
        w = np.array([weights.get(a, 0) for a in self.assets])
        Sigma = self.Sigma.values
        port_vol = np.sqrt(w @ Sigma @ w)

        # 风险贡献
        mrc = Sigma @ w / max(port_vol, 1e-10)  # marginal risk contribution
        rc = w * mrc  # risk contribution
        rc_pct = rc / max(rc.sum(), 1e-10)

        # 集中度
        herfindahl = float(np.sum(w ** 2))
        effective_n = float(1.0 / max(herfindahl, 1e-10))
        n = self.n
        diversification_score = float(min(100, effective_n / max(n, 1) * 100))

        # VaR / CVaR
        port_ret = float(w @ self.mu.values)
        daily_vol = port_vol / np.sqrt(365)
        var_95 = float(port_ret / 365 - 1.645 * daily_vol)
        cvar_95 = float(port_ret / 365 - 2.063 * daily_vol)

        # 建议
        recommendations = []
        if self._condition_number > 100:
            recommendations.append("🔴 协方差条件数过高, 推荐用 HRP 替代")
        elif self._condition_number > 50:
            recommendations.append("🟡 协方差条件数偏高, 建议使用 Ledoit-Wolf 收缩")

        if herfindahl > 0.5:
            recommendations.append("🔴 高度集中, 建议增加资产分散化")
        elif herfindahl > 0.3:
            recommendations.append("🟡 中度集中")

        if effective_n < 3:
            recommendations.append("🟡 有效资产数 < 3, 分散化不足")

        max_single = max(weights.values()) if weights else 0
        if max_single > self.profile["max_single_asset"]:
            recommendations.append(f"🔴 {max(max_single, key=lambda x: x[1]) if isinstance(max_single, dict) else ''} 超单币上限 {self.profile['max_single_asset']:.0%}")

        return {
            "condition_number": self._condition_number,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "risk_contribution": {a: float(rc_pct[i]) for i, a in enumerate(self.assets)},
            "herfindahl": herfindahl,
            "effective_n": effective_n,
            "diversification_score": diversification_score,
            "recommendations": recommendations,
        }

    # ── 再平衡检查 ──
    def rebalance_check(
        self,
        current_holdings_usdt: Dict[str, float],
        optimal_weights: Dict[str, float],
        total_capital: float,
        min_trade: float = 50.0,
        max_turnover: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        生成再平衡指令

        Args:
            current_holdings_usdt: {资产: 当前USDT价值}
            optimal_weights: {资产: 目标权重}
            total_capital: 总资金
            min_trade: 最小交易金额 (低于此值不交易)
            max_turnover: 最大换手率 (None=不限制)

        Returns:
            dict with 'instructions' list and 'summary'
        """
        instructions = []
        total_turnover = 0.0

        for asset in optimal_weights:
            target_value = optimal_weights[asset] * total_capital
            current_value = current_holdings_usdt.get(asset, 0)
            diff = target_value - current_value
            diff_pct = diff / max(total_capital, 1)

            if abs(diff) < min_trade:
                continue

            action = "BUY" if diff > 0 else "SELL"
            instructions.append({
                "asset": asset,
                "action": action,
                "amount_usdt": abs(diff),
                "weight_change": float(diff_pct),
                "current_weight": float(current_value / max(total_capital, 1)),
                "target_weight": float(optimal_weights[asset]),
            })
            total_turnover += abs(diff_pct)

        # 换手率约束
        if max_turnover is not None and total_turnover > max_turnover:
            scale = max_turnover / total_turnover
            for inst in instructions:
                inst["amount_usdt"] *= scale
                inst["weight_change"] *= scale
            total_turnover = max_turnover

        # 按 trade 金额排序 (大的在前)
        instructions.sort(key=lambda x: x["amount_usdt"], reverse=True)

        return {
            "instructions": instructions,
            "total_turnover": float(total_turnover),
            "n_trades": len(instructions),
            "total_capital": total_capital,
        }

    # ── 内部工具 ──
    def _build_result(self, method: str, weights: Dict[str, float]) -> MPTResult:
        """从权重构建完整 MPTResult"""
        w = np.array([weights.get(a, 0) for a in self.assets])
        Sigma = self.Sigma.values
        mu = self.mu.values
        rf = self.risk_free_rate

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ Sigma @ w))
        sharpe = float((port_ret - rf) / max(port_vol, 1e-10))

        # 诊断
        diag = self.comprehensive_diagnosis(weights)

        # 计算有效边界 (仅对主要方法)
        ef_points = None
        if method in ("max_sharpe", "min_risk"):
            try:
                ef_points = self.efficient_frontier(30)
            except Exception:
                pass

        warnings_list = diag.get("recommendations", [])

        return MPTResult(
            method=method,
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            condition_number=self._condition_number,
            risk_contributions=diag.get("risk_contribution", {}),
            diversification_score=diag.get("diversification_score", 50),
            effective_n=diag.get("effective_n", self.n),
            herfindahl=diag.get("herfindahl", 0.0),
            var_95=diag.get("var_95", 0),
            cvar_95=diag.get("cvar_95", 0),
            efficient_frontier_points=ef_points,
            warnings=warnings_list,
        )

    # ── 便捷 API: 一键全方法对比 ──
    def compare_all_methods(self, bl_views: Optional[Dict] = None,
                            bl_confs: Optional[Dict] = None) -> Dict[str, MPTResult]:
        """运行所有优化方法并返回对比"""
        results = {}

        results["max_sharpe"] = self.optimize_max_sharpe()

        if not self.is_near_singular:
            results["min_risk"] = self.optimize_min_risk()
            results["risk_parity"] = self.optimize_risk_parity()
        else:
            results["min_risk"] = None
            results["risk_parity"] = None

        results["hrp"] = self.optimize_hrp()

        if bl_views:
            results["black_litterman"] = self.optimize_black_litterman(bl_views, bl_confs)

        return results

    def compare_summary(self, results: Dict[str, MPTResult]) -> pd.DataFrame:
        """生成方法对比 DataFrame"""
        rows = []
        for method, r in results.items():
            if r is None:
                rows.append({"方法": method, "状态": "❌ 不可用 (协方差奇异)"})
                continue
            rows.append({
                "方法": {
                    "max_sharpe": "📈 最大夏普",
                    "min_risk": "🛡️ 最小风险",
                    "risk_parity": "⚖️ 风险平价",
                    "hrp": "🌳 HRP",
                    "black_litterman": "🧠 Black-Litterman",
                }.get(method, method),
                "预期收益": f"{r.expected_return:+.1%}",
                "波动率": f"{r.expected_volatility:.1%}",
                "夏普比率": f"{r.sharpe_ratio:.2f}",
                "有效N": f"{r.effective_n:.1f}",
                "分散化": f"{r.diversification_score:.0f}/100",
                "条件数": f"{r.condition_number:.0f}",
                "⚠️": len(r.warnings),
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# 与现有系统的集成桥接
# ═══════════════════════════════════════════

def optimize_from_portfolio(
    pf: Any,  # PortfolioManager
    symbols: List[str],
    risk_profile: str = "moderate",
) -> Optional[MPTResult]:
    """
    从 PortfolioManager 持仓出发做 MPT 优化
    自动获取价格数据并优化
    """
    try:
        import ccxt
        exchange = ccxt.binance()

        prices = {}
        for sym in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, '1d', limit=365)
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)
                prices[sym] = df['close']
            except Exception:
                continue

        if len(prices) < 2:
            return None

        prices_df = pd.DataFrame(prices).dropna()

        opt = MPTPortfolioOptimizer(
            prices_df=prices_df,
            risk_profile=risk_profile,
        )

        return opt.optimize_max_sharpe()

    except Exception as e:
        print(f"[MPT] optimize_from_portfolio error: {e}")
        return None
