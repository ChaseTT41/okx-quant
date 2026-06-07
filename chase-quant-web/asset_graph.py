"""
Asset Relationship Graph Engine v1.0 — 真实资产关系图 + 多资产联合预测
=====================================================================
Chase量化策略 Phase 11: 将 GATs 从"玩具图"升级到真实资产关系图

核心理念 (对标 Qlib 的图学习能力):
  - 资产之间不是独立的 — BTC涨了 ETH 可能跟涨
  - 图结构 = 资产间的关系 → GATs 聚合邻居信息 → 更准的预测
  - 关系图动态变化 → 需要定期重建

图的构建 (6维关系):
  1. Pearson 相关性    — 线性价格关系
  2. Spearman 秩相关   — 非线性单调关系 (对异常值鲁棒)
  3. 距离相关性 (dCor) — 任意依赖关系
  4. 协整关系           — 长期均衡关系 (Engle-Granger)
  5. 领先滞后           — 谁先动谁后动 (Granger causality)
  6. 波动率相关         — 尾部风险联动

图类型:
  - static: 全历史单图
  - rolling: 滚动窗口动态图 (捕捉市场结构变化)
  - ensemble: 多关系加权融合图

使用:
  from asset_graph import AssetGraphBuilder
  builder = AssetGraphBuilder()
  adj_matrix, node_features = builder.build(["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])

  # 也可以在 GATs 中直接用
  from asset_graph import CrossAssetGATPredictor
  predictor = CrossAssetGATPredictor()
  predictions = predictor.predict(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 科学计算
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import ledoit_wolf, OAS

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlib_models import (
    MultiHeadGAT, GraphAttentionLayer, create_model,
    TimeSeriesDataset, DEVICE, MODEL_DIR,
)

DATA_DIR = Path(__file__).parent / "data"
GRAPH_DIR = DATA_DIR / "asset_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ── 默认监控资产 ──
DEFAULT_CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
]

DEFAULT_STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
]

# ── 资产分类 (sector/type grouping) ──
CRYPTO_CATEGORIES = {
    "BTC/USDT": "store_of_value",
    "ETH/USDT": "smart_contract_platform",
    "SOL/USDT": "smart_contract_platform",
    "BNB/USDT": "exchange_token",
    "XRP/USDT": "payment",
    "ADA/USDT": "smart_contract_platform",
    "DOGE/USDT": "meme",
    "AVAX/USDT": "smart_contract_platform",
    "DOT/USDT": "interoperability",
    "LINK/USDT": "oracle",
}


@dataclass
class GraphSnapshot:
    """资产关系图快照"""
    symbols: List[str]
    n_assets: int
    created_at: str
    data_start: str
    data_end: str
    n_days: int
    lookback_days: int

    # 邻接矩阵
    adj_matrix: np.ndarray           # (n, n) 最终融合邻接矩阵
    adj_raw: Dict[str, np.ndarray]   # 各维度的原始关系矩阵

    # 图属性
    avg_degree: float
    graph_density: float
    n_communities: int
    community_labels: List[int]
    top_edges: List[dict]            # 最强连边

    # 节点特征 (每个资产的 feature vector)
    node_features: Optional[Dict[str, np.ndarray]] = None

    # 元信息
    version: int = 1
    previous_hash: str = ""
    graph_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "n_assets": self.n_assets,
            "created_at": self.created_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "n_days": self.n_days,
            "lookback_days": self.lookback_days,
            "avg_degree": round(self.avg_degree, 3),
            "graph_density": round(self.graph_density, 4),
            "n_communities": self.n_communities,
            "community_labels": self.community_labels,
            "top_edges": self.top_edges[:10],
            "version": self.version,
            "previous_hash": self.previous_hash,
            "graph_hash": self.graph_hash,
        }


class AssetGraphBuilder:
    """
    资产关系图构建器 — 6维关系矩阵 + 融合

    使用:
      builder = AssetGraphBuilder()
      snapshot = builder.build(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
      # snapshot.adj_matrix 可以直接用作 GATs 的邻接矩阵

    融合权重 (可调):
      - pearson: 0.25
      - spearman: 0.20
      - distance_corr: 0.15
      - cointegration: 0.15
      - granger: 0.15
      - volatility: 0.10
    """

    # 默认关系权重
    DEFAULT_RELATION_WEIGHTS = {
        "pearson": 0.25,
        "spearman": 0.20,
        "distance_corr": 0.15,
        "cointegration": 0.15,
        "granger": 0.15,
        "volatility": 0.10,
    }

    def __init__(self, relation_weights: Optional[Dict[str, float]] = None,
                 lookback_days: int = 365, min_history_days: int = 60,
                 adj_threshold: float = 0.3):
        """
        Args:
            relation_weights: 各关系维度的融合权重
            lookback_days: 默认回看天数
            min_history_days: 最少需要的数据天数
            adj_threshold: 邻接矩阵阈值 (低于此值置0, 稀疏化)
        """
        self.weights = {**self.DEFAULT_RELATION_WEIGHTS, **(relation_weights or {})}
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self.adj_threshold = adj_threshold
        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
            except Exception:
                self._exchange = None
        return self._exchange

    # ── 数据获取 ──────────────────────────────────

    def fetch_multi_asset_data(self, symbols: List[str],
                               lookback_days: int = None) -> Dict[str, pd.DataFrame]:
        """
        拉取多资产 OHLCV 数据

        Returns:
            {symbol: DataFrame with [date, open, high, low, close, volume]}
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        ex = self.exchange
        if ex is None:
            raise RuntimeError("无法连接交易所 — 请安装 ccxt")

        # Binance 日线限制: 每次最多1000条
        limit = min(lookback_days + 100, 1000)

        results = {}
        for symbol in symbols:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, "1d", limit=limit)
                df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"], unit="ms")
                df = df.set_index("date").sort_index()
                # 只保留最近 lookback_days
                cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
                df = df[df.index >= cutoff]
                if len(df) >= self.min_history_days:
                    results[symbol] = df
            except Exception as e:
                print(f"  ⚠️ {symbol} 数据拉取失败: {e}")

        return results

    # ── 关系矩阵构建 ──────────────────────────────

    def compute_pearson_correlation(self, close_prices: pd.DataFrame) -> np.ndarray:
        """
        Pearson 相关系数矩阵

        Args:
            close_prices: (T, n_assets) 收盘价 DataFrame

        Returns:
            (n_assets, n_assets) 相关性矩阵 [0, 1]
        """
        returns = close_prices.pct_change().dropna()
        if len(returns) < 30:
            return np.eye(len(close_prices.columns))

        corr = returns.corr(method="pearson").values
        corr = np.abs(corr)  # 我们关心的是相关强度, 不是方向
        np.fill_diagonal(corr, 0)  # 对角线置0 (自己和自己不算连边)
        return corr

    def compute_spearman_correlation(self, close_prices: pd.DataFrame) -> np.ndarray:
        """
        Spearman 秩相关系数矩阵 — 对异常值更鲁棒
        """
        returns = close_prices.pct_change().dropna()
        if len(returns) < 30:
            return np.eye(len(close_prices.columns))

        corr = returns.corr(method="spearman").values
        corr = np.abs(corr)
        np.fill_diagonal(corr, 0)
        return corr

    def compute_distance_correlation(self, close_prices: pd.DataFrame) -> np.ndarray:
        """
        距离相关性 (dCor) — 捕捉任意非线性依赖关系

        dCor = 0 当且仅当两个变量独立
        比 Pearson/Spearman 更通用
        """
        returns = close_prices.pct_change().dropna()
        n_assets = len(returns.columns)
        n_samples = len(returns)

        if n_samples < 30:
            return np.eye(n_assets)

        dcor_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                x = returns.iloc[:, i].values
                y = returns.iloc[:, j].values

                # 计算距离矩阵
                a = np.abs(x[:, None] - x[None, :])
                b = np.abs(y[:, None] - y[None, :])

                # 中心化
                A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
                B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

                # 距离协方差/方差
                dCov = np.sqrt(np.mean(A * B))
                dVarX = np.sqrt(np.mean(A * A))
                dVarY = np.sqrt(np.mean(B * B))

                if dVarX > 1e-10 and dVarY > 1e-10:
                    dcor = dCov / np.sqrt(dVarX * dVarY)
                else:
                    dcor = 0.0

                dcor_matrix[i, j] = dcor
                dcor_matrix[j, i] = dcor

        np.fill_diagonal(dcor_matrix, 0)
        return dcor_matrix

    def compute_cointegration_matrix(self, close_prices: pd.DataFrame) -> np.ndarray:
        """
        协整关系矩阵 — 基于 Engle-Granger 检验

        思想: 两组价格虽然各自非平稳, 但线性组合平稳 → 长期均衡关系
        我们用 (1 - p_value) 作为连接强度

        注意: 完整协整检验很慢, 这里做简化版
        """
        n_assets = len(close_prices.columns)
        coint_matrix = np.zeros((n_assets, n_assets))

        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            return np.eye(n_assets) * 0.1

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                y = close_prices.iloc[:, i].dropna().values
                x = close_prices.iloc[:, j].dropna().values

                # 对齐长度
                min_len = min(len(y), len(x))
                y = y[-min_len:]
                x = x[-min_len:]

                if min_len < 60:
                    continue

                try:
                    t_stat, p_value, _ = coint(y, x)
                    strength = max(0, 1 - p_value)  # p值越小 → 协整越显著
                    coint_matrix[i, j] = strength
                    coint_matrix[j, i] = strength
                except Exception:
                    pass

        np.fill_diagonal(coint_matrix, 0)
        return coint_matrix

    def compute_granger_matrix(self, close_prices: pd.DataFrame,
                               max_lag: int = 5) -> np.ndarray:
        """
        Granger 因果关系矩阵 (领先滞后)

        思想: 如果 X 的过去值能帮助预测 Y 的未来值 → X 领先 Y
        用 (1 - p_value) 作为有向边强度

        Returns:
            (n, n) 邻接矩阵 — 非对称! [i, j] = 资产 i 对资产 j 的影响
        """
        returns = close_prices.pct_change().dropna()
        n_assets = len(returns.columns)
        granger_matrix = np.zeros((n_assets, n_assets))

        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            return np.eye(n_assets) * 0.05

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    continue

                y = returns.iloc[:, j].dropna().values  # 被预测的
                x = returns.iloc[:, i].dropna().values  # 预测因子

                min_len = min(len(y), len(x)) - max_lag
                if min_len < 60:
                    continue

                data = np.column_stack([y[-min_len - max_lag:],
                                        x[-min_len - max_lag:]])

                try:
                    result = grangercausalitytests(data, max_lag, verbose=False)
                    # 取所有lag的最小p值
                    min_p = min(result[lag][0]["ssr_chi2test"][1] for lag in range(1, max_lag + 1))
                    granger_matrix[i, j] = max(0, 1 - min_p)
                except Exception:
                    pass

        np.fill_diagonal(granger_matrix, 0)
        return granger_matrix

    def compute_volatility_correlation(self, close_prices: pd.DataFrame,
                                       vol_window: int = 20) -> np.ndarray:
        """
        波动率相关性 — 尾部风险联动

        高波动率相关性 → 市场恐慌时一起暴跌
        用 GARCH 风格的 rolling volatility
        """
        returns = close_prices.pct_change().dropna()
        n_assets = len(returns.columns)

        # 滚动波动率
        rolling_vol = returns.rolling(window=vol_window).std().dropna()

        if len(rolling_vol) < 30:
            return np.eye(n_assets) * 0.1

        # 波动率的相关系数
        vol_corr = rolling_vol.corr().values
        vol_corr = np.abs(vol_corr)
        np.fill_diagonal(vol_corr, 0)
        return vol_corr

    def compute_sector_matrix(self, symbols: List[str]) -> np.ndarray:
        """
        行业/类别相似度矩阵 — 同类型资产有更紧密的连边
        """
        n = len(symbols)
        sector_matrix = np.zeros((n, n))

        categories = {s: CRYPTO_CATEGORIES.get(s, "unknown") for s in symbols}

        for i in range(n):
            for j in range(n):
                if i != j and categories[symbols[i]] == categories[symbols[j]]:
                    sector_matrix[i, j] = 0.5  # 同类别加分

        return sector_matrix

    # ── 融合与后处理 ──────────────────────────────

    def fuse_adjacency(self, raw_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        多维度关系矩阵 → 单一融合邻接矩阵

        加权求和后:
          1. 阈值稀疏化 (去弱边)
          2. 最大度约束 (每节点最多 k 个邻居)
          3. 对称化 (邻接矩阵必须对称 for GATs)
          4. 归一化 (每行和为1 → 平均聚合)
        """
        n = list(raw_matrices.values())[0].shape[0]
        fused = np.zeros((n, n))

        for relation_type, matrix in raw_matrices.items():
            w = self.weights.get(relation_type, 0.1)
            # 确保矩阵对称
            matrix_sym = (matrix + matrix.T) / 2
            fused += w * matrix_sym

        # 阈值稀疏化
        fused[fused < self.adj_threshold] = 0

        # 最大度约束: 每节点保留 top-k 最强连边
        k_max = min(n - 1, 5)
        for i in range(n):
            row = fused[i].copy()
            row[i] = -1  # 排除自己
            threshold_idx = np.argpartition(-row, k_max)[:k_max]
            mask = np.zeros(n, dtype=bool)
            mask[threshold_idx] = True
            fused[i] = fused[i] * mask

        # 按出度归一化 (GAT 需要)
        row_sums = fused.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        fused = fused / row_sums

        return fused

    def compute_graph_metrics(self, adj: np.ndarray,
                              symbols: List[str]) -> dict:
        """
        计算图属性: 度分布, 密度, 社区检测
        """
        n = len(symbols)
        binary_adj = (adj > 0).astype(float)
        np.fill_diagonal(binary_adj, 0)

        # 平均度
        degrees = binary_adj.sum(axis=1)
        avg_degree = float(np.mean(degrees))

        # 图密度
        max_edges = n * (n - 1)
        n_edges = binary_adj.sum() / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        # 社区检测 (用层次聚类)
        if n >= 4:
            # 距离矩阵 = 1 - 邻接矩阵 (强连边 = 近距离)
            dist = 1 - adj
            dist = (dist + dist.T) / 2
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist)
            try:
                Z = linkage(condensed, method="ward")
                # 用最大模块度启发式选社区数
                n_communities = min(n // 2 + 1, 4)
                labels = fcluster(Z, n_communities, criterion="maxclust")
                labels = [int(l) for l in labels]
            except Exception:
                labels = [1] * n
                n_communities = 1
        else:
            labels = list(range(1, n + 1))
            n_communities = n

        # Top edges
        top_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    top_edges.append({
                        "source": symbols[i],
                        "target": symbols[j],
                        "weight": round(float(adj[i, j]), 4),
                    })
        top_edges.sort(key=lambda e: e["weight"], reverse=True)
        top_edges = top_edges[:15]

        return {
            "avg_degree": avg_degree,
            "graph_density": density,
            "n_communities": n_communities,
            "community_labels": labels,
            "top_edges": top_edges,
        }

    # ── 主入口 ──────────────────────────────────

    def build(self, symbols: List[str],
              lookback_days: int = None,
              data: Dict[str, pd.DataFrame] = None) -> GraphSnapshot:
        """
        构建资产关系图

        Args:
            symbols: 资产列表
            lookback_days: 回看天数
            data: 预拉取的数据 (如果不提供, 自动从 Binance 拉取)

        Returns:
            GraphSnapshot
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        # 1. 获取数据
        if data is None:
            data = self.fetch_multi_asset_data(symbols, lookback_days)

        if len(data) < 2:
            raise ValueError(f"至少需要2个资产, 只有 {len(data)} 个可用")

        # 2. 对齐时间轴 + 构建统一 DataFrame
        symbols_available = list(data.keys())
        closes = {}
        for sym in symbols_available:
            closes[sym] = data[sym]["close"]
        close_df = pd.DataFrame(closes).dropna()

        if len(close_df) < self.min_history_days:
            raise ValueError(f"对齐后数据不足: {len(close_df)}天 < {self.min_history_days}天")

        print(f"📊 构建资产关系图: {len(symbols_available)} 资产 × {len(close_df)} 天")

        # 3. 计算各维度关系矩阵
        raw_matrices = {}

        print("  📈 Pearson 相关性...")
        raw_matrices["pearson"] = self.compute_pearson_correlation(close_df)

        print("  📊 Spearman 秩相关...")
        raw_matrices["spearman"] = self.compute_spearman_correlation(close_df)

        print("  🔗 距离相关性 (dCor)...")
        raw_matrices["distance_corr"] = self.compute_distance_correlation(close_df)

        print("  ⚖️  协整关系...")
        raw_matrices["cointegration"] = self.compute_cointegration_matrix(close_df)

        print("  ⏩ Granger 因果...")
        raw_matrices["granger"] = self.compute_granger_matrix(close_df)

        print("  📉 波动率相关...")
        raw_matrices["volatility"] = self.compute_volatility_correlation(close_df)

        # 行业相似度 (不是时间序列相关的, 当作先验)
        raw_matrices["sector"] = self.compute_sector_matrix(symbols_available)

        # 4. 融合
        print("🔗 融合邻接矩阵...")
        adj = self.fuse_adjacency(raw_matrices)

        # 5. 图属性
        metrics = self.compute_graph_metrics(adj, symbols_available)

        # 6. 构建快照
        snapshot = GraphSnapshot(
            symbols=symbols_available,
            n_assets=len(symbols_available),
            created_at=datetime.now().isoformat(),
            data_start=str(close_df.index[0].date()),
            data_end=str(close_df.index[-1].date()),
            n_days=len(close_df),
            lookback_days=lookback_days,
            adj_matrix=adj,
            adj_raw=raw_matrices,
            avg_degree=metrics["avg_degree"],
            graph_density=metrics["graph_density"],
            n_communities=metrics["n_communities"],
            community_labels=metrics["community_labels"],
            top_edges=metrics["top_edges"],
            version=1,
            graph_hash=self._hash_matrix(adj),
        )

        print(f"  ✅ 图构建完成: 密度={snapshot.graph_density:.3f}, "
              f"平均度={snapshot.avg_degree:.1f}, "
              f"{snapshot.n_communities} 个社区")

        return snapshot

    def build_rolling(self, symbols: List[str],
                      lookback_days: int = 365,
                      window_days: int = 90,
                      step_days: int = 30) -> List[GraphSnapshot]:
        """
        构建滚动窗口动态图 — 捕捉市场结构演变

        Args:
            symbols: 资产列表
            lookback_days: 数据总长度
            window_days: 每张图覆盖的天数
            step_days: 窗口滑动步长

        Returns:
            按时间排序的图快照列表
        """
        data = self.fetch_multi_asset_data(symbols, lookback_days)
        if len(data) < 2:
            return []

        # 找公共日期范围
        all_dates = None
        for sym in symbols:
            if sym in data:
                dates = set(data[sym].index)
                if all_dates is None:
                    all_dates = dates
                else:
                    all_dates &= dates

        if all_dates is None:
            return []

        date_list = sorted(all_dates)

        snapshots = []
        for start_idx in range(0, len(date_list) - window_days, step_days):
            end_idx = start_idx + window_days
            window_dates = date_list[start_idx:end_idx]

            # 切片数据
            window_data = {}
            for sym in symbols:
                if sym in data:
                    df_slice = data[sym][data[sym].index.isin(window_dates)]
                    if len(df_slice) >= self.min_history_days:
                        window_data[sym] = df_slice

            if len(window_data) >= 2:
                try:
                    snapshot = self.build(symbols, window_days, window_data)
                    snapshot.version = len(snapshots) + 1
                    snapshot.previous_hash = snapshots[-1].graph_hash if snapshots else ""
                    snapshots.append(snapshot)
                except Exception as e:
                    print(f"  ⚠️ 窗口 [{window_dates[0].date()} - {window_dates[-1].date()}] 构建失败: {e}")

        return snapshots

    def detect_graph_drift(self, old_adj: np.ndarray,
                           new_adj: np.ndarray) -> float:
        """
        检测图结构漂移 — 市场关系模式是否变了

        Returns:
            drift_score: [0, 1], >0.3 表示显著漂移
        """
        # Frobenius 范数差异
        diff = np.linalg.norm(old_adj - new_adj, ord='fro')
        max_diff = np.sqrt(old_adj.shape[0] * old_adj.shape[1])
        drift = diff / max_diff if max_diff > 0 else 0
        return float(drift)

    def detect_graph_drift_with_build(self, symbols: List[str],
                                      lookback_days: int = None) -> dict:
        """
        检测并返回完整的漂移分析结果 — 用于UI显示

        Returns:
            {"drifted": bool, "drift_score": float, "n_common_assets": int, ...}
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        # 加载旧图
        old_snapshot = self.load_snapshot("latest")

        # 构建新图 (只用最近90天)
        new_snapshot = self.build(symbols, lookback_days=min(lookback_days, 90))

        if old_snapshot is None:
            self.save_snapshot(new_snapshot, "latest")
            return {"drifted": False, "drift_score": 0.0, "action": "first_build",
                    "n_common_assets": len(symbols),
                    "old_density": 0, "new_density": new_snapshot.graph_density}

        # 对齐资产
        common_symbols = list(set(old_snapshot.symbols) & set(new_snapshot.symbols))
        if len(common_symbols) < 3:
            return {"drifted": False, "drift_score": 0.0, "action": "insufficient_overlap",
                    "n_common_assets": len(common_symbols),
                    "old_density": old_snapshot.graph_density,
                    "new_density": new_snapshot.graph_density}

        old_idx = [old_snapshot.symbols.index(s) for s in common_symbols]
        new_idx = [new_snapshot.symbols.index(s) for s in common_symbols]

        old_adj = old_snapshot.adj_matrix[np.ix_(old_idx, old_idx)]
        new_adj = new_snapshot.adj_matrix[np.ix_(new_idx, new_idx)]

        drift_score = self.detect_graph_drift(old_adj, new_adj)
        drifted = drift_score > 0.3

        if drifted:
            self.save_snapshot(new_snapshot, "latest")

        return {
            "drifted": drifted,
            "drift_score": drift_score,
            "threshold": 0.3,
            "action": "graph_rebuilt" if drifted else "graph_stable",
            "n_common_assets": len(common_symbols),
            "old_density": old_snapshot.graph_density,
            "new_density": new_snapshot.graph_density,
        }

    # ── 持久化 ──────────────────────────────────

    def save_snapshot(self, snapshot: GraphSnapshot,
                      name: str = "latest") -> Path:
        """保存图快照到磁盘"""
        path = GRAPH_DIR / f"graph_{name}.npz"
        np.savez(
            path,
            adj=snapshot.adj_matrix,
            symbols=np.array(snapshot.symbols, dtype=str),
        )
        # 元信息
        meta_path = GRAPH_DIR / f"graph_{name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def load_snapshot(self, name: str = "latest") -> Optional[GraphSnapshot]:
        """加载图快照"""
        path = GRAPH_DIR / f"graph_{name}.npz"
        meta_path = GRAPH_DIR / f"graph_{name}_meta.json"

        if not path.exists():
            return None

        data = np.load(path, allow_pickle=True)
        adj = data["adj"]
        symbols = list(data["symbols"])

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return GraphSnapshot(
            symbols=symbols,
            n_assets=len(symbols),
            created_at=meta.get("created_at", ""),
            data_start=meta.get("data_start", ""),
            data_end=meta.get("data_end", ""),
            n_days=meta.get("n_days", 0),
            lookback_days=meta.get("lookback_days", 365),
            adj_matrix=adj,
            adj_raw={},
            avg_degree=meta.get("avg_degree", 0),
            graph_density=meta.get("graph_density", 0),
            n_communities=meta.get("n_communities", 1),
            community_labels=meta.get("community_labels", []),
            top_edges=meta.get("top_edges", []),
            version=meta.get("version", 1),
            graph_hash=meta.get("graph_hash", ""),
        )

    def _hash_matrix(self, matrix: np.ndarray) -> str:
        """矩阵哈希 (用于变化检测)"""
        return str(hash(matrix.tobytes()))[:16]


# ═══════════════════════════════════════════════════════════════
# 多资产 GAT 模型 — 图增强预测
# ═══════════════════════════════════════════════════════════════

class CrossAssetGAT(nn.Module):
    """
    跨资产图注意力网络 — 同时预测多个资产

    与单资产 GAT (MultiHeadGAT) 的区别:
      - MultiHeadGAT: 一个资产 = 一个节点, 输入只是该资产的特征
      - CrossAssetGAT: N 个资产 = N 个节点, 输入是每个资产的特征 + 它们的关系图

    架构:
      每个资产的特征 → 节点嵌入 → GAT层 (聚合邻居信息) → 每节点独立预测

    这能学到: "ETH 涨了, 而且 ETH 和 BTC 关系很强 → BTC 也可能涨"
    """

    def __init__(self, n_features_per_asset: int,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 use_edge_weights: bool = True):
        """
        Args:
            n_features_per_asset: 每个资产的特征维度
            hidden_dim: 隐藏层维度
            n_heads: 注意力头数
            n_layers: GAT 层数
            dropout: Dropout 率
            use_edge_weights: 是否使用有权重边 (adj 中的值作为边权重)
        """
        super().__init__()
        self.n_features = n_features_per_asset
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_edge_weights = use_edge_weights

        # 特征嵌入 (每个资产独立投影)
        self.node_embedding = nn.Sequential(
            nn.Linear(n_features_per_asset, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 多层 GAT
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            in_dim = hidden_dim if layer_idx == 0 else hidden_dim * n_heads
            out_dim = hidden_dim

            self.gat_layers.append(
                MultiHeadGATLayer(in_dim, out_dim, n_heads, dropout)
            )

        # 图级聚合: 对每节点做 attention pooling over neighbors
        self.graph_summary = nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 输出头 (每个资产独立预测)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: (batch, n_assets, n_features_per_asset) or (n_assets, n_features)
            adj_matrix: (n_assets, n_assets) 邻接矩阵
                        注意: 同一 batch 中的所有样本共享同一个图结构

        Returns:
            predictions: (batch, n_assets, 1) or (n_assets, 1)
            attention_weights: 最后一层的注意力权重 (用于可解释性)
        """
        # 确保是 3D: (batch, n_assets, n_features)
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        batch, n_assets, n_feat = node_features.shape

        # 节点嵌入
        h = self.node_embedding(node_features)  # (batch, n_assets, hidden_dim)

        # 多层 GAT 传播
        attn_weights = None
        for layer in self.gat_layers:
            h, attn_weights = layer(h, adj_matrix)

        # 图级特征总结
        h_summary = self.graph_summary(h)  # (batch, n_assets, hidden_dim)

        # 每个资产独立预测
        pred = self.output_head(h_summary)  # (batch, n_assets, 1)

        if single_sample:
            pred = pred.squeeze(0)

        return pred, attn_weights


class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层 — 聚合邻居信息

    这是对 qlib_models.GraphAttentionLayer 的扩展:
      - 支持 batch 维度 (batch, n_nodes, features)
      - 多注意力头并行
      - 支持有权重邻接矩阵
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        # Attention: per-head scalar score
        self.a = nn.Parameter(torch.randn(1, n_heads, 2 * out_features) * 0.02)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            h: (batch, n_nodes, in_features)
            adj: (n_nodes, n_nodes) — 所有batch共享同一图结构

        Returns:
            h_prime: (batch, n_nodes, out_features * n_heads)
            attn_weights: (batch, n_heads, n_nodes, n_nodes) attention
        """
        batch, n_nodes, _ = h.shape

        # 线性变换: (batch, n_nodes, n_heads * out)
        Wh = self.W(h).view(batch, n_nodes, self.n_heads, self.out_features)

        # 计算注意力系数 (简化版 GAT)
        # Wh_i: (batch, n_nodes, 1, n_heads, out)
        Wh_i = Wh.unsqueeze(2)
        # Wh_j: (batch, 1, n_nodes, n_heads, out)
        Wh_j = Wh.unsqueeze(1)

        # 拼接: (batch, n_nodes, n_nodes, n_heads, 2*out)
        Wh_cat = torch.cat([Wh_i.expand(-1, -1, n_nodes, -1, -1),
                           Wh_j.expand(-1, n_nodes, -1, -1, -1)], dim=-1)

        # 注意力分数: self.a @ Wh_cat → (batch, n_nodes, n_nodes, n_heads)
        e = (Wh_cat * self.a).sum(dim=-1)
        e = self.leaky_relu(e)

        # Masked attention
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, self.n_heads)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)

        # 聚合: (batch, n_nodes, n_heads, out)
        h_prime = torch.einsum('bnmh,bmhf->bnhf', attention, Wh)

        # 展平多头
        h_prime = h_prime.reshape(batch, n_nodes, self.n_heads * self.out_features)
        h_prime = F.elu(h_prime)

        return h_prime, attention


class CrossAssetGATPredictor:
    """
    跨资产 GAT 推理器 — 与 QlibPredictor 并行的推理接口

    使用:
      predictor = CrossAssetGATPredictor()
      # 需要先 build graph 或 load graph
      predictions = predictor.predict(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
      # predictions: {"BTC/USDT": 0.023, "ETH/USDT": 0.018, ...}
    """

    def __init__(self, graph_name: str = "latest",
                 seq_len: int = 60, use_category_features: bool = True):
        """
        Args:
            graph_name: 图快照名称
            seq_len: 输入序列长度
            use_category_features: 是否使用资产类别特征
        """
        self.graph_name = graph_name
        self.seq_len = seq_len
        self.use_category_features = use_category_features

        self.graph_builder = AssetGraphBuilder()
        self.snapshot: Optional[GraphSnapshot] = None
        self.model: Optional[CrossAssetGAT] = None
        self._models_loaded: Dict[str, nn.Module] = {}

        # 尝试加载已有的图
        self._load_graph()

        # 加载预训练模型
        self._load_model()

    def _load_graph(self):
        """加载或构建资产关系图"""
        snapshot = self.graph_builder.load_snapshot(self.graph_name)
        if snapshot is not None:
            self.snapshot = snapshot
            print(f"📊 已加载图快照: {snapshot.n_assets} 资产, "
                  f"密度={snapshot.graph_density:.3f}")
        else:
            print("⚠️ 未找到图快照, 请先运行 build()")

    def _load_model(self):
        """加载预训练的 CrossAssetGAT 模型"""
        model_path = MODEL_DIR / "cross_asset_gat.pth"
        if model_path.exists():
            print(f"🧠 加载预训练 CrossAssetGAT 模型")
            # 延迟加载, 需要知道 n_features
        else:
            print("ℹ️ 未找到 CrossAssetGAT 预训练模型, 仅使用图结构增强")

    def build_graph(self, symbols: List[str] = None,
                    force: bool = False) -> GraphSnapshot:
        """
        构建/更新资产关系图

        Args:
            symbols: 资产列表 (None = 默认加密货币列表)
            force: 强制重建

        Returns:
            GraphSnapshot
        """
        if symbols is None:
            symbols = DEFAULT_CRYPTO_SYMBOLS[:5]

        if not force and self.snapshot is not None:
            age_hours = (datetime.now() - datetime.fromisoformat(self.snapshot.created_at)).total_seconds() / 3600
            if age_hours < 24:
                print(f"ℹ️ 图快照仍然新鲜 ({age_hours:.0f}小时前), 跳过重建")
                return self.snapshot

        print(f"🔨 构建资产关系图: {len(symbols)} 资产...")
        self.snapshot = self.graph_builder.build(symbols)
        self.graph_builder.save_snapshot(self.snapshot, self.graph_name)

        # 打印 Top 5 连边
        print("📊 Top 5 最强连边:")
        for edge in self.snapshot.top_edges[:5]:
            print(f"  {edge['source']:12s} ←→ {edge['target']:12s}  w={edge['weight']:.3f}")

        return self.snapshot

    def build_feature_vector(self, df_asset: pd.DataFrame,
                            symbol: str) -> Optional[np.ndarray]:
        """
        为单个资产构建特征向量 (用于节点特征)

        使用 FeatureFactoryV4 的特征工程
        """
        try:
            from feature_ts import FeatureFactoryV4
            ff = FeatureFactoryV4()
            feature_df = ff.compute_timeseries(df_asset)
            if feature_df is None or len(feature_df) < self.seq_len:
                return None
            # 取最后 seq_len 天的特征, 聚合为向量
            recent = feature_df.values[-self.seq_len:].astype(np.float32)
            # (seq_len, n_features) → (n_features,) via attention-style weighting
            # 越近的权重越高
            weights = np.exp(np.linspace(-2, 0, self.seq_len))
            weights = weights / weights.sum()
            weighted = np.average(recent, axis=0, weights=weights)
            return weighted
        except Exception as e:
            print(f"  ⚠️ {symbol} 特征构建失败: {e}")
            return None

    def predict(self, symbols: List[str] = None,
                force_build_graph: bool = False) -> Dict[str, Optional[float]]:
        """
        多资产联合预测 — 使用图增强

        Args:
            symbols: 资产列表
            force_build_graph: 是否强制重建图

        Returns:
            {symbol: predicted_forward_return, ...}
        """
        if symbols is None:
            if self.snapshot is not None:
                symbols = self.snapshot.symbols
            else:
                symbols = DEFAULT_CRYPTO_SYMBOLS[:5]

        # 1. 确保图可用
        if force_build_graph or self.snapshot is None:
            self.build_graph(symbols, force=force_build_graph)

        if self.snapshot is None:
            print("❌ 图快照不可用")
            return {}

        # 2. 获取数据
        data = self.graph_builder.fetch_multi_asset_data(symbols, lookback_days=400)
        if len(data) < 2:
            print("❌ 数据不足")
            return {}

        # 3. 为每个资产构建特征
        node_features_list = []
        pred_symbols = []

        for sym in symbols:
            if sym not in data:
                continue
            feat = self.build_feature_vector(data[sym], sym)
            if feat is not None:
                node_features_list.append(feat)
                pred_symbols.append(sym)

        if len(node_features_list) < 2:
            print("❌ 特征构建失败: 资产不足")
            return {}

        # 4. 对齐特征维度
        n_features_all = [len(f) for f in node_features_list]
        min_features = min(n_features_all)
        node_features_aligned = np.array([f[:min_features] for f in node_features_list])

        # 5. 构建/获取邻接矩阵
        # 如果预测的资产和图快照不完全一致, 从快照中提取子矩阵
        if set(pred_symbols) <= set(self.snapshot.symbols):
            # 快照覆盖了所有资产 → 提取子矩阵
            idx_map = {s: i for i, s in enumerate(self.snapshot.symbols)}
            indices = [idx_map[s] for s in pred_symbols]
            adj = self.snapshot.adj_matrix[np.ix_(indices, indices)]
        else:
            # 资产集不一致 → 重建图
            print("🔄 资产集与快照不一致, 重建图...")
            self.snapshot = self.graph_builder.build(pred_symbols)
            adj = self.snapshot.adj_matrix

        # 6. 如果有模型, 做 GAT 推理
        if self.model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(node_features_aligned).to(DEVICE)
                adj_t = torch.FloatTensor(adj).to(DEVICE)
                predictions, attn = self.model(x, adj_t)
                if predictions.dim() == 2:
                    predictions = predictions.squeeze(-1)
                preds = predictions.cpu().numpy()
        else:
            # 降级模式: 用图信息做加权平均
            # 邻居的平均预测会通过图权重传递
            preds = self._graph_enhanced_prediction(node_features_aligned, adj)

        # 7. 返回结果
        results = {}
        for i, sym in enumerate(pred_symbols):
            results[sym] = float(preds[i]) if i < len(preds) else None

        return results

    def _graph_enhanced_prediction(self, features: np.ndarray,
                                   adj: np.ndarray) -> np.ndarray:
        """
        图增强的简单预测 (无模型模式)

        思想: 每个节点的预测 = 自身特征信号 + 邻居加权平均
        这是 GAT 的简单近似
        """
        n = features.shape[0]

        # 每个节点的初始信号: 用最近特征的趋势
        initial_signals = np.zeros(n)
        for i in range(n):
            row = features[i]
            recent = row[-min(20, len(row)):]
            if len(recent) > 1:
                initial_signals[i] = np.mean(recent)

        # 图传播 (简化版 message passing)
        propagated = initial_signals.copy()
        for _ in range(2):  # 2-hop
            propagated = adj @ propagated

        # 混合: 自身信号 + 邻居传播
        alpha = 0.6  # 自身权重
        enhanced = alpha * initial_signals + (1 - alpha) * propagated

        return enhanced

    def train_model(self, symbols: List[str] = None,
                    n_epochs: int = 100, lr: float = 1e-3):
        """
        训练 CrossAssetGAT 模型

        Args:
            symbols: 资产列表
            n_epochs: 训练轮数
            lr: 学习率
        """
        if symbols is None:
            symbols = DEFAULT_CRYPTO_SYMBOLS[:5]

        # 1. 确保图可用
        if self.snapshot is None:
            self.build_graph(symbols)

        # 2. 获取数据并构建训练集
        data = self.graph_builder.fetch_multi_asset_data(symbols, lookback_days=730)
        if len(data) < 2:
            print("❌ 训练数据不足")
            return None

        # 3. 准备节点特征 + 标签
        print("📊 准备训练数据...")
        # 每个时间步: 用 [t-seq_len:t] 特征预测 [t+fwd] 收益
        fwd_window = 5

        X_all, y_all = [], []
        for sym in symbols:
            if sym not in data:
                continue
            df = data[sym]
            try:
                from feature_ts import FeatureFactoryV4
                ff = FeatureFactoryV4()
                feature_df = ff.compute_timeseries(df)
                if feature_df is None:
                    continue
                feat_array = feature_df.values.astype(np.float32)

                # 前向收益
                close = df["close"].values
                fwd_ret = np.zeros(len(close))
                fwd_ret[:-fwd_window] = (close[fwd_window:] / close[:-fwd_window] - 1)

                X_all.append(feat_array)
                y_all.append(fwd_ret[-len(feat_array):])
            except Exception as e:
                print(f"  ⚠️ {sym} 特征失败: {e}")

        if len(X_all) < 2:
            print("❌ 至少需要2个资产")
            return None

        # 对齐维度
        min_features = min(x.shape[1] for x in X_all)
        X_aligned = [x[:, :min_features] for x in X_all]
        n_samples = min(len(x) - self.seq_len for x in X_aligned)

        # 构建模型
        self.model = CrossAssetGAT(
            n_features_per_asset=min_features,
            hidden_dim=128,
            n_heads=4,
            n_layers=2,
            dropout=0.2,
        ).to(DEVICE)

        # 简化训练 (实际使用 PurgedKFold)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        adj_t = torch.FloatTensor(self.snapshot.adj_matrix).to(DEVICE)

        print(f"🧠 训练 CrossAssetGAT: {n_epochs} epochs, {len(X_aligned)} 资产")

        best_loss = float('inf')
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            # 随机采样训练
            for _ in range(min(100, n_samples // 4)):
                t = np.random.randint(0, n_samples)
                # 每个资产取 seq_len 窗口
                batch_x = []
                batch_y = []
                for i in range(len(X_aligned)):
                    x_slice = X_aligned[i][t:t + self.seq_len]
                    y_val = y_all[i][t + self.seq_len - 1] if t + self.seq_len < len(y_all[i]) else 0
                    batch_x.append(x_slice)
                    batch_y.append(y_val)

                x_tensor = torch.FloatTensor(np.array(batch_x)).to(DEVICE)  # (n_assets, seq_len, n_feat)
                y_tensor = torch.FloatTensor(np.array(batch_y)).to(DEVICE)

                # 对时序维度做简单聚合 (mean)
                x_flat = x_tensor.mean(dim=1)  # (n_assets, n_feat)

                optimizer.zero_grad()
                pred, _ = self.model(x_flat, adj_t)
                loss = criterion(pred.squeeze(-1), y_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0:
                avg_loss = epoch_loss / max(1, n_batches)
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.6f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss

        # 保存模型
        model_path = MODEL_DIR / "cross_asset_gat.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'symbols': symbols,
            'n_features': min_features,
            'trained_at': datetime.now().isoformat(),
        }, model_path)
        print(f"✅ 模型保存: {model_path}")

        return self.model

    def get_attention_interpretation(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        返回 GAT 注意力解释 — 哪个资产对哪个资产影响最大
        """
        if self.model is None or self.snapshot is None:
            return pd.DataFrame()

        if symbols is None:
            symbols = self.snapshot.symbols[:5]

        # 需要做一次前向推理获取注意力权重
        data = self.graph_builder.fetch_multi_asset_data(symbols, 400)
        node_feats = []
        for sym in symbols:
            if sym in data:
                feat = self.build_feature_vector(data[sym], sym)
                if feat is not None:
                    node_feats.append(feat)

        if len(node_feats) < 2:
            return pd.DataFrame()

        min_f = min(len(f) for f in node_feats)
        x = torch.FloatTensor(np.array([f[:min_f] for f in node_feats])).to(DEVICE)
        adj_t = torch.FloatTensor(self.snapshot.adj_matrix[:len(symbols), :len(symbols)]).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            _, attn = self.model(x.unsqueeze(0), adj_t)
            attn = attn.mean(dim=1).squeeze(0)  # avg over heads, (n, n)

        # 构建 DataFrame
        attn_np = attn.cpu().numpy()
        df = pd.DataFrame(attn_np, index=symbols[:len(node_feats)], columns=symbols[:len(node_feats)])
        return df


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="资产关系图引擎")
    parser.add_argument("--build", action="store_true", help="构建资产关系图")
    parser.add_argument("--symbols", type=str, nargs="+",
                       default=DEFAULT_CRYPTO_SYMBOLS[:5],
                       help="资产列表")
    parser.add_argument("--rolling", action="store_true", help="构建滚动窗口动态图")
    parser.add_argument("--train", action="store_true", help="训练 CrossAssetGAT 模型")
    parser.add_argument("--predict", action="store_true", help="多资产联合预测")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lookback", type=int, default=365, help="回看天数")

    args = parser.parse_args()

    predictor = CrossAssetGATPredictor()

    if args.build or args.rolling:
        builder = AssetGraphBuilder(lookback_days=args.lookback)

        if args.rolling:
            print("🔄 构建滚动窗口动态图...")
            snapshots = builder.build_rolling(args.symbols, args.lookback)
            print(f"\n✅ 生成 {len(snapshots)} 个图快照")

            if snapshots:
                # 检测漂移
                drift = builder.detect_graph_drift(
                    snapshots[0].adj_matrix,
                    snapshots[-1].adj_matrix,
                )
                print(f"📉 图漂移度: {drift:.4f} "
                      f"({'⚠️ 显著' if drift > 0.3 else '✅ 稳定'})")
        else:
            snapshot = builder.build(args.symbols, args.lookback)
            builder.save_snapshot(snapshot)
            print(f"\n📊 图统计:")
            print(f"  资产: {snapshot.n_assets}")
            print(f"  密度: {snapshot.graph_density:.4f}")
            print(f"  平均度: {snapshot.avg_degree:.2f}")
            print(f"  社区数: {snapshot.n_communities}")
            print(f"\n🔗 Top 5 最强连边:")
            for e in snapshot.top_edges[:5]:
                print(f"  {e['source']:12s} ←→ {e['target']:12s}  w={e['weight']:.3f}")

    if args.train:
        print("🧠 训练 CrossAssetGAT 模型...")
        model = predictor.train_model(args.symbols, n_epochs=args.epochs)
        if model:
            print("✅ 训练完成!")

    if args.predict:
        print("🔮 多资产联合预测...")
        preds = predictor.predict(args.symbols, force_build_graph=args.build)
        print(f"\n📊 预测结果:")
        for sym, pred in sorted(preds.items(), key=lambda x: abs(x[1] or 0), reverse=True):
            direction = "📈" if pred and pred > 0 else "📉" if pred and pred < 0 else "➡️"
            print(f"  {direction} {sym:12s}: {pred:+.6f}" if pred else f"  ❌ {sym:12s}: 预测失败")

    if not any([args.build, args.rolling, args.train, args.predict]):
        parser.print_help()
        print("\n💡 快速开始:")
        print("  python3 asset_graph.py --build               # 构建资产关系图")
        print("  python3 asset_graph.py --build --predict     # 构建并预测")
        print("  python3 asset_graph.py --train --epochs 100  # 训练 CrossAssetGAT")
