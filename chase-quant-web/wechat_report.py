"""
Chase量化策略 — 企业微信日报自动推送
======================================
Phase 14: 基于全算法框架的智能日报生成 + Webhook推送

核心理念:
  Yina不只是搬运数据 — 她解读盘面, 展示推算逻辑, 给出洞察。
  每份报告包含:
    1. 市场概览 (价格/成交量/恐慌指数)
    2. ML信号洞察 + 推算逻辑
    3. 模拟盘持仓 & 盈亏
    4. 风控状态
    5. 执行质量统计
    6. Yina的AI洞察 (多因子综合研判)

推送渠道:
  企业微信群「金融监控」
  Webhook: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769

推送时段:
  早报 08:30 — 隔夜行情 + 今日预判
  午报 14:00 — 盘中扫描 + 信号更新
  晚报 22:00 — 全天复盘 + 明日展望

使用:
  python3 wechat_report.py                    # 自动判断时段, 生成并推送
  python3 wechat_report.py --mode morning     # 早报
  python3 wechat_report.py --mode afternoon   # 午报
  python3 wechat_report.py --mode evening     # 晚报
  python3 wechat_report.py --dry-run          # 只打印, 不推送
  python3 wechat_report.py --test             # 测试webhook连通性
"""
from __future__ import annotations
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 企业微信 Webhook
WECHAT_WEBHOOK_URL = (
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"
    "?key=2c602b48-5da2-4989-9193-30c0e226c769"
)

# 推送时段判定
REPORT_HOURS = {
    "morning":   (5, 11),    # 05:00-11:00 → 早报
    "afternoon": (11, 17),   # 11:00-17:00 → 午报
    "evening":   (17, 23),   # 17:00-23:00 → 晚报
}

# 主要监控币种
WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]

# ═══════════════════════════════════════════
# 数据容器
# ═══════════════════════════════════════════


@dataclass
class MarketSnapshot:
    """单个市场快照"""
    symbol: str
    price: float = 0.0
    change_24h_pct: float = 0.0
    volume_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    spread_pct: float = 0.0


@dataclass
class SignalInsight:
    """ML信号洞察"""
    symbol: str
    action: str                      # BUY / SELL / HOLD
    confidence: float                # 0-1
    signal_score: float              # -1 to 1
    lgbm_direction: str = ""
    qlib_direction: str = ""
    graph_direction: str = ""        # asset graph signal
    alpha_direction: str = ""        # top alpha signal
    reasoning: str = ""              # 推算逻辑


@dataclass
class RiskStatus:
    """风控状态"""
    total_value: float
    total_pnl_pct: float
    open_positions: int
    daily_loss_used_pct: float
    monthly_progress_pct: float
    survival_score: float
    alerts: List[str]
    execution_quality: dict = field(default_factory=dict)


@dataclass
class DailyReport:
    """完整日报"""
    report_time: str
    report_mode: str                 # morning / afternoon / evening
    market_snapshots: List[MarketSnapshot]
    signal_insights: List[SignalInsight]
    portfolio_summary: dict
    risk_status: RiskStatus
    model_status: dict               # 模型新鲜度等
    alpha_discoveries: List[dict]    # 最新的Alpha
    graph_status: dict               # 资产关系图状态
    fear_greed: Optional[dict] = None  # 恐慌贪婪指数
    yina_insight: str = ""           # Yina的综合研判


# ═══════════════════════════════════════════
# 数据采集层 — 从各模块拉取数据
# ═══════════════════════════════════════════


class DataCollector:
    """从各系统模块采集数据, 容错设计 — 任一模块挂掉不影响整体"""

    def __init__(self):
        self.errors: List[str] = []

    # ── 市场数据 ──
    def fetch_market_snapshots(self, symbols: List[str] = None) -> List[MarketSnapshot]:
        """从 Binance 获取实时行情"""
        if symbols is None:
            symbols = WATCHLIST
        snapshots = []
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})

            # 批量获取 ticker
            tickers = {}
            try:
                all_tickers = exchange.fetch_tickers()
                tickers = {s: all_tickers.get(s, {}) for s in symbols}
            except Exception:
                # 逐个获取
                for sym in symbols:
                    try:
                        tickers[sym] = exchange.fetch_ticker(sym)
                    except Exception:
                        pass

            for sym in symbols:
                t = tickers.get(sym, {})
                if not t:
                    snapshots.append(MarketSnapshot(symbol=sym))
                    continue
                snapshots.append(MarketSnapshot(
                    symbol=sym,
                    price=t.get("last", 0) or t.get("close", 0),
                    change_24h_pct=t.get("percentage", 0) or t.get("change", 0) or 0,
                    volume_24h=t.get("quoteVolume", 0) or t.get("baseVolume", 0) or 0,
                    high_24h=t.get("high", 0) or 0,
                    low_24h=t.get("low", 0) or 0,
                    spread_pct=((t.get("ask", 0) or 0) - (t.get("bid", 0) or 0)) / max(t.get("last", 1), 1) * 100,
                ))

        except ImportError:
            self.errors.append("ccxt 未安装, 无法获取行情")
        except Exception as e:
            self.errors.append(f"行情获取失败: {e}")

        return snapshots

    # ── 恐惧贪婪指数 ──
    def fetch_fear_greed(self) -> Optional[dict]:
        """获取 Crypto Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "YinaQuant/2.4"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            item = data.get("data", [{}])[0]
            return {
                "value": int(item.get("value", 50)),
                "classification": item.get("value_classification", "Neutral"),
                "timestamp": item.get("timestamp", ""),
            }
        except Exception as e:
            self.errors.append(f"恐慌指数获取失败: {e}")
            return None

    # ── 组合数据 ──
    def fetch_portfolio(self) -> Optional[dict]:
        """获取虚拟盘组合状态"""
        try:
            from portfolio import get_portfolio
            pf = get_portfolio()

            positions_data = []
            for pos in pf.open_positions:
                # Extract clean symbol from position ID and data
                pos_symbol = pos.symbol
                # If symbol looks like an auto-generated ID, extract from position ID
                if pos_symbol.isdigit() or "_" not in pos_symbol:
                    # Try to extract from position ID: crypto_BTC/USDT_20260607...
                    pid_parts = pos.id.split("_", 2)
                    if len(pid_parts) >= 3:
                        pos_symbol = pid_parts[1]
                    elif len(pid_parts) >= 2:
                        pos_symbol = pid_parts[1]

                positions_data.append({
                    "symbol": pos_symbol,
                    "name": pos.name if pos.name != pos_symbol else pos_symbol,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "quantity": pos.quantity,
                    "value": pos.value,
                    "pnl": pos.pnl,
                    "pnl_pct": pos.pnl_pct,
                    "strategy": pos.execution_strategy,
                    "entry_time": pos.entry_time,
                })

            allocation = pf.get_allocation_summary()

            return {
                "total_value": pf.total_value,
                "total_cash": pf.total_cash,
                "total_pnl": pf.total_pnl,
                "total_pnl_pct": pf.total_pnl_pct,
                "positions": positions_data,
                "open_count": len(positions_data),
                "allocation": allocation,
            }
        except Exception as e:
            self.errors.append(f"组合数据获取失败: {e}")
            return None

    # ── 风控数据 ──
    def fetch_risk_status(self) -> Optional[RiskStatus]:
        """获取风控状态"""
        try:
            from portfolio import get_portfolio
            from risk import RiskController, DAILY_LOSS_LIMIT_PCT, MONTHLY_TARGET_PCT

            pf = get_portfolio()
            rc = RiskController(pf)
            report = rc.daily_risk_report()

            # 执行质量统计
            exec_quality = {}
            try:
                from execution import ExecutionStore
                store = ExecutionStore()
                stats = store.get_stats()
                exec_quality = {
                    "avg_shortfall_bps": stats.get("avg_shortfall_bps", 0),
                    "avg_slippage_bps": stats.get("avg_slippage_bps", 0),
                    "avg_fill_rate": stats.get("avg_fill_rate", 0),
                    "best_strategy": stats.get("best_strategy", ""),
                    "n_executions": stats.get("n_executions", 0),
                }
            except Exception:
                pass

            return RiskStatus(
                total_value=report["total_value"],
                total_pnl_pct=report["total_pnl_pct"],
                open_positions=report["open_positions"],
                daily_loss_used_pct=report.get("daily_loss_used", 0),
                monthly_progress_pct=report.get("monthly_progress", 0),
                survival_score=report.get("survival_score", 100),
                alerts=report.get("alerts", []),
                execution_quality=exec_quality,
            )
        except Exception as e:
            self.errors.append(f"风控数据获取失败: {e}")
            return None

    # ── ML信号 ──
    def fetch_ml_signals(self) -> List[SignalInsight]:
        """获取ML信号洞察 (v5融合引擎)"""
        insights = []
        try:
            from ml_signal_v5 import MLSignalEngineV5
            import pandas as pd
            import ccxt

            engine = MLSignalEngineV5()
            exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})

            for sym in WATCHLIST[:6]:  # Top 6 to avoid rate limits
                try:
                    # 获取OHLCV (generate_signal 内部会自己计算特征)
                    ohlcv = exchange.fetch_ohlcv(sym, "1h", limit=300)
                    df = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                    # 直接传 OHLCV 给 generate_signal, 它内部调用 FeatureFactory
                    signal = engine.generate_signal(df, sym)

                    if signal:
                        # 构建推算逻辑
                        reasoning = self._build_reasoning(signal, sym)

                        insights.append(SignalInsight(
                            symbol=sym,
                            action=signal.action,
                            confidence=signal.confidence,
                            signal_score=getattr(signal, "signal_weighted", 0),
                            lgbm_direction=getattr(signal, "signal_lgbm", 0) > 0.01 and "LONG" or
                                          getattr(signal, "signal_lgbm", 0) < -0.01 and "SHORT" or "NEUTRAL",
                            qlib_direction="FUSION" if getattr(signal, "n_models_active", 0) > 1 else "LGBM_ONLY",
                            graph_direction="",
                            alpha_direction="",
                            reasoning=reasoning,
                        ))
                except Exception as e:
                    self.errors.append(f"ML信号 {sym}: {e}")

            # 资产关系图增强
            try:
                from asset_graph import AssetGraphBuilder, GRAPH_AVAILABLE
                if GRAPH_AVAILABLE:
                    graph_insights = self._add_graph_insights(insights)
                    insights = graph_insights
            except Exception:
                pass

            # Alpha挖掘增强
            try:
                from alpha_miner import AlphaStore, ALPHA_AVAILABLE
                if ALPHA_AVAILABLE:
                    insights = self._add_alpha_insights(insights)
            except Exception:
                pass

        except ImportError as e:
            self.errors.append(f"ML引擎不可用: {e}")
        except Exception as e:
            self.errors.append(f"ML信号采集失败: {e}")

        return insights

    def _build_reasoning(self, signal, symbol: str) -> str:
        """从融合信号中提取推算逻辑"""
        parts = []

        # 信号强度
        score = getattr(signal, "signal_weighted", 0)
        if abs(score) > 0.3:
            parts.append(f"融合信号强度 {score:+.3f}, 方向明确")
        elif abs(score) > 0.1:
            parts.append(f"融合信号强度 {score:+.3f}, 方向温和")
        else:
            parts.append(f"融合信号强度 {score:+.3f}, 方向模糊")

        # 模型共识
        consensus = getattr(signal, "consensus_ratio", 0)
        if consensus > 0.8:
            parts.append(f"模型高度共识({consensus:.0%})")
        elif consensus > 0.5:
            parts.append(f"模型存在分歧({consensus:.0%})")
        else:
            parts.append(f"模型严重分歧({consensus:.0%}), 建议观望")

        # 分歧度
        divergence = getattr(signal, "divergence", 0)
        if divergence > 0.1:
            parts.append(f"分歧度偏高({divergence:.3f}), 降低仓位")

        # 模型数量
        n_models = getattr(signal, "n_models_active", 0)
        parts.append(f"{n_models}个模型参与投票")

        return "; ".join(parts)

    def _add_graph_insights(self, insights: List[SignalInsight]) -> List[SignalInsight]:
        """用资产关系图增强信号洞察"""
        try:
            from asset_graph import AssetGraphBuilder
            builder = AssetGraphBuilder()
            symbols = [i.symbol for i in insights if i.symbol]

            # 尝试加载已有图或快速构建
            adj, features = builder.build(symbols[:6])
            if adj is not None and len(adj) > 0:
                # 计算每个资产的邻居平均影响
                for i, ins in enumerate(insights):
                    if i < len(adj):
                        neighbors = adj[i]
                        strong_neighbors = np.sum(np.abs(neighbors) > 0.5)
                        ins.graph_direction = (
                            f"图连接{strong_neighbors}个强相关资产"
                            if strong_neighbors > 0 else "独立走势"
                        )
        except Exception:
            pass
        return insights

    def _add_alpha_insights(self, insights: List[SignalInsight]) -> List[SignalInsight]:
        """用Alpha挖掘增强信号洞察"""
        try:
            from alpha_miner import AlphaStore
            store = AlphaStore()
            top_alphas = store.list_alphas(n=3)
            if top_alphas:
                for ins in insights:
                    # 取最相关的alpha方向
                    ins.alpha_direction = f"Top Alpha: {top_alphas[0].get('name', '')}"
        except Exception:
            pass
        return insights

    # ── 模型状态 ──
    def fetch_model_status(self) -> dict:
        """获取模型新鲜度和训练状态"""
        status = {
            "lgbm_available": False,
            "qlib_available": False,
            "rolling_available": False,
            "model_freshness": "未知",
            "last_train": "",
            "models_count": 0,
        }
        try:
            # LightGBM 模型
            lgbm_dir = DATA_DIR / "models"
            if lgbm_dir.exists():
                lgbm_files = list(lgbm_dir.glob("*.pkl"))
                status["lgbm_available"] = len(lgbm_files) > 0
                status["models_count"] += len(lgbm_files)

            # Qlib 模型
            qlib_dir = DATA_DIR / "models" / "qlib"
            if qlib_dir.exists():
                qlib_files = list(qlib_dir.glob("*.pth"))
                status["qlib_available"] = len(qlib_files) > 0
                status["models_count"] += len(qlib_files)

            # 滚动训练状态
            try:
                from rolling_trainer import RollingModelRegistry
                registry = RollingModelRegistry()
                reg_status = registry.get_status()
                status["rolling_available"] = True
                status["model_freshness"] = reg_status.get("freshness", "未知")
                status["last_train"] = reg_status.get("last_train_time", "")
            except Exception:
                pass

        except Exception as e:
            self.errors.append(f"模型状态获取失败: {e}")

        return status

    # ── Alpha 发现 ──
    def fetch_alpha_discoveries(self) -> List[dict]:
        """获取最新的Alpha挖掘成果"""
        alphas = []
        try:
            from alpha_miner import AlphaStore
            store = AlphaStore()
            alphas = store.list_alphas(n=5)  # top 5
        except Exception:
            pass
        return alphas

    # ── 图状态 ──
    def fetch_graph_status(self) -> dict:
        """获取资产关系图状态"""
        status = {"available": False, "n_assets": 0, "last_build": ""}
        try:
            from asset_graph import GRAPH_AVAILABLE
            if GRAPH_AVAILABLE:
                status["available"] = True
                graph_dir = DATA_DIR / "asset_graphs"
                if graph_dir.exists():
                    files = list(graph_dir.glob("*.json"))
                    status["n_graphs"] = len(files)
                    if files:
                        latest = max(files, key=lambda f: f.stat().st_mtime)
                        status["last_build"] = datetime.fromtimestamp(
                            latest.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        return status

    # ── 综合采集 ──
    def collect_all(self, mode: str = "auto") -> DailyReport:
        """采集所有数据, 组装成完整日报"""
        now = datetime.now(timezone.utc)
        beijing_now = now + timedelta(hours=8)

        if mode == "auto":
            hour = beijing_now.hour
            mode = "evening"
            for m, (start, end) in REPORT_HOURS.items():
                if start <= hour < end:
                    mode = m
                    break

        # 并行采集 (实际串行, 但容错)
        market_snapshots = self.fetch_market_snapshots()
        signal_insights = self.fetch_ml_signals()
        portfolio = self.fetch_portfolio()
        risk_status = self.fetch_risk_status()
        model_status = self.fetch_model_status()
        alpha_discoveries = self.fetch_alpha_discoveries()
        graph_status = self.fetch_graph_status()
        fear_greed = self.fetch_fear_greed()

        # 生成 Yina 洞察
        yina_insight = self._generate_yina_insight(
            mode, market_snapshots, signal_insights, portfolio, risk_status, fear_greed
        )

        return DailyReport(
            report_time=beijing_now.strftime("%Y-%m-%d %H:%M"),
            report_mode=mode,
            market_snapshots=market_snapshots,
            signal_insights=signal_insights,
            portfolio_summary=portfolio or {},
            risk_status=risk_status or RiskStatus(
                total_value=10000, total_pnl_pct=0, open_positions=0,
                daily_loss_used_pct=0, monthly_progress_pct=0,
                survival_score=100, alerts=[]
            ),
            model_status=model_status,
            alpha_discoveries=alpha_discoveries,
            graph_status=graph_status,
            fear_greed=fear_greed,
            yina_insight=yina_insight,
        )

    def _generate_yina_insight(self, mode: str, snapshots: List[MarketSnapshot],
                                signals: List[SignalInsight], portfolio: Optional[dict],
                                risk: Optional[RiskStatus], fear_greed: Optional[dict]) -> str:
        """Yina的综合研判 — 多因子综合推理"""
        lines = []

        # 1. 市场情绪判断
        if fear_greed:
            fg = fear_greed["value"]
            if fg >= 75:
                lines.append(f"🟢 市场极度贪婪(F&G={fg}), 注意回调风险")
            elif fg >= 55:
                lines.append(f"🟡 市场偏贪婪(F&G={fg}), 趋势可能延续但需警惕")
            elif fg >= 45:
                lines.append(f"⚪ 市场中性(F&G={fg}), 适合趋势策略")
            elif fg >= 25:
                lines.append(f"🟠 市场偏恐惧(F&G={fg}), 可能出现超跌反弹")
            else:
                lines.append(f"🔴 市场极度恐惧(F&G={fg}), 历史看是布局机会")

        # 2. 涨跌比
        if snapshots:
            up = sum(1 for s in snapshots if s.change_24h_pct > 0)
            down = sum(1 for s in snapshots if s.change_24h_pct < 0)
            total = len(snapshots)
            if total > 0:
                if up > down * 2:
                    lines.append(f"📈 普涨格局 ({up}/{total}上涨), 追高需谨慎")
                elif down > up * 2:
                    lines.append(f"📉 普跌格局 ({down}/{total}下跌), 恐慌中找机会")
                else:
                    lines.append(f"📊 分化行情 ({up}涨{down}跌), 精选标的")

        # 3. 信号汇总
        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        if buy_signals:
            names = ", ".join(s.symbol.replace("/USDT", "") for s in buy_signals[:3])
            lines.append(f"🎯 买入信号: {names}")
        if sell_signals:
            names = ", ".join(s.symbol.replace("/USDT", "") for s in sell_signals[:3])
            lines.append(f"⚠️ 卖出信号: {names}")

        # 4. 持仓评估
        if portfolio and portfolio.get("positions"):
            pos_data = portfolio["positions"]
            profitable = sum(1 for p in pos_data if p["pnl_pct"] > 0)
            total_pos = len(pos_data)
            if total_pos > 0:
                lines.append(
                    f"💰 持仓{total_pos}只, {profitable}只盈利, "
                    f"总盈亏{portfolio['total_pnl_pct']:+.2f}%"
                )

        # 5. 风控提示
        if risk and risk.alerts:
            lines.append(f"🚨 风控警报: {'; '.join(risk.alerts[:3])}")

        # 6. 时段特定建议
        if mode == "morning":
            lines.append("🌅 早盘策略: 关注隔夜突破/跌破关键位的品种, 优先处理止损/止盈")
        elif mode == "afternoon":
            lines.append("☀️ 午盘策略: 欧盘开盘前后波动加大, 适合短线做T")
        else:
            lines.append("🌙 夜盘策略: 美盘时段流动性最佳, 关注美股联动效应")

        return "\n".join(lines)


# ═══════════════════════════════════════════
# 格式化层 — 企业微信 Markdown
# ═══════════════════════════════════════════


class ReportFormatter:
    """将 DailyReport 格式化为企业微信 Markdown 消息"""

    MODE_EMOJI = {
        "morning": "🌅",
        "afternoon": "☀️",
        "evening": "🌙",
    }
    MODE_LABEL = {
        "morning": "早报",
        "afternoon": "午报",
        "evening": "晚报",
    }

    @classmethod
    def format(cls, report: DailyReport) -> str:
        """主格式化入口"""
        emoji = cls.MODE_EMOJI.get(report.report_mode, "📊")
        label = cls.MODE_LABEL.get(report.report_mode, "日报")

        sections = [
            cls._header(report, emoji, label),
            cls._market_overview(report),
            cls._ml_signals(report),
            cls._portfolio_section(report),
            cls._risk_section(report),
            cls._execution_section(report),
            cls._yina_insight_section(report),
            cls._footer(report),
        ]

        return "\n".join(s for s in sections if s)

    @classmethod
    def _header(cls, report: DailyReport, emoji: str, label: str) -> str:
        return (
            f"{emoji} **Yina量化{label}** | {report.report_time}\n"
            f"> Chase哥的AI量化助手 · v2.4\n"
        )

    @classmethod
    def _market_overview(cls, report: DailyReport) -> str:
        """市场概览 — 表格形式"""
        lines = ["## 📊 市场概览\n"]
        lines.append("| 币种 | 价格 | 24h涨跌 | 24h成交量 |")
        lines.append("|------|------|---------|-----------|")

        for s in report.market_snapshots[:8]:
            if s.price <= 0:
                continue
            sym = s.symbol.replace("/USDT", "")
            change = f"{s.change_24h_pct:+.2f}%"

            # 成交量格式化
            vol = s.volume_24h
            if vol > 1e9:
                vol_str = f"${vol/1e9:.1f}B"
            elif vol > 1e6:
                vol_str = f"${vol/1e6:.1f}M"
            else:
                vol_str = f"${vol:,.0f}"

            # 价格格式化
            if s.price >= 1000:
                price_str = f"${s.price:,.0f}"
            elif s.price >= 1:
                price_str = f"${s.price:,.2f}"
            else:
                price_str = f"${s.price:.4f}"

            lines.append(f"| {sym} | {price_str} | {change} | {vol_str} |")

        # 恐惧贪婪指数
        fg = report.fear_greed
        if fg:
            fg_bar = "🟢" * min(5, max(1, fg["value"] // 20)) + "⚪" * max(0, 5 - fg["value"] // 20)
            lines.append(f"\n> 😱 恐惧&贪婪指数: **{fg['value']}** {fg_bar} — {fg['classification']}")

        return "\n".join(lines)

    @classmethod
    def _ml_signals(cls, report: DailyReport) -> str:
        """ML信号洞察 + 推算逻辑"""
        if not report.signal_insights:
            return "## 🤖 ML信号\n> 信号引擎暂不可用, 请检查模型状态\n"

        lines = ["## 🤖 ML信号洞察\n"]

        # 信号汇总表
        lines.append("| 币种 | 信号 | 置信度 | 强度 | 推算逻辑 |")
        lines.append("|------|------|:-----:|:----:|----------|")

        for ins in report.signal_insights[:6]:
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(ins.action, "⚪")
            sym = ins.symbol.replace("/USDT", "")
            conf = f"{ins.confidence:.0%}"
            score = f"{ins.signal_score:+.2f}"
            # 压缩推理逻辑
            reasoning = ins.reasoning[:80] + ("..." if len(ins.reasoning) > 80 else "")
            lines.append(f"| {sym} | {action_emoji} {ins.action} | {conf} | {score} | {reasoning} |")

        # 模型共识摘要
        buys = sum(1 for s in report.signal_insights if s.action == "BUY")
        sells = sum(1 for s in report.signal_insights if s.action == "SELL")
        holds = sum(1 for s in report.signal_insights if s.action == "HOLD")
        lines.append(f"\n> 🎯 信号分布: {buys}买 {sells}卖 {holds}观望")

        # 图增强信息
        graph_info = [s.graph_direction for s in report.signal_insights if s.graph_direction]
        if graph_info:
            lines.append(f"> 🔗 图分析: {graph_info[0]}")

        return "\n".join(lines)

    @classmethod
    def _portfolio_section(cls, report: DailyReport) -> str:
        """模拟盘持仓"""
        pf = report.portfolio_summary
        if not pf:
            return "## 📈 模拟盘\n> 组合数据不可用\n"

        lines = ["## 📈 模拟盘持仓\n"]

        # 总览
        pnl_emoji = "🟢" if pf["total_pnl_pct"] >= 0 else "🔴"
        lines.append(
            f"> 总资产: **¥{pf['total_value']:,.2f}** | "
            f"总盈亏: {pnl_emoji} **{pf['total_pnl_pct']:+.2f}%** (¥{pf['total_pnl']:+.2f}) | "
            f"持仓: **{pf['open_count']}**只"
        )

        # 各市场分配
        lines.append("\n| 市场 | 分配 | 现值 | 盈亏 |")
        lines.append("|------|------|------|------|")
        for market, alloc in pf.get("allocation", {}).items():
            pnl_str = f"{alloc['pnl_pct']:+.2f}%"
            lines.append(
                f"| {alloc['label']} | ¥{alloc['allocated']:,.0f} | "
                f"¥{alloc['total']:,.0f} | {pnl_str} |"
            )

        # 持仓明细
        positions = pf.get("positions", [])
        if positions:
            lines.append("\n| 持仓 | 入场价 | 现价 | 盈亏 | 策略 |")
            lines.append("|------|--------|------|------|------|")
            for pos in positions:
                pnl_str = f"{pos['pnl_pct']:+.2f}%"
                entry = f"¥{pos['entry_price']:,.0f}" if pos['entry_price'] >= 1 else f"${pos['entry_price']:.2f}"
                current = f"¥{pos['current_price']:,.0f}" if pos['current_price'] >= 1 else f"${pos['current_price']:.2f}"
                strat = pos.get("strategy", "") or "-"
                lines.append(
                    f"| {pos['symbol'][:12]} | {entry} | {current} | {pnl_str} | {strat} |"
                )
        else:
            lines.append("\n> 当前无持仓, 等待信号触发")

        return "\n".join(lines)

    @classmethod
    def _risk_section(cls, report: DailyReport) -> str:
        """风控状态"""
        risk = report.risk_status
        if not risk:
            return ""

        lines = ["## 🛡️ 风控状态\n"]

        # 风控仪表
        survival_emoji = "🟢" if risk.survival_score >= 80 else "🟡" if risk.survival_score >= 50 else "🔴"
        lines.append(
            f"> 生存分: {survival_emoji} **{risk.survival_score:.0f}/100** | "
            f"日损额度: {risk.daily_loss_used_pct:.0f}% | "
            f"月目标进度: {risk.monthly_progress_pct:.0f}%"
        )

        # 警报
        if risk.alerts:
            for alert in risk.alerts[:3]:
                lines.append(f"> {alert}")
        else:
            lines.append("> ✅ 无风控警报, 系统运行正常")

        # 模型状态
        model = report.model_status
        if model:
            freshness = model.get("model_freshness", "未知")
            lines.append(f"> 🧠 模型新鲜度: **{freshness}** | 已加载{model.get('models_count', 0)}个模型")

        return "\n".join(lines)

    @classmethod
    def _execution_section(cls, report: DailyReport) -> str:
        """执行质量"""
        risk = report.risk_status
        if not risk or not risk.execution_quality:
            return ""

        eq = risk.execution_quality
        if eq.get("n_executions", 0) == 0:
            return ""

        lines = ["## 📊 执行质量 (近期)\n"]
        lines.append(
            f"> 平均滑点: **{eq.get('avg_slippage_bps', 0):.1f}bps** | "
            f"执行缺口: **{eq.get('avg_shortfall_bps', 0):.1f}bps** | "
            f"成交率: **{eq.get('avg_fill_rate', 0):.1%}**"
        )
        if eq.get("best_strategy"):
            lines.append(f"> 最佳策略: **{eq['best_strategy'].upper()}** | 共{eq.get('n_executions', 0)}笔执行记录")

        return "\n".join(lines)

    @classmethod
    def _yina_insight_section(cls, report: DailyReport) -> str:
        """Yina的综合研判"""
        if not report.yina_insight:
            return ""

        lines = ["## 💡 Yina的综合研判\n"]
        for line in report.yina_insight.split("\n"):
            if line.strip():
                lines.append(f"> {line.strip()}")
        return "\n".join(lines)

    @classmethod
    def _footer(cls, report: DailyReport) -> str:
        return (
            f"\n---\n"
            f"🐾 Yina Quant v2.4 · {report.report_time}\n"
            f"> 本报告由AI自动生成, 仅供参考, 不构成投资建议\n"
            f"> 算法框架: Multi-LLM Ensemble + 五维评分 + 拆单执行\n"
        )


# ═══════════════════════════════════════════
# 推送层 — 企业微信 Webhook
# ═══════════════════════════════════════════


class WeChatPusher:
    """企业微信群机器人推送"""

    def __init__(self, webhook_url: str = WECHAT_WEBHOOK_URL):
        self.webhook_url = webhook_url
        self.max_msg_length = 4096  # 企业微信 markdown 消息上限

    def push_markdown(self, content: str) -> bool:
        """推送 Markdown 格式消息"""
        # 企业微信 markdown 消息体
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            }
        }

        # 如果内容超长, 分段发送
        if len(content) > self.max_msg_length:
            return self._push_long_message(content)

        return self._send(payload)

    def push_text(self, content: str) -> bool:
        """推送纯文本消息 (降级方案)"""
        payload = {
            "msgtype": "text",
            "text": {
                "content": content,
            }
        }
        return self._send(payload)

    def _push_long_message(self, content: str) -> bool:
        """分段发送超长消息"""
        # 按 ## 分段
        sections = content.split("\n## ")
        if len(sections) <= 1:
            sections = [content[i:i + self.max_msg_length - 200]
                       for i in range(0, len(content), self.max_msg_length - 200)]

        current = sections[0]
        success = True
        for section in sections[1:]:
            if len(current) + len(section) + 5 < self.max_msg_length:
                current += "\n## " + section
            else:
                if not self.push_markdown(current):
                    success = False
                current = "## " + section

        if current:
            if not self.push_markdown(current):
                success = False

        return success

    def _send(self, payload: dict) -> bool:
        """发送 HTTP POST"""
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Content-Length": str(len(data)),
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                if result.get("errcode") == 0:
                    return True
                else:
                    print(f"⚠️ Webhook返回错误: {result}", file=sys.stderr)
                    return False
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP错误 {e.code}: {e.reason}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"❌ 推送失败: {e}", file=sys.stderr)
            return False

    def test_connection(self) -> bool:
        """测试webhook连通性"""
        test_msg = f"🧪 Yina量化系统 — Webhook连通性测试\n> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n> 状态: 正常 ✅"
        return self.push_markdown(test_msg)


# ═══════════════════════════════════════════
# 日报引擎 — 编排采集→格式化→推送
# ═══════════════════════════════════════════


class DailyReportEngine:
    """日报引擎: 编排全流程"""

    def __init__(self, webhook_url: str = WECHAT_WEBHOOK_URL):
        self.collector = DataCollector()
        self.formatter = ReportFormatter()
        self.pusher = WeChatPusher(webhook_url)

    def generate_and_push(self, mode: str = "auto", dry_run: bool = False) -> DailyReport:
        """
        生成日报并推送

        Args:
            mode: morning / afternoon / evening / auto
            dry_run: True=只打印不推送

        Returns:
            DailyReport 对象
        """
        print(f"🔍 采集数据中... (mode={mode})")
        report = self.collector.collect_all(mode)

        print(f"📝 格式化报告中...")
        markdown = self.formatter.format(report)

        # 保存报告到本地
        self._save_report(report, markdown)

        if dry_run:
            print("=" * 60)
            print(markdown)
            print("=" * 60)
            print("🔇 Dry-run模式, 未推送到企业微信")
        else:
            print(f"📤 推送到企业微信...")
            success = self.pusher.push_markdown(markdown)
            if success:
                print("✅ 推送成功!")
            else:
                print("❌ 推送失败, 报告已保存到本地")
                # 降级: 尝试纯文本
                text_content = markdown.replace("#", "").replace("*", "").replace("|", " ")
                self.pusher.push_text(f"[降级模式] 日报内容过长, 已保存到:\nreports/{report.report_time}.md")

        # 打印采集错误
        if self.collector.errors:
            print(f"\n⚠️ 采集过程中有 {len(self.collector.errors)} 个非致命错误:")
            for err in self.collector.errors[:5]:
                print(f"  - {err}")

        return report

    def _save_report(self, report: DailyReport, markdown: str):
        """保存报告到本地"""
        report_file = REPORTS_DIR / f"report_{report.report_time.replace(':', '-').replace(' ', '_')}.md"
        report_file.write_text(markdown, encoding="utf-8")

        # 保存JSON版本供后续分析
        json_file = REPORTS_DIR / f"report_{report.report_time.replace(':', '-').replace(' ', '_')}.json"
        try:
            json.dump(asdict(report), json_file.open("w", encoding="utf-8"),
                     indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="🐾 Yina量化 — 企业微信日报推送",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 wechat_report.py                        # 自动判断时段, 生成并推送
  python3 wechat_report.py --mode morning         # 早报
  python3 wechat_report.py --mode evening         # 晚报
  python3 wechat_report.py --dry-run              # 只打印, 不推送
  python3 wechat_report.py --test                 # 测试webhook连通性
  python3 wechat_report.py --list                 # 列出历史报告
        """,
    )
    parser.add_argument("--mode", choices=["auto", "morning", "afternoon", "evening"],
                        default="auto", help="报告时段 (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="只打印不推送")
    parser.add_argument("--test", action="store_true", help="测试Webhook连通性")
    parser.add_argument("--list", action="store_true", dest="list_reports",
                        help="列出历史报告")
    parser.add_argument("--json", action="store_true", help="输出JSON格式 (dry-run模式)")
    args = parser.parse_args()

    # 测试连通性
    if args.test:
        print("🧪 测试企业微信Webhook连通性...")
        pusher = WeChatPusher()
        ok = pusher.test_connection()
        if ok:
            print("✅ Webhook连通正常! 可以开始推送日报。")
        else:
            print("❌ Webhook连通失败! 请检查网络和Key。")
        sys.exit(0 if ok else 1)

    # 列出历史报告
    if args.list_reports:
        if REPORTS_DIR.exists():
            files = sorted(REPORTS_DIR.glob("report_*.md"), reverse=True)
            print(f"📋 历史报告 ({len(files)}份):\n")
            for f in files[:20]:
                size = f.stat().st_size
                print(f"  {f.name} ({size:,} bytes)")
        else:
            print("暂无历史报告")
        sys.exit(0)

    # JSON输出
    if args.json:
        args.dry_run = True

    # 生成并推送
    engine = DailyReportEngine()
    report = engine.generate_and_push(mode=args.mode, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
