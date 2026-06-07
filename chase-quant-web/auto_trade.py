"""
Chase量化策略 — 自动交易脚本
由 Cron 触发, 扫描→决策→执行→通知

Phase 7 升级: 支持ML信号引擎 (MLSignalEngineV4) + 多币种扫描
  使用方式:
    python3 auto_trade.py --ml              # ML模式 (推荐)
    python3 auto_trade.py                   # 传统模式 (RSI/MACD)
    python3 auto_trade.py --markets crypto --ml  # 只扫加密货币
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from portfolio import PortfolioManager, ALLOCATION
from risk import RiskController
from signals import SignalEngine, CryptoSignals, AStockSignals, USStockSignals

# 偏差修正
try:
    from bias_correction import full_bias_audit, estimate_survival_bias
    BIAS_AWARE = True
except ImportError:
    BIAS_AWARE = False

# ML信号引擎 (Phase 7)
try:
    from ml_signal_v4 import MLSignalEngineV4, SubSignalBuilder, EnsembleSignal
    ML_SIGNAL_AVAILABLE = True
except ImportError:
    ML_SIGNAL_AVAILABLE = False

# Qlib融合信号引擎 (Phase 9)
try:
    from ml_signal_v5 import MLSignalEngineV5, FusionSignal
    ML_V5_AVAILABLE = True
except ImportError:
    ML_V5_AVAILABLE = False

# 滚动在线学习 (Phase 10)
try:
    from rolling_trainer import RollingTrainer, RollingModelRegistry
    ROLLING_AVAILABLE = True
except ImportError:
    ROLLING_AVAILABLE = False

# 资产关系图 (Phase 11)
try:
    from asset_graph import CrossAssetGATPredictor, AssetGraphBuilder
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

# Alpha挖掘 (Phase 12)
try:
    from alpha_miner import AlphaStore, evaluate_expression
    ALPHA_AVAILABLE = True
except ImportError:
    ALPHA_AVAILABLE = False

# 订单执行优化 (Phase 13)
try:
    from execution import (ExecutionEngine, ExecutionConfig, ExecutionStore,
                           SmartOrderRouter, MarketImpactModel)
    EXECUTION_AVAILABLE = True
except ImportError:
    EXECUTION_AVAILABLE = False


def is_market_open(market: str) -> bool:
    """判断市场是否交易时段"""
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 6=Sun

    if weekday >= 5:  # 周末
        return market == "crypto"

    hour = now.hour + now.minute / 60

    if market == "a_stock":
        return (9.5 <= hour <= 11.5) or (13.0 <= hour <= 15.0)
    elif market == "us_stock":
        # 美股夏令时 21:30-04:00 CST
        return hour >= 21.5 or hour <= 4.0
    elif market == "crypto":
        return True
    return False


class MLAutoTrader:
    """
    ML增强自动交易器 (Phase 7)

    用 MLSignalEngineV4 生成信号 → 多币种扫描 → 自动入场/出场
    跨市场数据共享: 拉一次, 所有symbol复用
    """

    # 加密货币扫描列表 (流动性好 + 数据充足)
    CRYPTO_WATCHLIST = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ]

    def __init__(self, use_lgbm: bool = True):
        if not ML_SIGNAL_AVAILABLE:
            raise ImportError("ML信号引擎不可用")
        self.use_lgbm = use_lgbm
        self.engine = MLSignalEngineV4()
        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
            except Exception:
                self._exchange = None
        return self._exchange

    def fetch_ohlcv(self, symbol: str, limit: int = 400) -> pd.DataFrame | None:
        """拉取OHLCV数据"""
        try:
            ex = self.exchange
            if ex is None:
                return None
            ohlcv = ex.fetch_ohlcv(symbol, "1d", limit=limit)
            df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            return df
        except Exception as e:
            print(f"  ⚠️ {symbol} 数据拉取失败: {e}")
            return None

    def scan(self, symbols: list = None) -> list[dict]:
        """
        扫描多币种，返回ML信号列表

        Returns:
            [{"symbol": "BTC/USDT", "signal": EnsembleSignal, "price": ..., "action": ...}, ...]
        """
        if symbols is None:
            symbols = self.CRYPTO_WATCHLIST

        results = []
        n = len(symbols)

        # 跨市场数据共享: 只拉一次 (用第一个symbol触发)
        cross_market_loaded = False

        for i, symbol in enumerate(symbols):
            print(f"  🔍 [{i+1}/{n}] {symbol}...", end=" ")

            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 200:
                print("数据不足")
                continue

            try:
                # 第一个symbol触发跨市场数据加载
                # 后续symbol跳过 (跨市场数据已在engine中缓存)
                signal = self.engine.generate_signal(df, symbol, use_lgbm=self.use_lgbm)
                price = float(df["close"].values[-1])

                results.append({
                    "symbol": symbol,
                    "name": symbol.replace("/USDT", ""),
                    "price": price,
                    "signal": signal,
                    "action": signal.action,
                    "signal_val": signal.signal_lgbm if (self.use_lgbm and signal.lgbm_available) else signal.signal_ic,
                    "confidence": signal.confidence,
                    "consensus": signal.consensus,
                    "active_themes": signal.n_sub_signals_active,
                })
                icon = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "⚪"
                print(f"{icon} {signal.action} | sig={results[-1]['signal_val']:+.2f} | "
                      f"置信={signal.confidence:.0%} | {signal.n_sub_signals_active}/7主题")

            except Exception as e:
                print(f"信号生成失败: {e}")

        # 按信号值排序 (绝对值越大越极端)
        results.sort(key=lambda r: abs(r["signal_val"]), reverse=True)
        return results

    def get_status_summary(self) -> str:
        """引擎状态摘要"""
        lines = [
            f"🧠 ML信号引擎 v4.0",
            f"  模型: {'✅ 已加载' if self.engine._lgbm_loaded else '⚠️ 未加载 (使用线性IC加权)'}",
            f"  跨市场: {'✅ 已激活' if self.engine.cross_market_available else '⚠️ 不可用'}",
            f"  扫描列表: {', '.join(self.CRYPTO_WATCHLIST)}",
        ]
        return "\n".join(lines)


class RollingAwareAutoTrader:
    """
    滚动训练感知的自动交易器 (Phase 10)

    在 MLAutoTrader 基础上增加:
      1. 交易前检查模型新鲜度
      2. 模型过期 → 自动触发滚动重训
      3. 使用 MLSignalEngineV5 (Qlib融合) 获取更稳健的信号

    使用:
      trader = RollingAwareAutoTrader()
      results, pf = trader.run()
    """

    CRYPTO_WATCHLIST = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ]

    def __init__(self, use_v5: bool = True, use_rolling: bool = True,
                 auto_retrain: bool = False, use_graph: bool = True,
                 use_alphas: bool = False, execution_config: "ExecutionConfig" = None):
        """
        Args:
            use_v5: 是否使用 MLSignalEngineV5 (Qlib融合)
            use_rolling: 是否检查模型新鲜度
            auto_retrain: 模型过期时是否自动重训 (true=全自动, false=仅警告)
            use_graph: 是否使用资产关系图增强 (Phase 11)
            use_alphas: 是否使用Alpha挖掘增强 (Phase 12)
            execution_config: 执行优化配置 (Phase 13), None=单笔市价
        """
        self.use_v5 = use_v5 and ML_V5_AVAILABLE
        self.use_rolling = use_rolling and ROLLING_AVAILABLE
        self.auto_retrain = auto_retrain
        self.use_graph = use_graph and GRAPH_AVAILABLE
        self.use_alphas = use_alphas and ALPHA_AVAILABLE
        self.use_execution = execution_config is not None and EXECUTION_AVAILABLE
        self.execution_config = execution_config

        # 初始化执行引擎
        if self.use_execution:
            self.execution_engine = ExecutionEngine(config=execution_config)
        else:
            self.execution_engine = None

        # 初始化引擎
        if self.use_v5:
            self.engine = MLSignalEngineV5(use_qlib=True, use_lgbm=True, use_graph=self.use_graph,
                                            use_alphas=self.use_alphas)
            label_parts = ["v5 (Qlib融合"]
            if self.use_graph:
                label_parts.append(" + 图增强")
            if self.use_alphas:
                label_parts.append(" + Alpha增强")
            if self.use_execution and self.execution_config:
                label_parts.append(f" + {self.execution_config.strategy.upper()}")
            label_parts.append(")")
            self.engine_label = "".join(label_parts)
        elif ML_SIGNAL_AVAILABLE:
            self.engine = MLSignalEngineV4()
            self.engine_label = "v4 (LightGBM)"

        # 滚动训练器
        if self.use_rolling:
            self.rolling_trainer = RollingTrainer()
            self.rolling_registry = RollingModelRegistry()
        else:
            self.rolling_trainer = None

        # 资产关系图
        self.graph_predictor = None
        if self.use_graph:
            try:
                self.graph_predictor = CrossAssetGATPredictor()
                if self.graph_predictor.snapshot is None:
                    print("🔨 首次运行, 构建资产关系图...")
                    self.graph_predictor.build_graph()
            except Exception as e:
                print(f"⚠️ 图引擎初始化失败: {e}")
                self.use_graph = False

        # Alpha挖掘
        self.alpha_store = None
        if self.use_alphas:
            try:
                self.alpha_store = AlphaStore()
                n = len(self.alpha_store.load("latest"))
                print(f"🔬 Alpha挖掘引擎已加载 ({n} 个Alpha)")
            except Exception as e:
                print(f"⚠️ Alpha引擎加载失败: {e}")
                self.use_alphas = False

        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
            except Exception:
                self._exchange = None
        return self._exchange

    def fetch_ohlcv(self, symbol: str, limit: int = 400) -> pd.DataFrame | None:
        """拉取OHLCV数据"""
        try:
            ex = self.exchange
            if ex is None:
                return None
            ohlcv = ex.fetch_ohlcv(symbol, "1d", limit=limit)
            df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            return df
        except Exception as e:
            print(f"  ⚠️ {symbol} 数据拉取失败: {e}")
            return None

    def pre_trade_check(self) -> dict:
        """
        交易前检查 — 模型健康度 + 新鲜度

        Returns:
            {"ok": bool, "warnings": [...], "model_age_days": int}
        """
        warnings_list = []

        # 检查 Qlib 模型
        if self.use_v5 and self.engine.qlib_predictor:
            n_models = len(self.engine.qlib_predictor._loaded_models)
            if n_models == 0:
                warnings_list.append("⚠️ 未加载任何 Qlib 模型 — 请先训练")
        elif self.use_v5:
            warnings_list.append("⚠️ Qlib 推理器未初始化")

        # 检查滚动模型新鲜度
        if self.use_rolling and self.rolling_trainer:
            status = self.rolling_trainer.status()
            max_age = status["max_age_days"]

            if status["should_retrain"]:
                reason = status["retrain_reason"]
                warnings_list.append(f"🔴 建议重训: {reason}")

                if self.auto_retrain:
                    print(f"🔄 自动触发滚动重训: {reason}")
                    # 拉取数据并重训
                    df_btc = self.fetch_ohlcv("BTC/USDT", limit=800)
                    if df_btc is not None and len(df_btc) >= 200:
                        result = self.rolling_trainer.run(df_btc, force=False, mode="update")
                        if result["action"] == "trained":
                            print(f"  ✅ 自动重训完成: {result['report']['summary']['n_models_trained']} 个模型")
                        else:
                            print(f"  ℹ️ {result['action']}: {result.get('reason', '')}")
                    else:
                        warnings_list.append("⚠️ 自动重训失败: 数据不足")
            elif max_age <= 7:
                print(f"✅ 模型新鲜 ({max_age}天) — 可以交易")
        else:
            max_age = 0

        return {
            "ok": len([w for w in warnings_list if w.startswith("🔴")]) == 0,
            "warnings": warnings_list,
            "model_age_days": max_age,
        }

    def scan(self, symbols: list = None) -> list[dict]:
        """
        扫描多币种 (支持 v4, v5, 和 v5+Graph 引擎)

        Phase 11: 使用资产关系图做多资产联合扫描, 邻居信息增强每个资产的信号

        Returns:
            [{"symbol": "BTC/USDT", "signal": ..., "price": ..., "action": ...}, ...]
        """
        if symbols is None:
            symbols = self.CRYPTO_WATCHLIST

        # Phase 11: 跨资产图增强扫描
        if self.use_graph and self.graph_predictor is not None:
            return self._scan_with_graph(symbols)

        # 标准扫描
        return self._scan_standard(symbols)

    def _scan_with_graph(self, symbols: list) -> list[dict]:
        """使用资产关系图的多资产联合扫描"""
        results = []
        n = len(symbols)
        print(f"🔗 图增强多资产扫描 [{len(symbols)} 个资产]")

        # 获取所有资产的图增强预测
        graph_preds = self.graph_predictor.predict(symbols)
        print(f"  📊 图预测: {len(graph_preds)}/{len(symbols)} 个资产可用")

        for i, symbol in enumerate(symbols):
            print(f"  🔍 [{i+1}/{n}] {symbol}...", end=" ")

            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 200:
                print("数据不足")
                continue

            try:
                if self.use_v5:
                    # v5 融合信号
                    fusion_signal = self.engine.generate_signal(df, symbol)

                    # 图增强调整
                    if symbol in graph_preds and graph_preds[symbol] is not None:
                        graph_pred = graph_preds[symbol]
                        # 图预测方向与融合信号一致 → 提升置信度
                        direction_agree = (graph_pred > 0) == (fusion_signal.signal_consensus > 0)
                        graph_bonus = 1.15 if direction_agree else 0.85

                        fusion_signal.confidence = min(1.0, fusion_signal.confidence * graph_bonus)
                        fusion_signal.signal_consensus *= graph_bonus

                        # 重新判定 action
                        if fusion_signal.signal_consensus > 0.5:
                            fusion_signal.action = "BUY"
                        elif fusion_signal.signal_consensus < -0.5:
                            fusion_signal.action = "SELL"
                        else:
                            fusion_signal.action = "HOLD"

                    price = fusion_signal.price
                    action = fusion_signal.action
                    signal_val = fusion_signal.signal_consensus
                    confidence = fusion_signal.confidence
                    consensus = fusion_signal.consensus_ratio
                    n_active = fusion_signal.n_models_active
                    suggested_size = fusion_signal.suggested_size_pct
                    engine_label = self.engine_label
                else:
                    signal = self.engine.generate_signal(df, symbol, use_lgbm=True)
                    price = float(df["close"].values[-1])
                    action = signal.action
                    signal_val = signal.signal_lgbm if signal.lgbm_available else signal.signal_ic
                    confidence = signal.confidence
                    consensus = signal.consensus
                    n_active = signal.n_sub_signals_active
                    suggested_size = signal.suggested_size_pct / 100
                    engine_label = "v4 (LightGBM + 图增强)"

                graph_info = f" | 图={'✅' if symbol in graph_preds else '❌'}"
                results.append({
                    "symbol": symbol,
                    "name": symbol.replace("/USDT", ""),
                    "price": price,
                    "action": action,
                    "signal_val": signal_val,
                    "confidence": confidence,
                    "consensus": consensus,
                    "n_active": n_active,
                    "suggested_size_pct": suggested_size,
                    "engine": engine_label + graph_info,
                })

                icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                print(f"{icon} {action} | sig={signal_val:+.3f} | "
                      f"置信={confidence:.0%} | {n_active}活跃")

            except Exception as e:
                print(f"信号生成失败: {e}")

        results.sort(key=lambda r: abs(r["signal_val"]), reverse=True)
        return results

    def _scan_standard(self, symbols: list) -> list[dict]:
        """标准扫描 (无图增强)"""
        results = []
        n = len(symbols)

        for i, symbol in enumerate(symbols):
            print(f"  🔍 [{i+1}/{n}] {symbol}...", end=" ")

            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 200:
                print("数据不足")
                continue

            try:
                if self.use_v5:
                    fusion_signal = self.engine.generate_signal(df, symbol)
                    price = fusion_signal.price
                    action = fusion_signal.action
                    signal_val = fusion_signal.signal_consensus
                    confidence = fusion_signal.confidence
                    consensus = fusion_signal.consensus_ratio
                    n_active = fusion_signal.n_models_active
                    suggested_size = fusion_signal.suggested_size_pct
                else:
                    signal = self.engine.generate_signal(df, symbol, use_lgbm=True)
                    price = float(df["close"].values[-1])
                    action = signal.action
                    signal_val = signal.signal_lgbm if signal.lgbm_available else signal.signal_ic
                    confidence = signal.confidence
                    consensus = signal.consensus
                    n_active = signal.n_sub_signals_active
                    suggested_size = signal.suggested_size_pct / 100

                results.append({
                    "symbol": symbol,
                    "name": symbol.replace("/USDT", ""),
                    "price": price,
                    "action": action,
                    "signal_val": signal_val,
                    "confidence": confidence,
                    "consensus": consensus,
                    "n_active": n_active,
                    "suggested_size_pct": suggested_size,
                    "engine": self.engine_label,
                })

                icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                print(f"{icon} {action} | sig={signal_val:+.3f} | "
                      f"置信={confidence:.0%} | {n_active}活跃")

            except Exception as e:
                print(f"信号生成失败: {e}")

        results.sort(key=lambda r: abs(r["signal_val"]), reverse=True)
        return results

    def run(self, symbols: list = None) -> tuple:
        """
        完整交易流程: 检查→扫描→决策→执行

        Returns:
            (results: list, pf: PortfolioManager)
        """
        print(f"🔄 Rolling-Aware 自动交易 [{self.engine_label}]")

        # 1. 交易前检查
        health = self.pre_trade_check()
        if health["warnings"]:
            for w in health["warnings"]:
                print(f"  {w}")

        if not health["ok"] and not self.auto_retrain:
            print("⚠️ 模型不健康且未启用自动重训 — 使用降级模式 (仅 LightGBM)")

        # 2. 初始化风控
        pf = PortfolioManager()
        rc = RiskController(pf)
        total_value = pf.total_value
        results = []

        # 3. 扫描信号
        ml_signals = self.scan(symbols)
        buy_sigs = [s for s in ml_signals if s["action"] == "BUY"]
        sell_sigs = [s for s in ml_signals if s["action"] == "SELL"]

        print(f"\n📊 扫描结果: {len(buy_sigs)} BUY, {len(sell_sigs)} SELL, "
              f"{len(ml_signals) - len(buy_sigs) - len(sell_sigs)} HOLD")

        # 4. 持仓管理 (止损/止盈)
        for pos in pf.open_positions:
            if pos.market != "crypto":
                continue

            # 更新价格
            for sig in ml_signals:
                if sig["symbol"] == pos.symbol:
                    pf.update_price(pos.id, sig["price"])
                    break

            # 止损
            if pos.pnl_pct <= -8.0:
                pf.sell(pos.id, pos.current_price, f"硬止损: {pos.pnl_pct:.1f}%")
                results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")
            # 止盈
            elif pos.pnl_pct >= 15.0:
                pf.sell(pos.id, pos.current_price, f"止盈: +{pos.pnl_pct:.1f}%")
                results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")
            # ML 出场信号
            else:
                for sig in sell_sigs:
                    if sig["symbol"] == pos.symbol:
                        pf.sell(pos.id, pos.current_price,
                               f"ML卖出: {sig['signal_val']:+.3f}")
                        results.append(f"🔴 ML卖出 {pos.symbol}: {pos.pnl_pct:+.1f}%")
                        break

        # 5. 入场信号
        held_symbols = [p.symbol for p in pf.open_positions if p.market == "crypto"]
        for sig in buy_sigs[:3]:
            if sig["symbol"] in held_symbols:
                continue

            cash = pf.cash.get("crypto", 0)
            size_pct = sig.get("suggested_size_pct", 0.05)
            if isinstance(size_pct, float) and size_pct < 1:
                pass  # already decimal
            elif isinstance(size_pct, float):
                size_pct = size_pct / 100
            max_size = min(cash * size_pct, cash * 0.5)

            if max_size < 200:
                continue

            ml_score = abs(sig["signal_val"]) * 30 + sig["confidence"] * 20 + sig["consensus"] * 30
            ml_score = min(100, ml_score)

            check = rc.pre_trade_check("crypto", max_size, ml_score, total_value)
            if not check.passed:
                print(f"  ⚠️ {sig['symbol']} 风控拦截: {check.reason}")
                continue

            quantity = max_size / sig["price"]
            reasoning = (
                f"[{self.engine_label}] 信号={sig['signal_val']:+.3f} | "
                f"置信={sig['confidence']:.0%} | "
                f"活跃={sig['n_active']}"
            )

            # Phase 13: 拆单执行 vs 单笔市价
            if self.use_execution and self.execution_engine is not None:
                # 执行层风控
                mdata = {
                    "price": sig["price"],
                    "avg_daily_volume": 35000 if "BTC" in sig["symbol"] else 500000,
                    "volatility": 0.025,
                    "spread": 0.0005,
                }
                exec_check = rc.execution_risk_check(
                    quantity, mdata["avg_daily_volume"],
                    mdata["spread"], mdata["volatility"],
                    n_slices=self.execution_config.n_slices or 10
                )
                if not exec_check.passed:
                    print(f"  ⚠️ 执行层风控拦截 ({sig['symbol']}): {exec_check.reason}")
                    # 降级为单笔市价
                    print(f"  ⬇️ 降级为单笔市价单")
                    trade = pf.buy(market="crypto", symbol=sig["symbol"],
                                   name=sig["name"], price=sig["price"],
                                   quantity=quantity, reason=reasoning)
                    if trade:
                        results.append(f"🧠 买入 {sig['symbol']} ¥{max_size:.0f} (降级市价) | {reasoning}")
                    continue

                # 拆单执行
                print(f"  🔪 拆单执行 [{self.execution_config.strategy.upper()}] "
                      f"{quantity:.4f} {sig['symbol']}...")
                exec_report = self.execution_engine.execute_paper(
                    symbol=sig["symbol"], side="buy",
                    quantity=quantity, price=sig["price"],
                    market_data=mdata, pf=pf
                )

                if exec_report.fill_rate > 0:
                    results.append(
                        f"🔪 拆单买入 {sig['symbol']} ¥{max_size:.0f} | "
                        f"[{exec_report.strategy_used.upper()}] "
                        f"{exec_report.n_slices_filled}/{exec_report.n_slices_total}片 | "
                        f"IS={exec_report.implementation_shortfall_bps:+.1f}bps | "
                        f"Fill={exec_report.fill_rate:.0%}"
                    )
                    print(f"  ✅ 执行完成: Avg Px=${exec_report.avg_execution_price:,.2f} | "
                          f"IS={exec_report.implementation_shortfall_bps:+.1f}bps | "
                          f"Fill={exec_report.fill_rate:.0%}")
                else:
                    print(f"  ❌ 执行失败: 0% 成交率")
            else:
                # 标准单笔市价
                trade = pf.buy(
                    market="crypto", symbol=sig["symbol"],
                    name=sig["name"],
                    price=sig["price"], quantity=quantity,
                    reason=reasoning,
                )
                if trade:
                    results.append(f"🧠 买入 {sig['symbol']} ¥{max_size:.0f} | {reasoning}")

        pf.take_snapshot()
        return results, pf

    def get_status_summary(self) -> str:
        """引擎状态摘要"""
        lines = [
            f"🔄 Rolling-Aware 自动交易 [{self.engine_label}]",
            f"  Qlib融合: {'✅' if self.use_v5 else '❌'}",
            f"  滚动检查: {'✅' if self.use_rolling else '❌'}",
            f"  自动重训: {'✅' if self.auto_retrain else '❌ (仅警告)'}",
            f"  资产关系图: {'✅' if self.use_graph else '❌'} (Phase 11)",
            f"  订单执行: {'✅ ' + self.execution_config.strategy.upper() if self.use_execution else '❌ 单笔市价'} (Phase 13)",
        ]
        if self.use_execution and self.execution_config:
            lines.append(f"    策略: {self.execution_config.strategy.upper()}")
            lines.append(f"    窗口: {self.execution_config.horizon_minutes}分钟")
            n = self.execution_config.n_slices
            lines.append(f"    切片: {n if n > 0 else '自动最优'}")
            lines.append(f"    紧急度: {self.execution_config.urgency:.1f}")

        if self.use_graph and self.graph_predictor and self.graph_predictor.snapshot:
            snap = self.graph_predictor.snapshot
            lines.append(f"  图资产: {snap.n_assets} 个")
            lines.append(f"  图密度: {snap.graph_density:.4f}")
            lines.append(f"  图社区: {snap.n_communities} 个")
            if snap.top_edges:
                top = snap.top_edges[0]
                lines.append(f"  最强边: {top['source']} ←→ {top['target']} (w={top['weight']:.3f})")

        if self.use_rolling and self.rolling_trainer:
            status = self.rolling_trainer.status()
            lines.append(f"  模型快照: {status['n_snapshots']} 个")
            lines.append(f"  最新训练: {status['latest_training'][:19] if status['latest_training'] != 'never' else '从未'}")
            lines.append(f"  最大年龄: {status['max_age_days']} 天")
            if status["should_retrain"]:
                lines.append(f"  ⚠️ 建议重训: {status['retrain_reason']}")

        lines.append(f"  扫描列表: {', '.join(self.CRYPTO_WATCHLIST)}")
        return "\n".join(lines)


def auto_scan_and_trade(markets: list = None, use_ml: bool = False,
                       use_rolling: bool = False):
    """
    自动扫描 → 决策 → 执行

    Args:
        markets: 市场列表 (默认["crypto"])
        use_ml: True=ML信号引擎, False=传统RSI/MACD引擎
        use_rolling: True=滚动训练感知模式 (Phase 10), 自动检查模型新鲜度
    Returns:
        (results, pf)
    """
    if markets is None:
        markets = ["crypto"]  # 默认只扫24/7市场

    pf = PortfolioManager()
    rc = RiskController(pf)
    total_value = pf.total_value
    results = []

    # ── Rolling-Aware ML模式 (Phase 10) ──
    if use_rolling and "crypto" in markets:
        if not ROLLING_AVAILABLE:
            print("⚠️ 滚动训练引擎不可用, 回退到 v4 模式")
            return auto_scan_and_trade(markets, use_ml=True, use_rolling=False)
        else:
            print("🔄 Rolling-Aware 自动交易模式 (Phase 10)")
            exec_cfg = None
            if hasattr(args, 'execution') and args.execution and EXECUTION_AVAILABLE:
                exec_cfg = ExecutionConfig(
                    strategy=args.execution,
                    horizon_minutes=args.execution_horizon,
                    n_slices=args.execution_slices,
                    urgency=args.execution_urgency,
                )
            trader = RollingAwareAutoTrader(use_v5=True, use_rolling=True, auto_retrain=False,
                                            use_alphas=args.alphas, execution_config=exec_cfg)
            print(trader.get_status_summary())

            results, pf = trader.run()
            return results, pf

    # ── ML模式 ──
    if use_ml and "crypto" in markets:
        if not ML_SIGNAL_AVAILABLE:
            print("⚠️ ML信号引擎不可用, 回退到传统模式")
            use_ml = False
        else:
            print("🧠 ML自动交易模式")
            trader = MLAutoTrader(use_lgbm=True)
            print(trader.get_status_summary())

            ml_signals = trader.scan()
            buy_sigs = [s for s in ml_signals if s["action"] == "BUY"]
            sell_sigs = [s for s in ml_signals if s["action"] == "SELL"]

            # 第一步: 检查持仓止损/止盈
            for pos in pf.open_positions:
                if pos.market != "crypto":
                    continue

                # 更新价格
                for sig in ml_signals:
                    if sig["symbol"] == pos.symbol:
                        pf.update_price(pos.id, sig["price"])
                        break

                # 止损
                if pos.pnl_pct <= -8.0:
                    pf.sell(pos.id, pos.current_price, f"硬止损触发: {pos.pnl_pct:.1f}%")
                    results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")
                # 止盈
                elif pos.pnl_pct >= 15.0:
                    pf.sell(pos.id, pos.current_price, f"止盈触发: +{pos.pnl_pct:.1f}%")
                    results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")
                # ML出场信号
                else:
                    for sig in sell_sigs:
                        if sig["symbol"] == pos.symbol:
                            pf.sell(pos.id, pos.current_price,
                                   f"ML卖出信号: {sig['signal_val']:+.2f}")
                            results.append(f"🔴 ML卖出 {pos.symbol}: {pos.pnl_pct:+.1f}%")
                            break

            # 第二步: ML入场机会
            held_symbols = [p.symbol for p in pf.open_positions if p.market == "crypto"]
            for sig in buy_sigs[:3]:  # Top 3
                if sig["symbol"] in held_symbols:
                    continue

                cash = pf.cash.get("crypto", 0)
                size_pct = sig["signal"].suggested_size_pct / 100
                max_size = min(cash * size_pct, cash * 0.5)

                if max_size < 200:
                    continue

                # ML信号分数 (映射到0-100)
                ml_score = abs(sig["signal_val"]) * 30 + sig["confidence"] * 20 + sig["consensus"] * 30
                ml_score = min(100, ml_score)

                check = rc.pre_trade_check("crypto", max_size, ml_score, total_value)
                if not check.passed:
                    print(f"  ⚠️ {sig['symbol']} 风控拦截: {check.reason}")
                    continue

                quantity = max_size / sig["price"]
                reasoning = (
                    f"ML信号={sig['signal_val']:+.2f} | "
                    f"置信={sig['confidence']:.0%} | "
                    f"活跃={sig['active_themes']}/7主题"
                )
                trade = pf.buy(
                    market="crypto", symbol=sig["symbol"],
                    name=sig["name"],
                    price=sig["price"], quantity=quantity,
                    reason=reasoning,
                )
                if trade:
                    results.append(
                        f"🧠 买入 {sig['symbol']} ¥{max_size:.0f} | {reasoning}"
                    )

            pf.take_snapshot()
            return results, pf

    # ── 传统模式 (RSI/MACD) ──
    engine = SignalEngine()

    for market in markets:
        if not is_market_open(market):
            continue

        scanners = {
            "crypto": engine.crypto,
            "a_stock": engine.a_stock,
            "us_stock": engine.us_stock,
        }
        scanner = scanners.get(market)
        if not scanner:
            continue

        all_sigs = scanner.scan()
        buy_sigs = [s for s in all_sigs if s.action == "BUY" and s.score >= 65]
        buy_sigs.sort(key=lambda s: s.score, reverse=True)

        # ── 第一步: 检查持仓止损/止盈 ──
        for pos in pf.open_positions:
            if pos.market != market or pos.status != "closed":
                continue
            for sig in all_sigs:
                if sig.symbol == pos.symbol:
                    pf.update_price(pos.id, sig.price)
                    break

            if pos.pnl_pct <= -8.0:
                pf.sell(pos.id, pos.current_price, f"硬止损触发: {pos.pnl_pct:.1f}%")
                results.append(f"🔴 止损 {pos.symbol}: {pos.pnl_pct:.1f}%")
            elif pos.pnl_pct >= 15.0:
                pf.sell(pos.id, pos.current_price, f"止盈触发: +{pos.pnl_pct:.1f}%")
                results.append(f"🟢 止盈 {pos.symbol}: +{pos.pnl_pct:.1f}%")

        # ── 第二步: 寻找新的入场机会 ──
        for sig in buy_sigs[:3]:
            held_symbols = [p.symbol for p in pf.open_positions if p.market == market]
            if sig.symbol in held_symbols:
                continue

            cash = pf.cash.get(market, 0)
            max_size = min(sig.suggested_size, cash * 0.5)

            if max_size < 200:
                continue

            check = rc.pre_trade_check(market, max_size, sig.score, total_value)
            if not check.passed:
                continue

            quantity = max_size / sig.price
            trade = pf.buy(
                market=market, symbol=sig.symbol, name=sig.name,
                price=sig.price, quantity=quantity,
                reason=" | ".join(sig.reasons),
            )
            if trade:
                results.append(
                    f"🟢 买入 {sig.symbol} ¥{max_size:.0f} | "
                    f"评分{sig.score:.0f} | {sig.reasons[0]}"
                )

    pf.take_snapshot()

    return results, pf


def generate_status_report() -> str:
    """生成状态报告"""
    pf = PortfolioManager()
    rc = RiskController(pf)

    lines = [
        "=" * 50,
        f"🐾 Chase量化策略 · 状态报告",
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 50,
        f"",
        f"💰 总资产: ¥{pf.total_value:,.2f} | 盈亏: {pf.total_pnl_pct:+.2f}%",
        f"💵 现金: ¥{pf.total_cash:,.2f} | 持仓: ¥{pf.positions_value:,.2f}",
        f"",
    ]

    # 持仓
    open_pos = pf.open_positions
    if open_pos:
        lines.append("📦 当前持仓:")
        for p in open_pos:
            emoji = "🟢" if p.pnl_pct >= 0 else "🔴"
            lines.append(f"  {emoji} {p.symbol} | ¥{p.entry_price:,.2f}→¥{p.current_price:,.2f} | {p.pnl_pct:+.2f}% | ¥{p.value:,.0f}")
    else:
        lines.append("📦 无持仓")

    lines.append("")

    # 风控
    alerts = pf.check_risk()
    if alerts:
        lines.append("🚨 风控告警:")
        for a in alerts:
            lines.append(f"  {a}")
    else:
        lines.append("✅ 风控正常")

    lines.append("")
    lines.append(f"🎯 月度目标: {pf.total_pnl_pct:+.1f}%/30%")

    # 偏差修正信息
    if BIAS_AWARE:
        bias = estimate_survival_bias("crypto")
        lines.append(f"🔍 偏差修正: 回测×{bias['correction_factor']} (虚高{bias['estimated_overstatement_pct']}%)")

    # ML状态 (Phase 7/10)
    if ROLLING_AVAILABLE:
        lines.append("🔄 Rolling引擎: ✅ 可用 (使用 --rolling 启动 Phase 10)")
    elif ML_SIGNAL_AVAILABLE:
        lines.append("🧠 ML引擎: ✅ 可用 (使用 --ml 启动)")
    else:
        lines.append("🧠 ML引擎: ⚠️ 不可用")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chase量化策略 · 自动交易")
    parser.add_argument("--markets", nargs="+", default=["crypto"],
                       choices=["crypto", "a_stock", "us_stock", "all"])
    parser.add_argument("--report", action="store_true",
                       help="仅生成状态报告")
    parser.add_argument("--ml", action="store_true",
                       help="使用ML信号引擎 (Phase 7)")
    parser.add_argument("--rolling", action="store_true",
                       help="使用滚动训练感知模式 (Phase 10, Qlib融合+模型新鲜度检查)")
    parser.add_argument("--alphas", action="store_true",
                       help="使用Alpha挖掘增强 (Phase 12)")
    parser.add_argument("--execution", type=str, default=None,
                       choices=["twap", "vwap", "adaptive", "iceberg", "smart"],
                       help="订单执行策略 (Phase 13, 默认=单笔市价)")
    parser.add_argument("--execution-horizon", type=int, default=60,
                       help="执行时间窗口 (分钟, 默认60)")
    parser.add_argument("--execution-slices", type=int, default=0,
                       help="切片数 (0=自动最优, 默认0)")
    parser.add_argument("--execution-urgency", type=float, default=0.5,
                       help="执行紧急度 0-1 (默认0.5)")
    parser.add_argument("--ml-scan", action="store_true",
                       help="仅ML扫描, 不执行交易 (调试用)")
    parser.add_argument("--wechat-report", action="store_true",
                       help="交易后推送日报到企业微信 (Phase 14)")
    args = parser.parse_args()

    if args.report:
        print(generate_status_report())
        sys.exit(0)

    # ML调试模式: 仅扫描
    if args.ml_scan:
        if ROLLING_AVAILABLE:
            # 构建执行配置
            exec_cfg = None
            if args.execution and EXECUTION_AVAILABLE:
                exec_cfg = ExecutionConfig(
                    strategy=args.execution,
                    horizon_minutes=args.execution_horizon,
                    n_slices=args.execution_slices,
                    urgency=args.execution_urgency,
                )
            print("🔄 Rolling-Aware 扫描模式\n")
            trader = RollingAwareAutoTrader(use_v5=True, use_rolling=True, auto_retrain=False,
                                            use_alphas=args.alphas, execution_config=exec_cfg)
            print(trader.get_status_summary())
            print()
            signals = trader.scan()
            print(f"\n📊 扫描结果 ({len(signals)}个):")
            for s in signals:
                icon = "🟢" if s["action"] == "BUY" else "🔴" if s["action"] == "SELL" else "⚪"
                print(f"  {icon} {s['symbol']:12s} | {s['action']:4s} | "
                      f"信号={s['signal_val']:+.3f} | "
                      f"置信={s['confidence']:.0%} | "
                      f"共识={s['consensus']:.0%} | "
                      f"{s['n_active']}活跃 | [{s['engine']}]")
        elif ML_SIGNAL_AVAILABLE:
            print("🧠 ML扫描模式 (仅查看信号)\n")
            trader = MLAutoTrader(use_lgbm=True)
            print(trader.get_status_summary())
            print()
            signals = trader.scan()
            print(f"\n📊 扫描结果 ({len(signals)}个):")
            for s in signals:
                icon = "🟢" if s["action"] == "BUY" else "🔴" if s["action"] == "SELL" else "⚪"
                print(f"  {icon} {s['symbol']:12s} | {s['action']:4s} | "
                      f"信号={s['signal_val']:+.3f} | "
                      f"置信={s['confidence']:.0%} | "
                      f"共识={s['consensus']:.0%} | "
                      f"{s['active_themes']}/7主题")
        else:
            print("❌ ML信号引擎不可用")
        sys.exit(0)

    markets = ["crypto", "a_stock", "us_stock"] if args.markets == ["all"] else args.markets
    results, pf = auto_scan_and_trade(markets, use_ml=args.ml, use_rolling=args.rolling)

    print(generate_status_report())

    if results:
        print("\n📋 本次操作:")
        for r in results:
            print(f"  {r}")
    else:
        print("\n💤 本次扫描无操作")

    # Phase 14: 交易后推送企业微信简报
    if args.wechat_report:
        print("\n📱 推送交易简报到企业微信...")
        try:
            from wechat_report import WeChatPusher, DataCollector
            pusher = WeChatPusher()
            collector = DataCollector()

            # 构建精简版交易简报
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            pf_local = PortfolioManager()

            lines = [
                f"📊 **Yina交易简报** | {now_str}",
                f"> Chase哥的AI量化助手 · 自动交易通知",
                "",
                "## 🏷️ 本次操作",
            ]

            if results:
                for r in results[:10]:
                    lines.append(f"> {r}")
            else:
                lines.append("> 💤 本次扫描无操作")

            lines.append("")
            lines.append("## 📈 当前持仓")
            open_pos = pf_local.open_positions
            if open_pos:
                for pos in open_pos[:5]:
                    pnl_str = f"{pos.pnl_pct:+.2f}%"
                    lines.append(f"> {pos.symbol}: {pnl_str} | 入场¥{pos.entry_price:,.0f} → 现价¥{pos.current_price:,.0f}")
            else:
                lines.append("> 当前无持仓")

            lines.append("")
            lines.append(f"> 🛡️ 总资产: ¥{pf_local.total_value:,.2f} | 总盈亏: {pf_local.total_pnl_pct:+.2f}%")
            lines.append("")
            lines.append("---")
            lines.append(f"🐾 Yina Quant v2.5 · {now_str}")

            brief = "\n".join(lines)
            success = pusher.push_markdown(brief)
            if success:
                print("✅ 交易简报已推送到企业微信!")
            else:
                print("⚠️ 推送失败 (不影响交易结果)")
        except Exception as e:
            print(f"⚠️ 推送失败: {e} (不影响交易结果)")
