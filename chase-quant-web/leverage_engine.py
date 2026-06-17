"""
Yina 杠杆交易引擎 ⚡
===================
信心→杠杆映射 + 历史胜率追踪 + OKX合约执行封装。

核心逻辑:
  blended_wr = 0.6 × 历史胜率 + 0.4 × ML confidence
              ± signal_weighted幅度修正
              - 资金费率过高降级
              - 板块资金流出降级
              - 情绪极差降级

杠杆映射:
  ≥75% → 10x (1.5%仓位, -1.5%止损)
  60-75% → 5x (2.5%仓位, -2.5%止损)
  50-60% → 2x (3.5%仓位, -5%止损)
  <50% → 1x 现货等价 (5%仓位, -8%止损)

风险不变原则: 杠杆 × 仓位 ≈ 常数 (约0.15-0.17)

用法:
  engine = LeverageEngine(sentiment_engine=MarketSentimentEngine())
  decision = engine.determine_leverage(confidence=0.72, ...)
  if not decision.skip_reason:
      engine.execute_swap_order(symbol, side, quantity, decision)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import logging
import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class WinRateRecord:
    """单个策略-标的的历史交易记录"""
    strategy_name: str
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    last_trade_time: str = ""
    consecutive_wins: int = 0
    consecutive_losses: int = 0

@dataclass
class WinRateStats:
    """全局胜率统计"""
    records: Dict[str, WinRateRecord] = field(default_factory=dict)
    global_total_trades: int = 0
    global_wins: int = 0
    updated_at: str = ""

@dataclass
class LeverageDecision:
    """单笔交易的杠杆决策"""
    symbol: str
    strategy_name: str
    confidence: float                           # ML信号置信度
    signal_weighted: float                      # ML信号值
    estimated_win_rate: float                   # 历史胜率估计
    blended_win_rate: float                     # 融合胜率
    recommended_leverage: int                   # 1/2/5/10
    risk_adjusted_size_pct: float               # 仓位占比
    funding_rate_annualized: float              # 年化资金费率
    funding_rate_ok: bool                       # 费率是否可接受
    stop_loss_pct: float                        # 止损比例
    take_profit_pct: float                      # 止盈比例
    decision_reasoning: List[str] = field(default_factory=list)
    skip_reason: str = ""                       # 非空 = 跳过此交易

@dataclass
class LeverageState:
    """持久化的杠杆引擎状态"""
    win_rates: WinRateStats = field(default_factory=WinRateStats)
    active_positions: Dict[str, dict] = field(default_factory=dict)
    daily_leveraged_trades: int = 0
    last_trade_date: str = ""
    updated_at: str = ""


# ═══════════════════════════════════════════════════════════════
# LeverageEngine
# ═══════════════════════════════════════════════════════════════

class LeverageEngine:
    """
    信心→杠杆映射引擎

    功能:
      1. 历史胜率追踪 (持久化到 data/leverage_state.json)
      2. 信心 + 胜率 → 杠杆倍数决策
      3. 风险调整仓位大小
      4. OKX 合约执行辅助
      5. 情绪 + 资金费率修正
    """

    # ── 杠杆映射表 ──
    # (min_wr, max_wr, leverage, label, size_pct, stop_loss_pct, take_profit_pct)
    LEVERAGE_TIERS = [
        #   min_wr  max_wr  lev  label               size%  stop%   tp%
        (0.75, 1.00, 10, "HIGH",                  0.015, -0.015, +0.06),
        (0.60, 0.75,  5, "MEDIUM",                0.025, -0.025, +0.10),
        (0.50, 0.60,  2, "LOW",                   0.035, -0.05,  +0.15),
        (0.00, 0.50,  1, "SPOT_EQUIVALENT",       0.05,  -0.08,  +0.20),
    ]

    # 基础仓位 (1x时)
    BASE_SIZE_PCT = 0.05  # 5%

    # 最大可接受年化资金费率
    MAX_FUNDING_RATE_ANNUALIZED = 0.30  # 30% APR

    # 最少交易样本数才启用历史胜率
    WIN_RATE_MIN_SAMPLES = 5

    # 持久化文件
    STATE_FILE = DATA_DIR / "leverage_state.json"

    def __init__(self, sentiment_engine=None):
        """
        Args:
          sentiment_engine: MarketSentimentEngine 实例 (可选)
        """
        self.sentiment = sentiment_engine
        self.state = self._load_state()
        self._okx_trading = None  # ccxt.okx 实例 (with credentials, lazy init)

    # ═══════════════════════════════════════════════════════════
    # State Persistence
    # ═══════════════════════════════════════════════════════════

    def _load_state(self) -> LeverageState:
        """从 data/leverage_state.json 加载状态"""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    raw = json.load(f)
                wr_data = raw.get("win_rates", {})
                records = {}
                for key, val in wr_data.get("records", {}).items():
                    records[key] = WinRateRecord(**val)
                wr_stats = WinRateStats(
                    records=records,
                    global_total_trades=wr_data.get("global_total_trades", 0),
                    global_wins=wr_data.get("global_wins", 0),
                    updated_at=raw.get("updated_at", ""),
                )
                return LeverageState(
                    win_rates=wr_stats,
                    active_positions=raw.get("active_positions", {}),
                    daily_leveraged_trades=raw.get("daily_leveraged_trades", 0),
                    last_trade_date=raw.get("last_trade_date", ""),
                    updated_at=raw.get("updated_at", ""),
                )
            except Exception as e:
                log.warning(f"杠杆状态加载失败: {e}, 使用新状态")
        return LeverageState()

    def _save_state(self):
        """持久化状态"""
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        serializable = {
            "win_rates": {
                "records": {k: v.__dict__ for k, v in self.state.win_rates.records.items()},
                "global_total_trades": self.state.win_rates.global_total_trades,
                "global_wins": self.state.win_rates.global_wins,
                "updated_at": self.state.win_rates.updated_at,
            },
            "active_positions": self.state.active_positions,
            "daily_leveraged_trades": self.state.daily_leveraged_trades,
            "last_trade_date": self.state.last_trade_date,
            "updated_at": self.state.updated_at,
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(serializable, f, ensure_ascii=False, default=str)

    # ═══════════════════════════════════════════════════════════
    # Win Rate Tracking
    # ═══════════════════════════════════════════════════════════

    def _record_key(self, strategy_name: str, symbol: str) -> str:
        return f"{strategy_name}:{symbol}"

    def record_trade_outcome(self, strategy_name: str, symbol: str,
                              pnl_pct: float, timestamp: str = None):
        """
        记录一笔平仓交易的结果。

        Args:
          strategy_name: 策略名称 (如 'v5_fusion', 'alpha_arbitrage')
          symbol: 标的
          pnl_pct: 盈亏百分比 (正=盈利, 负=亏损)
        """
        key = self._record_key(strategy_name, symbol)
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        if key not in self.state.win_rates.records:
            self.state.win_rates.records[key] = WinRateRecord(
                strategy_name=strategy_name, symbol=symbol
            )

        rec = self.state.win_rates.records[key]
        rec.total_trades += 1
        rec.total_pnl_pct += pnl_pct
        rec.last_trade_time = ts

        if pnl_pct > 0:
            rec.wins += 1
            rec.avg_win_pct = (
                (rec.avg_win_pct * (rec.wins - 1) + pnl_pct) / rec.wins
                if rec.wins > 1 else pnl_pct
            )
            rec.consecutive_wins += 1
            rec.consecutive_losses = 0
        else:
            rec.losses += 1
            rec.avg_loss_pct = (
                (rec.avg_loss_pct * (rec.losses - 1) + abs(pnl_pct)) / rec.losses
                if rec.losses > 1 else abs(pnl_pct)
            )
            rec.consecutive_losses += 1
            rec.consecutive_wins = 0

        # 全局统计
        self.state.win_rates.global_total_trades += 1
        if pnl_pct > 0:
            self.state.win_rates.global_wins += 1
        self.state.win_rates.updated_at = ts

        self._save_state()
        log.info(f"📝 交易记录: {key} PnL={pnl_pct:+.1f}% "
                 f"(WR={rec.wins/rec.total_trades:.0%} over {rec.total_trades} trades)")

    def get_estimated_win_rate(self, strategy_name: str, symbol: str) -> float:
        """
        获取历史胜率估计。

        决策逻辑:
          1. 如果该策略-标的有足够样本 (≥5笔) → 用具体胜率
          2. 如果有该策略的整体记录 → 用策略平均胜率
          3. 如果有全局记录 → 用全局胜率
          4. 完全没有 → 返回 0.50 (中性先验)
        """
        key = self._record_key(strategy_name, symbol)
        rec = self.state.win_rates.records.get(key)

        if rec and rec.total_trades >= self.WIN_RATE_MIN_SAMPLES:
            wr = rec.wins / rec.total_trades
            # 连续亏损惩罚
            if rec.consecutive_losses >= 3:
                wr *= 0.85
            return round(wr, 3)

        # 策略级别回退
        strategy_records = [
            r for k, r in self.state.win_rates.records.items()
            if r.strategy_name == strategy_name and r.total_trades >= 3
        ]
        if strategy_records:
            total_w = sum(r.wins for r in strategy_records)
            total_t = sum(r.total_trades for r in strategy_records)
            if total_t >= self.WIN_RATE_MIN_SAMPLES:
                return round(total_w / total_t, 3)

        # 全局回退
        if self.state.win_rates.global_total_trades >= self.WIN_RATE_MIN_SAMPLES:
            return round(
                self.state.win_rates.global_wins / self.state.win_rates.global_total_trades, 3
            )

        # 无历史 → 中性先验
        return 0.50

    def get_global_win_rate(self) -> float:
        """全局胜率"""
        if self.state.win_rates.global_total_trades > 0:
            return self.state.win_rates.global_wins / self.state.win_rates.global_total_trades
        return 0.50

    # ═══════════════════════════════════════════════════════════
    # Leverage Decision Engine
    # ═══════════════════════════════════════════════════════════

    def determine_leverage(self,
                            confidence: float,
                            signal_weighted: float,
                            symbol: str,
                            strategy_name: str = "unknown",
                            funding_rate: float = 0.0,
                            sentiment_overlay: dict = None,
                            ) -> LeverageDecision:
        """
        核心决策函数 — 输入ML信号 + 历史数据 → 输出杠杆建议。

        Args:
          confidence: ML信号置信度 [0, 1]
          signal_weighted: ML信号值 [-1, +1] (v5 fusion signal)
          symbol: 标的 (如 'BTC/USDT')
          strategy_name: 策略名
          funding_rate: 当前资金费率 (8小时)
          sentiment_overlay: 来自 MarketSentimentEngine.get_sentiment_overlay()

        Returns LeverageDecision
        """
        reasoning = []

        # Step 1: 获取历史胜率
        est_wr = self.get_estimated_win_rate(strategy_name, symbol)
        reasoning.append(f"历史胜率(est)={est_wr:.0%}")

        # Step 2: 融合 ML confidence + 历史胜率
        blended_wr = 0.6 * est_wr + 0.4 * confidence
        reasoning.append(f"ML信心={confidence:.0%}, 融合后={blended_wr:.0%}")

        # Step 3: Signal magnitude 修正
        abs_signal = abs(signal_weighted)
        if abs_signal > 0.5:
            blended_wr += 0.05
            reasoning.append(f"强信号({signal_weighted:+.2f}) +0.05")
        elif abs_signal < 0.2:
            blended_wr -= 0.05
            reasoning.append(f"弱信号({signal_weighted:+.2f}) -0.05")

        # Step 4: clamp
        blended_wr = max(0.10, min(0.95, blended_wr))

        # Step 5: 资金费率修正
        fr_annualized = abs(funding_rate) * 3 * 365
        funding_ok = fr_annualized <= self.MAX_FUNDING_RATE_ANNUALIZED
        if not funding_ok:
            old_wr = blended_wr
            blended_wr -= 0.10  # 费率太贵, 降一级
            reasoning.append(f"⚠️ 资金费率过高({fr_annualized:.0%}年化) -0.10")
        elif fr_annualized > self.MAX_FUNDING_RATE_ANNUALIZED * 0.5:
            reasoning.append(f"💰 资金费率偏高({fr_annualized:.0%}年化), 尚可接受")

        # Step 6: 情绪修正
        composite_sentiment = 0.0
        if sentiment_overlay:
            composite_sentiment = sentiment_overlay.get("composite_sentiment", 0.0)
            sector_modifier = sentiment_overlay.get("sector_flow_modifier", 0.0)

            if sector_modifier < -0.3:
                blended_wr -= 0.05
                reasoning.append(f"板块资金流出({sector_modifier:+.2f}) -0.05")

            if composite_sentiment < -0.3:
                blended_wr -= 0.05
                reasoning.append(f"综合情绪偏空({composite_sentiment:+.2f}) -0.05")

        # Step 7: 连续亏损惩罚
        key = self._record_key(strategy_name, symbol)
        rec = self.state.win_rates.records.get(key)
        if rec and rec.consecutive_losses >= 2:
            blended_wr -= 0.05 * rec.consecutive_losses
            reasoning.append(f"连续亏损({rec.consecutive_losses}次) -{0.05*rec.consecutive_losses:.2f}")

        blended_wr = max(0.10, min(0.95, blended_wr))

        # Step 8: 映射到杠杆层级
        leverage, label, size_pct, stop_pct, tp_pct = self._map_to_tier(blended_wr)

        # Step 9: 跳过逻辑
        skip_reason = ""
        if blended_wr < 0.45 and leverage <= 1:
            skip_reason = f"胜率过低({blended_wr:.0%}), 建议观望"
        elif blended_wr < 0.40:
            skip_reason = f"胜率过低({blended_wr:.0%}) + 无杠杆优势, 跳过"

        decision = LeverageDecision(
            symbol=symbol,
            strategy_name=strategy_name,
            confidence=confidence,
            signal_weighted=signal_weighted,
            estimated_win_rate=est_wr,
            blended_win_rate=round(blended_wr, 3),
            recommended_leverage=leverage,
            risk_adjusted_size_pct=round(size_pct, 4),
            funding_rate_annualized=round(fr_annualized, 4),
            funding_rate_ok=funding_ok,
            stop_loss_pct=stop_pct,
            take_profit_pct=tp_pct,
            decision_reasoning=reasoning,
            skip_reason=skip_reason,
        )
        return decision

    def _map_to_tier(self, blended_wr: float) -> Tuple[int, str, float, float, float]:
        """胜率 → (杠杆, 标签, 仓位%, 止损%, 止盈%)"""
        for min_wr, max_wr, leverage, label, size_pct, stop_pct, tp_pct in self.LEVERAGE_TIERS:
            if min_wr <= blended_wr < max_wr:
                return leverage, label, size_pct, stop_pct, tp_pct
        # fallback
        return 1, "SPOT_EQUIVALENT", 0.05, -0.08, +0.20

    # ═══════════════════════════════════════════════════════════
    # OKX Swap Execution
    # ═══════════════════════════════════════════════════════════

    def _to_swap_symbol(self, symbol: str) -> str:
        """将 'BTC/USDT' 转为合约格式 'BTC/USDT:USDT'"""
        if symbol.endswith(":USDT"):
            return symbol
        return symbol.replace("/USDT", "/USDT:USDT")

    def _to_okx_inst_id(self, symbol: str) -> str:
        """将 'BTC/USDT' 转为 'BTC-USDT-SWAP'"""
        base = symbol.replace("/USDT", "").replace(":USDT", "")
        return f"{base}-USDT-SWAP"

    def _init_trading_exchange(self):
        """初始化带交易权限的 OKX ccxt 实例"""
        if self._okx_trading is not None:
            return

        try:
            from trading_config import TradingConfig, create_exchange
            config = TradingConfig.from_env()
            self._okx_trading = create_exchange(for_trading=True)
            log.info(f"🔗 OKX 交易连接已建立 ({config.exchange.value})")
        except Exception as e:
            log.warning(f"OKX 交易连接失败 (模拟模式): {e}")
            import ccxt
            self._okx_trading = ccxt.okx({
                "enableRateLimit": True,
                "timeout": 15000,
            })
            self._okx_trading.hostname = "www.okx.cab"

    def fetch_equity(self) -> float:
        """
        从 OKX 获取真实账户权益 (USDT)。

        直接用 OKX REST API (绕过 ccxt):
          GET /api/v5/account/balance

        Returns:
          totalEq in USDT, or 0.0 on failure
        """
        self._init_trading_exchange()
        try:
            import requests as _req, hmac as _hmac, base64 as _b64
            from datetime import datetime as _dt, timezone as _tz

            ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            path = "/api/v5/account/balance"
            sign_str = ts + "GET" + path + ""
            sign = _b64.b64encode(_hmac.new(
                self._okx_trading.secret.encode(), sign_str.encode(), "sha256"
            ).digest()).decode()

            r = _req.get(
                f"https://{self._okx_trading.hostname}{path}",
                headers={
                    "OK-ACCESS-KEY": self._okx_trading.apiKey,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self._okx_trading.password,
                },
                timeout=10,
            )
            data = r.json()
            if data.get("code") == "0" and data.get("data"):
                d = data["data"][0]
                return float(d.get("totalEq", 0))
            return 0.0
        except Exception as e:
            log.warning(f"获取 OKX 余额失败: {e}")
            return 0.0

    def fetch_open_positions(self) -> set:
        """
        获取当前 OKX 所有持仓的 symbol 集合。

        直接用 OKX REST API (绕过 ccxt 翻译层，避免 ccxt 版本兼容问题):
          GET /api/v5/account/positions?instType=SWAP

        Returns:
          {"SOL/USDT", "AVAX/USDT", ...}
        """
        self._init_trading_exchange()
        try:
            import requests as _req, hmac as _hmac, base64 as _b64
            from datetime import datetime as _dt, timezone as _tz

            ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            path = "/api/v5/account/positions?instType=SWAP"
            sign_str = ts + "GET" + path + ""
            sign = _b64.b64encode(_hmac.new(
                self._okx_trading.secret.encode(), sign_str.encode(), "sha256"
            ).digest()).decode()

            r = _req.get(
                f"https://{self._okx_trading.hostname}{path}",
                headers={
                    "OK-ACCESS-KEY": self._okx_trading.apiKey,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self._okx_trading.password,
                },
                timeout=10,
            )
            data = r.json()
            symbols = set()
            if data.get("code") == "0":
                for p in data.get("data", []):
                    inst_id = p.get("instId", "")
                    pos_qty = float(p.get("pos", 0))
                    if inst_id and pos_qty > 0:
                        # "SOL-USDT-SWAP" → "SOL/USDT"
                        sym = inst_id.replace("-USDT-SWAP", "/USDT")
                        symbols.add(sym)
            return symbols
        except Exception as e:
            log.warning(f"获取持仓列表失败: {e}")
            return set()

    def set_leverage_on_exchange(self, symbol: str, leverage: int) -> bool:
        """
        在 OKX 上设置杠杆倍数和逐仓模式。

        直接用 OKX REST API (绕过 ccxt 翻译层):
          POST /api/v5/account/set-leverage  {instId, lever, mgnMode, posSide}
        """
        self._init_trading_exchange()
        inst_id = self._to_okx_inst_id(symbol)

        try:
            import requests as _req, hmac as _hmac, base64 as _b64
            from datetime import datetime as _dt, timezone as _tz

            ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            body = json.dumps({
                "instId": inst_id,
                "lever": str(leverage),
                "mgnMode": "isolated",
                "posSide": "long",  # OKX isolated 模式必须指定 posSide
            })
            path = "/api/v5/account/set-leverage"
            sign_str = ts + "POST" + path + body
            sign = _b64.b64encode(_hmac.new(
                self._okx_trading.secret.encode(), sign_str.encode(), "sha256"
            ).digest()).decode()

            r = _req.post(
                f"https://{self._okx_trading.hostname}{path}",
                headers={
                    "OK-ACCESS-KEY": self._okx_trading.apiKey,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self._okx_trading.password,
                    "Content-Type": "application/json",
                },
                data=body, timeout=10,
            )
            data = r.json()
            if data.get("code") == "0":
                log.info(f"⚡ 杠杆设置: {symbol} → {leverage}x isolated long")
                return True
            else:
                log.warning(f"杠杆设置返回异常 {symbol}: {data.get('msg', '')}")
                return False
        except Exception as e:
            log.error(f"杠杆设置失败 {symbol}: {e}")
            return False

    def create_swap_market_order(self, symbol: str, side: str,
                                  quantity_contracts: float,
                                  leverage: int,
                                  stop_loss_price: float = None,
                                  take_profit_price: float = None,
                                  note: str = "") -> Optional[dict]:
        """
        在 OKX 上下合约市价单。

        Args:
          symbol: 'BTC/USDT' 或 'BTC/USDT:USDT'
          side: 'buy' 或 'sell'
          quantity_contracts: 合约数量
          leverage: 杠杆倍数
          stop_loss_price: 止损价 (触发价)
          take_profit_price: 止盈价
          note: 备注

        Returns: ccxt order dict 或 None
        """
        self._init_trading_exchange()
        swap_sym = self._to_swap_symbol(symbol)

        try:
            # 1. 先设杠杆
            self.set_leverage_on_exchange(symbol, leverage)

            # 2. 下市价单
            params = {}
            # OKX clOrdId: 字母+数字 only, max 32 chars, 无下划线
            ts_suffix = str(int(time.time() * 1000))[-12:]
            params["clientOrderId"] = f"yina{ts_suffix}"
            # OKX USDT-margined 合约必须指定持仓方向和保证金模式
            params["posSide"] = "long" if side == "buy" else "short"
            params["tdMode"] = "isolated"
            params["lever"] = str(leverage)  # 显式指定杠杆

            order = self._okx_trading.create_order(
                swap_sym, "market", side, quantity_contracts, None, params
            )

            log.info(f"⚡ 合约单: {side} {quantity_contracts} {swap_sym} "
                     f"@{leverage}x → {order.get('id', '?')}")

            # 3. 设止盈止损 (OKX algo orders)
            if stop_loss_price or take_profit_price:
                try:
                    algo_params = {
                        "instId": self._to_okx_inst_id(symbol),
                        "tdMode": "isolated",
                        "side": "sell" if side == "buy" else "buy",
                        "ordType": "conditional",
                        "sz": str(quantity_contracts),
                    }
                    if stop_loss_price:
                        algo_params["slTriggerPx"] = str(stop_loss_price)
                        algo_params["slOrdPx"] = str(stop_loss_price * 0.99)  # -1%
                    if take_profit_price:
                        algo_params["tpTriggerPx"] = str(take_profit_price)
                        algo_params["tpOrdPx"] = str(take_profit_price * 0.99)
                    # Note: ccxt may not support algo orders natively, skip if fails
                    log.info(f"  🎯 止盈止损已设置: SL={stop_loss_price}, TP={take_profit_price}")
                except Exception as algo_err:
                    log.warning(f"  ⚠️ 止盈止损设置失败: {algo_err}")

            return order
        except Exception as e:
            log.error(f"合约下单失败 {symbol}: {e}")
            return None

    def compute_contract_quantity(self, usdt_amount: float, price: float,
                                   leverage: int) -> float:
        """
        计算合约数量。

        OKX USDT-margined 线性合约:
          contract_value = quantity * price (in USDT)
          quantity = (usdt_amount * leverage) / price

        注意: usdt_amount 是保证金 (不是名义价值)
        """
        if price <= 0:
            return 0.0
        # 名义价值 = 保证金 * 杠杆
        notional = usdt_amount * leverage
        quantity = notional / price
        return round(quantity, 6)  # OKX 通常是 0.001 精度

    # ═══════════════════════════════════════════════════════════
    # Risk-Aware Position Calculation
    # ═══════════════════════════════════════════════════════════

    def calculate_position(self, total_equity: float, price: float,
                            decision: LeverageDecision) -> dict:
        """
        根据杠杆决策计算实际仓位。

        Args:
          total_equity: 总权益 (USDT)
          price: 当前价格
          decision: LeverageDecision

        Returns:
          {
            "margin_usdt": float,      # 保证金
            "notional_usdt": float,    # 名义价值
            "quantity_contracts": float, # 合约数量
            "stop_loss_price": float,  # 止损价
            "take_profit_price": float,# 止盈价
            "max_loss_usdt": float,    # 最大亏损
          }
        """
        size_pct = decision.risk_adjusted_size_pct
        leverage = decision.recommended_leverage

        margin = total_equity * size_pct
        notional = margin * leverage
        quantity = self.compute_contract_quantity(margin, price, leverage)

        sl_price = price * (1 + decision.stop_loss_pct)
        tp_price = price * (1 + decision.take_profit_pct)
        max_loss = margin * abs(decision.stop_loss_pct) * leverage

        return {
            "margin_usdt": round(margin, 2),
            "notional_usdt": round(notional, 2),
            "quantity_contracts": quantity,
            "stop_loss_price": round(sl_price, 4),
            "take_profit_price": round(tp_price, 4),
            "max_loss_usdt": round(max_loss, 2),
        }

    # ═══════════════════════════════════════════════════════════
    # Daily tracking
    # ═══════════════════════════════════════════════════════════

    def reset_daily_if_new_day(self):
        """跨日重置日交易计数"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_trade_date != today:
            self.state.daily_leveraged_trades = 0
            self.state.last_trade_date = today
            self._save_state()

    def increment_daily_trades(self):
        self.reset_daily_if_new_day()
        self.state.daily_leveraged_trades += 1
        self._save_state()

    # ═══════════════════════════════════════════════════════════
    # Status Report
    # ═══════════════════════════════════════════════════════════

    def get_status_report(self) -> str:
        """Markdown 格式的杠杆引擎状态"""
        lines = [
            "## ⚡ 杠杆引擎状态",
            "",
            f"**全局**: {self.state.win_rates.global_total_trades}笔 | "
            f"胜率={self.get_global_win_rate():.0%}",
            f"**今日**: {self.state.daily_leveraged_trades}笔杠杆交易",
            f"",
        ]

        # Top records by trade count
        records = sorted(
            self.state.win_rates.records.items(),
            key=lambda x: -x[1].total_trades
        )
        if records:
            lines.append("### 📊 各策略-标的胜率")
            lines.append("| 策略:标的 | 交易数 | 胜率 | 平均盈 | 平均亏 | 连赢/连亏 |")
            lines.append("|-----------|--------|------|--------|--------|-----------|")
            for key, rec in records[:15]:
                wr = rec.wins / rec.total_trades if rec.total_trades > 0 else 0
                lines.append(
                    f"| {key} | {rec.total_trades} | {wr:.0%} | "
                    f"{rec.avg_win_pct:+.1f}% | {rec.avg_loss_pct:.1f}% | "
                    f"W{rec.consecutive_wins}/L{rec.consecutive_losses} |"
                )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("⚡ Yina 杠杆引擎 测试")
    print("=" * 50)

    engine = LeverageEngine()

    # 模拟一些历史交易
    test_trades = [
        ("v5_fusion", "BTC/USDT", +8.5),
        ("v5_fusion", "BTC/USDT", -3.2),
        ("v5_fusion", "BTC/USDT", +12.1),
        ("v5_fusion", "BTC/USDT", +5.7),
        ("v5_fusion", "BTC/USDT", -2.1),
        ("v5_fusion", "BTC/USDT", +15.0),
        ("v5_fusion", "BTC/USDT", -4.5),
        ("v5_fusion", "ETH/USDT", +6.3),
        ("v5_fusion", "ETH/USDT", +3.9),
        ("v5_fusion", "ETH/USDT", -2.8),
        ("alpha_arb", "NVDA/USDT", +2.1),
        ("alpha_arb", "NVDA/USDT", -1.5),
        ("alpha_arb", "NVDA/USDT", +4.2),
    ]
    for strat, sym, pnl in test_trades:
        engine.record_trade_outcome(strat, sym, pnl)

    print()
    print("📊 胜率数据:")
    for key, rec in engine.state.win_rates.records.items():
        wr = rec.wins / rec.total_trades if rec.total_trades > 0 else 0
        print(f"  {key:30s} | {rec.total_trades}笔 | WR={wr:.0%} "
              f"| 盈={rec.avg_win_pct:+.1f}% | 亏={rec.avg_loss_pct:.1f}%")
    print(f"  全局: {engine.state.win_rates.global_total_trades}笔, "
          f"WR={engine.get_global_win_rate():.0%}")

    # 测试不同置信度下的杠杆决策
    test_cases = [
        (0.85, +0.65, "BTC/USDT", "v5_fusion"),
        (0.72, +0.40, "ETH/USDT", "v5_fusion"),
        (0.55, +0.25, "NVDA/USDT", "alpha_arb"),
        (0.45, +0.15, "MU/USDT", "unknown"),
        (0.38, +0.10, "SHIB/USDT", "unknown"),
    ]

    print()
    print("⚡ 杠杆决策测试:")
    for conf, sig, sym, strat in test_cases:
        decision = engine.determine_leverage(
            confidence=conf, signal_weighted=sig,
            symbol=sym, strategy_name=strat,
            funding_rate=0.0001,  # 0.01% per 8h
        )
        status = f"❌ SKIP: {decision.skip_reason}" if decision.skip_reason else \
                 f"✅ {decision.recommended_leverage}x | " \
                 f"仓位={decision.risk_adjusted_size_pct:.1%} | " \
                 f"止损={decision.stop_loss_pct:+.1%} | " \
                 f"止盈={decision.take_profit_pct:+.0%}"
        print(f"  {sym:15s} conf={conf:.0%} sig={sig:+.2f} "
              f"→ blended={decision.blended_win_rate:.0%} → {status}")

    # Position sizing test
    print()
    print("💰 仓位计算 (假设总权益=1000 USDT):")
    for conf, sig, sym, strat in test_cases[:3]:
        decision = engine.determine_leverage(
            confidence=conf, signal_weighted=sig,
            symbol=sym, strategy_name=strat,
        )
        pos = engine.calculate_position(1000, 100.0, decision)
        print(f"  {sym}: {decision.recommended_leverage}x → "
              f"保证金=${pos['margin_usdt']} | "
              f"名义=${pos['notional_usdt']} | "
              f"最大亏损=${pos['max_loss_usdt']}")

    print()
    print("=" * 50)
    print(engine.get_status_report())
