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

杠杆映射 (v3 — 持有1-2天中频):
  ≥75% → 20x (2.5%仓位, -2.0%止损, +8%止盈)
  65-75% → 15x (3.0%仓位, -2.5%止损, +10%止盈)
  55-65% → 10x (3.5%仓位, -3.0%止损, +12%止盈)
  50-55% → 5x (4.0%仓位, -5.0%止损, +15%止盈)
  <50% → 1x (5.0%仓位, -8.0%止损, +20%止盈) — 跳过不开

风险原则: 杠杆 ↑ 止损%↓, 总风险可控
小资金阶梯: $≤100→40-50%权益, $100-200→30-40%, $200-500→20-30%

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

    # ── 杠杆映射表 (v4: 宽止损≥10% + 动态止盈, 配合安检门杠杆帽) ──
    # (min_wr, max_wr, leverage, label, size_pct, stop_loss_pct, take_profit_pct)
    # 🆕 v4: SL≥10% (不设紧止损, 给交易呼吸空间), TP=建议值(实际动态调整)
    LEVERAGE_TIERS = [
        #   min_wr  max_wr  lev  label               size%  stop%   tp%(建议)
        (0.75, 1.00, 20, "HIGH",                  0.020, -0.10, +0.30),   # 2%仓位×20x×10%SL=4%风险
        (0.65, 0.75, 15, "MEDIUM-HIGH",           0.030, -0.10, +0.30),   # 3%仓位×15x×10%SL=4.5%风险
        (0.55, 0.65, 10, "MEDIUM",                0.035, -0.12, +0.40),   # 3.5%×10x×12%SL=4.2%风险
        (0.50, 0.55,  5, "LOW",                   0.040, -0.15, +0.50),   # 4%×5x×15%SL=3%风险
        (0.00, 0.50,  1, "SPOT_EQUIVALENT",       0.050, -0.20, +0.80),   # 5%×1x×20%SL=1%风险
    ]

    # 🆕 v4: 止盈为动态值，这里只是默认建议
    # 实际止盈应在开仓前根据预期判断设定
    # 例如: BTC 64k看好到80k → TP≈+25% (而非固定30%)

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
        """初始化带交易权限的 OKX ccxt 实例 + REST 获取合约参数 (REST 优先，永不触发 load_markets bug)"""
        # ── 合约参数: REST 优先 (零 ccxt, 100% 可靠) ──
        if self._okx_trading is not None:
            if not getattr(self, '_markets_loaded', False):
                self._load_instruments_rest()
            return

        # ── 交易实例: 尝试 create_exchange (含凭证), 失败则降级 ──
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

        # ── 合约参数: 永远用 REST (绕过 ccxt load_markets NoneType 间歇性 bug) ──
        self._load_instruments_rest()

    def _load_instruments_rest(self):
        """
        🔗 OKX REST fallback: 获取所有 SWAP 合约参数 (绕过 ccxt load_markets bug)

        缓存到 self._rest_instruments = {instId: {ctVal, lotSz, minSz, ...}}
        用于 compute_contract_quantity 和 calculate_position。
        """
        try:
            import requests
            url = "https://www.okx.cab/api/v5/public/instruments"
            params = {"instType": "SWAP"}
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if data.get("code") == "0":
                self._rest_instruments = {}
                for inst in data.get("data", []):
                    if inst.get("state") != "live":
                        continue
                    inst_id = inst["instId"]
                    self._rest_instruments[inst_id] = {
                        "ctVal": float(inst.get("ctVal", 1)),
                        "lotSz": float(inst.get("lotSz", 1)),
                        "minSz": float(inst.get("minSz", 1)),
                        "tickSz": float(inst.get("tickSz", 0.1)),
                        "settleCcy": inst.get("settleCcy", ""),
                    }
                self._markets_loaded = True
                log.info(f"📚 OKX 合约参数已加载 (REST, {len(self._rest_instruments)} 个)")
            else:
                log.warning(f"OKX REST instruments 返回异常: {data.get('msg','?')}")
                self._markets_loaded = False
        except Exception as e:
            log.warning(f"OKX REST instruments 获取失败: {e}")
            self._markets_loaded = False

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

    def check_price_position(self, symbol: str, side: str, entry_price: float) -> dict:
        """
        🛡️ 入场时机校验: 检查价格在24h区间的位置，避免追高杀跌。

        规则:
          - 做多(long/buy): 价格 > 24h区间70%分位 → 拒绝 (追高风险)
          - 做空(short/sell): 价格 < 24h区间30%分位 → 拒绝 (杀跌风险)

        Returns:
          {
            "safe": bool,        # 是否可以入场
            "percentile": float, # 价格在24h区间的百分位 (0-100)
            "high_24h": float,
            "low_24h": float,
            "reason": str,       # 拒绝原因 (safe=False时)
          }
        """
        self._init_trading_exchange()
        result = {"safe": True, "percentile": 50.0, "high_24h": 0, "low_24h": 0, "reason": ""}

        try:
            import requests as _req, hmac as _hmac, base64 as _b64, json as _json
            from datetime import datetime as _dt, timezone as _tz

            inst_id = self._to_okx_inst_id(symbol)
            path = f"/api/v5/market/ticker?instId={inst_id}"
            ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            sign_str = ts + "GET" + path
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
            if data.get("code") != "0" or not data.get("data"):
                result["reason"] = f"ticker API 返回异常: {data.get('msg','?')}"
                return result

            ticker = data["data"][0]
            high_24h = float(ticker["high24h"])
            low_24h = float(ticker["low24h"])
            result["high_24h"] = high_24h
            result["low_24h"] = low_24h

            price_range = high_24h - low_24h
            if price_range <= 0:
                return result  # 无波动, 跳过检查

            percentile = (entry_price - low_24h) / price_range * 100.0
            result["percentile"] = round(percentile, 1)

            # 判断是否安全
            side_lower = side.lower()
            if side_lower in ("buy", "long"):
                if percentile > 70:
                    result["safe"] = False
                    result["reason"] = (
                        f"追高风险: 入场价${entry_price:.4f} 处于24h区间的{percentile:.0f}%分位 "
                        f"(高=${high_24h:.4f}, 低=${low_24h:.4f}), "
                        f"超过70%安全线——不做多"
                    )
            elif side_lower in ("sell", "short"):
                if percentile < 30:
                    result["safe"] = False
                    result["reason"] = (
                        f"杀跌风险: 入场价${entry_price:.4f} 处于24h区间的{percentile:.0f}%分位 "
                        f"(高=${high_24h:.4f}, 低=${low_24h:.4f}), "
                        f"低于30%安全线——不做空"
                    )

        except Exception as e:
            # 网络/解析异常不阻塞交易，只记录
            result["reason"] = f"ticker 查询失败({e})，跳过价格位置检查"
            log.warning(f"  ⚠️ [{symbol}] 价格位置检查异常: {e}")

        return result

    def verify_algo_order_live(self, algo_id: str, inst_id: str, max_retries: int = 2) -> bool:
        """
        🔴 验证算法单是否真正激活 (state=live)。

        开仓后必须逐个 algoId 确认 state=live，不能只依赖 POST 返回的 code==0。
        重试 max_retries 次，每次间隔 1 秒。
        """
        import requests as _req, hmac as _hmac, base64 as _b64
        from datetime import datetime as _dt, timezone as _tz
        import time as _time

        for attempt in range(1, max_retries + 1):
            try:
                path = f"/api/v5/trade/order-algo?algoId={algo_id}"
                ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                sign_str = ts + "GET" + path
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
                    state = data["data"][0].get("state", "")
                    fail_code = data["data"][0].get("failCode", "")
                    if state == "live" and str(fail_code) == "0":
                        return True
                    elif attempt < max_retries:
                        log.warning(f"  ⚠️ [{inst_id}] algo验证(尝试{attempt}/{max_retries}): "
                                   f"state={state}, failCode={fail_code}, 1秒后重试...")
                        _time.sleep(1.0)
                    else:
                        log.error(f"  🚨 [{inst_id}] algo验证失败({attempt}次): "
                                 f"state={state}, failCode={fail_code}")
                else:
                    if attempt < max_retries:
                        _time.sleep(1.0)
                    else:
                        log.error(f"  🚨 [{inst_id}] algo查询失败: {data.get('msg','?')}")
            except Exception as e:
                if attempt < max_retries:
                    _time.sleep(1.0)
                else:
                    log.error(f"  🚨 [{inst_id}] algo验证异常: {e}")
        return False

    def check_all_positions_sl_tp(self) -> dict:
        """
        🔴 遍历所有实盘持仓，检查每个持仓是否有活跃的 SL/TP 保护。
        在 daemon 启动和每轮扫描周期调用。

        Returns:
          {
            "all_protected": bool,
            "naked_positions": ["MU/USDT", ...],  # 无 SL/TP 的仓位
            "protected_positions": {"MU/USDT": {"sl": 1160, "tp": 1230}, ...},
          }
        """
        import requests as _req, hmac as _hmac, base64 as _b64
        from datetime import datetime as _dt, timezone as _tz

        result = {"all_protected": True, "naked_positions": [], "protected_positions": {}}

        try:
            self._init_trading_exchange()
            positions = self.fetch_open_positions()
            if not positions:
                return result

            # 获取所有 pending algo orders (okx.cab 必须带 ordType 参数)
            ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            path = "/api/v5/trade/orders-algo-pending"
            params_str = "?ordType=conditional"
            sign_str = ts + "GET" + path + params_str
            sign = _b64.b64encode(_hmac.new(
                self._okx_trading.secret.encode(), sign_str.encode(), "sha256"
            ).digest()).decode()

            r = _req.get(
                f"https://{self._okx_trading.hostname}{path}{params_str}",
                headers={
                    "OK-ACCESS-KEY": self._okx_trading.apiKey,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self._okx_trading.password,
                },
                timeout=10,
            )
            all_algos = r.json().get("data", [])

            # 逐个仓位检查
            for pos_symbol in sorted(positions):
                inst_id = self._to_okx_inst_id(pos_symbol)
                # 找到该 symbol 的所有 algo
                pos_algos = []
                for a in all_algos:
                    if a.get("instId") == inst_id:
                        pos_algos.append(a)
                    # 也检查 symbol 格式: MU/USDT vs MU-USDT-SWAP
                    a_symbol = a.get("instId", "").replace("-SWAP", "").replace("-", "/")
                    if a_symbol == pos_symbol:
                        pos_algos.append(a)

                has_sl = any(str(a.get("slTriggerPx", "")).strip() for a in pos_algos)
                has_tp = any(str(a.get("tpTriggerPx", "")).strip() for a in pos_algos)

                if has_sl or has_tp:
                    sl_val = None
                    tp_val = None
                    for a in pos_algos:
                        if str(a.get("slTriggerPx", "")).strip():
                            sl_val = a.get("slTriggerPx")
                        if str(a.get("tpTriggerPx", "")).strip():
                            tp_val = a.get("tpTriggerPx")
                    result["protected_positions"][pos_symbol] = {"sl": sl_val, "tp": tp_val}
                else:
                    result["naked_positions"].append(pos_symbol)
                    result["all_protected"] = False

        except Exception as e:
            log.warning(f"检查 SL/TP 状态异常: {e}")

        return result

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

            # 3. 设止盈止损 (OKX REST API 直接提交 algo orders)
            if stop_loss_price or take_profit_price:
                try:
                    import requests as _req, hmac as _hmac, base64 as _b64
                    from datetime import datetime as _dt, timezone as _tz

                    close_side = "sell" if side == "buy" else "buy"
                    pos_side = "long" if side == "buy" else "short"   # 仓位方向
                    inst_id = self._to_okx_inst_id(symbol)
                    results = []

                    # SL 和 TP 分别提交 conditional 订单（OKX 一个 conditional 只能设一种触发）
                    for ord_kind, trigger_px in [("SL", stop_loss_price), ("TP", take_profit_price)]:
                        if not trigger_px:
                            continue

                        algo_body = {
                            "instId": inst_id,
                            "tdMode": "isolated",
                            "side": close_side,
                            "posSide": pos_side,     # 🔑 OKX 必须参数
                            "ordType": "conditional",
                            "sz": str(quantity_contracts),
                        }
                        # 避免科学计数法 (OKX API 不接受 4.5523e-06 这种格式)
                        px_str = f"{trigger_px:.10f}" if trigger_px < 0.01 else str(trigger_px)
                        if ord_kind == "SL":
                            algo_body["slTriggerPx"] = px_str
                            algo_body["slOrdPx"] = "-1"   # 市价触发
                        else:
                            algo_body["tpTriggerPx"] = px_str
                            algo_body["tpOrdPx"] = "-1"   # 市价触发

                        body = json.dumps(algo_body)
                        ts = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        path = "/api/v5/trade/order-algo"
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
                            data=body,
                            timeout=10,
                        )
                        resp = r.json()
                        if resp.get("code") == "0":
                            algo_id = resp.get("data", [{}])[0].get("algoId", "")
                            results.append({"kind": ord_kind, "price": trigger_px, "algoId": algo_id})
                        else:
                            log.error(f"  🚨 {ord_kind} 设置失败 [{inst_id}]: {resp}")

                    # 🔴 逐个验证 algo order state=live
                    verified = []
                    for r_item in results:
                        if r_item.get("algoId"):
                            is_live = self.verify_algo_order_live(
                                r_item["algoId"], inst_id, max_retries=2
                            )
                            if is_live:
                                verified.append(f"{r_item['kind']}={r_item['price']}(✅live)")
                            else:
                                log.error(f"  🚨 {r_item['kind']} 未激活 [{inst_id}] algoId={r_item['algoId']} — 仓位裸奔!")
                        else:
                            log.error(f"  🚨 {r_item['kind']} 无algoId [{inst_id}] — 无法验证")
                    if verified:
                        log.info(f"  🎯 止盈止损已设置+验证: {', '.join(verified)}, instId={inst_id}")
                    elif results:
                        log.error(f"  🚨 止盈止损设置后验证全部失败 [{inst_id}] — 仓位无保护!")
                except Exception as algo_err:
                    log.error(f"  🚨 止盈止损设置异常 [{inst_id}]: {algo_err}")

            return order
        except Exception as e:
            log.error(f"合约下单失败 {symbol}: {e}")
            return None

    def _get_contract_params(self, symbol: str) -> tuple:
        """
        获取合约参数 (ctVal, lotSz)。

        先尝试 ccxt market 数据，失败则用 REST fallback 缓存。
        Returns: (contract_size, lot_sz) — defaults (1.0, 1.0)
        """
        contract_size = 1.0
        lot_sz = 1.0
        inst_id = self._to_okx_inst_id(symbol)

        # 1. 尝试 ccxt
        try:
            swap_sym = self._to_swap_symbol(symbol)
            market = self._okx_trading.market(swap_sym)
            contract_size = float(market.get('contractSize', 1.0) or 1.0)
            lot_sz = float(market.get('precision', {}).get('amount', 1.0) or 1.0)
            return contract_size, lot_sz
        except Exception:
            pass

        # 2. REST fallback
        if hasattr(self, '_rest_instruments') and inst_id in self._rest_instruments:
            info = self._rest_instruments[inst_id]
            return info.get("ctVal", 1.0), info.get("lotSz", 1.0)

        # 3. 尝试加载 REST 缓存
        try:
            self._load_instruments_rest()
            if hasattr(self, '_rest_instruments') and inst_id in self._rest_instruments:
                info = self._rest_instruments[inst_id]
                return info.get("ctVal", 1.0), info.get("lotSz", 1.0)
        except Exception:
            pass

        return contract_size, lot_sz

    def compute_contract_quantity(self, usdt_amount: float, price: float,
                                   leverage: int, symbol: str = None) -> float:
        """
        计算合约数量。

        OKX USDT-margined 线性合约:
          名义价值 = 保证金 * 杠杆
          合约数量 = 名义价值 / (价格 * ctVal)

        注意: usdt_amount 是保证金 (不是名义价值)
               ctVal 是合约面值 (如 ORDI ctVal=0.1, ETH ctVal=0.1, BTC ctVal=1)
               必须除以 ctVal，否则 ctVal!=1 的币种会下错数量！
        """
        if price <= 0:
            return 0.0

        contract_size = 1.0
        if symbol:
            contract_size, _ = self._get_contract_params(symbol)

        # 名义价值 = 保证金 * 杠杆
        notional = usdt_amount * leverage
        # 合约数量 = 名义价值 / (价格 * 合约面值)
        quantity = notional / (price * contract_size)
        return round(quantity, 6)  # OKX 通常是 0.001 精度

    # ═══════════════════════════════════════════════════════════
    # Risk-Aware Position Calculation
    # ═══════════════════════════════════════════════════════════

    def calculate_position(self, total_equity: float, price: float,
                            decision: LeverageDecision, side: str = "buy") -> dict:
        """
        根据杠杆决策计算实际仓位。

        Args:
          total_equity: 总权益 (USDT)
          price: 当前价格
          decision: LeverageDecision
          side: 'buy' 做多 / 'sell' 做空 (默认 buy)

        Returns:
          {
            "margin_usdt": float,      # 理论保证金
            "actual_margin_usdt": float, # 🔴 实际保证金 (从合约数量反算, 用于MIN_MARGIN检查)
            "notional_usdt": float,    # 理论名义价值
            "actual_notional_usdt": float, # 🔴 实际名义价值 (从合约数量反算)
            "quantity_contracts": float, # 合约数量
            "stop_loss_price": float,  # 止损价
            "take_profit_price": float,# 止盈价
            "max_loss_usdt": float,    # 最大亏损
          }
        """
        size_pct = decision.risk_adjusted_size_pct
        leverage = decision.recommended_leverage

        # 🔥 小资金集中火力: 阶梯仓位 — 不再撒胡椒面
        # $0-100:   单笔 40-50% (最多2仓, 另一仓做对冲)
        # $100-200: 单笔 30-40% (最多3仓)
        # $200-500: 单笔 20-30% (最多4仓)
        # $500+:    原版 tier 比例
        if total_equity <= 100:
            size_pct = max(size_pct, 0.40)   # 至少40%权益
            size_pct = min(size_pct, 0.50)   # 最多50% (留一半做对冲)
        elif total_equity <= 200:
            size_pct = max(size_pct, 0.25)   # 至少25%
            size_pct = min(size_pct, 0.40)
        elif total_equity <= 500:
            size_pct = max(size_pct, 0.15)
            size_pct = min(size_pct, 0.30)

        margin = total_equity * size_pct
        notional = margin * leverage
        quantity = self.compute_contract_quantity(margin, price, leverage, symbol=decision.symbol)

        # 🔴 反算实际保证金: 合约数量经过OKX lotSz截断后，实际所需的保证金可能不同
        # OKX 不会四舍五入到 lotSz，而是截断（向下取整到 lotSz 的倍数）
        # 例如: lotSz=0.01, quantity=0.0483 → OKX截断为 0.04
        contract_size, lot_sz = self._get_contract_params(decision.symbol)
        # 🔴 截断到 lotSz (OKX向下取整, 非四舍五入)
        import math
        if lot_sz > 0:
            quantity = math.floor(quantity / lot_sz) * lot_sz
        if quantity <= 0:
            quantity = lot_sz  # 至少 1 个 lot
        actual_notional = quantity * price * contract_size
        actual_margin = actual_notional / leverage

        # 止损方向: 多头止损在入场价下方, 空头止损在入场价上方
        if side == "sell":  # 空头 (做空)
            sl_price = price * (1 - decision.stop_loss_pct)  # stop_loss_pct为负, 减负得正 → 上方
            tp_price = price * (1 - decision.take_profit_pct)  # take_profit_pct为正, 减正得负 → 下方
        else:  # 多头 (做多, 默认)
            sl_price = price * (1 + decision.stop_loss_pct)
            tp_price = price * (1 + decision.take_profit_pct)
        max_loss = actual_margin * abs(decision.stop_loss_pct) * leverage

        return {
            "margin_usdt": round(actual_margin, 2),  # 🔴 用实际保证金替代理论值
            "actual_margin_usdt": round(actual_margin, 2),
            "notional_usdt": round(actual_notional, 2),  # 🔴 用实际名义价值
            "actual_notional_usdt": round(actual_notional, 2),
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
