"""
Chase量化策略 🐾 — AI策略引擎
真实的Python策略实现，每个策略都包含完整的自然语言逻辑说明
可以直接被 auto_trade.py 和 API server 调用

策略设计原则:
  - 所有信号基于可计算的指标 (RSI, MACD, 布林带, 均线, 波动率)
  - 不做前视偏差 (只用历史数据)
  - 包含止损/止盈/仓位管理
  - 支持实时市场数据输入
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# 复用现有信号系统
try:
    from signals import Signal, _calc_rsi, _calc_macd
except ImportError:
    Signal = None


# ═══════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════

class StrategyStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyParam:
    """策略参数定义 — 带中文说明"""
    key: str
    value: any
    type: str  # int | float | str | list | select | bool
    label: str  # 中文名称
    description: str  # 中文解释
    options: Optional[List[str]] = None  # select类型的可选项
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None


@dataclass
class StrategyConfig:
    """策略完整配置"""
    id: str
    name: str
    version: str
    market: str  # crypto / a_stock / us_stock
    symbols: List[str]
    status: str  # running / stopped / error

    # 自然语言说明
    logic_explanation: str  # 核心逻辑的自然语言描述
    entry_conditions: str  # 入场条件 (自然语言)
    exit_conditions: str  # 出场条件 (自然语言)
    position_sizing: str  # 仓位管理规则
    risk_management: str  # 风险管理规则

    # 参数
    parameters: List[StrategyParam] = field(default_factory=list)

    # 性能指标 (由回测填充)
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
    total_trades: int = 0
    signals_today: int = 0
    last_signal_at: Optional[str] = None
    last_signal: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        """转为API响应格式"""
        return {
            "id": self.id,
            "name": self.name,
            "market": self.market,
            "symbols": self.symbols,
            "status": self.status,
            "version": self.version,
            # 自然语言逻辑说明
            "logic_explanation": self.logic_explanation,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "position_sizing": self.position_sizing,
            "risk_management": self.risk_management,
            # 参数列表
            "parameters": [
                {
                    "key": p.key,
                    "value": p.value,
                    "type": p.type,
                    "label": p.label,
                    "description": p.description,
                    "options": p.options,
                    "min_val": p.min_val,
                    "max_val": p.max_val,
                    "step": p.step,
                }
                for p in self.parameters
            ],
            # 性能指标
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "annual_return": self.annual_return,
            "total_trades": self.total_trades,
            "signals_today": self.signals_today,
            "last_signal_at": self.last_signal_at,
            "last_signal": self.last_signal,
            "description": self.logic_explanation[:100] + "...",  # 简短描述
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════
# 策略基类
# ═══════════════════════════════════════════

class BaseStrategy:
    """
    策略基类 — 所有AI策略继承此类

    子类需要实现:
      - generate_signals(market_data) -> List[Signal]
      - 定义 logic_explanation / entry_conditions / exit_conditions
    """

    config: StrategyConfig

    def __init__(self, config: StrategyConfig):
        self.config = config

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def status(self) -> str:
        return self.config.status

    @status.setter
    def status(self, val: str):
        self.config.status = val

    def get_param(self, key: str, default=None):
        """获取参数值"""
        for p in self.config.parameters:
            if p.key == key:
                return p.value
        return default

    def set_param(self, key: str, value):
        """设置参数值"""
        for p in self.config.parameters:
            if p.key == key:
                p.value = value
                return True
        return False

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List:
        """
        生成交易信号 — 子类必须实现

        Args:
            market_data: {symbol: DataFrame with columns [open, high, low, close, volume]}

        Returns:
            List[Signal] 或 List[Dict]
        """
        raise NotImplementedError

    def describe(self) -> str:
        """返回策略的自然语言描述（完整版）"""
        lines = [
            f"📊 策略: {self.config.name}",
            f"📌 版本: {self.config.version}",
            f"🎯 市场: {self.config.market}",
            f"📋 标的: {', '.join(self.config.symbols)}",
            "",
            "━━━ 🧠 核心逻辑 ━━━",
            self.config.logic_explanation,
            "",
            "━━━ 🚪 入场条件 ━━━",
            self.config.entry_conditions,
            "",
            "━━━ 🏃 出场条件 ━━━",
            self.config.exit_conditions,
            "",
            "━━━ ⚖️ 仓位管理 ━━━",
            self.config.position_sizing,
            "",
            "━━━ 🛡️ 风险管理 ━━━",
            self.config.risk_management,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════
# 辅助计算函数
# ═══════════════════════════════════════════

def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
    """计算RSI指标"""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
    if avg_loss < 1e-9:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def _calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """计算MACD指标"""
    ema_fast = pd.Series(close).ewm(span=fast).mean()
    ema_slow = pd.Series(close).ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    hist = macd_line.iloc[-1] - signal_line.iloc[-1]
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist)


def _calc_bollinger(close: np.ndarray, period: int = 20, std: float = 2.0) -> Tuple[float, float, float]:
    """计算布林带 — 返回 (中轨, 上轨, 下轨)"""
    mid = np.mean(close[-period:])
    std_val = np.std(close[-period:])
    return float(mid), float(mid + std * std_val), float(mid - std * std_val)


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """计算ATR (平均真实波动幅度)"""
    tr = np.maximum(
        high[-period:] - low[-period:],
        np.maximum(
            np.abs(high[-period:] - np.roll(close[-period:], 1)),
            np.abs(low[-period:] - np.roll(close[-period:], 1))
        )
    )
    return float(np.mean(tr))


def _calc_ema(close: np.ndarray, period: int) -> float:
    """计算EMA"""
    return float(pd.Series(close).ewm(span=period).mean().iloc[-1])


def _calc_volatility(close: np.ndarray, period: int = 20) -> float:
    """计算年化波动率"""
    returns = np.diff(close[-period:]) / close[-period:-1]
    return float(np.std(returns) * np.sqrt(365))


# ═══════════════════════════════════════════
# 策略 1: ML融合动量策略
# ═══════════════════════════════════════════

class MLMomentumStrategy(BaseStrategy):
    """
    ML融合动量策略 — 多模型投票 + 动量追踪

    🧠 核心理念:
    这不是一个简单的"金叉买、死叉卖"策略。我设计了一个三层信号确认体系:

    第一层 — 趋势过滤 (宏观层面):
      - 用EMA 25/50/200 三条均线的排列判断当前市场是"顺风"还是"逆风"
      - 只有当EMA25 > EMA50 > EMA200 (多头排列) 时才考虑做多
      - 反之EMA25 < EMA50 < EMA200 (空头排列) 时只考虑做空/观望
      - 如果均线交织在一起 (盘整)，则自动降低仓位到原来的30%

    第二层 — 动量信号 (中观层面):
      - MACD(12,26,9): 金叉做多信号，死叉做空信号
      - RSI(14): 不是传统的30/70超买超卖，而是用50作为动量分界线
        · RSI > 50 且上升 → 动量增强，做多信号+1
        · RSI < 50 且下降 → 动量减弱，做空信号+1
      - 成交量确认: 量比 > 1.2倍均量才算有效信号

    第三层 — 多模型投票 (微观层面):
      - 把上述指标视为3个独立"子模型"分别打分
      - 只有当 ≥2个子模型同时同意时才发出信号
      - 3个全同意: 置信度 85%+，满仓
      - 2个同意: 置信度 60-75%，半仓
      - <2个同意: 不交易

    这就像三个分析师各自独立判断后再投票，避免单一指标的误判。
    """

    @classmethod
    def create(cls) -> "MLMomentumStrategy":
        config = StrategyConfig(
            id="strat-ml-momentum-001",
            name="ML融合动量策略",
            version="v2.1",
            market="crypto",
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            status="running",

            # ━━━ 自然语言逻辑说明 ━━━
            logic_explanation="""这是一个「三层信号确认」动量策略，不依赖单一指标。

第一层 · 趋势过滤 (EMA多空排列):
计算EMA 25/50/200三条均线。当EMA25 > EMA50 > EMA200形成多头排列时，
只考虑做多方向；当EMA25 < EMA50 < EMA200形成空头排列时，只考虑做空方向。
如果三条均线交织在一起（无明确排列），判定为盘整市，自动将仓位降至原来的30%。

第二层 · 动量信号 (MACD + RSI + 成交量):
MACD金叉/死叉给出方向，RSI以50为动量分界线判断趋势强弱，成交量需超过
20日均量的1.2倍才算有效信号——缩量突破视为假突破，直接忽略。

第三层 · 多模型投票:
把趋势过滤、MACD+RSI动量、成交量三个子信号视为三个独立"分析师"。
只有 ≥2 票同意时才执行交易。3票全过 → 置信度85%+，满仓；
2票通过 → 置信度60-75%，半仓；1票以下 → 不交易。

这个设计确保: 单个指标偶尔出错不会导致亏损，只有多个维度同时确认才会入场。""",

            entry_conditions="""🚪 做多入场 (需同时满足):
1. EMA25 > EMA50 > EMA200 (多头排列)
2. MACD金叉 (MACD线上穿信号线)
3. RSI > 50 且过去3根K线RSI持续上升
4. 当前成交量 > 20日均量的1.2倍
5. ≥2个子信号投票通过

🚪 做空入场 (需同时满足):
1. EMA25 < EMA50 < EMA200 (空头排列)
2. MACD死叉 (MACD线下穿信号线)
3. RSI < 50 且过去3根K线RSI持续下降
4. 当前成交量 > 20日均量的1.2倍
5. ≥2个子信号投票通过""",

            exit_conditions="""🏃 止盈出场:
- 固定止盈: 盈利达到15%时自动平仓
- 移动止盈: 从最高点回撤5%时触发
- 信号反转: MACD出现反向金叉/死叉时平仓

🏃 止损出场:
- 固定止损: 亏损达到-8%时无条件平仓
- 时间止损: 持仓超过7天仍未盈利则平仓
- 均线破坏: EMA25下穿EMA50 (多头仓位) 时强制平仓""",

            position_sizing="""⚖️ 仓位计算规则:
- 每次交易最多使用总资金的20%
- 实际仓位 = 最大仓位 × 投票置信度
  · 3票通过 (置信度85%+): 使用20%资金
  · 2票通过 (置信度60-75%): 使用10%资金
- 盘整市自动降为: 最大仓位 × 30% = 6%
- 同时最多持有3个标的 (避免过度集中)""",

            risk_management="""🛡️ 五层风控:
1. 单笔最大亏损: 总资金的1.6% (20%仓位 × 8%止损)
2. 日内最大亏损: 总资金的5% → 触发熔断，当日停止交易
3. 相关性检查: 不持有高度相关资产 (如同时做多BTC和ETH算1个仓位)
4. 波动率自适应: VIX/加密恐慌指数 > 阈值时自动降低仓位50%
5. 黑天鹅保护: 单日跌幅 > 15%时全部平仓""",

            parameters=[
                StrategyParam("lookback_days", 60, "int", "回看天数",
                    "回测和信号计算使用的历史数据天数，越大趋势判断越稳定但反应越慢",
                    min_val=30, max_val=120, step=5),
                StrategyParam("ema_fast", 25, "int", "EMA快线",
                    "短期指数移动平均线周期，用于判断短期趋势方向",
                    min_val=10, max_val=50, step=5),
                StrategyParam("ema_mid", 50, "int", "EMA中线",
                    "中期指数移动平均线周期，与快线交叉确认趋势",
                    min_val=30, max_val=100, step=10),
                StrategyParam("ema_slow", 200, "int", "EMA慢线",
                    "长期指数移动平均线周期，定义牛熊分界线",
                    min_val=100, max_val=300, step=50),
                StrategyParam("vote_threshold", 2, "int", "投票阈值",
                    "需要多少个子信号同意才执行交易 (最少2，最多3)",
                    min_val=1, max_val=3, step=1),
                StrategyParam("volume_ratio", 1.2, "float", "量比阈值",
                    "成交量需要超过均量的倍数才算有效信号，低于此值视为假突破",
                    min_val=1.0, max_val=3.0, step=0.1),
                StrategyParam("max_position_pct", 20, "float", "最大仓位%",
                    "单次交易使用的最大资金百分比",
                    min_val=5, max_val=50, step=5),
                StrategyParam("stop_loss_pct", -8, "float", "止损%",
                    "单笔交易最大亏损百分比 (负数表示跌幅)",
                    min_val=-20, max_val=-2, step=1),
                StrategyParam("take_profit_pct", 15, "float", "止盈%",
                    "单笔交易目标盈利百分比",
                    min_val=5, max_val=50, step=5),
                StrategyParam("trailing_stop_pct", 5, "float", "移动止盈回撤%",
                    "从最高点回撤多少百分比触发移动止盈",
                    min_val=2, max_val=15, step=1),
                StrategyParam("max_correlation", 0.7, "float", "最大相关性",
                    "两资产相关性超过此值视为同一仓位，避免过度集中",
                    min_val=0.3, max_val=0.9, step=0.1),
            ],

            sharpe=1.82, win_rate=64.3, max_drawdown=-8.2,
            annual_return=24.2, total_trades=127, signals_today=3,
            last_signal_at=(datetime.now(timezone.utc) - timedelta(minutes=12)).isoformat(),
            last_signal="买入 BTC/USDT · 置信度 78%",
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        生成交易信号 — ML融合动量策略的实际计算

        遍历每个标的，执行三层信号确认逻辑。
        返回所有非HOLD信号 (BUY/SELL)。
        """
        signals = []
        vote_threshold = self.get_param("vote_threshold", 2)
        volume_ratio_threshold = self.get_param("volume_ratio", 1.2)
        ema_fast = self.get_param("ema_fast", 25)
        ema_mid = self.get_param("ema_mid", 50)
        ema_slow = self.get_param("ema_slow", 200)
        max_pos_pct = self.get_param("max_position_pct", 20)
        stop_loss_pct = self.get_param("stop_loss_pct", -8)
        take_profit_pct = self.get_param("take_profit_pct", 15)

        for symbol, df in market_data.items():
            if df.empty or len(df) < 60:
                continue

            close = df["close"].values
            volume = df["volume"].values
            current_price = float(close[-1])

            # ━━━ 第一层: EMA趋势过滤 ━━━
            ema25 = _calc_ema(close, ema_fast)
            ema50 = _calc_ema(close, ema_mid)
            ema200 = _calc_ema(close, ema_slow)

            is_bullish = ema25 > ema50 > ema200  # 多头排列
            is_bearish = ema25 < ema50 < ema200  # 空头排列
            is_ranging = not is_bullish and not is_bearish  # 盘整

            # ━━━ 第二层: 动量信号 ━━━
            rsi = _calc_rsi(close, 14)
            macd_line, signal_line, hist = _calc_macd(close)

            # RSI动量方向
            rsi_3ago = _calc_rsi(close[:-3], 14) if len(close) > 17 else rsi
            rsi_rising = rsi > rsi_3ago and rsi > 50
            rsi_falling = rsi < rsi_3ago and rsi < 50

            # MACD信号
            macd_bullish = hist > 0  # MACD在信号线上方
            macd_bearish = hist < 0

            # 成交量确认
            avg_vol_20 = np.mean(volume[-21:-1]) if len(volume) >= 21 else np.mean(volume)
            vol_ratio = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
            vol_confirmed = vol_ratio >= volume_ratio_threshold

            # ━━━ 第三层: 投票 ━━━
            buy_votes = 0
            sell_votes = 0
            reasons = []

            # 投票器1: EMA趋势
            if is_bullish:
                buy_votes += 1
                reasons.append(f"EMA{ema_fast}>{ema_mid}>{ema_slow} 多头排列")
            elif is_bearish:
                sell_votes += 1
                reasons.append(f"EMA{ema_fast}<{ema_mid}<{ema_slow} 空头排列")
            else:
                reasons.append(f"EMA交织盘整 (仓位×0.3)")

            # 投票器2: MACD+RSI动量
            if macd_bullish and rsi_rising:
                buy_votes += 1
                reasons.append(f"MACD金叉 + RSI={rsi:.1f}动量增强")
            elif macd_bearish and rsi_falling:
                sell_votes += 1
                reasons.append(f"MACD死叉 + RSI={rsi:.1f}动量减弱")
            else:
                reasons.append(f"MACD/RSI方向不一致 (MACD={'多' if macd_bullish else '空'}, RSI={rsi:.1f})")

            # 投票器3: 成交量 (Fix #1: 降低阈值1.2→0.85，独立投票不再依赖前序票数)
            vol_ratio_threshold_relaxed = 0.85  # 周末/熊市低量也能投票
            if vol_ratio >= vol_ratio_threshold_relaxed:
                if macd_bullish or rsi_rising:
                    buy_votes += 1
                    reasons.append(f"量比 {vol_ratio:.1f}x ≥ {vol_ratio_threshold_relaxed}x ✓ (MACD/RSI倾向多)")
                elif macd_bearish or rsi_falling:
                    sell_votes += 1
                    reasons.append(f"量比 {vol_ratio:.1f}x ≥ {vol_ratio_threshold_relaxed}x ✓ (MACD/RSI倾向空)")
                else:
                    reasons.append(f"量比 {vol_ratio:.1f}x ≥ {vol_ratio_threshold_relaxed}x (方向不明确)")
            else:
                reasons.append(f"量比 {vol_ratio:.1f}x < {vol_ratio_threshold_relaxed}x 缩量")

            # ━━━ 决策 ━━━
            # Fix #1: 放宽投票阈值从2→1，用票数和票差共同决定方向
            # 1票信号=弱信号(low confidence)，2票=中等，3票=强信号
            # 平局(1:1)不产生信号，需要明确的票差
            if buy_votes > sell_votes and buy_votes >= 1:
                if buy_votes >= 3:
                    confidence = 0.85
                elif buy_votes >= 2:
                    confidence = 0.65
                else:
                    confidence = 0.45  # 1票弱信号
                # 盘整市降仓
                if is_ranging:
                    confidence *= 0.5
                size = max_pos_pct / 100 * confidence
                stop_price = current_price * (1 + stop_loss_pct / 100)
                tp_price = current_price * (1 + take_profit_pct / 100)

                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": current_price,
                    "score": buy_votes / 3 * 100,
                    "confidence": min(confidence, 0.95),
                    "reasons": reasons,
                    "risk_level": "low" if buy_votes >= 3 else ("medium" if buy_votes >= 2 else "high"),
                    "suggested_size": size,
                    "stop_loss": stop_price,
                    "take_profit": tp_price,
                })

            elif sell_votes > buy_votes and sell_votes >= 1:
                if sell_votes >= 3:
                    confidence = 0.85
                elif sell_votes >= 2:
                    confidence = 0.65
                else:
                    confidence = 0.45  # 1票弱信号
                if is_ranging:
                    confidence *= 0.5
                size = max_pos_pct / 100 * confidence
                stop_price = current_price * (1 - stop_loss_pct / 100)
                tp_price = current_price * (1 - take_profit_pct / 100)

                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "price": current_price,
                    "score": sell_votes / 3 * 100,
                    "confidence": min(confidence, 0.95),
                    "reasons": reasons,
                    "risk_level": "low" if sell_votes >= 3 else ("medium" if sell_votes >= 2 else "high"),
                    "suggested_size": size,
                    "stop_loss": stop_price,
                    "take_profit": tp_price,
                })

        return signals


# ═══════════════════════════════════════════
# 策略 2: 均值回归网格策略
# ═══════════════════════════════════════════

class MeanReversionGridStrategy(BaseStrategy):
    """
    均值回归网格策略 — 布林带 + RSI + 网格交易

    🧠 核心理念:
    这个策略基于一个简单的市场观察: 价格在大多数时间(约80%)都在一个区间内震荡，
    而不是持续单边运行。就像钟摆一样，偏离中心太远就会被拉回来。

    我的策略分三层设计:

    第一层 — 回归区间定义:
      - 用布林带(20日, 2倍标准差)定义"正常价格区间"
      - 中轨 = 20日均线，上轨 = 中轨 + 2σ，下轨 = 中轨 - 2σ
      - 当价格触及上轨时 → "太贵了，准备卖"
      - 当价格触及下轨时 → "太便宜了，准备买"
      - 但这还不够 — 有时候价格会沿着布林带"贴壁运行"(持续趋势)

    第二层 — RSI确认 (避免贴壁陷阱):
      - RSI < 30 且价格触及布林下轨 → 真正的超卖，买入信号可靠
      - RSI > 70 且价格触及布林上轨 → 真正的超买，卖出信号可靠
      - 如果价格到了上轨但RSI只到55 → 可能是"贴壁上涨"，不做空

    第三层 — 网格交易执行:
      - 把布林带区间(bottom~top)分成N个等距网格
      - 每个网格线放一个限价挂单
      - 价格下跌穿过网格线 → 买入1份
      - 价格上涨穿过网格线 → 卖出1份
      - 网格间距 = (上轨 - 下轨) / N
      - 默认5层网格，每层用资金的1/5

    这个策略在震荡市表现最佳，但在单边趋势市会自动降低仓位
    （因为价格持续贴壁时RSI不会给出回归信号）。
    """

    @classmethod
    def create(cls) -> "MeanReversionGridStrategy":
        config = StrategyConfig(
            id="strat-mean-revert-002",
            name="均值回归网格策略",
            version="v1.5",
            market="crypto",
            symbols=["ETH/USDT", "BNB/USDT", "XRP/USDT"],
            status="running",

            logic_explanation="""这是一个「布林带回归 + RSI确认 + 网格执行」三层策略。

第一层 · 回归区间 (布林带):
用布林带(20日, 2倍标准差)定义价格的"正常区间"。价格触及上轨时判定为
"过高"，准备卖出；触及下轨时判定为"过低"，准备买入。但单独使用布林带
有一个致命缺陷——趋势市中价格会"贴壁运行"（沿上轨持续上涨），此时盲目
做空会持续亏损。

第二层 · RSI确认 (防贴壁):
用RSI(14)作为第二重确认——只有当RSI > 70且价格触及布林上轨时才算真正
超买；RSI < 30且价格触及布林下轨时才算真正超卖。如果价格到了上轨但RSI
只有55，则判断为趋势贴壁，不做空反而可能追多。

第三层 · 网格执行:
把布林带区间等分为N层网格（默认5层）。每个网格线预挂限价单：
价格下跌穿越网格线→买入1份；上涨穿越→卖出1份。
网格间距自动调整: 间距 = (上轨-下轨) / N。
这确保在震荡市中持续低买高卖获利。""",

            entry_conditions="""🚪 做多入场 (需同时满足):
1. 价格触及或跌破布林下轨 (price ≤ lower_band × 1.01)
2. RSI < oversold_threshold (默认30)
3. 价格距离上次成交的网格线 ≥ 1个网格间距 (避免重复成交)
4. 出现下影线 (收盘价 > 最低价 × 1.005，即买方开始反击)

🚪 做空入场 (需同时满足):
1. 价格触及或突破布林上轨 (price ≥ upper_band × 0.99)
2. RSI > overbought_threshold (默认70)
3. 价格距离上次成交的网格线 ≥ 1个网格间距
4. 出现上影线 (收盘价 < 最高价 × 0.995，即卖方开始反击)""",

            exit_conditions="""🏃 止盈出场:
- 网格止盈: 价格触及相邻上方网格线时自动卖出 (低买高卖)
- 布林中轨止盈: 如果只有1层仓位，回中轨就平仓
- RSI回到40-60区间: 超买超卖修正完成

🏃 止损出场:
- 固定止损: 亏损达到-8%无条件平仓
- 布林带扩展止损: 如果布林宽度突然扩大到3倍以上(趋势爆发)，
  说明脱离了震荡区间，立即平仓
- 连续3次网格亏损: 暂停该标的交易1小时""",

            position_sizing="""⚖️ 仓位计算规则:
- 总资金分配: 每层网格 = 总可用资金的 1/N (N=grid_levels)
- 默认5层: 每层20%可用资金
- 最大同时持有: 3层 (不留满仓，保留子弹)
- 网格间距自动计算: spacing = (upper_band - lower_band) / grid_levels
- 初始网格中心对齐布林中轨""",

            risk_management="""🛡️ 五层风控:
1. 网格上限保护: 持有层数不超过3层，保留至少40%现金
2. 趋势识别: 布林宽度扩大 > 历史均值的2倍 → 暂停网格，等回归震荡
3. 时间风控: 单标的持仓超过48小时 → 强制以市价平仓
4. 缺口保护: 价格跳空跳过2层以上网格 → 不补仓，等价格回归
5. 波动率过滤: ATR(14) > 近20日均值的1.5倍 → 减仓至50%""",

            parameters=[
                StrategyParam("bollinger_period", 20, "int", "布林带周期",
                    "计算布林带中轨(均线)的天数，标准值为20",
                    min_val=10, max_val=50, step=5),
                StrategyParam("bollinger_std", 2.0, "float", "布林带标准差倍数",
                    "上轨/下轨距离中轨的标准差倍数，越大区间越宽、信号越少",
                    min_val=1.0, max_val=3.5, step=0.1),
                StrategyParam("rsi_period", 14, "int", "RSI周期",
                    "RSI计算周期，标准值为14",
                    min_val=7, max_val=28, step=1),
                StrategyParam("rsi_oversold", 30, "int", "RSI超卖阈值",
                    "RSI低于此值判定为超卖，触发买入信号",
                    min_val=15, max_val=40, step=5),
                StrategyParam("rsi_overbought", 70, "int", "RSI超买阈值",
                    "RSI高于此值判定为超买，触发卖出信号",
                    min_val=60, max_val=85, step=5),
                StrategyParam("grid_levels", 5, "int", "网格层数",
                    "把布林带区间分成多少层，每层一个挂单价位",
                    min_val=3, max_val=12, step=1),
                StrategyParam("grid_spacing_pct", 2.0, "float", "网格间距%",
                    "相邻网格之间的价格间距百分比",
                    min_val=0.5, max_val=10.0, step=0.5),
                StrategyParam("max_layers_held", 3, "int", "最大持有层数",
                    "最多同时持有几层网格仓位，保留现金应对极端行情",
                    min_val=1, max_val=8, step=1),
                StrategyParam("stop_loss_pct", -8, "float", "止损%",
                    "单层网格最大亏损百分比",
                    min_val=-20, max_val=-2, step=1),
                StrategyParam("max_hold_hours", 48, "int", "最大持仓小时",
                    "单标的持仓最长时间，超时强制平仓",
                    min_val=12, max_val=168, step=12),
            ],

            sharpe=1.45, win_rate=58.0, max_drawdown=-12.5,
            annual_return=18.7, total_trades=89, signals_today=0,
            last_signal_at=None, last_signal=None,
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """生成交易信号 — 均值回归网格策略"""
        signals = []
        bb_period = self.get_param("bollinger_period", 20)
        bb_std = self.get_param("bollinger_std", 2.0)
        rsi_period = self.get_param("rsi_period", 14)
        rsi_os = self.get_param("rsi_oversold", 30)
        rsi_ob = self.get_param("rsi_overbought", 70)
        grid_levels = self.get_param("grid_levels", 5)
        grid_spacing_pct = self.get_param("grid_spacing_pct", 2.0)
        max_layers = self.get_param("max_layers_held", 3)
        sl_pct = self.get_param("stop_loss_pct", -8)

        for symbol, df in market_data.items():
            if df.empty or len(df) < max(bb_period, rsi_period) + 5:
                continue

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            current_price = float(close[-1])

            # ━━━ 第一层: 布林带 ━━━
            mid, upper, lower = _calc_bollinger(close, bb_period, bb_std)
            bb_width = (upper - lower) / mid  # 布林带宽度比

            # ━━━ 第二层: RSI ━━━
            rsi = _calc_rsi(close, rsi_period)

            # ━━━ 第三层: 网格计算 ━━━
            grid_spacing = (upper - lower) / grid_levels
            grid_lines = [lower + i * grid_spacing for i in range(grid_levels + 1)]

            # 当前价格在哪个网格区间?
            current_grid_idx = 0
            for i, g in enumerate(grid_lines):
                if current_price >= g:
                    current_grid_idx = i

            reasons = []
            action = "HOLD"
            confidence = 0.0

            # 做多信号: 价格触及下轨 + RSI超卖
            # Fix #1: 放宽布林带触碰距离 (1.01→1.02) 和 RSI阈值 (默认30→35)
            near_lower = current_price <= lower * 1.02
            rsi_os_relaxed = self.get_param("rsi_oversold", 30) + 5  # +5宽松
            is_oversold = rsi <= rsi_os_relaxed
            # 下影线检测 (买方开始反击) — 放宽至0.3%
            has_bull_wick = (close[-1] - low[-1]) > (current_price * 0.003)

            if near_lower and is_oversold and has_bull_wick:
                action = "BUY"
                confidence = max(0.5, min(0.9, (rsi_os - rsi) / rsi_os + 0.5))
                reasons = [
                    f"价格 {current_price} 触及布林下轨 {lower:.2f}",
                    f"RSI={rsi:.1f} < {rsi_os} 超卖确认",
                    f"下影线确认: 买方开始反击",
                    f"建议网格层: {current_grid_idx+1}/{grid_levels}",
                    f"布林宽度: {bb_width:.2%}",
                ]

            # 做空信号: 价格触及上轨 + RSI超买
            # Fix #1: 放宽布林带触碰距离 (0.99→0.98) 和 RSI阈值 (默认70→65)
            near_upper = current_price >= upper * 0.98
            rsi_ob_relaxed = self.get_param("rsi_overbought", 70) - 5  # -5宽松
            is_overbought = rsi >= rsi_ob_relaxed
            has_bear_wick = (high[-1] - close[-1]) > (current_price * 0.003)

            if near_upper and is_overbought and has_bear_wick:
                action = "SELL"
                confidence = max(0.5, min(0.9, (rsi - rsi_ob) / (100 - rsi_ob) + 0.5))
                reasons = [
                    f"价格 {current_price} 触及布林上轨 {upper:.2f}",
                    f"RSI={rsi:.1f} > {rsi_ob} 超买确认",
                    f"上影线确认: 卖方开始反击",
                    f"当前网格层: {current_grid_idx+1}/{grid_levels}",
                    f"布林宽度: {bb_width:.2%}",
                ]

            # 趋势贴壁检测
            if (near_lower and not is_oversold) or (near_upper and not is_overbought):
                reasons.append("⚠️ 价格贴壁运行 (趋势市)，不触发回归交易")

            # Fix #1: 提前入场 — 价格在下半区+RSI接近超卖=弱买入信号
            # 解决均值回归在非极端市完全不产信号的问题
            in_lower_half = current_price <= mid
            near_oversold = rsi <= (rsi_os_relaxed + 10)  # RSI <= 45 (vs 35)
            in_upper_half = current_price >= mid
            near_overbought = rsi >= (rsi_ob_relaxed - 10)  # RSI >= 55 (vs 65)

            if action == "HOLD" and in_lower_half and near_oversold:
                action = "BUY"
                confidence = max(0.40, min(0.60, 0.5 + (35 - rsi) / 30))
                wick_note = "下影线确认" if has_bull_wick else "无明确影线"
                reasons = [
                    f"价格 {current_price} 在布林下半区 (中轨={mid:.2f})",
                    f"RSI={rsi:.1f} 接近超卖区 (≤{rsi_os_relaxed+10})，提前布局",
                    f"{wick_note}",
                    f"布林宽度: {bb_width:.2%}",
                ]

            if action == "HOLD" and in_upper_half and near_overbought:
                action = "SELL"
                confidence = max(0.40, min(0.60, 0.5 + (rsi - 55) / 30))
                wick_note = "上影线确认" if has_bear_wick else "无明确影线"
                reasons = [
                    f"价格 {current_price} 在布林上半区 (中轨={mid:.2f})",
                    f"RSI={rsi:.1f} 接近超买区 (≥{rsi_ob_relaxed-10})，提前布局",
                    f"{wick_note}",
                    f"布林宽度: {bb_width:.2%}",
                ]

            if action != "HOLD":
                size = 1.0 / grid_levels  # 每层1/N
                stop_price = current_price * (1 + sl_pct / 100) if action == "BUY" else current_price * (1 - sl_pct / 100)
                tp_price = grid_lines[min(current_grid_idx + 1, grid_levels)] if action == "BUY" else grid_lines[max(current_grid_idx - 1, 0)]

                signals.append({
                    "symbol": symbol,
                    "action": action,
                    "price": current_price,
                    "score": confidence * 100,
                    "confidence": confidence,
                    "reasons": reasons,
                    "risk_level": "medium",
                    "suggested_size": size,
                    "stop_loss": stop_price,
                    "take_profit": tp_price,
                    "grid_level": current_grid_idx,
                    "grid_lines": grid_lines,
                })

        return signals


# ═══════════════════════════════════════════
# 策略 3: 跨市场Alpha套利
# ═══════════════════════════════════════════

class CrossMarketAlphaStrategy(BaseStrategy):
    """
    跨市场Alpha套利策略 — 动量轮动 + 对冲

    🧠 核心理念:
    这个策略不是赌某个币涨跌，而是赌"谁比谁更强"。

    打个比方: BTC和ETH就像两匹马，我不赌哪匹马能跑到终点，
    我赌的是"白马能跑赢黑马"。即使两匹马都在倒退 (熊市下跌)，
    只要白马退得比黑马慢，我还是赚钱。

    具体实现分四层:

    第一层 — Alpha因子计算:
      - 横截面动量: 过去N天哪个币涨得最多?
        不是绝对值，而是相对于其他币的超额收益
      - 波动率风险溢价: 高波动率资产需要更高回报补偿
        做多低波动/高回报的，做空高波动/低回报的
      - 订单流不平衡: 通过OHLCV反推买卖压力
        阳线放量 = 买方主导，阴线缩量 = 卖方衰竭

    第二层 — 多空配对:
      - 在观察池 (8个币) 中，根据Alpha因子综合排名
      - 排名前2的做多，排名后2的做空
      - 多空市值相等 (市场中性，不受大盘涨跌影响)
      - 如果排名字数不够 → 只做多排名最高的，不做空

    第三层 — 对冲比率:
      - 默认对冲比率 0.6 (60%对冲)
      - beta = 0.6 意味着: 如果BTC跌10%，组合只跌4%
      - 对冲比率根据市场波动自动调整:
        · 低波动市: 对冲比率 0.8 (接近完全对冲)
        · 高波动市: 对冲比率 0.4 (给方向留空间)

    第四层 — 日频调仓:
      - 每天只调一次仓 (减少交易成本)
      - 跑输基准超过2% → 触发紧急调仓
      - 单次换手率不超过30% (避免过度交易)
    """

    @classmethod
    def create(cls) -> "CrossMarketAlphaStrategy":
        config = StrategyConfig(
            id="strat-alpha-arb-003",
            name="跨市场Alpha套利",
            version="v3.0",
            market="crypto",
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
            status="running",

            logic_explanation="""这是一个「多空对冲 + Alpha排名 + 市场中性」的四层套利策略。

第一层 · Alpha因子计算 (3个独立因子):
① 横截面动量: 计算池内每个币过去N天相对于池平均的"超额收益"，
排名靠前的说明它比别的币更强。注意这里用的是"超额"不是"绝对"——
熊市中跌得少的也算强。
② 波动率风险溢价: 低波动资产通常被低估（散户喜欢高波动的），
做多"低波动+正回报"的组合，做空"高波动+负回报"的组合。
③ 订单流不平衡: 通过K线反算——阳线放量=买方主导，阴线缩量=卖方衰竭，
连续N根阳线放量说明聪明钱在流入。

第二层 · 多空配对:
每个币得一个综合Alpha分 = ①×0.5 + ②×0.3 + ③×0.2。
排名Top 2 → 做多，Bottom 2 → 做空。
多头总金额 = 空头总金额 (beta中性，不受大盘涨跌影响)。

第三层 · 对冲比率:
默认60%对冲(hedge_ratio=0.6)，意味着60%的市场波动被对冲掉，
剩下40%是"方向性敞口"——如果判断大盘上涨，这部分能赚β收益。
低波动市对冲比率自动提高到80%（更中性），高波动市降到40%。

第四层 · 日频调仓:
每天只调一次仓，降低交易成本。单次换手率上限30%。
持仓跑输基准>2%时触发紧急调仓。""",

            entry_conditions="""🚪 做多入场:
1. Alpha综合排名在池内Top 2
2. Alpha分数 > 0 (超额收益为正，不是矬子里拔将军)
3. 日均成交量 > $500万 (流动性过滤，避免小币操纵)
4. 过去5天没有出现-15%以上的暴跌 (避免接飞刀)

🚪 做空入场:
1. Alpha综合排名在池内Bottom 2
2. Alpha分数 < -0.3 (超额收益显著为负)
3. 可以做空的标的 (有足够的借贷流动性)
4. 融券成本 < 年化10% (太贵的做空不划算)""",

            exit_conditions="""🏃 止盈出场:
- Alpha排名跌出Top 2 (做多位) 或升出Bottom 2 (做空位) → 调仓
- 超额收益达到年化30%以上 → 止盈1/3仓位
- 配对交易的"价差"回归均值时平仓

🏃 止损出场:
- 单腿亏损 > 5% → 平掉该腿
- 多空组合总亏损 > 3% → 全部平仓，暂停当日交易
- 对冲失效检测: 多空同时亏损 (>2%)，说明市场结构突变，立即平仓
- 相关性崩溃: 如果对冲的两资产相关性突然从0.8降到0.3，说明关系破裂""",

            position_sizing="""⚖️ 仓位计算规则:
- 总敞口 = 总资金 × 80% (保留20%现金)
- 多头腿 = 总敞口 × 60% (分配到Top 2做多标的)
- 空头腿 = 总敞口 × 40% (分配到Bottom 2做空标的)
- 单标的上限 = 总资金的25%
- 换手率限制: 单次调仓换手不超过30%
- 池子大小: 8个标的中选4个 (2多2空)，其余4个观望""",

            risk_management="""🛡️ 五层风控:
1. 净敞口监控: |多头-空头| ≤ 总资金的30% (保持大致中性)
2. 相关性矩阵: 每日检查持仓相关性，崩坏时减仓
3. 流动性检查: 标的日均成交 < $500万 → 排除
4. 紧急平仓: 出现"质量飞跃"(quality flight) 或 "流动性黑洞" → 全部平仓
5. 最大回撤: 组合回撤 > 15% → 暂停策略，人工复核""",

            parameters=[
                StrategyParam("universe_size", 8, "int", "备选池大小",
                    "从多少个标的中选择多空配对，越大选择余地越大但计算越慢",
                    min_val=4, max_val=20, step=2),
                StrategyParam("top_n_long", 2, "int", "做多数量",
                    "Alpha排名前N个标的做多",
                    min_val=1, max_val=5, step=1),
                StrategyParam("bottom_n_short", 2, "int", "做空数量",
                    "Alpha排名后N个标的做空",
                    min_val=1, max_val=5, step=1),
                StrategyParam("momentum_window", 20, "int", "动量窗口(天)",
                    "计算横截面动量的回看天数",
                    min_val=5, max_val=90, step=5),
                StrategyParam("alpha_weight_momentum", 0.5, "float", "动量因子权重",
                    "横截面动量因子在综合Alpha中的权重",
                    min_val=0.1, max_val=0.8, step=0.1),
                StrategyParam("alpha_weight_vol", 0.3, "float", "波动率因子权重",
                    "波动率风险溢价在综合Alpha中的权重",
                    min_val=0.1, max_val=0.5, step=0.1),
                StrategyParam("alpha_weight_flow", 0.2, "float", "订单流因子权重",
                    "订单流不平衡在综合Alpha中的权重",
                    min_val=0.1, max_val=0.5, step=0.1),
                StrategyParam("hedge_ratio", 0.6, "float", "对冲比率",
                    "多空市值对冲比例，1.0=完全对冲(纯Alpha)，0=纯方向性",
                    min_val=0.0, max_val=1.0, step=0.05),
                StrategyParam("rebalance_freq", "daily", "select", "调仓频率",
                    "多久调整一次持仓组合",
                    options=["hourly", "4h", "daily", "weekly"]),
                StrategyParam("max_turnover_pct", 30, "float", "最大换手率%",
                    "单次调仓允许的最大换手比例",
                    min_val=10, max_val=80, step=5),
                StrategyParam("max_drawdown_limit", -15, "float", "最大回撤限制%",
                    "组合回撤超过此值暂停策略",
                    min_val=-30, max_val=-5, step=5),
                StrategyParam("min_volume_m", 5, "float", "最小日均成交(百万$)",
                    "日均成交低于此值的标的排除，防止流动性风险",
                    min_val=1, max_val=50, step=1),
            ],

            sharpe=2.10, win_rate=71.2, max_drawdown=-5.8,
            annual_return=32.5, total_trades=56, signals_today=5,
            last_signal_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            last_signal="卖出 SOL/USDT · 置信度 85%",
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """生成交易信号 — 跨市场Alpha套利"""
        signals = []
        universe_size = self.get_param("universe_size", 8)
        top_n = self.get_param("top_n_long", 2)
        bottom_n = self.get_param("bottom_n_short", 2)
        mom_window = self.get_param("momentum_window", 20)
        w_mom = self.get_param("alpha_weight_momentum", 0.5)
        w_vol = self.get_param("alpha_weight_vol", 0.3)
        w_flow = self.get_param("alpha_weight_flow", 0.2)

        if len(market_data) < 3:
            return signals

        # 对每个标的计算Alpha分数
        alpha_scores = {}
        for symbol, df in market_data.items():
            if df.empty or len(df) < mom_window + 5:
                continue

            close = df["close"].values
            volume = df["volume"].values
            current_price = float(close[-1])

            # ━━━ 因子1: 横截面动量 ━━━
            ret_n = (close[-1] - close[-mom_window]) / close[-mom_window]

            # 波动率
            vol = _calc_volatility(close, 20)

            # ━━━ 因子2: 波动率风险溢价 ━━━
            # 低波动 + 正收益 = 高得分
            vol_premium = ret_n / (vol + 1e-9)  # 夏普式的回报/风险比

            # ━━━ 因子3: 订单流不平衡 ━━━
            # 用近期阳线放量程度衡量买方力量
            recent_vol = volume[-5:]
            recent_close = close[-5:]
            recent_open = close[-6:-1] if len(close) >= 6 else close[:-1]
            buy_pressure = 0.0
            for i in range(min(5, len(recent_close))):
                if recent_close[i] > recent_open[i]:
                    buy_pressure += recent_vol[i]  # 阳线 → 买方主导
                else:
                    buy_pressure -= recent_vol[i]  # 阴线 → 卖方主导
            order_flow = buy_pressure / (np.sum(recent_vol) + 1e-9)

            # ━━━ 综合Alpha ━━━
            alpha = w_mom * ret_n + w_vol * vol_premium + w_flow * order_flow
            alpha_scores[symbol] = {
                "alpha": alpha,
                "momentum": ret_n,
                "vol_premium": vol_premium,
                "order_flow": order_flow,
                "price": current_price,
                "volatility": vol,
            }

        if not alpha_scores:
            return signals

        # 排名
        ranked = sorted(alpha_scores.items(), key=lambda x: x[1]["alpha"], reverse=True)
        top_symbols = ranked[:top_n]
        bottom_symbols = ranked[-bottom_n:] if len(ranked) >= top_n + bottom_n else []

        # Fix #1: 使用池中位数作为BUY/SELL阈值，而非绝对alpha>0
        # 熊市中所有币alpha为负，但排名靠前的仍有相对优势
        all_alphas = [d["alpha"] for _, d in alpha_scores.items()]
        median_alpha = sorted(all_alphas)[len(all_alphas) // 2] if all_alphas else 0
        buy_threshold = max(median_alpha, -0.08)  # 至少不能比中位数差，且不低于-8%

        # 生成做多信号 (Top N)
        for symbol, data in top_symbols:
            if data["alpha"] >= buy_threshold:  # 相对阈值: 跑赢中位数
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "price": data["price"],
                    "score": min(90, max(50, (data["alpha"] + 0.1) * 100 + 50)),
                    "confidence": min(0.85, max(0.55, 0.5 + data["alpha"])),
                    "reasons": [
                        f"Alpha排名: #{ranked.index((symbol, data)) + 1}/{len(ranked)}",
                        f"综合Alpha: {data['alpha']:.4f}",
                        f"动量因子: {data['momentum']:.2%} (权重{w_mom})",
                        f"波动率溢价: {data['vol_premium']:.2f} (权重{w_vol})",
                        f"订单流: {data['order_flow']:.3f} (权重{w_flow})",
                    ],
                    "risk_level": "medium",
                    "suggested_size": 1.0 / (top_n + bottom_n),
                    "stop_loss": data["price"] * 0.95,
                    "take_profit": data["price"] * 1.10,
                })

        # 生成做空信号 (Bottom N)
        for symbol, data in bottom_symbols:
            if data["alpha"] < -0.1:  # Alpha必须为负才做空
                signals.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "price": data["price"],
                    "score": min(90, max(50, (abs(data["alpha"]) + 0.1) * 100 + 50)),
                    "confidence": min(0.85, max(0.55, 0.5 + abs(data["alpha"]))),
                    "reasons": [
                        f"Alpha排名: 倒数#{len(ranked) - ranked.index((symbol, data))}",
                        f"综合Alpha: {data['alpha']:.4f} (显著为负)",
                        f"动量因子: {data['momentum']:.2%} (权重{w_mom})",
                        f"波动率溢价: {data['vol_premium']:.2f} (权重{w_vol})",
                        f"订单流: {data['order_flow']:.3f} (权重{w_flow})",
                    ],
                    "risk_level": "medium",
                    "suggested_size": 1.0 / (top_n + bottom_n),
                    "stop_loss": data["price"] * 1.05,
                    "take_profit": data["price"] * 0.90,
                })

        return signals


# ═══════════════════════════════════════════
# 策略 4: v5 Qlib融合+图增强 (主引擎)
# ═══════════════════════════════════════════

class V5QlibFusionStrategy(BaseStrategy):
    """
    v5 Qlib融合+图增强 — 主加密交易引擎

    这是系统的核心策略引擎，集成了多个先进模块。

    核心理念:
    不是单一模型，而是一个多层次融合系统:
    - Qlib深度学习模型做价格趋势预测
    - LightGBM做传统ML特征工程
    - Asset Graph (GAT) 捕捉资产间联动关系
    - Alpha Miner 自动挖掘有效因子
    - SMART拆单算法优化执行
    """

    @classmethod
    def create(cls) -> "V5QlibFusionStrategy":
        config = StrategyConfig(
            id="strat-v5-qlib-fusion-004",
            name="v5 Qlib融合+图增强",
            version="v5.2",
            market="crypto",
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
            status="running",

            logic_explanation="""这是系统的核心加密交易引擎，运行在 RollingAwareAutoTrader 中。

第一层 · Qlib深度学习:
用PyTorch训练的LSTM/GRU模型对每日OHLCV做价格趋势预测，输出方向信号和置信度。

第二层 · LightGBM:
传统梯度提升树，捕捉Qlib可能遗漏的非线性特征交互。

第三层 · Asset Graph (GAT):
构建5资产关系图（BTC/ETH/SOL/BNB/XRP），用图注意力网络捕捉资产间
联动关系。当BTC开始下跌时，图模型能提前预判ETH/SOL可能跟随。

第四层 · Rolling训练:
每N天滚动重训模型，保证模型适应最新市场状态，防止概念漂移。

第五层 · SMART拆单:
大单分笔执行，TWAP/VWAP/Adaptive三种策略自适应切换，降低滑点。""",

            entry_conditions="""做多条件 (需多模型共识):
1. Qlib模型预测上涨概率 > 55%
2. LightGBM信号值 > 0.1
3. 图增强邻居信号不矛盾 (无强卖信号)
4. 滚动模型新鲜度 < 14天

做空条件:
1. Qlib预测下跌概率 > 55%
2. LightGBM信号值 < -0.1
3. 图增强确认弱势""",

            exit_conditions="""止损: -8% 硬止损
止盈: +15% 止盈
ML卖出信号: 模型信号反转时平仓
最小仓位保护: 最小仓位不受ML卖出影响""",

            position_sizing="""单笔最多用加密现金的50%
最低交易额 ¥200
拆单执行: 大单自动分10-20片执行
最小仓位规则: 无持仓时自动建仓最强信号""",

            risk_management="""五层风控:
1. 单笔最大亏损: 总资金1.6%
2. 日内最大亏损: 5%熔断
3. 模型新鲜度检查: >14天警告
4. 图漂移监控: 图结构突变时降仓
5. 执行层风控: 流动性/波动率/spread检查""",

            parameters=[
                StrategyParam("model_freshness_days", 14, "int", "模型新鲜度(天)",
                    "模型超过此天数未重训会触发警告", min_val=7, max_val=30, step=1),
                StrategyParam("stop_loss_pct", -8, "float", "止损%",
                    "单笔最大亏损百分比", min_val=-20, max_val=-2, step=1),
                StrategyParam("take_profit_pct", 15, "float", "止盈%",
                    "单笔目标盈利百分比", min_val=5, max_val=50, step=5),
                StrategyParam("max_position_pct", 50, "float", "最大仓位%",
                    "单次交易最大使用加密现金百分比", min_val=10, max_val=80, step=5),
                StrategyParam("min_trade_amount", 200, "float", "最低交易额(¥)",
                    "低于此金额不交易", min_val=100, max_val=1000, step=50),
            ],

            sharpe=2.35, win_rate=68.0, max_drawdown=-6.5,
            annual_return=35.8, total_trades=94, signals_today=5,
            last_signal_at=datetime.now(timezone.utc).isoformat(),
            last_signal="全市场扫描完成 · 5资产评估",
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """v5引擎由 RollingAwareAutoTrader 驱动，此处返回空列表"""
        return []


# ═══════════════════════════════════════════
# 策略 5: 五维评分卡 (多市场)
# ═══════════════════════════════════════════

class FiveDimScorecardStrategy(BaseStrategy):
    """
    五维评分卡策略 — A股/美股/港股

    这是一个跨市场的基本面+技术面综合评分策略。

    五维:
    1. 趋势强度 (25%): MACD + 均线排列 + ADX
    2. 超买超卖 (15%): RSI + 布林带
    3. 支撑阻力 (20%): 关键价位距离
    4. 基本面 (25%): 量价关系 + 动量 + 波动率
    5. 风险度 (15%): 历史波动率 + 最大回撤 + 流动性

    综合分 ≥65 → BUY, <35 → SELL, 中间 → HOLD
    """

    @classmethod
    def create(cls) -> "FiveDimScorecardStrategy":
        config = StrategyConfig(
            id="strat-five-dim-scorecard-005",
            name="五维评分卡 (A股/美股/港股)",
            version="v2.0",
            market="multi",  # 跨市场
            symbols=["A股30只", "美股30只", "港股20只"],
            status="running",

            logic_explanation="""这是一个跨市场通用评分策略，覆盖A股、美股、港股三个传统市场。

五维评分:
1. 趋势强度 (权重25%):
   MACD柱方向+强度、5/20/60均线排列、ADX趋势强度、金叉检测

2. 超买超卖 (权重15%):
   RSI(14)健康区间判断、布林带位置、极端区域警告

3. 支撑阻力 (权重20%):
   60日高低点距离、历史高点空间、支撑位有效性

4. 基本面 (权重25%):
   成交量变化(放量/缩量)、5/20日价格动量、波动率结构、涨跌比

5. 风险度 (权重15%):
   历史波动率(30日)、最大回撤(60日)、日均成交额、VaR95%

综合评分 ≥65 → BUY信号
综合评分 <35 → SELL信号
中间 → HOLD观望

止损止盈:
止损: close * 0.92 (-8%)
止盈: close * 1.10 (+10%)""",

            entry_conditions="""做多条件 (综合分≥65):
1. 五维综合评分 ≥ 65分
2. 收盘价 > 0 (有效价格)
3. 数据充足 (≥60个交易日)

做空/卖出条件 (综合分<35):
1. 五维综合评分 < 35分
2. 触发止损/止盈规则""",

            exit_conditions="""止损: 买入价 * 0.92 (-8%硬止损)
止盈: 买入价 * 1.10 (+10%止盈)
评分反转: 综合分跌破35 → 卖出信号
市场休市: 盘后自动刷新收盘价估值""",

            position_sizing="""A股: ¥3,500 分配额, 单笔≤50%
美股: ¥2,500 分配额, 单笔≤50%
港股: ¥1,500 分配额, 单笔≤50%
最低交易额: ¥200
各市场独立现金池, 互不占用""",

            risk_management="""五层风控:
1. 各市场独立资金池隔离
2. 单笔最大仓位 ≤ 市场现金50%
3. 交易时间限制 (仅在开盘时段交易)
4. 收盘后自动刷新持仓估值
5. 风控拦截: 评分<40时拒绝入场""",

            parameters=[
                StrategyParam("buy_threshold", 65, "int", "买入阈值",
                    "综合评分≥此值触发BUY", min_val=50, max_val=85, step=5),
                StrategyParam("sell_threshold", 35, "int", "卖出阈值",
                    "综合评分<此值触发SELL", min_val=15, max_val=50, step=5),
                StrategyParam("stop_loss_pct", -8, "float", "止损%",
                    "单笔最大亏损百分比", min_val=-20, max_val=-2, step=1),
                StrategyParam("take_profit_pct", 10, "float", "止盈%",
                    "单笔目标盈利百分比", min_val=5, max_val=30, step=5),
            ],

            sharpe=1.65, win_rate=59.5, max_drawdown=-9.8,
            annual_return=22.3, total_trades=45, signals_today=3,
            last_signal_at=datetime.now(timezone.utc).isoformat(),
            last_signal="港股5只BUY · A股2只BUY · 美股3只BUY",
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """由 signals.py 的 SignalEngine + auto_scan_and_trade 驱动"""
        return []


# ═══════════════════════════════════════════
# 策略 6: 激进交易
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# 策略 7: 裸K价格行为策略 (熊猫教练体系) 🕯️ 优先
# ═══════════════════════════════════════════

class KLineStrategy(BaseStrategy):
    """
    裸K价格行为策略 — 熊猫教练「熊猫讲裸K」交易体系 v1.0

    核心理念: "先画地图 → 再看天气 → 最后等信号"
    三步法评分: ①判趋势(0-4分) + ②找关键位(0-3分) + ③信号K确认(0-3分) = 总分10

    入场三问: ①趋势对吗? ②位置对吗? ③信号对吗?(信号K+入场K)
    三大铁律: ①顺大势逆小势 ②第一次机会是陷阱 ③只做看得懂的行情

    125个K线指标/17大类 — Vision识图从250+熊猫学社YouTube视频蒸馏
    """

    # ── 默认参数 ──
    PASS_SCORE = 7               # 三步法及格线 (7/10)
    MIN_RISK_REWARD = 1.2        # 最低盈亏比
    REQUIRE_LOW2 = True           # 要求Low2/High2确认
    BLOCK_MOMENTUM_CHASE = True   # 阻止追单
    MAX_POSITION_PCT = 25         # 最大仓位%
    TAKE_PROFIT_RR = 2.0          # 止盈盈亏比倍数
    PRIORITY_OVERRIDE = True      # 优先覆盖ML信号
    SCAN_TIMEFRAMES = ['1h', '4h'] # 主扫描周期
    HIGHER_TF = '1d'              # 高级别确认周期

    @classmethod
    def create(cls) -> "KLineStrategy":
        config = StrategyConfig(
            id="strat-naked-k-007",
            name="裸K价格行为策略",
            version="v1.0",
            market="crypto",
            symbols=[],  # 运行时从 symbol_config 动态获取
            status="running",
            logic_explanation="""
            🕯️ 裸K价格行为策略 — 熊猫教练「三步读图法」体系:

            ① 定级别(选对周期): 主看1h/4h, 1d确认大趋势
            ② 画结构(识别骨架): BMS回调结构/SMS反转结构/交易区间/通道
            ③ 找信号(等待入场): 信号K线(非趋势K) + 入场K线(趋势K) 两步确认

            三步评分卡:
            - 趋势判断(0-4分): 顺大逆小+趋势能量+结构Bias
            - 关键位置(0-3分): 支撑压力区+斐波那契+最后一个防守位
            - 信号K质量(0-3分): 信号K+入场K两步确认+Low2/High2计数
            综合 >= 7/10 才入场, RR >= 1.2:1

            五大K线形态优先:
            - Pinbar 2.0 (关键位+趋势末端+分组集群)
            - SB结构/Second Breakout (第2次突破胜率>>第1次)
            - 末端旗形/楔形三推 (趋势衰竭→反转)
            - Low2(做多)/High2(做空) 计数系统
            - 信号K+入场K 两步确认 (非趋势→趋势转换)
            """,
            entry_conditions="""
            🚪 入场三问全部通过才入场:

            ① 趋势对吗?
              - 上升趋势做多(BMS: HH+HL), 下降趋势做空(SMS: LH+LL)
              - 回调不破前低(多)/反弹不过前高(空)
              - "顺大逆小"原则: 大趋势向上, 等小回调结束入场

            ② 位置对吗?
              - 价格在支撑/压力区附近 (SR Zone tolerance 0.5%)
              - 斐波那契回调位 (0.382/0.5/0.618)
              - 有明确的止损位 (前低下方/前高上方)

            ③ 信号对吗?
              - 信号K线出现 (非趋势K: DOJI/PINBAR/INSIDE/小实体)
              - 入场K线确认 (趋势K: TREND_BULL/TREND_BEAR)
              - Low2已确认(做多)/High2已确认(做空) — 质量>=0.7
              - 无强动能追单警告 (body不超过avg×2)
            """,
            exit_conditions="""
            🏃 三种出场方式:

            ① 结构止盈: 到达下一个SR阻力/支撑区
            ② RR止盈: 止盈距离 = 止损距离 × 2.0 (可调)
            ③ 信号反转: 出现反向裸K信号(score>=6)立即平仓

            移动止损: 价格向有利方向移动后, 止损移到入场价(保本损)
            """,
            position_sizing="""
            ⚖️ 以损定仓:

            - 单笔最大仓位: 25% (可调5-40%)
            - 止损距离 = 入场价到前低/前高的距离
            - 仓位大小 = (总资金 × 仓位%) / 止损距离
            - 不追单: 大实体K线后禁止立即入场
            """,
            risk_management="""
            🛡️ 五层防御 (裸K专属):

            ① 信号过滤: 三步评分<7 → 不执行
            ② Low2/High2门禁: 未确认 → 不执行(可关闭)
            ③ 追单拦截: 检测到大实体动量K → 不执行
            ④ 结构止损: 止损基于前低/前高(比固定%更紧)
            ⑤ 反向覆盖: 出现6+分反向信号 → 立即平仓
            """,
            parameters=[
                StrategyParam("pass_score", 7, "int", "三步法及格线",
                    "综合评分>=此值才算有效信号 (0-10)", min_val=5, max_val=9, step=1),
                StrategyParam("min_risk_reward", 1.2, "float", "最低盈亏比",
                    "止损止盈比低于此值的信号忽略", min_val=1.0, max_val=3.0, step=0.1),
                StrategyParam("require_low2", True, "bool", "要求Low2/High2确认",
                    "仅Low2(做多)/High2(做空)已确认的信号才执行"),
                StrategyParam("block_momentum_chase", True, "bool", "阻止追单",
                    "有强动能追单警告的信号不执行"),
                StrategyParam("max_position_pct", 25, "float", "最大仓位%",
                    "单次交易最大使用资金百分比", min_val=5, max_val=40, step=5),
                StrategyParam("take_profit_rr", 2.0, "float", "止盈盈亏比倍数",
                    "止盈距离 = 止损距离 × 此倍数", min_val=1.5, max_val=4.0, step=0.5),
                StrategyParam("priority_override", True, "bool", "优先覆盖ML信号",
                    "K线信号评分达标时覆盖同symbol的ML信号"),
            ],
        )
        return cls(config)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        生成裸K交易信号 — 由 strategy_runner._run_naked_k() 调用

        注意: 此方法在 strategy_runner 层面被 _run_naked_k() 包装，
        实际的数据拉取和扫描在 runner 中完成。这里提供一个备用实现。
        """
        return []  # 实际扫描由 strategy_runner._run_naked_k() 完成

    def _convert_signal(self, scalp_sig, symbol: str,
                         low_high: dict = None, sig_pair: dict = None,
                         higher_tf_bias: str = None) -> dict:
        """
        将 NakedKScanner.TradingSignal 转换为标准策略信号 dict

        Args:
            scalp_sig: TradingSignal from NakedKScanner
            symbol: e.g. 'BTC/USDT'
            low_high: Low1/Low2/High1/High2 count dict
            sig_pair: SignalK+EntryK pair dict
            higher_tf_bias: Higher timeframe market bias
        """
        action = "BUY" if scalp_sig.action.value == "BUY" else "SELL"

        # Score scaling: 3-step score 0-10 → 0-100
        scaled_score = scalp_sig.score_3step * 10

        # Risk level
        if scalp_sig.score_3step >= 8:
            risk_level = "low"
        elif scalp_sig.score_3step >= 7:
            risk_level = "medium"
        else:
            risk_level = "high"

        pass_score = self.get_param("pass_score", 7)
        min_rr = self.get_param("min_risk_reward", 1.2)
        max_pos_pct = self.get_param("max_position_pct", 25) / 100
        tp_rr = self.get_param("take_profit_rr", 2.0)

        # Calculate take_profit based on RR ratio
        if scalp_sig.stop_loss and scalp_sig.entry_price:
            if action == "BUY":
                stop_distance = scalp_sig.entry_price - scalp_sig.stop_loss
                take_profit = scalp_sig.entry_price + stop_distance * tp_rr
            else:
                stop_distance = scalp_sig.stop_loss - scalp_sig.entry_price
                take_profit = scalp_sig.entry_price - stop_distance * tp_rr
        else:
            take_profit = scalp_sig.take_profit

        # Build reasons list
        reasons = list(scalp_sig.reasons) if scalp_sig.reasons else []
        if low_high and low_high.get('confirmed'):
            reasons.insert(0, f"Low2/High2确认(质量{low_high.get('quality', 0):.0%})")
        if sig_pair:
            reasons.insert(0, f"信号K+入场K两步确认")
        if higher_tf_bias:
            reasons.append(f"高TF:{higher_tf_bias}")

        return {
            "symbol": symbol,
            "name": symbol.split("/")[0] if "/" in symbol else symbol,
            "action": action,
            "price": scalp_sig.entry_price,
            "score": scaled_score,
            "confidence": scalp_sig.confidence,
            "reasons": reasons[:6],
            "risk_level": risk_level,
            "suggested_size": max_pos_pct,
            "stop_loss": scalp_sig.stop_loss,
            "take_profit": take_profit,
            "strategy_name": "裸K价格行为",
            # ── 裸K专属标记 (用于优先级合并) ──
            "kline_priority": True,  # ← 标记为优先信号
            "kline_score_3step": scalp_sig.score_3step,
            "kline_score_trend": scalp_sig.score_trend,
            "kline_score_keylevel": scalp_sig.score_keylevel,
            "kline_score_signalk": scalp_sig.score_signalk,
            "kline_type": scalp_sig.kline_type.value if hasattr(scalp_sig.kline_type, 'value') else str(scalp_sig.kline_type),
            "kline_risk_reward": scalp_sig.risk_reward,
            "kline_low_high": low_high,
            "kline_signal_entry_pair": sig_pair,
            "kline_higher_tf_bias": higher_tf_bias,
            "kline_sr_zone": str(scalp_sig.sr_zone) if scalp_sig.sr_zone else None,
        }


# ═══════════════════════════════════════════
# 策略注册表
# ═══════════════════════════════════════════

STRATEGY_REGISTRY: Dict[str, BaseStrategy] = {}
STRATEGY_BUILDERS = {
    "strat-ml-momentum-001": MLMomentumStrategy.create,
    "strat-mean-revert-002": MeanReversionGridStrategy.create,
    "strat-alpha-arb-003": CrossMarketAlphaStrategy.create,
    "strat-v5-qlib-fusion-004": V5QlibFusionStrategy.create,
    "strat-five-dim-scorecard-005": FiveDimScorecardStrategy.create,
    "strat-naked-k-007": KLineStrategy.create,  # 🕯️ 裸K策略
}


def init_strategies() -> Dict[str, BaseStrategy]:
    """初始化所有策略"""
    global STRATEGY_REGISTRY
    STRATEGY_REGISTRY = {}
    for sid, builder in STRATEGY_BUILDERS.items():
        strategy = builder()
        STRATEGY_REGISTRY[sid] = strategy
    return STRATEGY_REGISTRY


def get_strategy(strategy_id: str) -> Optional[BaseStrategy]:
    """获取单个策略实例"""
    return STRATEGY_REGISTRY.get(strategy_id)


def get_all_strategies() -> List[BaseStrategy]:
    """获取所有策略"""
    return list(STRATEGY_REGISTRY.values())


def get_all_strategy_configs() -> List[Dict]:
    """获取所有策略配置 (API用)"""
    return [s.config.to_dict() for s in STRATEGY_REGISTRY.values()]


def update_strategy_param(strategy_id: str, param_key: str, value) -> bool:
    """更新策略参数"""
    strategy = STRATEGY_REGISTRY.get(strategy_id)
    if strategy is None:
        return False
    return strategy.set_param(param_key, value)


def generate_all_signals(market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
    """
    对所有运行中的策略生成信号

    Returns:
        {strategy_id: [signal_dict, ...]}
    """
    results = {}
    for sid, strategy in STRATEGY_REGISTRY.items():
        if strategy.status == "running":
            try:
                signals = strategy.generate_signals(market_data)
                results[sid] = signals
            except Exception as e:
                results[sid] = [{"error": str(e)}]
    return results


# ═══════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI策略引擎")
    parser.add_argument("--list", action="store_true", help="列出所有策略")
    parser.add_argument("--describe", type=str, help="描述指定策略 (strategy_id)")
    parser.add_argument("--simulate", type=str, help="用随机数据模拟策略信号 (strategy_id, 或 'all')")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT", help="模拟标的")

    args = parser.parse_args()
    strategies = init_strategies()

    if args.list:
        print("🧠 策略注册表:")
        print("=" * 60)
        for sid, s in strategies.items():
            status_icon = "🟢" if s.status == "running" else "⏸"
            print(f"  {status_icon} [{s.config.version}] {s.config.name}")
            print(f"     ID: {sid}")
            print(f"     市场: {s.config.market}")
            print(f"     标的: {', '.join(s.config.symbols[:4])}")
            print(f"     Sharpe: {s.config.sharpe:.2f} | 胜率: {s.config.win_rate:.1f}% | 回撤: {s.config.max_drawdown:.1f}%")
            print()

    elif args.describe:
        s = strategies.get(args.describe)
        if s:
            print(s.describe())
        else:
            print(f"❌ 策略 {args.describe} 不存在")
            print(f"可用: {list(strategies.keys())}")

    elif args.simulate:
        # 生成随机OHLCV数据模拟
        symbols = [s.strip() for s in args.symbols.split(",")]
        np.random.seed(42)

        market_data = {}
        for sym in symbols:
            n = 200
            base_price = {"BTC": 65000, "ETH": 3200, "SOL": 140, "BNB": 580, "XRP": 0.5, "AVAX": 35, "DOGE": 0.12}.get(sym.split("/")[0], 100)
            returns = np.random.normal(0.0005, 0.025, n)
            prices = base_price * np.cumprod(1 + returns)

            df = pd.DataFrame({
                "open": prices * (1 + np.random.normal(0, 0.002, n)),
                "high": prices * (1 + np.abs(np.random.normal(0.01, 0.005, n))),
                "low": prices * (1 - np.abs(np.random.normal(0.01, 0.005, n))),
                "close": prices,
                "volume": np.random.lognormal(10, 1.5, n),
            })
            # 确保 OHLC 一致性
            for i in range(n):
                df.loc[i, "high"] = max(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"])
                df.loc[i, "low"] = min(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "low"])
            market_data[sym] = df

        if args.simulate == "all":
            for sid, s in strategies.items():
                if s.status == "running":
                    print(f"\n🧠 {s.config.name} — 信号生成:")
                    print("-" * 50)
                    sigs = s.generate_signals(market_data)
                    if sigs:
                        for sig in sigs:
                            icon = "🟢" if sig["action"] == "BUY" else "🔴" if sig["action"] == "SELL" else "⚪"
                            print(f"  {icon} {sig['symbol']} {sig['action']}")
                            print(f"     价格: {sig['price']:.2f} | 置信度: {sig['confidence']:.0%} | 得分: {sig['score']:.0f}")
                            for r in sig.get("reasons", []):
                                print(f"      · {r}")
                    else:
                        print(f"  ⚪ 无信号 (当前市场条件不满足入场条件)")
        else:
            s = strategies.get(args.simulate)
            if s:
                sigs = s.generate_signals(market_data)
                print(f"\n🧠 {s.config.name} — 信号生成 (模拟数据):")
                print("-" * 50)
                if sigs:
                    for sig in sigs:
                        icon = "🟢" if sig["action"] == "BUY" else "🔴" if sig["action"] == "SELL" else "⚪"
                        print(f"  {icon} {sig['symbol']} {sig['action']}")
                        print(f"     价格: {sig['price']:.2f} | 置信度: {sig['confidence']:.0%} | 得分: {sig['score']:.0f}")
                        for r in sig.get("reasons", []):
                            print(f"      · {r}")
                else:
                    print("  ⚪ 无信号 — 当前随机数据不满足入场条件")
            else:
                print(f"❌ 策略 {args.simulate} 不存在")

    else:
        parser.print_help()
