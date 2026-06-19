"""
裸K价格行为扫描引擎 — 基于熊猫教练「熊猫讲裸K」交易体系
============================================================
Chase量化策略: 自动化K线形态识别 + 三步法入场信号评分

理论来源:
  - 熊猫教练 31讲《熊猫讲裸K》免费课 + 300+集VIP系统课
  - Al Brooks《Reading Price Charts Bar by Bar》四部曲
  - 12金K精讲 (反转金K + 进攻金K)
  - SMC订单流 (OB/FVG/流动性原理)

核心理念:
  "先画地图(结构+关键位) → 再看天气(趋势方向) → 最后等信号(2+3评分)"
  "看见了再交易" — 每根K线收定后才做判断，绝不做左侧预测

引擎能力:
  1. 市场结构识别 — HH/HL/LH/LL序列 + BMS回调结构 + SMS反转结构
  2. K线三分类   — 趋势K/非趋势K/信号K (含Pinbar/吞没/外包/十字星)
  3. 孤立支点检测 — 3根法Wickoff支点 → 支撑阻力Zone构建
  4. 三步法评分   — ①判趋势 ②找关键位 ③2+3确认 → 综合入场评分
  5. 通道识别     — 平行通道 + 交易区间检测
  6. 多周期验证   — 不逆4倍原则 (大周期定方向, 小周期找入场)

使用:
  from naked_k_scanner import NakedKScanner

  scanner = NakedKScanner(df_ohlcv)
  result = scanner.scan()  # 全量扫描

  # result.signals        — 所有入场信号
  # result.market_bias    — 'BULLISH' / 'BEARISH' / 'NEUTRAL'
  # result.structure      — 市场结构分析
  # result.kline_summary  — K线类型统计

依赖: numpy, pandas (纯Python计算, 无额外依赖)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════

class KLineKind(str, Enum):
    """K线种类 — 熊猫教练三分类体系"""
    TREND_BULL = "TREND_BULL"         # 看涨趋势K: 实体大, 重叠少, 收在顶部
    TREND_BEAR = "TREND_BEAR"         # 看跌趋势K: 实体大, 重叠少, 收在底部
    NON_TREND = "NON_TREND"           # 非趋势K: 实体小, 大量重叠, 多空平衡
    DOJI = "DOJI"                     # 十字星: 实体极小, 开盘≈收盘
    PINBAR_BULL = "PINBAR_BULL"       # 看涨Pinbar: 长下影线, 关键位反转信号
    PINBAR_BEAR = "PINBAR_BEAR"       # 看跌Pinbar: 长上影线, 关键位反转信号
    ENGULFING_BULL = "ENGULFING_BULL" # 看涨吞没: 阳包阴
    ENGULFING_BEAR = "ENGULFING_BEAR" # 看跌吞没: 阴包阳
    OUTSIDE_BULL = "OUTSIDE_BULL"     # 看涨外包: 波动扩大+收阳
    OUTSIDE_BEAR = "OUTSIDE_BEAR"     # 看跌外包: 波动扩大+收阴
    INSIDE = "INSIDE"                 # 内包K (孕线母/子): 被前K完全包含


class SwingKind(str, Enum):
    """摆动点类型"""
    HH = "HH"  # Higher High — 更高的高点
    HL = "HL"  # Higher Low  — 更高的低点
    LH = "LH"  # Lower High  — 更低的高点
    LL = "LL"  # Lower Low   — 更低的低点


class MarketBias(str, Enum):
    """市场倾向"""
    BULLISH = "BULLISH"       # 多头趋势: HH + HL 序列
    BEARISH = "BEARISH"       # 空头趋势: LL + LH 序列
    CHANNEL_UP = "CHANNEL_UP" # 上升通道
    CHANNEL_DOWN = "CHANNEL_DOWN" # 下降通道
    RANGE = "RANGE"           # 交易区间 (横盘)
    TRANSITION = "TRANSITION" # 转换期 (SMS出现, 趋势可能反转)


class SignalAction(str, Enum):
    """信号动作"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class SwingPoint:
    """孤立支点 — Wickoff 3根法"""
    index: int
    price: float
    kind: SwingKind
    strength: float = 0.0  # 0-1, 支点有多"干净"
    left_bars: int = 1     # 左侧确认K线数
    right_bars: int = 1    # 右侧确认K线数

    def __repr__(self):
        return f"SwingPoint({self.kind.value} @ {self.price:.4f} idx={self.index})"


@dataclass
class SRZone:
    """支撑/阻力区域"""
    top: float
    bottom: float
    kind: str             # 'support' / 'resistance'
    touches: int = 0       # 历史触及次数
    strength: float = 0.0  # 0-1 综合强度
    recent_test: bool = False  # 最近是否被测试过
    swing_points: List[SwingPoint] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def width_pct(self) -> float:
        return (self.top - self.bottom) / self.mid * 100

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class KLineInfo:
    """单根K线的完整分类信息"""
    index: int
    kind: KLineKind
    body_ratio: float      # 实体/总区间 比值
    overlap_ratio: float   # 与前K重叠比例
    upper_wick_pct: float  # 上影线占比
    lower_wick_pct: float  # 下影线占比
    body_position: str     # 'top' / 'bottom' / 'middle'
    is_signal: bool = False
    signal_score: int = 0  # 2+3评分 (仅信号K有值)

    def __repr__(self):
        return f"KLineInfo({self.kind.value} body={self.body_ratio:.1%})"


@dataclass
class BMSSignal:
    """BMS/SMS 结构信号"""
    kind: str              # 'BMS_LONG' / 'BMS_SHORT' / 'SMS_WARNING'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float      # 0-1
    description: str


@dataclass
class TradingSignal:
    """完整入场信号 — 三步法最终输出"""
    action: SignalAction
    entry_price: float
    stop_loss: float
    take_profit: float
    score_3step: int       # 三步法综合评分 0-10
    score_2plus3: int      # 2+3评分 (信号K部分)
    confidence: float      # 综合置信度 0-1
    risk_reward: float     # 盈亏比
    reasons: List[str]
    signal_bar_idx: int
    kline_type: KLineKind
    sr_zone: Optional[SRZone] = None  # 触发的关键位

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "entry": round(self.entry_price, 4),
            "stop": round(self.stop_loss, 4),
            "target": round(self.take_profit, 4),
            "score": self.score_3step,
            "confidence": round(self.confidence, 2),
            "risk_reward": round(self.risk_reward, 2),
            "reasons": self.reasons,
            "bar_idx": self.signal_bar_idx,
            "kline": self.kline_type.value,
            "sr_zone": f"{self.sr_zone.bottom:.4f}-{self.sr_zone.top:.4f}" if self.sr_zone else None,
        }


@dataclass
class StructureAnalysis:
    """市场结构分析报告"""
    bias: MarketBias
    swing_points: List[SwingPoint]
    sr_zones: List[SRZone]
    trend_strength: float       # 0-1 趋势强度
    trend_energy: int           # 0-10 熊猫教练"能量评分"
    trend_k_ratio: float        # 趋势K占比
    bms_signals: List[BMSSignal] = field(default_factory=list)
    description: str = ""


@dataclass
class ComplexPullback:
    """复杂回调结构"""
    start_idx: int
    end_idx: int
    leg1_length: float       # 第一段回调长度
    leg2_length: float       # 第二段回调长度
    ratio: float             # leg2/leg1 比例
    target_price: float      # 测量目标价 (腿1≈腿2)
    direction: str           # 'bullish' / 'bearish' (回调结束后目标方向)
    confidence: float


@dataclass
class FVGap:
    """Fair Value Gap — SMC合理价值缺口"""
    index: int               # 缺口形成位置 (第2根K的索引)
    kind: str                # 'BISI' (看涨缺口) / 'SIBI' (看跌缺口)
    gap_top: float
    gap_bottom: float
    gap_size_pct: float      # 缺口大小%
    filled: bool = False     # 是否已被回补
    age_bars: int = 0        # 形成后经过了多少根K

@dataclass
class LowHighCount:
    """Low1/Low2/High1/High2 入场计数 — Vision识图: 熊猫教练核心入场系统

    Low 1: 上升趋势中第一次回调产生的低点 (打底)
    Low 2: 上升趋势中第二次回调产生的低点 (出击, 胜率更高)
    High 1: 下降趋势中第一次反弹产生的高点
    High 2: 下降趋势中第二次反弹产生的高点

    "低1打底, 低2出击" — 熊猫教练原话
    "有效信号K +入场K: 低2" — 两步确认+低2=高胜率
    """
    index: int
    count: int                # 1 or 2
    price: float
    kind: str                 # 'low1' / 'low2' / 'high1' / 'high2'
    signal_k_idx: int = -1    # 对应的信号K位置
    entry_k_idx: int = -1     # 对应的入场K位置
    quality: float = 0.0      # 0-1 信号质量
    confirmed: bool = False   # 信号K+入场K两步都满足?

    def __repr__(self):
        return f"{self.kind.upper()} @ {self.price:.4f} q={self.quality:.1f} conf={self.confirmed}"


@dataclass
class SignalEntryPair:
    """信号K+入场K两步确认对 — Vision识图核心概念

    信号K = 非趋势K线 (逆主趋势方向的小K线, "警报")
    入场K = 趋势K线 (顺主趋势方向的确认K线, "出击")

    "不需要数入场K线" — 入场K只看第1根顺趋势K, 不是数K线数量
    """
    signal_idx: int           # 信号K索引
    entry_idx: int            # 入场K索引
    direction: str            # 'LONG' / 'SHORT'
    signal_k_kind: str        # K线类型
    entry_k_kind: str         # 入场K类型
    at_key_level: bool = False  # 是否在关键位
    low_high_count: str = ''  # 'low1'/'low2'/'high1'/'high2'
    quality: float = 0.0


@dataclass
class ScanResult:
    """一次完整扫描的结果"""
    symbol: str
    timeframe: str
    market_bias: MarketBias
    bias_confidence: float
    signals: List[TradingSignal]
    structure: StructureAnalysis
    kline_summary: Dict[str, int]    # 各类型K线计数
    complex_pullbacks: List[ComplexPullback] = field(default_factory=list)
    fvgs: List[FVGap] = field(default_factory=list)
    order_blocks: List[Dict] = field(default_factory=list)
    channel_outcome: Optional[str] = None  # 通道三结局预测
    low_high_counts: List[Dict] = field(default_factory=list)  # 🆕 Low1/Low2/High1/High2
    signal_entry_pairs: List[Dict] = field(default_factory=list)  # 🆕 信号K+入场K对
    momentum_warnings: List[Dict] = field(default_factory=list)  # 🆕 强动能追单警告
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_bias": self.market_bias.value,
            "bias_confidence": round(self.bias_confidence, 2),
            "signals": [s.to_dict() for s in self.signals],
            "structure": {
                "trend_strength": round(self.structure.trend_strength, 2),
                "trend_energy": self.structure.trend_energy,
                "trend_k_ratio": round(self.structure.trend_k_ratio, 2),
                "swing_count": len(self.structure.swing_points),
                "sr_zone_count": len(self.structure.sr_zones),
                "bms_count": len(self.structure.bms_signals),
                "description": self.structure.description,
            },
            "kline_summary": self.kline_summary,
            "complex_pullbacks": [
                {"start": cp.start_idx, "end": cp.end_idx,
                 "leg1": round(cp.leg1_length, 4), "leg2": round(cp.leg2_length, 4),
                 "ratio": round(cp.ratio, 2), "target": round(cp.target_price, 4),
                 "direction": cp.direction, "confidence": round(cp.confidence, 2)}
                for cp in self.complex_pullbacks
            ],
            "fvgs": [
                {"index": f.index, "kind": f.kind,
                 "gap_top": round(f.gap_top, 4), "gap_bottom": round(f.gap_bottom, 4),
                 "size_pct": round(f.gap_size_pct, 2), "filled": f.filled}
                for f in self.fvgs
            ],
            "order_blocks": self.order_blocks,
            "channel_outcome": self.channel_outcome,
            "low_high_counts": self.low_high_counts,       # 🆕
            "signal_entry_pairs": self.signal_entry_pairs, # 🆕
            "momentum_warnings": self.momentum_warnings,   # 🆕
            "timestamp": self.timestamp,
        }
        return result


# ═══════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════

class NakedKScanner:
    """
    裸K价格行为扫描引擎

    一站式扫描: 输入OHLCV DataFrame → 输出完整交易信号

    使用:
        scanner = NakedKScanner(df_ohlcv, symbol="BTC/USDT", timeframe="1h")
        result = scanner.scan()
        for sig in result.signals:
            print(f"{sig.action.value} @ {sig.entry_price} | 评分={sig.score_3step}")
    """

    # ── 可调参数 ──
    # K线分类阈值 (Al Brooks 量化标准, 适配加密货币波动特征)
    TREND_BODY_MIN = 0.35       # 趋势K实体最小占比 (加密货币1h周期)
    DOJI_BODY_MAX = 0.10        # 十字星实体最大占比
    NON_TREND_BODY_MAX = 0.25   # 非趋势K实体最大占比
    OVERLAP_TREND_MAX = 0.45    # 趋势K最大重叠度 (<45% = 几乎不重叠)
    OVERLAP_NON_TREND_MIN = 0.45  # 非趋势K最小重叠度 (>45% = 明显重叠)
    CLOSE_EXTREME_ZONE = 0.40   # 收盘在顶部/底部40%区间内

    # Pinbar阈值
    PINBAR_WICK_RATIO = 2.0     # 长影线 ≥ 实体 × 2
    PINBAR_BODY_ZONE = 0.35     # 实体必须在K线的一端35%区间内
    PINBAR_NOSE_RATIO = 0.10    # 反向影线 ≤ 总区间的10%

    # 摆动点参数
    SWING_LOOKBACK = 1           # 3根法: 左右各1根
    SWING_STRONG_LOOKBACK = 2    # 强支点: 左右各2根 (5根法)

    # 支撑阻力区域
    ZONE_TOLERANCE = 0.005       # Zone容差 0.5% (合并相近支点)
    ZONE_MIN_TOUCHES = 1         # 最少触及次数
    ZONE_MAX_COUNT = 8           # 最多保留的Zone数量 (只保留最强)

    # 三步法评分权重
    SCORE_TREND_MAX = 4          # 第一步: 趋势判断 最高4分
    SCORE_ZONE_MAX = 3           # 第二步: 关键位 最高3分
    SCORE_SIGNAL_MAX = 3         # 第三步: 信号K 最高3分
    SCORE_PASS_THRESHOLD = 7     # 入场及格线 (总分10)
    MIN_RISK_REWARD = 1.2        # 最低盈亏比要求

    # 盈亏比要求
    RISK_REWARD_MIN = 1.5        # 最低盈亏比

    # 通道/区间参数
    CHANNEL_MIN_TOUCHES = 3      # 通道至少3点触及边界
    RANGE_MIN_BARS = 10          # 交易区间最少K线数

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        higher_tf_df: Optional[pd.DataFrame] = None,  # 大周期数据(不逆4倍原则)
    ):
        """
        Args:
            df: OHLCV DataFrame, 列名 ['open','high','low','close','volume']
            symbol: 交易对名称
            timeframe: K线周期 ('5m','15m','1h','4h','1d')
            higher_tf_df: 4倍以上大周期数据, 用于趋势方向确认
        """
        self.df = df.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.higher_tf_df = higher_tf_df

        # 预计算
        self._n = len(df)
        self._body = (df['close'] - df['open']).abs().values
        self._range = (df['high'] - df['low']).values
        self._body_ratio = np.divide(self._body, self._range,
                                     out=np.zeros_like(self._body),
                                     where=self._range > 0)
        self._direction = np.sign(df['close'].values - df['open'].values)  # 1=阳, -1=阴, 0=十字

        # 缓存
        self._swing_points: List[SwingPoint] = []
        self._kline_infos: List[KLineInfo] = []
        self._sr_zones: List[SRZone] = []

        # 大周期方向
        self._higher_tf_bias: Optional[MarketBias] = None
        if higher_tf_df is not None and len(higher_tf_df) >= 20:
            htf_scanner = NakedKScanner(higher_tf_df, symbol, "higher_tf")
            htf_struct = htf_scanner._analyze_structure()
            self._higher_tf_bias = htf_struct.bias

    # ═══════════════════════════════════════
    # 公开API
    # ═══════════════════════════════════════

    def scan(self) -> ScanResult:
        """执行全量扫描, 返回结构化结果"""
        from datetime import datetime

        # 1. 分类每一根K线
        self._classify_all_klines()

        # 2. 检测孤立支点
        self._detect_swing_points()

        # 3. 构建支撑阻力Zone
        self._build_sr_zones()

        # 4. 分析市场结构
        structure = self._analyze_structure()

        # 5. 三步法扫描入场信号
        signals = self._scan_signals(structure)

        # 6. 🆕 2B/SB结构信号
        signals_2b_sb = self._scan_2b_sb_signals(structure)
        signals.extend(signals_2b_sb)
        signals = self._deduplicate_signals(signals)

        # 7. 🆕 复杂回调检测
        complex_pullbacks = self._detect_complex_pullback(structure)

        # 8. 🆕 FVG检测
        fvgs = self._detect_fvg()

        # 9. 🆕 Order Block检测
        order_blocks = self._detect_order_blocks()

        # 10. 🆕 通道三结局预测
        channel_outcome = self._predict_channel_outcome(structure)

        # 11. 🆕 Low1/Low2/High1/High2 入场计数 (Vision识图核心)
        low_high_counts = self._detect_low_high_sequence(structure)

        # 12. 🆕 信号K+入场K两步确认对
        signal_entry_pairs = self._detect_signal_entry_pairs(structure)

        # 13. 🆕 强动能追单警告
        momentum_warnings = self._detect_momentum_chase(structure)

        # 14. 汇总
        kline_counts = {}
        for ki in self._kline_infos:
            kline_counts[ki.kind.value] = kline_counts.get(ki.kind.value, 0) + 1

        return ScanResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            market_bias=structure.bias,
            bias_confidence=structure.trend_strength,
            signals=signals,
            structure=structure,
            kline_summary=kline_counts,
            complex_pullbacks=complex_pullbacks,
            fvgs=fvgs,
            order_blocks=order_blocks,
            channel_outcome=channel_outcome,
            low_high_counts=[lh.__dict__ for lh in low_high_counts],      # 🆕
            signal_entry_pairs=[sp.__dict__ for sp in signal_entry_pairs], # 🆕
            momentum_warnings=momentum_warnings,                           # 🆕
            timestamp=datetime.now().isoformat(),
        )

    def scan_latest(self) -> Optional[TradingSignal]:
        """只返回最新一根K线的信号 (用于实时监控)"""
        result = self.scan()
        if result.signals:
            return result.signals[-1]
        return None

    def get_kline_info(self, idx: int) -> Optional[KLineInfo]:
        """获取指定位置K线的分类信息"""
        if 0 <= idx < len(self._kline_infos):
            return self._kline_infos[idx]
        return None

    # ═══════════════════════════════════════
    # 第一步: K线三分类
    # ═══════════════════════════════════════

    def _classify_all_klines(self):
        """对所有K线执行趋势K/非趋势K/信号K三分法"""
        self._kline_infos = []

        for i in range(self._n):
            ki = self._classify_single_kline(i)
            self._kline_infos.append(ki)

    def _classify_single_kline(self, idx: int) -> KLineInfo:
        """分类单根K线 — 核心分类逻辑"""
        body_r = self._body_ratio[idx]
        rng = self._range[idx]
        direction = self._direction[idx]
        o, h, l, c = (self.df.iloc[idx]['open'], self.df.iloc[idx]['high'],
                       self.df.iloc[idx]['low'], self.df.iloc[idx]['close'])

        # 影线计算
        if direction > 0:  # 阳线
            upper_wick = (h - c) / rng if rng > 0 else 0
            lower_wick = (o - l) / rng if rng > 0 else 0
            body_pos = 'top' if upper_wick < self.CLOSE_EXTREME_ZONE else 'middle'
        elif direction < 0:  # 阴线
            upper_wick = (h - o) / rng if rng > 0 else 0
            lower_wick = (c - l) / rng if rng > 0 else 0
            body_pos = 'bottom' if lower_wick < self.CLOSE_EXTREME_ZONE else 'middle'
        else:
            upper_wick = (h - c) / rng if rng > 0 else 0.5
            lower_wick = (c - l) / rng if rng > 0 else 0.5
            body_pos = 'middle'

        # 重叠度 (与前一根K线)
        overlap_r = self._calc_overlap(idx)

        # ── 分类决策树 ──
        kind = self._determine_kline_kind(
            idx, body_r, rng, direction, upper_wick, lower_wick, body_pos, overlap_r
        )

        return KLineInfo(
            index=idx,
            kind=kind,
            body_ratio=body_r,
            overlap_ratio=overlap_r,
            upper_wick_pct=upper_wick,
            lower_wick_pct=lower_wick,
            body_position=body_pos,
            is_signal=kind.value.startswith("PINBAR") or kind.value.startswith("ENGULFING"),
        )

    def _determine_kline_kind(
        self, idx: int, body_r: float, rng: float, direction: float,
        upper_w: float, lower_w: float, body_pos: str, overlap_r: float
    ) -> KLineKind:
        """K线分类决策树 — 熊猫教练 + Al Brooks 量化标准"""

        # ── 十字星 ──
        if body_r <= self.DOJI_BODY_MAX:
            return KLineKind.DOJI

        # ── Pinbar 检测 (在非趋势K判定前, 因为Pinbar实体也不大) ──
        pinbar_result = self._check_pinbar(idx, body_r, rng, direction, upper_w, lower_w)
        if pinbar_result is not None:
            return pinbar_result

        # ── 吞没检测 ──
        engulf_result = self._check_engulfing(idx)
        if engulf_result is not None:
            return engulf_result

        # ── 外包K检测 ──
        outside_result = self._check_outside_bar(idx)
        if outside_result is not None:
            return outside_result

        # ── 内包K ──
        if idx > 0:
            prev_h = self.df.iloc[idx - 1]['high']
            prev_l = self.df.iloc[idx - 1]['low']
            if self.df.iloc[idx]['high'] <= prev_h and self.df.iloc[idx]['low'] >= prev_l:
                return KLineKind.INSIDE

        # ── 非趋势K ──
        if body_r <= self.NON_TREND_BODY_MAX and overlap_r >= self.OVERLAP_NON_TREND_MIN:
            return KLineKind.NON_TREND

        # ── 趋势K ──
        if body_r >= self.TREND_BODY_MIN and overlap_r <= self.OVERLAP_TREND_MAX:
            if body_pos == 'top' and direction >= 0:
                return KLineKind.TREND_BULL
            if body_pos == 'bottom' and direction <= 0:
                return KLineKind.TREND_BEAR
            # 实体大但位置不对 — 按方向分
            if direction > 0:
                return KLineKind.TREND_BULL
            if direction < 0:
                return KLineKind.TREND_BEAR

        # ── 默认: 非趋势 ──
        return KLineKind.NON_TREND

    def _check_pinbar(
        self, idx: int, body_r: float, rng: float, direction: float,
        upper_w: float, lower_w: float
    ) -> Optional[KLineKind]:
        """Pinbar检测 — 长影线≥实体×2, 反向影线≤10%, 实体在一端35%区间内"""
        if rng <= 0:
            return None

        # 看涨Pinbar: 长下影线, 实体在顶部
        if (lower_w >= body_r * self.PINBAR_WICK_RATIO
                and upper_w <= self.PINBAR_NOSE_RATIO
                and lower_w >= 0.50):  # 下影线至少占一半
            # 实体必须在K线上端35%内 (close靠近high)
            o, h, l, c = (self.df.iloc[idx]['open'], self.df.iloc[idx]['high'],
                           self.df.iloc[idx]['low'], self.df.iloc[idx]['close'])
            body_top = max(o, c)
            if (h - body_top) / rng <= self.PINBAR_BODY_ZONE:
                return KLineKind.PINBAR_BULL

        # 看跌Pinbar: 长上影线, 实体在底部
        if (upper_w >= body_r * self.PINBAR_WICK_RATIO
                and lower_w <= self.PINBAR_NOSE_RATIO
                and upper_w >= 0.50):  # 上影线至少占一半
            o, h, l, c = (self.df.iloc[idx]['open'], self.df.iloc[idx]['high'],
                           self.df.iloc[idx]['low'], self.df.iloc[idx]['close'])
            body_bottom = min(o, c)
            if (body_bottom - l) / rng <= self.PINBAR_BODY_ZONE:
                return KLineKind.PINBAR_BEAR

        return None

    def _check_engulfing(self, idx: int) -> Optional[KLineKind]:
        """吞没形态检测 — 实体完全包裹前K实体, 方向相反, 当前必须是趋势K级别"""
        if idx < 1:
            return None

        prev_o, prev_c = self.df.iloc[idx - 1]['open'], self.df.iloc[idx - 1]['close']
        curr_o, curr_c = self.df.iloc[idx]['open'], self.df.iloc[idx]['close']

        prev_body_top = max(prev_o, prev_c)
        prev_body_bot = min(prev_o, prev_c)
        curr_body_top = max(curr_o, curr_c)
        curr_body_bot = min(curr_o, curr_c)
        prev_body = abs(prev_c - prev_o)
        curr_body = abs(curr_c - curr_o)
        curr_range = self.df.iloc[idx]['high'] - self.df.iloc[idx]['low']
        curr_body_r = curr_body / curr_range if curr_range > 0 else 0

        if curr_body <= 0:
            return None

        # 必须满足趋势K实体阈值
        if curr_body_r < self.TREND_BODY_MIN:
            return None

        # 看涨吞没: 前阴后阳, 当前实体完全包裹前实体
        if (prev_c < prev_o  # 前一根阴线
                and curr_c > curr_o  # 当前阳线
                and curr_body_top >= prev_body_top
                and curr_body_bot <= prev_body_bot
                and curr_body > prev_body):  # 实体必须更大
            return KLineKind.ENGULFING_BULL

        # 看跌吞没: 前阳后阴
        if (prev_c > prev_o  # 前一根阳线
                and curr_c < curr_o  # 当前阴线
                and curr_body_top >= prev_body_top
                and curr_body_bot <= prev_body_bot
                and curr_body > prev_body):
            return KLineKind.ENGULFING_BEAR

        return None

    def _check_outside_bar(self, idx: int) -> Optional[KLineKind]:
        """外包K检测 — 最高>前K最高 且 最低<前K最低, 强动能信号"""
        if idx < 1:
            return None

        prev_h, prev_l = self.df.iloc[idx - 1]['high'], self.df.iloc[idx - 1]['low']
        curr_h, curr_l = self.df.iloc[idx]['high'], self.df.iloc[idx]['low']
        curr_c, curr_o = self.df.iloc[idx]['close'], self.df.iloc[idx]['open']

        if curr_h > prev_h and curr_l < prev_l:
            if curr_c >= curr_o:
                return KLineKind.OUTSIDE_BULL
            else:
                return KLineKind.OUTSIDE_BEAR

        return None

    def _calc_overlap(self, idx: int) -> float:
        """计算与前一根K线的重叠比例"""
        if idx < 1:
            return 0.0

        curr_h, curr_l = self.df.iloc[idx]['high'], self.df.iloc[idx]['low']
        prev_h, prev_l = self.df.iloc[idx - 1]['high'], self.df.iloc[idx - 1]['low']

        curr_range = curr_h - curr_l
        prev_range = prev_h - prev_l

        if curr_range <= 0 or prev_range <= 0:
            return 0.0

        # 重叠区间
        overlap_top = min(curr_h, prev_h)
        overlap_bot = max(curr_l, prev_l)
        overlap_size = max(0, overlap_top - overlap_bot)

        # 与前K重叠比例
        return overlap_size / prev_range

    # ═══════════════════════════════════════
    # 第二步: 孤立支点检测
    # ═══════════════════════════════════════

    def _detect_swing_points(self):
        """检测所有孤立高低点 — 3根法 + 5根强支点"""
        self._swing_points = []

        for i in range(1, self._n - 1):
            curr_h = self.df.iloc[i]['high']
            curr_l = self.df.iloc[i]['low']

            # 3根法检测
            left_h = self.df.iloc[i - 1]['high']
            left_l = self.df.iloc[i - 1]['low']
            right_h = self.df.iloc[i + 1]['high']
            right_l = self.df.iloc[i + 1]['low']

            # 孤立高点: 当前High > 左右High
            if curr_h > left_h and curr_h > right_h:
                strength = self._calc_swing_strength(i, is_high=True, lookback=self.SWING_LOOKBACK)
                self._swing_points.append(SwingPoint(
                    index=i, price=curr_h, kind=SwingKind.HH if i > 0 else SwingKind.HH,
                    strength=strength, left_bars=1, right_bars=1
                ))

            # 孤立低点: 当前Low < 左右Low
            if curr_l < left_l and curr_l < right_l:
                strength = self._calc_swing_strength(i, is_high=False, lookback=self.SWING_LOOKBACK)
                self._swing_points.append(SwingPoint(
                    index=i, price=curr_l, kind=SwingKind.LL if i > 0 else SwingKind.LL,
                    strength=strength, left_bars=1, right_bars=1
                ))

        # 标注 HH/HL/LH/LL (与前一摆动点比较)
        self._label_swing_sequence()

    def _calc_swing_strength(self, idx: int, is_high: bool, lookback: int = 1) -> float:
        """计算支点强度 — 左右K线的价格差距越大越强"""
        score = 0.0
        n_checks = 0

        for offset in range(1, lookback + 1):
            li = idx - offset
            ri = idx + offset
            if li < 0 or ri >= self._n:
                continue

            if is_high:
                gap_l = self.df.iloc[idx]['high'] - self.df.iloc[li]['high']
                gap_r = self.df.iloc[idx]['high'] - self.df.iloc[ri]['high']
            else:
                gap_l = self.df.iloc[li]['low'] - self.df.iloc[idx]['low']
                gap_r = self.df.iloc[ri]['low'] - self.df.iloc[idx]['low']

            avg_range = (self._range[li] + self._range[ri]) / 2
            if avg_range > 0:
                score += min(1.0, (gap_l + gap_r) / (2 * avg_range))
            n_checks += 1

        return score / max(1, n_checks)

    def _label_swing_sequence(self):
        """标注摆动点序列: HH/HL/LH/LL"""
        if len(self._swing_points) < 2:
            return

        highs = [sp for sp in self._swing_points
                 if sp.price == self.df.iloc[sp.index]['high']]
        lows = [sp for sp in self._swing_points
                if sp.price == self.df.iloc[sp.index]['low']]

        # 重新标注高点序列
        for i in range(len(highs)):
            if i == 0:
                highs[i].kind = SwingKind.HH  # 第一个默认为HH
            else:
                highs[i].kind = (SwingKind.HH if highs[i].price > highs[i - 1].price
                                 else SwingKind.LH)

        # 重新标注低点序列
        for i in range(len(lows)):
            if i == 0:
                lows[i].kind = SwingKind.HL
            else:
                lows[i].kind = (SwingKind.HL if lows[i].price > lows[i - 1].price
                                else SwingKind.LL)

    # ═══════════════════════════════════════
    # 第三步: 支撑阻力Zone
    # ═══════════════════════════════════════

    def _build_sr_zones(self):
        """从孤立支点构建支撑阻力区域"""
        self._sr_zones = []
        if len(self._swing_points) < 2:
            return

        avg_price = float(self.df.iloc[-1]['close'])
        tolerance = avg_price * self.ZONE_TOLERANCE

        # 分组: 相近的支点合并为一个Zone
        highs = sorted(
            [sp for sp in self._swing_points if sp.kind in (SwingKind.HH, SwingKind.LH)],
            key=lambda x: x.price
        )
        lows = sorted(
            [sp for sp in self._swing_points if sp.kind in (SwingKind.HL, SwingKind.LL)],
            key=lambda x: x.price
        )

        # 阻力Zone (从高点构建)
        res_zones = self._cluster_swings_to_zones(highs, 'resistance', tolerance, avg_price)
        # 支撑Zone (从低点构建)
        sup_zones = self._cluster_swings_to_zones(lows, 'support', tolerance, avg_price)

        self._sr_zones = res_zones + sup_zones

        # 按强度排序, 只保留最强的前N个
        self._sr_zones.sort(key=lambda z: (z.strength * z.touches), reverse=True)
        self._sr_zones = self._sr_zones[:self.ZONE_MAX_COUNT]

    def _cluster_swings_to_zones(
        self, swings: List[SwingPoint], kind: str, tolerance: float, avg_price: float
    ) -> List[SRZone]:
        """将相近摆动点聚类为支撑/阻力Zone"""
        zones = []
        used = set()

        for i, sp in enumerate(swings):
            if i in used:
                continue

            cluster = [sp]
            used.add(i)

            # 收集容差范围内的所有支点
            for j in range(i + 1, len(swings)):
                if j in used:
                    continue
                if abs(swings[j].price - sp.price) <= tolerance:
                    cluster.append(swings[j])
                    used.add(j)

            if len(cluster) >= 1:
                prices = [s.price for s in cluster]
                zone_top = max(prices) + tolerance * 0.5
                zone_bot = min(prices) - tolerance * 0.5
                strength = min(1.0, len(cluster) / 3.0)  # 3个以上支点=满强度

                # 检查是否最近被测试过
                recent_test = False
                lookback = min(20, self._n)
                for k in range(self._n - lookback, self._n):
                    if zone_bot <= self.df.iloc[k]['high'] <= zone_top or \
                       zone_bot <= self.df.iloc[k]['low'] <= zone_top:
                        recent_test = True
                        break

                zones.append(SRZone(
                    top=zone_top,
                    bottom=zone_bot,
                    kind=kind,
                    touches=len(cluster),
                    strength=strength,
                    recent_test=recent_test,
                    swing_points=cluster,
                ))

        return zones

    # ═══════════════════════════════════════
    # 第四步: 市场结构分析
    # ═══════════════════════════════════════

    def _analyze_structure(self) -> StructureAnalysis:
        """分析市场结构 — 趋势/通道/区间 + BMS/SMS"""
        highs = [sp for sp in self._swing_points
                 if sp.kind in (SwingKind.HH, SwingKind.LH)]
        lows = [sp for sp in self._swing_points
                if sp.kind in (SwingKind.HL, SwingKind.LL)]

        # ── 趋势判定 ──
        n_recent = min(4, len(highs), len(lows))  # 最近N个支点
        hh_count = sum(1 for sp in highs[-n_recent:] if sp.kind == SwingKind.HH)
        hl_count = sum(1 for sp in lows[-n_recent:] if sp.kind == SwingKind.HL)
        lh_count = sum(1 for sp in highs[-n_recent:] if sp.kind == SwingKind.LH)
        ll_count = sum(1 for sp in lows[-n_recent:] if sp.kind == SwingKind.LL)

        # 趋势K占比
        trend_k_count = sum(1 for ki in self._kline_infos
                            if ki.kind in (KLineKind.TREND_BULL, KLineKind.TREND_BEAR))
        total_k = max(1, len(self._kline_infos))
        trend_k_ratio = trend_k_count / total_k

        # 能量评分 (0-10): 趋势K占比 + 重叠度
        avg_overlap = np.mean([ki.overlap_ratio for ki in self._kline_infos[-20:]])
        energy = int(trend_k_ratio * 6 + (1 - avg_overlap) * 4)
        energy = max(0, min(10, energy))

        # ── 判断市场倾向 ──
        bias = MarketBias.RANGE
        trend_strength = 0.0

        if hh_count >= 2 and hl_count >= 2:
            # 最近低点是否高于前低? (HH+HL序列)
            recent_lows = sorted(lows[-4:], key=lambda x: x.index)
            if len(recent_lows) >= 2 and recent_lows[-1].price > recent_lows[-2].price:
                bias = MarketBias.BULLISH
                trend_strength = min(1.0, (hh_count + hl_count) / 6.0)
        elif ll_count >= 2 and lh_count >= 2:
            recent_highs = sorted(highs[-4:], key=lambda x: x.index)
            if len(recent_highs) >= 2 and recent_highs[-1].price < recent_highs[-2].price:
                bias = MarketBias.BEARISH
                trend_strength = min(1.0, (ll_count + lh_count) / 6.0)

        # 检查通道
        if bias == MarketBias.RANGE and len(highs) >= 3 and len(lows) >= 3:
            # 简单通道检测: 高点和低点是否各自近似在平行线上
            if self._is_channel(highs, lows):
                # 判断通道方向
                if highs[-1].price > highs[-3].price:
                    bias = MarketBias.CHANNEL_UP
                else:
                    bias = MarketBias.CHANNEL_DOWN
                trend_strength = 0.5

        # SMS检测 (结构转换预警)
        bms_signals = self._detect_bms_sms(highs, lows)

        # ── 描述文字 ──
        desc_parts = []
        if bias == MarketBias.BULLISH:
            desc_parts.append(f"多头趋势 (HH+HL序列, 趋势K占比{trend_k_ratio:.0%}, 能量{energy}/10)")
        elif bias == MarketBias.BEARISH:
            desc_parts.append(f"空头趋势 (LL+LH序列, 趋势K占比{trend_k_ratio:.0%}, 能量{energy}/10)")
        elif bias in (MarketBias.CHANNEL_UP, MarketBias.CHANNEL_DOWN):
            desc_parts.append(f"通道行情 (趋势K占比{trend_k_ratio:.0%})")
        else:
            desc_parts.append(f"交易区间/横盘 (趋势K占比{trend_k_ratio:.0%})")

        if self._higher_tf_bias is not None:
            desc_parts.append(f"大周期方向: {self._higher_tf_bias.value}")

        for bms in bms_signals:
            desc_parts.append(f"[{bms.kind}] {bms.description}")

        return StructureAnalysis(
            bias=bias,
            swing_points=self._swing_points,
            sr_zones=self._sr_zones,
            trend_strength=trend_strength,
            trend_energy=energy,
            trend_k_ratio=trend_k_ratio,
            bms_signals=bms_signals,
            description=" | ".join(desc_parts),
        )

    def _is_channel(self, highs: List[SwingPoint], lows: List[SwingPoint]) -> bool:
        """简单通道检测 — 高点和低点各自大致在平行线上"""
        if len(highs) < 3 or len(lows) < 3:
            return False

        recent_highs = highs[-4:]
        recent_lows = lows[-4:]

        h_prices = [sp.price for sp in recent_highs]
        l_prices = [sp.price for sp in recent_lows]

        # 简单判断: 高点之间距离差异不太大, 低点同理
        if len(h_prices) >= 3:
            h_dists = [abs(h_prices[i] - h_prices[i - 1]) for i in range(1, len(h_prices))]
            h_avg = np.mean(h_dists) if h_dists else 0
            h_consistent = all(d <= h_avg * 2 + 0.01 * np.mean(h_prices) for d in h_dists)

            l_dists = [abs(l_prices[i] - l_prices[i - 1]) for i in range(1, len(l_prices))]
            l_avg = np.mean(l_dists) if l_dists else 0
            l_consistent = all(d <= l_avg * 2 + 0.01 * np.mean(l_prices) for d in l_dists)

            # 通道宽度相对稳定
            widths = [h_prices[i] - l_prices[i] for i in range(min(len(h_prices), len(l_prices)))]
            if len(widths) >= 2:
                avg_w = np.mean(widths)
                w_stable = all(abs(w - avg_w) <= avg_w * 0.3 + 0.01 * np.mean(h_prices)
                              for w in widths)
                return h_consistent and l_consistent and w_stable

        return False

    def _detect_bms_sms(
        self, highs: List[SwingPoint], lows: List[SwingPoint]
    ) -> List[BMSSignal]:
        """检测BMS回调结构和SMS反转结构"""
        signals = []
        avg_price = float(self.df.iloc[-1]['close'])

        if len(lows) >= 3:
            recent_lows = sorted(lows, key=lambda x: x.index)

            # BMS做多: 回调不破前低 + 突破回调高点
            if len(recent_lows) >= 2:
                prev_swing_low = recent_lows[-2]
                last_low = recent_lows[-1]

                if last_low.price > prev_swing_low.price:
                    # Higher Low — 可能的BMS回调结构
                    # 找回调期间的高点
                    callback_highs = [h for h in highs
                                      if h.index > prev_swing_low.index
                                      and h.index < self._n - 1]
                    if callback_highs:
                        breakout_level = callback_highs[-1].price
                        if self.df.iloc[-1]['close'] > breakout_level:
                            stop = last_low.price * 0.995
                            target = breakout_level + (breakout_level - last_low.price)
                            rr = abs(target - avg_price) / max(abs(avg_price - stop), 0.0001)
                            signals.append(BMSSignal(
                                kind='BMS_LONG',
                                entry_price=avg_price,
                                stop_loss=stop,
                                take_profit=target,
                                confidence=min(0.9, 0.5 + len(callback_highs) * 0.15),
                                description=f"BMS回调结构: HL不破→突破回调高点 {breakout_level:.4f}"
                            ))

            # SMS预警: 跌破前低
            if len(recent_lows) >= 2:
                prev_low = recent_lows[-2]
                curr_low = recent_lows[-1]
                if curr_low.price < prev_low.price:
                    signals.append(BMSSignal(
                        kind='SMS_WARNING',
                        entry_price=avg_price,
                        stop_loss=0,
                        take_profit=0,
                        confidence=0.4,
                        description=f"⚠️ SMS反转预警: 跌破前低 {prev_low.price:.4f}, 趋势可能反转"
                    ))

        return signals

    # ═══════════════════════════════════════
    # 第五步: 三步法入场信号扫描
    # ═══════════════════════════════════════

    def _scan_signals(self, structure: StructureAnalysis) -> List[TradingSignal]:
        """三步法扫描: ①趋势 ②关键位 ③信号K → 综合评分"""
        signals = []

        for i in range(2, self._n):  # 从第3根K开始(需要前2根确认)
            ki = self._kline_infos[i]

            # 只看信号K
            if not ki.is_signal:
                continue

            # ── 第一步: 判断趋势方向 (1-4分) ──
            trend_score, trade_direction = self._score_trend(structure, i)

            # ── 第二步: 判断是否在关键位 (1-3分) ──
            zone_score, matched_zone = self._score_zone(i, trade_direction)

            # ── 第三步: 2+3信号K评分 (1-3分) ──
            signal_score, signal_details = self._score_signal_k(i, ki, trade_direction, matched_zone)

            total_score = trend_score + zone_score + signal_score

            # 基本条件不满足 → 直接跳过
            if signal_score == 0:
                continue

            # 不及格 → 跳过
            if total_score < self.SCORE_PASS_THRESHOLD:
                continue

            # ── 计算止损止盈 ──
            entry_price = float(self.df.iloc[i]['close'])
            stop_loss, take_profit, rr = self._calc_stop_target(
                i, ki, trade_direction, matched_zone, structure
            )

            # 盈亏比过滤
            if rr < self.MIN_RISK_REWARD:
                continue  # 盈亏比不达标, 直接放弃

            # ── 构建信号 ──
            action = SignalAction.BUY if trade_direction == 'LONG' else SignalAction.SELL
            reasons = [f"趋势评分: {trend_score}/4", f"关键位评分: {zone_score}/3"]
            reasons.extend(signal_details)
            reasons.append(f"综合评分: {total_score}/10")
            reasons.append(f"盈亏比: {rr:.1f}:1")

            signals.append(TradingSignal(
                action=action,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                score_3step=total_score,
                score_2plus3=signal_score,
                confidence=min(0.95, total_score / 10),
                risk_reward=rr,
                reasons=reasons,
                signal_bar_idx=i,
                kline_type=ki.kind,
                sr_zone=matched_zone,
            ))

        # 去重: 同一Zone只保留最高评分的信号
        signals = self._deduplicate_signals(signals)
        signals.sort(key=lambda s: s.score_3step, reverse=True)

        return signals

    def _score_trend(self, structure: StructureAnalysis, idx: int) -> Tuple[int, str]:
        """
        第一步: 趋势方向判断评分
        - 多头趋势 + Pinbar看涨 → 满分
        - 顺大周期方向 → 加分
        """
        score = 0
        ki = self._kline_infos[idx]
        is_bull_signal = ki.kind in (KLineKind.PINBAR_BULL, KLineKind.ENGULFING_BULL, KLineKind.OUTSIDE_BULL)
        is_bear_signal = ki.kind in (KLineKind.PINBAR_BEAR, KLineKind.ENGULFING_BEAR, KLineKind.OUTSIDE_BEAR)

        # 信号方向 vs 市场倾向
        if is_bull_signal and structure.bias == MarketBias.BULLISH:
            score = self.SCORE_TREND_MAX  # 4分: 多头趋势+看涨信号 = 顺势
            direction = 'LONG'
        elif is_bear_signal and structure.bias == MarketBias.BEARISH:
            score = self.SCORE_TREND_MAX
            direction = 'SHORT'
        elif is_bull_signal and structure.bias == MarketBias.CHANNEL_UP:
            score = 3  # 上升通道: 顺势但不如纯趋势
            direction = 'LONG'
        elif is_bear_signal and structure.bias == MarketBias.CHANNEL_DOWN:
            score = 3
            direction = 'SHORT'
        elif is_bull_signal and structure.bias == MarketBias.RANGE:
            score = 2  # 横盘: 低分, 需要更严格的信号确认
            direction = 'LONG'
        elif is_bear_signal and structure.bias == MarketBias.RANGE:
            score = 2
            direction = 'SHORT'
        elif is_bull_signal:
            # 逆势信号 — 需要SMS确认
            has_sms = any(b.kind == 'SMS_WARNING' for b in structure.bms_signals)
            score = 2 if has_sms else 0
            direction = 'LONG'
        elif is_bear_signal:
            has_sms = any(b.kind == 'SMS_WARNING' for b in structure.bms_signals)
            score = 2 if has_sms else 0
            direction = 'SHORT'
        else:
            direction = 'LONG'
            score = 1

        # 大周期验证 (不逆4倍原则)
        if self._higher_tf_bias is not None:
            if direction == 'LONG' and self._higher_tf_bias == MarketBias.BEARISH:
                score = max(0, score - 2)
            if direction == 'SHORT' and self._higher_tf_bias == MarketBias.BULLISH:
                score = max(0, score - 2)

        # 能量加分 (能量≥8且有明显趋势K占比才加分)
        if structure.trend_energy >= 8 and structure.trend_k_ratio >= 0.30:
            score = min(self.SCORE_TREND_MAX, score + 1)

        return score, direction

    def _score_zone(self, idx: int, trade_direction: str) -> Tuple[int, Optional[SRZone]]:
        """
        第二步: 关键位判断评分
        - 信号K的低点/高点精准触及Zone边界 → 高分
        - 只是穿过Zone → 低分
        - 不在任何Zone → 0分
        """
        curr_low = float(self.df.iloc[idx]['low'])
        curr_high = float(self.df.iloc[idx]['high'])
        curr_close = float(self.df.iloc[idx]['close'])
        avg_price = curr_close

        best_zone = None
        best_score = 0

        for zone in self._sr_zones:
            # 判断是否触发Zone
            triggered = False
            proximity = 99.0  # 距离Zone边界的远近 (越小越好)

            if trade_direction == 'LONG' and zone.kind == 'support':
                # 做多: K线低点或收盘价在支撑区附近
                if zone.contains(curr_low) or zone.contains(curr_close):
                    triggered = True
                    # 计算距离Zone下边界的距离
                    proximity = abs(curr_low - zone.bottom) / avg_price
            elif trade_direction == 'SHORT' and zone.kind == 'resistance':
                if zone.contains(curr_high) or zone.contains(curr_close):
                    triggered = True
                    proximity = abs(curr_high - zone.top) / avg_price

            if not triggered:
                continue

            # Zone评分: 基础分(根据距离) + 强度 + 触及次数
            if proximity < 0.002:  # <0.2% = 精准触及
                z_base = 3
            elif proximity < 0.005:  # <0.5% = 接近
                z_base = 2
            else:
                z_base = 1  # >0.5% = 只是穿过

            z_score = z_base + int(zone.strength * 1)  # +0~1
            if zone.touches >= 3:
                z_score += 1

            z_score = min(self.SCORE_ZONE_MAX, z_score)
            if z_score > best_score:
                best_score = z_score
                best_zone = zone

        return best_score, best_zone

    def _score_signal_k(
        self, idx: int, ki: KLineInfo, trade_direction: str, zone: Optional[SRZone]
    ) -> Tuple[int, List[str]]:
        """
        第三步: 2+3信号K评分

        2个基本条件:
          ✅ A. 在关键位Zone内
          ✅ B. 在波段高位/低位

        5个加分项 (≥3项合格):
          ① 信号K在盘面上够明显
          ② 有假突破确认
          ③ 实体被前K包含且幅度>前K
          ④ 顺大级别趋势方向
          ⑤ 盈亏比≥1.5:1
        """
        score = 0
        details = []

        # ── 基本条件A: 在关键位 ──
        in_zone = zone is not None
        if in_zone:
            details.append("✅ 基本A: 在关键位Zone内")
        else:
            # 即使不在Zone, 也可能在波段极值附近
            details.append("⚠️ 基本A: 未在明确关键位 (减分)")

        # ── 基本条件B: 在波段高位/低位 (硬性要求!) ──
        lookback = min(20, idx)
        at_swing_extreme = False
        if lookback > 0:
            prices_high = self.df.iloc[idx - lookback:idx]['high'].values
            prices_low = self.df.iloc[idx - lookback:idx]['low'].values
            curr_h = self.df.iloc[idx]['high']
            curr_l = self.df.iloc[idx]['low']

            is_high = curr_h >= np.percentile(prices_high, 85)
            is_low = curr_l <= np.percentile(prices_low, 15)

            if (trade_direction == 'SHORT' and is_high) or (trade_direction == 'LONG' and is_low):
                at_swing_extreme = True
                details.append("✅ 基本B: 在波段极值区域")
                score += 1
            else:
                details.append("❌ 基本B: 不在波段极值区域 (硬性条件不满足)")

        # ── 硬性门槛: 2个基本条件缺一不可 ──
        if not in_zone or not at_swing_extreme:
            missing = []
            if not in_zone:
                missing.append("关键位")
            if not at_swing_extreme:
                missing.append("波段极值")
            details.append(f"❌ 缺少: {', '.join(missing)} → 信号无效")
            return 0, details

        # ── 加分项 ──
        bonus = 0

        # ① 信号K够明显
        if ki.body_ratio >= 0.40 or ki.kind == KLineKind.PINBAR_BULL or ki.kind == KLineKind.PINBAR_BEAR:
            bonus += 1
            details.append("✅ 加分①: 信号K明显 (实体或影线突出)")

        # ② 假突破确认
        if self._check_fake_breakout(idx, trade_direction):
            bonus += 1
            details.append("✅ 加分②: 假突破确认 (穿刺关键位后拉回)")

        # ③ 实体被前K包含且幅度>前K
        if self._check_body_contained(idx):
            bonus += 1
            details.append("✅ 加分③: 实体被前K包含+幅度>前K")

        # ④ 顺大级别趋势
        if self._higher_tf_bias is not None:
            if (trade_direction == 'LONG' and self._higher_tf_bias in (MarketBias.BULLISH, MarketBias.CHANNEL_UP)) or \
               (trade_direction == 'SHORT' and self._higher_tf_bias in (MarketBias.BEARISH, MarketBias.CHANNEL_DOWN)):
                bonus += 1
                details.append("✅ 加分④: 顺大周期趋势 (不逆4倍)")
            else:
                details.append("⚠️ 加分④: 逆大周期方向 (不加分)")

        # ⑤ 盈亏比预判 (实际盈亏比在后续计算, 这里先预估)
        # 在Zone内做多: 上方阻力 = 目标, 下方支撑 = 止损
        if in_zone:
            bonus += 1
            details.append("✅ 加分⑤: 关键位提供了明确的止损止盈参考")
        else:
            details.append("⚠️ 加分⑤: 无关键位参考, 盈亏比不确定")

        final_score = min(self.SCORE_SIGNAL_MAX, 1 + bonus)  # 基础1分 + 加分
        return final_score, details

    def _check_fake_breakout(self, idx: int, trade_direction: str) -> bool:
        """检查假突破: 穿刺关键位后快速收回"""
        if idx < 2 or not self._sr_zones:
            return False

        for zone in self._sr_zones:
            if (trade_direction == 'LONG' and zone.kind == 'support') or \
               (trade_direction == 'SHORT' and zone.kind == 'resistance'):

                # 前1-2根K线是否穿刺了Zone
                for lookback in range(1, min(3, idx)):
                    prev_low = self.df.iloc[idx - lookback]['low']
                    prev_high = self.df.iloc[idx - lookback]['high']

                    if trade_direction == 'LONG':
                        # 做多: 前K穿刺支撑区下方 → 收回 → 假突破
                        if prev_low < zone.bottom and self.df.iloc[idx]['close'] > zone.bottom:
                            return True
                    else:
                        # 做空: 前K穿刺阻力区上方 → 收回 → 假突破
                        if prev_high > zone.top and self.df.iloc[idx]['close'] < zone.top:
                            return True

        return False

    def _check_body_contained(self, idx: int) -> bool:
        """检查实体是否被前K包含且当前K幅度>前K"""
        if idx < 1:
            return False

        prev_h, prev_l = self.df.iloc[idx - 1]['high'], self.df.iloc[idx - 1]['low']
        curr_o, curr_c = self.df.iloc[idx]['open'], self.df.iloc[idx]['close']
        curr_body_top = max(curr_o, curr_c)
        curr_body_bot = min(curr_o, curr_c)
        curr_range = self._range[idx]
        prev_range = self._range[idx - 1]

        return (prev_l <= curr_body_bot and curr_body_top <= prev_h
                and curr_range > prev_range)

    # ═══════════════════════════════════════
    # 🆕 2B结构 & SB结构检测
    # ═══════════════════════════════════════

    def _scan_2b_sb_signals(self, structure: StructureAnalysis) -> List[TradingSignal]:
        """扫描2B反转结构和SB二次突破结构的入场信号"""
        signals = []
        if self._n < 6:
            return signals

        # 2B结构: 价格突破前高(低)后立即被拉回 → 假突破反转
        for i in range(3, self._n - 1):
            highs = [float(self.df.iloc[j]['high']) for j in range(i-3, i+1)]
            lows = [float(self.df.iloc[j]['low']) for j in range(i-3, i+1)]
            closes = [float(self.df.iloc[j]['close']) for j in range(i-3, i+1)]

            # 2B顶: 突破前高→被拉回收阴
            if highs[-2] > highs[-3] and closes[-1] < highs[-3]:
                # 突破后立即回落
                entry = closes[-1]
                stop = highs[-2] * 1.005
                target = entry - (stop - entry) * 2.0
                if target < entry * 0.9:  # RR >= 2
                    signals.append(TradingSignal(
                        action=SignalAction.SELL,
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        score_3step=8,
                        score_2plus3=5,
                        confidence=0.75,
                        risk_reward=2.0,
                        reasons=["2B顶反转: 突破前高后立即被拉回收阴", "假突破陷阱"],
                        signal_bar_idx=i,
                        kline_type=KLineKind.PINBAR_BEAR,
                    ))

            # 2B底: 跌破前低→被拉回收阳
            if lows[-2] < lows[-3] and closes[-1] > lows[-3]:
                entry = closes[-1]
                stop = lows[-2] * 0.995
                target = entry + (entry - stop) * 2.0
                if target > entry * 1.01:
                    signals.append(TradingSignal(
                        action=SignalAction.BUY,
                        entry_price=entry,
                        stop_loss=stop,
                        take_profit=target,
                        score_3step=8,
                        score_2plus3=5,
                        confidence=0.75,
                        risk_reward=2.0,
                        reasons=["2B底反转: 跌破前低后立即被拉回收阳", "假突破陷阱"],
                        signal_bar_idx=i,
                        kline_type=KLineKind.PINBAR_BULL,
                    ))

        # SB结构 (二次突破): 高1+低1→回调→高2+低2 顺势
        if structure.bias in (MarketBias.BULLISH, MarketBias.BEARISH):
            swing_points = structure.swing_points
            if len(swing_points) >= 4:
                recent = swing_points[-4:]
                # 看涨SB: 两个逐步抬高的低点(HL) + 其间有小回调
                if structure.bias == MarketBias.BULLISH:
                    hl_points = [sp for sp in recent if sp.kind == SwingKind.HL]
                    if len(hl_points) >= 2:
                        h1, h2 = hl_points[-2], hl_points[-1]
                        bars_between = abs(h2.index - h1.index)
                        if 2 <= bars_between <= 8 and h2.price > h1.price:
                            # SB做多信号
                            entry = float(self.df.iloc[-1]['close'])
                            stop = h2.price * 0.995
                            target = entry + (entry - stop) * 2.5
                            signals.append(TradingSignal(
                                action=SignalAction.BUY,
                                entry_price=entry,
                                stop_loss=stop,
                                take_profit=target,
                                score_3step=9,
                                score_2plus3=5,
                                confidence=0.80,
                                risk_reward=2.5,
                                reasons=[f"SB结构做多: HL#1→HL#2({bars_between}根K线)", "高胜率二次突破"],
                                signal_bar_idx=h2.index,
                                kline_type=KLineKind.TREND_BULL,
                            ))

                # 看跌SB
                else:
                    lh_points = [sp for sp in recent if sp.kind == SwingKind.LH]
                    if len(lh_points) >= 2:
                        l1, l2 = lh_points[-2], lh_points[-1]
                        bars_between = abs(l2.index - l1.index)
                        if 2 <= bars_between <= 8 and l2.price < l1.price:
                            entry = float(self.df.iloc[-1]['close'])
                            stop = l2.price * 1.005
                            target = entry - (stop - entry) * 2.5
                            signals.append(TradingSignal(
                                action=SignalAction.SELL,
                                entry_price=entry,
                                stop_loss=stop,
                                take_profit=target,
                                score_3step=9,
                                score_2plus3=5,
                                confidence=0.80,
                                risk_reward=2.5,
                                reasons=[f"SB结构做空: LH#1→LH#2({bars_between}根K线)", "高胜率二次突破"],
                                signal_bar_idx=l2.index,
                                kline_type=KLineKind.TREND_BEAR,
                            ))

        return signals

    # ═══════════════════════════════════════
    # 🆕 复杂回调检测 (腿1≈腿2)
    # ═══════════════════════════════════════

    def _detect_complex_pullback(self, structure: StructureAnalysis) -> List[ComplexPullback]:
        """检测复杂回调: 两段式ABC调整, 腿1≈腿2测量目标"""
        results = []
        if self._n < 20 or len(structure.swing_points) < 4:
            return results

        closes = self.df['close'].values.astype(float)

        # 找最近的两段式回调
        swing_points = structure.swing_points
        for i in range(len(swing_points) - 3):
            s1, s2, s3, s4 = swing_points[i:i+4]

            # 看涨复杂回调: HH→HL→LH→LL 或 上升趋势中的两段回调
            if s1.kind == SwingKind.HH and s3.kind == SwingKind.HH:
                leg1 = abs(s1.price - s2.price) if s2.kind in (SwingKind.HL, SwingKind.LL) else 0
                leg2 = abs(s3.price - s4.price) if s4.kind in (SwingKind.HL, SwingKind.LL) else 0
                if leg1 > 0 and leg2 > 0:
                    ratio = leg2 / leg1 if leg1 > 0 else 0
                    if 0.5 <= ratio <= 2.0:  # 腿1≈腿2
                        target = float(closes[-1]) + leg1  # 测量目标
                        results.append(ComplexPullback(
                            start_idx=s1.index, end_idx=s4.index,
                            leg1_length=leg1, leg2_length=leg2,
                            ratio=ratio, target_price=target,
                            direction='bullish',
                            confidence=min(1.0, 1.0 - abs(1.0 - ratio)),
                        ))

            # 看跌复杂回调
            if s1.kind == SwingKind.LL and s3.kind == SwingKind.LL:
                leg1 = abs(s1.price - s2.price) if s2.kind in (SwingKind.LH, SwingKind.HH) else 0
                leg2 = abs(s3.price - s4.price) if s4.kind in (SwingKind.LH, SwingKind.HH) else 0
                if leg1 > 0 and leg2 > 0:
                    ratio = leg2 / leg1 if leg1 > 0 else 0
                    if 0.5 <= ratio <= 2.0:
                        target = float(closes[-1]) - leg1
                        results.append(ComplexPullback(
                            start_idx=s1.index, end_idx=s4.index,
                            leg1_length=leg1, leg2_length=leg2,
                            ratio=ratio, target_price=target,
                            direction='bearish',
                            confidence=min(1.0, 1.0 - abs(1.0 - ratio)),
                        ))

        return results

    # ═══════════════════════════════════════
    # 🆕 FVG (Fair Value Gap) 检测
    # ═══════════════════════════════════════

    def _detect_fvg(self) -> List[FVGap]:
        """检测SMC Fair Value Gap (合理价值缺口)

        BISI: 第1根K高点 > 第3根K低点 → 买方不平衡缺口
        SIBI: 第1根K低点 < 第3根K高点 → 卖方不平衡缺口
        """
        fvgs = []
        if self._n < 3:
            return fvgs

        for i in range(1, self._n - 1):
            bar0_high = float(self.df.iloc[i-1]['high'])
            bar0_low = float(self.df.iloc[i-1]['low'])
            bar1_high = float(self.df.iloc[i]['high'])    # 中间K (形成FVG的K)
            bar1_low = float(self.df.iloc[i]['low'])
            bar2_high = float(self.df.iloc[i+1]['high'])
            bar2_low = float(self.df.iloc[i+1]['low'])

            # BISI: 看涨FVG (第1根高 > 第3根低, 之间有空隙)
            if bar0_low > bar2_high:
                gap_top = bar0_low
                gap_bottom = bar2_high
                gap_size = (gap_top - gap_bottom) / gap_bottom * 100
                if gap_size > 0.05:  # 至少0.05%
                    # 检查是否已被回补
                    filled = False
                    age = self._n - 1 - i
                    for j in range(i+2, self._n):
                        if float(self.df.iloc[j]['low']) <= gap_bottom:
                            filled = True
                            break
                    fvgs.append(FVGap(
                        index=i, kind='BISI',
                        gap_top=gap_top, gap_bottom=gap_bottom,
                        gap_size_pct=gap_size, filled=filled, age_bars=age,
                    ))

            # SIBI: 看跌FVG
            if bar0_high < bar2_low:
                gap_top = bar2_low
                gap_bottom = bar0_high
                gap_size = (gap_top - gap_bottom) / gap_bottom * 100
                if gap_size > 0.05:
                    filled = False
                    age = self._n - 1 - i
                    for j in range(i+2, self._n):
                        if float(self.df.iloc[j]['high']) >= gap_top:
                            filled = True
                            break
                    fvgs.append(FVGap(
                        index=i, kind='SIBI',
                        gap_top=gap_top, gap_bottom=gap_bottom,
                        gap_size_pct=gap_size, filled=filled, age_bars=age,
                    ))

        return fvgs

    # ═══════════════════════════════════════
    # 🆕 Order Block 检测
    # ═══════════════════════════════════════

    def _detect_order_blocks(self) -> List[Dict]:
        """检测Order Block: 大阳线/大阴线启动前的那根反向K线

        推进块: 趋势中推动方向的OB
        拒绝块: 关键位附近被拒绝的OB
        """
        obs = []
        if self._n < 5:
            return obs

        closes = self.df['close'].values.astype(float)
        opens = self.df['open'].values.astype(float)
        highs = self.df['high'].values.astype(float)
        lows = self.df['low'].values.astype(float)
        avg_range = np.mean(self._range[-20:]) if self._n >= 20 else np.mean(self._range)

        for i in range(2, self._n - 1):
            # 看涨OB: 一根大阴线后出现大阳线, OB=那根大阴线
            prev_range = self._range[i-1]
            curr_range = self._range[i]
            prev_body = abs(closes[i-1] - opens[i-1])
            curr_body = abs(closes[i] - opens[i])

            if (prev_body > avg_range * 1.2 and closes[i-1] < opens[i-1] and  # 前一根大阴线
                curr_body > avg_range * 1.2 and closes[i] > opens[i] and       # 当前大阳线
                closes[i] > highs[i-1]):                                       # 覆盖前K
                obs.append({
                    "index": i-1,
                    "kind": "bullish_ob",
                    "price_top": highs[i-1],
                    "price_bottom": lows[i-1],
                    "mid": round((highs[i-1] + lows[i-1]) / 2, 6),
                    "strength": "strong" if curr_body > avg_range * 2 else "normal",
                })

            # 看跌OB: 一根大阳线后出现大阴线, OB=那根大阳线
            if (prev_body > avg_range * 1.2 and closes[i-1] > opens[i-1] and
                curr_body > avg_range * 1.2 and closes[i] < opens[i] and
                closes[i] < lows[i-1]):
                obs.append({
                    "index": i-1,
                    "kind": "bearish_ob",
                    "price_top": highs[i-1],
                    "price_bottom": lows[i-1],
                    "mid": round((highs[i-1] + lows[i-1]) / 2, 6),
                    "strength": "strong" if curr_body > avg_range * 2 else "normal",
                })

        return obs

    # ═══════════════════════════════════════
    # 🆕 通道三结局预测
    # ═══════════════════════════════════════

    def _predict_channel_outcome(self, structure: StructureAnalysis) -> Optional[str]:
        """预测通道的三种结局: 加速突破/向下破位/演变为区间

        判断依据:
        - 通道角度变化 (变陡→加速, 变平→区间)
        - 最近触及边界的反应强度
        - 是否出现端点旗形
        """
        if structure.bias not in (MarketBias.CHANNEL_UP, MarketBias.CHANNEL_DOWN):
            return None

        swing_points = structure.swing_points
        if len(swing_points) < 6:
            return None

        closes = self.df['close'].values.astype(float)
        recent_close = closes[-1]

        # 取最近的极点计算通道斜率
        highs = sorted([sp for sp in swing_points if sp.kind in (SwingKind.HH, SwingKind.LH)],
                       key=lambda x: x.index)
        lows = sorted([sp for sp in swing_points if sp.kind in (SwingKind.HL, SwingKind.LL)],
                      key=lambda x: x.index)

        if len(highs) < 3 or len(lows) < 3:
            return None

        # 通道中线位置和宽度
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        # 趋势K占比 — 如果趋势K越来越少→可能变区间
        recent_klines = self._kline_infos[-10:]
        trend_count = sum(1 for k in recent_klines
                         if k.kind in (KLineKind.TREND_BULL, KLineKind.TREND_BEAR))
        trend_ratio = trend_count / len(recent_klines) if recent_klines else 0

        # 判断通道结局
        if trend_ratio >= 0.5:
            # 趋势K占比高 → 可能在加速
            # 检查是否在通道上轨附近
            if structure.bias == MarketBias.CHANNEL_UP:
                upper_bound = max(h.price for h in recent_highs)
                if recent_close > upper_bound * 0.995:
                    return "accelerate_breakout_up"  # 即将加速突破
                return "continue_channel_up"
            else:
                lower_bound = min(l.price for l in recent_lows)
                if recent_close < lower_bound * 1.005:
                    return "accelerate_breakout_down"
                return "continue_channel_down"

        elif trend_ratio <= 0.2:
            # 趋势K占比很低 → 通道在变平, 可能转为区间
            return "flattening_to_range"

        else:
            # 中性: K线越来越小 → 末端旗形 → 可能反转
            recent_ranges = self._range[-5:]
            if len(recent_ranges) >= 3:
                if recent_ranges[-1] < np.mean(recent_ranges[:-1]) * 0.7:
                    return "terminal_flag_wedge"  # 末端旗形/楔形反转预警

            return "normal_channel"

    # ═══════════════════════════════════════
    # 🆕 末端旗形 & 楔形反转检测
    # ═══════════════════════════════════════

    def _detect_terminal_flag_wedge(self) -> List[Dict]:
        """检测末端旗形和楔形反转形态

        末端旗形: 趋势末尾出现的紧凑旗形整理, 幅度越来越小
        楔形反转: 三推楔形, 每次推动幅度递减
        """
        results = []
        if self._n < 15:
            return results

        closes = self.df['close'].values.astype(float)
        highs = self.df['high'].values.astype(float)
        lows = self.df['low'].values.astype(float)

        # 检测最近10-15根K线是否形成收敛形态
        window = min(15, self._n - 1)
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]

        # 高点越来越低 + 低点越来越高 = 收敛三角形/楔形
        h_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        l_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

        if h_trend < 0 and l_trend > 0:
            # 收敛!
            # 测量收敛幅度
            first_range = recent_highs[0] - recent_lows[0]
            last_range = recent_highs[-1] - recent_lows[-1]
            compression = last_range / first_range if first_range > 0 else 1.0

            if compression < 0.5:  # 幅度压缩超过50%
                # 判断是末端旗形还是楔形
                # 看之前是否有明显趋势
                prev_closes = closes[-window*2:-window]
                if len(prev_closes) > 5:
                    prev_trend = np.polyfit(range(len(prev_closes)), prev_closes, 1)[0]
                    wedge_width = recent_highs[0] - recent_lows[0]

                    results.append({
                        "type": "terminal_flag" if compression < 0.3 else "wedge",
                        "compression_ratio": round(compression, 2),
                        "prev_trend": "up" if prev_trend > 0 else "down",
                        "wedge_width": round(wedge_width, 4),
                        "breakout_target": round(
                            closes[-1] + wedge_width if prev_trend > 0
                            else closes[-1] - wedge_width, 4
                        ),
                        "signal": "Trend exhaustion — prepare for reversal",
                        "confidence": round(min(0.9, 1.0 - compression), 2),
                    })

        return results

    # ═══════════════════════════════════════
    # 🆕 斐波那契扩张目标计算 (公用方法)
    # ═══════════════════════════════════════

    def calc_fib_expansion_targets(self, swing_low: float, swing_high: float,
                                     direction: str = 'long') -> Dict[str, float]:
        """计算斐波那契扩张目标位

        用于测量趋势空间的扩展目标:
          -0.618: 回调目标
           1.000: 腿1=腿2 等距目标
           1.618: 大级别扩张
           2.618: 超大级别扩张
        """
        base_range = abs(swing_high - swing_low)

        if direction == 'long':
            return {
                '-0.618': round(swing_high - base_range * 0.618, 4),
                '1.000': round(swing_low + base_range * 1.000, 4),
                '1.618': round(swing_low + base_range * 1.618, 4),
                '2.618': round(swing_low + base_range * 2.618, 4),
            }
        else:
            return {
                '-0.618': round(swing_low + base_range * 0.618, 4),
                '1.000': round(swing_high - base_range * 1.000, 4),
                '1.618': round(swing_high - base_range * 1.618, 4),
                '2.618': round(swing_high - base_range * 2.618, 4),
            }

    def _calc_stop_target(
        self, idx: int, ki: KLineInfo, trade_direction: str,
        zone: Optional[SRZone], structure: StructureAnalysis
    ) -> Tuple[float, float, float]:
        """计算止损、止盈、盈亏比 — 优先用Zone, 同时保证最小RR"""
        entry_price = float(self.df.iloc[idx]['close'])
        bar_low = float(self.df.iloc[idx]['low'])
        bar_high = float(self.df.iloc[idx]['high'])
        atr = float(np.mean(self._range[-14:])) if self._n >= 14 else entry_price * 0.008
        min_target_atr = atr * 2.0  # 目标至少2倍ATR

        if trade_direction == 'LONG':
            # 止损: 信号K低点下方 - 0.5ATR 或 Zone底部下方
            if zone and zone.kind == 'support':
                stop = min(bar_low, zone.bottom) - atr * 0.5
            else:
                stop = bar_low - atr * 0.5
            stop = min(stop, entry_price - atr * 0.5)  # 止损必须低于入场价

            # 目标: 找上方≥2ATR的阻力Zone, 否则用ATR目标
            target = entry_price + min_target_atr  # 默认2ATR
            for z in sorted(self._sr_zones, key=lambda z: z.bottom):
                if z.kind == 'resistance' and z.bottom > entry_price + min_target_atr:
                    target = z.bottom
                    break

        else:
            if zone and zone.kind == 'resistance':
                stop = max(bar_high, zone.top) + atr * 0.5
            else:
                stop = bar_high + atr * 0.5
            stop = max(stop, entry_price + atr * 0.5)

            target = entry_price - min_target_atr
            for z in sorted(self._sr_zones, key=lambda z: z.top, reverse=True):
                if z.kind == 'support' and z.top < entry_price - min_target_atr:
                    target = z.top
                    break

        risk = abs(entry_price - stop)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0

        return round(stop, 6), round(target, 6), round(rr, 2)

    # ═══════════════════════════════════════
    # 🆕 Low1/Low2/High1/High2 入场计数 (Vision识图: 熊猫教练核心)
    # ═══════════════════════════════════════

    def _detect_low_high_sequence(self, structure: StructureAnalysis) -> List[LowHighCount]:
        """检测 Low1/Low2 (做多) 和 High1/High2 (做空) 序列

        熊猫教练Vision识图核心:
          - Low 1: 上升趋势中第一次回调低点 → "打底", 胜率较低
          - Low 2: 上升趋势中第二次回调低点 → "出击", 胜率更高
          - "有效信号K +入场K: 低2" → 信号K+入场K+低2 = 最高胜率组合
          - "信号K = 非趋势K线" (逆主趋势的小K线, 表示方向犹豫)
          - "入场K = 趋势K线" (顺主趋势方向, 确认出击)

        算法:
          1. 在上升趋势中找 pullback lows → 标为 Low1, Low2
          2. 在下降趋势中找 rally highs → 标为 High1, High2
          3. 检查每个 Low/High 是否有信号K+入场K确认
        """
        results = []
        if self._n < 10 or len(structure.swing_points) < 3:
            return results

        closes = self.df['close'].values.astype(float)
        highs_arr = self.df['high'].values.astype(float)
        lows_arr = self.df['low'].values.astype(float)
        avg_price = closes[-1]

        # ── 上升趋势中找 Low1/Low2 ──
        if structure.bias in (MarketBias.BULLISH, MarketBias.CHANNEL_UP):
            swing_lows = sorted(
                [sp for sp in structure.swing_points
                 if sp.kind in (SwingKind.HL, SwingKind.LL) and sp.index > self._n - 50],
                key=lambda x: x.index
            )

            for i, sl in enumerate(swing_lows[-6:]):  # 最近6个低点
                # 是否在关键支撑位附近?
                near_key_level = False
                for zone in self._sr_zones:
                    if zone.kind == 'support' and zone.contains(sl.price):
                        near_key_level = True
                        break

                # 检查信号K+入场K
                sig_entry = self._find_signal_entry_at_swing(
                    sl.index, 'LONG', sl.price
                )

                # 判断是Low1还是Low2
                low_count = 1 if i % 2 == 0 else 2  # 简化为交替
                # 精确判断: 与前一个低点比较
                if i > 0:
                    prev_sl = swing_lows[-6:][i-1]
                    low_count = 2 if sl.price > prev_sl.price else 1

                lh = LowHighCount(
                    index=sl.index,
                    count=low_count,
                    price=sl.price,
                    kind=f'low{low_count}',
                    signal_k_idx=sig_entry[0] if sig_entry else -1,
                    entry_k_idx=sig_entry[1] if sig_entry else -1,
                    quality=0.7 if near_key_level else 0.4,
                    confirmed=sig_entry is not None,
                )

                # 低2 + 关键位 + 信号K确认 = 最高质量
                if lh.count == 2 and near_key_level and lh.confirmed:
                    lh.quality = 0.95
                elif lh.count == 2 and lh.confirmed:
                    lh.quality = 0.85

                results.append(lh)

        # ── 下降趋势中找 High1/High2 ──
        elif structure.bias in (MarketBias.BEARISH, MarketBias.CHANNEL_DOWN):
            swing_highs = sorted(
                [sp for sp in structure.swing_points
                 if sp.kind in (SwingKind.LH, SwingKind.HH) and sp.index > self._n - 50],
                key=lambda x: x.index
            )

            for i, sh in enumerate(swing_highs[-6:]):
                near_key_level = False
                for zone in self._sr_zones:
                    if zone.kind == 'resistance' and zone.contains(sh.price):
                        near_key_level = True
                        break

                sig_entry = self._find_signal_entry_at_swing(
                    sh.index, 'SHORT', sh.price
                )

                high_count = 1 if i % 2 == 0 else 2
                if i > 0:
                    prev_sh = swing_highs[-6:][i-1]
                    high_count = 2 if sh.price < prev_sh.price else 1

                lh = LowHighCount(
                    index=sh.index,
                    count=high_count,
                    price=sh.price,
                    kind=f'high{high_count}',
                    signal_k_idx=sig_entry[0] if sig_entry else -1,
                    entry_k_idx=sig_entry[1] if sig_entry else -1,
                    quality=0.7 if near_key_level else 0.4,
                    confirmed=sig_entry is not None,
                )

                if lh.count == 2 and near_key_level and lh.confirmed:
                    lh.quality = 0.95
                elif lh.count == 2 and lh.confirmed:
                    lh.quality = 0.85

                results.append(lh)

        return results

    def _find_signal_entry_at_swing(
        self, swing_idx: int, direction: str, swing_price: float
    ) -> Optional[Tuple[int, int]]:
        """在摆动点附近找信号K+入场K两步确认

        信号K = 非趋势K线 (表示多空犹豫, 原趋势暂停)
        入场K = 趋势K线 (确认方向, 重新启动)

        Returns: (signal_k_idx, entry_k_idx) or None
        """
        if swing_idx >= self._n - 2:
            return None

        closes = self.df['close'].values.astype(float)

        # 在摆动点后1-5根K线内寻找
        for look in range(1, min(6, self._n - swing_idx - 1)):
            si = swing_idx + look
            ki = self._kline_infos[si] if si < len(self._kline_infos) else None
            if ki is None:
                continue

            # 信号K: 非趋势K线 (NON_TREND / DOJI / INSIDE)
            is_signal_k = ki.kind in (
                KLineKind.NON_TREND, KLineKind.DOJI, KLineKind.INSIDE,
                KLineKind.PINBAR_BULL, KLineKind.PINBAR_BEAR
            )

            if not is_signal_k:
                continue

            # 确认信号K方向正确
            if direction == 'LONG' and ki.kind == KLineKind.PINBAR_BULL:
                pass  # 看涨Pinbar = 优秀信号K
            elif direction == 'SHORT' and ki.kind == KLineKind.PINBAR_BEAR:
                pass  # 看跌Pinbar = 优秀信号K
            elif ki.kind in (KLineKind.NON_TREND, KLineKind.DOJI, KLineKind.INSIDE):
                pass  # 非趋势K = 可接受信号K
            else:
                continue

            # 入场K: 在信号K后1-3根内找趋势K
            for entry_look in range(1, min(4, self._n - si - 1)):
                ei = si + entry_look
                eki = self._kline_infos[ei] if ei < len(self._kline_infos) else None
                if eki is None:
                    continue

                if direction == 'LONG':
                    is_entry_k = eki.kind in (
                        KLineKind.TREND_BULL, KLineKind.ENGULFING_BULL,
                        KLineKind.OUTSIDE_BULL
                    )
                    if is_entry_k and closes[ei] > closes[si]:
                        return (si, ei)
                else:
                    is_entry_k = eki.kind in (
                        KLineKind.TREND_BEAR, KLineKind.ENGULFING_BEAR,
                        KLineKind.OUTSIDE_BEAR
                    )
                    if is_entry_k and closes[ei] < closes[si]:
                        return (si, ei)

        return None

    # ═══════════════════════════════════════
    # 🆕 信号K+入场K两步确认对 (Vision识图)
    # ═══════════════════════════════════════

    def _detect_signal_entry_pairs(self, structure: StructureAnalysis) -> List[SignalEntryPair]:
        """检测所有信号K+入场K两步确认对

        "不需要数入场K线" — 只看第1根确认的顺趋势K
        信号K = 非趋势K线 (不能是趋势K!)
        入场K = 趋势K线 (顺主趋势方向的第一根)
        """
        pairs = []
        if self._n < 5:
            return pairs

        closes = self.df['close'].values.astype(float)
        avg_price = closes[-1]

        for i in range(1, self._n - 3):
            ki = self._kline_infos[i]

            # 信号K必须是 非趋势K / 十字星 / 内包K / Pinbar
            is_signal = ki.kind in (
                KLineKind.NON_TREND, KLineKind.DOJI, KLineKind.INSIDE,
                KLineKind.PINBAR_BULL, KLineKind.PINBAR_BEAR
            )
            if not is_signal:
                continue

            # 检查关键位
            at_key_level = False
            for zone in self._sr_zones:
                if zone.contains(self.df.iloc[i]['low']) or zone.contains(self.df.iloc[i]['high']):
                    at_key_level = True
                    break

            # 找接下来的入场K (1-3根内)
            for j in range(i + 1, min(i + 4, self._n)):
                ek = self._kline_infos[j]

                # ── 做多信号对 ──
                if ki.kind in (KLineKind.PINBAR_BULL, KLineKind.NON_TREND, KLineKind.DOJI, KLineKind.INSIDE):
                    if ek.kind in (KLineKind.TREND_BULL, KLineKind.ENGULFING_BULL, KLineKind.OUTSIDE_BULL):
                        if closes[j] > closes[i]:  # 入场K收盘高于信号K
                            # 判断是Low1还是Low2
                            lh_count = self._classify_low_high_count(i, 'LONG', structure)

                            quality = 0.5
                            if at_key_level:
                                quality += 0.2
                            if ki.kind == KLineKind.PINBAR_BULL:
                                quality += 0.15  # Pinbar信号K加分
                            if lh_count == 'low2':
                                quality += 0.15  # Low2加分

                            pairs.append(SignalEntryPair(
                                signal_idx=i, entry_idx=j,
                                direction='LONG',
                                signal_k_kind=ki.kind.value,
                                entry_k_kind=ek.kind.value,
                                at_key_level=at_key_level,
                                low_high_count=lh_count,
                                quality=min(0.95, quality),
                            ))
                            break  # 找到第一根入场K就够了

                # ── 做空信号对 ──
                elif ki.kind in (KLineKind.PINBAR_BEAR, KLineKind.NON_TREND, KLineKind.DOJI, KLineKind.INSIDE):
                    if ek.kind in (KLineKind.TREND_BEAR, KLineKind.ENGULFING_BEAR, KLineKind.OUTSIDE_BEAR):
                        if closes[j] < closes[i]:
                            lh_count = self._classify_low_high_count(i, 'SHORT', structure)

                            quality = 0.5
                            if at_key_level:
                                quality += 0.2
                            if ki.kind == KLineKind.PINBAR_BEAR:
                                quality += 0.15
                            if lh_count == 'high2':
                                quality += 0.15

                            pairs.append(SignalEntryPair(
                                signal_idx=i, entry_idx=j,
                                direction='SHORT',
                                signal_k_kind=ki.kind.value,
                                entry_k_kind=ek.kind.value,
                                at_key_level=at_key_level,
                                low_high_count=lh_count,
                                quality=min(0.95, quality),
                            ))
                            break

        # 按质量排序, 只保留最近的高质量对
        pairs.sort(key=lambda p: p.quality, reverse=True)
        return pairs[:10]

    def _classify_low_high_count(self, idx: int, direction: str,
                                  structure: StructureAnalysis) -> str:
        """判断当前信号K处于Low1/Low2或High1/High2"""
        swing_points = structure.swing_points
        if len(swing_points) < 2:
            return ''

        # 找idx之前的最近摆动点
        if direction == 'LONG':
            prior_lows = [sp for sp in swing_points
                         if sp.kind in (SwingKind.HL, SwingKind.LL) and sp.index < idx]
            if len(prior_lows) >= 2:
                return 'low2'
            elif len(prior_lows) >= 1:
                return 'low1'
        else:
            prior_highs = [sp for sp in swing_points
                          if sp.kind in (SwingKind.HH, SwingKind.LH) and sp.index < idx]
            if len(prior_highs) >= 2:
                return 'high2'
            elif len(prior_highs) >= 1:
                return 'high1'

        return ''

    # ═══════════════════════════════════════
    # 🆕 强动能追单警告 (Vision识图: "不要在强动能K线追单")
    # ═══════════════════════════════════════

    def _detect_momentum_chase(self, structure: StructureAnalysis) -> List[Dict]:
        """检测强动能K线追单风险

        熊猫教练原话: "不要在强动能K线追单"
        - 大实体趋势K线之后立即入场 = 追单
        - 应该等回调到关键位再入场
        - 追单的风险: 入场后立即被回调止损

        检测条件:
          1. 最近1-2根K线实体 >= 平均实体的2倍 (强动能)
          2. 价格远离关键位 (>1 ATR)
          3. 没有信号K确认
        """
        warnings = []
        if self._n < 5:
            return warnings

        closes = self.df['close'].values.astype(float)
        avg_body = float(np.mean(self._body[-20:])) if self._n >= 20 else float(np.mean(self._body))
        if avg_body <= 0:
            return warnings

        # 检查最近2根K线
        for lookback in range(1, min(4, self._n)):
            idx = self._n - lookback
            ki = self._kline_infos[idx] if idx < len(self._kline_infos) else None
            if ki is None:
                continue

            # 是否强动能?
            body_r = self._body[idx] / avg_body if avg_body > 0 else 0
            is_strong_momentum = (
                body_r >= 2.0 and
                ki.kind in (KLineKind.TREND_BULL, KLineKind.TREND_BEAR,
                           KLineKind.OUTSIDE_BULL, KLineKind.OUTSIDE_BEAR)
            )

            if not is_strong_momentum:
                continue

            # 距离关键位多远?
            nearest_zone_dist = 999.0
            nearest_zone_kind = ''
            for zone in self._sr_zones:
                if ki.kind == KLineKind.TREND_BULL:
                    dist = (closes[idx] - zone.bottom) / closes[idx]
                    if zone.kind == 'resistance' and dist < nearest_zone_dist:
                        nearest_zone_dist = dist
                        nearest_zone_kind = 'resistance'
                elif ki.kind == KLineKind.TREND_BEAR:
                    dist = (zone.top - closes[idx]) / closes[idx]
                    if zone.kind == 'support' and dist < nearest_zone_dist:
                        nearest_zone_dist = dist
                        nearest_zone_kind = 'support'

            atr = float(np.mean(self._range[-14:]))
            far_from_zone = nearest_zone_dist > (atr / closes[idx]) if atr > 0 else True

            if far_from_zone:
                warnings.append({
                    "bar_idx": int(idx),
                    "warning": "强动能追单风险!",
                    "kline_type": ki.kind.value,
                    "body_vs_avg": round(body_r, 1),
                    "distance_to_zone_pct": round(nearest_zone_dist * 100, 2),
                    "advice": "不要在强动能K线追单 — 等回调到关键位+信号K确认后再入场",
                })

        return warnings

    def _deduplicate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """去重: 相邻信号只保留评分最高的"""
        if len(signals) <= 1:
            return signals

        kept = []
        used_indices = set()

        for sig in sorted(signals, key=lambda s: s.score_3step, reverse=True):
            # 检查是否有索引接近的更高分信号
            conflict = False
            for used_idx in used_indices:
                if abs(sig.signal_bar_idx - used_idx) <= 3:  # 3根K线内不重复
                    conflict = True
                    break

            if not conflict:
                kept.append(sig)
                used_indices.add(sig.signal_bar_idx)

        return kept


# ═══════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════

def scan_symbol(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
    higher_tf_df: Optional[pd.DataFrame] = None,
) -> ScanResult:
    """扫描单个交易对的便捷函数"""
    scanner = NakedKScanner(df, symbol=symbol, timeframe=timeframe,
                            higher_tf_df=higher_tf_df)
    return scanner.scan()


def scan_multi_timeframe(
    dfs: Dict[str, pd.DataFrame],
    symbol: str = "UNKNOWN",
) -> Dict[str, ScanResult]:
    """多周期扫描 — 从大到小, 大周期结果传入小周期做趋势确认

    Args:
        dfs: {'1d': df_daily, '4h': df_4h, '1h': df_1h, '15m': df_15m}
    """
    tf_order = ['1d', '4h', '1h', '15m', '5m']
    results = {}

    # 从大到小扫描
    for tf in sorted(dfs.keys(), key=lambda x: tf_order.index(x) if x in tf_order else 99):
        df = dfs[tf]
        # 找到4倍以上的大周期结果
        higher_result = None
        for htf in tf_order:
            if htf in results and _tf_minutes(htf) >= _tf_minutes(tf) * 4:
                higher_result = results[htf]
                break

        higher_df = dfs.get(higher_result.timeframe) if higher_result else None
        scanner = NakedKScanner(df, symbol=symbol, timeframe=tf,
                                higher_tf_df=higher_df)
        results[tf] = scanner.scan()

    return results


def _tf_minutes(tf: str) -> int:
    """周期字符串 → 分钟数"""
    tf_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30,
              '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
    return tf_map.get(tf, 60)


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == "__main__":
    # 生成模拟数据测试
    np.random.seed(42)
    n = 200

    # 模拟一个上升趋势 + 回调 + BMS结构
    price = 50000
    closes = []
    highs = []
    lows = []
    opens = []

    for i in range(n):
        # 趋势: 0-100 上升, 100-130 回调, 130-170 反弹突破, 170-200 震荡
        if i < 100:
            drift = 50 + np.random.randn() * 150
        elif i < 130:
            drift = -80 + np.random.randn() * 120  # 回调
        elif i < 170:
            drift = 100 + np.random.randn() * 130  # 反弹
        else:
            drift = np.random.randn() * 100  # 震荡

        price += drift
        price = max(100, price)

        bar_range = price * 0.01 * abs(np.random.randn() + 1)
        o = price - bar_range * np.random.randn() * 0.3
        c = price + bar_range * np.random.randn() * 0.3
        h = max(o, c) + abs(np.random.randn()) * bar_range * 0.4
        l = min(o, c) - abs(np.random.randn()) * bar_range * 0.4

        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(l)

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'volume': np.random.rand(n) * 100
    })

    # 测试
    scanner = NakedKScanner(df, symbol="BTC/USDT", timeframe="1h")
    result = scanner.scan()

    print(f"=== 裸K扫描结果: {result.symbol} {result.timeframe} ===")
    print(f"市场倾向: {result.market_bias.value} (置信度={result.bias_confidence:.2f})")
    print(f"结构描述: {result.structure.description}")
    print(f"趋势能量: {result.structure.trend_energy}/10")
    print(f"趋势K占比: {result.structure.trend_k_ratio:.1%}")
    print(f"摆动点: {len(result.structure.swing_points)}个")
    print(f"支撑阻力Zone: {len(result.structure.sr_zones)}个")
    print(f"BMS/SMS信号: {len(result.structure.bms_signals)}个")
    print(f"\nK线分类统计:")
    for k, v in sorted(result.kline_summary.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}根")

    print(f"\n入场信号 ({len(result.signals)}个):")
    for sig in result.signals[:5]:
        print(f"  {sig.action.value} @ {sig.entry_price:.2f} | "
              f"评分={sig.score_3step}/10 | 盈亏比={sig.risk_reward:.1f} | "
              f"置信度={sig.confidence:.0%}")
        for r in sig.reasons:
            print(f"    {r}")
