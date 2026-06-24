"""
Yina 股票合约扫描器 🐾
专门扫描 OKX 股票永续合约（半导体/AI软件/太空等），基于技术指标 + 板块资金流 + 情绪

为什么需要这个模块：
  strategy_runner.py 的策略 universe 只有 crypto（BTC/ETH/SOL等~80个币），
  完全不覆盖 MU/NVDA/ASML 等股票合约。本模块填补这个空白。

策略逻辑（规则+技术指标驱动，不依赖 ML 重训）：
  1. 超卖反弹: RSI < 30 + 价格触布林下轨 + 板块资金流入
  2. 动量突破: MACD 金叉 + 成交量放大 + 价格站上 MA20
  3. 趋势跟踪: MA20 > MA50 + RSI 50-70 + 板块资金流正
  4. 恐慌反转: F&G < 30 (Extreme Fear) + 连跌3天 + 板块资金拐点
"""

import os, sys, json, time, requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent

# ── 优先扫描列表 (Chase哥 核心持仓 + 重点关注) ──
PRIORITY_WATCHLIST = [
    "MU/USDT",       # 🔥 美光科技 — Chase哥持仓
    "NVDA/USDT",     # 🥇 英伟达 — AI算力之王
    "AMD/USDT",      # 🥈 AMD — 数据中心GPU
    "ASML/USDT",     # 🥇 阿斯麦 — 光刻机垄断
    "AVGO/USDT",     # 博通 — 网络芯片
    "TSM/USDT",      # 台积电 — 芯片代工
    "QCOM/USDT",     # 高通 — 移动芯片
    "ARM/USDT",      # ARM — 芯片架构
    "INTC/USDT",     # 英特尔 — 老牌芯片
    "MRVL/USDT",     # Marvell — 数据中心芯片
]

# ── 扩展列表 (其他有流动性的股票/ETF/商品合约) ──
EXTENDED_WATCHLIST = [
    "MSFT/USDT", "GOOGL/USDT", "META/USDT", "AAPL/USDT",
    "TSLA/USDT", "AMZN/USDT", "NFLX/USDT",
    "RKLB/USDT", "ASTS/USDT", "LUNR/USDT",
    "SPCX/USDT",     # 🚀 太空ETF — 间接持有SpaceX! 全球最大未上市公司的唯一敞口
    "SPY/USDT", "QQQ/USDT",
    "XAU/USDT", "XAG/USDT",
]

# ── 🚀 策略资产特权 (Chase哥指定核心标的, 放宽阈值) ──
STRATEGIC_ASSETS = {
    "SPCX/USDT": {
        "name": "SpaceX太空ETF",
        "rsi_oversold": 40,        # 超卖RSI阈值 (默认35, SPCX放宽到40)
        "fear_contrarian_min": 28, # 恐慌反转最低分 (默认35, SPCX降到28)
        "confidence_boost": 0.05,  # 额外置信度加成
        "score_boost": 5,          # 额外评分加成
    },
}

# ── 风控参数 ──
MAX_CONCURRENT_STOCK_POSITIONS = 5      # 股票合约最大同时持仓
MAX_STOCK_POSITION_PCT = 0.03           # 单笔最大仓位 3% (比crypto保守)
MIN_CONFIDENCE_FOR_ENTRY = 0.55         # 最低入场置信度
SCAN_COOLDOWN_CYCLES = 2                # 同一标的连续信号的冷却周期


@dataclass
class StockSignal:
    """股票合约交易信号"""
    symbol: str
    name: str
    action: str                        # BUY / SELL / HOLD
    price: float
    score: float                       # 0-100 综合评分
    confidence: float                  # 0-1 置信度
    signal_type: str                   # oversold_bounce / momentum_breakout / trend_following / fear_contrarian
    reasons: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    suggested_size: float = 0.02       # 建议仓位占比
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_name: str = "stock_swap_scanner"
    indicators: dict = field(default_factory=dict)  # 技术指标快照


class StockSwapScanner:
    """OKX 股票永续合约扫描器"""

    def __init__(self):
        self.base_url = os.environ.get('OKX_API_URL', 'https://www.okx.cab')
        self._cache: Dict[str, dict] = {}
        self._last_full_scan: Dict[str, float] = {}  # symbol → last_scan_ts
        self._signal_history: Dict[str, list] = {}    # symbol → [(ts, signal_type)]

    # ═══════════════════════════════════════════
    # 数据获取
    # ═══════════════════════════════════════════

    def _okx_get(self, path: str, timeout: int = 10) -> dict:
        """OKX 公开 API 请求"""
        try:
            r = requests.get(f'{self.base_url}{path}', timeout=timeout)
            return r.json()
        except Exception as e:
            return {"code": "-1", "msg": str(e)}

    def fetch_candles(self, symbol: str, bar: str = "1H", limit: int = 200) -> Optional[np.ndarray]:
        """
        拉取 OKX K线数据

        OKX 返回格式 (倒序: 最新在前):
          [[ts, open, high, low, close, vol, volCcy], ...]

        Returns:
          numpy array [limit, 6] 正序排列: [ts, open, high, low, close, volume]
        """
        inst_id = symbol.replace("/", "-") + "-SWAP"
        try:
            data = self._okx_get(
                f"/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
            )
            if data.get('code') != '0' or not data.get('data'):
                return None

            candles = []
            for c in data['data']:
                candles.append([
                    float(c[0]), float(c[1]), float(c[2]),
                    float(c[3]), float(c[4]), float(c[5])
                ])
            # 倒序转正序
            arr = np.array(candles[::-1])
            if len(arr) < 50:
                return None
            return arr
        except Exception:
            return None

    def fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """批量获取当前价格"""
        prices = {}
        try:
            data = self._okx_get("/api/v5/market/tickers?instType=SWAP", timeout=15)
            if data.get('code') == '0':
                for t in data.get('data', []):
                    inst = t['instId'].replace('-USDT-SWAP', '')
                    prices[inst] = float(t['last'])
        except Exception:
            pass

        # 补漏: 逐个查询
        for sym in symbols:
            if sym not in prices:
                try:
                    inst_id = sym.replace("/", "-") + "-SWAP"
                    data = self._okx_get(f"/api/v5/market/ticker?instId={inst_id}")
                    if data.get('code') == '0' and data.get('data'):
                        prices[sym] = float(data['data'][0]['last'])
                except Exception:
                    pass
        return prices

    # ═══════════════════════════════════════════
    # 技术指标计算
    # ═══════════════════════════════════════════

    def compute_indicators(self, candles: np.ndarray) -> dict:
        """
        计算全套技术指标

        Args:
          candles: [n, 6] — [ts, open, high, low, close, volume]

        Returns:
          dict with rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle,
          bb_lower, ma20, ma50, vol_ratio, consecutive_down_days
        """
        close = candles[:, 4]
        high = candles[:, 2]
        low = candles[:, 3]
        volume = candles[:, 5]
        n = len(close)

        result = {}

        # RSI(14)
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[13] = gain[:14].mean()
        avg_loss[13] = loss[:14].mean()
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 1e9), where=avg_loss != 0)
        result['rsi'] = float(100 - (100 / (1 + rs[-1])))
        result['rsi_prev'] = float(100 - (100 / (1 + (rs[-2] if len(rs) > 1 else rs[-1]))))

        # MACD(12, 26, 9)
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = self._ema(macd_line, 9)
        result['macd'] = float(macd_line[-1])
        result['macd_signal'] = float(macd_signal[-1])
        result['macd_hist'] = float(macd_line[-1] - macd_signal[-1])
        # 金叉/死叉检测
        result['macd_crossover'] = (
            macd_line[-2] <= macd_signal[-2] and macd_line[-1] > macd_signal[-1]
        )
        result['macd_crossunder'] = (
            macd_line[-2] >= macd_signal[-2] and macd_line[-1] < macd_signal[-1]
        )

        # 布林带 (20, 2)
        ma20 = np.convolve(close, np.ones(20)/20, mode='valid')
        bb_mid = ma20[-1]
        bb_std = np.std(close[-20:])
        result['bb_upper'] = float(bb_mid + 2 * bb_std)
        result['bb_middle'] = float(bb_mid)
        result['bb_lower'] = float(bb_mid - 2 * bb_std)
        result['bb_position'] = float((close[-1] - result['bb_lower']) /
                                      (result['bb_upper'] - result['bb_lower'] + 1e-9))

        # 移动均线
        result['ma20'] = float(np.mean(close[-20:]))
        result['ma50'] = float(np.mean(close[-min(50, n):]))
        result['price_vs_ma20'] = float(close[-1] / result['ma20'] - 1)
        result['price_vs_ma50'] = float(close[-1] / result['ma50'] - 1)

        # ATR(14) — 波动率
        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        # pad to same length
        tr_full = np.zeros(n)
        tr_full[1:] = tr
        result['atr'] = float(np.mean(tr_full[-14:]))
        result['atr_pct'] = float(result['atr'] / close[-1])

        # 成交量
        vol_ma20 = np.mean(volume[-20:])
        result['vol_ratio'] = float(volume[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

        # 连跌天数
        consecutive = 0
        for i in range(len(close)-1, 0, -1):
            if close[i] < close[i-1]:
                consecutive += 1
            else:
                break
        result['consecutive_down_days'] = consecutive

        # 日内涨跌幅 (24h from OKX)
        result['price'] = float(close[-1])
        result['change_1d'] = float(close[-1] / close[-min(24, n)] - 1) if n >= 24 else 0.0
        result['change_3d'] = float(close[-1] / close[-min(72, n)] - 1) if n >= 72 else 0.0

        return result

    @staticmethod
    def _ema(data: np.ndarray, span: int) -> np.ndarray:
        """指数移动平均"""
        alpha = 2 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    # ═══════════════════════════════════════════
    # 信号生成
    # ═══════════════════════════════════════════

    def scan_single(self, symbol: str, name: str, ind: dict,
                    sector_flow: float = 0.0, fear_greed: int = 50,
                    funding_rate: float = 0.0) -> Optional[StockSignal]:
        """
        对单个标的生成信号

        Args:
          symbol: e.g. "MU/USDT"
          name: 中文名称
          ind: compute_indicators() 返回的指标字典
          sector_flow: 板块资金流强度 [-1, +1]
          fear_greed: 恐惧贪婪指数 0-100
          funding_rate: 年化资金费率

        Returns:
          StockSignal or None (无信号)
        """
        reasons = []
        score = 50.0       # 基础分
        confidence = 0.50  # 基础置信度
        signal_type = "hold"
        action = "HOLD"
        risk_level = "medium"
        suggested_size = 0.02
        stop_loss_pct = 0.05
        take_profit_pct = 0.10

        price = ind['price']
        rsi = ind['rsi']

        # ── 🚀 策略资产特权: 放宽阈值 ──
        strat_cfg = STRATEGIC_ASSETS.get(symbol, {})
        rsi_oversold_limit = strat_cfg.get('rsi_oversold', 35)
        fear_contrarian_min = strat_cfg.get('fear_contrarian_min', 35)
        is_strategic = bool(strat_cfg)

        # ── 策略1: 超卖反弹 ──
        oversold_score = 0
        if rsi < rsi_oversold_limit:
            oversold_score += 15
            tag = "🚀" if is_strategic else ""
            reasons.append(f"{tag}RSI超卖({rsi:.0f})")
        if ind['bb_position'] < 0.15:
            oversold_score += 10
            reasons.append(f"触及布林下轨({ind['bb_position']:.1%})")
        if sector_flow > 0:
            oversold_score += 10
            reasons.append(f"板块资金流入({sector_flow:+.2f})")
        if ind['consecutive_down_days'] >= 3:
            oversold_score += 5
            reasons.append(f"连跌{ind['consecutive_down_days']}天")

        if oversold_score >= 30:
            signal_type = "oversold_bounce"
            action = "BUY"
            # 连跌越狠反弹越强，但置信度不过高
            confidence = min(0.65, 0.50 + oversold_score / 200)
            score = 55 + oversold_score
            stop_loss_pct = 0.04     # 超卖止损紧一点
            take_profit_pct = 0.08
            risk_level = "medium"

        # ── 策略2: 动量突破 ──
        breakout_score = 0
        if ind.get('macd_crossover'):
            breakout_score += 20
            reasons.append("MACD金叉")
        if ind['vol_ratio'] > 1.5:
            breakout_score += 15
            reasons.append(f"放量{ind['vol_ratio']:.1f}x")
        if ind['price_vs_ma20'] > 0:
            breakout_score += 10
            reasons.append(f"站上MA20 ({ind['price_vs_ma20']:+.1%})")
        if ind['rsi'] > 45 and ind['rsi'] < 70:
            breakout_score += 5
            reasons.append(f"RSI健康({ind['rsi']:.0f})")

        if breakout_score >= 30 and action == "HOLD":
            # 动量突破优于超卖（如果两者同时触发，超卖优先因为更极端）
            signal_type = "momentum_breakout"
            action = "BUY"
            confidence = min(0.70, 0.50 + breakout_score / 150)
            score = 60 + breakout_score
            stop_loss_pct = 0.05
            take_profit_pct = 0.12
            risk_level = "medium"
            suggested_size = 0.025
        elif breakout_score >= 25 and action == "BUY":
            # 超卖+动量双重确认 → 提升置信度
            confidence += 0.05
            score += 5

        # ── 策略3: 趋势跟踪 ──
        trend_score = 0
        if ind['ma20'] > ind['ma50']:
            trend_score += 15
        if 45 < rsi < 75:
            trend_score += 10
        if ind['price_vs_ma20'] > 0.01:
            trend_score += 10
            reasons.append(f"MA20趋势向上 ({ind['price_vs_ma20']:+.1%})")
        if sector_flow > 0.1:
            trend_score += 10
        if ind['macd'] > ind['macd_signal']:
            trend_score += 5

        if trend_score >= 35 and action == "HOLD":
            signal_type = "trend_following"
            action = "BUY"
            confidence = min(0.72, 0.50 + trend_score / 150)
            score = 58 + trend_score * 0.8
            stop_loss_pct = 0.06
            take_profit_pct = 0.15
            risk_level = "low"
            suggested_size = 0.03

        # ── 策略4: 恐慌反转 (F&G contrarian) ──
        if fear_greed < 30 and action == "HOLD":
            contrarian_score = 30 if fear_greed < 20 else 20
            if ind['consecutive_down_days'] >= 3:
                contrarian_score += 15
                reasons.append(f"极度恐慌F&G={fear_greed} + 连跌{ind['consecutive_down_days']}天")
            if sector_flow > -0.2:  # 板块资金不再流出
                contrarian_score += 10
            if contrarian_score >= fear_contrarian_min:
                signal_type = "fear_contrarian"
                action = "BUY"
                confidence = min(0.68, 0.50 + contrarian_score / 200)
                score = 55 + contrarian_score * 0.8
                stop_loss_pct = 0.04
                take_profit_pct = 0.10
                risk_level = "medium"
                reasons.append("恐慌情绪极值→均值回归")
                if is_strategic:
                    reasons.append(f"🚀 策略资产: SpaceX太空ETF")

        # ── 负面过滤: 即使触发入场，也要检查风险 ──
        if action == "BUY":
            # 资金费率过高 (>50% 年化) → 降级或跳过
            if abs(funding_rate) > 0.50:
                reasons.append(f"⚠️ 资金费率过高({funding_rate:.0%}年化)")
                confidence -= 0.10
                score -= 10

            # 价格在 BB 上轨以上 + RSI > 75 → 不追高
            if ind['bb_position'] > 0.90 and rsi > 70:
                reasons.append(f"⚠️ 价格过高 (BB{ind['bb_position']:.0%}, RSI{rsi:.0f})")
                confidence -= 0.15
                score -= 15

            # 置信度不足 → 放弃
            if confidence < MIN_CONFIDENCE_FOR_ENTRY:
                return None

            # 连续同标的信号 → 降低置信度（避免重复扫入）
            history = self._signal_history.get(symbol, [])
            if history and len(history) >= 1:
                last_ts, last_type = history[-1]
                cycles_ago = (time.time() - last_ts) / 600  # 10分钟一个周期
                if cycles_ago < SCAN_COOLDOWN_CYCLES:
                    confidence -= 0.05

            if confidence < MIN_CONFIDENCE_FOR_ENTRY:
                return None

            # 记录信号
            self._signal_history.setdefault(symbol, []).append(
                (time.time(), signal_type)
            )

        # ── SELL 信号 (对已有持仓的止盈/止损提醒) ──
        # 这里主要生成 BUY，SELL 由杠杆引擎的止损止盈处理

        if action == "BUY":
            # ── 🚀 策略资产特权: 额外评分 + 置信度加成 ──
            if is_strategic:
                score += strat_cfg.get('score_boost', 0)
                confidence += strat_cfg.get('confidence_boost', 0)
                reasons.append(f"🚀 策略资产加成 +{strat_cfg.get('score_boost',0)}分")

            # ── 🎯 动态止盈: 基于布林上轨+ATR, 拒绝死板公式 ──
            bb_upper = ind.get('bb_upper', price * 1.15)
            bb_distance = (bb_upper / price) - 1 if price > 0 else 0.10
            atr_pct = ind.get('atr_pct', 0.02)
            # 技术止盈 = 布林上轨距离 × 1.2 (略穿出上轨), ATR高波动加10%空间
            raw_tp = bb_distance * 1.2 if bb_distance > 0.02 else 0.08
            if atr_pct > 0.04:
                raw_tp *= 1.1
            # 股票不像币: 止盈5%-20%合理区间
            technical_tp_pct = max(0.05, min(0.20, raw_tp))

            return StockSignal(
                symbol=symbol,
                name=name,
                action=action,
                price=price,
                score=round(score, 1),
                confidence=round(min(confidence, 0.80), 3),
                signal_type=signal_type,
                reasons=reasons,
                risk_level=risk_level,
                suggested_size=suggested_size,
                stop_loss=round(price * (1 - stop_loss_pct), 1),
                take_profit=round(price * (1 + technical_tp_pct), 1),
                indicators=ind,
            )

        return None

    # ═══════════════════════════════════════════
    # 主扫描入口
    # ═══════════════════════════════════════════

    def scan_all(
        self,
        sentiment_engine=None,
        existing_positions: set = None,
        force_full: bool = False,
    ) -> List[dict]:
        """
        扫描所有股票合约，返回标准格式信号列表

        Args:
          sentiment_engine: MarketSentimentEngine 实例 (获取板块流+情绪)
          existing_positions: 已有持仓的 symbol 集合 (避免重复)
          force_full: 强制扫描扩展列表

        Returns:
          标准信号 dict 列表，可直接喂给 _run_leverage_decisions()
        """
        if existing_positions is None:
            existing_positions = set()

        # ── 获取情绪数据 ──
        fear_greed = 50
        sector_flows = {}
        if sentiment_engine:
            try:
                fg = sentiment_engine.fetch_fear_greed(use_cache=True)
                fear_greed = fg.current_value if hasattr(fg, 'current_value') else 50
            except Exception:
                pass
            try:
                # 尝试获取最新的 sector_flows
                snapshot = getattr(sentiment_engine, '_last_snapshot', None)
                if snapshot and hasattr(snapshot, 'sector_flows'):
                    sector_flows = snapshot.sector_flows
            except Exception:
                pass

        # ── 确定扫描列表 ──
        scan_symbols = list(PRIORITY_WATCHLIST)
        if force_full:
            scan_symbols += EXTENDED_WATCHLIST
        # 去重
        scan_symbols = list(dict.fromkeys(scan_symbols))

        # 过滤已有持仓
        scan_symbols = [s for s in scan_symbols if s not in existing_positions]

        if not scan_symbols:
            return []

        # ── 并行拉取K线 + 当前价 ──
        prices = self.fetch_current_prices(scan_symbols)

        # ── 获取资金费率 ──
        funding_rates = {}
        if sentiment_engine:
            try:
                fr_data = sentiment_engine.fetch_funding_rates(scan_symbols, use_cache=True)
                for sym, fr in fr_data.items():
                    funding_rates[sym] = getattr(fr, 'annualized_rate', 0.0)
            except Exception:
                pass

        # ── 逐个扫描 ──
        signals = []
        semicon_flow = sector_flows.get('semiconductor', 0.0)
        ai_sw_flow = sector_flows.get('ai_software', 0.0)
        space_flow = sector_flows.get('space', 0.0)

        # 逐个扫描日志前缀
        space_markers = {"SPCX/USDT": "🚀", "RKLB/USDT": "🛰️", "ASTS/USDT": "📡", "LUNR/USDT": "🌙"}
        skipped = 0

        for sym in scan_symbols:
            # 判断板块
            sector_flow = self._get_sector_flow(sym, semicon_flow, ai_sw_flow, space_flow)
            name = self._stock_name(sym)

            candles = self.fetch_candles(sym, bar="1H", limit=200)
            if candles is None or len(candles) < 50:
                skipped += 1
                continue

            ind = self.compute_indicators(candles)
            fr = funding_rates.get(sym, 0.0)

            signal = self.scan_single(
                sym, name, ind,
                sector_flow=sector_flow,
                fear_greed=fear_greed,
                funding_rate=fr,
            )

            if signal is None:
                # 逐股可见: 被扫但未出信号 (含RSI/价格/板块)
                marker = space_markers.get(sym, "📊")
                rsi = ind.get('rsi', 50) if ind else 50
                px = prices.get(sym, 0)
                print(f"  {marker} {sym:15s} {name:20s} | ${px:>8.2f} | RSI={rsi:.0f} | ⚪ 无信号")
                continue

            # 转换为标准 dict 格式 (兼容 _run_leverage_decisions)
            is_strategic = bool(STRATEGIC_ASSETS.get(sym, {}))
            px_sig = signal.price if signal.price > 0 else prices.get(sym, 100.0)
            # 🎯 技术面止盈百分比 (从信号中反推)
            _tech_tp_pct = (signal.take_profit / px_sig - 1) if px_sig > 0 else 0.10
            signals.append({
                "symbol": signal.symbol,
                "name": signal.name,
                "action": signal.action,
                "price": px_sig,
                "score": signal.score,
                "confidence": signal.confidence,
                "reasons": signal.reasons,
                "risk_level": signal.risk_level,
                "suggested_size": signal.suggested_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "strategy_name": f"stock_{signal.signal_type}",
                "signal_val": (signal.confidence - 0.5) * 2,  # 转换为 [-1, +1]
                "_is_strategic": is_strategic,  # 🚀 策略资产特权标记
                "_strategic_config": STRATEGIC_ASSETS.get(sym, {}),  # 策略资产配置
                "_technical_tp_pct": round(_tech_tp_pct, 3),  # 🎯 基于布林上轨的动态止盈
            })
            # 信号日志
            marker = space_markers.get(sym, "📊")
            rsi = ind.get('rsi', 50) if ind else 50
            px = prices.get(sym, 0)
            icon = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "🟡"
            print(f"  {marker} {sym:15s} {name:20s} | ${px:>8.2f} | RSI={rsi:.0f} | {icon} {signal.action} | "
                  f"评分{signal.score:.0f} | 置信{signal.confidence:.1%} | {', '.join(signal.reasons[:2])}")

        # 逐股扫描完成摘要
        print(f"  📋 股票扫描完成: {len(scan_symbols)}只 → {len(signals)}信号 | {skipped}只数据不足 | force_full={force_full}")

        # 🚀 策略资产优先: 先分离策略信号，再按score排序
        strategic_signals = [s for s in signals if s.get('_is_strategic')]
        regular_signals = [s for s in signals if not s.get('_is_strategic')]
        strategic_signals.sort(key=lambda s: s['score'], reverse=True)
        regular_signals.sort(key=lambda s: s['score'], reverse=True)

        # 限制信号数量（避免一次开太多仓）
        current_stock_positions = len(existing_positions)
        max_new = MAX_CONCURRENT_STOCK_POSITIONS - current_stock_positions
        if max_new <= 0:
            return []

        # 🚀 策略资产至少保留1个名额，其余按score排序
        result = []
        if strategic_signals and max_new >= 1:
            result.append(strategic_signals[0])  # 至少保留1个策略资产
            max_new -= 1
        # 剩余名额: 其余策略资产 + 普通信号合并排序
        remaining = strategic_signals[1:] + regular_signals
        remaining.sort(key=lambda s: s['score'], reverse=True)
        result.extend(remaining[:max_new])

        return result

    def _get_sector_flow(self, sym: str, semicon: float, ai_sw: float, space: float) -> float:
        """根据 symbol 判断所属板块，返回对应 flow"""
        ticker = sym.split('/')[0]
        semicon_stocks = {'MU', 'NVDA', 'AMD', 'INTC', 'MRVL', 'AVGO', 'QCOM', 'TSM', 'ARM', 'ASML', 'AMAT', 'COHR', 'CIEN', 'CRDO', 'CGNX', 'AXTI', 'POET', 'WDC', 'SNDK', 'FLNC'}
        ai_sw_stocks = {'MSFT', 'GOOGL', 'META', 'AAPL', 'AMZN', 'NFLX', 'CRM', 'ADBE', 'ORCL', 'NOW', 'PLTR', 'SNOW', 'NET', 'MDB', 'DDOG', 'CRWD', 'ZS', 'PANW', 'FTNT', 'CYBR', 'PATH', 'AI', 'BBAI', 'SOUN', 'BMR', 'UPST', 'AFRM', 'SOFI', 'HOOD', 'COIN', 'MSTR', 'SQ'}
        space_stocks = {'RKLB', 'ASTS', 'LUNR', 'RDW', 'SPCE'}

        if ticker in semicon_stocks:
            return semicon
        elif ticker in ai_sw_stocks:
            return ai_sw
        elif ticker in space_stocks:
            return space
        return 0.0

    def _stock_name(self, sym: str) -> str:
        """MU/USDT → 美光科技 Micron"""
        names = {
            "MU": "美光科技", "NVDA": "英伟达", "AMD": "AMD", "INTC": "英特尔",
            "MRVL": "Marvell", "AVGO": "博通", "QCOM": "高通", "TSM": "台积电",
            "ARM": "ARM", "ASML": "阿斯麦", "AMAT": "应用材料",
            "MSFT": "微软", "GOOGL": "谷歌", "META": "Meta", "AAPL": "苹果",
            "TSLA": "特斯拉", "AMZN": "亚马逊", "NFLX": "奈飞",
            "RKLB": "Rocket Lab", "ASTS": "AST SpaceMobile", "LUNR": "Intuitive Machines",
            "WDC": "西部数据", "SNDK": "闪迪", "COHR": "Coherent",
            "CIEN": "Ciena", "CRDO": "Credo", "POET": "POET Technologies",
            "SPY": "标普500ETF", "QQQ": "纳斯达克ETF",
            "XAU": "黄金", "XAG": "白银",
        }
        ticker = sym.split('/')[0]
        return names.get(ticker, ticker)


# ═══════════════════════════════════════════════════════════
# 便捷函数: 供 daemon 直接调用
# ═══════════════════════════════════════════════════════════

_stock_scanner: Optional[StockSwapScanner] = None


def get_stock_scanner() -> StockSwapScanner:
    global _stock_scanner
    if _stock_scanner is None:
        _stock_scanner = StockSwapScanner()
    return _stock_scanner


def scan_stock_swaps(
    sentiment_engine=None,
    existing_positions: set = None,
    force_full: bool = False,
) -> List[dict]:
    """
    扫描股票合约，返回标准信号列表。
    可直接在 auto_trade_daemon.py 中调用。
    """
    scanner = get_stock_scanner()
    return scanner.scan_all(
        sentiment_engine=sentiment_engine,
        existing_positions=existing_positions,
        force_full=force_full,
    )


# ═══════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 加载环境变量
    env_path = PROJECT_DIR / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    print("🐾 Stock Swap Scanner 自测\n")

    scanner = StockSwapScanner()

    # 测试 MU
    print("═══ MU/USDT (美光科技) ═══")
    candles = scanner.fetch_candles("MU/USDT", bar="1H", limit=200)
    if candles is not None:
        print(f"  ✅ K线: {len(candles)} 根")
        ind = scanner.compute_indicators(candles)
        print(f"  价格: ${ind['price']:.2f}")
        print(f"  RSI(14): {ind['rsi']:.1f}")
        print(f"  MACD: {ind['macd']:.4f} (signal={ind['macd_signal']:.4f})")
        print(f"  金叉: {'✅' if ind['macd_crossover'] else '❌'}")
        print(f"  布林带: ${ind['bb_lower']:.1f} ~ ${ind['bb_upper']:.1f} (pos={ind['bb_position']:.1%})")
        print(f"  MA20: ${ind['ma20']:.1f} | MA50: ${ind['ma50']:.1f}")
        print(f"  量比: {ind['vol_ratio']:.1f}x")
        print(f"  连跌: {ind['consecutive_down_days']}天")
        print(f"  1日涨跌: {ind['change_1d']:+.2%}")
        print(f"  3日涨跌: {ind['change_3d']:+.2%}")

        signal = scanner.scan_single(
            "MU/USDT", "美光科技", ind,
            sector_flow=0.3, fear_greed=22
        )
        if signal:
            print(f"\n  🎯 信号: {signal.action} | {signal.signal_type}")
            print(f"  评分: {signal.score:.0f} | 置信: {signal.confidence:.1%}")
            print(f"  止损: ${signal.stop_loss:.1f} | 止盈: ${signal.take_profit:.1f}")
            print(f"  理由: {signal.reasons}")
        else:
            print(f"\n  💤 无信号")
    else:
        print("  ❌ K线获取失败")

    # 测试 NVDA
    print("\n═══ NVDA/USDT (英伟达) ═══")
    candles = scanner.fetch_candles("NVDA/USDT", bar="1H", limit=200)
    if candles is not None:
        ind = scanner.compute_indicators(candles)
        print(f"  价格: ${ind['price']:.2f} | RSI: {ind['rsi']:.1f} | 量比: {ind['vol_ratio']:.1f}x")
        signal = scanner.scan_single(
            "NVDA/USDT", "英伟达", ind,
            sector_flow=0.3, fear_greed=22
        )
        if signal:
            print(f"  🎯 信号: {signal.action} | {signal.signal_type} | 置信: {signal.confidence:.1%}")
        else:
            print(f"  💤 无信号")
    else:
        print("  ❌ K线获取失败")

    # 全量扫描
    print("\n═══ 全量扫描 ═══")
    signals = scanner.scan_all(force_full=False)
    print(f"  信号: {len(signals)} 个")
    for s in signals:
        print(f"  🎯 [{s['strategy_name']}] {s['symbol']} | "
              f"评分{s['score']:.0f} | 置信{s['confidence']:.1%} | "
              f"{', '.join(s['reasons'][:3])}")
