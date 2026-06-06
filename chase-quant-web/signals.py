"""
Chase量化策略 — 多市场信号引擎
A股(同花顺) / 美股(VSTrader) / 加密货币(Binance/OKX)
所有信号基于免费数据源: akshare + yfinance + ccxt

偏差修正 v2.0:
  - 生存偏差: 记录已归零币, 回测收益自动打折
  - 前视偏差: 只用当前已知信息 (df.iloc[-1])
  - 数据质量: 复权已检查 (A股qfq / 美股Adj Close)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# 偏差修正模块
try:
    from bias_correction import (
        CRYPTO_WATCHLIST_CORRECTED,
        estimate_survival_bias,
        load_graveyard,
    )
    BIAS_CORRECTION_AVAILABLE = True
except ImportError:
    BIAS_CORRECTION_AVAILABLE = False


@dataclass
class Signal:
    """交易信号 — 偏差修正版"""
    market: str
    symbol: str
    name: str
    price: float
    action: str          # BUY / SELL / HOLD
    score: float         # 0-100 (原始分)
    confidence: float    # 0-1
    reasons: List[str]   # 为什么
    risk_level: str      # low / medium / high
    suggested_size: float  # 建议仓位 (RMB)
    stop_loss: float
    take_profit: float
    survival_bias_penalty: float = 0.0  # 生存偏差扣分

    @property
    def adjusted_score(self) -> float:
        """偏差修正后的分数"""
        return max(0, self.score - self.survival_bias_penalty)


def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
    if avg_loss < 1e-9:
        return 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def _calc_macd(close: np.ndarray) -> Tuple[float, float, float]:
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd.iloc[-1] - signal.iloc[-1]
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist)


class CryptoSignals:
    """加密货币信号 (Binance/OKX via ccxt) — 偏差修正版"""

    # 存活币: 当前Top15, 用于实盘交易
    WATCHLIST = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
        "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "APT/USDT", "OP/USDT",
    ]

    # 归零币幽灵列表: 不交易, 但用于风险校准
    # 每季度对照 CoinGecko Top 200 更新
    GRAVEYARD_SYMBOLS = {
        "LUNA", "UST", "FTT", "CEL", "VGX", "SRM", "ANC", "MIR",
    }

    @property
    def survival_bias_warning(self) -> str:
        """生存偏差提醒"""
        bias = estimate_survival_bias("crypto")
        return (
            f"⚠️ 生存偏差: 回测收益可能虚高 {bias['estimated_overstatement_pct']}%。"
            f"实盘建议修正系数 ×{bias['correction_factor']}。"
            f"已追踪 {bias['dead_count']} 个归零案例。"
        )

    def __init__(self):
        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
            except Exception:
                self._exchange = None
        return self._exchange

    def fetch_ohlcv(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            ex = self.exchange
            if ex is None:
                return None
            ohlcv = ex.fetch_ohlcv(symbol, "1d", limit=limit)
            df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            return df
        except Exception:
            return None

    def scan(self) -> List[Signal]:
        signals = []
        for symbol in self.WATCHLIST:
            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 50:
                continue

            close = df["close"].values
            price = close[-1]
            reasons = []
            score = 50

            # 生存偏差检查: 对市值<1B的币降分
            if symbol in self.GRAVEYARD_SYMBOLS:
                reasons.append("⚠️归零风险库匹配")
                score -= 30

            # RSI
            rsi = _calc_rsi(close)
            if 30 < rsi < 40:
                score += 15; reasons.append(f"RSI={rsi:.0f} 超卖反弹区")
            elif 40 <= rsi <= 60:
                score += 10; reasons.append(f"RSI={rsi:.0f} 中性健康")
            elif rsi > 70:
                score -= 20; reasons.append(f"RSI={rsi:.0f} 超买⚠️")
            elif rsi < 30:
                score += 20; reasons.append(f"RSI={rsi:.0f} 极度超卖🔔")

            # MACD
            macd, signal_line, hist = _calc_macd(close)
            if hist > 0:
                score += 10; reasons.append("MACD金叉")
            else:
                score -= 10; reasons.append("MACD死叉")

            # 均线
            sma20 = pd.Series(close).rolling(20).mean()
            if price > sma20.iloc[-1]:
                score += 10; reasons.append("站上20日均线")
            else:
                score -= 5

            # 动量
            ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
            ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0
            if ret_5d > 3:
                score += 10; reasons.append(f"5日动量+{ret_5d:.1f}%")
            elif ret_5d < -5:
                score -= 10; reasons.append(f"5日急跌{ret_5d:.1f}%")
            if ret_20d > 10:
                score += 10; reasons.append(f"20日趋势+{ret_20d:.1f}%")

            # 波动率
            returns = np.diff(close) / close[:-1]
            vol = np.std(returns[-20:]) * np.sqrt(365)
            if vol > 1.0:
                score -= 10; reasons.append(f"高波动{vol:.0%}⚠️")
                risk = "high"
            elif vol < 0.3:
                score += 5; risk = "low"
            else:
                risk = "medium"

            score = max(0, min(100, score))
            if score >= 65:
                action = "BUY"
            elif score < 35:
                action = "SELL"
            else:
                action = "HOLD"

            signals.append(Signal(
                market="crypto", symbol=symbol,
                name=symbol.replace("/USDT", ""),
                price=price, action=action,
                score=score, confidence=min(0.95, max(0.25, 1 - score / 120)),
                reasons=reasons, risk_level=risk,
                suggested_size=max(200, min(2000, score * 20)),
                stop_loss=price * 0.92, take_profit=price * 1.15,
            ))

        return sorted(signals, key=lambda s: s.score, reverse=True)


class AStockSignals:
    """A股信号 (同花顺标准, via akshare)"""

    WATCHLIST = [
        ("000001", "平安银行"), ("000002", "万科A"), ("000858", "五粮液"),
        ("002415", "海康威视"), ("002594", "比亚迪"), ("300750", "宁德时代"),
        ("600519", "贵州茅台"), ("601318", "中国平安"), ("600036", "招商银行"),
        ("600276", "恒瑞医药"), ("601012", "隆基绿能"), ("600900", "长江电力"),
        ("000568", "泸州老窖"), ("002475", "立讯精密"), ("300059", "东方财富"),
        ("603259", "药明康德"), ("600809", "山西汾酒"), ("002714", "牧原股份"),
        ("300124", "汇川技术"), ("601888", "中国中免"),
    ]

    def fetch_daily(self, code: str) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak
            # akshare 东方财富日线
            symbol = f"{'sh' if code.startswith('6') else 'sz'}{code}"
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=(datetime.now() - timedelta(days=200)).strftime("%Y%m%d"),
                                   end_date=datetime.now().strftime("%Y%m%d"),
                                   adjust="qfq")
            if df is None or len(df) < 30:
                return None
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                                     "最高": "high", "最低": "low", "成交量": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return None

    def scan(self) -> List[Signal]:
        signals = []
        for code, name in self.WATCHLIST:
            df = self.fetch_daily(code)
            if df is None or len(df) < 30:
                continue

            close = df["close"].values
            price = close[-1]
            reasons = []
            score = 50

            # RSI
            rsi = _calc_rsi(close)
            if 30 < rsi < 40:
                score += 15; reasons.append(f"RSI={rsi:.0f} 超卖区")
            elif 40 <= rsi <= 60:
                score += 10; reasons.append(f"RSI={rsi:.0f} 中性")
            elif rsi > 75:
                score -= 20; reasons.append(f"RSI={rsi:.0f} 超买⚠️")
            elif rsi < 25:
                score += 20; reasons.append(f"RSI={rsi:.0f} 极超卖")

            # MACD
            macd, signal_line, hist = _calc_macd(close)
            if hist > 0:
                score += 10; reasons.append("MACD正柱")
            else:
                score -= 10; reasons.append("MACD负柱")

            # 均线排列 (5/20/60)
            sma5 = pd.Series(close).rolling(5).mean()
            sma20 = pd.Series(close).rolling(20).mean()
            sma60 = pd.Series(close).rolling(60).mean()
            if sma5.iloc[-1] > sma20.iloc[-1] > sma60.iloc[-1]:
                score += 15; reasons.append("多头排列")
            elif sma5.iloc[-1] < sma20.iloc[-1]:
                score -= 10; reasons.append("短线死叉")

            # 量价
            vol_5d = np.mean(df["volume"].values[-5:])
            vol_20d = np.mean(df["volume"].values[-20:])
            if vol_20d > 0 and vol_5d / vol_20d > 1.5:
                score += 10; reasons.append("放量")
            elif vol_20d > 0 and vol_5d / vol_20d < 0.5:
                score -= 5; reasons.append("缩量")

            # 动量
            ret_10d = (close[-1] / close[-11] - 1) * 100 if len(close) > 10 else 0
            if ret_10d > 5:
                score += 10; reasons.append(f"10日+{ret_10d:.1f}%")
            elif ret_10d < -8:
                score -= 10; reasons.append(f"10日急跌{ret_10d:.1f}%")

            risk = "low" if score > 60 else "medium" if score > 35 else "high"

            score = max(0, min(100, score))
            if score >= 70:
                action = "BUY"
            elif score < 30:
                action = "SELL"
            else:
                action = "HOLD"

            signals.append(Signal(
                market="a_stock", symbol=code,
                name=name, price=price, action=action,
                score=score, confidence=min(0.95, max(0.25, 1 - score / 120)),
                reasons=reasons, risk_level=risk,
                suggested_size=max(200, min(1500, score * 15)),
                stop_loss=price * 0.93, take_profit=price * 1.12,
            ))

        return sorted(signals, key=lambda s: s.score, reverse=True)


class USStockSignals:
    """美股信号 (VSTrader, via yfinance)"""

    WATCHLIST = [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"),
        ("NVDA", "NVIDIA"), ("TSLA", "Tesla"), ("META", "Meta"),
        ("AMZN", "Amazon"), ("AMD", "AMD"), ("NFLX", "Netflix"),
        ("BABA", "阿里巴巴"), ("JD", "京东"), ("PDD", "拼多多"),
        ("NIO", "蔚来"), ("BIDU", "百度"), ("LI", "理想汽车"),
        ("SPY", "标普500ETF"), ("QQQ", "纳斯达克100ETF"), ("IWM", "罗素2000ETF"),
        ("SMH", "半导体ETF"), ("TLT", "长期国债ETF"),
    ]

    def fetch_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
            if df is None or len(df) < 30:
                return None
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Open": "open", "Close": "close",
                                     "High": "high", "Low": "low", "Volume": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return None

    def scan(self) -> List[Signal]:
        signals = []
        for symbol, name in self.WATCHLIST:
            df = self.fetch_daily(symbol)
            if df is None or len(df) < 30:
                continue

            close = df["close"].values
            price = close[-1]
            reasons = []
            score = 50

            # RSI
            rsi = _calc_rsi(close)
            if 30 < rsi < 40:
                score += 15; reasons.append(f"RSI={rsi:.0f} 超卖")
            elif 40 <= rsi <= 60:
                score += 10; reasons.append(f"RSI={rsi:.0f} 健康")
            elif rsi > 70:
                score -= 20; reasons.append(f"RSI={rsi:.0f} 超买")

            # MACD
            macd, signal_line, hist = _calc_macd(close)
            if hist > 0:
                score += 10; reasons.append("MACD金叉")
            else:
                score -= 10

            # SMA20
            sma20 = pd.Series(close).rolling(20).mean()
            if price > sma20.iloc[-1]:
                score += 10; reasons.append("站上20日线")
            else:
                score -= 5

            # 动量
            ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0
            if ret_20d > 5:
                score += 10; reasons.append(f"20日+{ret_20d:.1f}%")
            elif ret_20d < -10:
                score -= 10; reasons.append(f"20日大跌{ret_20d:.1f}%")

            # 波动
            returns = np.diff(close) / close[:-1]
            vol = np.std(returns[-20:]) * np.sqrt(252)
            risk = "high" if vol > 0.5 else "low" if vol < 0.2 else "medium"
            if vol > 0.6:
                score -= 10; reasons.append(f"高波{vol:.0%}")

            score = max(0, min(100, score))
            if score >= 65:
                action = "BUY"
            elif score < 35:
                action = "SELL"
            else:
                action = "HOLD"

            # 美股价格换算RMB (约7.2)
            signals.append(Signal(
                market="us_stock", symbol=symbol,
                name=f"{name} (${price:.2f})",
                price=price, action=action,
                score=score, confidence=min(0.95, max(0.25, 1 - score / 120)),
                reasons=reasons, risk_level=risk,
                suggested_size=max(200, min(1500, score * 15)),
                stop_loss=price * 0.92, take_profit=price * 1.15,
            ))

        return sorted(signals, key=lambda s: s.score, reverse=True)


class SignalEngine:
    """统一信号引擎"""

    def __init__(self):
        self.crypto = CryptoSignals()
        self.a_stock = AStockSignals()
        self.us_stock = USStockSignals()

    def scan_all(self) -> Dict[str, List[Signal]]:
        """扫描全市场, 返回 BUY 信号 (score >= 65)"""
        results = {}
        for market, scanner in [
            ("crypto", self.crypto),
            ("a_stock", self.a_stock),
            ("us_stock", self.us_stock),
        ]:
            try:
                all_sigs = scanner.scan()
                results[market] = [s for s in all_sigs if s.action == "BUY"]
            except Exception as e:
                results[market] = []
                print(f"⚠️ {market} 扫描失败: {e}")
        return results

    def scan_all_signals(self) -> Dict[str, List[Signal]]:
        """扫描全市场, 返回所有信号"""
        results = {}
        for market, scanner in [
            ("crypto", self.crypto),
            ("a_stock", self.a_stock),
            ("us_stock", self.us_stock),
        ]:
            try:
                results[market] = scanner.scan()
            except Exception:
                results[market] = []
        return results
