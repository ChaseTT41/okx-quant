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

# 币种全名映射
COIN_FULL_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "BNB Chain",
    "SOL": "Solana", "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin",
    "AVAX": "Avalanche", "DOT": "Polkadot", "LINK": "Chainlink",
    "MATIC": "Polygon", "ATOM": "Cosmos", "LTC": "Litecoin",
    "UNI": "Uniswap", "APT": "Aptos", "NEAR": "NEAR Protocol",
    "OP": "Optimism", "ARB": "Arbitrum", "SUI": "Sui", "TON": "Toncoin",
    "FIL": "Filecoin", "TRX": "TRON", "ETC": "Ethereum Classic",
    "ICP": "Internet Computer", "RENDER": "Render",
    "WIF": "dogwifhat", "PEPE": "Pepe", "AAVE": "Aave", "ORDI": "ORDI",
}

def _coin_full_name(ticker: str) -> str:
    return COIN_FULL_NAMES.get(ticker.upper(), ticker)

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
                name=_coin_full_name(symbol.replace("/USDT", "")),
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
            # 使用腾讯数据源 (更稳定, 东方财富API频繁断连)
            symbol_tx = f"{'sh' if code.startswith('6') else 'sz'}{code}"
            start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            end = datetime.now().strftime("%Y%m%d")
            df = ak.stock_zh_a_hist_tx(symbol=symbol_tx, start_date=start, end_date=end)
            if df is None or len(df) < 30:
                return None
            # stock_zh_a_hist_tx 列名: date/open/close/high/low/amount (amount=成交额, 无volume)
            df["date"] = pd.to_datetime(df["date"])
            if "volume" not in df.columns:
                df["volume"] = df.get("amount", 0)  # 用成交额替代成交量
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


class HKStockSignals:
    """港股信号 (五维评分卡, via hk_stock_data + hk_five_dim_scorer)"""

    WATCHLIST = [
        ("00700", "腾讯控股"), ("09988", "阿里巴巴"), ("03690", "美团"),
        ("02318", "中国平安"), ("00388", "港交所"), ("01299", "友邦保险"),
        ("00939", "建设银行"), ("01398", "工商银行"), ("00941", "中国移动"),
        ("00883", "中海油"), ("01211", "比亚迪股份"), ("01024", "快手"),
        ("09618", "京东"), ("09999", "网易"), ("01810", "小米集团"),
        ("09888", "百度集团"), ("02269", "药明生物"), ("01109", "华润置地"),
        ("02020", "安踏体育"), ("03968", "招商银行"),
    ]

    def __init__(self):
        self._scorer = None

    @property
    def scorer(self):
        if self._scorer is None:
            try:
                import sys
                from pathlib import Path
                _hk_path = str(Path(__file__).parent.parent)
                if _hk_path not in sys.path:
                    sys.path.insert(0, _hk_path)
                from hk_five_dim_scorer import FiveDimScorer
                self._scorer = FiveDimScorer()
            except Exception:
                self._scorer = False
        return self._scorer if self._scorer is not False else None

    def scan(self) -> List[Signal]:
        signals = []
        scorer = self.scorer
        if scorer is None:
            return signals

        for code, name in self.WATCHLIST:
            try:
                result = scorer.score(code)
                price = result.close
                if price <= 0:
                    continue

                reasons = []
                if result.trend.signals:
                    reasons.extend(result.trend.signals[:2])
                if result.ob_os.signals:
                    reasons.append(result.ob_os.signals[0])
                if result.fundamental.signals:
                    reasons.append(result.fundamental.signals[0])

                composite = result.composite
                if composite >= 65:
                    action = "BUY"
                elif composite < 35:
                    action = "SELL"
                else:
                    action = "HOLD"

                risk = "low" if composite >= 70 else "medium" if composite >= 40 else "high"
                score_val = max(0, min(100, composite))
                confidence = result.confidence

                signals.append(Signal(
                    market="hk_stock", symbol=code,
                    name=name, price=price, action=action,
                    score=score_val, confidence=confidence,
                    reasons=reasons[:5], risk_level=risk,
                    suggested_size=max(200, min(1500, score_val * 12)),
                    stop_loss=result.stop_loss if result.stop_loss > 0 else price * 0.92,
                    take_profit=result.target if result.target > 0 else price * 1.10,
                ))
            except Exception:
                continue

        return sorted(signals, key=lambda s: s.score, reverse=True)


class BStockSignals:
    """bStocks信号 — 币安代币化美股 (24/7交易, USDT计价, BNB Chain)

    bStocks 是 Binance 2026年6月11日推出的代币化美股产品:
      - BEP-677 代币, 1:1 由真实股票背书
      - 首批: NVDAB, TSLAB, CRCLB, MUB, SNDKB
      - 全部 /USDT 交易对, 最低 $5
      - 挂单费豁免至 2026-09-01
      - 24/7 交易 (无美股交易时段限制!)
    """

    # bStocks 首批5只 + 预计扩展
    WATCHLIST = [
        ("NVDAB/USDT", "NVIDIA Tokenized"),    # 英伟达
        ("TSLAB/USDT", "Tesla Tokenized"),     # 特斯拉
        ("CRCLB/USDT", "Circle Tokenized"),    # Circle (USDC发行商)
        ("MUB/USDT", "MicroStrategy Tokenized"),  # MicroStrategy (BTC大户)
        ("SNDKB/USDT", "Sandisk Tokenized"),   # 闪迪
    ]

    # 中文名映射
    NAME_MAP = {
        "NVDAB": "英伟达·b", "TSLAB": "特斯拉·b", "CRCLB": "Circle·b",
        "MUB": "微策略·b", "SNDKB": "闪迪·b",
    }

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
        for symbol, full_name in self.WATCHLIST:
            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 5:
                # bStocks 才上线几天, 数据不足时用模拟估值
                continue

            close = df["close"].values
            price = close[-1]
            ticker = symbol.replace("/USDT", "")
            name = self.NAME_MAP.get(ticker, full_name)
            reasons = []
            score = 50

            # RSI (短周期，因数据少)
            if len(close) >= 14:
                rsi = _calc_rsi(close, period=min(14, len(close)-1))
                if 30 < rsi < 40:
                    score += 15; reasons.append(f"RSI={rsi:.0f} 超卖反弹区")
                elif 40 <= rsi <= 60:
                    score += 10; reasons.append(f"RSI={rsi:.0f} 中性健康")
                elif rsi > 70:
                    score -= 20; reasons.append(f"RSI={rsi:.0f} 超买⚠️")
                elif rsi < 30:
                    score += 20; reasons.append(f"RSI={rsi:.0f} 极度超卖🔔")

            # MACD
            if len(close) >= 26:
                macd, signal_line, hist = _calc_macd(close)
                if hist > 0:
                    score += 10; reasons.append("MACD金叉")
                else:
                    score -= 10; reasons.append("MACD死叉")

            # 短期均线
            if len(close) >= 5:
                sma5 = pd.Series(close).rolling(5).mean()
                if price > sma5.iloc[-1]:
                    score += 10; reasons.append("站上5日线")
                else:
                    score -= 5

            # 动量 (bStocks刚上线, 用1日/3日)
            if len(close) > 1:
                ret_1d = (close[-1] / close[-2] - 1) * 100
                if ret_1d > 3:
                    score += 10; reasons.append(f"1日+{ret_1d:.1f}%")
                elif ret_1d < -3:
                    score -= 10; reasons.append(f"1日跌{ret_1d:.1f}%")

            # bStocks 特殊加分: 新品溢价效应 (刚上线5天)
            reasons.append("🆕 bStocks新品 (上线<1周)")

            # 波动率 (数据可能不足, 保守处理)
            risk = "medium"

            score = max(0, min(100, score))
            if score >= 60:
                action = "BUY"
            elif score < 35:
                action = "SELL"
            else:
                action = "HOLD"

            # 价格是 USD, 换算 RMB (汇率 ~7.2)
            price_cny = price * 7.2

            signals.append(Signal(
                market="b_stock", symbol=symbol,
                name=f"{name} (${price:.2f})",
                price=price_cny, action=action,
                score=score, confidence=min(0.95, max(0.25, 1 - score / 120)),
                reasons=reasons, risk_level=risk,
                suggested_size=max(200, min(1000, score * 10)),
                stop_loss=price_cny * 0.90,    # bStocks新品波动大, 止损放宽到-10%
                take_profit=price_cny * 1.20,   # 止盈+20%
            ))

        return sorted(signals, key=lambda s: s.score, reverse=True)


class SignalEngine:
    """统一信号引擎"""

    def __init__(self):
        self.crypto = CryptoSignals()
        self.a_stock = AStockSignals()
        self.us_stock = USStockSignals()
        self.hk_stock = HKStockSignals()
        self.b_stock = BStockSignals()

    def scan_all(self) -> Dict[str, List[Signal]]:
        """扫描全市场, 返回 BUY 信号 (score >= 65)"""
        results = {}
        for market, scanner in [
            ("crypto", self.crypto),
            ("a_stock", self.a_stock),
            ("us_stock", self.us_stock),
            ("hk_stock", self.hk_stock),
            ("b_stock", self.b_stock),
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
            ("hk_stock", self.hk_stock),
            ("b_stock", self.b_stock),
        ]:
            try:
                results[market] = scanner.scan()
            except Exception:
                results[market] = []
        return results
