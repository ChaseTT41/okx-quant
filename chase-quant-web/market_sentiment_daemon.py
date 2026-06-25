#!/usr/bin/env python3
"""
🏛️ Yina 市场情绪守护进程 v1.0
================================
独立运行, 持续监控市场情绪, 生成辅助决策信号。

核心功能:
  1. 🔴🟢 双源 F&G 融合 (CMC + alternative.me) → 加权恐慌贪婪指数
  2. 📰 Gemini 新闻情绪 → 多币种情绪分数 + 关键驱动因素
  3. 📊 市场环境判定 → regime + bias + conviction + 杠杆乘数
  4. 💾 写入共享状态 → data/market_sentiment_state.json

设计原则:
  - 完全独立于 auto_trade_daemon.py, 可单独启停
  - 通过共享 JSON 文件 IPC, 零耦合
  - 旧 daemon 只需读文件, 改动极小

使用:
  python3 market_sentiment_daemon.py              # 前台运行
  python3 market_sentiment_daemon.py --daemon     # 后台运行
  python3 market_sentiment_daemon.py --once       # 单次运行

Chase哥 的设想:
  "当你有个标的分数很高，而且连市场分析也都有非常好的表现
   那你可以更加放心大胆的去建仓，而且可以多加杠杆"
  → market_regime.leverage_multiplier 实现
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

BEIJING_TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "market_sentiment_state.json"
LOG_FILE = DATA_DIR / "sentiment_daemon.log"

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

FNG_REFRESH_SEC = 300       # F&G 每 5 分钟刷新
NEWS_REFRESH_SEC = 1800     # 新闻情绪每 30 分钟刷新
REGIME_REFRESH_SEC = 300    # 市场环境每 5 分钟刷新

# 双源 F&G 权重
CMC_WEIGHT = 0.6            # CMC 权重 (更新更频繁)
ALT_ME_WEIGHT = 0.4         # alternative.me 权重 (包含历史数据)

# 情绪→信号 杠杆乘数
# 当市场情绪方向与交易信号方向一致时，提升杠杆上限
SENTIMENT_LEVERAGE_MAP = {
    "strong_aligned": 2.0,   # 情绪极度一致 → 杠杆×2 (比如5x→10x)
    "aligned": 1.5,          # 情绪一致 → 杠杆×1.5
    "neutral": 1.0,          # 中性 → 不变
    "opposed": 0.7,          # 情绪相反 → 杠杆×0.7 (降杠杆)
    "strongly_opposed": 0.5, # 情绪极度相反 → 杠杆×0.5 (大幅降杠杆)
}

# 日志
def _log(msg: str):
    """写日志到 stdout + 文件"""
    ts = datetime.now(BEIJING_TZ).strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════
# 双源 F&G 融合引擎
# ═══════════════════════════════════════════

class DualFearGreedEngine:
    """双源恐惧贪婪指数融合 — CMC (主) + alternative.me (辅)"""

    def __init__(self):
        self._cached_cmc: Optional[dict] = None
        self._cached_alt: Optional[dict] = None
        self._last_cmc_fetch = 0.0
        self._last_alt_fetch = 0.0
        self._cmc_ttl = 600    # CMC 缓存 10 分钟
        self._alt_ttl = 3600   # alt.me 缓存 1 小时

    def _fetch_cmc(self) -> Optional[dict]:
        """从 CoinMarketCap SSR 抓取 F&G"""
        import re, requests
        now = time.time()
        if self._cached_cmc and (now - self._last_cmc_fetch) < self._cmc_ttl:
            return self._cached_cmc

        try:
            resp = requests.get(
                "https://coinmarketcap.com/charts/",
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=15,
            )
            resp.raise_for_status()
            m = re.search(
                r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text, re.DOTALL
            )
            if not m:
                return None

            data = json.loads(m.group(1))
            props = data.get("props", {}).get("pageProps", {})

            # 深度搜索
            def deep_search(obj, key, depth=0):
                if depth > 20 or obj is None: return None
                if isinstance(obj, dict):
                    if key in obj: return obj[key]
                    for k, v in obj.items():
                        r = deep_search(v, key, depth + 1)
                        if r is not None: return r
                elif isinstance(obj, list):
                    for item in obj[:100]:
                        r = deep_search(item, key, depth + 1)
                        if r is not None: return r
                return None

            fg = deep_search(props, "fearGreedIndexData")
            if not fg:
                return None

            cur = fg.get("currentIndex", {})
            result = {
                "value": int(cur.get("score", 50)),
                "classification": cur.get("name", "Neutral"),
                "updated_at": cur.get("updateTime", ""),
                "source": "CoinMarketCap",
            }
            self._cached_cmc = result
            self._last_cmc_fetch = now
            return result
        except Exception as e:
            _log(f"  ⚠️ CMC F&G 抓取失败: {e}")
            return self._cached_cmc  # 返回旧缓存

    def _fetch_alt(self) -> Optional[dict]:
        """从 alternative.me 获取 F&G"""
        import requests
        now = time.time()
        if self._cached_alt and (now - self._last_alt_fetch) < self._alt_ttl:
            return self._cached_alt

        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=30", timeout=15
            )
            data = resp.json().get("data", [])
            if not data:
                return None

            current = int(data[0]["value"])
            v1d = int(data[1]["value"]) if len(data) > 1 else current
            v7d = int(data[7]["value"]) if len(data) > 7 else current
            time_until = int(data[0].get("time_until_update", 0))

            # 分类
            if current <= 25: cls = "Extreme Fear"
            elif current <= 45: cls = "Fear"
            elif current <= 55: cls = "Neutral"
            elif current <= 75: cls = "Greed"
            else: cls = "Extreme Greed"

            result = {
                "value": current,
                "classification": cls,
                "value_1d_ago": v1d,
                "value_7d_ago": v7d,
                "change_1d": current - v1d,
                "change_7d": current - v7d,
                "time_until_update_sec": time_until,
                "source": "alternative.me",
            }
            self._cached_alt = result
            self._last_alt_fetch = now
            return result
        except Exception as e:
            _log(f"  ⚠️ alt.me F&G 失败: {e}")
            return self._cached_alt

    def get_combined(self) -> dict:
        """获取双源融合 F&G"""
        cmc = self._fetch_cmc()
        alt = self._fetch_alt()

        result = {
            "updated_at": datetime.now(BEIJING_TZ).isoformat(),
            "cmc": cmc,
            "alternative_me": alt,
        }

        # 计算加权融合值
        values = []
        weights = []
        if cmc:
            values.append(cmc["value"])
            weights.append(CMC_WEIGHT)
        if alt:
            values.append(alt["value"])
            weights.append(ALT_ME_WEIGHT)

        if values:
            total_w = sum(weights)
            combined_value = sum(v * w for v, w in zip(values, weights)) / total_w
            combined_value = round(combined_value)

            # 分类
            if combined_value <= 20: cls = "Extreme Fear"
            elif combined_value <= 40: cls = "Fear"
            elif combined_value <= 60: cls = "Neutral"
            elif combined_value <= 80: cls = "Greed"
            else: cls = "Extreme Greed"

            result["combined"] = {
                "value": combined_value,
                "classification": cls,
                "weight": f"{CMC_WEIGHT}*CMC + {ALT_ME_WEIGHT}*alt.me",
                "sources_available": len(values),
            }

            # 双源分歧度
            if len(values) == 2:
                divergence = abs(cmc["value"] - alt["value"])
                if divergence <= 5:
                    result["combined"]["divergence"] = "tight"      # 双源一致
                elif divergence <= 15:
                    result["combined"]["divergence"] = "moderate"   # 略有分歧
                else:
                    result["combined"]["divergence"] = "wide"       # 分歧较大
        else:
            result["combined"] = {
                "value": 50, "classification": "Unknown",
                "weight": "none", "sources_available": 0
            }

        return result


# ═══════════════════════════════════════════
# 市场环境判定器
# ═══════════════════════════════════════════

class MarketRegimeClassifier:
    """
    🏛️ 市场环境分类器

    根据 F&G + 新闻情绪, 判定当前市场环境, 输出:
      - regime: extreme_fear / fear / neutral / greed / extreme_greed
      - bias: short_only / short_preferred / neutral / long_preferred / long_only
      - conviction: 0-100 (环境信号强度)
      - leverage_multiplier: 杠杆乘数
    """

    @staticmethod
    def classify(fng_combined: dict, news_sentiment: dict) -> dict:
        fg_val = fng_combined.get("value", 50)
        fg_cls = fng_combined.get("classification", "Neutral")
        divergence = fng_combined.get("divergence", "moderate")

        # ── 1. F&G → 基础判定 ──
        if fg_val <= 20:
            regime = "extreme_fear"
            bias = "short_only"
            base_conviction = 85
        elif fg_val <= 40:
            regime = "fear"
            bias = "short_preferred"
            base_conviction = 65
        elif fg_val <= 60:
            regime = "neutral"
            bias = "neutral"
            base_conviction = 40
        elif fg_val <= 80:
            regime = "greed"
            bias = "long_preferred"
            base_conviction = 65
        else:
            regime = "extreme_greed"
            bias = "long_only"
            base_conviction = 85

        # ── 2. 新闻情绪修正 ──
        news_scores = []
        for sym, ns in news_sentiment.items():
            if isinstance(ns, dict):
                score = ns.get("score", 0)  # -100 ~ +100
                conf = min(1.0, ns.get("confidence", 50) / 100)
                news_scores.append(score * conf)

        news_avg = sum(news_scores) / len(news_scores) if news_scores else 0
        # news_avg: 负=看跌, 正=看涨

        # 新闻情绪与 F&G 共振/背离
        if news_avg < -20 and regime in ("extreme_fear", "fear"):
            resonance = "aligned"       # 新闻+F&G都恐慌 → 共振
            conviction_boost = 15
        elif news_avg > 20 and regime in ("extreme_greed", "greed"):
            resonance = "aligned"       # 新闻+F&G都贪婪 → 共振
            conviction_boost = 15
        elif news_avg < -20 and regime in ("extreme_greed", "greed"):
            resonance = "divergent"     # 新闻恐慌+F&G贪婪 → 背离
            conviction_boost = -20
            divergence = "wide"
        elif news_avg > 20 and regime in ("extreme_fear", "fear"):
            resonance = "divergent"     # 新闻乐观+F&G恐慌 → 背离(可能是底部信号)
            conviction_boost = -10
        else:
            resonance = "neutral"
            conviction_boost = 0

        conviction = max(10, min(100, base_conviction + conviction_boost))

        # ── 3. 杠杆乘数 ──
        if resonance == "aligned" and conviction >= 80:
            sentiment_strength = "strong_aligned"
        elif resonance == "aligned":
            sentiment_strength = "aligned"
        elif resonance == "divergent":
            sentiment_strength = "strongly_opposed"
        else:
            sentiment_strength = "neutral"

        leverage_mult = SENTIMENT_LEVERAGE_MAP.get(sentiment_strength, 1.0)

        # ── 4. 双源分歧 → 降低杠杆 ──
        if divergence == "wide":
            leverage_mult *= 0.8  # 双源分歧大, 减杠杆
            conviction = max(10, conviction - 10)

        return {
            "regime": regime,
            "bias": bias,
            "conviction": conviction,
            "resonance": resonance,
            "leverage_multiplier": round(leverage_mult, 2),
            "news_sentiment_avg": round(news_avg, 1),
            "fg_value": fg_val,
            "fg_classification": fg_cls,
            "divergence": divergence,
        }


# ═══════════════════════════════════════════
# 主守护进程
# ═══════════════════════════════════════════

class MarketSentimentDaemon:
    """市场情绪守护进程 — 独立运行, 写入共享状态"""

    def __init__(self):
        self.fg_engine = DualFearGreedEngine()
        self._last_fg_refresh = 0.0
        self._last_news_refresh = 0.0
        self._last_regime_refresh = 0.0
        self.running = True

        # 当前状态
        self.fear_greed: dict = {}
        self.news_sentiment: dict = {}
        self.market_regime: dict = {}

    def _refresh_fear_greed(self):
        """刷新双源 F&G"""
        self.fear_greed = self.fg_engine.get_combined()
        fg = self.fear_greed.get("combined", {})
        div = fg.get("divergence", "?")
        _log(f"😱 双源F&G: {fg.get('value')} ({fg.get('classification')}) "
             f"[CMC={self.fear_greed.get('cmc',{}).get('value','?')} "
             f"alt={self.fear_greed.get('alternative_me',{}).get('value','?')} "
             f"分歧={div}]")

    def _refresh_news_sentiment(self):
        """刷新 Gemini 新闻情绪 (BTC/ETH/SOL)"""
        try:
            from sentiment_analyzer import MarketSentimentAnalyzer
            analyzer = MarketSentimentAnalyzer()
            syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            for sym in syms:
                try:
                    report = analyzer.analyze(sym, market="crypto", force_refresh=True)
                    raw_score = report.sentiment_score  # 0~100
                    norm_score = (raw_score - 50) * 2   # -100~+100
                    drivers = [d.get("factor", "") for d in (report.key_drivers or [])[:3]]
                    self.news_sentiment[sym] = {
                        "score": round(norm_score),
                        "raw_score": raw_score,
                        "label": report.sentiment_label,
                        "confidence": report.data_count * 10 if report.data_count else 50,
                        "drivers": drivers,
                        "data_count": report.data_count,
                        "model": report.model_used,
                    }
                    emoji = "🔴" if raw_score < 30 else "🟡" if raw_score < 70 else "🟢"
                    _log(f"  📰 {emoji} {sym}: 情绪={norm_score:.0f} "
                         f"({report.sentiment_label}) | {report.data_count}条")
                except Exception as e:
                    _log(f"  ⚠️ {sym} 新闻分析跳过: {e}")
        except Exception as e:
            _log(f"⚠️ 新闻情绪引擎不可用: {e}")

    def _refresh_regime(self):
        """刷新市场环境判定"""
        fg_combined = self.fear_greed.get("combined", {})
        self.market_regime = MarketRegimeClassifier.classify(
            fg_combined, self.news_sentiment
        )
        regime = self.market_regime
        lev_emoji = "⚡" if regime.get("leverage_multiplier", 1.0) >= 1.5 else "🔒"
        _log(f"🏛️ 市场环境: {regime.get('regime','?')} | "
             f"偏向={regime.get('bias','?')} | "
             f"信度={regime.get('conviction','?')}% | "
             f"{lev_emoji}杠杆×{regime.get('leverage_multiplier',1.0)} | "
             f"共振={regime.get('resonance','?')}")

    def _write_state(self):
        """写入共享状态文件"""
        state = {
            "updated_at": datetime.now(BEIJING_TZ).isoformat(),
            "fear_greed": self.fear_greed,
            "news_sentiment": self.news_sentiment,
            "market_regime": self.market_regime,
        }
        try:
            # 原子写入
            tmp_path = STATE_FILE.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            tmp_path.rename(STATE_FILE)
        except Exception as e:
            _log(f"⚠️ 状态写入失败: {e}")

    def run_once(self):
        """单次运行"""
        _log("🏛️ 市场情绪分析 单次运行...")
        self._refresh_fear_greed()
        self._refresh_news_sentiment()
        self._refresh_regime()
        self._write_state()
        _log("✅ 完成!")
        self._print_summary()

    def run_loop(self):
        """持续运行守护循环"""
        _log("🏛️ 市场情绪守护进程 启动 v1.0")
        _log(f"   F&G刷新: {FNG_REFRESH_SEC}s | 新闻刷新: {NEWS_REFRESH_SEC}s | 环境刷新: {REGIME_REFRESH_SEC}s")
        _log(f"   共享状态: {STATE_FILE}")

        # 启动时立即运行一次
        self._refresh_fear_greed()
        self._refresh_news_sentiment()
        self._refresh_regime()
        self._write_state()

        while self.running:
            try:
                now = time.time()

                if now - self._last_fg_refresh >= FNG_REFRESH_SEC:
                    self._refresh_fear_greed()
                    self._last_fg_refresh = now

                if now - self._last_news_refresh >= NEWS_REFRESH_SEC:
                    self._refresh_news_sentiment()
                    self._last_news_refresh = now

                if now - self._last_regime_refresh >= REGIME_REFRESH_SEC:
                    self._refresh_regime()
                    self._last_regime_refresh = now

                # 每次刷新后写入状态
                if (now - self._last_fg_refresh < 2 or
                    now - self._last_regime_refresh < 2):
                    self._write_state()

                time.sleep(30)  # 每30秒检查一次

            except KeyboardInterrupt:
                _log("⏹️ 收到中断信号，退出...")
                break
            except Exception as e:
                _log(f"❌ 循环异常: {traceback.format_exc()}")
                time.sleep(60)

        _log("👋 市场情绪守护进程已停止")

    def _print_summary(self):
        """打印当前状态摘要"""
        fg = self.fear_greed.get("combined", {})
        regime = self.market_regime
        print("\n" + "=" * 60)
        print("🏛️  市场情绪快报")
        print("=" * 60)
        print(f"😱 双源F&G: {fg.get('value','?')}/100 ({fg.get('classification','?')})")
        cmc = self.fear_greed.get("cmc", {})
        alt = self.fear_greed.get("alternative_me", {})
        if cmc:
            print(f"   CMC: {cmc.get('value')} ({cmc.get('classification')})")
        if alt:
            print(f"   alt.me: {alt.get('value')} ({alt.get('classification')}) "
                  f"[1d:{alt.get('change_1d',0):+d}]")
        print()
        print(f"🏛️ 市场环境: {regime.get('regime','?')}")
        print(f"   📍 偏向: {regime.get('bias','?')}")
        print(f"   🎯 信度: {regime.get('conviction','?')}%")
        print(f"   ⚡ 杠杆乘数: ×{regime.get('leverage_multiplier',1.0)}")
        print(f"   🔄 共振: {regime.get('resonance','?')}")
        print()
        print("📰 新闻情绪:")
        for sym, ns in self.news_sentiment.items():
            emoji = "🔴" if ns.get("score", 0) < -20 else "🟡" if ns.get("score", 0) < 20 else "🟢"
            print(f"   {emoji} {sym}: {ns.get('score',0):.0f} "
                  f"({ns.get('label','?')}) | {ns.get('data_count',0)}条 | {ns.get('model','?')}")
        print("=" * 60)


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Yina 市场情绪守护进程 🏛️")
    parser.add_argument("--daemon", action="store_true", help="后台运行")
    parser.add_argument("--once", action="store_true", help="单次运行")
    args = parser.parse_args()

    daemon = MarketSentimentDaemon()

    if args.once:
        daemon.run_once()
        return

    if args.daemon:
        # 后台运行
        pid = os.fork()
        if pid > 0:
            print(f"🏛️ 市场情绪守护进程已后台启动, PID={pid}")
            print(f"   日志: {LOG_FILE}")
            print(f"   状态: {STATE_FILE}")
            return
        # 子进程
        os.setsid()
        daemon.run_loop()
    else:
        # 前台运行
        def _sig_handler(signum, frame):
            daemon.running = False
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
        daemon.run_loop()


if __name__ == "__main__":
    main()
