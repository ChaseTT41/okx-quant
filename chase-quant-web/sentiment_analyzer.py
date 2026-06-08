"""
Yina 市场情绪分析引擎 🎭
========================
基于 Gemini API + 多源数据采集的市场情绪深度分析

核心理念:
  不依赖单一数据源 — 融合国内外新闻、社交媒体、链上数据等多维信号
  Gemini 作为 NLP 大脑，对采集的文本做情绪打分、关键驱动因素提取、平台对比

数据源:
  国际: Google News, Yahoo Finance, CoinDesk, CoinTelegraph, Reddit
  国内: 东方财富, 雪球, 微博热搜, 知乎 (通过 RSS + 爬虫)

使用:
  from sentiment_analyzer import MarketSentimentAnalyzer
  analyzer = MarketSentimentAnalyzer()
  report = analyzer.analyze("BTC/USDT")  # 加密货币
  report = analyzer.analyze("中芯国际")    # 港股
  report = analyzer.analyze("沪深300")    # 指数
"""

from __future__ import annotations
import json
import os
import re
import time
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import ssl
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup
import feedparser

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data" / "sentiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BEIJING_TZ = timezone(timedelta(hours=8))

# Gemini API
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash-lite"  # 免费层, 15 RPM

# 加载 API Key
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not _GEMINI_API_KEY:
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    _GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

# 也检查 vision tools 的 .env
if not _GEMINI_API_KEY:
    _vision_env = Path.home() / "Mac-For-Claude" / "free-vision-image-tools" / ".env"
    if _vision_env.exists():
        with open(_vision_env) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    _GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

# 缓存有效期 (秒)
CACHE_TTL = 3600  # 1小时

# 请求超时
REQUEST_TIMEOUT = 15


# ═══════════════════════════════════════════
# 数据采集器
# ═══════════════════════════════════════════

class NewsCollector:
    """多源新闻采集器 — RSS + 爬虫"""

    # 各标的的 RSS 源配置
    RSS_FEEDS = {
        "crypto": [
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("CoinTelegraph", "https://cointelegraph.com/rss"),
            ("CryptoSlate", "https://cryptoslate.com/feed/"),
            ("Decrypt", "https://decrypt.co/feed"),
        ],
        "stock_cn": [
            ("东方财富", "https://rss.sina.com.cn/finance/stock.xml"),
            ("证券时报", "https://www.stcn.com/rss.xml"),
        ],
        "global": [
            ("Google News Business", "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"),
            ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
            ("Reuters", "https://www.rss-bridge.org/bridge01/?action=display&bridge=Reuters&feed=home%2FtopNews&format=Atom"),
        ],
    }

    @staticmethod
    def fetch_rss(url: str, max_items: int = 10) -> List[dict]:
        """抓取 RSS 源"""
        try:
            # 绕过 SSL 验证 (某些 RSS 源证书有问题)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries[:max_items]:
                items.append({
                    "title": entry.get("title", ""),
                    "summary": re.sub(r'<[^>]+>', '', entry.get("summary", "")),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
            return items
        except Exception as e:
            return []

    @staticmethod
    def fetch_url_text(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
        """抓取网页文本"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # 移除 script/style
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            # 清理空白
            text = re.sub(r'\s+', ' ', text)
            return text[:5000]  # 限制长度
        except Exception:
            return ""

    @staticmethod
    def search_google_news(query: str, max_items: int = 10) -> List[dict]:
        """通过 Google News RSS 搜索"""
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        items = NewsCollector.fetch_rss(url, max_items)

        # 也搜英文
        url_en = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        items_en = NewsCollector.fetch_rss(url_en, max_items)

        # 合并去重
        seen = set()
        merged = []
        for item in items + items_en:
            if item["title"] not in seen:
                seen.add(item["title"])
                merged.append(item)
        return merged[:max_items]

    @classmethod
    def collect_all(cls, symbol: str, market: str = "crypto") -> Dict[str, List[dict]]:
        """
        全源采集

        Args:
            symbol: 标的符号 (BTC/USDT, ETH/USDT, 中芯国际, etc.)
            market: 市场类型 (crypto, stock_cn, stock_us)

        Returns:
            {"google_news": [...], "rss_feeds": [...], "source_name": [...]}
        """
        result = {}

        # 1. Google News 搜索
        search_query = symbol.replace("/USDT", "").replace("/USD", "")
        if market == "crypto":
            search_query = f"{search_query} cryptocurrency"
        elif market == "stock_cn":
            search_query = f"{symbol} 股票"
        result["google_news"] = cls.search_google_news(search_query, max_items=15)

        # 2. 根据市场选择 RSS 源
        feed_categories = ["global"]
        if market == "crypto":
            feed_categories.append("crypto")
        elif market in ("stock_cn", "stock_us"):
            feed_categories.append("stock_cn")

        for cat in feed_categories:
            for source_name, url in cls.RSS_FEEDS.get(cat, []):
                items = cls.fetch_rss(url, max_items=8)
                if items:
                    result[source_name] = items

        return result


# ═══════════════════════════════════════════
# Gemini 情绪分析引擎
# ═══════════════════════════════════════════

@dataclass
class SentimentReport:
    """市场情绪分析报告"""
    symbol: str
    market: str
    analyzed_at: str

    # 概述
    current_price_range: str = ""
    recent_performance: str = ""

    # 核心指标
    sentiment_score: int = 50  # 0-100, 0=极度看跌, 100=极度看涨
    sentiment_label: str = "中性"  # 极度看跌/看跌/中性偏跌/中性/中性偏涨/看涨/极度看涨
    sentiment_summary: str = ""

    # 关键驱动因素 (Top 5-7)
    key_drivers: List[dict] = field(default_factory=list)
    # [{"factor": "...", "impact": "positive/negative/neutral", "strength": 1-10, "source": "..."}]

    # 平台情绪对比
    platform_sentiment: Dict[str, dict] = field(default_factory=dict)
    # {"国际/X": {"score": 65, "summary": "..."}, "国内/雪球": {...}, ...}

    # 风险与机会
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # ── 周期分层分析 (v2 升级) ──
    # 宏观因素按影响周期分层
    macro_cycle: Dict[str, dict] = field(default_factory=dict)
    # {
    #   "short_term": {  # 1-4周
    #     "label": "短期 (1-4周)",
    #     "score": 45,
    #     "drivers": [{"factor": "...", "impact": "negative", "detail": "..."}],
    #     "summary": "..."
    #   },
    #   "mid_term": { ... },   # 1-3月
    #   "long_term": { ... }   # 3-12月+
    # }
    cycle_position: str = ""  # 当前所处周期位置: 恐慌底部/修复初期/上升中期/顶部过热/下跌初期/下跌中继
    cycle_confidence: int = 50  # 周期判断置信度 0-100

    # 情绪趋势
    sentiment_trend: str = ""  # 改善/恶化/稳定
    divergence_note: str = ""  # 与价格背离情况

    # 结论
    conclusion: str = ""
    trading_suggestion: str = ""  # 仅供参考

    # 元数据
    data_sources: List[str] = field(default_factory=list)
    data_count: int = 0
    model_used: str = GEMINI_MODEL
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """转为 Markdown 格式 (企微推送用)"""
        lines = [
            f"## 🎭 市场情绪分析: {self.symbol}",
            f"> 📅 {self.analyzed_at[:16]}",
            "",
            f"**整体情绪得分**: {self.sentiment_score}/100 ({self.sentiment_label})",
            f"> {self.sentiment_summary}",
            "",
            "### 🔑 关键驱动因素",
        ]
        for i, d in enumerate(self.key_drivers[:7], 1):
            emoji = "🟢" if d.get("impact") == "positive" else "🔴" if d.get("impact") == "negative" else "⚪"
            lines.append(f"{i}. {emoji} **{d.get('factor', '')}** (强度: {d.get('strength', 5)}/10)")
            if d.get("detail"):
                lines.append(f"   {d['detail']}")

        lines.append("")
        lines.append("### 📊 平台情绪对比")
        for platform, data in self.platform_sentiment.items():
            score = data.get("score", 50)
            bar = "🟩" * (score // 20) + "⬜" * (5 - score // 20)
            lines.append(f"- {platform}: {bar} {score}/100")

        if self.risks:
            lines.append("")
            lines.append("### ⚠️ 风险")
            for r in self.risks:
                lines.append(f"- 🔴 {r}")

        if self.opportunities:
            lines.append("")
            lines.append("### 💡 机会")
            for o in self.opportunities:
                lines.append(f"- 🟢 {o}")

        lines.append("")
        lines.append(f"### 🎯 结论")
        lines.append(self.conclusion)
        lines.append(f"> *{self.trading_suggestion}*")
        lines.append("")
        lines.append(f"📊 数据来源: {', '.join(self.data_sources[:5])}")
        lines.append(f"🧠 模型: {self.model_used} | 样本: {self.data_count}条")

        return "\n".join(lines)


class GeminiSentimentEngine:
    """使用 Gemini API 做情绪分析的引擎"""

    SENTIMENT_PROMPT = """你是一个专业的金融市场情绪分析师，兼具宏观经济学家视角。请根据以下采集的最新新闻标题和摘要，对 {symbol} 进行深度市场情绪分析，并严格区分宏观因素的**影响周期**。

**采集数据** (共 {total_items} 条):
{collected_text}

**分析要求**:
1. 综合所有标题和摘要，判断整体市场情绪偏向
2. 识别关键驱动因素（政策、技术、资金、基本面、消息面等）
3. **重要: 按影响周期分层** — 每个驱动因素必须明确标注其影响的时间维度:
   - 🔴 **短期 (1-4周)**: 事件驱动、情绪冲击、技术面信号、资金异动、监管突击
   - 🟡 **中期 (1-3月)**: 货币政策转向、财报季、行业周期、地缘政治演变、技术升级
   - 🟢 **长期 (3-12月+)**: 结构性问题、人口趋势、技术革命、全球化/去全球化、能源转型
4. 区分国际和国内平台的情绪差异
5. 发现潜在风险和机会
6. 判断当前市场所处周期位置

**请严格输出以下 JSON 格式** (不要输出其他内容):
```json
{{
  "sentiment_score": <0-100整数, 0=极度看跌, 50=中性, 100=极度看涨>,
  "sentiment_label": "<极度看跌/看跌/中性偏跌/中性/中性偏涨/看涨/极度看涨>",
  "sentiment_summary": "<一句话总结当前市场情绪，30字以内>",
  "current_price_range": "<当前价格区间描述>",
  "recent_performance": "<最近表现描述>",
  "key_drivers": [
    {{"factor": "<因素名>", "impact": "positive/negative/neutral", "strength": <1-10>, "cycle": "short/mid/long", "detail": "<一句话说明,需提及影响时间跨度>", "source": "<来源>"}}
  ],
  "platform_sentiment": {{
    "国际/英文媒体": {{"score": <0-100>, "summary": "<15字>"}},
    "国际/社交媒体": {{"score": <0-100>, "summary": "<15字>"}},
    "国内/中文媒体": {{"score": <0-100>, "summary": "<15字>"}},
    "国内/投资社区": {{"score": <0-100>, "summary": "<15字>"}}
  }},
  "risks": ["<风险1>", "<风险2>", "..."],
  "opportunities": ["<机会1>", "<机会2>", "..."],
  "sentiment_trend": "<改善/恶化/稳定>",
  "divergence_note": "<情绪与价格背离情况，无则填'无明显背离'>",

  "macro_cycle": {{
    "short_term": {{
      "score": <0-100, 短期情绪得分>,
      "summary": "<短期(1-4周)展望，25字以内>",
      "drivers": [{{"factor": "<因素>", "impact": "positive/negative/neutral", "detail": "<说明>"}}]
    }},
    "mid_term": {{
      "score": <0-100, 中期情绪得分>,
      "summary": "<中期(1-3月)展望，25字以内>",
      "drivers": [{{"factor": "<因素>", "impact": "positive/negative/neutral", "detail": "<说明>"}}]
    }},
    "long_term": {{
      "score": <0-100, 长期情绪得分>,
      "summary": "<长期(3-12月+)展望，25字以内>",
      "drivers": [{{"factor": "<因素>", "impact": "positive/negative/neutral", "detail": "<说明>"}}]
    }}
  }},
  "cycle_position": "<恐慌底部/修复初期/上升中期/顶部过热/下跌初期/下跌中继/横盘整理>",
  "cycle_confidence": <0-100, 周期位置判断置信度>,

  "conclusion": "<150字以内综合结论，必须包含短中长期三层判断>",
  "trading_suggestion": "<50字以内交易思路，必须注明'仅供参考，非投资建议'>"
}}
```

**重要**:
- 必须基于提供的采集数据进行分析，不要凭空编造
- **短期≠长期**: 例如"非农超预期"是短期利空（影响1-2周），"AI产业革命"是长期利好（影响6-12月+），两者必须分层
- 如果某个周期缺乏数据支撑，score可填50并标注"数据不足"
- 国内来源重点分析政策面和资金面情绪
- 周期位置判断要保守，置信度低于40时标注"横盘整理"
- 保持客观中立专业"""

    def __init__(self):
        self.api_key = _GEMINI_API_KEY

    def analyze(self, symbol: str, collected_data: Dict[str, List[dict]],
                market: str = "crypto") -> SentimentReport:
        """
        调用 Gemini API 分析情绪

        Args:
            symbol: 标的
            collected_data: NewsCollector.collect_all() 的结果
            market: 市场类型

        Returns:
            SentimentReport
        """
        # 构建采集文本
        all_items = []
        sources = list(collected_data.keys())

        for source_name, items in collected_data.items():
            for item in items:
                title = item.get("title", "")
                summary = item.get("summary", "")[:200]
                if title:
                    all_items.append(f"[{source_name}] {title}")
                    if summary and summary != title:
                        all_items.append(f"  → {summary}")

        # 限制总长度 (Gemini 上下文限制)
        max_chars = 15000
        text_block = ""
        for line in all_items:
            if len(text_block) + len(line) + 1 > max_chars:
                break
            text_block += line + "\n"

        total_items = len(all_items)

        if total_items == 0:
            return self._empty_report(symbol, market, sources)

        # 构建 prompt
        prompt = self.SENTIMENT_PROMPT.format(
            symbol=symbol,
            total_items=total_items,
            collected_text=text_block,
        )

        # 调用 Gemini API
        try:
            raw_response = self._call_gemini(prompt)
            report = self._parse_response(raw_response, symbol, market, sources, total_items)
            return report
        except Exception as e:
            # 降级: 返回基础报告
            report = self._empty_report(symbol, market, sources)
            report.sentiment_summary = f"分析暂时不可用: {str(e)[:50]}"
            return report

    def _call_gemini(self, prompt: str) -> str:
        """调用 Gemini REST API"""
        if not self.api_key:
            raise ValueError("未配置 GEMINI_API_KEY")

        url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,  # 低温度 → 更确定性的输出
                "topP": 0.9,
                "maxOutputTokens": 2048,
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not text:
                    raise ValueError("Gemini 返回空文本")
                return text
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"Gemini API HTTP {e.code}: {error_body[:200]}")
        except Exception as e:
            raise RuntimeError(f"Gemini API 调用失败: {e}")

    def _parse_response(self, raw: str, symbol: str, market: str,
                        sources: List[str], total_items: int) -> SentimentReport:
        """解析 Gemini 返回的 JSON"""
        try:
            # 提取 JSON 块
            json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接找 JSON 对象
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = raw

            data = json.loads(json_str)

            report = SentimentReport(
                symbol=symbol,
                market=market,
                analyzed_at=datetime.now(BEIJING_TZ).isoformat(),
                sentiment_score=int(data.get("sentiment_score", 50)),
                sentiment_label=data.get("sentiment_label", "中性"),
                sentiment_summary=data.get("sentiment_summary", ""),
                current_price_range=data.get("current_price_range", ""),
                recent_performance=data.get("recent_performance", ""),
                key_drivers=data.get("key_drivers", []),
                platform_sentiment=data.get("platform_sentiment", {}),
                risks=data.get("risks", []),
                opportunities=data.get("opportunities", []),
                sentiment_trend=data.get("sentiment_trend", "稳定"),
                divergence_note=data.get("divergence_note", "无明显背离"),
                macro_cycle=data.get("macro_cycle", {}),
                cycle_position=data.get("cycle_position", "横盘整理"),
                cycle_confidence=int(data.get("cycle_confidence", 50)),
                conclusion=data.get("conclusion", ""),
                trading_suggestion=data.get("trading_suggestion", "仅供参考，非投资建议"),
                data_sources=sources,
                data_count=total_items,
            )

            # 限制得分范围
            report.sentiment_score = max(0, min(100, report.sentiment_score))
            return report

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 解析失败: 用原始文本构建基础报告
            report = SentimentReport(
                symbol=symbol,
                market=market,
                analyzed_at=datetime.now(BEIJING_TZ).isoformat(),
                sentiment_score=50,
                sentiment_label="中性",
                sentiment_summary=raw[:100].replace("\n", " "),
                conclusion=raw[:200],
                trading_suggestion="数据解析异常，建议人工复核 (仅供参考)",
                data_sources=sources,
                data_count=total_items,
            )
            return report

    def _empty_report(self, symbol: str, market: str, sources: List[str]) -> SentimentReport:
        """空报告 (无数据时)"""
        return SentimentReport(
            symbol=symbol,
            market=market,
            analyzed_at=datetime.now(BEIJING_TZ).isoformat(),
            sentiment_score=50,
            sentiment_label="数据不足",
            sentiment_summary="未能采集到足够的新闻数据，建议稍后重试",
            conclusion="当前无足够数据生成情绪分析。可能原因：网络问题、RSS源暂时不可用、或标的名称不匹配。",
            trading_suggestion="数据不足，无法给出建议 (仅供参考，非投资建议)",
            data_sources=sources,
            data_count=0,
        )


# ═══════════════════════════════════════════
# 主分析器
# ═══════════════════════════════════════════

class MarketSentimentAnalyzer:
    """市场情绪主分析器 — 采集 + 分析 + 缓存"""

    # 默认分析标的
    DEFAULT_SYMBOLS = [
        ("BTC/USDT", "crypto"),
        ("ETH/USDT", "crypto"),
        ("SOL/USDT", "crypto"),
    ]

    def __init__(self):
        self.collector = NewsCollector()
        self.engine = GeminiSentimentEngine()

    def _cache_key(self, symbol: str) -> str:
        return hashlib.md5(symbol.encode()).hexdigest()[:12]

    def _cache_path(self, symbol: str) -> Path:
        return DATA_DIR / f"{self._cache_key(symbol)}.json"

    def _load_cache(self, symbol: str) -> Optional[SentimentReport]:
        """加载缓存 (1小时内有效)"""
        path = self._cache_path(symbol)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            cache_time = datetime.fromisoformat(data.get("analyzed_at", ""))
            age = (datetime.now(BEIJING_TZ) - cache_time).total_seconds()

            if age < CACHE_TTL:
                report = SentimentReport(**{k: v for k, v in data.items() if k in SentimentReport.__dataclass_fields__})
                report.cache_hit = True
                return report
        except Exception:
            pass

        return None

    def _save_cache(self, report: SentimentReport):
        """保存缓存"""
        try:
            with open(self._cache_path(report.symbol), "w") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def analyze(self, symbol: str, market: str = None,
                force_refresh: bool = False) -> SentimentReport:
        """
        分析市场情绪

        Args:
            symbol: 标的 (BTC/USDT, ETH/USDT, 中芯国际, etc.)
            market: 市场 (crypto/stock_cn/stock_us), None=自动判断
            force_refresh: 强制刷新 (忽略缓存)

        Returns:
            SentimentReport
        """
        # 自动判断市场
        if market is None:
            if "/USDT" in symbol or "/USD" in symbol:
                market = "crypto"
            elif any(k in symbol for k in ["上证", "深证", "沪深", "创业板", "科创板"]):
                market = "stock_cn"
            elif re.search(r'[A-Z]{1,5}\.[A-Z]{2}', symbol):
                market = "stock_us"
            else:
                market = "stock_cn"  # 默认国内

        # 检查缓存
        if not force_refresh:
            cached = self._load_cache(symbol)
            if cached:
                return cached

        # 采集数据
        collected = self.collector.collect_all(symbol, market)

        # 分析
        report = self.engine.analyze(symbol, collected, market)

        # 保存缓存
        self._save_cache(report)

        return report

    def analyze_batch(self, symbols: List[Tuple[str, str]] = None,
                      force_refresh: bool = False) -> List[SentimentReport]:
        """批量分析"""
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS

        reports = []
        for symbol, market in symbols:
            try:
                report = self.analyze(symbol, market, force_refresh)
                reports.append(report)
            except Exception as e:
                # 失败时创建空报告
                reports.append(SentimentReport(
                    symbol=symbol, market=market,
                    analyzed_at=datetime.now(BEIJING_TZ).isoformat(),
                    sentiment_summary=f"分析失败: {str(e)[:50]}",
                ))
        return reports

    def get_market_overview(self) -> dict:
        """获取整体市场情绪概览 (用于仪表板)"""
        reports = self.analyze_batch()

        total_score = 0
        n = 0
        overview = {
            "analyzed_at": datetime.now(BEIJING_TZ).isoformat(),
            "overall_sentiment_score": 50,
            "overall_label": "中性",
            "symbols": [],
            "top_risks": [],
            "top_opportunities": [],
            "dominant_drivers": [],
        }

        all_risks = set()
        all_opps = set()
        driver_counts = {}

        for r in reports:
            total_score += r.sentiment_score
            n += 1

            overview["symbols"].append({
                "symbol": r.symbol,
                "score": r.sentiment_score,
                "label": r.sentiment_label,
                "summary": r.sentiment_summary,
            })

            for risk in r.risks[:2]:
                all_risks.add(risk)
            for opp in r.opportunities[:2]:
                all_opps.add(opp)
            for driver in r.key_drivers[:3]:
                factor = driver.get("factor", "")
                if factor:
                    driver_counts[factor] = driver_counts.get(factor, 0) + 1

        if n > 0:
            overview["overall_sentiment_score"] = total_score // n

        # 标签
        score = overview["overall_sentiment_score"]
        if score >= 80:
            overview["overall_label"] = "极度看涨 🔥"
        elif score >= 65:
            overview["overall_label"] = "看涨 📈"
        elif score >= 55:
            overview["overall_label"] = "中性偏涨 ↗️"
        elif score >= 45:
            overview["overall_label"] = "中性 ➡️"
        elif score >= 35:
            overview["overall_label"] = "中性偏跌 ↘️"
        elif score >= 20:
            overview["overall_label"] = "看跌 📉"
        else:
            overview["overall_label"] = "极度看跌 🧊"

        overview["top_risks"] = list(all_risks)[:5]
        overview["top_opportunities"] = list(all_opps)[:5]
        overview["dominant_drivers"] = sorted(driver_counts.items(), key=lambda x: -x[1])[:5]

        return overview


# ═══════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yina 市场情绪分析 🎭")
    parser.add_argument("symbol", nargs="?", default="BTC/USDT",
                        help="分析标的 (默认: BTC/USDT)")
    parser.add_argument("--market", "-m", choices=["crypto", "stock_cn", "stock_us"],
                        help="市场类型 (默认自动判断)")
    parser.add_argument("--batch", action="store_true",
                        help="批量分析默认标的")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制刷新 (忽略缓存)")
    parser.add_argument("--overview", action="store_true",
                        help="输出市场概览 JSON")
    parser.add_argument("--json", action="store_true",
                        help="输出 JSON 格式")

    args = parser.parse_args()

    analyzer = MarketSentimentAnalyzer()

    if args.overview:
        overview = analyzer.get_market_overview()
        print(json.dumps(overview, ensure_ascii=False, indent=2))
    elif args.batch:
        reports = analyzer.analyze_batch(force_refresh=args.force)
        for r in reports:
            if args.json:
                print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(r.to_markdown())
                print("\n" + "=" * 60 + "\n")
    else:
        report = analyzer.analyze(args.symbol, args.market, args.force)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_markdown())
