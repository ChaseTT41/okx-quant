#!/usr/bin/env python3
"""
🐾 AI 信源采集模块 — 每日从5大宝藏博主信源抓取最新动态
   Stratechery / Dwarkesh / 张小珺 / 晚点 / 海外独角兽
   集成到 hourly_analysis.py 日报推送
"""
import json, sys, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ = timezone(timedelta(hours=7))
UA = "Yina-AI-Source/1.0"
CACHE_FILE = Path(__file__).parent / "data" / "hourly" / "ai_news_cache.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════
# 五大信源定义
# ═══════════════════════════════════════════

SOURCES = {
    "stratechery": {
        "name": "Stratechery",
        "author": "Ben Thompson",
        "rss": "https://stratechery.com/feed/",
        "category": "科技战略分析",
        "focus": "AI产业/平台战略/科技巨头博弈",
        "emoji": "🏛️",
    },
    "dwarkesh": {
        "name": "Dwarkesh Podcast",
        "author": "Dwarkesh Patel",
        "rss": "https://www.dwarkesh.com/feed",
        "category": "AGI深度访谈",
        "focus": "AGI前沿/行业领袖对话/技术趋势",
        "emoji": "🎙️",
    },
    "zhangxiaojun": {
        "name": "张小珺商业访谈录",
        "author": "张小珺",
        "rss": None,  # 无公开RSS
        "category": "中国科技创投",
        "focus": "国内创业者采访/人脉广/资源顶级/大模型季报",
        "emoji": "🎤",
        "search_query": "张小珺 播客 AI 最新",
    },
    "latepost": {
        "name": "晚点 LatePost",
        "author": "晚点团队",
        "rss": None,
        "category": "深度科技媒体",
        "focus": "独家新闻/记者专业/深度报道",
        "emoji": "📰",
        "search_query": "晚点LatePost 科技 AI 最新",
    },
    "overseas_unicorn": {
        "name": "海外独角兽",
        "author": "海外独角兽",
        "rss": None,
        "category": "AI行业深度",
        "focus": "海外AI公司分析/独角兽追踪",
        "emoji": "🦄",
        "search_query": "海外独角兽 AI 科技 最新",
    },
}


def fetch_rss(url: str, max_items: int = 3) -> list:
    """抓取 RSS feed，返回最新 N 条标题+链接"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        items = []
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pubdate_el = item.find("pubDate")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            desc = desc_el.text if desc_el is not None else ""
            pubdate = pubdate_el.text if pubdate_el is not None else ""
            # 清理 HTML
            import re
            desc = re.sub(r"<[^>]+>", "", desc)[:200] if desc else ""
            title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title)
            items.append({"title": title.strip(), "link": link.strip(), "desc": desc.strip(), "pubdate": pubdate})
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        return [{"error": str(e)}]


def fetch_ai_news() -> dict:
    """采集所有AI信源最新动态"""
    results = {}
    print("🤖 采集AI信源动态...")

    for src_id, src in SOURCES.items():
        if src.get("rss"):
            items = fetch_rss(src["rss"], max_items=3)
            results[src_id] = {
                "name": src["name"],
                "author": src["author"],
                "emoji": src["emoji"],
                "category": src["category"],
                "focus": src["focus"],
                "items": items,
                "fetched_at": datetime.now(TZ).isoformat(),
            }
            if items and "error" not in items[0]:
                print(f"   ✅ {src['emoji']} {src['name']}: {len(items)}条")
                for i, item in enumerate(items):
                    print(f"      [{i+1}] {item['title'][:80]}")
            else:
                print(f"   ⚠️ {src['emoji']} {src['name']}: 抓取失败")
        else:
            results[src_id] = {
                "name": src["name"],
                "author": src["author"],
                "emoji": src["emoji"],
                "category": src["category"],
                "focus": src["focus"],
                "items": [],
                "note": "无公开RSS，请关注抖音/小宇宙/官网获取最新内容",
                "search_query": src.get("search_query", ""),
                "fetched_at": datetime.now(TZ).isoformat(),
            }
            print(f"   📌 {src['emoji']} {src['name']}: 无RSS (已记录关注点)")

    return results


def format_ai_news_section(news: dict) -> str:
    """格式化为日报推送的 Markdown 段落"""
    now = datetime.now(TZ)
    lines = []
    lines.append("### 🤖 AI信源 · 五大宝藏博主动态")
    lines.append("")
    lines.append(f"> *{now.strftime('%m/%d')} 自动采集 · 筛选标准：人脉广/更新快/观点独特经时间检验*")
    lines.append("")

    # 先展示有RSS的信源（有实时内容）
    for src_id in ["stratechery", "dwarkesh"]:
        src = news.get(src_id, {})
        if not src:
            continue
        items = src.get("items", [])
        emoji = src.get("emoji", "📌")
        lines.append(f"**{emoji} {src['name']}** — {src['author']} | {src.get('category', '')}")
        lines.append(f"> 📍 {src.get('focus', '')}")
        if items and "error" not in items[0]:
            for i, item in enumerate(items[:3], 1):
                title = item.get("title", "")[:100]
                link = item.get("link", "")
                date_str = item.get("pubdate", "")[:25]
                if date_str:
                    lines.append(f"> {i}. [{title}]({link}) — {date_str}")
                else:
                    lines.append(f"> {i}. [{title}]({link})")
        else:
            lines.append(f"> *(暂无最新内容)*")
        lines.append("")

    # 中文信源（无RSS，标注关注点+搜索入口）
    lines.append("**📡 中文深度信源** (小宇宙/抖音/官网订阅)")
    lines.append("")
    for src_id in ["zhangxiaojun", "latepost", "overseas_unicorn"]:
        src = news.get(src_id, {})
        if not src:
            continue
        emoji = src.get("emoji", "📌")
        lines.append(
            f"> {emoji} **{src['name']}** — {src.get('category', '')} | {src.get('focus', '')}"
        )

    lines.append("")
    lines.append("> 💡 *每日AI信源由 Yina 自动采集 · 建议订阅原播客/公众号获取完整内容*")
    return "\n".join(lines)


def load_cached_news() -> dict:
    """加载缓存的AI新闻"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
                cache_time = cache.get("fetched_at", "")
                if cache_time:
                    cache_dt = datetime.fromisoformat(cache_time)
                    age = (datetime.now(TZ) - cache_dt).total_seconds()
                    if age < 3600:  # 1小时内有效
                        return cache
        except:
            pass
    return {}


def save_cached_news(news: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=2)


def get_ai_news_section(force_refresh: bool = False) -> str:
    """获取AI信源段落（带缓存）"""
    news = None
    if not force_refresh:
        news = load_cached_news()

    if not news:
        news = fetch_ai_news()
        save_cached_news(news)

    return format_ai_news_section(news)


if __name__ == "__main__":
    # 单独运行：抓取并打印AI信源内容
    print("🐾 Yina AI信源采集测试")
    print()
    news = fetch_ai_news()
    save_cached_news(news)
    print()
    section = format_ai_news_section(news)
    print("=" * 60)
    print(section)
    print("=" * 60)
    print(f"✅ 已缓存到 {CACHE_FILE}")
