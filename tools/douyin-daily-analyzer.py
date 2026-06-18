#!/usr/bin/env python3
"""
抖音每日智能分析器 v1 — Gemini AI 分析收藏+喜欢，识别 Yina 可执行任务
用法: python3 douyin-daily-analyzer.py [--output /tmp/douyin_report.json]
"""
import os, sys, json, re, time
from datetime import datetime
from pathlib import Path

INPUT_FILE = '/tmp/douyin_fav_result.json'
OUTPUT_FILE = '/tmp/douyin_report.json'
ENV_FILE = os.path.expanduser('~/Mac-For-Claude/free-vision-image-tools/.env')
MAX_TITLES = 80  # 最多送多少条标题去分析

# ── 加载 .env ──
def load_env(path):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

env = load_env(ENV_FILE)
GEMINI_API_KEY = env.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
OPENROUTER_API_KEY = env.get('OPENROUTER_API_KEY', os.getenv('OPENROUTER_API_KEY', ''))

# ── LLM API 调用（多后端+多模型自动切换）──

# 免费模型列表，按优先级排列
FREE_MODELS = [
    "qwen/qwen3-coder:free",              # Qwen3 Coder 480B — 强推理
    "nvidia/nemotron-3-super-120b-a12b:free",  # NVIDIA Super 120B
    "google/gemma-4-31b-it:free",          # Google Gemma 4 31B
    "meta-llama/llama-3.3-70b-instruct:free",  # Llama 3.3 70B
    "qwen/qwen3-next-80b-a3b-instruct:free",   # Qwen3 Next 80B
]

def call_openrouter(prompt: str, max_tokens: int = 4096) -> str:
    """通过 OpenRouter 调用免费模型，支持自动切换+重试"""
    import urllib.request
    import urllib.error

    for model in FREE_MODELS:
        for attempt in range(3):  # 每个模型最多重试3次
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                body = json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                }).encode('utf-8')

                req = urllib.request.Request(url, data=body, headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                    'HTTP-Referer': 'https://github.com/yina-tools',
                    'X-Title': 'Yina Douyin Analyzer',
                })

                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read())
                    content = data['choices'][0]['message']['content']
                    print(f"   ✅ 模型 {model} 返回 {len(content)} 字符", flush=True)
                    return content

            except urllib.error.HTTPError as e:
                err_body = e.read().decode()[:300]
                if e.code == 429 and attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"   ⏳ 模型 {model} 限流，{wait}秒后重试...", flush=True)
                    time.sleep(wait)
                    continue
                else:
                    print(f"   ⚠️ 模型 {model} 失败 ({e.code})，切换下一个...", flush=True)
                    break  # 换下一个模型
            except Exception as e:
                print(f"   ⚠️ 模型 {model} 异常: {e}，切换下一个...", flush=True)
                break

    raise RuntimeError(f"所有 {len(FREE_MODELS)} 个免费模型均不可用")

def call_gemini_direct(prompt: str, max_tokens: int = 4096) -> str:
    """直接调用 Gemini REST API"""
    import urllib.request
    import urllib.error

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        }
    }).encode('utf-8')

    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data['candidates'][0]['content']['parts'][0]['text']
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()[:500]
        print(f"❌ Gemini 错误 {e.code}: {err_body}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"❌ Gemini 调用失败: {e}", file=sys.stderr)
        raise

def call_llm(prompt: str, max_tokens: int = 4096) -> str:
    """智能选择后端：OpenRouter → Gemini Direct"""
    # 优先 OpenRouter（走 Gemini Flash，免费额度独立）
    if OPENROUTER_API_KEY:
        try:
            print("   📡 使用 OpenRouter (Gemini Flash)...", flush=True)
            return call_openrouter(prompt, max_tokens)
        except Exception as e:
            print(f"   ⚠️ OpenRouter 失败: {e}，尝试 Gemini Direct...", flush=True)

    # 回退到 Gemini Direct
    if GEMINI_API_KEY:
        print("   📡 使用 Gemini Direct...", flush=True)
        return call_gemini_direct(prompt, max_tokens)

    raise RuntimeError("没有可用的 LLM 后端！请设置 OPENROUTER_API_KEY 或 GEMINI_API_KEY")


# ── 构建分析 Prompt ──
def build_prompt(favorites, likes):
    """构建给 AI 的分析 prompt"""
    lines = []
    lines.append("你是 Yina，Chase哥的 AI 小助手（犬系人格）。Chase哥 是一个开发者+量化交易员+自媒体创作者。")
    lines.append("")
    lines.append("以下是他抖音收藏和喜欢的视频标题列表。请分析这些内容，找出 Yina 可以落地执行的事情。")
    lines.append("")

    if favorites:
        lines.append(f"## 📌 收藏 ({len(favorites)}条)")
        lines.append("")
        for i, item in enumerate(favorites[:MAX_TITLES//2]):
            title = item.get('title', '(无标题)')[:150]
            lines.append(f"{i+1}. {title}")

    if likes:
        lines.append("")
        lines.append(f"## ❤️ 喜欢 ({len(likes)}条)")
        lines.append("")
        for i, item in enumerate(likes[:MAX_TITLES//2]):
            title = item.get('title', '(无标题)')[:150]
            lines.append(f"{i+1}. {title}")

    lines.append("")
    lines.append("---")
    lines.append("请按以下 JSON 格式输出（只输出 JSON，不要其他文字）：")
    lines.append("""```json
{
  "summary": "一句话总结Chase哥最近的兴趣焦点",
  "actionable_items": [
    {
      "priority": "high|medium|low",
      "category": "工具部署|教程跟进|投资线索|代码技巧|内容创作|研究调研|自动化",
      "title": "任务简短标题",
      "what_to_do": "Yina 具体该做什么（1-2句话）",
      "source_video": "相关的视频标题"
    }
  ],
  "interesting_patterns": ["观察到的兴趣趋势1", "趋势2"],
  "suggested_next_action": "建议 Chase哥 优先关注的一件事"
}
```""")
    lines.append("")
    lines.append("规则：")
    lines.append("1. 只输出真正可执行的，不要泛泛而谈")
    lines.append("2. 「工具部署」= 提到了某个具体工具/平台/网站，Yina可以帮忙搭建或注册")
    lines.append("3. 「教程跟进」= 教学类内容，Yina可以提取要点或实际操作一遍")
    lines.append("4. 「投资线索」= 跟股票/加密货币/SpaceX/半导体相关的信号")
    lines.append("5. 「代码技巧」= 编程/量化/AI开发相关，Yina可以实现或整合到现有系统")
    lines.append("6. 「内容创作」= 值得Chase哥做视频的选题方向")
    lines.append("7. 「自动化」= 可以写脚本自动化的事情")
    lines.append("8. 去重：相似主题的视频合并成一个 action item")
    lines.append("9. 输出 3-8 个 actionable items，宁缺毋滥")
    lines.append("10. 输出必须是合法的 JSON")

    return '\n'.join(lines)


# ── 解析 AI 输出 ──
def parse_response(text: str) -> dict:
    """从 AI 回复中提取 JSON"""
    # 尝试提取 ```json ... ``` 块
    m = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if m:
        text = m.group(1)

    # 尝试提取 { ... }
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        text = m.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        text = text.replace('\n', ' ').replace('\r', '')
        try:
            return json.loads(text)
        except:
            return {"raw_response": text, "parse_error": True}


# ── 格式化输出 ──
def format_report(report: dict) -> str:
    """将分析结果格式化为可读文本"""
    lines = []
    lines.append("=" * 60)
    lines.append("🐾 Yina 抖音每日智能分析")
    lines.append("=" * 60)
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    if report.get('summary'):
        lines.append(f"📊 焦点: {report['summary']}")
        lines.append("")

    items = report.get('actionable_items', [])
    if items:
        lines.append(f"🎯 可执行任务 ({len(items)}项):")
        lines.append("-" * 40)

        emoji_map = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        for i, item in enumerate(items):
            pri = item.get('priority', 'medium')
            cat = item.get('category', '其他')
            title = item.get('title', '未命名')
            todo = item.get('what_to_do', '')
            src = item.get('source_video', '')

            lines.append(f"\n{emoji_map.get(pri, '⚪')} [{cat}] {title}")
            lines.append(f"   📋 {todo}")
            if src:
                lines.append(f"   🎬 来源: {src[:80]}")

    patterns = report.get('interesting_patterns', [])
    if patterns:
        lines.append(f"\n\n📈 兴趣趋势:")
        for p in patterns:
            lines.append(f"   • {p}")

    if report.get('suggested_next_action'):
        lines.append(f"\n💡 建议优先: {report['suggested_next_action']}")

    lines.append(f"\n{'=' * 60}")
    return '\n'.join(lines)


# ==================== MAIN ====================

def main():
    print("🔍 Yina 抖音每日分析器 v1", flush=True)

    # 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到 {INPUT_FILE}，请先运行提取器", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容 v3 (顶层items) 和 v4 (嵌套 favorites/likes)
    if 'favorites' in data or 'likes' in data:
        favorites = data.get('favorites', {}).get('items', [])
        likes = data.get('likes', {}).get('items', [])
    else:
        # v3 格式：只有收藏
        favorites = data.get('items', [])
        likes = []

    print(f"📌 收藏: {len(favorites)} 条", flush=True)
    print(f"❤️ 喜欢: {len(likes)} 条", flush=True)

    if not favorites and not likes:
        print("⚠️ 没有数据可分析", file=sys.stderr)
        sys.exit(0)

    # 构建 prompt 并调用 AI
    print("🤖 正在让 Gemini 分析...", flush=True)
    prompt = build_prompt(favorites, likes)

    try:
        response = call_llm(prompt)
        report = parse_response(response)
    except Exception as e:
        print(f"❌ AI 调用失败: {e}", file=sys.stderr)
        # Fallback: 保存原始数据和错误
        report = {
            "summary": f"AI分析失败: {e}",
            "actionable_items": [],
            "error": str(e),
        }

    # 添加元数据
    report['_meta'] = {
        'analyzed_at': datetime.now().isoformat(),
        'favorites_count': len(favorites),
        'likes_count': len(likes),
        'source_file': INPUT_FILE,
    }

    # 保存 JSON 报告
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印可读报告
    print("\n" + format_report(report))
    print(f"\n💾 JSON 报告: {OUTPUT_FILE}")

    return report


if __name__ == '__main__':
    main()
