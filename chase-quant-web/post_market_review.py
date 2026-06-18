#!/usr/bin/env python3
"""
🐾 Yina AI 盘后复盘与策略推演引擎 v1.0
========================================
= 五步链式推理 (Chain-of-Thought)
= 观察 → 对比 → 归因 → 假设 → 行动
= 每日 22:05 晚间深度复盘 + 08:35 盘前简报
= 推送企业微信「金融监控」群

核心理念:
  不是简单打分，而是像人类分析师一样一步步推演：
  今天发生了什么？→ 和预期比有什么意外？→ 为什么会这样？
  → 明天有哪些可能？→ 具体该怎么做？
  每一步的推理都会保存，第二天验证，形成可追踪的决策闭环。

用法:
  python3 post_market_review.py --mode evening    # 晚间深度复盘(5步完整CoT)
  python3 post_market_review.py --mode morning    # 盘前简报(轻量)
  python3 post_market_review.py --dry-run         # 只打印不推送
  python3 post_market_review.py --track-record    # 查看历史预测准确率
"""
import os, sys, json, time, re
import urllib.request, urllib.error
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

TZ = timezone(timedelta(hours=7))  # 曼谷 UTC+7
API_BASE = "http://localhost:8766"
DATA_DIR = Path(__file__).parent / "data"
REVIEWS_DIR = DATA_DIR / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# 企微 Webhook (从 .env 加载，fallback 到硬编码)
_WEBHOOK_KEY = ""
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("WECHAT_WEBHOOK_KEY="):
                _WEBHOOK_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
if not _WEBHOOK_KEY:
    _WEBHOOK_KEY = "2c602b48-5da2-4989-9193-30c0e226c769"  # fallback

WEBHOOK_URL = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={_WEBHOOK_KEY}"

# OpenRouter API Key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    _or_env = os.path.expanduser("~/Mac-For-Claude/free-vision-image-tools/.env")
    if os.path.exists(_or_env):
        with open(_or_env) as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    OPENROUTER_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ═══════════════════════════════════════════
# 数据容器
# ═══════════════════════════════════════════

@dataclass
class ChainStep:
    """单步推理结果"""
    step: int
    name: str
    content: str
    model: str
    tokens: int = 0
    latency_ms: int = 0

@dataclass
class ReviewRecord:
    """一次完整复盘"""
    date: str
    mode: str
    timestamp: str
    chain_steps: List[Dict] = field(default_factory=list)
    summary: str = ""
    scenarios: List[Dict] = field(default_factory=list)
    action_plan: str = ""
    market_snapshot: Dict = field(default_factory=dict)
    prediction_count: int = 0

@dataclass
class TrackRecord:
    """预测追踪"""
    total_reviews: int = 0
    total_predictions: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0
    by_market: Dict[str, Dict] = field(default_factory=dict)


# ═══════════════════════════════════════════
# API 数据获取
# ═══════════════════════════════════════════

def api_get(path: str, timeout: int = 90) -> dict:
    """从本地 API server (port 8766) 获取数据"""
    url = f"{API_BASE}{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Yina-Review/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def safe_api_get(path: str, fallback: dict = None, timeout: int = 30) -> dict:
    """安全获取，失败返回 fallback 不中断"""
    try:
        result = api_get(path, timeout)
        if "error" in result:
            return fallback or {"error": result["error"], "available": False}
        return result
    except Exception as e:
        return fallback or {"error": str(e), "available": False}

def collect_market_context() -> dict:
    """并行收集所有市场数据（用串行实现，未来可改并行）"""
    ctx = {
        "timestamp": datetime.now(TZ).isoformat(),
        "markets": {},
        "portfolio": None,
        "dashboard": None,
        "sentiment": None,
    }

    # 五市场信号
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        data = safe_api_get(f"/api/signals?market={market}")
        signals = data.get("signals", {}).get(market, []) if "signals" in data else []
        ctx["markets"][market] = {
            "available": len(signals) > 0,
            "count": len(signals),
            "signals": signals[:20],  # 每市场最多20个
        }

    # 组合 & 仪表板 & 情绪
    ctx["portfolio"] = safe_api_get("/api/portfolio")
    ctx["dashboard"] = safe_api_get("/api/dashboard")
    ctx["sentiment"] = safe_api_get("/api/sentiment/overview")

    # Binance 24hr ticker（crypto+bStocks实时价格）
    try:
        crypto_syms = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
                        "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
        url = f'https://api.binance.com/api/v3/ticker/24hr?symbols={json.dumps(crypto_syms)}'
        req = urllib.request.Request(url, headers={"User-Agent": "Yina-Review/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tickers = json.loads(resp.read())
            ctx["crypto_24hr"] = {t["symbol"]: {
                "price": float(t["lastPrice"]),
                "change_pct": float(t["priceChangePercent"]),
                "volume_usdt": float(t["quoteVolume"]),
                "high": float(t["highPrice"]),
                "low": float(t["lowPrice"]),
            } for t in tickers}
    except Exception as e:
        ctx["crypto_24hr"] = {"error": str(e)}

    return ctx


# ═══════════════════════════════════════════
# OpenRouter LLM 客户端
# ═══════════════════════════════════════════

# 免费模型 — 按任务类型分配
MODELS = {
    "observe": "google/gemma-4-31b-it:free",       # Gemma: 输出稳，适合数据整理
    "compare": "meta-llama/llama-3.3-70b-instruct:free",  # Llama: 对比分析强
    "attribute": "deepseek/deepseek-chat:free",     # DeepSeek: 因果推理强
    "hypothesize": "deepseek/deepseek-chat:free",   # DeepSeek: 情景推演
    "act": "deepseek/deepseek-chat:free",           # DeepSeek: 行动计划
    "morning": "google/gemma-4-31b-it:free",        # 早报: Gemma够用
    "fallback": "google/gemma-4-31b-it:free",       # 备用
}

def call_openrouter(system_prompt: str, user_prompt: str, step_name: str = "analysis",
                    max_tokens: int = 1500, temperature: float = 0.3) -> str:
    """调用 OpenRouter 模型，带自动回退"""
    model = MODELS.get(step_name, MODELS["fallback"])

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/yina-tools",
        "X-Title": "Yina Post-Market Review",
    }

    t0 = time.time()
    last_error = None

    # 尝试主模型 → fallback
    for attempt_model in [model, MODELS["fallback"]]:
        if attempt_model != model:
            payload = json.dumps({
                "model": attempt_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode("utf-8")

        for retry in range(2):
            try:
                req = urllib.request.Request(f"{OPENROUTER_BASE}/chat/completions",
                                             data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                    content = data["choices"][0]["message"]["content"]
                    elapsed = int((time.time() - t0) * 1000)
                    print(f"   ✅ [{step_name}] {attempt_model} → {len(content)}字符 ({elapsed}ms)")
                    return content
            except urllib.error.HTTPError as e:
                err = e.read().decode()[:200]
                if e.code == 429 and retry == 0:
                    time.sleep(3)
                    continue
                last_error = f"HTTP {e.code}: {err}"
                break
            except Exception as e:
                last_error = str(e)
                break

    raise RuntimeError(f"OpenRouter调用失败 [{step_name}]: {last_error}")


# ═══════════════════════════════════════════
# 五步链式推理 Prompt
# ═══════════════════════════════════════════

def build_data_summary(ctx: dict) -> str:
    """将市场数据转为 LLM 可读的文本摘要"""
    lines = []
    for market, mdata in ctx.get("markets", {}).items():
        sigs = mdata.get("signals", [])
        if not sigs:
            lines.append(f"**{market}**: 数据不可用")
            continue
        n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
        n_sell = sum(1 for s in sigs if s.get("action") == "SELL")
        lines.append(f"\n### {market} ({len(sigs)}标的, 🟢{n_buy}买 🟡{len(sigs)-n_buy-n_sell}观 🔴{n_sell}卖)")
        for s in sorted(sigs, key=lambda x: x.get("score", 0), reverse=True)[:5]:
            name = s.get("name", s.get("symbol", "?"))
            lines.append(f"- {s['symbol']} ({name}): 评分{s.get('score','?')} | "
                         f"建议{s.get('action','?')} | "
                         f"理由: {', '.join(s.get('reasons', [])[:2])}")

    # Crypto 实时价格
    c24 = ctx.get("crypto_24hr", {})
    if c24 and "error" not in c24:
        lines.append("\n### 实时价格(24h变动)")
        for sym, t in list(c24.items())[:8]:
            lines.append(f"- {sym}: ${t['price']:.2f} ({t['change_pct']:+.2f}%)")

    # 组合
    pf = ctx.get("portfolio", {})
    if pf and "error" not in pf:
        lines.append(f"\n### 组合状态")
        lines.append(f"总价值: {pf.get('total_value', '?')}")
        positions = pf.get("positions", [])
        if positions:
            for p in positions[:5]:
                lines.append(f"- {p.get('symbol','?')}: {p.get('size','?')} | "
                             f"盈亏 {p.get('pnl_pct', 0):+.2f}%")

    return "\n".join(lines)


# ── 第一步：观察 ──

STEP1_SYSTEM = """你是 Yina 的市场数据分析师。你的任务：纯数据复述 —— 不加解读、不预测、不评论。
像播音员一样，客观报告今天各大市场发生了什么。

规则:
1. 只复述数据，不说"这意味着"、"值得关注"等主观判断
2. 数据要具体：价格、涨跌幅、评分、买卖信号数量
3. 如果某市场缺失，直接标注"数据不可用"
4. 格式简洁，每市场2-3行"""

def build_step1_prompt(ctx: dict) -> str:
    return f"""## 今日市场数据 (曼谷时间 {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')})

请客观复述以下数据：

{ctx['data_summary']}

请按以下格式输出：

### 观察 · 今日数据事实

**₿ 加密货币**: [BTC/ETH等核心币种价格+涨跌]
**🇨🇳 A股**: [亮点标的+涨跌方向]
**🇺🇸 美股**: [亮点标的+涨跌方向]
**🇭🇰 港股**: [亮点标的+涨跌方向]
**🏦 bStocks**: [币安代币化股票表现]
**💰 组合状态**: [总价值+盈亏]
**📊 信号统计**: 各市场买卖比"""


# ── 第二步：对比 ──

STEP2_SYSTEM = """你是 Yina 的偏离检测分析师。你的唯一任务：找出「和预期不一样的」地方。

检查维度（按优先级）:
1. 跨市场背离：加密货币和股票是否同向？不同向就是信号
2. 趋势变化：加速上涨？减速？反转？
3. 评分vs实际：AI给的评分和实际涨跌是否一致？
4. 极端值：有没有单一资产涨跌幅远超其他？

规则:
- 每个偏离标注严重程度 (⚠️轻微 / 🔴重要 / 🚨关键)
- 没有偏离就说没有，不要硬找
- 具体数字对比，不说模糊的话"""

def build_step2_prompt(ctx: dict, step1_output: str) -> str:
    prev_review = _load_last_review()
    prev_text = ""
    if prev_review:
        prev_text = f"\n昨日复盘摘要:\n{prev_review.get('summary', '无')}\n"
        scenarios = prev_review.get('scenarios', [])
        if scenarios:
            prev_text += "昨日情景预测:\n"
            for s in scenarios:
                prev_text += f"- {s.get('name','?')} (概率{s.get('probability','?')}%): {s.get('narrative','')[:100]}\n"

    return f"""## 今日数据
{ctx['data_summary'][:3000]}

## AI第一步观察
{step1_output[:1500]}
{prev_text}

请对比今日实际数据 vs 昨日预期，找出所有偏离：

### 对比 · 偏离预期检测
**跨市场背离**:
**趋势变化**:
**评分与实际偏差**:
**极端异常值**:
**意外度评分** (1-5分, 5=极度意外):"""


# ── 第三步：归因 ──

STEP3_SYSTEM = """你是 Yina 的因果链分析师。你的任务：解释 WHY。

推理框架:
1. 供给链瓶颈传导: 瓶颈节点(高评分)的变动会被放大 → 盯住瓶颈
2. 宏观联动: Fed政策 → 利率 → 美元 → 风险资产 → 加密货币
3. 资金流向: 聪明钱在哪？防御性还是进攻性配置？
4. 叙事动力学: 什么故事在驱动情绪？新叙事还是旧叙事？

规则:
- 用"可能因为..."而非"这是因为..."（保持认知谦逊）
- 每条因果链标注置信度 (高/中/低)
- 提供至少1个替代解释
- 用具体数据支撑"""

def build_step3_prompt(ctx: dict, step1: str, step2: str) -> str:
    return f"""## 市场数据
{ctx['data_summary'][:2500]}

## 观察结果
{step1[:1000]}

## 偏离检测
{step2[:1500]}

请进行因果链推演：

### 归因 · 因果链推演
**主驱动因素**: [1-2句核心原因，标注置信度]
**因果链**:
1. [上游事件] → [中间传导] → [市场反应] (置信度: X%)
2. (继续列举2-4条)
**瓶颈节点分析**: 供给链瓶颈是否放大了波动？
**替代解释**: 如果主因不对，还可能是...
**关键不确定性**: 目前最大的未知数是？"""


# ── 第四步：假设 ──

STEP4_SYSTEM = """你是 Yina 的情景规划专家。生成3个互斥的明天市场情景。

关键要求:
1. 三个情景必须真正不同（不是乐观/中性/悲观的三段套话）
2. 每个有具体触发条件
3. 每个有可验证的证伪条件（"如果看到X，这个情景就错了"）
4. 概率总和接近100%
5. 每个情景有具体的市场影响（不同资产方向可能不同）

坏的情景规划: A=涨 B=平 C=跌
好的情景规划: A=AI算力瓶颈加剧,B=监管黑天鹅,C=流动性意外宽松"""

def build_step4_prompt(ctx: dict, step1: str, step2: str, step3: str) -> str:
    return f"""## 今日情况
{ctx['data_summary'][:2000]}

## 归因分析
{step3[:1500]}

## 偏离情况
{step2[:1000]}

请生成3个互斥的明日情景：

### 假设 · 明日情景推演

**情景A: [独特命名]** (概率: XX%)
> 叙事: [1-2句讲清楚这个情景下明天会发生什么]
> 触发: [什么事件/价位会激活这个情景]
> 市场影响:
> - 加密货币: [方向+幅度]
> - A股: [方向+幅度]
> - 美股: [方向+幅度]
> - 港股: [方向+幅度]
> **证伪条件**: "如果____，这个情景就错了"
> 关键价位: [要盯住的水平]

(情景B和C同样格式)

**情景概率校准说明**: [为什么这样分配概率]"""


# ── 第五步：行动 ──

STEP5_SYSTEM = """你是 Yina 的交易行动规划师。基于完整的推理链，制定明天的具体行动计划。

Chase哥 的约束条件:
- 10,000 RMB 虚拟盘
- 最多5个仓位
- 硬止损 -8%
- 日亏损熔断 -5%
- 单笔最大亏损 200 RMB (2%)

规则:
1. 行动要具体：标的、方向、仓位%、入场价、止损、止盈
2. 现有持仓要说明「持有/减仓/加仓/清仓」+ 理由
3. 一定要有「不做什么」清单 —— 管住手比管住钱难
4. 仓位大小要匹配置信度"""

def build_step5_prompt(ctx: dict, step1: str, step2: str, step3: str, step4: str) -> str:
    pf = ctx.get("portfolio", {})
    pos_text = "无持仓"
    positions = pf.get("positions", [])
    if positions:
        pos_text = "\n".join([
            f"- {p.get('symbol','?')}: 仓位{p.get('size','?')} | "
            f"盈亏{p.get('pnl_pct',0):+.2f}% | 入场{p.get('entry_price','?')}"
            for p in positions[:5]
        ])

    return f"""## 组合现状
{pos_text}

## 观察总结
{step1[:800]}

## 偏离与归因
{step2[:600]}
{step3[:800]}

## 明日情景
{step4[:1500]}

请生成明日行动计划：

### 行动 · 明日执行计划

**核心判断**: [1句话方向性判断 + 置信度]

**现有持仓调整**:
| 持仓 | 行动 | 理由 |
|------|------|------|

**新开仓建议** (如果有):
| 标的 | 方向 | 仓位% | 入场价 | 止损 | 止盈 | 理由 |
|------|------|-------|--------|------|------|------|

**组合权重调整**: [如果需要再平衡]

**风控提醒**: [明天特别注意的风险]

**不做什么**: [管住手的清单 —— 想冲动交易但决定不做的事]"""


# ═══════════════════════════════════════════
# 链式推理引擎
# ═══════════════════════════════════════════

def run_chain_of_thought(ctx: dict) -> List[ChainStep]:
    """执行完整的5步链式推理"""
    steps = []

    # Step 1: 观察
    print("\n🧠 第一步: 观察 (数据复述)...")
    try:
        s1 = call_openrouter(STEP1_SYSTEM, build_step1_prompt(ctx), "observe", max_tokens=800)
        steps.append(ChainStep(1, "观察·数据事实", s1, MODELS["observe"]))
    except Exception as e:
        print(f"   ⚠️ 观察步骤失败: {e}")
        steps.append(ChainStep(1, "观察·数据事实", f"[AI推理不可用: {e}]", "none"))

    # Step 2: 对比
    print("🧠 第二步: 对比 (偏离检测)...")
    try:
        s2 = call_openrouter(STEP2_SYSTEM, build_step2_prompt(ctx, steps[-1].content),
                            "compare", max_tokens=800)
        steps.append(ChainStep(2, "对比·偏离检测", s2, MODELS["compare"]))
    except Exception as e:
        print(f"   ⚠️ 对比步骤失败: {e}")
        steps.append(ChainStep(2, "对比·偏离检测", f"[AI推理不可用: {e}]", "none"))

    # Step 3: 归因
    print("🧠 第三步: 归因 (因果推演)...")
    try:
        s3 = call_openrouter(STEP3_SYSTEM,
                            build_step3_prompt(ctx, steps[0].content, steps[1].content),
                            "attribute", max_tokens=1200)
        steps.append(ChainStep(3, "归因·因果推演", s3, MODELS["attribute"]))
    except Exception as e:
        print(f"   ⚠️ 归因步骤失败: {e}")
        steps.append(ChainStep(3, "归因·因果推演", f"[AI推理不可用: {e}]", "none"))

    # Step 4: 假设
    print("🧠 第四步: 假设 (情景推演)...")
    try:
        s4 = call_openrouter(STEP4_SYSTEM,
                            build_step4_prompt(ctx, steps[0].content, steps[1].content, steps[2].content),
                            "hypothesize", max_tokens=1200)
        steps.append(ChainStep(4, "假设·情景推演", s4, MODELS["hypothesize"]))
    except Exception as e:
        print(f"   ⚠️ 假设步骤失败: {e}")
        steps.append(ChainStep(4, "假设·情景推演", f"[AI推理不可用: {e}]", "none"))

    # Step 5: 行动
    print("🧠 第五步: 行动 (执行计划)...")
    try:
        s5 = call_openrouter(STEP5_SYSTEM,
                            build_step5_prompt(ctx,
                                steps[0].content, steps[1].content,
                                steps[2].content, steps[3].content),
                            "act", max_tokens=1000)
        steps.append(ChainStep(5, "行动·执行计划", s5, MODELS["act"]))
    except Exception as e:
        print(f"   ⚠️ 行动步骤失败: {e}")
        steps.append(ChainStep(5, "行动·执行计划", f"[AI推理不可用: {e}]", "none"))

    return steps


# ═══════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════

def build_market_snapshot_table(ctx: dict) -> str:
    """生成各市场快照表格"""
    lines = ["## 📊 市场快照 · 五市场一览", ""]
    for market in ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]:
        mdata = ctx["markets"].get(market, {})
        sigs = mdata.get("signals", [])
        if not sigs:
            lines.append(f"**{market}**: 数据不可用")
            continue

        n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
        n_sell = sum(1 for s in sigs if s.get("action") == "SELL")
        n_watch = len(sigs) - n_buy - n_sell

        labels = {"crypto": "₿ 加密货币", "a_stock": "🇨🇳 A股", "us_stock": "🇺🇸 美股",
                   "hk_stock": "🇭🇰 港股", "b_stock": "🏦 bStocks"}
        lines.append(f"### {labels.get(market, market)}  🟢{n_buy} 🟡{n_watch} 🔴{n_sell}")
        lines.append("| 标的 | 名称 | 价格 | 涨跌 | 评分 | 建议 | 理由 |")
        lines.append("|------|------|------|------|------|------|------|")
        for s in sorted(sigs, key=lambda x: x.get("score", 0), reverse=True)[:8]:
            name = s.get("name", s["symbol"])[:10]
            price = s.get("price", 0)
            score = s.get("adjusted_score", s.get("score", "?"))
            action = s.get("action", "?")
            reasons = ", ".join(s.get("reasons", [])[:2])
            action_emoji = "🟢" if action == "BUY" else "🟡" if action == "WATCH" else "🔴"
            lines.append(f"| {s['symbol']} | {name} | {price} | - | {score} | {action_emoji} {action} | {reasons} |")
        lines.append("")
    return "\n".join(lines)


def build_report_markdown(ctx: dict, steps: List[ChainStep], mode: str = "evening") -> str:
    """组装完整报告"""
    now = datetime.now(TZ)
    lines = []

    # 头部
    if mode == "evening":
        lines.append(f"# 🐾 Yina AI 盘后复盘与策略推演")
        lines.append(f"## {now.strftime('%Y-%m-%d')}  {now.strftime('%H:%M')} 曼谷时间 · 五步链式推理")
        lines.append("")
        lines.append("> 🔴高优先 🟡中优先 🟢低优先 · 止盈/止损严格 · 证伪条件可追踪")
    else:
        lines.append(f"# 🌅 Yina AI 盘前简报")
        lines.append(f"## {now.strftime('%Y-%m-%d')}  {now.strftime('%H:%M')} 曼谷时间")
        lines.append("")

    lines.append("")

    # 市场快照
    lines.append(build_market_snapshot_table(ctx))

    # 链式推理
    lines.append("---")
    lines.append("## 🧠 五步链式推理")
    lines.append("")

    step_emojis = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣"}
    for step in steps:
        emoji = step_emojis.get(step.step, "➡️")
        lines.append(f"### {emoji} 第{step.step}步: {step.name}")
        lines.append(f"*模型: {step.model}*")
        lines.append("")
        lines.append(step.content)
        lines.append("")

    # 组合状态（如果有）
    pf = ctx.get("portfolio", {})
    if pf and "error" not in pf and pf.get("total_value"):
        lines.append("---")
        lines.append("## 💼 组合状态")
        lines.append(f"- 总价值: {pf.get('total_value', '?')}")
        lines.append(f"- 现金: {pf.get('cash', '?')}")
        positions = pf.get("positions", [])
        if positions:
            lines.append("| 持仓 | 数量 | 入场价 | 当前盈亏 |")
            lines.append("|------|------|--------|----------|")
            for p in positions:
                lines.append(f"| {p.get('symbol','?')} | {p.get('size','?')} | "
                            f"{p.get('entry_price','?')} | {p.get('pnl_pct',0):+.2f}% |")

    # 尾部
    lines.append("")
    lines.append("---")
    lines.append(f"🐾 Yina AI 盘后复盘引擎 v1.0 · 五步链式推理 · 证伪可追踪 · {now.strftime('%H:%M')}")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 历史追踪
# ═══════════════════════════════════════════

def _today_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")

def _load_last_review() -> Optional[dict]:
    """加载最近一次晚间复盘"""
    files = sorted(REVIEWS_DIR.glob("*_evening.json"), reverse=True)
    if files:
        try:
            with open(files[0]) as f:
                return json.load(f)
        except:
            pass
    return None

def save_review(steps: List[ChainStep], ctx: dict, mode: str):
    """保存复盘到磁盘"""
    today = _today_str()
    now = datetime.now(TZ).isoformat()

    # JSON（结构化，用于追踪）
    record = {
        "date": today,
        "mode": mode,
        "timestamp": now,
        "chain_steps": [{"step": s.step, "name": s.name, "content": s.content,
                          "model": s.model} for s in steps],
        "summary": steps[0].content[:300] if steps else "",
        "market_snapshot": {
            m: {"count": d["count"], "buy": sum(1 for s in d.get("signals", [])
                   if s.get("action") == "BUY"),
                "sell": sum(1 for s in d.get("signals", [])
                   if s.get("action") == "SELL")}
            for m, d in ctx.get("markets", {}).items()
        },
        "prediction_count": len(steps) if steps else 0,
    }
    json_path = REVIEWS_DIR / f"{today}_{mode}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"   💾 JSON: {json_path}")

    # Markdown（可读）
    md_path = REVIEWS_DIR / f"{today}_{mode}.md"
    md_content = build_report_markdown(ctx, steps, mode)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"   📝 MD: {md_path}")


def compute_track_record() -> TrackRecord:
    """计算历史预测准确率"""
    tr = TrackRecord()
    review_files = sorted(REVIEWS_DIR.glob("*_evening.json"))
    tr.total_reviews = len(review_files)
    # 基础统计 — 未来可扩展情景验证
    for rf in review_files:
        try:
            with open(rf) as f:
                r = json.load(f)
            tr.total_predictions += r.get("prediction_count", 0)
        except:
            pass
    return tr


def show_track_record():
    """展示预测追踪"""
    tr = compute_track_record()
    print(f"\n📈 预测追踪记录")
    print(f"   总复盘次数: {tr.total_reviews}")
    print(f"   总预测数: {tr.total_predictions}")
    print(f"   (更多追踪指标将在积累足够数据后启用)")
    files = sorted(REVIEWS_DIR.glob("*_evening.json"))
    if files:
        print(f"\n   最近5次复盘:")
        for f in files[-5:]:
            print(f"   📅 {f.stem.replace('_evening', '')}")
    if not files:
        print("   暂无复盘记录")


# ═══════════════════════════════════════════
# 企业微信推送
# ═══════════════════════════════════════════

def push_to_wechat(md_content: str) -> bool:
    """推送到企微「金融监控」群，自动分片"""
    max_bytes = 3800
    chunks = []
    current = ""
    for line in md_content.split("\n"):
        test = current + line + "\n"
        if len(test.encode("utf-8")) > max_bytes and current:
            chunks.append(current)
            current = line + "\n"
        else:
            current = test
    if current:
        chunks.append(current)

    success = True
    for i, chunk in enumerate(chunks):
        prefix = f"({i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
        payload = json.dumps({
            "msgtype": "markdown",
            "markdown": {"content": prefix + chunk}
        }).encode("utf-8")

        req = urllib.request.Request(WEBHOOK_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                if result.get("errcode") != 0:
                    print(f"   ⚠️ 分段{i+1}推送异常: {result}")
                    success = False
                else:
                    print(f"   ✅ 分段{i+1}/{len(chunks)} 推送成功")
        except Exception as e:
            print(f"   ❌ 分段{i+1}推送失败: {e}")
            success = False
    return success


# ═══════════════════════════════════════════
# 盘前简报
# ═══════════════════════════════════════════

MORNING_SYSTEM = """你是 Yina 的盘前简报员。基于昨日复盘和隔夜数据，生成简洁的盘前指南。

格式要求:
1. 列出隔夜关键变动（加密货币+美股收盘+亚洲期货）
2. 昨日3个情景中哪个正在激活？证伪条件触发了吗？
3. 今日重点关注事件和价位
4. 简洁行动建议

200字以内，要点形式。"""

def generate_morning_brief(ctx: dict) -> str:
    """生成盘前简报"""
    prev = _load_last_review()
    prev_text = "无昨日复盘"
    if prev:
        steps = prev.get("chain_steps", [])
        if len(steps) >= 4:
            prev_text = f"昨日情景推演:\n{steps[3].get('content', '')[:1500]}"

    # Crypto overnight
    c24 = ctx.get("crypto_24hr", {})
    crypto_text = ""
    if c24 and "error" not in c24:
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            t = c24.get(sym, {})
            if t:
                crypto_text += f"{sym}: ${t['price']:.2f} ({t['change_pct']:+.2f}%)\n"

    prompt = f"""## 隔夜市场数据 (曼谷时间 {datetime.now(TZ).strftime('%H:%M')})

加密货币:
{crypto_text}

## 昨日复盘
{prev_text}

请生成盘前简报 (200字以内):

### 盘前简报
**隔夜动态**:
**情景激活**: 昨日3情景哪个在发生？证伪条件触发了吗？
**今日关注**: 关键事件+价位
**简洁建议**: 1-2句话"""

    try:
        content = call_openrouter(MORNING_SYSTEM, prompt, "morning", max_tokens=600, temperature=0.2)
    except Exception as e:
        content = f"[AI简报生成失败: {e}]"

    # 组装
    now = datetime.now(TZ)
    report = f"""# 🌅 Yina 盘前简报
## {now.strftime('%Y-%m-%d %H:%M')} 曼谷时间

{content}

---
🐾 Yina 盘前简报 · 完整复盘见昨日22:00"""
    return report


# ═══════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Yina AI 盘后复盘与策略推演引擎")
    parser.add_argument("--mode", choices=["evening", "morning", "auto"],
                        default="auto", help="运行模式 (默认auto=根据时间自动判断)")
    parser.add_argument("--dry-run", action="store_true", help="只打印报告不推送")
    parser.add_argument("--test", action="store_true", help="测试API连通性")
    parser.add_argument("--track-record", action="store_true", help="查看历史预测准确率")
    parser.add_argument("--no-llm", action="store_true", help="跳过LLM推理(只用数据表格)")
    args = parser.parse_args()

    # --track-record
    if args.track_record:
        show_track_record()
        return

    # --test
    if args.test:
        print("🔍 测试连通性...")
        print(f"   企微 Webhook: {'✅' if _WEBHOOK_KEY else '❌'} "
              f"({_WEBHOOK_KEY[:20]}...)" if _WEBHOOK_KEY else "   企微: ❌ 无key")
        print(f"   OpenRouter: {'✅' if OPENROUTER_API_KEY else '❌'}")
        sig = api_get("/api/signals?market=crypto")
        print(f"   API信号(8766): {'✅' if 'error' not in sig else '❌ ' + sig.get('error','')}")
        return

    # 自动判断模式
    now = datetime.now(TZ)
    if args.mode == "auto":
        hour = now.hour
        args.mode = "morning" if 5 <= hour < 12 else "evening"

    print(f"🐾 Yina AI 盘后{'复盘' if args.mode == 'evening' else '简报'} v1.0")
    print(f"   {now.strftime('%Y-%m-%d %H:%M')} 曼谷时间 | 模式: {args.mode}")
    print()

    # ── 数据采集 ──
    print("📡 采集五市场数据...")
    ctx = collect_market_context()
    n_signals = sum(m["count"] for m in ctx["markets"].values())
    print(f"   ✅ 共 {n_signals} 个信号")
    ctx["data_summary"] = build_data_summary(ctx)

    # ── 推理 / 简报 ──
    if args.mode == "morning":
        print("\n🌅 生成盘前简报...")
        if args.no_llm:
            report = "# 🌅 盘前简报\n\n[LLM跳过 — 仅数据]\n\n" + build_market_snapshot_table(ctx)
        else:
            report = generate_morning_brief(ctx)
        steps = []
    else:
        if args.no_llm:
            print("\n⚠️ --no-llm 模式，仅生成数据表格")
            steps = [ChainStep(1, "数据快照(无AI)", build_market_snapshot_table(ctx), "none")]
        else:
            steps = run_chain_of_thought(ctx)
        report = build_report_markdown(ctx, steps, args.mode)

    # ── 输出 ──
    if args.dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] 以下为报告预览（未推送）")
        print("=" * 60)
        print(report)
    else:
        # 保存
        if steps:
            save_review(steps, ctx, args.mode)

        # 推送
        print(f"\n📤 推送企业微信「金融监控」...")
        report_bytes = len(report.encode('utf-8'))
        print(f"   报告大小: {report_bytes} bytes")
        ok = push_to_wechat(report)
        if ok:
            print("✅ 推送成功!")
        else:
            print("❌ 推送失败")
            sys.exit(1)

    print(f"\n✅ Yina 盘后{'复盘' if args.mode == 'evening' else '简报'}完成! 🐾")


if __name__ == "__main__":
    main()
