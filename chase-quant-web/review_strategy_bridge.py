#!/usr/bin/env python3
"""
🐾 Yina AI复盘 → 策略参数桥接器 v1.0
========================================
读取 AI 盘后复盘的五步链式推理结果，提取结构化策略参数，
注入到虚拟盘和实盘交易引擎中。

核心功能:
  1. 解析 CoT 情景推演 → 市场方向偏见
  2. 解析行动方案 → 仓位/风控参数调整
  3. 解析「不做什么」→ 交易黑名单
  4. 生成 strategy_overlay.json 供交易引擎消费

用法:
  python3 review_strategy_bridge.py                    # 生成最新 overlay
  python3 review_strategy_bridge.py --review 2026-06-18_evening  # 指定复盘
  python3 review_strategy_bridge.py --show             # 显示当前 overlay
"""
import json, os, re, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

DATA_DIR = Path(__file__).parent / "data"
REVIEWS_DIR = DATA_DIR / "reviews"
OVERLAY_FILE = DATA_DIR / "strategy_overlay.json"
TZ = timezone(timedelta(hours=7))  # Bangkok


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class MarketBias:
    """单市场方向偏见"""
    direction: str = "neutral"       # bullish | bearish | neutral
    confidence: float = 0.5          # 0-1
    key_drivers: List[str] = field(default_factory=list)
    key_levels: Dict[str, float] = field(default_factory=dict)  # support/resistance

@dataclass
class ScenarioWeight:
    """情景权重"""
    name: str = ""
    probability: float = 0.0
    crypto_direction: str = "neutral"
    a_stock_direction: str = "neutral"
    us_stock_direction: str = "neutral"
    hk_stock_direction: str = "neutral"
    b_stock_direction: str = "neutral"
    trigger_condition: str = ""
    falsification: str = ""

@dataclass
class StrategyOverlay:
    """策略叠加参数"""
    generated_at: str = ""
    review_date: str = ""
    source_review: str = ""
    # 核心判断
    core_judgment: str = ""
    core_confidence: float = 0.5
    # 市场偏见
    market_biases: Dict[str, MarketBias] = field(default_factory=dict)
    # 情景权重
    scenarios: List[ScenarioWeight] = field(default_factory=list)
    # 仓位参数
    max_crypto_positions: int = 6
    single_trade_max_pct: float = 0.15       # 单笔最大仓位占比
    confidence_multiplier: float = 1.0        # 置信度乘数 (>1=加仓, <1=减仓)
    # 风控参数
    stop_loss_pct: float = -8.0
    take_profit_pct: float = 15.0
    trailing_stop_pct: float = 5.0            # 移动止盈回撤%
    daily_loss_meltdown_pct: float = -5.0
    # 资产偏好
    favor_assets: List[str] = field(default_factory=list)     # 优先考虑
    avoid_assets: List[str] = field(default_factory=list)     # 暂时回避
    # 行为约束
    dont_do: List[str] = field(default_factory=list)          # 不做什么
    risk_reminders: List[str] = field(default_factory=list)    # 风控提醒
    # 关键价位
    key_levels: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 元数据
    is_stale: bool = False                   # 超过24h未更新标记陈旧
    ttl_hours: int = 24


# ═══════════════════════════════════════════
# CoT 文本解析器
# ═══════════════════════════════════════════

def _extract_section(text: str, header: str, next_headers: List[str] = None) -> str:
    """从 markdown 中提取指定 section"""
    pattern = rf'\*?\*?{re.escape(header)}\*?\*?\s*\n(.*?)(?=\n\*?\*?\w|\n###|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        # 截断到下一个 header
        if next_headers:
            for nh in next_headers:
                idx = content.find(f"**{nh}**")
                if idx > 0:
                    content = content[:idx].strip()
                    break
        return content
    return ""


def parse_scenarios(step4_text: str) -> List[ScenarioWeight]:
    """从第四步(假设)解析情景推演 — 处理多种 markdown 格式的 CoT 输出"""
    scenarios = []

    # 匹配情景标题: **情景A: [名称]** (概率: 50%)
    # 或: **情景A: 名称** (概率: 50%)
    # 注意: [名称]** — 名称括号后可能有 bold 结束标记 **
    pattern = r'\*{0,2}情景([A-C])\*{0,2}\s*[:：]\s*\[?([^\]]*?)\]?\*{0,2}\s*\(概率[:：]\s*(\d+)\s*%\)'

    for m in re.finditer(pattern, step4_text):
        letter = m.group(1)
        name = m.group(2).strip().rstrip(']').strip()
        prob = int(m.group(3))

        scenario = ScenarioWeight(name=name, probability=prob)

        # 提取此情景的文本块 (到下一个情景头或section头)
        block_start = m.end()
        next_scenario = re.search(r'\*{0,2}情景[B-C]\*{0,2}\s*[:：]', step4_text[block_start:])
        next_section = re.search(r'\n\*{0,2}情景概率校准', step4_text[block_start:])
        end_pos = len(step4_text) - block_start
        if next_scenario:
            end_pos = min(end_pos, next_scenario.start())
        if next_section:
            end_pos = min(end_pos, next_section.start())
        if next_scenario is None and next_section is None:
            end_pos = min(1500, end_pos)
        block = step4_text[block_start:block_start + end_pos]

        # 提取各市场方向
        # 格式1: > - **加密货币**: 🟢 强劲上涨 (...)
        # 格式2: **加密货币**: 🟢 上涨
        # emoji: 🟢=bullish, 🔴=bearish, 🟡/⚪=neutral
        market_map = {
            "加密货币": "crypto_direction",
            "美股": "us_stock_direction",
            "港股": "hk_stock_direction",
            "A股": "a_stock_direction",
            "bStock": "b_stock_direction",
        }
        for label, attr in market_map.items():
            dir_pattern = rf'{label}\*{{0,2}}\s*[:：]\s*([🟢🔴🟡⚪])\s*([^\n]*)'
            dm = re.search(dir_pattern, block)
            if dm:
                emoji = dm.group(1)
                desc = dm.group(2).strip()
                if emoji == '🟢':
                    setattr(scenario, attr, 'bullish')
                elif emoji == '🔴':
                    setattr(scenario, attr, 'bearish')
                elif emoji in ('🟡', '⚪'):
                    setattr(scenario, attr, 'neutral')
                else:
                    # Fallback to text analysis
                    if any(w in desc for w in ['涨', 'bull', '上升', '走强', '反弹', '上涨', 'up', '补涨']):
                        setattr(scenario, attr, 'bullish')
                    elif any(w in desc for w in ['跌', 'bear', '下降', '走弱', '回调', '下跌', 'down', '承压', '阴跌', '震荡']):
                        setattr(scenario, attr, 'bearish')
                    else:
                        setattr(scenario, attr, 'neutral')

        # 提取触发条件: > **触发**: text
        trigger_m = re.search(r'触发\*{0,2}\s*[:：]\s*(.+?)(?:\n>|\n\*|\n\n|\Z)', block, re.DOTALL)
        if trigger_m:
            scenario.trigger_condition = trigger_m.group(1).strip()[:200]

        # 提取证伪条件: > **证伪条件**: "text"
        falsify_m = re.search(
            r'证伪条件\*{0,2}\s*[:：]\s*["""]?(.+?)["”""]?(?:\n>|\n\*|\n\n|\Z)',
            block, re.DOTALL
        )
        if falsify_m:
            scenario.falsification = falsify_m.group(1).strip().strip('"').strip('"').strip('"')[:200]

        scenarios.append(scenario)

    return scenarios


def parse_action_plan(step5_text: str) -> dict:
    """从第五步(行动)解析执行计划 — 处理 markdown 格式"""
    result = {
        "core_judgment": "",
        "core_confidence": 0.5,
        "position_adjustments": [],
        "new_positions": [],
        "dont_do": [],
        "risk_reminders": [],
        "favor_assets": [],
        "avoid_assets": [],
    }

    # 核心判断 — 处理 **核心判断**: 文本 (置信度: XX%)
    core_m = re.search(r'\*{0,2}核心判断\*{0,2}\s*[:：]\s*(.+?)(?:\n|$)', step5_text)
    if core_m:
        core_text = core_m.group(1).strip()
        conf_m = re.search(r'置信度[:：]?\s*(\d+)%', core_text)
        if conf_m:
            result["core_confidence"] = int(conf_m.group(1)) / 100
        result["core_judgment"] = core_text

    # 新开仓建议 — markdown 表格: | **SOL** | 做多 | 20% | $150 | $130 | $180 | 理由 |
    new_table = re.search(
        r'(?:新开仓建议|新开仓).*?\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n)+)',
        step5_text
    )
    if new_table:
        for row in new_table.group(1).strip().split('\n'):
            # Strip markdown bold and leading/trailing pipes
            cells = [c.strip().strip('*') for c in row.strip('|').split('|')]
            if len(cells) >= 5 and cells[0] and cells[0] not in ['标的', '------', '']:
                try:
                    entry = {
                        "symbol": cells[0].replace('/USDT', ''),
                        "direction": cells[1] if len(cells) > 1 else "做多",
                        "size_pct": 0.05,
                        "entry_price": 0.0,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "reason": "",
                    }
                    # Parse size percentage
                    size_str = cells[2] if len(cells) > 2 else "5%"
                    size_m = re.search(r'(\d+)', size_str.replace('%', ''))
                    if size_m:
                        entry["size_pct"] = int(size_m.group(1)) / 100
                    # Parse prices (handle $ prefix and commas)
                    for i, field in [(3, "entry_price"), (4, "stop_loss"), (5, "take_profit")]:
                        if len(cells) > i and cells[i]:
                            price_m = re.search(r'[\d,]+\.?\d*', cells[i].replace('$', '').replace(',', ''))
                            if price_m:
                                entry[field] = float(price_m.group(0).replace(',', ''))
                    if len(cells) > 6:
                        entry["reason"] = cells[6]
                    result["new_positions"].append(entry)
                except (ValueError, IndexError):
                    pass

    # 现有持仓调整 — markdown 表格
    pos_table = re.search(
        r'现有持仓调整.*?\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n)+)',
        step5_text
    )
    if pos_table:
        for row in pos_table.group(1).strip().split('\n'):
            cells = [c.strip().strip('*') for c in row.strip('|').split('|')]
            if len(cells) >= 2 and cells[0] not in ['持仓', '------', '', '无']:
                result["position_adjustments"].append({
                    "symbol": cells[0],
                    "action": cells[1] if len(cells) > 1 else "",
                    "reason": cells[2] if len(cells) > 2 else "",
                })

    # 不做什么 — 匹配 markdown 的 **不做什么** 部分，提取后续列表项
    dont_patterns = [
        r'\*{0,2}不做什么\*{0,2}\s*[:：]?\s*\n((?:(?:[-•>]\s*.+?\n)|(?:\d+\.\s*.+?\n))+)',
        r'\*{0,2}不做什么\*{0,2}\s*[:：]?\s*(.+?)(?=\n\*{0,2}(?:风控|组合|核心|$)|\Z)',
    ]
    for dp in dont_patterns:
        dont_m = re.search(dp, step5_text, re.DOTALL)
        if dont_m:
            section_text = dont_m.group(1)
            for line in section_text.strip().split('\n'):
                line = re.sub(r'^[-•>\d.]+\s*', '', line).strip()
                # Strip markdown formatting
                line = re.sub(r'\*{1,3}', '', line)
                line = re.sub(r'❌\s*', '', line)
                line = line.strip()
                if line and len(line) > 5 and not line.startswith('(管住'):
                    result["dont_do"].append(line)
            break

    # 风控提醒
    risk_patterns = [
        r'\*{0,2}风控提醒\*{0,2}\s*[:：]?\s*\n((?:(?:[-•>]\s*.+?\n)|(?:\d+\.\s*.+?\n))+)',
        r'\*{0,2}风控提醒\*{0,2}\s*[:：]?\s*(.+?)(?=\n\*{0,2}(?:不做|组合|核心|$)|\Z)',
    ]
    for rp in risk_patterns:
        risk_m = re.search(rp, step5_text, re.DOTALL)
        if risk_m:
            section_text = risk_m.group(1)
            for line in section_text.strip().split('\n'):
                line = re.sub(r'^[-•>\d.]+\s*', '', line).strip()
                if line and len(line) > 3:
                    result["risk_reminders"].append(line)
            break

    # 标的提及提取 — 从行动方案中识别被明确推荐/回避的标的
    # 优先/看好
    favor_matches = re.findall(
        r'(?:优先|看好|加仓|推荐|关注|布局|领涨|领头).*?\b([A-Z]{2,6})(?:/USDT)?\b',
        step5_text
    )
    for asset in favor_matches:
        if asset.upper() not in ["TLT", "IWM"]:  # filter non-crypto ETFs
            if asset not in result["favor_assets"]:
                result["favor_assets"].append(asset)

    # 回避/减仓
    avoid_matches = re.findall(
        r'(?:回避|减仓|清仓|不追|谨慎|放弃|弱势).*?\b([A-Z]{2,6})(?:/USDT)?\b',
        step5_text
    )
    for asset in avoid_matches:
        if asset not in result["avoid_assets"] and asset not in result["favor_assets"]:
            result["avoid_assets"].append(asset)

    return result


def parse_market_bias(scenarios: List[ScenarioWeight], step1_text: str = "", step3_text: str = "") -> Dict[str, MarketBias]:
    """从情景概率加权计算各市场方向偏见"""
    biases = {}
    markets = ["crypto", "a_stock", "us_stock", "hk_stock", "b_stock"]

    for market in markets:
        direction_score = 0.0  # +1=bullish, -1=bearish
        total_prob = 0.0

        for s in scenarios:
            prob = s.probability / 100.0
            total_prob += prob
            attr = f"{'crypto' if market == 'crypto' else market}_direction"
            dir_val = getattr(s, attr, 'neutral')
            if dir_val == 'bullish':
                direction_score += prob
            elif dir_val == 'bearish':
                direction_score -= prob

        if total_prob > 0:
            direction_score /= total_prob  # normalize

        if direction_score > 0.2:
            direction = "bullish"
        elif direction_score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        biases[market] = MarketBias(
            direction=direction,
            confidence=abs(direction_score),
        )

    # 从归因文本中提取关键价位
    # 匹配类似 "BTC 支撑 85000" 或 "BTC: $85,000 支撑"
    all_text = step1_text + step3_text
    level_pattern = r'([A-Z]{2,6})\S*\s*[:：]?\s*\$?([\d,]+)\s*(?:支撑|阻力|support|resistance)'
    for m in re.finditer(level_pattern, all_text, re.IGNORECASE):
        symbol = m.group(1)
        level = float(m.group(2).replace(',', ''))
        label = m.group(0)

        if symbol not in biases.get("crypto", MarketBias()).key_levels:
            if "crypto" not in [k for k in biases]:
                pass
        # attach to crypto market bias
        if symbol.upper() in ["BTC", "ETH", "SOL", "BNB", "AVAX", "XRP", "LINK"]:
            if "crypto" in biases:
                level_type = "support" if "支撑" in label or "support" in label.lower() else "resistance"
                biases["crypto"].key_levels[f"{symbol}_{level_type}"] = level

    return biases


# ═══════════════════════════════════════════
# 核心桥接逻辑
# ═══════════════════════════════════════════

def load_latest_review() -> Optional[dict]:
    """加载最新的复盘 JSON"""
    if not REVIEWS_DIR.exists():
        return None

    review_files = sorted(REVIEWS_DIR.glob("*_evening.json"), reverse=True)
    if not review_files:
        review_files = sorted(REVIEWS_DIR.glob("*.json"), reverse=True)

    if not review_files:
        return None

    try:
        with open(review_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载复盘失败: {e}")
        return None


def build_overlay(review: dict) -> StrategyOverlay:
    """从复盘记录构建策略叠加参数"""
    overlay = StrategyOverlay(
        generated_at=datetime.now(TZ).isoformat(),
        review_date=review.get("date", ""),
        source_review=f"data/reviews/{review.get('date', 'unknown')}_{review.get('mode', 'unknown')}.json",
    )

    chain_steps = review.get("chain_steps", [])
    steps_dict = {s.get("step", 0): s.get("content", "") for s in chain_steps}

    step1 = steps_dict.get(1, "")
    step2 = steps_dict.get(2, "")
    step3 = steps_dict.get(3, "")
    step4 = steps_dict.get(4, "")
    step5 = steps_dict.get(5, "")

    # ── 解析情景 ──
    overlay.scenarios = parse_scenarios(step4)

    # ── 解析行动方案 ──
    action = parse_action_plan(step5)
    overlay.core_judgment = action["core_judgment"]
    overlay.core_confidence = action["core_confidence"]
    overlay.dont_do = action["dont_do"]
    overlay.risk_reminders = action["risk_reminders"]
    overlay.favor_assets = action["favor_assets"]
    overlay.avoid_assets = action["avoid_assets"]

    # ── 市场偏见 ──
    overlay.market_biases = parse_market_bias(overlay.scenarios, step1, step3)

    # ── 从情绪推断仓位参数 ──
    crypto_bias = overlay.market_biases.get("crypto")
    if crypto_bias:
        if crypto_bias.direction == "bullish":
            overlay.confidence_multiplier = min(1.3, 1.0 + crypto_bias.confidence * 0.5)
            overlay.single_trade_max_pct = min(0.25, 0.15 * (1 + crypto_bias.confidence))
        elif crypto_bias.direction == "bearish":
            overlay.confidence_multiplier = max(0.5, 1.0 - crypto_bias.confidence * 0.5)
            overlay.single_trade_max_pct = max(0.05, 0.15 * (1 - crypto_bias.confidence))
            overlay.stop_loss_pct = max(-5.0, -8.0 * (1 + crypto_bias.confidence * 0.3))

    # ── 解析关键价位 ──
    overlay.key_levels = {}
    # 从归因和观察中提取
    for text in [step1, step3]:
        level_pattern = r'(\$?[\d,]+)\s*(?:支撑|阻力|support|resistance|关键|突破|跌破)'
        for m in re.finditer(level_pattern, text):
            pass  # complex extraction handled in parse_market_bias

    # 传递 crypto key_levels
    if "crypto" in overlay.market_biases:
        overlay.key_levels = overlay.market_biases["crypto"].key_levels

    # ── 时效性检查 ──
    try:
        review_dt = datetime.fromisoformat(review.get("timestamp", ""))
        age_hours = (datetime.now(TZ) - review_dt).total_seconds() / 3600
        overlay.is_stale = age_hours > overlay.ttl_hours
    except (ValueError, TypeError):
        overlay.is_stale = True

    return overlay


def save_overlay(overlay: StrategyOverlay) -> Path:
    """保存策略叠加参数到 JSON"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = asdict(overlay)
    # Convert MarketBias dataclasses to dicts
    data["market_biases"] = {
        k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
        for k, v in overlay.market_biases.items()
    }

    with open(OVERLAY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 策略叠加已保存: {OVERLAY_FILE}")
    return OVERLAY_FILE


def load_overlay() -> Optional[StrategyOverlay]:
    """加载已保存的策略叠加参数"""
    if not OVERLAY_FILE.exists():
        return None

    try:
        with open(OVERLAY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        overlay = StrategyOverlay(**{k: v for k, v in data.items()
                                     if k in StrategyOverlay.__dataclass_fields__})

        # Rebuild MarketBias objects
        biases = {}
        for mk, mv in data.get("market_biases", {}).items():
            biases[mk] = MarketBias(**mv) if isinstance(mv, dict) else mv
        overlay.market_biases = biases

        # Rebuild ScenarioWeight objects
        overlay.scenarios = [ScenarioWeight(**s) for s in data.get("scenarios", [])]

        # Check staleness
        try:
            gen_dt = datetime.fromisoformat(overlay.generated_at)
            age_hours = (datetime.now(TZ) - gen_dt).total_seconds() / 3600
            overlay.is_stale = age_hours > overlay.ttl_hours
        except (ValueError, TypeError):
            overlay.is_stale = True

        return overlay
    except Exception as e:
        print(f"⚠️ 加载 overlay 失败: {e}")
        return None


# ═══════════════════════════════════════════
# 策略应用函数 (供交易引擎调用)
# ═══════════════════════════════════════════

def apply_overlay_to_signals(signals: List[dict], overlay: StrategyOverlay) -> List[dict]:
    """
    将策略叠加应用到信号列表

    调整项:
      1. 优先资产的置信度 +10%
      2. 回避资产的置信度 -30% (或直接降为 HOLD)
      3. 市场偏见一致的 BUY 置信度 +5%
      4. 市场偏见相反的 BUY 置信度 -15%

    Returns:
        调整后的信号列表
    """
    if overlay.is_stale:
        return signals  # 陈旧不调整

    adjusted = []
    for sig in signals:
        sig = dict(sig)  # copy
        symbol_base = sig.get("symbol", "").replace("/USDT", "").replace("USDT", "")

        # 回避资产 → 降级
        if symbol_base in overlay.avoid_assets:
            if sig.get("action") == "BUY":
                sig["action"] = "HOLD"
                sig["confidence"] = max(0.1, sig.get("confidence", 0.5) * 0.5)
                sig["overlay_note"] = f"AI复盘回避: {symbol_base}"

        # 优先资产 → 增强
        if symbol_base in overlay.favor_assets:
            sig["confidence"] = min(1.0, sig.get("confidence", 0.5) * 1.15)
            sig["overlay_note"] = f"AI复盘优先: {symbol_base}"

        # 市场偏见调整
        crypto_bias = overlay.market_biases.get("crypto")
        if crypto_bias and sig.get("action") == "BUY":
            if crypto_bias.direction == "bullish":
                sig["confidence"] = min(1.0, sig.get("confidence", 0.5) * 1.08)
            elif crypto_bias.direction == "bearish":
                sig["confidence"] = max(0.1, sig.get("confidence", 0.5) * 0.85)

        adjusted.append(sig)

    return adjusted


def apply_overlay_to_risk(overlay: StrategyOverlay) -> dict:
    """
    返回调整后的风控参数

    Returns:
        {"stop_loss_pct": float, "take_profit_pct": float,
         "max_positions": int, "single_trade_max_pct": float,
         "daily_loss_meltdown": float}
    """
    if overlay.is_stale:
        return {}  # 使用默认值

    return {
        "stop_loss_pct": overlay.stop_loss_pct,
        "take_profit_pct": overlay.take_profit_pct,
        "max_positions": overlay.max_crypto_positions,
        "single_trade_max_pct": overlay.single_trade_max_pct,
        "daily_loss_meltdown": overlay.daily_loss_meltdown_pct,
        "confidence_multiplier": overlay.confidence_multiplier,
    }


def get_dont_do_list(overlay: StrategyOverlay) -> List[str]:
    """获取「不做什么」清单"""
    if overlay.is_stale:
        return []
    return overlay.dont_do


def get_risk_reminders(overlay: StrategyOverlay) -> List[str]:
    """获取风控提醒"""
    if overlay.is_stale:
        return []
    return overlay.risk_reminders


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def print_overlay(overlay: StrategyOverlay):
    """打印可读的策略叠加摘要"""
    print("\n" + "=" * 60)
    print(f"🐾 Yina AI复盘 → 策略叠加参数")
    print("=" * 60)
    print(f"📅 复盘日期: {overlay.review_date}")
    print(f"⏰ 生成时间: {overlay.generated_at[:19]}")
    print(f"📂 来源: {overlay.source_review}")
    print(f"{'⚠️ 数据陈旧(>24h)' if overlay.is_stale else '✅ 数据新鲜'}")
    print()

    print(f"💡 核心判断: {overlay.core_judgment}")
    print(f"   置信度: {overlay.core_confidence:.0%}")
    print()

    print("📊 市场方向偏见:")
    for market, bias in overlay.market_biases.items():
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(bias.direction, "⚪")
        print(f"   {emoji} {market}: {bias.direction} (置信 {bias.confidence:.0%})")

    print(f"\n🎯 情景推演 ({len(overlay.scenarios)}个):")
    for s in overlay.scenarios:
        print(f"   [{s.probability}%] {s.name}")
        print(f"        Crypto: {s.crypto_direction} | 触发: {s.trigger_condition[:60]}...")

    print(f"\n💰 仓位参数:")
    print(f"   最大持仓: {overlay.max_crypto_positions} | 单笔上限: {overlay.single_trade_max_pct:.0%}")
    print(f"   置信乘数: {overlay.confidence_multiplier:.2f}")
    print(f"   止损: {overlay.stop_loss_pct:+.0f}% | 止盈: +{overlay.take_profit_pct:.0f}%")

    if overlay.favor_assets:
        print(f"\n⭐ 优先资产: {', '.join(overlay.favor_assets)}")
    if overlay.avoid_assets:
        print(f"🚫 回避资产: {', '.join(overlay.avoid_assets)}")

    if overlay.dont_do:
        print(f"\n🛑 不做什么:")
        for d in overlay.dont_do:
            print(f"   • {d}")

    if overlay.risk_reminders:
        print(f"\n⚠️ 风控提醒:")
        for r in overlay.risk_reminders:
            print(f"   • {r}")

    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Yina AI复盘 → 策略桥接器")
    parser.add_argument("--review", type=str, help="指定复盘文件 (e.g. 2026-06-18_evening)")
    parser.add_argument("--show", action="store_true", help="显示当前 overlay")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--stale-ok", action="store_true", help="即使数据陈旧也使用")
    args = parser.parse_args()

    if args.show:
        overlay = load_overlay()
        if overlay:
            if args.json:
                print(json.dumps(asdict(overlay), ensure_ascii=False, indent=2))
            else:
                print_overlay(overlay)
        else:
            print("⚠️ 暂无策略叠加数据，请先运行复盘引擎")
        return

    # 加载复盘
    if args.review:
        review_path = REVIEWS_DIR / f"{args.review}.json"
        if not review_path.exists():
            print(f"❌ 复盘文件不存在: {review_path}")
            sys.exit(1)
        with open(review_path, 'r', encoding='utf-8') as f:
            review = json.load(f)
    else:
        review = load_latest_review()
        if not review:
            print("❌ 没有找到复盘记录，请先运行 post_market_review.py")
            sys.exit(1)

    # 构建 & 保存
    print(f"🔍 分析复盘: {review.get('date', '?')}_{review.get('mode', '?')}")
    overlay = build_overlay(review)

    if overlay.is_stale and not args.stale_ok:
        print(f"⚠️ 复盘数据已超过{overlay.ttl_hours}小时 (陈旧)，使用 --stale-ok 强制生成")

    save_overlay(overlay)
    print_overlay(overlay)

    # 也输出 JSON
    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(asdict(overlay), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
