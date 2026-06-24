#!/usr/bin/env python3
"""
🔭 Multi-Perspective Scanner — 多视角标的扫描器

每个视角独立产生信号，然后汇聚到共识引擎(L2)投票。

四大独立视角:
  1. 🔍 K线技术面 — 裸K+ML动量+Alpha+激进+均值回归 (已有)
  2. 💰 权哥价值投资 — 趋势风口+估值三灯+进攻防守+生意思维
  3. 🧮 西蒙斯量化统计 — 统计优势+多频段对齐+风险调整
  4. 🐼 熊猫裸K — 三步读图法+多时间框架共振 (裸K策略已有)

设计原则:
  - 每个视角独立运行，不互相污染
  - 标准化信号格式，直接接入 StrategyConsensusEngine
  - 权哥和西蒙斯视角赋予"high"质量等级 (独立来源)
  - 失败降级: 数据不足时优雅返回空信号
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

# 聚焦资产: 流动性最好的标的，避免分散
FOCUS_ASSETS = [
    # Tier 1: 大盘
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    # Tier 2: 高流动性山寨
    "AVAX/USDT", "NEAR/USDT", "RENDER/USDT",
    # Stock swaps (半导体/AI 核心)
    "AAPL/USDT", "NVDA/USDT", "AMZN/USDT",
]

# 权哥2026年认定的"飞天猪"赛道 (风口行业)
QUANGE_HOT_SECTORS_2026 = {
    "半导体": ["NVDA/USDT", "TSLA/USDT"],
    "AI基础设施": ["RENDER/USDT", "NEAR/USDT", "FET/USDT"],
    "L1公链": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
    "太空经济": [],  # 无对应合约
}

# 权哥M+组合配置参考 (2026)
QUANGE_M_PORTFOLIO = {
    "SMH": 0.595, "QQQM": 0.11, "XLK": 0.07,
    "TSLA": 0.04, "GOOG": 0.042, "BTC": 0.109, "ETH": 0.034,
}


# ═══════════════════════════════════════════════════════
# Perspective 1: 权哥价值投资视角
# ═══════════════════════════════════════════════════════

class QuangeValuePerspective:
    """
    权哥价值投资信号生成器

    核心理念: "用生意思维做投资" + "趋势风口(飞天猪)"
    三大判断维度:
      1. 风口判断 — 这个赛道现在是风口吗？
      2. 估值判断 — 三灯系统: 贵不贵？
      3. 模式判断 — 进攻/防守/观望

    信号规则:
      - 在风口 + 估值合理 + 进攻模式 → BUY (score 70-90)
      - 不在风口 + 估值贵 → SELL (score 60-80)
      - 防守模式 → 任何买入信号降级
    """

    QUALITY = "high"  # 独立价值投资视角，高质量
    STRATEGY_NAME = "权哥价值投资"

    def scan(self, market_context: dict) -> List[dict]:
        """
        生成权哥视角的交易信号

        Args:
            market_context: {
                "fg_value": int, "btc_trend": str, "btc_price": float,
                "sector_flows": dict, "funding_rates": dict,
            }
        Returns:
            [{symbol, action, score, confidence, strategy_name, reasons}, ...]
        """
        fg = market_context.get("fg_value", 50)
        btc_trend = market_context.get("btc_trend", "sideways")

        signals = []

        # Step 1: 判断进攻/防守模式 (全哥模型5: 顺势而为)
        # 关键: F&G极端恐惧 + BTC涨 = 反向买入机会 (股灾翻身符!)
        #       F&G贪婪 + BTC跌 = 危险信号
        if fg <= 25 and btc_trend == "uptrend":
            mode = "contrarian_buy"   # 🐻→🐂 别人恐惧我贪婪
            mode_multiplier = 1.10    # 加分: 极端恐惧+趋势向上=最佳买入时机
        elif fg <= 25 and btc_trend == "downtrend":
            mode = "defense"
            mode_multiplier = 0.5     # 恐惧+下跌 = 防守
        elif fg >= 50 and btc_trend == "uptrend":
            mode = "offense"
            mode_multiplier = 1.0     # 正常进攻
        elif fg >= 75 and btc_trend == "downtrend":
            mode = "danger"           # 贪婪+下跌 = 危险
            mode_multiplier = 0.4
        else:
            mode = "neutral"
            mode_multiplier = 0.85

        # Step 2: 扫描聚焦资产
        for symbol in FOCUS_ASSETS:
            base = symbol.split("/")[0]
            sector = self._find_sector(symbol)

            # 2a. 风口判断 (飞天猪模型)
            is_hot_sector = sector is not None
            if not is_hot_sector:
                continue  # 不在任何风口赛道，跳过

            # 2b. 生意思维: 这个"商品"值得买吗？
            # L1公链 = 基础设施，长期持有价值
            # AI/半导体 = 当前最大的飞天猪
            # 其他 = 需要更多证据
            if sector == "L1公链":
                business_score = 75 if base in ("BTC", "ETH") else 60
            elif sector in ("半导体", "AI基础设施"):
                business_score = 80
            else:
                business_score = 55

            # 2c. 估值判断 (简化版 — 用F&G和BTC趋势代理)
            # 极端恐惧 = 低估 (全哥: "恐惧变机会")
            # 贪婪 = 高估
            if fg <= 25:
                valuation_score = 70  # 低估，买入机会
                valuation_note = "极端恐惧→低估机会(股灾翻身符)"
            elif fg >= 75:
                valuation_score = 30  # 高估
                valuation_note = "贪婪→估值偏高"
            else:
                valuation_score = 50
                valuation_note = "估值中性"

            # 2d. 趋势确认
            if btc_trend == "uptrend":
                trend_score = 70
            elif btc_trend == "downtrend":
                trend_score = 30
            else:
                trend_score = 50

            # 2e. 综合评分
            raw_score = (business_score * 0.40 +
                        valuation_score * 0.30 +
                        trend_score * 0.30)
            score = raw_score * mode_multiplier

            # 方向判定
            if score >= 60:
                action = "BUY"
                confidence = min(0.85, 0.5 + (score - 60) / 100)
                reasons = [f"风口:{sector}", valuation_note,
                          f"模式:{mode}", f"生意分:{business_score}"]
            elif score <= 40:
                action = "SELL"
                confidence = min(0.75, 0.5 + (40 - score) / 100)
                reasons = [f"非风口/估值贵", valuation_note,
                          f"模式:{mode}"]
            else:
                continue  # HOLD，不产生信号

            signals.append({
                "symbol": symbol,
                "name": base,
                "action": action,
                "price": 0,
                "score": round(score),
                "confidence": round(confidence, 2),
                "reasons": reasons,
                "risk_level": "medium",
                "suggested_size": 0.03 if mode == "defense" else 0.05,
                "stop_loss": 0,
                "take_profit": 0,
                "strategy_name": self.STRATEGY_NAME,
                "perspective": "value",
            })

        return signals

    def _find_sector(self, symbol: str) -> Optional[str]:
        for sector, assets in QUANGE_HOT_SECTORS_2026.items():
            if symbol in assets:
                return sector
        return None


# ═══════════════════════════════════════════════════════
# Perspective 2: 西蒙斯量化统计视角
# ═══════════════════════════════════════════════════════

class SimonsQuantPerspective:
    """
    西蒙斯量化统计信号生成器

    核心理念: "It's just math" — 只关注统计上显著的重复模式
    三大判断维度:
      1. 统计优势 — 是否存在正期望值的信号？
      2. 多频段对齐 — 日内/日频/周频是否一致？
      3. 风险调整 — 波动率调整后的收益是否显著？

    信号规则:
      - 多频段对齐 + 正期望值 → BUY/SELL (score 65-85)
      - 单频段信号 → 弱信号 (score 40-55)
      - 噪音 → 无信号
    """

    QUALITY = "high"  # 独立量化统计视角
    STRATEGY_NAME = "西蒙斯量化统计"

    def scan(self, market_context: dict,
             ohlcv_data: Optional[Dict[str, list]] = None) -> List[dict]:
        """
        生成西蒙斯视角的交易信号

        Args:
            market_context: 市场环境数据
            ohlcv_data: {symbol: [(ts,o,h,l,c,vol), ...]} 可选
        """
        signals = []
        fg = market_context.get("fg_value", 50)
        btc_trend = market_context.get("btc_trend", "sideways")

        for symbol in FOCUS_ASSETS:
            base = symbol.split("/")[0]

            # 1. 均值回归检验: 极端恐惧时，统计上回归均值的概率更高
            # 西蒙斯会说: "恐惧指数20的历史数据表明，未来5天BTC正收益概率>65%"
            if fg <= 30:
                mean_reversion_score = 70  # 统计上有利
                mr_confidence = 0.60 + (30 - fg) * 0.01  # 恐惧越深，回归越确定
            elif fg >= 70:
                mean_reversion_score = 30  # 统计上不利(可能均值回归向下)
                mr_confidence = 0.55
            else:
                mean_reversion_score = 50
                mr_confidence = 0.45

            # 2. 趋势统计: BTC趋势的持续性
            # "趋势是金融市场中最稳健的统计现象" — 西蒙斯
            if btc_trend == "uptrend":
                trend_persistence = 0.60  # 上涨趋势次日继续上涨的概率
                trend_score = 65
            elif btc_trend == "downtrend":
                trend_persistence = 1 - 0.60  # 下跌趋势持续
                trend_score = 35
            else:
                trend_persistence = 0.50
                trend_score = 50

            # 3. 波动率调整: 低波动环境下信号更可靠
            vol = market_context.get("vol_level", "medium")
            if vol == "low":
                vol_confidence_boost = 1.15
            elif vol == "high":
                vol_confidence_boost = 0.85
            else:
                vol_confidence_boost = 1.0

            # 4. 多频段对齐 (简化: 用均值回归+趋势是否同向)
            # 两者同向 = 多频段对齐 → 强信号
            mr_bullish = mean_reversion_score >= 55
            trend_bullish = trend_score >= 55

            if mr_bullish and trend_bullish:
                # 均值回归+趋势都看多 → 强BUY
                action = "BUY"
                raw_score = (mean_reversion_score + trend_score) / 2
                confidence = min(0.85, (mr_confidence + trend_persistence) / 2 * vol_confidence_boost)
                reasons = ["多频段对齐", f"均值回归看多({mean_reversion_score})",
                          f"趋势持续({trend_score})"]
            elif not mr_bullish and not trend_bullish:
                # 两者都看空 → 强SELL
                action = "SELL"
                raw_score = (100 - mean_reversion_score + 100 - trend_score) / 2
                confidence = min(0.75, ((1-mr_confidence) + (1-trend_persistence)) / 2 * vol_confidence_boost)
                reasons = ["多频段对齐看空", f"均值回归看空({mean_reversion_score})",
                          f"趋势走弱({trend_score})"]
            elif abs(mean_reversion_score - trend_score) > 20:
                # 明显分歧 → 弱信号，跟随较强的那个
                if mean_reversion_score > trend_score:
                    action = "BUY"
                    raw_score = mean_reversion_score * 0.7  # 降级
                else:
                    action = "SELL"
                    raw_score = (100 - trend_score) * 0.7
                confidence = 0.45  # 低置信
                reasons = ["频段分歧", f"MR={mean_reversion_score} vs Trend={trend_score}"]
            else:
                continue  # 无明确信号

            # L1链 (BTC/ETH/SOL) 流动性好，统计更可靠
            if base in ("BTC", "ETH", "SOL"):
                confidence = min(0.90, confidence * 1.10)

            signals.append({
                "symbol": symbol,
                "name": base,
                "action": action,
                "price": 0,
                "score": round(raw_score),
                "confidence": round(confidence, 2),
                "reasons": reasons,
                "risk_level": "medium",
                "suggested_size": 0.04,
                "stop_loss": 0,
                "take_profit": 0,
                "strategy_name": self.STRATEGY_NAME,
                "perspective": "quant",
            })

        return signals


# ═══════════════════════════════════════════════════════
# Perspective 4: Serenity 长期价值投资视角
# ═══════════════════════════════════════════════════════

class SerenityValuePerspective:
    """
    Serenity 长期价值投资信号生成器

    核心理念: 长期持有优质资产，关注商业模式和成长性
    三大判断维度:
      1. 资产质量 — 这个标的是否有长期持有的价值？
      2. 成长空间 — 未来12个月的成长预期
      3. 安全边际 — 当前价格是否提供了足够的安全边际？

    信号规则:
      - 优质资产 + 高成长 + 安全边际 → BUY (score 70-90)
      - 优质资产 + 估值过高 → HOLD (观望)
      - 劣质资产 → 无信号
    """

    QUALITY = "high"
    STRATEGY_NAME = "Serenity长期价值"

    # 长期优质资产清单 (基本面过硬)
    LONG_TERM_QUALITY = {
        "BTC/USDT": 90,   # 数字黄金, 长期持有价值最高
        "ETH/USDT": 80,   # 智能合约平台龙头
        "SOL/USDT": 70,   # 高性能L1
        "AAPL/USDT": 85,  # 全球最强消费科技
        "NVDA/USDT": 90,  # AI芯片垄断
        "AMZN/USDT": 80,  # 电商+云计算双引擎
        "RENDER/USDT": 65, # AI基础设施
        "NEAR/USDT": 60,   # L1新秀
    }

    # 成长空间评分 (12个月展望)
    GROWTH_OUTLOOK = {
        "BTC/USDT": 75,    # ETF持续流入 + 机构采用
        "ETH/USDT": 70,     # L2生态扩展
        "SOL/USDT": 65,     # 高性能链竞争激烈
        "AAPL/USDT": 60,    # 成熟期，成长放缓
        "NVDA/USDT": 90,    # AI需求爆发
        "AMZN/USDT": 70,    # 云计算增长+AI
        "RENDER/USDT": 80,  # AI算力需求
        "NEAR/USDT": 70,    # 生态建设
    }

    def scan(self, market_context: dict) -> List[dict]:
        """
        生成Serenity视角的交易信号

        核心理念: 只在有安全边际时买入优质资产，长期持有
        """
        fg = market_context.get("fg_value", 50)
        btc_trend = market_context.get("btc_trend", "sideways")
        signals = []

        for symbol in FOCUS_ASSETS:
            if symbol not in self.LONG_TERM_QUALITY:
                continue

            quality = self.LONG_TERM_QUALITY[symbol]
            growth = self.GROWTH_OUTLOOK.get(symbol, 50)

            # 安全边际: 极端恐惧 = 优质资产打折
            if fg <= 25:
                margin_of_safety = 85  # 别人恐惧时安全边际最高
                safety_note = "极端恐惧→优质资产打折(安全边际高)"
            elif fg <= 40:
                margin_of_safety = 65
                safety_note = "恐惧→有一定安全边际"
            elif fg <= 60:
                margin_of_safety = 50
                safety_note = "中性→合理估值"
            else:
                margin_of_safety = 30
                safety_note = "贪婪→估值偏高(安全边际低)"

            # 趋势确认
            if btc_trend == "uptrend":
                trend_bonus = 1.05
            elif btc_trend == "downtrend":
                trend_bonus = 0.90
            else:
                trend_bonus = 1.0

            # 综合评分: 质量40% + 成长30% + 安全边际30%
            raw_score = (quality * 0.40 + growth * 0.30 + margin_of_safety * 0.30) * trend_bonus

            # 长期视角: 质量分≥80的资产，即使短期不利也保持BUY倾向
            if quality >= 80 and margin_of_safety >= 50:
                action = "BUY"
                confidence = min(0.85, 0.55 + (quality - 80) * 0.01 + (margin_of_safety - 50) * 0.005)
                reasons = [f"优质资产(质量{quality})", safety_note,
                          f"成长空间{growth}", "长期持有视角"]
            elif raw_score >= 65:
                action = "BUY"
                confidence = min(0.80, 0.50 + (raw_score - 65) / 80)
                reasons = [f"综合评分{raw_score:.0f}", safety_note,
                          f"质量{quality} 成长{growth}"]
            elif raw_score <= 35:
                action = "SELL"
                confidence = min(0.70, 0.50 + (35 - raw_score) / 80)
                reasons = ["估值过高+安全边际不足", safety_note]
            else:
                continue  # HOLD

            signals.append({
                "symbol": symbol,
                "name": symbol.split("/")[0],
                "action": action,
                "price": 0,
                "score": round(raw_score),
                "confidence": round(confidence, 2),
                "reasons": reasons,
                "risk_level": "low",  # 长期价值 = 低风险
                "suggested_size": 0.04,
                "stop_loss": 0.10,     # ≥10% SL (长期视角，宽止损)
                "take_profit": 0,       # 动态止盈，不设固定值
                "strategy_name": self.STRATEGY_NAME,
                "perspective": "value_longterm",
            })

        return signals


# ═══════════════════════════════════════════════════════
# Multi-Perspective Scanner (Orchestrator)
# ═══════════════════════════════════════════════════════

class MultiPerspectiveScanner:
    """
    多视角扫描器 — 协调所有独立视角，产出综合信号

    工作流:
      1. 收集市场环境数据 (F&G, BTC, vol, sector flows)
      2. 运行所有独立视角扫描
      3. 标准化信号 → 送入共识引擎
    """

    def __init__(self):
        self.quange = QuangeValuePerspective()
        self.simons = SimonsQuantPerspective()
        # Serenity 长期价值: 已禁用 (待明确方法论来源后重新激活) — Chase哥说先不要了

    def scan_all(self,
                 market_context: dict,
                 technical_signals: Optional[List[dict]] = None,
                 ohlcv_data: Optional[Dict] = None) -> List[dict]:
        """
        运行所有独立视角扫描，返回综合信号列表

        Args:
            market_context: {"fg_value", "btc_trend", "vol_level", "sector_flows"}
            technical_signals: 已有的技术面信号 (来自 strategy_runner)
            ohlcv_data: 可选OHLCV数据
        """
        all_signals = []

        # 1. 技术面信号 (已有)
        if technical_signals:
            all_signals.extend(technical_signals)

        # 2. 权哥价值投资视角
        try:
            value_signals = self.quange.scan(market_context)
            if value_signals:
                buy_n = sum(1 for s in value_signals if s["action"] == "BUY")
                sell_n = sum(1 for s in value_signals if s["action"] == "SELL")
                log.info(f"💰 权哥视角: {len(value_signals)}信号 ({buy_n}BUY/{sell_n}SELL)")
                all_signals.extend(value_signals)
        except Exception as e:
            log.warning(f"权哥视角扫描失败: {e}")

        # 3. 西蒙斯量化统计视角
        try:
            quant_signals = self.simons.scan(market_context, ohlcv_data)
            if quant_signals:
                buy_n = sum(1 for s in quant_signals if s["action"] == "BUY")
                sell_n = sum(1 for s in quant_signals if s["action"] == "SELL")
                log.info(f"🧮 西蒙斯视角: {len(quant_signals)}信号 ({buy_n}BUY/{sell_n}SELL)")
                all_signals.extend(quant_signals)
        except Exception as e:
            log.warning(f"西蒙斯视角扫描失败: {e}")

        # 4. Serenity 长期价值 — 已禁用
        # serenity_signals = self.serenity.scan(market_context)

        return all_signals


# ═══════════════════════════════════════════════════════
# Convenience Function
# ═══════════════════════════════════════════════════════

def scan_multi_perspective(fg_value: int = 50,
                           fg_label: str = "Neutral",
                           btc_trend: str = "sideways",
                           vol_level: str = "medium",
                           sector_flows: Optional[dict] = None,
                           technical_signals: Optional[List[dict]] = None,
                           ohlcv_data: Optional[Dict] = None) -> List[dict]:
    """
    便捷函数 — 一键多视角扫描

    Args:
        fg_value: F&G指数
        fg_label: F&G分类
        btc_trend: BTC趋势
        vol_level: 波动率等级
        sector_flows: 板块资金流
        technical_signals: 已有技术面信号
        ohlcv_data: OHLCV数据
    """
    context = {
        "fg_value": fg_value,
        "fg_label": fg_label,
        "btc_trend": btc_trend,
        "vol_level": vol_level,
        "sector_flows": sector_flows or {},
    }
    scanner = MultiPerspectiveScanner()
    return scanner.scan_all(context, technical_signals, ohlcv_data)


# ═══════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔭 Multi-Perspective Scanner — 测试\n")

    context = {
        "fg_value": 23,
        "fg_label": "Extreme Fear",
        "btc_trend": "uptrend",
        "vol_level": "low",
        "sector_flows": {},
    }

    scanner = MultiPerspectiveScanner()

    # 测试 权哥
    print("=" * 60)
    print("💰 权哥价值投资视角")
    print("=" * 60)
    q_signals = scanner.quange.scan(context)
    for s in q_signals:
        icon = "🟢" if s["action"] == "BUY" else "🔴"
        print(f"  {icon} {s['symbol']:15s} {s['action']:4s} "
              f"score={s['score']:3d} conf={s['confidence']:.0%} "
              f"| {', '.join(s['reasons'][:3])}")

    # 测试 西蒙斯
    print(f"\n{'='*60}")
    print("🧮 西蒙斯量化统计视角")
    print("=" * 60)
    s_signals = scanner.simons.scan(context)
    for s in s_signals:
        icon = "🟢" if s["action"] == "BUY" else "🔴"
        print(f"  {icon} {s['symbol']:15s} {s['action']:4s} "
              f"score={s['score']:3d} conf={s['confidence']:.0%} "
              f"| {', '.join(s['reasons'][:3])}")

    # 合并
    all_s = scanner.scan_all(context)
    print(f"\n📊 多视角总计: {len(all_s)} 信号")
    bu = sum(1 for s in all_s if s["action"] == "BUY")
    se = sum(1 for s in all_s if s["action"] == "SELL")
    print(f"   BUY: {bu}, SELL: {se}")
    for s in sorted(all_s, key=lambda x: x["confidence"], reverse=True):
        icon = "🟢" if s["action"] == "BUY" else "🔴"
        print(f"   {icon} [{s['strategy_name'][:8]:8s}] {s['symbol']:15s} "
              f"{s['action']:4s} {s['score']:3d} conf={s['confidence']:.0%}")
