#!/bin/bash
# ============================================
# 抖音每日智能分析 — 一键工作流
# 1. 提取收藏+喜欢
# 2. AI 分析可执行任务
# 3. 保存报告
# ============================================
set -e

SCRIPT_DIR="$HOME/.claude/tools"
EXTRACTOR="$SCRIPT_DIR/douyin-fav-extractor.py"
ANALYZER="$SCRIPT_DIR/douyin-daily-analyzer.py"
REPORT="/tmp/douyin_report.json"
RESULT="/tmp/douyin_fav_result.json"

echo "════════════════════════════════════════"
echo "🐾 Yina 抖音每日分析工作流"
echo "📅 $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════"
echo ""

# ── Step 1: 提取 ──
echo "📥 Step 1/2: 提取收藏+喜欢..."
python3 "$EXTRACTOR"
if [ $? -ne 0 ]; then
    echo "❌ 提取失败！"
    exit 1
fi
echo ""

# ── Step 2: 分析 ──
echo "🧠 Step 2/2: AI 智能分析..."
python3 "$ANALYZER"
if [ $? -ne 0 ]; then
    echo "❌ 分析失败！"
    exit 1
fi
echo ""

# ── 最终输出 ──
echo "════════════════════════════════════════"
echo "✅ 完成！"
echo "📊 原始数据: $RESULT"
echo "📋 分析报告: $REPORT"
echo "════════════════════════════════════════"
