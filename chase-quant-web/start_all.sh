#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Chase量化策略 — 一键启动全部服务                          ║
# ║  用法: bash start_all.sh                                    ║
# ║  包含: API Server + Cloudflare隧道 + Streamlit + 交易守护   ║
# ╚══════════════════════════════════════════════════════════════╝

set -e
PROJECT_DIR="/Users/chasett/yina-app/chase-quant-web"
STREAMLIT_BIN="/Users/chasett/Library/Python/3.9/bin/streamlit"

echo "🚀 Chase量化策略 — 全服务启动"
echo "================================"
echo ""

# ── 1. 清理旧进程 ──
echo "🧹 清理旧进程..."
lsof -ti:8766 | xargs kill -9 2>/dev/null && echo "   已清理端口8766" || echo "   端口8766空闲"
lsof -ti:8501 | xargs kill -9 2>/dev/null && echo "   已清理端口8501" || echo "   端口8501空闲"
pkill -f "cloudflared tunnel" 2>/dev/null && echo "   已清理cloudflared" || echo "   cloudflared未运行"
pkill -f auto_trade_daemon 2>/dev/null && echo "   已清理交易守护" || echo "   交易守护未运行"
pkill -f market_sentiment_daemon 2>/dev/null && echo "   已清理情绪守护" || echo "   情绪守护未运行"
pkill -f chase-quant-watchdog 2>/dev/null && echo "   已清理看门狗" || echo "   看门狗未运行"
sleep 1

# ── 2. 加载环境变量 ──
cd "$PROJECT_DIR"
export $(grep -v '^#' .env | xargs 2>/dev/null)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_DYNAMIC=FALSE

# ── 3. 启动 API Server ──
echo ""
echo "🌐 启动 API Server (8766)..."
nohup python3 api_server.py --production --port 8766 > /tmp/chase-quant-server.log 2>&1 &
API_PID=$!
sleep 2
if lsof -ti:8766 > /dev/null 2>&1; then
    echo "   ✅ API Server 就绪 (PID: $API_PID)"
else
    echo "   ❌ API Server 启动失败！检查日志: /tmp/chase-quant-server.log"
    exit 1
fi

# ── 4. 启动 Cloudflare 隧道 ──
echo "☁️  启动 Cloudflare 隧道..."
nohup cloudflared tunnel --url http://localhost:8766 > /tmp/cloudflared-chase-quant.log 2>&1 &
CF_PID=$!
sleep 4
CF_URL=$(grep -o 'https://[^ ]*\.trycloudflare\.com' /tmp/cloudflared-chase-quant.log | tail -1)
if [ -n "$CF_URL" ]; then
    echo "   ✅ 公网地址: $CF_URL"
else
    echo "   ⚠️  隧道启动中，稍后查看 /tmp/cloudflared-chase-quant.log"
fi

# ── 5. 启动 Streamlit 仪表板 ──
echo "📊 启动 Streamlit 仪表板 (8501)..."
export STREAMLIT_SERVER_HEADLESS=true
nohup "$STREAMLIT_BIN" run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false > /tmp/chase-quant-streamlit.log 2>&1 &
ST_PID=$!
sleep 4
if lsof -ti:8501 > /dev/null 2>&1; then
    echo "   ✅ Streamlit 就绪 (PID: $ST_PID)"
    echo "   本地: http://localhost:8501"
else
    echo "   ❌ Streamlit 启动失败！检查日志: /tmp/chase-quant-streamlit.log"
fi

# ── 6. 启动市场情绪守护 ──
echo "🏛️  启动市场情绪守护进程..."
nohup python3 -u market_sentiment_daemon.py --daemon > /tmp/yina-sentiment-daemon.log 2>&1 &
SD_PID=$!
sleep 1
if pgrep -f market_sentiment_daemon > /dev/null 2>&1; then
    echo "   ✅ 情绪守护就绪 (PID: $SD_PID)"
else
    echo "   ⚠️  情绪守护启动异常，检查日志"
fi

# ── 7. 启动交易守护 ──
echo "🤖 启动交易守护进程..."
nohup python3 -u auto_trade_daemon.py --daemon --live > /tmp/yina-trade-daemon.log 2>&1 &
TD_PID=$!
sleep 1
if pgrep -f auto_trade_daemon > /dev/null 2>&1; then
    echo "   ✅ 交易守护就绪 (PID: $TD_PID)"
else
    echo "   ⚠️  交易守护启动异常，检查日志"
fi

# ── 8. 启动看门狗 ──
echo "🛡️  启动自动重启看门狗..."
nohup /tmp/chase-quant-watchdog.sh &
WD_PID=$!
echo "   ✅ 看门狗就绪 (PID: $WD_PID)"

# ── 9. 状态总览 ──
echo ""
echo "================================"
echo "🎉 全部服务启动完成！"
echo ""
echo "┌──────────────────┬────────┐"
echo "│ 服务             │ 状态   │"
echo "├──────────────────┼────────┤"
printf "│ 🌐 API Server    │  ✅    │\n"
printf "│ ☁️  Cloudflare   │  ✅    │\n"
printf "│ 📊 Streamlit     │  ✅    │\n"
printf "│ 🏛️  情绪守护   │  ✅    │\n"
printf "│ 🤖 交易守护     │  ✅    │\n"
printf "│ 🛡️  看门狗      │  ✅    │\n"
printf "│ 🇭🇰 港股面板     │  ✅    │\n"
echo "└──────────────────┴────────┘"
echo ""
echo "🔗 公网: ${CF_URL:-获取中...}"
echo "🌍 Vercel: https://chase-quant-web.vercel.app"
echo "🏠 本地: http://localhost:8501"
echo ""
echo "📋 日志文件:"
echo "   API:    /tmp/chase-quant-server.log"
echo "   隧道:   /tmp/cloudflared-chase-quant.log"
echo "   Streamlit: /tmp/chase-quant-streamlit.log"
echo "   交易:   /tmp/yina-trade-daemon.log"
echo "   看门狗: /tmp/chase-quant-watchdog.log"
