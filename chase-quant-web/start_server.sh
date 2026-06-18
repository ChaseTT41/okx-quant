#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  🐾 Chase量化策略 — 生产环境一键启动脚本                    ║
# ║  Usage: bash start_server.sh [soft|hard]                     ║
# ║    soft - 软重启: 只重启 API Server, 保持 tunnel URL 不变   ║
# ║    hard - 硬重启: 全杀重启, URL 会变                         ║
# ╚══════════════════════════════════════════════════════════════╝
set -e

MODE="${1:-soft}"
PORT=8766
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_SERVER="/tmp/chase-quant-server.log"
LOG_TUNNEL="/tmp/cloudflared-chase-quant.log"
LOG_WATCHDOG="/tmp/chase-quant-watchdog.log"
LOG_DAEMON="/tmp/yina-trade-daemon.log"
WATCHDOG_SCRIPT="$DIR/watchdog.sh"
ENV_FILE="$DIR/.env"

cd "$DIR"

# ── 加载 .env ──
if [ -f "$ENV_FILE" ]; then
  set -a
  source <(grep -v '^#' "$ENV_FILE" | grep -v '^$')
  set +a
  echo "✅ 已加载 .env"
else
  echo "⚠️  .env 文件不存在，跳过"
fi

# ╔══════════════════════════════════════════════════════════════╗
# ║  🩺 OpenMP 线程安全环境变量 (macOS 防 SIGSEGV)              ║
# ╚══════════════════════════════════════════════════════════════╝
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

echo "🛡️  OpenMP 线程安全: OMP_NUM_THREADS=$OMP_NUM_THREADS KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"

# ╔══════════════════════════════════════════════════════════════╗
# ║  🩺 libomp 统一性检查 (pip install 可能恢复旧版本)          ║
# ╚══════════════════════════════════════════════════════════════╝
HOMEBREW_OMP="/opt/homebrew/opt/libomp/lib/libomp.dylib"
SKLEARN_OMP="$HOME/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs/libomp.dylib"
TORCH_OMP="$HOME/Library/Python/3.9/lib/python/site-packages/torch/lib/libomp.dylib"

_fix_omp() {
  local target="$1" label="$2"
  if [ -f "$target" ] && [ ! -L "$target" ]; then
    echo "⚠️  $label 不是软链，替换为 Homebrew 版本..."
    mv "$target" "${target}.bak.$(date +%Y%m%d)" 2>/dev/null
    ln -sf "$HOMEBREW_OMP" "$target"
    echo "✅ $label 已修复"
  elif [ -L "$target" ]; then
    local dest; dest=$(readlink "$target" 2>/dev/null)
    if [ "$dest" != "$HOMEBREW_OMP" ]; then
      ln -sf "$HOMEBREW_OMP" "$target"
      echo "✅ $label 重新指向 Homebrew"
    fi
  fi
}

if [ -f "$HOMEBREW_OMP" ]; then
  _fix_omp "$SKLEARN_OMP" "sklearn libomp"
  _fix_omp "$TORCH_OMP" "torch libomp"
fi

# ── 硬重启模式：全杀 ──
if [ "$MODE" = "hard" ]; then
  echo "💣 硬重启模式: 全杀所有进程..."
  lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  pkill -f cloudflared 2>/dev/null || true
  pkill -f auto_trade_daemon 2>/dev/null || true
  pkill -f chase-quant-watchdog 2>/dev/null || true
  pkill -f "$WATCHDOG_SCRIPT" 2>/dev/null || true
  sleep 2
else
  echo "🔄 软重启模式: 只重启 API Server..."
  lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# ── 启动 API Server ──
echo "🚀 启动 API Server (port $PORT)..."
nohup python3 api_server.py --production --port $PORT > "$LOG_SERVER" 2>&1 &
sleep 3

if lsof -ti:$PORT > /dev/null 2>&1; then
  SERVER_PID=$(lsof -ti:$PORT | head -1)
  echo "✅ API Server 已启动 (PID: $SERVER_PID)"
else
  echo "❌ API Server 启动失败！查看日志: tail -50 $LOG_SERVER"
  exit 1
fi

# ── 启动 cloudflared tunnel (仅硬重启) ──
if [ "$MODE" = "hard" ]; then
  echo "🌐 启动 Cloudflare Tunnel..."
  nohup cloudflared tunnel --url "http://localhost:$PORT" > "$LOG_TUNNEL" 2>&1 &
  sleep 4
  TUNNEL_URL=$(grep -o 'https://[^ ]*\.trycloudflare\.com' "$LOG_TUNNEL" 2>/dev/null | tail -1)
  if [ -n "$TUNNEL_URL" ]; then
    echo "🔗 公网地址: $TUNNEL_URL"
    echo "⚠️  Tunnel URL 已变更！请更新 vercel.json 中的 destination 字段:"
    echo "   sed -i '' 's|https://[^/]*\.trycloudflare\.com|${TUNNEL_URL}|g' $DIR/vercel.json"
    echo "   然后 git push 让 Vercel 自动部署新配置"
  else
    echo "⚠️  Tunnel URL 未就绪，检查: tail -20 $LOG_TUNNEL"
  fi
else
  TUNNEL_URL=$(grep -o 'https://[^ ]*\.trycloudflare\.com' "$LOG_TUNNEL" 2>/dev/null | tail -1)
  echo "🔗 公网地址 (不变): ${TUNNEL_URL:-未知}"
fi

# ── 启动交易守护进程 ──
if ! pgrep -f auto_trade_daemon > /dev/null 2>&1; then
  echo "🤖 启动交易守护进程..."
  nohup python3 auto_trade_daemon.py --daemon > "$LOG_DAEMON" 2>&1 &
  echo "✅ 交易守护进程已启动"
else
  echo "✅ 交易守护进程已在运行"
fi

# ── 启动 Watchdog 自愈守护 ──
echo "🐶 启动 Watchdog 自愈守护..."
pkill -f "$WATCHDOG_SCRIPT" 2>/dev/null || true
pkill -f chase-quant-watchdog 2>/dev/null || true
sleep 1
nohup bash "$WATCHDOG_SCRIPT" > "$LOG_WATCHDOG" 2>&1 &
echo "✅ Watchdog 已启动 (每30s自检, 日志: $LOG_WATCHDOG)"

# ── 健康检查 ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🐾 全部就绪! "
echo "  📊 仪表板: http://localhost:$PORT"
echo "  🌍 Vercel:  https://chase-quant-web.vercel.app"
echo "  📖 API文档: http://localhost:$PORT/docs"
echo "  ❤️  健康:   http://localhost:$PORT/api/health"
echo "  🐶 Watchdog: 自动守护中"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
