#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  🐶 Chase量化 — Watchdog 自愈守护进程                       ║
# ║  每30秒检测 port 8766，挂了自动拉起                          ║
# ║  用法: bash watchdog.sh &                                    ║
# ╚══════════════════════════════════════════════════════════════╝

PORT=8766
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/chase-quant-watchdog.log"
CRASH_COUNT_FILE="/tmp/chase-quant-crash-count"
MAX_CRASHES_PER_HOUR=10  # 每小时最多自动重启10次，超过说明有问题

# ── 确保 OpenMP 环境变量在 watchdog 子进程中也生效 ──
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

echo "[$(date)] 🐶 Watchdog 启动, 守护 port $PORT, PID: $$"

# 滚动 crash 计数（每小时窗口）
touch "$CRASH_COUNT_FILE"

while true; do
  sleep 30

  # 检查 server 是否存活
  if lsof -ti:$PORT > /dev/null 2>&1; then
    continue  # 活着，继续睡
  fi

  # ── Server 挂了！准备重启 ──
  TIMESTAMP=$(date +%s)
  echo "[$(date)] ⚠️  Port $PORT 无响应 — Server 已挂!"

  # 检查最近1小时重启次数
  ONE_HOUR_AGO=$((TIMESTAMP - 3600))
  RECENT_CRASHES=$(awk -v cutoff="$ONE_HOUR_AGO" '$1 >= cutoff {count++} END {print count+0}' "$CRASH_COUNT_FILE" 2>/dev/null)

  if [ "$RECENT_CRASHES" -ge "$MAX_CRASHES_PER_HOUR" ]; then
    echo "[$(date)] 🚨 过去1小时已自动重启 $RECENT_CRASHES 次 (上限 $MAX_CRASHES_PER_HOUR)，停止自愈！请手动排查！"
    echo "[$(date)] 🚨 Crash 日志: tail -100 /tmp/chase-quant-server.log"
    # 推送通知 (如果有企微 webhook)
    if [ -n "$WECOM_WEBHOOK" ]; then
      curl -s "$WECOM_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"🚨 Chase量化 Server 反复崩溃！1小时内已达 $RECENT_CRASHES 次，Watchdog 停止自愈，请手动排查。\"}}" \
        > /dev/null 2>&1 || true
    fi
    exit 1
  fi

  # ── 执行重启 ──
  echo "$TIMESTAMP" >> "$CRASH_COUNT_FILE"
  echo "[$(date)] 🔄 第 $((RECENT_CRASHES + 1)) 次自动重启..."

  cd "$DIR"

  # 加载 .env
  if [ -f "$DIR/.env" ]; then
    set -a
    source <(grep -v '^#' "$DIR/.env" | grep -v '^$')
    set +a
  fi

  # 杀掉残留进程
  lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  sleep 1

  # 启动
  nohup python3 api_server.py --production --port $PORT >> /tmp/chase-quant-server.log 2>&1 &
  sleep 3

  if lsof -ti:$PORT > /dev/null 2>&1; then
    NEW_PID=$(lsof -ti:$PORT | head -1)
    echo "[$(date)] ✅ 自动重启成功! (PID: $NEW_PID)"
  else
    echo "[$(date)] ❌ 自动重启失败！查看日志: tail -50 /tmp/chase-quant-server.log"
  fi
done
