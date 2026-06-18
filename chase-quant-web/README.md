# 🐾 Yina OKX 量化交易系统 v2.0

[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel&style=flat-square)](https://chase-quant-web.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-okx--quant-blue?logo=github&style=flat-square)](https://github.com/ChaseTT41/okx-quant)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **多策略 ML 融合 + 市场情绪引擎 + 自适应杠杆 + 板块轮动预测**
>
> 支持加密货币 + 美股永续合约（OKX Perpetual Swaps）

---

## 🎯 系统概览

```
每10分钟扫描周期:
┌──────────────────────────────────────────────────┐
│ 🎭 情绪引擎 → OKX链上数据 + Fear&Greed           │
│ 📊 板块聚合 → 9大板块资金流向                     │
│ 🔮 轮动预测 → 下个领涨板块                        │
│ 🧠 ML信号 → Qlib GAT + LightGBM + 多策略融合     │
│ ⚡ 杠杆引擎 → 胜率驱动 1x/2x/5x/10x              │
│ 🛡️ 五层风控 → 事前→订单→持仓→组合→异常           │
│ 📱 企微推送 → 实时交易通知                        │
└──────────────────────────────────────────────────┘
```

## 📊 标的覆盖

| 类别 | 数量 | 策略深度 |
|------|------|----------|
| 🔵 加密 Tier1 | 35 | ML深度扫 (Qlib GAT + LightGBM + Alpha因子) |
| 🟢 加密 Tier2 | 45 | 技术面快扫 |
| ⚪ 加密 Tier3 | 216 | 动量过滤 |
| 🏭 美股半导体 | 20 | 情绪监控 + NVDA/MU/AMD/AVGO 专项 |
| 🤖 AI软件 | 14 | 情绪监控 + AI叙事相关性 |
| 🚀 太空科技 | 5 | SPCX/RKLB/AST SpaceMobile |
| 🇰🇷 韩国股票 | 3 | SK海力士 + 三星 + 现代 |
| 📈 ETF | 12 | SPY/QQQ/SOXL/SMH... |
| 🏆 商品 | 5 | 金银铜钯铂 |

## 🧠 策略引擎

| 策略 | 类型 | 描述 |
|------|------|------|
| **ML融合动量** | 趋势跟踪 | EMA三确认 + MACD + RSI投票 + LightGBM |
| **均值回归网格** | 反转 | 布林带 + RSI + 动态网格间距 |
| **跨市场Alpha套利** | 对冲 | 多空配对 + 动量轮动 + 协整检验 |
| **激进交易** | 高胜率狙击 | 多时间框架RSI + MACD + 布林带共振 |

## 🎭 市场情绪引擎 (`market_sentiment.py`)

五大链上数据源，独立缓存层：

| 数据 | 来源 | 缓存 TTL |
|------|------|----------|
| 💰 资金费率 | OKX `/api/v5/public/funding-rate` | 1h |
| 📊 持仓量 (OI) | OKX `/api/v5/public/open-interest` | 0.5h |
| 📐 多空比 | OKX Rubik long-short-account-ratio | 2h |
| 🔄 Taker买卖量比 | OKX Tickers 推断 | 1h |
| 😱 Fear & Greed | alternative.me | 4h |

**聚合输出：**
- 九大板块资金流向 (flow_score)
- AI叙事相关性矩阵 (NVDA↔FET, TSLA↔RENDER...)
- 板块轮动预测 (资金动量 + FG极值修正)
- 单标的情绪叠加因子 (composite_sentiment ∈ [-1,+1])

## ⚡ 自适应杠杆引擎 (`leverage_engine.py`)

**"看你的胜率有多少来决定杠杆"** — Chase哥

| 估计胜率 | 杠杆 | 仓位占比 | 止损 |
|----------|------|----------|------|
| ≥75% | **10x** | 1.5% | -1.5% |
| 60-75% | **5x** | 2.5% | -2.5% |
| 50-60% | **2x** | 3.5% | -5% |
| <50% | **1x** (现货) | 5% | -8% |

```
blended_wr = 0.6 × 历史胜率 + 0.4 × ML置信度
            ± 信号幅度修正
            - 资金费率过高降级 (>30%年化)
            - 板块资金流出降级
            - 情绪极端恐慌降级 (F&G<30)
```

**风险不变原则：** `杠杆 × 仓位 ≈ 常数 (0.15-0.17)`

## 🛡️ 五层风控体系

| 层级 | 检查项 |
|------|--------|
| 1️⃣ 事前过滤 | 最小交易量、黑名单、波动率过滤 |
| 2️⃣ 订单风控 | 单笔上限、最大杠杆10x、保证金≤10%权益 |
| 3️⃣ 持仓风控 | 总敞口≤3x、最大持仓数、止损止盈强制 |
| 4️⃣ 组合风控 | 相关性分散、最大回撤熔断、日损上限 |
| 5️⃣ 异常检测 | 连续亏损熔断、API异常降级、资金费率预警 |

## 🚀 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OKX API Key / Binance API Key

# 3. 启动全部服务 (API + 隧道 + Streamlit + 守护进程)
./start_all.sh

# 4. 或单独启动守护进程
python3 auto_trade_daemon.py --daemon      # 后台持续运行
python3 auto_trade_daemon.py --once        # 单次扫描测试
python3 auto_trade_daemon.py --once --live # 实盘单次扫描

# 5. 手动测试引擎
python3 market_sentiment.py                # 情绪引擎测试
python3 leverage_engine.py --test          # 杠杆引擎测试
python3 strategy_runner.py                 # 策略运行器测试
```

## 📋 架构

```
chase-quant-web/
├── market_sentiment.py   # 🎭 市场情绪引擎 (NEW v2.0)
├── leverage_engine.py    # ⚡ 自适应杠杆引擎 (NEW v2.0)
├── symbol_config.py      # 📋 统一标的配置中心 (NEW v2.0)
├── auto_trade_daemon.py  # 🔄 守护进程主循环
├── auto_trade.py         # 🤖 AutoTrader + 情绪叠加
├── strategy_runner.py    # 🧠 多策略并行运行器
├── strategies.py         # 📈 策略实现 (ML动量/均值回归/Alpha套利)
├── ml_signal_v5.py       # 🧬 ML信号引擎 v5 (LightGBM)
├── qlib_trainer.py       # 🎓 Qlib GAT 图神经网络训练
├── alpha_miner.py        # ⛏️ Alpha因子挖掘
├── risk.py               # 🛡️ 风控控制器
├── execution.py          # 💹 订单执行引擎
├── binance_live.py       # 📡 OKX/Binance 实盘交易接口
├── portfolio.py          # 💼 投资组合管理
├── mpt_engine.py         # 📊 现代投资组合理论引擎
├── wechat_report.py      # 📱 企业微信推送
└── api_server.py         # 🌐 FastAPI 量化API服务
```

## ⚙️ 配置

编辑 `trading_config.py` 调整：
- 扫描间隔 (默认10分钟)
- 夜间休眠时段 (04:00-07:00 北京时间)
- 策略开关 (ml_momentum/mean_reversion/cross_market_alpha/aggressive)
- 风险参数 (最大回撤/日损上限/持仓上限)

## 🌐 相关链接

| 链接 | 地址 |
|------|------|
| 🌍 **Vercel 前端** | [chase-quant-web.vercel.app](https://chase-quant-web.vercel.app) |
| 🏠 Streamlit 仪表板 | http://localhost:8501 |
| 📖 量化API | http://localhost:8766 |
| 📖 API文档 | http://localhost:8766/docs |
| 🔗 Cloudflare Tunnel | `trycloudflare.com` (动态) |

### ⚡ GitHub → Vercel 自动部署

```
git push origin main → Vercel 自动构建 → 全球 CDN 生效 (~2秒)
```

- **前端**: 纯静态 SPA，`vercel.json` 配置 zero-build 部署
- **API**: Vercel Rewrites 反向代理 `/api/*` → Cloudflare Tunnel 后端
- **前后端分离**: 改 UI 不碰交易后端，策略运行不中断

---

<p align="center">
  <b>🐾 Made with ❤️ by Yina, for Chase哥</b><br>
  <i>"市场永远有机会，但风控永远是第一"</i>
</p>
