# 🐾 Yina Quant — Chase哥的AI量化交易系统

> **Multi-Market + ML/DL + Auto Trade + Risk Control**  
> 加密货币 · A股 · 美股 · 港股 | Qlib深度学习增强 | Streamlit 仪表板  
> Built with ❤️ by Yina for Chase哥

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-00b159.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧬 核心亮点

| 能力 | 状态 | 说明 |
|------|:---:|------|
| **4市场覆盖** | ✅ | 加密货币 (Binance) + A股 (akshare) + 美股 (yfinance) + 港股 (3136只) |
| **500+特征工程** | ✅ | 17类别 · FDR筛选 · 西蒙斯风格 "让数据说话" |
| **7主题子信号** | ✅ | 趋势/均值回归/量价/波动/尾部风险/动量/跨市场 |
| **LightGBM** | ✅ | 7个独立主题模型 · PurgedKFold · ICIR 0.11-0.71 |
| **🧠 Qlib DL模型** | ✅ **NEW** | ALSTM + Transformer + TabNet + GATs 深度学习 |
| **模型融合 v5.0** | ✅ **NEW** | LightGBM + Qlib模型 ICIR加权融合 + 共识投票 |
| **🔄 滚动在线学习** | ✅ **NEW** | 自适应模型更新 · 特征漂移检测 · 自动重训调度 |
| **🔗 资产关系图** | ✅ **NEW** | 6维关系矩阵 · CrossAssetGAT · 图漂移检测 · 多资产联合预测 |
| **🔬 自动Alpha挖掘** | ✅ **NEW** | 表达式引擎 · 46模板 · Grid/Genetic/Random · FDR筛选 |
| **📊 订单执行优化** | ✅ **NEW** | TWAP/VWAP/Adaptive/Iceberg/Smart · Almgren-Chriss冲击模型 |
| **📱 企业微信日报推送** | ✅ **NEW** | 智能日报生成 · 算法洞察+推算逻辑 · 企微Webhook自动推送 |
| **自动交易** | ✅ | auto_trade.py · ML驱动 · 多币种扫描 |
| **五层风控** | ✅ | 事前→订单→持仓→组合→异常 · 硬止损-8% |
| **Walk-Forward** | ✅ | 滚动窗口OOS验证 · 参数稳定性评分 |
| **Web仪表板** | ✅ | Streamlit · 实时净值 · 一键交易 · 7个Tab |
| **港股量化** | ✅ | 3136只全覆盖 · 五维评分 · 动量轮动 |

---

## 🆚 vs Microsoft Qlib

> 详细对比见 [Qlib对比分析](https://github.com/ChaseTT41/yina-quant#vs-microsoft-qlib)

| 维度 | 🐾 Yina Quant | 🏢 Microsoft Qlib |
|------|:---:|:---:|
| 市场覆盖 | **4市场** | A股为主 |
| 深度学习模型 | ALSTM/Transformer/TabNet/GATs | **20+模型** |
| 自动Alpha挖掘 | ✅ **NEW! 表达式引擎** | ✅ |
| 订单执行优化 | ✅ **NEW! 拆单算法** | ❌ |
| 企业微信日报推送 | ✅ **NEW! 智能日报+企微推送** | ❌ |
| 实盘交易 | ✅ | ❌ |
| 风控体系 | **五层铁律** | 薄弱 |
| 在线学习 | ✅ **滚动在线学习** | ✅ |
| 资产关系图 | ✅ **6维关系+图漂移** | ⚠️ 研究为主 |
| **我们的策略** | **用Qlib的AI + 我们的实盘 = 🚀** | |

---

## 📁 项目结构

```
yina-app/
├── chase-quant-web/              # 🎯 核心量化仪表板
│   ├── app.py                    # Streamlit Web App (7个Tab)
│   ├── api_server.py             # 🆕 FastAPI 数据接口 (为前端仪表板提供API)
│   ├── static/                   # 🆕 前端仪表板 (HTML5 + Tailwind + TradingView图表)
│   │   └── index.html            # 模拟盘单页应用 · 深色金融UI · 响应式
│   ├── auto_trade.py             # 自动交易脚本 (Cron触发)
│   │
│   ├── feature_engine.py         # 500+特征工厂 (单值版)
│   ├── feature_ts.py             # 时序特征工厂 v4.0 (500+特征一次计算)
│   ├── feature_backtest_v4.py   # FDR校正 + PurgedKFold + ICIR筛选
│   │
│   ├── ml_signal_v4.py          # 信号引擎 v4.0 (LightGBM + 7子信号)
│   ├── ml_signal_v5.py          # 🆕 信号引擎 v5.0 (Qlib DL + LightGBM 融合)
│   ├── ml_lightgbm_trainer.py   # LightGBM训练器 + 模型注册表
│   │
│   ├── qlib_models.py           # 🆕 Qlib DL模型 (ALSTM/Transformer/TabNet/GATs)
│   ├── qlib_trainer.py          # 🆕 Qlib模型训练器 + PurgedKFold
│   ├── rolling_trainer.py       # 🆕 滚动在线学习引擎 (Phase 10)
│   ├── asset_graph.py           # 🆕 资产关系图引擎 (Phase 11)
│   ├── alpha_miner.py           # 🆕 自动Alpha挖掘引擎 (Phase 12)
│   ├── execution.py             # 🆕 订单执行优化引擎 (Phase 13)
│   ├── wechat_report.py         # 🆕 企业微信日报推送 (Phase 14)
│   │
│   ├── ml_cross_market.py       # 跨市场数据 (ETH/SPY/DXY/VIX/F&G)
│   ├── strategy_backtest.py     # 策略回测引擎
│   ├── hyperparam_optimizer.py  # Optuna + Grid Search参数优化
│   ├── walk_forward_validator.py # Walk-Forward滚动验证
│   ├── bias_correction.py       # 偏差修正 (Survival Bias)
│   │
│   ├── signals.py               # 传统信号引擎 (RSI/MACD)
│   ├── portfolio.py             # 组合管理 (3市场虚拟盘)
│   ├── risk.py                  # 五层风控控制器
│   │
│   ├── data/                    # 数据 & 模型 & 回测结果
│   │   ├── models/              # LightGBM模型 (.pkl) + Qlib模型 (.pth)
│   │   ├── backtest_results/    # 回测报告
│   │   └── optimization_results/ # 参数优化结果
│   │
│   └── requirements.txt
│
├── hk_stock_data.py             # 港股数据加载器 (3136只)
├── hk_five_dim_scorer.py        # 五维评分卡 (港股版)
├── hk_stock_screener.py         # 港股全市场批量筛选
├── hk_momentum_rotation.py     # 动量轮动策略 (backtrader)
│
├── data/hk_stocks/              # 港股数据 (145万条日线)
├── reports/                     # 每日筛选报告
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/yina-app/chase-quant-web

# 基础依赖
pip install streamlit pandas numpy plotly ccxt yfinance lightgbm scikit-learn scipy

# Qlib 深度学习增强 (Phase 9)
pip install torch torchvision torchaudio

# 参数优化
pip install optuna
```

### 2. 启动全新模拟盘仪表板 🆕 (推荐)

```bash
cd ~/yina-app/chase-quant-web
pip install fastapi uvicorn

# 启动 API Server + 前端仪表板
python3 api_server.py --port 8766
# 浏览器打开: http://localhost:8766
```

> 🎨 现代化深色金融UI · K线图 (TradingView轻量图表) · 持仓面板 · 资金曲线 · 绩效指标 · ML信号洞察 · 响应式设计

### 3. 启动 Streamlit 分析仪表板 (经典版)

```bash
cd ~/yina-app/chase-quant-web
streamlit run app.py --server.port 8501
# 浏览器打开: http://localhost:8501
```

### 3. 训练模型

```bash
# LightGBM 训练
python3 ml_lightgbm_trainer.py

# Qlib 深度学习训练 (NEW!)
python3 qlib_trainer.py --model alstm --epochs 50
python3 qlib_trainer.py --compare   # 对比所有模型 vs LightGBM

# 🔄 滚动在线学习 (NEW! Phase 10)
python3 rolling_trainer.py --init      # 首次全量训练
python3 rolling_trainer.py --update    # 增量更新 (自动检查是否需要)
python3 rolling_trainer.py --status    # 查看模型新鲜度
python3 rolling_trainer.py --backtest  # 回测滚动窗口性能

# 🔗 资产关系图 (NEW! Phase 11)
python3 asset_graph.py --build          # 构建6维资产关系图
python3 asset_graph.py --build --predict # 构建并多资产联合预测
python3 asset_graph.py --rolling        # 滚动窗口动态图
python3 asset_graph.py --train --epochs 100  # 训练 CrossAssetGAT
python3 rolling_trainer.py --check-graph  # 检查图漂移
python3 rolling_trainer.py --build-graph  # 重建资产关系图

# 🔬 自动Alpha挖掘 (NEW! Phase 12)
python3 alpha_miner.py --evaluate "ts_delta(close,5)/ts_std(close,20)"  # 评估表达式
python3 alpha_miner.py --mine --n 500        # Grid Search挖掘500个Alpha
python3 alpha_miner.py --evolve --generations 30  # 遗传进化
python3 alpha_miner.py --list               # 查看已入库Alpha
python3 alpha_miner.py --list-templates     # 查看模板库

# 📊 订单执行优化 (NEW! Phase 13)
python3 execution.py --estimate 0.5 BTC/USDT              # 预交易成本估算
python3 execution.py --simulate BTC/USDT --qty 0.1 --strategy smart  # 模拟拆单
python3 execution.py --compare                             # 策略对比
python3 execution.py --stats                               # 执行质量统计
python3 auto_trade.py --rolling --execution smart --ml-scan  # 扫描+执行配置预览

# 📱 企业微信日报推送 (NEW! Phase 14)
python3 wechat_report.py                         # 自动判断时段, 生成并推送
python3 wechat_report.py --mode morning          # 早报 (08:30)
python3 wechat_report.py --mode evening          # 晚报 (22:00)
python3 wechat_report.py --dry-run               # 预览不推送
python3 wechat_report.py --test                  # 测试Webhook连通性
python3 auto_trade.py --rolling --ml --wechat-report  # 交易后推送简报

# 策略回测
python3 strategy_backtest.py

# 参数优化
python3 hyperparam_optimizer.py

# Walk-Forward 验证
python3 walk_forward_validator.py
```

### 4. 部署自动交易

```bash
# 添加到 crontab (每30分钟)
*/30 * * * * cd ~/yina-app/chase-quant-web && python3 auto_trade.py --ml

# Phase 10: 滚动训练感知模式 (推荐!)
*/30 * * * * cd ~/yina-app/chase-quant-web && python3 auto_trade.py --rolling
```

---

## 🧠 Qlib 深度学习模型详解 (Phase 9)

### 模型架构

| 模型 | 类型 | 擅长 | 论文 |
|------|------|------|------|
| **ALSTM** | Attention LSTM | 时序模式识别 + 注意力加权 | Feng et al., 2019 |
| **Transformer** | Self-Attention | 长程依赖 + 全局时序建模 | Vaswani et al., 2017 |
| **TabNet** | Attentive Tabular | 稀疏特征选择 + 可解释性 | Arik & Pfister, 2019 |
| **GATs** | Graph Attention | 资产关系图 + 邻居聚合 | Veličković et al., 2018 |

### 融合策略

```
286个特征 (FeatureFactoryV4)
    │
    ├──→ LightGBM × 7 主题  →  7个预测
    ├──→ ALSTM    × 7 主题  →  7个预测
    ├──→ Transformer × 7 主题 → 7个预测
    └──→ TabNet   × 7 主题  →  7个预测
                                    │
                             ICIR 加权融合
                            + 共识投票机制
                            + 分歧惩罚
                                    │
                              最终信号
                        BUY (>0.5) / HOLD / SELL (<-0.5)
```

### 滚动在线学习 (Phase 10) 🆕

```
┌─────────────────────────────────────────────────────────┐
│                   Rolling Trainer                        │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Window    │ →  │   Retrain    │ →  │   Model    │ │
│  │   Manager   │    │   Scheduler  │    │  Registry  │ │
│  │             │    │              │    │            │ │
│  │ expanding/  │    │ daily/weekly/│    │ versioned  │ │
│  │ sliding/    │    │ monthly      │    │ .pth +     │ │
│  │ hybrid      │    │              │    │ metadata   │ │
│  └─────────────┘    └──────────────┘    └────────────┘ │
│                                                         │
│  触发条件:                                               │
│  🔴 模型过期 (>21天)  →  自动重训                          │
│  🟡 ICIR退化 (>15%)   →  自动重训                          │
│  🟠 特征漂移 (>0.3)   →  自动重训                          │
│  🟢 模型新鲜 (<7天)   →  跳过                              │
└─────────────────────────────────────────────────────────┘
```

### 资产关系图 (Phase 11) 🆕

```
┌──────────────────────────────────────────────────────────────┐
│                    Asset Graph Engine                         │
│                                                              │
│  6维关系矩阵                  多资产GAT预测                    │
│  ┌────────────────┐         ┌──────────────────────┐         │
│  │ 1. Pearson 相关│         │                      │         │
│  │ 2. Spearman 秩 │  融合   │  N个资产 → N个节点     │         │
│  │ 3. 距离相关dCor│ ────→  │  N×N 邻接矩阵        │         │
│  │ 4. 协整关系    │         │  GAT 聚合邻居信息     │         │
│  │ 5. Granger因果 │         │  → N个联合预测       │         │
│  │ 6. 波动率相关  │         │                      │         │
│  └────────────────┘         └──────────────────────┘         │
│                                                              │
│  图漂移监控:                                                  │
│  🔴 漂移 > 0.3  →  自动重建图 →  GATs 重新训练                │
│  🟢 图结构稳定   →  继续使用现有图                              │
└──────────────────────────────────────────────────────────────┘
```

### 自动Alpha挖掘 (Phase 12) 🆕

```
┌──────────────────────────────────────────────────────────────┐
│                   Alpha Mining Engine                         │
│                                                              │
│  表达式语言                    3大挖掘策略                     │
│  ┌────────────────┐         ┌──────────────────────┐         │
│  │ 变量: OHLCV    │         │ 1. Grid Search       │         │
│  │ 函数: 20+ TS   │  驱动   │    46模板 × 参数网格  │         │
│  │ 运算符: +-*/^  │ ────→  │ 2. Genetic Evolve    │         │
│  │ 单目: abs/log.. │        │    锦标赛+交叉+变异   │         │
│  │ 截面: cs_rank  │         │ 3. Random Explore    │         │
│  └────────────────┘         │    语法随机生成       │         │
│                              └──────────────────────┘         │
│                                         │                    │
│                              批量IC评估 + FDR校正              │
│                              Rank IC / ICIR / Decay          │
│                              Long-Short Sharpe / Turnover     │
│                                         │                    │
│                              Top-N入库 (JSON)                  │
│                              → ML信号引擎特征增强              │
│                              → 纯Alpha信号 (无ML)             │
└──────────────────────────────────────────────────────────────┘
```

### 订单执行优化 (Phase 13) 🆕

```
┌──────────────────────────────────────────────────────────────┐
│                   Execution Engine                             │
│                                                              │
│  5大拆单策略                   Market Impact Model            │
│  ┌────────────────┐         ┌──────────────────────┐         │
│  │ TWAP 时间加权   │         │ Almgren-Chriss 改编  │         │
│  │ VWAP 成交量加权 │  驱动   │  暂时冲击 η·σ·X^β   │         │
│  │ Adaptive 自适应 │ ────→  │  永久冲击 γ·σ·X^α   │         │
│  │ Iceberg 冰山    │         │  最优切片数 闭式解   │         │
│  │ Smart 智能路由  │         │  点差/延迟成本分解   │         │
│  └────────────────┘         └──────────────────────┘         │
│                                         │                    │
│                              纸交易模拟 + 仿真滑点              │
│                              Implementation Shortfall        │
│                              VWAP Slippage / Fill Rate        │
│                                         │                    │
│                              执行质量入库 (JSON)                │
│                              → auto_trade 拆单自动交易          │
│                              → 策略对比分析仪表板               │
└──────────────────────────────────────────────────────────────┘
```

### 企业微信日报推送 (Phase 14) 🆕

```
┌──────────────────────────────────────────────────────────────┐
│                   WeChat Report Engine                        │
│                                                              │
│  数据采集层                    日报生成层                      │
│  ┌────────────────┐         ┌──────────────────────┐         │
│  │ Binance行情     │         │ 📊 市场概览           │         │
│  │ F&G恐慌指数     │         │   价格/涨跌/成交量    │         │
│  │ ML信号引擎 v5   │  融合   │ 🤖 ML信号洞察         │         │
│  │ 虚拟盘持仓      │ ────→  │   融合方向+推算逻辑    │         │
│  │ 五层风控状态    │         │ 📈 模拟盘持仓+盈亏    │         │
│  │ 执行质量统计    │         │ 🛡️ 风控状态           │         │
│  │ 模型新鲜度      │         │ 💡 Yina综合研判       │         │
│  │ Alpha挖掘库     │         │   (多因子推理)        │         │
│  └────────────────┘         └──────────────────────┘         │
│                                         │                    │
│                              企业微信 Markdown 格式化           │
│                              超长消息自动分段                   │
│                              Webhook 推送 + 降级方案            │
│                                         │                    │
│                              定时调度 (Cron)                    │
│                              早报08:30 / 午报14:00 / 晚报22:00  │
│                              → 企业微信群「金融监控」            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 风控铁律 (5层防御)

| 层级 | 规则 | 触发条件 |
|------|------|---------|
| 🔴 硬止损 | 无条件-8%止损 | 单笔亏损≥8% |
| 💰 单笔上限 | ≤ 总资金 2% | 潜在亏损超限 |
| 📦 持仓上限 | ≤ 5 只 | 同时持仓数 |
| 📊 仓位上限 | ≤ 40% 单仓位 | 单仓位集中 |
| 🛑 日熔断 | 日亏损 > 5% 停交易 | 日内回撤 |

---

## 📊 港股量化模块

```bash
# 单只评分
python3 hk_five_dim_scorer.py 00700

# 全市场扫描 (3136只)
python3 hk_stock_screener.py --full --top 50

# 动量轮动回测
python3 hk_momentum_rotation.py
```

### 五维评分卡

| 维度 | 权重 | 指标 |
|------|:---:|------|
| 📈 趋势强度 | 25% | MACD + 均线排列 + ADX |
| 🔄 超买超卖 | 15% | RSI + 布林带 |
| 🏗️ 支撑阻力 | 20% | 关键价位 + 距离 |
| 💎 基本面 | 25% | 量价 + 动量 + 波动率 |
| ⚡ 风险度 | 15% | 历史波动率 + VaR |

---

## 🔮 路线图

- [x] 港股数据接入 + 五维评分 (3136只)
- [x] Streamlit Web仪表板 (3市场虚拟盘)
- [x] ML特征工厂 v4.0 (500+特征)
- [x] LightGBM 7主题模型训练
- [x] Walk-Forward优化 + 参数验证
- [x] 跨市场数据 (ETH/SPY/DXY/VIX/F&G)
- [x] **🧠 Qlib深度学习模型集成 (ALSTM/Transformer/TabNet/GATs)**
- [x] **模型融合 v5.0 (LightGBM + Qlib DL)**
- [x] 模型版本管理 (自动老化监控)
- [x] **🔄 滚动在线学习 (Rolling Training)** — 自适应模型更新
- [x] **🔗 GATs 真实资产关系图 + 多资产联合预测** — 6维关系矩阵 + CrossAssetGAT
- [x] **🔬 自动Alpha挖掘 (表达式引擎)** — Grid/Genetic/Random + 46模板 + FDR筛选
- [x] **📊 订单执行优化 (拆单算法)** — TWAP/VWAP/Adaptive/Iceberg/Smart + Almgren-Chriss
- [x] **📱 企业微信日报自动推送** — 智能日报 + 算法洞察 + 推算逻辑 + 企微推送

---

## 📄 License

MIT License — 欢迎 Star ⭐ & Fork

---

<p align="center">
  <b>🐾 Chase的量化策略 v2.5 — Qlib增强 + 在线学习 + 资产关系图 + Alpha挖掘 + 订单执行优化 + 企微日报推送</b><br>
  <i>Built with ❤️ by Yina for Chase哥</i><br>
  <sub>用Qlib的AI大脑 + 我们的实盘肌肉 + 资产关系网 + 自动Alpha挖掘 + 智能拆单 + 企微日报 = 🚀</sub>
</p>
