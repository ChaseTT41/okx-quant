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
| 自动Alpha挖掘 | ❌ | ✅ |
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

### 2. 启动量化仪表板

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
- [ ] 自动Alpha挖掘 (表达式引擎)
- [ ] 企业微信日报自动推送
- [ ] 订单执行优化 (拆单算法)

---

## 📄 License

MIT License — 欢迎 Star ⭐ & Fork

---

<p align="center">
  <b>🐾 Chase的量化策略 v2.2 — Qlib增强 + 滚动在线学习 + 资产关系图</b><br>
  <i>Built with ❤️ by Yina for Chase哥</i><br>
  <sub>用Qlib的AI大脑 + 我们的实盘肌肉 + 资产之间的关系网 = 无敌组合 🚀</sub>
</p>
