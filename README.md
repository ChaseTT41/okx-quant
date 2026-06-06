# Yina App — Chase哥的量化交易系统 🐾

> 港股 + 加密货币 双轨量化分析平台
> 蒸馏自 QuantDinger 全体系, 适配 Claude Code 交互

---

## 📁 项目结构

```
yina-app/
├── data/
│   └── hk_stocks/                  # 港股数据 (40MB → 75MB 解压)
│       ├── stock_list.parquet       # 3136只港股 (代码+名称)
│       ├── all_daily.parquet        # 145万条合并日线 (2007-2026)
│       └── daily/                   # 每只股票单独日线 (3136个文件)
│
├── hk_stock_data.py                # 港股数据加载器
├── hk_five_dim_scorer.py           # 五维评分卡 (港股版)
├── hk_stock_screener.py            # 全市场批量筛选器
│
├── reports/                        # 输出报告
│   └── hk_screener_YYYYMMDD.md     # 每日筛选报告
│
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

## 🚀 快速开始

```bash
cd ~/yina-app

# 1. 安装依赖
pip3 install pandas numpy pyarrow

# 2. 单只股票评分
python3 hk_five_dim_scorer.py

# 3. 批量筛选 (自选股)
python3 hk_stock_screener.py

# 4. 全市场扫描 (耗时较长)
python3 hk_stock_screener.py --full
```

## 🎯 模块说明

### hk_stock_data.py — 数据加载器
- `HKStockData.search("腾讯")` → 按名称/代码搜索
- `HKStockData.get_daily("00700")` → 单只股票完整日线
- `HKStockData.get_multi_daily([...])` → 批量获取
- `get_data()` → 全局单例

### hk_five_dim_scorer.py — 五维评分
- `FiveDimScorer().score("00700")` → 单只股票完整评分
- `FiveDimScorer().score_batch([...])` → 批量评分
- 输出: [买/卖/观] + ⭐(1-5) + 置信度% + 五维明细 + 止损/目标

### hk_stock_screener.py — 批量筛选
- `HKScreener().run_quick(codes)` → 快速评分指定列表
- `HKScreener().run(top_n=50)` → 全市场扫描
- `HKScreener().report(df)` → 格式化报告

## 🤖 Claude Code 集成

在 Claude Code 中直接说以下关键词触发:

| 命令 | 效果 |
|------|------|
| "港股评分 00700" | 腾讯五维评分 + 操作建议 |
| "港股搜索 腾讯" | 搜索港股代码 |
| "港股扫描" | 自选股批量评分 Top 榜 |
| "港股日报" | 今日港股市场日报 |
| "港股对比 00700 09988" | 腾讯 vs 阿里 多空对比 |

## 📊 五维评分体系

| 维度 | 权重 | 关键指标 |
|------|:---:|---------|
| 📈 趋势强度 | 25% | MACD + 均线排列 + ADX + 金叉 |
| 🔄 超买超卖 | 15% | RSI(14) + 布林带位置 |
| 🏗️ 支撑阻力 | 20% | 关键价位 + 距支撑/阻力距离 |
| 💎 基本面 | 25% | 量价关系 + 动量 + 波动率 + 涨跌比 |
| ⚡ 风险度 | 15% | 历史波动率 + 最大回撤 + 流动性 + VaR |

## 🔮 路线图

- [x] 港股数据接入
- [x] 五维评分卡 (港股版)
- [x] 批量筛选器
- [ ] 动量轮动回测 (backtrader)
- [ ] Multi-LLM Ensemble 集成
- [ ] 企业微信日报推送
- [ ] 纸交易订单模拟
