# 多Agent编排架构 — Chase量化交易系统自主决策与错误诊断

> 基于抖音收藏「多Agent无人值守跑了4天，14万行代码，用了什么编排？」调研
> 设计日期: 2026-06-20

---

## 一、调研结论：2026多Agent编排格局

### 框架选型对比

| 框架 | 核心范式 | 控制流 | 确定性 | 持久化 | 适合交易? |
|------|---------|--------|--------|--------|----------|
| **LangGraph** | 有状态图(节点+边) | 高度确定性, 循环图 | ⭐⭐⭐⭐⭐ | SQLite/PG/Redis | ✅ 最佳 |
| CrewAI | 角色团队(crew) | 顺序/层级 | ⭐⭐⭐ | 进程内 | ⚠️ 原型 |
| MS Agent Framework | 图+对话混合 | 图+聊天 | ⭐⭐⭐⭐ | Session | ✅ Azure栈 |
| AutoGen (legacy) | 对话驱动 | 动态涌现 | ⭐⭐ | 进程内 | ❌ 维护模式 |
| OpenAI Agents SDK | 握手(handoff) | 模型驱动路由 | ⭐⭐⭐ | 进程内 | ⚠️ 需Temporal |

### 核心结论：LangGraph + Temporal = 2026生产标准

```
LangGraph (定义Agent做什么) →  Temporal (保证Agent活下来)
     ↓                               ↓
 有状态图编排                      事件历史回放
 条件路由+循环                     跨进程重启存活
 HITL断点中断                      Activity级重试
 OTel全链路追踪                    6语言SDK
```

**为什么LangGraph最适合量化交易：**
1. **确定性执行** — 每个状态转换可审计（金融合规刚需）
2. **原生HITL** — `interrupt()`在风控阈值触发时暂停，等待人类确认后恢复
3. **循环图** — 重试→检查→重试模式是图原生语义，不需要hack
4. **Python生态** — 与现有chase-quant-web技术栈完美匹配

---

## 二、Chase量化系统现状分析

### 现有架构

```
auto_trade_daemon.py (10min循环)
  │
  ├── RollingAwareAutoTrader.run()        [ML信号: V5/LGBM/Graph]
  │
  ├── run_all_strategies()                 [多策略并行]
  │     ├── ML融合动量
  │     ├── 均值回归网格
  │     ├── 跨市场Alpha套利
  │     ├── 激进交易
  │     └── 🕯️ 裸K策略 (优先)
  │
  ├── _merge_with_priority()              [裸K≥7分覆盖ML]
  │
  ├── execute_strategy_signals()           [执行: 裸K优先]
  │
  └── K线持仓管理 (结构止盈止损)
```

### 痛点识别

1. **无自我诊断** — 策略亏损时不会自动分析原因，需人工复盘
2. **无自适应** — 市场regime切换时参数不变，错过最佳窗口
3. **单线程决策** — 所有策略在同一个循环里串行/简单并行，无对抗验证
4. **错误不自治** — API超时/数据异常只打日志，不会自动降级或切换数据源
5. **风控被动** — 熔断阈值固定，不会根据市场波动率动态调整

---

## 三、多Agent自主交易架构设计

### 3.1 总架构：五层Agent协作网

```
                        ┌──────────────────────────────┐
                        │   🎯 Supervisor Agent         │
                        │   任务分发 + 冲突仲裁         │
                        │   (LangGraph Supervisor)      │
                        └──────┬───────────┬───────────┘
               ┌───────────────┤           ├───────────────┐
               ▼               ▼           ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ 🔍 Scout     │ │ 🧠 Analyst   │ │ ⚡ Executor  │ │ 🛡️ Guardian  │
    │ 数据侦察     │ │ 策略分析     │ │ 交易执行     │ │ 风控守护     │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  📊 Shared State (TypedDict + PostgreSQL Checkpointer)       │
    │  market_data | signals | positions | risk_metrics | logs     │
    └──────────────────────────────────────────────────────────────┘
```

### 3.2 各Agent职责

#### 🔍 Scout Agent — 数据侦察兵
```python
# 职责: 多源数据采集 + 质量校验 + 异常检测
class ScoutAgent:
    tools = [
        "fetch_binance_ohlcv",    # 币安K线
        "fetch_okx_ohlcv",        # OKX K线(备用)
        "fetch_fear_greed",       # 恐慌指数
        "fetch_dollar_index",     # 美元指数
        "validate_data_quality",  # 数据质量校验
        "detect_data_anomaly",    # 异常检测(价格跳空/成交量异常)
    ]
    # 输出: MarketSnapshot { prices, volumes, anomalies, quality_score }
```

#### 🧠 Analyst Agent — 策略分析脑
```python
# 职责: 多策略并行分析 + 交叉验证 + 置信度校准
class AnalystAgent:
    sub_analysts = [
        "ml_analyst",        # ML融合信号分析
        "kline_analyst",     # 裸K结构分析 (≥7分优先)
        "mean_rev_analyst",  # 均值回归分析
        "alpha_analyst",     # 跨市场Alpha分析
        "sentiment_analyst", # 情绪面分析
    ]
    # 输出: ConsensusSignal { direction, confidence, dissenting_views, risk_score }
```

#### ⚡ Executor Agent — 交易执行器
```python
# 职责: 信号执行 + 滑点控制 + 仓位管理
class ExecutorAgent:
    tools = [
        "place_order",           # 下单(限价/市价)
        "cancel_order",          # 撤单
        "adjust_position",       # 调仓
        "check_slippage",        # 滑点检测
        "manage_kline_sltp",     # 裸K结构止盈止损
    ]
    # 约束: 必须经Guardian审批后才能执行
```

#### 🛡️ Guardian Agent — 风控守护神
```python
# 职责: 实时风控 + 熔断 + 合规检查
class GuardianAgent:
    checks = [
        "max_position_check",       # 单币种仓位上限
        "total_exposure_check",     # 总敞口上限
        "daily_loss_limit",         # 日亏损熔断
        "consecutive_loss_check",   # 连续亏损熔断
        "volatility_adj_limits",    # 波动率自适应限额
        "kline_structure_guard",    # 裸K结构风控
    ]
    # 输出: Approval { approved, reason, adjusted_params }
```

#### 🎯 Supervisor Agent — 总调度
```python
# 职责: 任务分发 + 冲突仲裁 + 错误恢复
class SupervisorAgent:
    # 路由规则:
    # 1. Scout异常 → 暂停交易, 切换备用数据源
    # 2. Analyst分歧 → 降仓或等待(裸K优先规则不变)
    # 3. Guardian否决 → 记录原因, 通知Chase哥
    # 4. Executor失败 → 重试3次 → 降级为手动
```

### 3.3 LangGraph状态图设计

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from datetime import datetime

class TradingState(TypedDict):
    # 市场数据
    market_snapshot: dict
    data_quality_score: float
    
    # 信号
    raw_signals: List[dict]
    consensus_signals: List[dict]
    kline_priority_signals: List[dict]  # 裸K优先
    
    # 风控
    risk_metrics: dict
    guardian_approvals: List[dict]
    
    # 执行
    executed_orders: List[dict]
    execution_errors: List[dict]
    
    # 元数据
    cycle_start: str
    error_count: int
    degradation_level: int  # 0=正常, 1=降级, 2=只平仓, 3=熔断

# 构建状态图
def build_trading_graph():
    graph = StateGraph(TradingState)
    
    # 节点
    graph.add_node("scout", scout_node)           # 数据采集
    graph.add_node("validate", validate_node)      # 数据校验
    graph.add_node("analyze", analyze_node)        # 策略分析
    graph.add_node("deliberate", deliberate_node)  # 多分析师讨论
    graph.add_node("guardian_check", guardian_node)# 风控审批
    graph.add_node("execute", execute_node)        # 执行交易
    graph.add_node("report", report_node)          # 生成报告
    graph.add_node("error_handler", error_node)    # 错误处理
    graph.add_node("human_review", human_node)     # 人工审核
    
    # 边 — 正常流程
    graph.add_edge("scout", "validate")
    graph.add_conditional_edges("validate", quality_router, {
        "good": "analyze",
        "degraded": "error_handler",
        "bad": "human_review"
    })
    graph.add_edge("analyze", "deliberate")
    graph.add_edge("deliberate", "guardian_check")
    graph.add_conditional_edges("guardian_check", guardian_router, {
        "approved": "execute",
        "rejected": "report",
        "adjusted": "execute"  # 调整参数后执行
    })
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)
    
    # 错误恢复边
    graph.add_edge("error_handler", "scout")  # 切换数据源重试
    graph.add_edge("human_review", END)        # 等待Chase哥决策
    
    return graph.compile(checkpointer=postgres_checkpointer)
```

### 3.4 关键创新点

#### 创新1: 多分析师对抗验证 (Adversarial Deliberation)
```python
def deliberate_node(state: TradingState) -> TradingState:
    """
    5个分析师独立出信号 → 互相质疑 → 投票 → 输出共识
    裸K(≥7分)有否决权: 直接覆盖任何反对意见
    """
    signals = state["raw_signals"]
    kline_signals = [s for s in signals if s.get("kline_priority")]
    
    # 裸K优先: ≥7分的裸K信号直接通过, 不参与投票
    priority_signals = [s for s in kline_signals if s["kline_score"] >= 7]
    
    # 其余信号交叉验证
    for signal in signals:
        if signal in priority_signals:
            continue
        # 3个独立验证器分别从不同角度质疑
        votes = parallel_verify(signal, lenses=["correctness", "risk", "timing"])
        signal["confidence"] = sum(votes) / len(votes)
    
    state["consensus_signals"] = priority_signals + [
        s for s in signals if s.get("confidence", 0) >= 0.67
    ]
    return state
```

#### 创新2: 自适应风控 (Volatility-Aware Risk Limits)
```python
def guardian_node(state: TradingState) -> TradingState:
    """
    风控参数不再固定，根据实时波动率自适应调整
    """
    vol = state["market_snapshot"]["btc_volatility_24h"]
    
    # 波动率越高 → 仓位越小
    base_position_pct = 25.0  # 基础仓位
    vol_multiplier = max(0.3, 1.0 - (vol - 0.02) * 20)  # vol>2%时开始缩减
    adjusted_position_pct = base_position_pct * vol_multiplier
    
    # 波动率越高 → 止损越紧 (保护本金)
    base_sl_pct = 8.0
    adjusted_sl_pct = base_sl_pct * vol_multiplier
    
    # 连续亏损 → 自动降级
    if state["risk_metrics"]["consecutive_losses"] >= 3:
        state["degradation_level"] = 1  # 降仓50%
    if state["risk_metrics"]["daily_pnl_pct"] <= -10:
        state["degradation_level"] = 3  # 熔断, 只平仓
    
    return state
```

#### 创新3: 自动错误诊断与自愈 (Self-Healing)
```python
def error_node(state: TradingState) -> TradingState:
    """
    遇到错误不崩溃，自动诊断 → 切换备用方案 → 重试
    """
    error = state["execution_errors"][-1]
    
    # 诊断错误类型
    if "rate_limit" in str(error).lower():
        # API限流 → 切换备用交易所
        state["market_snapshot"]["primary_exchange"] = "okx"  # 从binance切okx
    elif "timeout" in str(error).lower():
        # 超时 → 降低数据量重试
        state["market_snapshot"]["reduced_candles"] = True  # 200→100根K线
    elif "insufficient_balance" in str(error).lower():
        # 余额不足 → 暂停开仓, 通知Chase哥
        state["degradation_level"] = 2
    
    state["error_count"] += 1
    return state
```

---

## 四、实施路线图

### Phase 1: 最小可行多Agent (1-2周)
- [ ] 安装LangGraph + 搭建基础StateGraph
- [ ] 实现Supervisor + Analyst + Guardian三个核心Agent
- [ ] 接入现有strategy_runner的信号输出
- [ ] 用PostgreSQL checkpointer持久化状态
- [ ] 在PAPER模式并跑验证

### Phase 2: 对抗验证 + 自适应风控 (1周)
- [ ] 实现多分析师对抗验证(deliberate_node)
- [ ] 实现波动率自适应风控参数
- [ ] 实现连续亏损自动降级
- [ ] 添加决策审计日志

### Phase 3: 自愈 + 持久化 (1周)
- [ ] 实现自动错误诊断+切换备用方案
- [ ] 接入Temporal(可选,先用LangGraph checkpointer)
- [ ] OTel全链路追踪集成
- [ ] 测试网验证48h

### Phase 4: 生产上线
- [ ] 实盘LIVE模式
- [ ] 人工审核断点(HITL)用于大额交易
- [ ] 日报自动生成多Agent决策摘要

---

## 五、与现有系统的集成点

```python
# strategy_runner.py — 最小侵入式集成
# 在run_all_strategies()末尾加一行:

from multi_agent_orchestrator import TradingOrchestrator

def run_all_strategies(market_data=None):
    # ... 现有逻辑不变 ...
    all_signals = _merge_with_priority(all_signals)
    
    # 🆕 多Agent验证 (可选, feature flag控制)
    if ENABLE_MULTI_AGENT:
        orchestrator = TradingOrchestrator(checkpointer=pg_checkpointer)
        all_signals = orchestrator.deliberate(all_signals, market_data)
    
    return all_signals
```

### 关键原则
1. **不破坏现有逻辑** — 多Agent是增强层, 不是替换层
2. **Feature Flag控制** — 可随时开关, 出问题立即回退
3. **裸K优先不变** — ≥7分裸K信号仍然直接覆盖所有ML信号
4. **渐进式采用** — 先PAPER→TESTNET→LIVE, 每阶段验证

---

## 六、参考资源

- LangGraph官方文档: https://langchain-ai.github.io/langgraph/
- Temporal + LangGraph集成: https://temporal.io/blog/langgraph-tutorial
- B站原视频: 「多Agent无人值守跑了4天，14万行代码，用了什么编排？」
- Meridian项目: AI Worker Orchestration (VibeCoding大赏)
- 2026多Agent框架对比: futureagi.com/blog/best-ai-agent-orchestration-platforms-2026/

---

*设计: Yina 🐾 | 日期: 2026-06-20 | 状态: 待Chase哥审核后进入Phase 1实施*
