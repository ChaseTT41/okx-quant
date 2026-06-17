# 🐾 Yina Agent Orchestrator — YAML声明式多Agent编排器

> 参考: Niuma图编排 + ceo-thread-orchestrator + Agentic Sprint
> YAML定义工作流 → 自动编译为Workflow脚本 → 执行+Checkpoint断点续传

---

## 🎯 触发词

- "编排任务" / "多Agent执行" / "启动工作流"
- "代码审查" / "全面审查" → 运行 code-review 模板
- "自修复" / "修bug自动验证" → 运行 self-heal 模板
- "多市场日报" / "并行采集" → 运行 multi-market-report 模板
- "查看编排模板" → 列出所有可用模板
- "断点续传" / "查看进度" → 检查checkpoint状态

---

## 🏗️ 架构

```
YAML模板 (声明式定义)
    │
    ▼
engine.py compile  →  Workflow JS脚本
    │
    ▼
Claude Code Workflow 工具执行
    │
    ▼
checkpoint.json (每阶段保存)
    │
    ▼
下一会话 resume → 继续执行
```

### 三模式执行

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **parallel** ∥ | 所有Agent同时运行 | 多维度审查、多市场采集 |
| **pipeline** → | 数据流经多个Agent依次处理 | 审查→验证→报告 |
| **barrier** ⏸ | 等待上一阶段全部完成再启动 | 依赖前一阶段结果的场景 |
| **single** • | 单个Agent执行 | 最终合成/推送 |

### DAG图编排（核心参考Niuma）

```
                    ┌──────────┐
                    │ 安全审查  │
                    └────┬─────┘
                         │
  ┌──────────┐     ┌────▼─────┐     ┌──────────┐
  │ 代码质量  │────▶│ 交叉验证  │────▶│ 合成报告  │
  └──────────┘     └────▲─────┘     └──────────┘
                         │
                    ┌────┴─────┐
                    │ 性能审查  │
                    └──────────┘
```

---

## 📦 内置模板

### 1. `code-review` — 多维度代码审查
```
4维度并行审查 (安全/质量/性能/架构)
    → 对抗性交叉验证 (3视角质疑)
    → 合成统一报告
```

**触发**: "全面审查 src/" / "审查这段代码"
**变量**: `target` (必需), `language`, `focus`

### 2. `multi-market-report` — 多市场日报
```
5市场并行信号采集 + AI信源
    → 交叉分析 (Top5 + 瓶颈叙事)
    → 推送企业微信
```

**触发**: "执行金融日报" / "多市场日报"
**变量**: `push_webhook`, `force_refresh`

### 3. `self-heal` — 自修复闭环
```
诊断(3维度并行) → Fix → Test → Verify (循环直到通过)
    → 修复报告 + 可选自动commit
```

**触发**: "修这个bug" / "自动修复并验证"
**变量**: `error_description` (必需), `target_files` (必需), `test_command`, `auto_commit`

---

## 🔧 使用方式

### 命令行

```bash
# 查看所有模板
python3 ~/.claude/skills/agent-orchestrator/engine.py list

# 预览工作流结构
python3 ~/.claude/skills/agent-orchestrator/engine.py preview templates/code-review.yaml

# 编译为Workflow脚本
python3 ~/.claude/skills/agent-orchestrator/engine.py compile templates/code-review.yaml \
  -v target=src/ -o /tmp/workflow.js

# 查看断点状态
python3 ~/.claude/skills/agent-orchestrator/engine.py resume code-review

# 查看/清空断点
python3 ~/.claude/skills/agent-orchestrator/engine.py checkpoint list
python3 ~/.claude/skills/agent-orchestrator/engine.py checkpoint show code-review
python3 ~/.claude/skills/agent-orchestrator/engine.py checkpoint clear code-review
```

### 在对话中直接使用

说触发词即可，例如:
- **"全面审查 chase-quant-web 的安全问题"** → 自动运行 code-review 模板
- **"执行金融日报推送到企微"** → 自动运行 multi-market-report 模板
- **"这个报错帮我修，自动测试验证"** → 自动运行 self-heal 模板

---

## 📝 YAML模板编写指南

### 最小模板

```yaml
name: my-workflow
description: "我的第一个工作流"

phases:
  - name: Research
    parallel:
      - id: search_1
        agent_type: Explore
        prompt: "搜索关于 X 的最新信息"
      - id: search_2
        agent_type: Explore
        prompt: "搜索关于 Y 的最佳实践"

  - name: Synthesize
    agent:
      prompt: "基于 Research 阶段的结果，写一份总结"
```

### 完整模板（带变量+Schema+循环）

```yaml
name: advanced-workflow
description: "带变量的高级工作流"
checkpoint_enabled: true

variables:
  topic:
    description: "要研究的话题"
    required: true

phases:
  - name: Research
    parallel:
      - id: depth_1
        agent_type: Explore
        prompt: "深度研究 {{topic}} 的技术细节"
        schema:
          type: object
          properties:
            findings: {type: array, items: {type: object, properties: {title: {type: string}, detail: {type: string}}}}

  - name: Verify
    loop_until: "verified_count >= 3"
    max_iterations: 5
    parallel:
      - agent_type: verify-agent
        prompt: "验证这个发现: {{item}}"
```

### 变量语法

- `{{variable_name}}` — 简单变量替换
- `{{phase_name.agent_id.field}}` — 引用前阶段结果
- `{{item}}` — pipeline中当前项
- `{{item.field}}` — pipeline中当前项的字段

---

## 🔄 Checkpoint断点续传机制

### 工作原理

1. 每个Phase执行完毕 → 自动保存到 `state/{workflow_name}_checkpoint.json`
2. 会话中断 → 下次自动检测checkpoint → 从断点继续
3. 全部完成 → state标记为 `completed`

### 断点文件结构

```json
{
  "workflow": "code-review",
  "state": "running",
  "completed_phases": ["Review"],
  "current_phase_index": 1,
  "total_phases": 3,
  "phase_results": {
    "Review": {
      "security": {"findings": [...]},
      "quality": {"findings": [...]}
    }
  }
}
```

### 搭配CronCreate实现长任务无人值守

```
CronCreate 每30分钟:
  → 新会话 → 读checkpoint
  → 有未完成phase → 继续执行
  → 全部完成 → 发通知 + 停止cron
```

---

## 📊 与Niuma视频的对应

| Niuma概念 | Yina编排器实现 |
|-----------|--------------|
| 图编排 (节点+边) | `parallel` / `pipeline` / `barrier` DAG |
| 编排者Agent | Workflow script 本身 |
| YAML编排模板 | `templates/*.yaml` |
| 环境隔离 | `isolation: "worktree"` |
| 14万行代码 | `loop_until` + `max_iterations` |
| 自己重构自己 | `self-heal` 模板 |

---

## ⚠️ 限制

1. **Workflow工具限制**: 最多4096 items per pipeline, 单次最长几小时
2. **长任务**: 推荐搭配 CronCreate + Checkpoint 分段执行
3. **API消耗**: 多Agent并行 = 多倍token消耗，注意控制并发数
4. **上下文**: 每个Agent独立上下文，不会自动共享（通过checkpoint传递结果）

---

_2026-06-18: v1.0 — 参考Niuma视频架构 + ceo-thread-orchestrator + Agentic Sprint 设计~ 🐾_
