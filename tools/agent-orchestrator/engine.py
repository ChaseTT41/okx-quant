#!/usr/bin/env python3
"""
🐾 Yina Agent Orchestrator — YAML声明式多Agent编排引擎
   参考: Niuma图编排 + ceo-thread-orchestrator + Agentic Sprint

   核心能力:
   1. YAML → Workflow Script 编译
   2. DAG阶段编排 (parallel / pipeline / barrier)
   3. Checkpoint持久化 (断点续传)
   4. 模板变量解析 {{placeholder}}
   5. 自重构闭环 (agent改代码 → 自动验证 → commit)
"""

import yaml, json, os, sys, re, hashlib, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from copy import deepcopy

TZ = timezone(timedelta(hours=7))
SKILL_DIR = Path(__file__).parent
STATE_DIR = SKILL_DIR / "state"
TEMPLATE_DIR = SKILL_DIR / "templates"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════

@dataclass
class AgentDef:
    """Single agent invocation"""
    id: str
    agent_type: str = "general-purpose"
    prompt: str = ""
    schema: Optional[Dict] = None
    model: Optional[str] = None
    isolation: Optional[str] = None  # "worktree" for sandbox
    max_retries: int = 2

@dataclass
class PhaseDef:
    """One execution phase"""
    name: str
    description: str = ""
    mode: str = "parallel"  # parallel | pipeline | barrier | single
    agents: List[AgentDef] = field(default_factory=list)
    pipeline_stages: List[AgentDef] = field(default_factory=list)
    input_from: List[str] = field(default_factory=list)  # refs to previous phase outputs
    loop_until: Optional[str] = None  # JS expression for loop condition
    max_iterations: int = 10

@dataclass
class WorkflowDef:
    """Complete workflow definition"""
    name: str
    description: str = ""
    version: str = "1.0"
    variables: Dict[str, Dict] = field(default_factory=dict)
    phases: List[PhaseDef] = field(default_factory=list)
    checkpoint_enabled: bool = True
    max_runtime_hours: int = 24

# ═══════════════════════════════════════════
# YAML Parser & Validator
# ═══════════════════════════════════════════

class YAMLParser:
    """Parse and validate YAML workflow definitions"""

    REQUIRED_TOP_KEYS = ["name", "phases"]
    VALID_MODES = ["parallel", "pipeline", "barrier", "single"]
    VALID_AGENT_TYPES = [
        "general-purpose", "code-reviewer", "security-reviewer",
        "verify-agent", "architect", "Explore", "planner",
        "claude-forge:code-reviewer", "claude-forge:security-reviewer",
        "claude-forge:verify-agent", "claude-forge:architect",
        "tdd-guide", "database-reviewer", "refactor-cleaner",
    ]

    @classmethod
    def parse(cls, yaml_path: str) -> WorkflowDef:
        """Parse YAML file into WorkflowDef"""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        # Validate required keys
        for key in cls.REQUIRED_TOP_KEYS:
            if key not in raw:
                raise ValueError(f"Missing required key: '{key}'")

        # Parse variables
        variables = {}
        for var_name, var_def in (raw.get("variables") or {}).items():
            variables[var_name] = {
                "description": var_def.get("description", ""),
                "required": var_def.get("required", False),
                "default": var_def.get("default", None),
            }

        # Parse phases
        phases = []
        for i, phase_raw in enumerate(raw["phases"]):
            phase = cls._parse_phase(phase_raw, i)
            phases.append(phase)

        return WorkflowDef(
            name=raw["name"],
            description=raw.get("description", ""),
            version=raw.get("version", "1.0"),
            variables=variables,
            phases=phases,
            checkpoint_enabled=raw.get("checkpoint_enabled", True),
            max_runtime_hours=raw.get("max_runtime_hours", 24),
        )

    @classmethod
    def _parse_phase(cls, raw: dict, index: int) -> PhaseDef:
        """Parse a single phase"""
        if "name" not in raw:
            raise ValueError(f"Phase {index}: missing 'name'")

        # Determine mode + extract agents
        mode = raw.get("mode", "single")
        agents_raw = []
        pipeline_stages = []
        input_from = raw.get("input_from", [])

        if "parallel" in raw:
            mode = "parallel"
            agents_raw = raw["parallel"]
        elif "pipeline" in raw:
            mode = "pipeline"
            pipeline_raw = raw["pipeline"]
            if isinstance(pipeline_raw, dict):
                agents_raw = pipeline_raw.get("stages", [])
                input_from = pipeline_raw.get("input_from", input_from)
            else:
                agents_raw = pipeline_raw
        elif "barrier" in raw:
            mode = "barrier"
            barrier_val = raw["barrier"]
            # barrier can be a boolean flag or an agent list
            if isinstance(barrier_val, bool):
                # Flag mode: agents come from "agent" or "agents" key
                agents_raw = cls._extract_agent_list(raw)
            elif isinstance(barrier_val, list):
                agents_raw = barrier_val
            else:
                agents_raw = [barrier_val]
        elif "agent" in raw:
            mode = "single" if not raw.get("loop_until") else mode
            agents_raw = cls._extract_agent_list(raw)
        else:
            agents_raw = raw.get("agents", [])

        # If loop_until present but no explicit mode, default to single/parallel based on agent count
        if raw.get("loop_until") and mode == "single" and len(agents_raw) > 1:
            mode = "parallel"

        # Parse agent definitions
        agents = []
        for j, agent_raw in enumerate(agents_raw):
            if isinstance(agent_raw, str):
                agent = AgentDef(id=f"agent_{index}_{j}", prompt=agent_raw)
            elif isinstance(agent_raw, dict):
                agent = AgentDef(
                    id=agent_raw.get("id", f"agent_{index}_{j}"),
                    agent_type=agent_raw.get("agent_type", "general-purpose"),
                    prompt=agent_raw.get("prompt", ""),
                    schema=agent_raw.get("schema"),
                    model=agent_raw.get("model"),
                    isolation=agent_raw.get("isolation"),
                    max_retries=agent_raw.get("max_retries", 2),
                )
            else:
                raise ValueError(f"Phase {index} agent {j}: invalid type {type(agent_raw)}")

            if mode == "pipeline":
                pipeline_stages.append(agent)
            else:
                agents.append(agent)

        return PhaseDef(
            name=raw["name"],
            description=raw.get("description", ""),
            mode=mode,
            agents=agents,
            pipeline_stages=pipeline_stages,
            input_from=input_from,
            loop_until=raw.get("loop_until"),
            max_iterations=raw.get("max_iterations", 10),
        )

    @classmethod
    def _extract_agent_list(cls, raw: dict) -> list:
        """Extract agent list from various YAML shapes"""
        if "agent" in raw:
            val = raw["agent"]
            if isinstance(val, list):
                return val
            elif isinstance(val, dict):
                return [val]
            elif isinstance(val, str):
                return [{"prompt": val}]
        if "agents" in raw and isinstance(raw["agents"], list):
            return raw["agents"]
        return []

# ═══════════════════════════════════════════
# Template Variable Resolver
# ═══════════════════════════════════════════

class TemplateResolver:
    """Resolve {{variables}} in workflow definitions"""

    @classmethod
    def resolve(cls, text: str, context: Dict[str, Any]) -> str:
        """Replace {{key}} and {{key.nested}} with context values"""
        def replace_match(match):
            key_path = match.group(1).strip()
            try:
                value = cls._get_nested(context, key_path)
                if value is None:
                    return match.group(0)  # Keep placeholder if no value
                if isinstance(value, (list, dict)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value)
            except (KeyError, IndexError, TypeError):
                return match.group(0)

        return re.sub(r'\{\{(.+?)\}\}', replace_match, text)

    @classmethod
    def _get_nested(cls, data: dict, key_path: str) -> Any:
        """Get nested dict value by dot-separated path"""
        keys = key_path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                current = current[int(key)]
            else:
                return None
        return current

    @classmethod
    def validate_variables(cls, workflow: WorkflowDef, context: Dict) -> List[str]:
        """Check required variables are provided, return missing"""
        missing = []
        for var_name, var_def in workflow.variables.items():
            if var_def.get("required") and var_name not in context:
                if var_def.get("default") is None:
                    missing.append(var_name)
        return missing

# ═══════════════════════════════════════════
# Workflow Script Compiler (YAML → JS)
# ═══════════════════════════════════════════

class WorkflowCompiler:
    """Compile WorkflowDef YAML → Claude Code Workflow JS script"""

    @classmethod
    def compile(cls, workflow: WorkflowDef, context: Optional[Dict] = None) -> str:
        """Generate executable Workflow JS script"""
        context = context or {}
        lines = []

        # 1. Meta block
        lines.append("export const meta = {")
        lines.append(f"  name: '{cls._js_str(workflow.name)}',")
        lines.append(f"  description: '{cls._js_str(workflow.description)}',")
        phases_meta = [{"title": p.name, "detail": p.description or cls._phase_detail(p)} for p in workflow.phases]
        lines.append(f"  phases: {json.dumps(phases_meta)},")
        lines.append("}")
        lines.append("")

        # 2. Schema definitions (collect from all agents)
        schemas = cls._collect_schemas(workflow)
        for schema_name, schema_def in schemas.items():
            lines.append(f"const {schema_name} = {json.dumps(schema_def)};")
        if schemas:
            lines.append("")

        # 3. Phase execution
        phase_outputs = {}

        for i, phase in enumerate(workflow.phases):
            phase_var = f"p{i}_{cls._safe_var(phase.name)}"
            lines.append(f"phase('{phase.name}')")

            if phase.loop_until:
                # Loop wrapper
                lines.append(f"let {phase_var}_results = [];")
                lines.append(f"let {phase_var}_iter = 0;")
                lines.append(f"while ({phase_var}_iter < {phase.max_iterations}) {{")
                indent = "  "
            else:
                indent = ""

            # Compile phase body
            body_lines = cls._compile_phase_body(phase, phase_var, context)
            for bl in body_lines:
                lines.append(f"{indent}{bl}")

            if phase.loop_until:
                # Loop check
                lines.append(f"  {phase_var}_iter++;")
                lines.append(f"  if ({phase.loop_until}) break;")
                lines.append(f"  log(`Loop iteration ${{{phase_var}_iter}}/{phase.max_iterations}`);")
                lines.append("}")
                lines.append("")

            phase_outputs[phase.name] = phase_var
            lines.append("")

        # 4. Return synthesis
        lines.append("return {")
        for name, var in phase_outputs.items():
            lines.append(f"  '{cls._js_str(name)}': {var},")
        lines.append("  'completed_at': new Date().toISOString(),")
        lines.append("};")

        return "\n".join(lines)

    @classmethod
    def _compile_phase_body(cls, phase: PhaseDef, var_name: str, context: Dict) -> List[str]:
        """Compile one phase's body"""
        lines = []

        if phase.mode == "parallel":
            # Parallel execution
            lines.append(f"const {var_name} = await parallel([")
            for agent in phase.agents:
                prompt = TemplateResolver.resolve(agent.prompt, context)
                schema_var = cls._schema_var_name(agent)
                opts = cls._agent_opts(agent, schema_var)
                lines.append(f"  () => agent({cls._js_str(prompt)}, {opts}),")
            lines.append("]);")
            lines.append(f"log(`Phase '{phase.name}': ${{{var_name}.filter(Boolean).length}} agents completed`);")

        elif phase.mode == "pipeline":
            # Pipeline: items flow through stages
            if phase.input_from:
                # Flatten items from previous phases
                input_expr = ", ".join([f"({ref}.flat ? {ref}.flat() : [{ref}])" for ref in phase.input_from])
                lines.append(f"const pipeline_input = [{input_expr}].flat().filter(Boolean);")
                lines.append(f"log(`Pipeline input: ${{pipeline_input.length}} items`);")
            else:
                lines.append("const pipeline_input = args?.items || [];")

            lines.append(f"const {var_name} = await pipeline(")
            lines.append("  pipeline_input,")
            for stage in phase.pipeline_stages:
                prompt = TemplateResolver.resolve(stage.prompt, context)
                schema_var = cls._schema_var_name(stage)
                opts = cls._agent_opts(stage, schema_var)
                lines.append(f"  (prev, item, idx) => agent({cls._js_str(prompt)}, {opts}),")
            lines.append(");")

        elif phase.mode == "barrier":
            # Wait for ALL previous phases, then run agents
            lines.append(f"const {var_name} = await parallel([")
            for agent in phase.agents:
                prompt = TemplateResolver.resolve(agent.prompt, context)
                schema_var = cls._schema_var_name(agent)
                opts = cls._agent_opts(agent, schema_var)
                lines.append(f"  () => agent({cls._js_str(prompt)}, {opts}),")
            lines.append("]);")

        elif phase.mode == "single":
            agent = phase.agents[0] if phase.agents else AgentDef(id="default")
            prompt = TemplateResolver.resolve(agent.prompt, context)
            schema_var = cls._schema_var_name(agent)
            opts = cls._agent_opts(agent, schema_var)
            lines.append(f"const {var_name} = await agent({cls._js_str(prompt)}, {opts});")

        return lines

    @classmethod
    def _agent_opts(cls, agent: AgentDef, schema_var: str) -> str:
        """Build agent() options object"""
        parts = []
        if agent.agent_type != "general-purpose":
            parts.append(f"agentType: '{agent.agent_type}'")
        if schema_var:
            parts.append(f"schema: {schema_var}")
        if agent.isolation:
            parts.append(f"isolation: '{agent.isolation}'")
        if agent.model:
            parts.append(f"model: '{agent.model}'")
        if parts:
            return "{ " + ", ".join(parts) + " }"
        return "{}"

    @classmethod
    def _schema_var_name(cls, agent: AgentDef) -> Optional[str]:
        """Generate schema variable name"""
        if agent.schema:
            safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', agent.id)
            return f"{safe_id.upper()}_SCHEMA"
        return None

    @classmethod
    def _collect_schemas(cls, workflow: WorkflowDef) -> Dict[str, Dict]:
        """Collect all unique schemas"""
        schemas = {}
        for phase in workflow.phases:
            for agent in phase.agents + phase.pipeline_stages:
                if agent.schema:
                    var_name = cls._schema_var_name(agent)
                    schemas[var_name] = agent.schema
        return schemas

    @classmethod
    def _phase_detail(cls, phase: PhaseDef) -> str:
        """Generate phase detail string"""
        mode_emoji = {"parallel": "∥", "pipeline": "→", "barrier": "⏸", "single": "•"}
        emoji = mode_emoji.get(phase.mode, "•")
        n_agents = len(phase.agents) + len(phase.pipeline_stages)
        return f"{emoji} {phase.mode} · {n_agents} agents"

    @staticmethod
    def _js_str(s: str) -> str:
        """Escape string for JS single-quoted literal"""
        if s is None:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

    @staticmethod
    def _safe_var(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

# ═══════════════════════════════════════════
# Checkpoint Manager
# ═══════════════════════════════════════════

class CheckpointManager:
    """Persist and resume workflow state"""

    @classmethod
    def checkpoint_path(cls, workflow_name: str) -> Path:
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', workflow_name)
        return STATE_DIR / f"{safe}_checkpoint.json"

    @classmethod
    def save(cls, workflow_name: str, phase_name: str, results: Any,
             phase_index: int, total_phases: int, context: Dict = None):
        """Save checkpoint after phase completion"""
        checkpoint = {
            "workflow": workflow_name,
            "updated_at": datetime.now(TZ).isoformat(),
            "current_phase_index": phase_index,
            "total_phases": total_phases,
            "completed_phases": [],
            "phase_results": {},
            "context": context or {},
        }

        # Load existing checkpoint if any
        path = cls.checkpoint_path(workflow_name)
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
                    checkpoint["completed_phases"] = existing.get("completed_phases", [])
                    checkpoint["phase_results"] = existing.get("phase_results", {})
                    checkpoint["context"] = existing.get("context", {})
            except:
                pass

        # Update with new phase
        if phase_name not in checkpoint["completed_phases"]:
            checkpoint["completed_phases"].append(phase_name)
        checkpoint["phase_results"][phase_name] = cls._serialize_result(results)
        checkpoint["state"] = "running" if phase_index < total_phases - 1 else "completed"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        return checkpoint

    @classmethod
    def load(cls, workflow_name: str) -> Optional[Dict]:
        """Load existing checkpoint"""
        path = cls.checkpoint_path(workflow_name)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except:
                pass
        return None

    @classmethod
    def get_next_phase(cls, workflow_name: str, total_phases: int) -> int:
        """Get index of next phase to execute (0-based)"""
        checkpoint = cls.load(workflow_name)
        if checkpoint and checkpoint.get("state") == "running":
            return checkpoint.get("current_phase_index", 0) + 1
        return 0

    @classmethod
    def is_completed(cls, workflow_name: str) -> bool:
        checkpoint = cls.load(workflow_name)
        return checkpoint is not None and checkpoint.get("state") == "completed"

    @classmethod
    def _serialize_result(cls, result: Any) -> Any:
        """Make result JSON-serializable"""
        if result is None:
            return None
        if isinstance(result, (str, int, float, bool)):
            return result
        if isinstance(result, (list, tuple)):
            return [cls._serialize_result(r) for r in result]
        if isinstance(result, dict):
            return {k: cls._serialize_result(v) for k, v in result.items()}
        try:
            return str(result)
        except:
            return None

# ═══════════════════════════════════════════
# Orchestrator Runner
# ═══════════════════════════════════════════

class OrchestratorRunner:
    """Main orchestrator: parse → validate → compile → execute → checkpoint"""

    def __init__(self, yaml_path: str, context: Dict = None):
        self.yaml_path = Path(yaml_path)
        self.context = context or {}
        self.workflow = YAMLParser.parse(str(self.yaml_path))

        # Resolve context with defaults
        for var_name, var_def in self.workflow.variables.items():
            if var_name not in self.context and var_def.get("default") is not None:
                self.context[var_name] = var_def["default"]

    def validate(self) -> List[str]:
        """Validate workflow, return list of issues"""
        issues = []

        # Check required variables
        missing = TemplateResolver.validate_variables(self.workflow, self.context)
        if missing:
            issues.append(f"缺少必需变量: {', '.join(missing)}")

        # Check phases
        if not self.workflow.phases:
            issues.append("至少需要一个phase")

        for i, phase in enumerate(self.workflow.phases):
            all_agents = phase.agents + phase.pipeline_stages
            if not all_agents:
                issues.append(f"Phase '{phase.name}': 没有定义agent")

        return issues

    def compile(self) -> str:
        """Compile to Workflow JS script"""
        return WorkflowCompiler.compile(self.workflow, self.context)

    def get_resume_info(self) -> Dict:
        """Get resume information"""
        checkpoint = CheckpointManager.load(self.workflow.name)
        if not checkpoint:
            return {"can_resume": False, "message": "无断点，从头开始"}

        completed = checkpoint.get("completed_phases", [])
        total = checkpoint.get("total_phases", 0)
        return {
            "can_resume": checkpoint.get("state") == "running",
            "completed_phases": completed,
            "next_phase_index": checkpoint.get("current_phase_index", 0) + 1,
            "total_phases": total,
            "progress": f"{len(completed)}/{total}",
            "last_updated": checkpoint.get("updated_at"),
        }

    def preview(self) -> str:
        """Generate human-readable preview"""
        lines = []
        lines.append(f"🐾 {self.workflow.name} v{self.workflow.version}")
        lines.append(f"   {self.workflow.description}")
        lines.append("")

        if self.workflow.variables:
            lines.append("📋 变量:")
            for var_name, var_def in self.workflow.variables.items():
                req = "🔴必需" if var_def.get("required") else "🟢可选"
                default = f" (默认: {var_def['default']})" if var_def.get("default") is not None else ""
                lines.append(f"   {req} {var_name}: {var_def.get('description', '')}{default}")
            lines.append("")

        lines.append("📊 执行阶段:")
        for i, phase in enumerate(self.workflow.phases):
            mode_emoji = {"parallel": "∥", "pipeline": "→", "barrier": "⏸", "single": "•"}
            emoji = mode_emoji.get(phase.mode, "•")
            n_agents = len(phase.agents) + len(phase.pipeline_stages)
            loop = f" 🔄×{phase.max_iterations}" if phase.loop_until else ""
            lines.append(f"   {i+1}. {emoji} {phase.name} ({phase.mode} · {n_agents} agents{loop})")
            if phase.description:
                lines.append(f"      {phase.description}")

        lines.append("")
        lines.append(f"⚙️ 断点续传: {'✅' if self.workflow.checkpoint_enabled else '❌'}")
        lines.append(f"⏱️ 最长运行: {self.workflow.max_runtime_hours}h")

        return "\n".join(lines)

# ═══════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="🐾 Yina Agent Orchestrator — YAML多Agent编排引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览工作流
  python3 engine.py preview templates/code-review.yaml

  # 编译为Workflow脚本
  python3 engine.py compile templates/code-review.yaml -v target=src/

  # 检查断点状态
  python3 engine.py resume templates/code-review.yaml

  # 列出所有模板
  python3 engine.py list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # preview
    preview_parser = subparsers.add_parser("preview", help="预览工作流结构")
    preview_parser.add_argument("yaml_path", help="YAML工作流文件路径")
    preview_parser.add_argument("-v", "--var", nargs="*", help="变量 key=value", default=[])

    # compile
    compile_parser = subparsers.add_parser("compile", help="编译为Workflow JS脚本")
    compile_parser.add_argument("yaml_path", help="YAML工作流文件路径")
    compile_parser.add_argument("-v", "--var", nargs="*", help="变量 key=value", default=[])
    compile_parser.add_argument("-o", "--output", help="输出JS文件路径")

    # validate
    validate_parser = subparsers.add_parser("validate", help="验证工作流定义")
    validate_parser.add_argument("yaml_path", help="YAML工作流文件路径")
    validate_parser.add_argument("-v", "--var", nargs="*", help="变量 key=value", default=[])

    # resume
    resume_parser = subparsers.add_parser("resume", help="查看断点状态")
    resume_parser.add_argument("yaml_path", help="YAML工作流文件路径")

    # list
    subparsers.add_parser("list", help="列出所有可用模板")

    # checkpoint
    cp_parser = subparsers.add_parser("checkpoint", help="查看/管理断点")
    cp_parser.add_argument("action", choices=["show", "clear", "list"], help="操作")
    cp_parser.add_argument("workflow_name", nargs="?", help="工作流名称 (show/clear时需要)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        print("🐾 可用工作流模板:\n")
        for yaml_file in sorted(TEMPLATE_DIR.glob("*.yaml")):
            try:
                wf = YAMLParser.parse(str(yaml_file))
                print(f"  📄 {yaml_file.stem}")
                print(f"     {wf.name} — {wf.description[:80]}")
                print(f"     {len(wf.phases)} phases · v{wf.version}")
                print()
            except Exception as e:
                print(f"  ⚠️ {yaml_file.name}: {e}")
        return

    if args.command == "checkpoint":
        if args.action == "list":
            print("🐾 所有断点:\n")
            found = False
            for f in STATE_DIR.glob("*_checkpoint.json"):
                try:
                    with open(f) as fp:
                        cp = json.load(fp)
                    status = cp.get("state", "unknown")
                    updated = cp.get("updated_at", "")[:19]
                    progress = f"{len(cp.get('completed_phases', []))}/{cp.get('total_phases', '?')}"
                    status_emoji = {"running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
                    print(f"  {status_emoji} {cp['workflow']} — {progress} phases · {updated}")
                    found = True
                except:
                    pass
            if not found:
                print("  (无断点)")
            return

        if args.action in ("show", "clear") and args.workflow_name:
            path = CheckpointManager.checkpoint_path(args.workflow_name)
            if args.action == "show":
                if path.exists():
                    with open(path) as f:
                        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
                else:
                    print(f"❌ 未找到断点: {args.workflow_name}")
            elif args.action == "clear":
                if path.exists():
                    path.unlink()
                    print(f"✅ 已清除断点: {args.workflow_name}")
                else:
                    print(f"⚠️ 断点不存在: {args.workflow_name}")
        return

    # Parse variables
    context = {}
    var_list = getattr(args, 'var', []) or []
    for v in var_list:
        if "=" in v:
            k, v = v.split("=", 1)
            # Try to convert numbers
            try:
                if "." in v:
                    context[k] = float(v)
                else:
                    context[k] = int(v)
            except ValueError:
                context[k] = v

    # Resolve yaml_path (can be a short name for templates)
    yaml_path = args.yaml_path
    if not yaml_path.endswith(".yaml") and not yaml_path.endswith(".yml"):
        # Try as template name
        candidate = TEMPLATE_DIR / f"{yaml_path}.yaml"
        if candidate.exists():
            yaml_path = str(candidate)

    runner = OrchestratorRunner(yaml_path, context)

    if args.command == "validate":
        issues = runner.validate()
        if issues:
            print("❌ 验证失败:")
            for issue in issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print("✅ 验证通过!")

    elif args.command == "preview":
        print(runner.preview())

    elif args.command == "compile":
        if runner.validate():
            print("❌ 验证失败，请先修复:")
            for issue in runner.validate():
                print(f"   - {issue}")
            sys.exit(1)

        js_code = runner.compile()
        output_path = getattr(args, 'output', None)
        if output_path:
            with open(output_path, "w") as f:
                f.write(js_code)
            print(f"✅ 已编译到 {output_path}")
        else:
            print(js_code)

    elif args.command == "resume":
        info = runner.get_resume_info()
        print("🐾 断点续传状态:\n")
        if info["can_resume"]:
            print(f"  🔄 可续传: {info['progress']} phases已完成")
            print(f"  📍 下一阶段: phase {info['next_phase_index'] + 1}/{info['total_phases']}")
            print(f"  🕐 最后更新: {info['last_updated']}")
        else:
            print(f"  {info['message']}")

if __name__ == "__main__":
    main()
