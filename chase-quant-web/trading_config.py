"""
Chase量化策略 — 全局交易配置
==============================
支持三种模式: PAPER (模拟盘) / TESTNET (币安测试网) / LIVE (币安实盘)

读取 .env 环境变量，所有组件统一从此模块获取交易模式。
24 处现有 ccxt.binance() 无凭证调用完全不受影响。

使用:
    from trading_config import TradingConfig, TradingMode, create_exchange

    config = TradingConfig.from_env()
    if config.is_live:
        exchange = create_exchange(for_trading=True)
    else:
        exchange = create_exchange(for_trading=False)  # 公开行情, 无凭证
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional


# ═══════════════════════════════════════════
# 枚举 & 数据类
# ═══════════════════════════════════════════

class TradingMode(Enum):
    """交易模式"""
    PAPER = "paper"       # 模拟盘 — 现有逻辑完全不改
    TESTNET = "testnet"   # 币安测试网 — 免费, 真实 API 交互
    LIVE = "live"         # 币安实盘 — 真金白银

    @property
    def is_real(self) -> bool:
        """是否涉及真实资金"""
        return self in (TradingMode.TESTNET, TradingMode.LIVE)

    @property
    def label(self) -> str:
        labels = {
            TradingMode.PAPER: "📝 模拟盘",
            TradingMode.TESTNET: "🧪 测试网",
            TradingMode.LIVE: "🔴 实盘",
        }
        return labels.get(self, "❓ 未知")


@dataclass
class TradingConfig:
    """全局交易配置 — 从环境变量读取"""

    mode: TradingMode = TradingMode.PAPER

    # 币安 API 凭证
    api_key: str = ""
    secret_key: str = ""

    # ── 实盘风控参数（100-300 USDT 小额适配）──
    min_balance_usdt: float = 50.0          # 最低余额保护
    max_daily_trades: int = 5               # 每日最大交易次数（保守）
    max_drawdown_pct: float = 0.10          # 最大回撤 10% 阻止新开仓
    max_position_size_pct: float = 0.50     # 单币最大仓位 50%
    api_error_fuse_count: int = 3           # 连续 API 错误熔断次数
    api_error_fuse_minutes: int = 30        # 熔断冷却时间（分钟）
    cooldown_after_stop_minutes: int = 30   # 止损后冷静期（分钟）
    min_notional_usdt: float = 10.0         # 币安现货最小交易额

    # 路径
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    kill_switch_path: Path = field(default=None)

    def __post_init__(self):
        if self.kill_switch_path is None:
            self.kill_switch_path = self.data_dir / "kill_switch.json"

    @classmethod
    def from_env(cls, env_file: Path = None) -> "TradingConfig":
        """
        从 .env 文件 + 环境变量构造配置

        优先级: 环境变量 > .env 文件 > 默认值

        环境变量:
          TRADING_MODE=paper|testnet|live
          BINANCE_API_KEY=<key>
          BINANCE_SECRET_KEY=<secret>
          LIVE_MIN_BALANCE_USDT=50
          LIVE_MAX_DAILY_TRADES=5
          LIVE_MAX_DRAWDOWN=0.10
        """
        # 加载 .env 文件（如果存在）
        if env_file is None:
            env_file = Path(__file__).parent / ".env"

        if env_file.exists():
            _load_dotenv(env_file)

        # 模式
        mode_str = os.environ.get("TRADING_MODE", "paper").strip().lower()
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            print(f"⚠️ 未知交易模式 '{mode_str}', 回退到 paper")
            mode = TradingMode.PAPER

        # 凭证
        api_key = os.environ.get("BINANCE_API_KEY", "").strip()
        secret_key = os.environ.get("BINANCE_SECRET_KEY", "").strip()

        # 实盘参数（支持环境变量覆盖）
        min_balance = float(os.environ.get("LIVE_MIN_BALANCE_USDT", "50"))
        max_trades = int(os.environ.get("LIVE_MAX_DAILY_TRADES", "5"))
        max_dd = float(os.environ.get("LIVE_MAX_DRAWDOWN", "0.10"))

        return cls(
            mode=mode,
            api_key=api_key,
            secret_key=secret_key,
            min_balance_usdt=min_balance,
            max_daily_trades=max_trades,
            max_drawdown_pct=max_dd,
        )

    # ── 便捷属性 ──

    @property
    def is_paper(self) -> bool:
        return self.mode == TradingMode.PAPER

    @property
    def is_testnet(self) -> bool:
        return self.mode == TradingMode.TESTNET

    @property
    def is_live(self) -> bool:
        return self.mode == TradingMode.LIVE

    @property
    def is_real_trading(self) -> bool:
        """是否涉及真实交易所交互（testnet 或 live）"""
        return self.mode.is_real

    @property
    def has_credentials(self) -> bool:
        """API 凭证是否已配置"""
        return bool(self.api_key and self.secret_key)

    def validate(self) -> list[str]:
        """
        校验配置完整性，返回问题列表

        Returns:
            空列表 = 配置正确
        """
        issues = []

        if self.is_real_trading and not self.has_credentials:
            issues.append(
                f"{self.mode.label} 模式需要 BINANCE_API_KEY 和 BINANCE_SECRET_KEY, "
                "请在 .env 中配置"
            )

        if self.is_live and self.min_balance_usdt < 10:
            issues.append(
                f"最低余额保护 {self.min_balance_usdt} USDT 低于币安最小交易额 $10"
            )

        if self.is_live:
            # 检查是不是测试网密钥误用
            if self.api_key.startswith("test") or "testnet" in self.api_key.lower():
                issues.append(
                    "⚠️ API Key 看起来像测试网密钥, 但 TRADING_MODE=live"
                )

        return issues

    # ── 紧急停止 ──

    def is_kill_switch_active(self) -> bool:
        """检查紧急停止开关"""
        if not self.kill_switch_path or not self.kill_switch_path.exists():
            return False
        try:
            data = json.loads(self.kill_switch_path.read_text())
            if not data.get("active"):
                return False
            # 检查是否过期（超过设定时间自动恢复）
            activated_at = data.get("activated_at", "")
            if activated_at:
                then = datetime.fromisoformat(activated_at)
                timeout_hours = data.get("timeout_hours", 24)
                if datetime.now(timezone.utc) - then > timedelta(hours=timeout_hours):
                    return False
            return True
        except Exception:
            return False

    def activate_kill_switch(self, reason: str = "", timeout_hours: int = 24):
        """激活紧急停止"""
        self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        self.kill_switch_path.write_text(json.dumps({
            "active": True,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "timeout_hours": timeout_hours,
        }, ensure_ascii=False, indent=2))

    def deactivate_kill_switch(self):
        """解除紧急停止"""
        if self.kill_switch_path.exists():
            self.kill_switch_path.write_text(json.dumps({
                "active": False,
                "deactivated_at": datetime.now(timezone.utc).isoformat(),
            }))


# ═══════════════════════════════════════════
# ccxt Exchange 工厂
# ═══════════════════════════════════════════

_LOADED_DOTENV = False


def _load_dotenv(env_file: Path):
    """简易 dotenv 加载器（不依赖 python-dotenv）"""
    global _LOADED_DOTENV
    if _LOADED_DOTENV:
        return
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        _LOADED_DOTENV = True
    except Exception:
        pass


def create_exchange(for_trading: bool = False, testnet: bool = False) -> "ccxt.binance":
    """
    创建 ccxt Binance Exchange 实例

    Args:
        for_trading: True = 带 API 凭证（交易用）
                     False = 无凭证（公开行情，24 处现有代码保持不变）
        testnet: True = 连接币安测试网
                 False = 连接币安主网

    Returns:
        ccxt.binance() 实例

    使用示例:
        # 公开行情（现有代码不动）
        ex = create_exchange()

        # 实盘交易
        config = TradingConfig.from_env()
        ex = create_exchange(for_trading=True)

        # 测试网交易
        ex = create_exchange(for_trading=True, testnet=True)
    """
    import ccxt

    exchange_params = {
        "enableRateLimit": True,
        "timeout": 15000,
    }

    if for_trading:
        # 从环境变量读取凭证（如果还没加载）
        if not os.environ.get("BINANCE_API_KEY"):
            env_file = Path(__file__).parent / ".env"
            if env_file.exists():
                _load_dotenv(env_file)

        exchange_params.update({
            "apiKey": os.environ.get("BINANCE_API_KEY", ""),
            "secret": os.environ.get("BINANCE_SECRET_KEY", ""),
        })

    exchange = ccxt.binance(exchange_params)

    if testnet:
        exchange.set_sandbox_mode(True)
        exchange.urls["api"] = exchange.urls.get("test", ccxt.binance().urls.get("test", {}))

    return exchange


# ═══════════════════════════════════════════
# CLI 工具
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import sys

    config = TradingConfig.from_env()
    print(f"交易模式: {config.mode.label}")
    print(f"API 凭证: {'✅ 已配置' if config.has_credentials else '❌ 未配置'}")
    print(f"紧急停止: {'🔴 已激活' if config.is_kill_switch_active() else '🟢 正常'}")

    issues = config.validate()
    if issues:
        print("\n⚠️ 配置问题:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("✅ 配置无问题")

    if config.has_credentials:
        print(f"\nAPI Key 前缀: {config.api_key[:8]}... ({len(config.api_key)} 字符)")
        if config.is_real_trading:
            try:
                ex = create_exchange(for_trading=True, testnet=config.is_testnet)
                balance = ex.fetch_balance()
                usdt = balance.get("USDT", {})
                print(f"USDT 余额: {usdt.get('free', 0):.2f} (可用) / {usdt.get('total', 0):.2f} (总计)")
                print("✅ API 连接成功!")
            except Exception as e:
                print(f"❌ API 连接失败: {e}")
