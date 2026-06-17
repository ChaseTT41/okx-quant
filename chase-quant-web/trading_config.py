"""
Chase量化策略 — 全局交易配置
==============================
支持三种模式: PAPER (模拟盘) / TESTNET (交易所测试网/Demo) / LIVE (实盘)
支持多交易所: Binance / OKX

读取 .env 环境变量，所有组件统一从此模块获取交易模式。
24 处现有 ccxt 无凭证调用完全不受影响。

使用:
    from trading_config import TradingConfig, TradingMode, Exchange, create_exchange

    config = TradingConfig.from_env()
    if config.is_live:
        exchange = create_exchange(for_trading=True)
    else:
        exchange = create_exchange(for_trading=False)  # 公开行情, 无凭证

交易所适配:
    OKX 需要 passphrase (密码短语), Binance 不需要
    OKX Demo 交易 = sandbox mode, Binance 测试网 = testnet
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

class Exchange(Enum):
    """支持的交易所"""
    BINANCE = "binance"
    OKX = "okx"

    @property
    def label(self) -> str:
        labels = {
            Exchange.BINANCE: "币安 Binance",
            Exchange.OKX: "OKX",
        }
        return labels.get(self, "❓ 未知")

    @property
    def requires_passphrase(self) -> bool:
        """是否需要 passphrase (OKX 需要)"""
        return self == Exchange.OKX

    @property
    def ccxt_id(self) -> str:
        """ccxt exchange ID"""
        return self.value

    @property
    def default_fee_rate(self) -> float:
        """默认现货 taker 费率"""
        rates = {
            Exchange.BINANCE: 0.0010,   # 0.10%
            Exchange.OKX: 0.0010,       # 0.10% taker (maker 0.08%)
        }
        return rates.get(self, 0.0010)


class TradingMode(Enum):
    """交易模式"""
    PAPER = "paper"       # 模拟盘 — 现有逻辑完全不改
    TESTNET = "testnet"   # 交易所测试网/Demo — 免费, 真实 API 交互
    LIVE = "live"         # 实盘 — 真金白银

    @property
    def is_real(self) -> bool:
        """是否涉及真实交易所交互"""
        return self in (TradingMode.TESTNET, TradingMode.LIVE)

    @property
    def label(self) -> str:
        labels = {
            TradingMode.PAPER: "📝 模拟盘",
            TradingMode.TESTNET: "🧪 测试网/Demo",
            TradingMode.LIVE: "🔴 实盘",
        }
        return labels.get(self, "❓ 未知")


@dataclass
class TradingConfig:
    """全局交易配置 — 从环境变量读取"""

    mode: TradingMode = TradingMode.PAPER
    exchange: Exchange = Exchange.OKX       # 默认交易所

    # API 凭证 (兼容 Binance / OKX)
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""                    # OKX 专用 (Binance 忽略)

    # ── 实盘风控参数（100-300 USDT 小额适配）──
    min_balance_usdt: float = 50.0          # 最低余额保护
    max_daily_trades: int = 100             # 每日最大交易次数（信号驱动，不做硬限制）
    max_drawdown_pct: float = 0.10          # 最大回撤 10% 阻止新开仓
    max_position_size_pct: float = 0.50     # 单币最大仓位 50%
    api_error_fuse_count: int = 3           # 连续 API 错误熔断次数
    api_error_fuse_minutes: int = 30        # 熔断冷却时间（分钟）
    cooldown_after_stop_minutes: int = 30   # 止损后冷静期（分钟）
    min_notional_usdt: float = 10.0         # 现货最小交易额

    # 路径
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    kill_switch_path: Path = field(default=None)

    # 代理配置 (中国大陆/泰国访问 OKX 需要)
    proxy_url: str = ""                    # e.g. http://127.0.0.1:7890
    okx_api_url: str = ""                  # OKX 备用域名 (e.g. https://www.okx.cab)

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
          EXCHANGE=binance|okx
          OKX_API_KEY=<key>         (OKX)
          OKX_SECRET_KEY=<secret>   (OKX)
          OKX_PASSPHRASE=<phrase>   (OKX 专用)
          BINANCE_API_KEY=<key>     (Binance, 兼容旧配置)
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

        # 交易所选择
        exchange_str = os.environ.get("EXCHANGE", "okx").strip().lower()
        try:
            exchange = Exchange(exchange_str)
        except ValueError:
            print(f"⚠️ 未知交易所 '{exchange_str}', 回退到 OKX")
            exchange = Exchange.OKX

        # 模式
        mode_str = os.environ.get("TRADING_MODE", "paper").strip().lower()
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            print(f"⚠️ 未知交易模式 '{mode_str}', 回退到 paper")
            mode = TradingMode.PAPER

        # 凭证 — 根据交易所读取对应环境变量
        if exchange == Exchange.OKX:
            api_key = os.environ.get("OKX_API_KEY", "").strip()
            secret_key = os.environ.get("OKX_SECRET_KEY", "").strip()
            passphrase = os.environ.get("OKX_PASSPHRASE", "").strip()
        else:
            # Binance (兼容旧字段名)
            api_key = os.environ.get("BINANCE_API_KEY", "").strip()
            secret_key = os.environ.get("BINANCE_SECRET_KEY", "").strip()
            passphrase = ""

        # 实盘参数（支持环境变量覆盖）
        min_balance = float(os.environ.get("LIVE_MIN_BALANCE_USDT", "50"))
        max_trades = int(os.environ.get("LIVE_MAX_DAILY_TRADES", "100"))
        max_dd = float(os.environ.get("LIVE_MAX_DRAWDOWN", "0.10"))

        # 代理 & 备用域名 (中国大陆/泰国访问 OKX 需要)
        proxy_url = os.environ.get("PROXY_URL", "").strip()
        okx_api_url = os.environ.get("OKX_API_URL", "").strip()

        return cls(
            mode=mode,
            exchange=exchange,
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            min_balance_usdt=min_balance,
            max_daily_trades=max_trades,
            max_drawdown_pct=max_dd,
            proxy_url=proxy_url,
            okx_api_url=okx_api_url,
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

    @property
    def has_credentials(self) -> bool:
        """API 凭证是否已配置"""
        if self.exchange == Exchange.OKX:
            return bool(self.api_key and self.secret_key and self.passphrase)
        return bool(self.api_key and self.secret_key)

    def validate(self) -> list[str]:
        """
        校验配置完整性，返回问题列表

        Returns:
            空列表 = 配置正确
        """
        issues = []

        if self.is_real_trading and not self.has_credentials:
            if self.exchange == Exchange.OKX:
                issues.append(
                    f"{self.mode.label} 模式需要 OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, "
                    "请在 .env 中配置"
                )
            else:
                issues.append(
                    f"{self.mode.label} 模式需要 BINANCE_API_KEY 和 BINANCE_SECRET_KEY, "
                    "请在 .env 中配置"
                )

        if self.exchange == Exchange.OKX and self.is_real_trading and not self.passphrase:
            issues.append("OKX 实盘需要 OKX_PASSPHRASE (创建 API Key 时设置的密码短语)")

        if self.is_live and self.min_balance_usdt < 10:
            issues.append(
                f"最低余额保护 {self.min_balance_usdt} USDT 低于最小交易额 $10"
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


def create_exchange(for_trading: bool = False, testnet: bool = False,
                   exchange_type: Exchange = None) -> "ccxt.Exchange":
    """
    创建 ccxt Exchange 实例（支持 Binance / OKX）

    Args:
        for_trading: True = 带 API 凭证（交易用）
                     False = 无凭证（公开行情，24 处现有代码保持不变）
        testnet: True = 连接交易所测试网/Demo
                 False = 连接主网
        exchange_type: Exchange.BINANCE / Exchange.OKX (None = 从环境变量读取)

    Returns:
        ccxt 交易所实例 (ccxt.binance() 或 ccxt.okx())

    使用示例:
        # 公开行情（现有代码不动）
        ex = create_exchange()

        # OKX 实盘交易
        ex = create_exchange(for_trading=True, exchange_type=Exchange.OKX)

        # OKX Demo 交易
        ex = create_exchange(for_trading=True, testnet=True, exchange_type=Exchange.OKX)

        # Binance 实盘（兼容旧代码）
        ex = create_exchange(for_trading=True, exchange_type=Exchange.BINANCE)
    """
    import ccxt

    # 确定交易所
    if exchange_type is None:
        exchange_str = os.environ.get("EXCHANGE", "okx").strip().lower()
        try:
            exchange_type = Exchange(exchange_str)
        except ValueError:
            exchange_type = Exchange.OKX

    exchange_params = {
        "enableRateLimit": True,
        "timeout": 15000,
    }

    if for_trading:
        # 从环境变量读取凭证（如果还没加载）
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            _load_dotenv(env_file)

        if exchange_type == Exchange.OKX:
            exchange_params.update({
                "apiKey": os.environ.get("OKX_API_KEY", ""),
                "secret": os.environ.get("OKX_SECRET_KEY", ""),
                "password": os.environ.get("OKX_PASSPHRASE", ""),
            })
        else:
            # Binance
            exchange_params.update({
                "apiKey": os.environ.get("BINANCE_API_KEY", ""),
                "secret": os.environ.get("BINANCE_SECRET_KEY", ""),
            })

    # 创建交易所实例
    if exchange_type == Exchange.OKX:
        exchange = ccxt.okx(exchange_params)
        # 应用代理（如果配置了）
        proxy_url = os.environ.get("PROXY_URL", "").strip()
        okx_api_url = os.environ.get("OKX_API_URL", "").strip()
        if proxy_url:
            exchange.proxies = {
                'http': proxy_url,
                'https': proxy_url,
            }
        if okx_api_url:
            # OKX 使用 {hostname} 模板, 需设置 hostname 而非替换整个 url
            from urllib.parse import urlparse
            parsed = urlparse(okx_api_url)
            exchange.hostname = parsed.netloc  # www.okx.cab
    else:
        exchange = ccxt.binance(exchange_params)

    # 测试网/Demo 模式
    if testnet:
        if exchange_type == Exchange.OKX:
            # OKX Demo 交易 — 使用 sandbox mode
            exchange.set_sandbox_mode(True)
        else:
            # Binance 测试网
            exchange.set_sandbox_mode(True)
            exchange.urls["api"] = exchange.urls.get("test", {})

    return exchange


# ═══════════════════════════════════════════
# CLI 工具
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import sys

    config = TradingConfig.from_env()
    print(f"交易所:   {config.exchange.label}")
    print(f"交易模式: {config.mode.label}")
    print(f"API 凭证: {'✅ 已配置' if config.has_credentials else '❌ 未配置'}")
    if config.exchange == Exchange.OKX:
        print(f"Passphrase: {'✅ 已配置' if config.passphrase else '❌ 未配置'}")
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
                ex = create_exchange(for_trading=True, testnet=config.is_testnet,
                                    exchange_type=config.exchange)
                balance = ex.fetch_balance()
                usdt = balance.get("USDT", {})
                free = usdt.get('free', 0) if isinstance(usdt, dict) else (usdt.free if hasattr(usdt, 'free') else 0)
                total = usdt.get('total', 0) if isinstance(usdt, dict) else (usdt.total if hasattr(usdt, 'total') else 0)
                print(f"USDT 余额: {free:.2f} (可用) / {total:.2f} (总计)")
                print("✅ API 连接成功!")
            except Exception as e:
                print(f"❌ API 连接失败: {e}")
