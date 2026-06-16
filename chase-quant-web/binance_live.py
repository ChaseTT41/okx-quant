"""
Chase量化策略 — 币安实盘交易封装
==================================
封装 ccxt 币安现货交易，适配小额资金（100-300 USDT）。

核心保护:
  - 每笔交易前检查紧急停止开关
  - 金额低于币安最小交易额自动拒绝
  - API 错误自动重试（最多 3 次），连续失败触发熔断
  - 所有成交记录到 data/live_orders/{date}.jsonl

使用:
    from trading_config import TradingConfig, TradingMode
    from binance_live import BinanceLiveTrader

    config = TradingConfig.from_env()
    trader = BinanceLiveTrader(config)

    # 查询余额
    balances = trader.get_balance()

    # 市价买入 $50 的 BTC
    result = trader.market_buy("BTC/USDT", 50.0)

    # 市价卖出 0.001 BTC
    result = trader.market_sell("BTC/USDT", 0.001)

    # CLI: python3 binance_live.py --balance
    #      python3 binance_live.py --buy BTC/USDT --amount 50
"""
from __future__ import annotations
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
ORDERS_DIR = DATA_DIR / "live_orders"
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

# 熔断状态文件
FUSE_STATE_FILE = DATA_DIR / "live_fuse_state.json"


# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class OrderResult:
    """下单结果"""
    order_id: str
    symbol: str
    side: str                        # buy | sell
    type: str = "market"             # market | limit
    amount_usdt: float = 0.0         # 投入 USDT
    quantity: float = 0.0            # 成交数量
    price: float = 0.0              # 成交均价
    fee: float = 0.0               # 手续费
    fee_currency: str = "USDT"
    timestamp: str = ""
    status: str = "filled"
    error: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"


@dataclass
class BalanceInfo:
    """余额信息"""
    asset: str
    free: float                      # 可用
    locked: float                    # 冻结
    total: float                     # 总计

    @property
    def usdt_value(self) -> float:
        return self.total


# ═══════════════════════════════════════════
# 币安实盘交易器
# ═══════════════════════════════════════════

class LiveTradingBlockedError(Exception):
    """交易被阻止（紧急停止/熔断）"""
    pass


class LiveInsufficientFundsError(Exception):
    """资金不足"""
    pass


class LiveAPIError(Exception):
    """API 调用错误"""
    pass


class BinanceLiveTrader:
    """
    币安现货交易封装

    适配小额资金 (100-300 USDT):
      - 自动检查币安最小交易额 ($10)
      - 市价单为主（避免挂单不成交）
      - 每笔交易记录到 JSONL
    """

    # 币安现货最小名义交易额（美元）
    MIN_NOTIONAL_USD = 10.0

    # 支持的交易对
    SUPPORTED_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT",
    ]

    # bStocks 代币化股票（观察模式）
    BSTOCK_SYMBOLS = [
        "NVDAB/USDT", "TSLAB/USDT", "AAPLB/USDT", "CRCLB/USDT",
        "MUB/USDT", "SNDKB/USDT",
    ]

    def __init__(self, config=None):
        """
        Args:
            config: TradingConfig 实例（None = 从环境变量自动加载）
        """
        # 延迟导入避免循环依赖
        from trading_config import TradingConfig, create_exchange

        if config is None:
            config = TradingConfig.from_env()
        self.config = config

        # 创建 ccxt exchange（带凭证）
        self._exchange = create_exchange(
            for_trading=True,
            testnet=config.is_testnet
        )
        self._min_notional_cache: Dict[str, float] = {}

        # 错误计数器（用于熔断）
        self._consecutive_errors = 0
        self._fuse_blown_until: Optional[datetime] = None
        self._load_fuse_state()

    # ── 交易所属性 ──

    @property
    def exchange(self):
        return self._exchange

    # ── 紧急停止 ──

    def _check_kill_switch(self):
        """检查紧急停止开关，激活时抛出异常"""
        if self.config.is_kill_switch_active():
            raise LiveTradingBlockedError(
                "🚨 紧急停止开关已激活! 所有交易被阻止。\n"
                f"原因: {self._read_kill_switch_reason()}\n"
                f"解除: python3 -c \"from trading_config import TradingConfig; "
                f"TradingConfig.from_env().deactivate_kill_switch()\""
            )

    def _read_kill_switch_reason(self) -> str:
        try:
            if self.config.kill_switch_path.exists():
                data = json.loads(self.config.kill_switch_path.read_text())
                return data.get("reason", "未知")
        except Exception:
            pass
        return "未知"

    # ── API 错误熔断 ──

    def _load_fuse_state(self):
        """加载熔断状态"""
        try:
            if FUSE_STATE_FILE.exists():
                data = json.loads(FUSE_STATE_FILE.read_text())
                self._consecutive_errors = data.get("consecutive_errors", 0)
                blown = data.get("fuse_blown_until")
                if blown:
                    self._fuse_blown_until = datetime.fromisoformat(blown)
                    if self._fuse_blown_until and self._fuse_blown_until < datetime.now(timezone.utc):
                        # 冷却期已过，自动恢复
                        self._fuse_blown_until = None
                        self._consecutive_errors = 0
                        self._save_fuse_state()
        except Exception:
            self._consecutive_errors = 0
            self._fuse_blown_until = None

    def _save_fuse_state(self):
        """保存熔断状态"""
        try:
            FUSE_STATE_FILE.write_text(json.dumps({
                "consecutive_errors": self._consecutive_errors,
                "fuse_blown_until": self._fuse_blown_until.isoformat()
                    if self._fuse_blown_until else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    def _check_fuse(self):
        """检查 API 错误熔断状态"""
        if self._fuse_blown_until:
            if self._fuse_blown_until > datetime.now(timezone.utc):
                remaining = (self._fuse_blown_until - datetime.now(timezone.utc)).seconds // 60
                raise LiveTradingBlockedError(
                    f"⚡ API 错误熔断中! 剩余冷却 {remaining} 分钟。\n"
                    f"连续 {self._consecutive_errors} 次 API 错误触发熔断。"
                )
            else:
                # 冷却期已过
                self._fuse_blown_until = None
                self._consecutive_errors = 0
                self._save_fuse_state()

    def _record_api_error(self, error_msg: str):
        """记录 API 错误，可能触发熔断"""
        self._consecutive_errors += 1
        print(f"  ⚠️ API 错误 ({self._consecutive_errors}/{self.config.api_error_fuse_count}): {error_msg}")

        if self._consecutive_errors >= self.config.api_error_fuse_count:
            self._fuse_blown_until = (
                datetime.now(timezone.utc) +
                timedelta(minutes=self.config.api_error_fuse_minutes)
            )
            print(f"  🚨 连续 {self._consecutive_errors} 次错误，触发熔断! "
                  f"冷却 {self.config.api_error_fuse_minutes} 分钟。")

        self._save_fuse_state()

    def _record_api_success(self):
        """记录 API 成功，重置错误计数"""
        if self._consecutive_errors > 0:
            self._consecutive_errors = 0
            self._save_fuse_state()

    # ── 余额查询 ──

    def get_balance(self, asset: str = None) -> dict:
        """
        查询账户余额

        Args:
            asset: 指定资产（如 "USDT"），None 返回全部

        Returns:
            {"USDT": BalanceInfo, ...} 或 BalanceInfo
        """
        self._check_kill_switch()
        self._check_fuse()

        try:
            raw = self._exchange.fetch_balance()
            self._record_api_success()

            balances = {}
            for sym in (self.SUPPORTED_SYMBOLS + self.BSTOCK_SYMBOLS):
                base = sym.split("/")[0]
                if base in raw and base not in balances:
                    b = raw[base]
                    if isinstance(b, dict):
                        balances[base] = BalanceInfo(
                            asset=base,
                            free=b.get("free", 0) or 0,
                            locked=b.get("used", 0) or 0,
                            total=(b.get("free", 0) or 0) + (b.get("used", 0) or 0),
                        )

            # USDT
            usdt = raw.get("USDT", {})
            if isinstance(usdt, dict):
                balances["USDT"] = BalanceInfo(
                    asset="USDT",
                    free=usdt.get("free", 0) or 0,
                    locked=usdt.get("used", 0) or 0,
                    total=(usdt.get("free", 0) or 0) + (usdt.get("used", 0) or 0),
                )

            if asset:
                return balances.get(asset)
            return balances

        except Exception as e:
            self._record_api_error(f"余额查询失败: {e}")
            raise LiveAPIError(f"余额查询失败: {e}")

    def get_usdt_balance(self) -> float:
        """查询 USDT 可用余额"""
        bal = self.get_balance("USDT")
        if bal:
            return bal.free
        return 0.0

    # ── 交易信息 ──

    def get_min_notional(self, symbol: str) -> float:
        """
        获取币安最小名义交易额

        币安现货市场的最小交易额通常是 $10，但不同交易对可能不同。
        """
        if symbol in self._min_notional_cache:
            return self._min_notional_cache[symbol]

        try:
            markets = self._exchange.load_markets()
            market = markets.get(symbol, {})
            limits = market.get("limits", {})
            min_notional = limits.get("cost", {}).get("min", 0) or 0

            if min_notional <= 0:
                min_notional = self.MIN_NOTIONAL_USD

            self._min_notional_cache[symbol] = min_notional
            return min_notional
        except Exception:
            return self.MIN_NOTIONAL_USD

    def get_current_price(self, symbol: str) -> float:
        """获取当前市价"""
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker.get("last", 0) or 0
        except Exception as e:
            raise LiveAPIError(f"获取 {symbol} 价格失败: {e}")

    # ── 下单 ──

    def market_buy(self, symbol: str, amount_usdt: float,
                   note: str = "") -> OrderResult:
        """
        市价买入 — 用 USDT 买入指定金额的币

        Args:
            symbol: 交易对（如 "BTC/USDT"）
            amount_usdt: 买入金额（USDT）
            note: 备注（记录到订单日志）

        Returns:
            OrderResult

        Raises:
            LiveTradingBlockedError: 紧急停止或熔断
            LiveInsufficientFundsError: 资金不足
            LiveAPIError: API 调用失败
        """
        self._check_kill_switch()
        self._check_fuse()

        # 校验
        if amount_usdt < self.MIN_NOTIONAL_USD:
            raise LiveInsufficientFundsError(
                f"买入金额 ${amount_usdt:.2f} 低于币安最小交易额 ${self.MIN_NOTIONAL_USD}"
            )

        # 查询余额
        usdt_balance = self.get_usdt_balance()
        if usdt_balance < amount_usdt:
            raise LiveInsufficientFundsError(
                f"USDT 余额不足: 需要 ${amount_usdt:.2f}, 可用 ${usdt_balance:.2f}"
            )

        # 获取市价
        price = self.get_current_price(symbol)
        if price <= 0:
            raise LiveAPIError(f"{symbol} 价格异常: {price}")

        # 计算数量（留 0.1% 给滑点）
        quantity = (amount_usdt / price) * 0.999

        # 精度处理
        quantity = self._round_quantity(symbol, quantity)

        # 校验最小名义交易额
        notional = quantity * price
        min_notional = self.get_min_notional(symbol)
        if notional < min_notional:
            raise LiveInsufficientFundsError(
                f"名义交易额 ${notional:.2f} 低于 {symbol} 最小交易额 ${min_notional:.2f}"
            )

        # 执行下单
        return self._place_market_order(symbol, "buy", quantity, amount_usdt, note)

    def market_sell(self, symbol: str, quantity: float,
                    note: str = "") -> OrderResult:
        """
        市价卖出

        Args:
            symbol: 交易对
            quantity: 卖出数量（base currency）
            note: 备注

        Returns:
            OrderResult
        """
        self._check_kill_switch()
        self._check_fuse()

        # 查询持仓
        base = symbol.split("/")[0]
        bal = self.get_balance(base)
        if bal is None or bal.free < quantity:
            available = bal.free if bal else 0
            raise LiveInsufficientFundsError(
                f"{base} 余额不足: 需要 {quantity}, 可用 {available}"
            )

        # 精度处理
        quantity = self._round_quantity(symbol, quantity)

        # 校验最小名义交易额
        price = self.get_current_price(symbol)
        notional = quantity * price
        min_notional = self.get_min_notional(symbol)
        if notional < min_notional:
            raise LiveInsufficientFundsError(
                f"名义交易额 ${notional:.2f} 低于 {symbol} 最小交易额 ${min_notional:.2f}"
            )

        return self._place_market_order(symbol, "sell", quantity, 0, note)

    def _place_market_order(self, symbol: str, side: str,
                            quantity: float, amount_usdt: float,
                            note: str = "") -> OrderResult:
        """执行市价单（带重试）"""
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"  🔄 重试 {attempt}/{max_retries}...")
                    time.sleep(min(2 ** attempt, 5))  # 指数后退

                # 下单
                order = self._exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=quantity,
                )

                self._record_api_success()

                # 解析结果
                result = self._parse_order_result(order, symbol, side, amount_usdt, note)

                # 记录到日志
                self._log_order(result)
                return result

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    continue
                self._record_api_error(f"下单失败 (已重试{max_retries}次): {last_error}")

        raise LiveAPIError(f"下单失败: {last_error}")

    def _parse_order_result(self, order, symbol: str, side: str,
                            amount_usdt: float, note: str) -> OrderResult:
        """解析 ccxt 订单响应"""
        # ccxt create_market_order 返回的格式
        order_id = order.get("id", str(uuid.uuid4().hex[:12]))
        status = order.get("status", "unknown")

        # 成交均价
        avg_price = order.get("average", order.get("price", 0)) or 0
        if avg_price <= 0:
            # 从 fills 计算
            fills = order.get("fills", []) or []
            if fills:
                total_qty = sum(f.get("qty", 0) or 0 for f in fills)
                total_cost = sum(
                    (f.get("qty", 0) or 0) * (f.get("price", 0) or 0)
                    for f in fills
                )
                avg_price = total_cost / total_qty if total_qty > 0 else 0

        # 成交数量
        quantity = order.get("filled", order.get("amount", 0)) or 0

        # 手续费
        fee_info = order.get("fee", {}) or {}
        fee = fee_info.get("cost", 0) or 0
        fee_currency = fee_info.get("currency", "USDT")
        if fee <= 0 and "fees" in order and order["fees"]:
            for f in order["fees"]:
                fee += f.get("cost", 0) or 0
                if not fee_currency or fee_currency == "USDT":
                    fee_currency = f.get("currency", "USDT")

        return OrderResult(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            type="market",
            amount_usdt=amount_usdt,
            quantity=quantity,
            price=avg_price,
            fee=fee,
            fee_currency=fee_currency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
        )

    # ── 辅助 ──

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """根据交易对精度要求截断数量"""
        try:
            markets = self._exchange.load_markets()
            market = markets.get(symbol, {})
            precision = market.get("precision", {})
            amount_prec = precision.get("amount", 8) or 8

            # 计算小数位数
            if amount_prec < 1:
                decimals = len(str(amount_prec).split(".")[-1])
            elif amount_prec >= 1:
                decimals = 0
            else:
                decimals = 8  # fallback

            factor = 10 ** decimals
            return float(np.floor(quantity * factor) / factor)
        except Exception:
            return quantity

    def _log_order(self, result: OrderResult):
        """记录订单到 JSONL"""
        try:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            log_file = ORDERS_DIR / f"{date_str}.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  ⚠️ 订单日志写入失败: {e}")

    # ── 状态查询 ──

    def get_status(self) -> dict:
        """获取交易器状态摘要"""
        fuse_active = (
            self._fuse_blown_until is not None and
            self._fuse_blown_until > datetime.now(timezone.utc)
        )

        status = {
            "mode": self.config.mode.label,
            "testnet": self.config.is_testnet,
            "live": self.config.is_live,
            "kill_switch_active": self.config.is_kill_switch_active(),
            "fuse_active": fuse_active,
            "consecutive_errors": self._consecutive_errors,
            "api_ok": not fuse_active,
        }

        if fuse_active:
            remaining = (self._fuse_blown_until - datetime.now(timezone.utc)).seconds // 60
            status["fuse_remaining_minutes"] = remaining

        try:
            status["usdt_balance"] = self.get_usdt_balance()
        except Exception:
            status["usdt_balance"] = -1

        return status

    def print_status(self):
        """打印交易器状态"""
        s = self.get_status()
        print(f"交易模式: {s['mode']}")
        print(f"API 状态: {'✅ 正常' if s['api_ok'] else '🔴 熔断'}")
        print(f"紧急停止: {'🔴 已激活' if s['kill_switch_active'] else '🟢 正常'}")
        if s['fuse_active']:
            print(f"熔断剩余: {s.get('fuse_remaining_minutes', '?')} 分钟")
        if s.get('usdt_balance', -1) >= 0:
            print(f"USDT 余额: ${s['usdt_balance']:.2f}")


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="币安实盘交易器")
    parser.add_argument("--balance", action="store_true", help="查询余额")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--buy", type=str, help="买入交易对 (如 BTC/USDT)")
    parser.add_argument("--sell", type=str, help="卖出交易对 (如 BTC/USDT)")
    parser.add_argument("--amount", type=float, default=0, help="金额 (USDT) / 数量")
    parser.add_argument("--quantity", type=float, default=0, help="卖出数量")
    parser.add_argument("--kill", type=str, nargs="?", const="手动触发", help="激活紧急停止")
    parser.add_argument("--unkill", action="store_true", help="解除紧急停止")
    args = parser.parse_args()

    from trading_config import TradingConfig
    config = TradingConfig.from_env()

    # 紧急停止管理
    if args.kill:
        config.activate_kill_switch(reason=args.kill)
        print("🚨 紧急停止已激活!")
        sys.exit(0)
    if args.unkill:
        config.deactivate_kill_switch()
        print("🟢 紧急停止已解除!")
        sys.exit(0)

    trader = BinanceLiveTrader(config)

    # 余额
    if args.balance:
        print("💰 账户余额:")
        balances = trader.get_balance()
        for asset, info in sorted(balances.items()):
            if info.total > 0:
                print(f"  {asset:6s}: {info.free:>12.6f} (可用) / {info.total:>12.6f} (总计)")
        sys.exit(0)

    # 状态
    if args.status:
        trader.print_status()
        sys.exit(0)

    # 买入
    if args.buy and args.amount > 0:
        try:
            result = trader.market_buy(args.buy, args.amount)
            print(f"✅ 买入 {args.buy}:")
            print(f"  订单ID: {result.order_id}")
            print(f"  数量: {result.quantity:.6f}")
            print(f"  均价: ${result.price:,.2f}")
            print(f"  手续费: {result.fee:.4f} {result.fee_currency}")
        except Exception as e:
            print(f"❌ 买入失败: {e}")
            sys.exit(1)

    # 卖出
    elif args.sell and args.quantity > 0:
        try:
            result = trader.market_sell(args.sell, args.quantity)
            print(f"✅ 卖出 {args.sell}:")
            print(f"  订单ID: {result.order_id}")
            print(f"  数量: {result.quantity:.6f}")
            print(f"  均价: ${result.price:,.2f}")
            print(f"  手续费: {result.fee:.4f} {result.fee_currency}")
        except Exception as e:
            print(f"❌ 卖出失败: {e}")
            sys.exit(1)

    else:
        trader.print_status()
