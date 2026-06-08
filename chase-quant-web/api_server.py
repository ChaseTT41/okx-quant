"""
Chase量化策略 🐾 — 模拟盘 API Server
FastAPI 后端，为前端仪表板提供数据接口
启动: python3 api_server.py --port 8765
"""
from __future__ import annotations
import sys
import os
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("❌ FastAPI 未安装，请运行: pip install fastapi uvicorn")

from portfolio import PortfolioManager, ALLOCATION, INITIAL_CAPITAL, Position, Trade
from risk import RiskController

# ── 数据源 ──
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# ── 信号引擎 ──
from signals import SignalEngine, CryptoSignals

try:
    from ml_signal_v5 import MLSignalEngineV5, FusionSignal
    ML_V5_AVAILABLE = True
except ImportError:
    ML_V5_AVAILABLE = False

try:
    from alpha_miner import AlphaStore
    ALPHA_AVAILABLE = True
except ImportError:
    ALPHA_AVAILABLE = False

try:
    from execution import ExecutionStore
    EXECUTION_AVAILABLE = True
except ImportError:
    EXECUTION_AVAILABLE = False

try:
    from sentiment_analyzer import MarketSentimentAnalyzer, SentimentReport
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# ── 安全中间件 ──
try:
    from security import install_security_middleware, verify_admin_token, is_admin_enabled, ADMIN_TOKEN as _ADMIN_TOKEN
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# ── 配置 ──
DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 生产模式检测
PRODUCTION_MODE = os.environ.get("PRODUCTION", "false").lower() in ("1", "true", "yes")

CRYPTO_WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
]

# ═══════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════
app = FastAPI(
    title="Chase量化策略 API",
    description="🐾 自主量化交易模拟盘数据接口",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 压缩 — 大幅减少传输体积（147KB → ~35KB）
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── 安装安全中间件 ──
if SECURITY_AVAILABLE:
    install_security_middleware(app, production=PRODUCTION_MODE)
elif PRODUCTION_MODE:
    # 即使 security.py 不可用, 生产模式仍要禁用 API 文档
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None

# ── 全局实例（懒加载）──
_pf: Optional[PortfolioManager] = None
_risk: Optional[RiskController] = None
_exchange: Optional[object] = None
_signal_engine: Optional[SignalEngine] = None
_ml_engine: Optional[object] = None


def get_pf() -> PortfolioManager:
    global _pf
    if _pf is None:
        _pf = PortfolioManager()
    return _pf


def get_risk() -> RiskController:
    global _risk
    if _risk is None:
        _risk = RiskController(get_pf())
    return _risk


def get_exchange():
    global _exchange
    if _exchange is None and CCXT_AVAILABLE:
        try:
            _exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
        except Exception:
            pass
    return _exchange


def get_signal_engine() -> SignalEngine:
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = SignalEngine()
    return _signal_engine


def get_ml_engine():
    global _ml_engine
    if _ml_engine is None and ML_V5_AVAILABLE:
        try:
            _ml_engine = MLSignalEngineV5()
        except Exception:
            pass
    return _ml_engine


# ═══════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.5.0",
        "modules": {
            "ccxt": CCXT_AVAILABLE,
            "ml_v5": ML_V5_AVAILABLE,
            "alpha": ALPHA_AVAILABLE,
            "execution": EXECUTION_AVAILABLE,
        },
    }


# ── 管理员鉴权 ──

@app.get("/api/auth/status")
async def auth_status():
    """查询管理员鉴权是否已启用"""
    return {
        "admin_enabled": is_admin_enabled(),
        "admin_configured": bool(_ADMIN_TOKEN),
    }


@app.post("/api/auth/verify")
async def verify_auth(body: dict):
    """
    验证管理员令牌

    Body: { "token": "xxx" }
    Returns: { "valid": true/false }
    """
    token = str(body.get("token", "")).strip()
    if not token:
        return {"valid": False, "detail": "令牌不能为空"}
    valid = verify_admin_token(token)
    if valid:
        return {"valid": True, "detail": "验证成功"}
    else:
        return {"valid": False, "detail": "令牌无效"}


# ── 组合总览 ──

@app.get("/api/portfolio")
async def get_portfolio():
    """获取组合概览"""
    pf = get_pf()
    pf.take_snapshot()  # 每次请求记录快照

    total_val = pf.total_value
    total_pnl = pf.total_pnl
    total_pnl_pct = pf.total_pnl_pct

    # 各市场分配
    alloc = pf.get_allocation_summary()
    markets = []
    for mkt, info in alloc.items():
        markets.append({
            "market": mkt,
            "label": info["label"],
            "color": info["color"],
            "cash": info["cash"],
            "positions_value": info["positions"],
            "total": info["total"],
            "pnl": info["pnl"],
            "pnl_pct": info["pnl_pct"],
        })

    return {
        "initial_capital": INITIAL_CAPITAL,
        "total_value": round(total_val, 2),
        "total_cash": round(pf.total_cash, 2),
        "positions_value": round(pf.positions_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions_count": len(pf.open_positions),
        "monthly_target_pct": 30.0,
        "monthly_progress_pct": round(total_pnl_pct / 30.0 * 100, 1),
        "markets": markets,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/positions")
async def get_positions():
    """获取持仓列表"""
    pf = get_pf()
    positions = []
    for pos in pf.open_positions:
        positions.append({
            "id": pos.id,
            "market": pos.market,
            "symbol": pos.symbol,
            "name": pos.name,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "avg_entry_price": pos.avg_entry_price if pos.avg_entry_price > 0 else pos.entry_price,
            "quantity": pos.quantity,
            "cost": round(pos.cost, 2),
            "value": round(pos.value, 2),
            "pnl": round(pos.pnl, 2),
            "pnl_pct": round(pos.pnl_pct, 2),
            "entry_time": pos.entry_time,
            "entry_reason": pos.entry_reason,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "status": pos.status,
            "execution_strategy": getattr(pos, "execution_strategy", ""),
        })
    return {
        "positions": positions,
        "count": len(positions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/trades")
async def get_trades(limit: int = Query(50, ge=1, le=200)):
    """获取交易历史"""
    pf = get_pf()
    trades = sorted(pf.trades, key=lambda t: t.time, reverse=True)[:limit]
    return {
        "trades": [
            {
                "id": t.id,
                "market": t.market,
                "symbol": t.symbol,
                "name": t.name,
                "side": t.side,
                "price": t.price,
                "quantity": t.quantity,
                "amount": round(t.amount, 2),
                "fee": round(t.fee, 4),
                "time": t.time,
                "reason": t.reason,
                "pnl": round(t.pnl, 2) if t.side == "sell" else 0,
            }
            for t in trades
        ],
        "count": len(trades),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/equity-curve")
async def get_equity_curve():
    """获取净值曲线数据"""
    pf = get_pf()
    snapshots = pf.snapshots

    data = []
    # 添加初始点
    data.append({
        "time": pf.snapshots[0].time if snapshots else datetime.now(timezone.utc).isoformat(),
        "value": INITIAL_CAPITAL,
        "pnl": 0.0,
        "pnl_pct": 0.0,
    })

    for s in snapshots:
        data.append({
            "time": s.time,
            "value": round(s.total_value, 2),
            "pnl": round(s.pnl_total, 2),
            "pnl_pct": round(s.pnl_pct, 2),
        })

    return {
        "equity_curve": data,
        "initial_capital": INITIAL_CAPITAL,
        "count": len(data),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/performance")
async def get_performance():
    """计算绩效指标"""
    pf = get_pf()
    snaps = pf.snapshots

    if len(snaps) < 2:
        return {
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "win_rate_pct": 0,
            "profit_factor": 0,
            "total_trades": len(pf.trades),
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "best_trade_pct": 0,
            "worst_trade_pct": 0,
            "annualized_return_pct": 0,
            "calmar_ratio": 0,
            "note": "数据不足，需要至少2个净值快照",
        }

    # 净值序列
    values = np.array([s.total_value for s in snaps])
    initial_val = INITIAL_CAPITAL

    # 日收益率（如果有多个快照）
    returns = np.diff(values) / values[:-1]

    # 最大回撤
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak * 100
    max_dd = abs(float(np.min(drawdowns)))

    # Sharpe (假设无风险利率 2%)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float((np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)))
    else:
        sharpe = 0.0

    # 年化收益
    total_return = (values[-1] / initial_val - 1) * 100
    if len(snaps) >= 2:
        days = (pd.to_datetime(snaps[-1].time) - pd.to_datetime(snaps[0].time)).total_seconds() / 86400
        days = max(days, 1)
        annualized = float(((1 + total_return / 100) ** (365 / days) - 1) * 100)
    else:
        annualized = 0.0

    # Calmar
    calmar = annualized / max_dd if max_dd > 0 else 0.0

    # 交易统计
    sell_trades = [t for t in pf.trades if t.side == "sell"]
    if sell_trades:
        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl <= 0]
        win_rate = len(wins) / len(sell_trades) * 100
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0
        best = max(t.pnl_pct for t in sell_trades)
        worst = min(t.pnl_pct for t in sell_trades)
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = best = worst = 0
        profit_factor = 0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999,
        "total_trades": len(pf.trades),
        "closed_trades": len(sell_trades),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "annualized_return_pct": round(annualized, 2),
        "calmar_ratio": round(calmar, 2),
        "total_return_pct": round(total_return, 2),
    }


# ── 仪表盘聚合数据 ──

@app.get("/api/dashboard")
async def get_dashboard():
    """获取首页仪表盘聚合数据 — 一次请求返回所有仪表盘需要的数据"""
    pf = get_pf()
    risk_ctrl = get_risk()
    pf.take_snapshot()

    # ━━ 组合快照 ━━
    total_val = pf.total_value
    total_pnl = pf.total_pnl
    total_pnl_pct = pf.total_pnl_pct

    # ━━ 绩效指标 ━━
    snaps = pf.snapshots
    values = np.array([s.total_value for s in snaps]) if snaps else np.array([INITIAL_CAPITAL])
    initial_val = INITIAL_CAPITAL

    # 日内收益率 (相比初始值)
    if len(values) >= 2:
        returns = np.diff(values) / values[:-1]
    else:
        returns = np.array([])

    # 最大回撤
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak * 100
    max_dd = abs(float(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float((np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)))
    else:
        sharpe = 0.0

    # 年化收益
    total_return = (values[-1] / initial_val - 1) * 100
    if len(snaps) >= 2:
        days = (pd.to_datetime(snaps[-1].time) - pd.to_datetime(snaps[0].time)).total_seconds() / 86400
        days = max(days, 1)
        if days >= 7:  # Only annualize with >= 1 week of data
            annualized = float(((1 + total_return / 100) ** (365 / days) - 1) * 100)
        else:
            annualized = float(total_return * (365 / max(days, 1)))  # Simple linear extrapolation
        # Cap extreme values (fresh data with very short history)
        annualized = max(-99.9, min(annualized, 999.0))
    else:
        annualized = 0.0

    # Calmar
    calmar = annualized / max_dd if max_dd > 0 else 0.0

    # 交易统计
    sell_trades = [t for t in pf.trades if t.side == "sell"]
    if sell_trades:
        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl <= 0]
        win_rate = len(wins) / len(sell_trades) * 100
        avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0
        avg_loss = float(np.mean([abs(t.pnl_pct) for t in losses])) if losses else 0
        best = max(t.pnl_pct for t in sell_trades)
        worst = min(t.pnl_pct for t in sell_trades)
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = best = worst = 0
        profit_factor = 0

    # ━━ 风控 ━━
    try:
        risk_report = risk_ctrl.daily_risk_report()
        survival_score = round(risk_report.get("survival_score", 100), 1)
        risk_alerts = risk_report.get("alerts", [])
        risk_warnings = risk_report.get("warnings", [])
    except Exception:
        survival_score = 100
        risk_alerts = []
        risk_warnings = []

    # ━━ 策略状态 ━━
    strategies = []
    if STRATEGIES_AVAILABLE:
        configs = get_all_strategy_configs()
        for cfg in configs:
            strategies.append({
                "id": cfg["id"],
                "name": cfg["name"],
                "status": cfg["status"],
                "version": cfg["version"],
                "symbols": cfg["symbols"],
                "sharpe": cfg.get("sharpe", 0),
                "win_rate": cfg.get("win_rate", 0),
                "max_drawdown": cfg.get("max_drawdown", 0),
                "annual_return": cfg.get("annual_return", 0),
                "signals_today": cfg.get("signals_today", 0),
                "last_signal_at": cfg.get("last_signal_at"),
                "last_signal": cfg.get("last_signal"),
                "logic_explanation": (cfg.get("logic_explanation", "") or "")[:120],
            })

    # ━━ 持仓 ━━
    positions = []
    for pos in pf.open_positions:
        positions.append({
            "id": pos.id,
            "symbol": pos.symbol,
            "name": pos.name,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "quantity": pos.quantity,
            "value": round(pos.value, 2),
            "pnl": round(pos.pnl, 2),
            "pnl_pct": round(pos.pnl_pct, 2),
            "entry_time": pos.entry_time,
            "entry_reason": pos.entry_reason,
            "status": pos.status,
        })

    # ━━ 最近交易 ━━
    recent_trades = sorted(pf.trades, key=lambda t: t.time, reverse=True)[:20]
    trades_list = []
    for t in recent_trades:
        trades_list.append({
            "id": t.id,
            "symbol": t.symbol,
            "name": t.name,
            "side": t.side,
            "price": t.price,
            "quantity": t.quantity,
            "amount": round(t.amount, 2),
            "time": t.time,
            "reason": t.reason,
            "pnl": round(t.pnl, 2) if t.side == "sell" else 0,
        })

    # ━━ 净值曲线 ━━
    equity_data = [{
        "time": snaps[0].time if snaps else datetime.now(timezone.utc).isoformat(),
        "value": INITIAL_CAPITAL,
        "pnl": 0.0,
        "pnl_pct": 0.0,
    }]
    for s in snaps:
        equity_data.append({
            "time": s.time,
            "value": round(s.total_value, 2),
            "pnl": round(s.pnl_total, 2),
            "pnl_pct": round(s.pnl_pct, 2),
        })

    # ━━ 资产分配 ━━
    alloc = pf.get_allocation_summary()
    allocation = []
    for mkt, info in alloc.items():
        allocation.append({
            "market": mkt,
            "label": info["label"],
            "color": info["color"],
            "total": info["total"],
            "positions": info["positions"],
            "cash": info["cash"],
            "pnl": info["pnl"],
            "pnl_pct": info["pnl_pct"],
        })

    return {
        # 组合
        "portfolio": {
            "initial_capital": INITIAL_CAPITAL,
            "total_value": round(total_val, 2),
            "total_cash": round(pf.total_cash, 2),
            "positions_value": round(pf.positions_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions_count": len(pf.open_positions),
            "monthly_target_pct": 30.0,
            "daily_pnl": round(total_pnl, 2),
            "daily_pnl_pct": round(total_pnl_pct, 2),
        },
        # 绩效
        "performance": {
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999,
            "total_trades": len(pf.trades),
            "closed_trades": len(sell_trades),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "best_trade_pct": round(best, 2),
            "worst_trade_pct": round(worst, 2),
            "annualized_return_pct": round(annualized, 2),
            "calmar_ratio": round(calmar, 2),
            "total_return_pct": round(total_return, 2),
        },
        # 风控
        "risk": {
            "survival_score": survival_score,
            "alerts": risk_alerts,
            "warnings": risk_warnings,
        },
        # 策略
        "strategies": strategies,
        "strategies_running": sum(1 for s in strategies if s["status"] == "running"),
        "strategies_total": len(strategies),
        # 持仓
        "positions": positions,
        # 交易
        "recent_trades": trades_list,
        "trades_count": len(pf.trades),
        # 净值
        "equity_curve": equity_data,
        # 分配
        "allocation": allocation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── 市场数据 ──

@app.get("/api/market-snapshots")
async def get_market_snapshots():
    """获取加密货币实时报价"""
    exchange = get_exchange()
    snapshots = []

    if exchange is None:
        # 返回模拟数据
        mock_prices = {
            "BTC/USDT": 87200, "ETH/USDT": 3210, "BNB/USDT": 710,
            "SOL/USDT": 185, "XRP/USDT": 2.45, "DOGE/USDT": 0.32,
            "ADA/USDT": 0.85, "AVAX/USDT": 38.5, "DOT/USDT": 8.2, "LINK/USDT": 22.5,
        }
        for sym in CRYPTO_WATCHLIST:
            price = mock_prices.get(sym, 100)
            snapshots.append({
                "symbol": sym,
                "name": sym.replace("/USDT", ""),
                "price": price,
                "change_24h_pct": round(np.random.uniform(-5, 5), 2),
                "volume_24h": round(np.random.uniform(1e8, 1e10), 0),
                "source": "mock",
            })
        return {"snapshots": snapshots, "count": len(snapshots), "source": "mock"}

    try:
        tickers = exchange.fetch_tickers(CRYPTO_WATCHLIST)
        for sym in CRYPTO_WATCHLIST:
            t = tickers.get(sym, {})
            snapshots.append({
                "symbol": sym,
                "name": sym.replace("/USDT", ""),
                "price": t.get("last", 0),
                "change_24h_pct": round(t.get("percentage", 0) or 0, 2),
                "volume_24h": t.get("quoteVolume", 0) or 0,
                "high_24h": t.get("high", 0),
                "low_24h": t.get("low", 0),
                "bid": t.get("bid", 0),
                "ask": t.get("ask", 0),
                "source": "binance",
            })
        return {"snapshots": snapshots, "count": len(snapshots), "source": "binance"}
    except Exception as e:
        return {"snapshots": [], "count": 0, "error": str(e)}


@app.get("/api/market-data")
async def get_market_data(
    symbol: str = Query(..., description="交易对，如 BTC/USDT"),
    timeframe: str = Query("1h", regex="^(1m|5m|15m|30m|1h|4h|1d|1w)$"),
    limit: int = Query(200, ge=1, le=500),
):
    """获取K线数据 — symbol 通过 query 参数传递（避免 / 路径问题）"""
    sym = symbol.replace("-", "/")
    exchange = get_exchange()

    if exchange is None:
        # 生成模拟K线
        return _mock_klines(sym, timeframe, limit)

    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
        candles = []
        for row in ohlcv:
            candles.append({
                "time": int(row[0] / 1000),  # 转为秒级Unix时间戳
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return {
            "symbol": sym,
            "timeframe": timeframe,
            "candles": candles,
            "count": len(candles),
            "source": "binance",
        }
    except Exception as e:
        return _mock_klines(sym, timeframe, limit)


def _mock_klines(symbol: str, timeframe: str, limit: int) -> dict:
    """生成模拟K线数据"""
    base_prices = {
        "BTC/USDT": 87200, "ETH/USDT": 3210, "BNB/USDT": 710,
        "SOL/USDT": 185, "XRP/USDT": 2.45, "DOGE/USDT": 0.32,
        "ADA/USDT": 0.85, "AVAX/USDT": 38.5, "DOT/USDT": 8.2, "LINK/USDT": 22.5,
    }
    base = base_prices.get(symbol, 100)
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
    interval = tf_seconds.get(timeframe, 3600)

    np.random.seed(hash(symbol) % 2**31)
    returns = np.random.randn(limit) * 0.015
    prices = base * np.exp(np.cumsum(returns) * 0.3)

    now = int(datetime.now().timestamp())
    candles = []
    for i in range(limit):
        t = now - (limit - i) * interval
        p = float(prices[i])
        o = p * (1 + np.random.uniform(-0.005, 0.005))
        h = max(o, p) * (1 + abs(np.random.uniform(0, 0.01)))
        l = min(o, p) * (1 - abs(np.random.uniform(0, 0.01)))
        c = p
        v = abs(np.random.uniform(100, 10000))
        candles.append({
            "time": t,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": round(v, 2),
        })

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "count": len(candles),
        "source": "mock",
    }


# ── 信号 ──

@app.get("/api/signals")
async def get_signals(market: str = Query("crypto", regex="^(crypto|a_stock|us_stock|all)$")):
    """获取交易信号"""
    try:
        engine = get_signal_engine()
        all_signals = engine.scan_all()

        result = {}
        if market == "all":
            target = all_signals
        else:
            target = {market: all_signals.get(market, [])}

        for mkt, signals in target.items():
            result[mkt] = []
            for s in signals[:10]:
                result[mkt].append({
                    "symbol": s.symbol,
                    "name": s.name,
                    "market": s.market,
                    "price": round(s.price, 4),
                    "action": s.action,
                    "score": round(s.score, 1),
                    "confidence": round(s.confidence, 2),
                    "reasons": s.reasons[:5],
                    "risk_level": s.risk_level,
                    "suggested_size": round(s.suggested_size, 0),
                    "stop_loss": round(s.stop_loss, 4),
                    "take_profit": round(s.take_profit, 4),
                    "adjusted_score": round(s.adjusted_score, 1),
                })

        return {
            "signals": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"signals": {}, "error": str(e)}


@app.get("/api/ml-insights")
async def get_ml_insights():
    """获取ML信号洞察"""
    if not ML_V5_AVAILABLE:
        return {"insights": [], "error": "ML Signal Engine V5 不可用"}

    try:
        engine = get_ml_engine()
        if engine is None:
            return {"insights": [], "error": "ML引擎初始化失败"}

        insights = []
        for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]:
            try:
                # 获取OHLCV数据
                exchange = get_exchange()
                if exchange:
                    ohlcv = exchange.fetch_ohlcv(sym, "1d", limit=365)
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    # 模拟数据
                    dates = pd.date_range(end=datetime.now(), periods=365, freq="D")
                    np.random.seed(hash(sym) % 2**31)
                    base = {"BTC/USDT": 87200, "ETH/USDT": 3210, "SOL/USDT": 185, "BNB/USDT": 710, "XRP/USDT": 2.45, "DOGE/USDT": 0.32}.get(sym, 100)
                    prices = base * (1 + np.cumsum(np.random.randn(365) * 0.02))
                    df = pd.DataFrame({
                        "timestamp": dates,
                        "open": prices * (1 + np.random.uniform(-0.01, 0.01, 365)),
                        "high": prices * (1 + abs(np.random.uniform(0, 0.02, 365))),
                        "low": prices * (1 - abs(np.random.uniform(0, 0.02, 365))),
                        "close": prices,
                        "volume": abs(np.random.uniform(1000, 100000, 365)),
                    })

                signal = engine.generate_signal(df, sym)
                insights.append({
                    "symbol": sym,
                    "name": sym.replace("/USDT", ""),
                    "direction": signal.direction if hasattr(signal, "direction") else "HOLD",
                    "confidence": round(signal.confidence * 100, 1) if hasattr(signal, "confidence") else 50,
                    "strength": round(signal.strength * 100, 1) if hasattr(signal, "strength") else 50,
                    "consensus": round(signal.consensus * 100, 1) if hasattr(signal, "consensus") else 50,
                    "divergence": round(signal.divergence * 100, 1) if hasattr(signal, "divergence") else 50,
                    "model_count": signal.model_count if hasattr(signal, "model_count") else 0,
                    "reasoning": signal.reasoning if hasattr(signal, "reasoning") else "ML信号分析中...",
                })
            except Exception:
                insights.append({
                    "symbol": sym,
                    "name": sym.replace("/USDT", ""),
                    "direction": "HOLD",
                    "confidence": 50,
                    "strength": 50,
                    "consensus": 50,
                    "divergence": 50,
                    "model_count": 0,
                    "reasoning": "数据获取失败",
                })

        return {
            "insights": insights,
            "count": len(insights),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"insights": [], "error": str(e)}


# ── 风控 ──

@app.get("/api/risk")
async def get_risk_status():
    """获取风控状态"""
    try:
        risk = get_risk()
        report = risk.daily_risk_report()
        return {
            "survival_score": round(report.get("survival_score", 100), 1),
            "alerts": report.get("alerts", []),
            "warnings": report.get("warnings", []),
            "market_risk": report.get("market_risk", "unknown"),
            "position_risk": report.get("position_risk", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"survival_score": 100, "alerts": [], "error": str(e)}


# ── Alpha 挖掘 ──

@app.get("/api/alpha-discoveries")
async def get_alpha_discoveries(limit: int = Query(10, ge=1, le=50)):
    """获取Alpha因子发现"""
    if not ALPHA_AVAILABLE:
        return {"discoveries": [], "error": "Alpha Store 不可用"}

    try:
        store = AlphaStore()
        discoveries = store.load(limit)
        return {
            "discoveries": [
                {
                    "id": d.get("id", ""),
                    "expression": d.get("expression", ""),
                    "ic": round(d.get("ic", 0), 4),
                    "icir": round(d.get("icir", 0), 4),
                    "sharpe": round(d.get("sharpe", 0), 2),
                    "description": d.get("description", ""),
                    "discovered_at": d.get("discovered_at", ""),
                }
                for d in discoveries
            ],
            "count": len(discoveries),
        }
    except Exception as e:
        return {"discoveries": [], "error": str(e)}


# ── 平仓 + 交易执行 ──

@app.post("/api/positions/close")
async def close_position(body: dict):
    """平仓指定持仓 — 在 body 中传入 position_id 和 price"""
    pf = get_pf()
    position_id = str(body.get("position_id", "")).strip()

    if not position_id or len(position_id) > 128:
        raise HTTPException(status_code=400, detail="缺少或无效的 position_id 参数")

    # 输入清理: 防止路径遍历和特殊字符注入
    if ".." in position_id or "\x00" in position_id:
        raise HTTPException(status_code=400, detail="position_id 包含非法字符")

    # Find position
    pos = next((p for p in pf.open_positions if p.id == position_id), None)
    if not pos:
        raise HTTPException(status_code=404, detail=f"持仓 {position_id} 不存在或已平仓")

    price = body.get("price", pos.current_price)

    try:
        result = pf.sell(
            position_id=position_id,
            price=price,
            reason="手动平仓",
        )
        if result is None:
            raise HTTPException(status_code=400, detail="平仓失败：资金不足或持仓异常")

        return {
            "success": True,
            "position_id": position_id,
            "symbol": pos.symbol,
            "close_price": price,
            "quantity": pos.quantity,
            "pnl": round(result.pnl, 2),
            "pnl_pct": round(result.pnl_pct, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade")
async def execute_trade(body: dict):
    """手动执行交易（虚拟盘）— 接受 JSON body"""
    pf = get_pf()

    market = str(body.get("market", "crypto")).strip()
    symbol = str(body.get("symbol", "")).strip()
    side = str(body.get("side", "buy")).strip().lower()
    reason = str(body.get("reason", "手动交易")).strip()[:256]

    # 输入验证 + 安全清理
    if not symbol or ".." in symbol or "\x00" in symbol or len(symbol) > 32:
        raise HTTPException(status_code=400, detail="无效的交易对符号")
    if side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="side 必须是 buy 或 sell")
    if market not in ("crypto", "a_stock", "us_stock"):
        raise HTTPException(status_code=400, detail="无效的市场类型")

    try:
        quantity = float(body.get("quantity", 0))
        price = float(body.get("price", 0))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="quantity 和 price 必须是有效数字")

    if quantity <= 0 or price <= 0:
        raise HTTPException(status_code=400, detail="quantity 和 price 必须大于 0")
    if quantity > 1_000_000 or price > 1_000_000_000:
        raise HTTPException(status_code=400, detail="交易参数超出合理范围")

    name = symbol.replace("/USDT", "")

    try:
        if side == "buy":
            result = pf.buy(market, symbol, name, price, quantity, reason)
        else:
            result = pf.sell(market, symbol, name, price, quantity, reason)

        if result is None:
            raise HTTPException(status_code=400, detail="交易失败：资金不足或持仓不存在")

        return {
            "success": True,
            "trade": {
                "id": result.id,
                "side": side,
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
                "amount": round(price * quantity, 2),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 策略管理 (真实策略引擎) ──

try:
    from strategies import (
        init_strategies, get_all_strategy_configs,
        get_strategy, get_all_strategies,
        update_strategy_param, STRATEGY_REGISTRY,
        generate_all_signals,
    )
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    print("⚠️ strategies.py 未找到，策略引擎不可用")


if STRATEGIES_AVAILABLE:
    # 启动时初始化策略引擎
    init_strategies()
    print(f"🧠 策略引擎已初始化: {len(STRATEGY_REGISTRY)} 个策略已加载")
    for sid, s in STRATEGY_REGISTRY.items():
        status_icon = "🟢" if s.status == "running" else "⏸"
        print(f"   {status_icon} {s.config.name} ({sid})")


@app.get("/api/strategies")
async def get_strategies():
    """获取所有策略列表 — 来自真实策略引擎"""
    if not STRATEGIES_AVAILABLE:
        return {"strategies": [], "count": 0, "running_count": 0, "timestamp": datetime.now(timezone.utc).isoformat()}
    configs = get_all_strategy_configs()
    return {
        "strategies": configs,
        "count": len(configs),
        "running_count": sum(1 for s in configs if s["status"] == "running"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/strategies/{strategy_id}")
async def get_strategy_detail(strategy_id: str):
    """获取单个策略的完整详情 — 包含自然语言逻辑说明和可编辑参数"""
    if not STRATEGIES_AVAILABLE:
        raise HTTPException(status_code=503, detail="策略引擎未就绪")
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    return {
        "strategy": strategy.config.to_dict(),
        "description_text": strategy.describe(),  # 自然语言完整描述
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.put("/api/strategies/{strategy_id}")
async def update_strategy(strategy_id: str, body: Dict):
    """更新策略参数 — 支持批量修改"""
    if not STRATEGIES_AVAILABLE:
        raise HTTPException(status_code=503, detail="策略引擎未就绪")
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

    updates = body.get("updates", {})  # {param_key: new_value}
    updated = []
    failed = []

    for key, value in updates.items():
        if update_strategy_param(strategy_id, key, value):
            updated.append(key)
        else:
            failed.append(key)

    return {
        "success": len(failed) == 0,
        "message": f"参数已更新: {len(updated)} 项" + (f"，{len(failed)} 项失败" if failed else ""),
        "strategy_id": strategy_id,
        "updated": updated,
        "failed": failed,
        "strategy": strategy.config.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """删除指定策略 — 从注册表中移除"""
    if not STRATEGIES_AVAILABLE:
        raise HTTPException(status_code=503, detail="策略引擎未就绪")
    global STRATEGY_REGISTRY
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

    name = strategy.config.name
    del STRATEGY_REGISTRY[strategy_id]
    return {
        "success": True,
        "message": f"策略「{name}」已删除",
        "strategy_id": strategy_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/strategies/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str):
    """启动/停止指定策略"""
    if not STRATEGIES_AVAILABLE:
        raise HTTPException(status_code=503, detail="策略引擎未就绪")
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

    old_status = strategy.status
    new_status = "stopped" if old_status == "running" else "running"
    strategy.status = new_status

    return {
        "success": True,
        "message": f"策略「{strategy.config.name}」已{'停止' if old_status == 'running' else '启动'}",
        "strategy_id": strategy_id,
        "old_status": old_status,
        "new_status": new_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



def _fetch_crypto_ohlcv(symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """获取加密货币OHLCV数据为DataFrame (供策略引擎使用)"""
    try:
        exchange = get_exchange()
        if exchange is None:
            return _mock_ohlcv_df(symbol, limit)
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=limit)
        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        return df
    except Exception:
        return _mock_ohlcv_df(symbol, limit)


def _mock_ohlcv_df(symbol: str, limit: int = 200) -> pd.DataFrame:
    """生成模拟OHLCV DataFrame"""
    base_prices = {
        "BTC/USDT": 87200, "ETH/USDT": 3210, "BNB/USDT": 710,
        "SOL/USDT": 185, "XRP/USDT": 2.45, "DOGE/USDT": 0.32,
        "ADA/USDT": 0.85, "AVAX/USDT": 38.5, "DOT/USDT": 8.2, "LINK/USDT": 22.5,
    }
    base = base_prices.get(symbol, 100)
    np.random.seed(hash(symbol) % 2**31)
    returns = np.random.randn(limit) * 0.025
    prices = base * np.cumprod(1 + returns)

    dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, limit)),
        "high": prices * (1 + np.abs(np.random.uniform(0.01, 0.02, limit))),
        "low": prices * (1 - np.abs(np.random.uniform(0.01, 0.02, limit))),
        "close": prices,
        "volume": np.random.lognormal(10, 1.5, limit),
    })
    # 确保 OHLC 一致性
    for i in range(limit):
        df.loc[i, "high"] = max(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"])
        df.loc[i, "low"] = min(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "low"])
    return df


@app.get("/api/strategies/{strategy_id}/signals")
async def get_strategy_signals(strategy_id: str, symbol: str = Query(None)):
    """对策略实时生成交易信号 — 用实际市场数据运行策略逻辑，返回AI决策信号"""
    if not STRATEGIES_AVAILABLE:
        raise HTTPException(status_code=503, detail="策略引擎未就绪")
    strategy = get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

    # 获取市场数据
    symbols = [symbol] if symbol else strategy.config.symbols[:4]
    market_data = {}
    for sym in symbols:
        try:
            df = _fetch_crypto_ohlcv(sym)
            if df is not None and not df.empty:
                market_data[sym] = df
        except Exception as e:
            pass  # 该标的不可用时跳过

    if not market_data:
        raise HTTPException(status_code=503, detail="无法获取市场数据，请稍后重试")

    # 用真实策略逻辑生成信号
    try:
        signals = strategy.generate_signals(market_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号生成失败: {str(e)}")

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy.config.name,
        "signals": signals,
        "signal_count": len(signals),
        "data_symbols": list(market_data.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════
# 市场情绪分析 (Gemini AI 驱动)
# ═══════════════════════════════════════════

if SENTIMENT_AVAILABLE:
    _sentiment_analyzer = MarketSentimentAnalyzer()
else:
    _sentiment_analyzer = None


@app.get("/api/sentiment/overview")
async def get_sentiment_overview():
    """获取整体市场情绪概览"""
    if not SENTIMENT_AVAILABLE or _sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="市场情绪分析模块暂不可用")
    try:
        overview = _sentiment_analyzer.get_market_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情绪分析失败: {str(e)}")


@app.get("/api/sentiment/{symbol:path}")
async def get_sentiment(
    symbol: str,
    market: str = Query(default=None, description="市场类型: crypto/stock_cn/stock_us"),
    force: bool = Query(default=False, alias="force_refresh", description="强制刷新"),
):
    """获取指定标的的市场情绪分析

    Args:
        symbol: 标的 (BTC/USDT, ETH/USDT, 中芯国际, etc.)
        market: 市场类型 (可选，自动判断)
        force: 强制刷新缓存
    """
    if not SENTIMENT_AVAILABLE or _sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="市场情绪分析模块暂不可用")
    try:
        report = _sentiment_analyzer.analyze(symbol, market, force_refresh=force)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情绪分析失败: {str(e)}")


@app.post("/api/sentiment/refresh")
async def refresh_sentiment():
    """强制刷新所有默认标的的情绪分析"""
    if not SENTIMENT_AVAILABLE or _sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="市场情绪分析模块暂不可用")
    try:
        reports = _sentiment_analyzer.analyze_batch(force_refresh=True)
        return {
            "refreshed": len(reports),
            "symbols": [r.symbol for r in reports],
            "timestamp": reports[0].analyzed_at if reports else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量刷新失败: {str(e)}")


# ═══════════════════════════════════════════
# 静态文件 & SPA
# ═══════════════════════════════════════════

@app.get("/")
async def serve_dashboard():
    """仪表板首页"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        from starlette.responses import FileResponse as StarletteFileResponse
        resp = FileResponse(index_file)
        # 缓存策略: 浏览器缓存 5 分钟，CDN 缓存 10 分钟
        resp.headers["Cache-Control"] = "public, max-age=300, s-maxage=600"
        resp.headers["ETag"] = f"\"{index_file.stat().st_mtime:.0f}\""
        return resp
    return {
        "message": "Chase量化策略 API Server 🐾",
        "docs": "/docs",
        "dashboard": "请将 index.html 放入 static/ 目录",
        "endpoints": [
            "/api/health",
            "/api/portfolio",
            "/api/positions",
            "/api/trades",
            "/api/equity-curve",
            "/api/performance",
            "/api/market-snapshots",
            "/api/market-data?symbol=",
            "/api/signals",
            "/api/ml-insights",
            "/api/risk",
            "/api/alpha-discoveries",
            "/api/strategies",
            "/api/sentiment/overview",
            "/api/sentiment/{symbol}",
            "/api/sentiment/refresh",
        ],
    }


# ── 静态文件挂载 (CSS, JS, 图片等) ──
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Chase量化策略 API Server 🐾")
    parser.add_argument("--port", type=int, default=8766, help="服务端口 (默认: 8766)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--production", action="store_true", help="生产模式: 启用速率限制+禁用/docs+安全加固")
    args = parser.parse_args()

    # 设置环境变量供 security 模块读取
    if args.production:
        os.environ["PRODUCTION"] = "true"

    mode_badge = "🔒 生产模式" if (args.production or PRODUCTION_MODE) else "🛠️ 开发模式"
    print(f"""
╔══════════════════════════════════════════╗
║  🐾 Chase量化策略 API Server v2.5      ║
║                                          ║
║  📊 仪表板:  http://localhost:{args.port}    ║
║  📖 API文档: http://localhost:{args.port}/docs ║
║  ❤️  健康检查: http://localhost:{args.port}/api/health ║
║  🛡️  运行模式: {mode_badge:<20}      ║
╚══════════════════════════════════════════╝
""")

    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
