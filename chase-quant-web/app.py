"""
Chase的量化策略 🐾 — 自主量化交易仪表板
Streamlit 本地 Web APP · 虚拟盘 · 三市场
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))

from portfolio import PortfolioManager, ALLOCATION, INITIAL_CAPITAL
from signals import SignalEngine
from risk import RiskController

# ML增强信号引擎
try:
    from ml_signal_v4 import MLSignalEngineV4, SubSignalBuilder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Qlib深度学习模型 (Phase 9)
try:
    from ml_signal_v5 import MLSignalEngineV5, FusionSignal
    from qlib_trainer import QlibTrainer
    from qlib_models import MODEL_REGISTRY
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

# Rolling Trainer (Phase 10)
try:
    from rolling_trainer import RollingTrainer, RollingModelRegistry, auto_rolling_check
    ROLLING_AVAILABLE = True
except ImportError:
    ROLLING_AVAILABLE = False

# ── 页面配置 ──
st.set_page_config(
    page_title="Chase的量化策略 🐾",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 暗色主题 ──
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .metric-card {
        background: #1a1d24; border-radius: 12px; padding: 16px;
        margin: 4px; border: 1px solid #2a2d34;
    }
    .buy-signal { color: #00ff88; font-weight: bold; }
    .sell-signal { color: #ff4444; font-weight: bold; }
    .hold-signal { color: #888888; }
    .alert-danger { color: #ff4444; font-weight: bold; padding: 8px;
                     background: #2a1010; border-radius: 8px; margin: 4px 0; }
    .alert-warning { color: #ffaa00; padding: 8px;
                     background: #2a2010; border-radius: 8px; margin: 4px 0; }
    .alert-info { color: #4488ff; padding: 8px;
                  background: #101a2a; border-radius: 8px; margin: 4px 0; }
    .reason-box {
        background: #151820; border-left: 3px solid #4488ff;
        padding: 8px 12px; margin: 4px 0; border-radius: 0 8px 8px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── 初始化 ──
@st.cache_resource
def get_managers():
    pf = PortfolioManager()
    risk = RiskController(pf)
    return pf, risk

pf, risk_ctrl = get_managers()

# ── 侧边栏 ──
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/dog-face.png", width=64)
    st.title("Chase的量化策略")
    st.caption("🐾 自主量化交易 · 虚拟盘")
    st.divider()

    # 净值总览
    total_val = pf.total_value
    total_pnl = pf.total_pnl
    total_pnl_pct = pf.total_pnl_pct

    col1, col2 = st.columns(2)
    with col1:
        st.metric("总资产", f"¥{total_val:,.2f}")
    with col2:
        st.metric("总盈亏",
                  f"{'¥' if total_pnl >=0 else '-¥'}{abs(total_pnl):,.2f}",
                  delta=f"{total_pnl_pct:+.2f}%")

    st.divider()

    # 月度目标进度
    monthly_progress = total_pnl_pct / 30.0 * 100
    st.caption(f"🎯 月目标 30% 进度")
    st.progress(min(100, max(0, monthly_progress / 100)), text=f"{total_pnl_pct:+.1f}%/30%")

    st.divider()

    # 各市场分配
    st.caption("💰 资金分配")
    alloc = pf.get_allocation_summary()
    for market, info in alloc.items():
        pct = info["total"] / total_val * 100 if total_val > 0 else 0
        st.metric(
            f"{info['label']}  {pct:.0f}%",
            f"¥{info['total']:,.0f}",
            delta=f"{info['pnl_pct']:+.1f}%",
        )

    st.divider()
    st.caption(f"⚡ 初始本金: ¥{INITIAL_CAPITAL:,.0f}")
    st.caption(f"📅 运行中 · 实时监控")

    # 刷新按钮
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Tab 页 ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 总览", "📈 信号", "💼 持仓", "📋 交易记录", "🛡️ 风控", "🧬 ML信号", "🧠 Qlib深度模型"
])

# ═══════════════════════════════════════════
# Tab 1: 总览
# ═══════════════════════════════════════════
with tab1:
    st.header("📊 仪表板总览")

    # 顶部指标卡
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总资产", f"¥{total_val:,.2f}",
                  delta=f"{total_pnl_pct:+.2f}%")
    with col2:
        pos_count = len(pf.open_positions)
        st.metric("持仓数", f"{pos_count}/5")
    with col3:
        pos_val = pf.positions_value
        st.metric("持仓市值", f"¥{pos_val:,.2f}")
    with col4:
        st.metric("可用现金", f"¥{pf.total_cash:,.2f}")
    with col5:
        risk_report = risk_ctrl.daily_risk_report()
        survival = risk_report["survival_score"]
        st.metric("存活评分", f"{survival:.0f}/100",
                  delta="🟢 安全" if survival > 60 else "🟡 注意" if survival > 30 else "🔴 危险")

    # 净值曲线
    st.subheader("📈 净值曲线")
    if pf.snapshots:
        snap_df = pd.DataFrame([{
            "时间": pd.to_datetime(s.time),
            "总资产": s.total_value,
        } for s in pf.snapshots])
        # 添加初始点
        if len(snap_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=snap_df["时间"], y=snap_df["总资产"],
                mode="lines+markers", name="净值",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.05)",
            ))
            fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash",
                         line_color="gray", annotation_text="本金线")
            fig.update_layout(
                template="plotly_dark",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="", yaxis_title="¥",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # 初始快照
        pf.take_snapshot()
        st.info("📡 净值曲线开始记录... 刷新后显示")

    # 资产分布饼图 + 各市场表现
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🥧 资产分布")
        alloc = pf.get_allocation_summary()
        pie_labels = []
        pie_values = []
        pie_colors = []
        for market, info in alloc.items():
            if info["total"] > 0:
                pie_labels.append(info["label"])
                pie_values.append(info["total"])
                pie_colors.append(info["color"])
        if pie_values:
            fig = go.Figure(data=[go.Pie(
                labels=pie_labels, values=pie_values,
                marker=dict(colors=pie_colors),
                hole=0.4, textinfo="label+percent",
            )])
            fig.update_layout(template="plotly_dark", height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 各市场盈亏")
        bars = []
        for market, info in alloc.items():
            bars.append({
                "市场": info["label"],
                "盈亏": info["pnl"],
                "盈亏%": info["pnl_pct"],
            })
        bar_df = pd.DataFrame(bars)
        colors = ["#00ff88" if x >= 0 else "#ff4444" for x in bar_df["盈亏"]]
        fig = go.Figure(data=[go.Bar(
            x=bar_df["市场"], y=bar_df["盈亏"],
            marker_color=colors,
            text=[f"¥{v:+.0f}" for v in bar_df["盈亏"]],
            textposition="outside",
        )])
        fig.update_layout(template="plotly_dark", height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 最近交易
    st.subheader("🔄 最近交易")
    if pf.trades:
        recent = sorted(pf.trades, key=lambda t: t.time, reverse=True)[:10]
        trade_data = []
        for t in recent:
            trade_data.append({
                "时间": pd.to_datetime(t.time).strftime("%m-%d %H:%M"),
                "市场": t.market, "标的": f"{t.symbol} {t.name}",
                "方向": "🟢 买入" if t.side == "buy" else "🔴 卖出",
                "价格": f"¥{t.price:.2f}",
                "金额": f"¥{t.amount:.0f}",
                "盈亏": f"¥{t.pnl:+.0f}" if t.side == "sell" else "-",
                "原因": t.reason[:60],
            })
        st.dataframe(
            pd.DataFrame(trade_data),
            use_container_width=True,
            column_config={"原因": st.column_config.TextColumn(width="large")},
        )
    else:
        st.info("📭 暂无交易记录 — 等待信号触发")


# ═══════════════════════════════════════════
# Tab 2: 信号
# ═══════════════════════════════════════════
with tab2:
    st.header("📈 今日交易信号")

    if st.button("🔍 扫描全市场", type="primary"):
        with st.spinner("正在扫描三大市场..."):
            engine = SignalEngine()
            all_signals = engine.scan_all()

            for market, signals in all_signals.items():
                market_names = {"crypto": "₿ 加密货币", "a_stock": "🇨🇳 A股", "us_stock": "🇺🇸 美股"}
                st.subheader(f"{market_names.get(market, market)} — Top 5 买入信号")

                if not signals:
                    st.warning("无符合条件的买入信号 (score ≥ 65)")
                    continue

                for s in signals[:5]:
                    score_color = "#00ff88" if s.score >= 75 else "#ffaa00" if s.score >= 65 else "#888"
                    st.markdown(f"""
                    <div style="background:#1a1d24; border-radius:12px; padding:16px; margin:8px 0; border:1px solid #2a2d34;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:18px; font-weight:bold;">{s.symbol}</span>
                                <span style="color:#888; margin-left:8px;">{s.name}</span>
                            </div>
                            <div style="text-align:right;">
                                <span style="font-size:24px; color:{score_color}; font-weight:bold;">{s.score:.0f}分</span>
                                <span class="buy-signal" style="margin-left:8px;">{s.action}</span>
                            </div>
                        </div>
                        <div style="margin-top:8px; color:#aaa;">
                            现价: ¥{s.price:.2f} | 置信度: {s.confidence:.0%} |
                            风险: {s.risk_level.upper()} |
                            建议仓位: ¥{s.suggested_size:.0f} |
                            止损: ¥{s.stop_loss:.2f}
                        </div>
                        <div style="margin-top:8px;">
                            {''.join(f'<span class="reason-box">{r}</span>' for r in s.reasons)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 执行交易按钮
                    col_btn1, col_btn2 = st.columns([1, 4])
                    with col_btn1:
                        if st.button(f"✅ 执行买入 {s.symbol}", key=f"buy_{s.symbol}_{s.market}"):
                            quantity = s.suggested_size / s.price
                            trade = pf.buy(
                                market=s.market, symbol=s.symbol, name=s.name,
                                price=s.price, quantity=quantity,
                                reason="\n".join(s.reasons),
                            )
                            if trade:
                                pf.take_snapshot()
                                st.success(f"🎉 {s.symbol} 买入成功! ¥{s.suggested_size:.0f}")
                                st.rerun()
                            else:
                                st.error("❌ 买入失败 (资金不足或风控拦截)")

    # 自动刷新提示
    st.caption("💡 点击「扫描全市场」更新信号 | 信号每5分钟自动更新建议")

    # 历史信号表现的简单统计
    st.divider()
    st.subheader("📊 信号胜率")
    if pf.trades:
        sell_trades = [t for t in pf.trades if t.side == "sell" and t.pnl != 0]
        if sell_trades:
            wins = sum(1 for t in sell_trades if t.pnl > 0)
            total = len(sell_trades)
            st.metric("已完成交易胜率", f"{wins}/{total} ({wins/total*100:.0f}%)")
            total_pnl = sum(t.pnl for t in sell_trades)
            st.metric("已实现盈亏", f"¥{total_pnl:+.2f}")
    else:
        st.info("尚无完成交易")


# ═══════════════════════════════════════════
# Tab 3: 持仓
# ═══════════════════════════════════════════
with tab3:
    st.header("💼 当前持仓")

    open_pos = pf.open_positions
    if not open_pos:
        st.info("📭 无持仓 — 等待交易信号触发")
    else:
        for pos in open_pos:
            pnl_color = "#00ff88" if pos.pnl_pct >= 0 else "#ff4444"
            market_icons = {"crypto": "₿", "a_stock": "🇨🇳", "us_stock": "🇺🇸"}

            # 计算仓位占比
            pos_pct = pos.value / pf.total_value * 100 if pf.total_value > 0 else 0

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"""
                **{market_icons.get(pos.market, '')} {pos.symbol}** — {pos.name}
                <br><small>入场: ¥{pos.entry_price:.2f} | 现价: ¥{pos.current_price:.2f} |
                数量: {pos.quantity:.4f}</small>
                <br><small style="color:{pnl_color}">盈亏: ¥{pos.pnl:+.2f} ({pos.pnl_pct:+.2f}%) |
                仓位: {pos_pct:.1f}%</small>
                """, unsafe_allow_html=True)

            with col2:
                # 进度条
                pnl_pct_clamped = max(-10, min(15, pos.pnl_pct))
                st.progress((pnl_pct_clamped + 10) / 25,
                           text=f"{pos.pnl_pct:+.1f}%")

            with col3:
                if st.button("🔴 平仓", key=f"sell_{pos.id}"):
                    trade = pf.sell(pos.id, pos.current_price, "手动平仓")
                    if trade:
                        pf.take_snapshot()
                        st.rerun()

            # 入场理由
            st.markdown(f'<div class="reason-box">📝 {pos.entry_reason}</div>',
                       unsafe_allow_html=True)
            st.divider()


# ═══════════════════════════════════════════
# Tab 4: 交易记录
# ═══════════════════════════════════════════
with tab4:
    st.header("📋 完整交易记录")

    if not pf.trades:
        st.info("📭 暂无交易")
    else:
        trades_sorted = sorted(pf.trades, key=lambda t: t.time, reverse=True)
        data = []
        for t in trades_sorted:
            data.append({
                "时间": pd.to_datetime(t.time).strftime("%m-%d %H:%M:%S"),
                "市场": t.market,
                "代码": t.symbol,
                "名称": t.name,
                "方向": "买入" if t.side == "buy" else "卖出",
                "价格": round(t.price, 2),
                "数量": round(t.quantity, 4),
                "金额": round(t.amount, 2),
                "手续费": round(t.fee, 2),
                "盈亏": round(t.pnl, 2) if t.side == "sell" else 0,
                "盈亏%": f"{t.pnl_pct:+.1f}%" if t.side == "sell" else "-",
                "原因": t.reason[:80],
            })

        df = pd.DataFrame(data)

        # 统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总交易", len(df))
        with col2:
            buy_count = len(df[df["方向"] == "买入"])
            sell_count = len(df[df["方向"] == "卖出"])
            st.metric("买/卖", f"{buy_count}/{sell_count}")
        with col3:
            total_realized = df[df["方向"] == "卖出"]["盈亏"].sum()
            st.metric("已实现盈亏", f"¥{total_realized:+.2f}")

        st.dataframe(df, use_container_width=True, height=400)


# ═══════════════════════════════════════════
# Tab 5: 风控
# ═══════════════════════════════════════════
with tab5:
    st.header("🛡️ 风控中心")

    rr = risk_ctrl.daily_risk_report()

    # 存活评分大卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "🟢" if rr["survival_score"] > 60 else "🟡" if rr["survival_score"] > 30 else "🔴"
        st.metric(f"{color} 存活评分", f"{rr['survival_score']}/100")
    with col2:
        st.metric("📉 日亏损限额使用", f"{rr['daily_loss_used']:.0f}%",
                  delta="安全" if rr["daily_loss_used"] < 60 else "警告")
    with col3:
        st.metric("🎯 月度目标进度", f"{rr['monthly_progress']:.1f}%",
                  delta=f"{rr['total_pnl_pct']:+.1f}%/30%")
    with col4:
        st.metric("⚠️ 活跃警报", len(rr["alerts"]))

    st.divider()

    # 风控规则
    st.subheader("📜 风控铁律")
    rules = [
        ("🔴 硬止损", "-8% 无条件止损", "单笔亏损触及-8%立即平仓"),
        ("💰 单笔上限", "≤ 总资金 2% (¥200)", f"当前最大单笔亏损允许: ¥{total_val * 0.02:.0f}"),
        ("📦 持仓上限", "≤ 5 只同时持有", f"当前持仓: {rr['open_positions']} 只"),
        ("📊 仓位上限", "≤ 40% 单仓位", "超过40%自动减仓"),
        ("🛑 日熔断", "日亏损 > 5% 停止交易", f"当前日亏损: {total_pnl_pct:.1f}%"),
        ("✅ 入场最低分", "信号 ≥ 65 分", "过滤低质量信号"),
    ]
    for title, rule, status in rules:
        st.markdown(f"""
        <div style="background:#1a1d24; border-radius:8px; padding:12px; margin:6px 0;">
            <strong>{title}</strong>: {rule}
            <br><small style="color:#888;">{status}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 实时警报
    st.subheader("🚨 实时警报")
    alerts = pf.check_risk()
    if alerts:
        for alert in alerts:
            if "🔴" in alert:
                st.markdown(f'<div class="alert-danger">{alert}</div>', unsafe_allow_html=True)
            elif "⚠️" in alert:
                st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-info">{alert}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ 无风险警报 — 一切正常")

    # 持仓风险详情
    st.subheader("🔍 持仓风险扫描")
    for pos in pf.open_positions:
        checks = risk_ctrl.position_check(pos.id)
        if checks:
            for c in checks:
                if not c.passed:
                    cls = "alert-danger" if c.alert_level == "danger" else "alert-warning"
                    st.markdown(f'<div class="{cls}">{pos.symbol}: {c.reason}</div>',
                               unsafe_allow_html=True)

# ═══════════════════════════════════════════
# Tab 6: ML增强信号
# ═══════════════════════════════════════════
with tab6:
    st.header("🧬 ML增强信号引擎 v4.0")

    if not ML_AVAILABLE:
        st.warning("⚠️ ML信号引擎暂不可用 — 请先运行 feature_backtest_v4.py 生成回测结果")
    else:
        st.markdown("""
        > 🧬 **西蒙斯风格**: 286个特征 → FDR筛选28+ → 7个独立子信号 → 组合决策
        >
        > 不靠人拍脑袋判断"这个指标有道理" — **让数据说话**。
        """)

        # LightGBM toggle + Phase 6 cross-market status
        lgbm_col1, lgbm_col2 = st.columns([1, 3])
        with lgbm_col1:
            use_lgbm = st.checkbox("🧠 LightGBM增强", value=True,
                                   help="启用独立LightGBM模型预测各主题收益")
        with lgbm_col2:
            status_lines = []
            if use_lgbm:
                model_dir = Path(__file__).parent / "data" / "models"
                lgbm_models = list(model_dir.glob("lgbm_*.pkl")) if model_dir.exists() else []
                if lgbm_models:
                    status_lines.append(f"✅ {len(lgbm_models)}个模型已加载")
                else:
                    status_lines.append("⚠️ 模型未训练 — 运行 `python3 ml_lightgbm_trainer.py` 生成")
            # Phase 6: 跨市场状态
            from ml_cross_market import CrossMarketFetcher
            cm = CrossMarketFetcher(cache_ttl_hours=4)
            if cm.available_count() > 0:
                status_lines.append(f"🌐 跨市场数据: {cm.available_count()}/9源可用")
            else:
                status_lines.append("🌐 跨市场: 缓存未建立, 首次信号生成时拉取")
            st.caption(" | ".join(status_lines))

        if st.button("🧠 生成ML信号", type="primary", key="ml_generate"):
            spinner_text = "正在计算286个特征 + 构建7个子信号 + 跨市场数据"
            if use_lgbm:
                spinner_text += " + LightGBM预测..."
            with st.spinner(spinner_text):
                try:
                    import ccxt
                    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})

                    # BTC
                    ohlcv_btc = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
                    df_btc = pd.DataFrame(ohlcv_btc, columns=["date","open","high","low","close","volume"])

                    engine = MLSignalEngineV4()
                    signal = engine.generate_signal(df_btc, "BTC/USDT", use_lgbm=use_lgbm)

                    # ── 信号总览卡 ──
                    action_color = {
                        "BUY": "#00ff88", "SELL": "#ff4444", "HOLD": "#ffaa00"
                    }.get(signal.action, "#888")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:16px; padding:20px;
                                    text-align:center; border:2px solid {action_color};">
                            <div style="font-size:14px; color:#888;">操作</div>
                            <div style="font-size:36px; font-weight:bold; color:{action_color};">
                                {signal.action}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        primary_label = "LGBM信号" if (use_lgbm and signal.lgbm_available) else "IC加权信号"
                        primary_val = signal.signal_lgbm if (use_lgbm and signal.lgbm_available) else signal.signal_ic
                        st.metric(primary_label, f"{primary_val:+.2f}",
                                 delta=f"等权:{signal.signal_equal:+.2f} / 波动:{signal.signal_vol:+.2f}")
                    with col3:
                        st.metric("共识度", f"{signal.consensus:.0%}",
                                 delta=f"置信度:{signal.confidence:.0%}")
                    with col4:
                        st.metric("建议仓位", f"{signal.suggested_size_pct:.0f}%",
                                 delta=f"风险调整:{signal.risk_adjusted:+.2f}")

                    # ── LGBM vs 线性对比 ──
                    if use_lgbm and signal.lgbm_available:
                        st.divider()
                        st.subheader("🧠 LightGBM vs 线性IC加权 对比")
                        compare_col1, compare_col2, compare_col3, compare_col4 = st.columns(4)
                        with compare_col1:
                            st.metric("线性IC加权", f"{signal.signal_ic:+.3f}")
                        with compare_col2:
                            st.metric("LightGBM", f"{signal.signal_lgbm:+.3f}",
                                     delta=f"{(signal.signal_lgbm - signal.signal_ic):+.3f}")
                        with compare_col3:
                            agreement = "✅ 一致" if np.sign(signal.signal_ic) == np.sign(signal.signal_lgbm) else "⚠️ 分歧"
                            st.metric("一致性", agreement)
                        with compare_col4:
                            lgbm_action = "BUY" if signal.signal_lgbm > 0.5 else "SELL" if signal.signal_lgbm < -0.5 else "HOLD"
                            st.metric("LGBM操作", lgbm_action)

                    st.divider()

                    # ── 子信号雷达图 ──
                    st.subheader("🎯 7个子信号")
                    sig_data = []
                    for s in signal.sub_signals:
                        sig_data.append({
                            "子信号": s.name,
                            "信号值": s.value,
                            "置信度": s.confidence,
                            "方向": s.direction,
                            "贡献特征": len(s.contributing_features),
                        })
                    sig_df = pd.DataFrame(sig_data)

                    # 用条形图展示子信号
                    colors = [
                        "#00ff88" if d == "LONG" else "#ff4444" if d == "SHORT" else "#888"
                        for d in sig_df["方向"]
                    ]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=sig_df["子信号"], x=sig_df["信号值"],
                        orientation="h", marker_color=colors,
                        text=[f"{v:+.2f}" for v in sig_df["信号值"]],
                        textposition="outside",
                        hovertemplate="%{y}: %{x:+.2f}<br>置信度: %{customdata:.0%}<br>特征数: %{text}",
                        customdata=sig_df["置信度"],
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        template="plotly_dark", height=250,
                        margin=dict(l=0, r=40, t=10, b=0),
                        xaxis=dict(range=[-3.5, 3.5], title="信号值"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── 子信号详情 ──
                    st.subheader("🔍 子信号推理")
                    for s in signal.sub_signals:
                        icon = "🟢" if s.direction == "LONG" else "🔴" if s.direction == "SHORT" else "⚪"
                        feat_list = ", ".join(s.contributing_features[:5])
                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:12px; padding:12px;
                                    margin:6px 0; border-left:3px solid {colors[sig_df[sig_df['子信号']==s.name].index[0]] if s.name in sig_df['子信号'].values else '#888'}">
                            <strong>{icon} {s.name}</strong> — 信号: {s.value:+.2f} | 置信: {s.confidence:.0%}
                            <br><small style="color:#888;">特征: {feat_list}</small>
                            <br><small style="color:#aaa;">{s.reasoning}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    # ── 三市场扫描 ──
                    st.subheader("🌍 三市场ML扫描")

                    markets = {
                        "₿ 加密货币": [
                            ("BTC/USDT", ohlcv_btc),
                            ("ETH/USDT", None),  # lazy fetch
                        ],
                    }

                    for market_name, assets in markets.items():
                        for sym, data in assets:
                            if data is None:
                                try:
                                    data = exchange.fetch_ohlcv(sym, "1d", limit=400)
                                except Exception:
                                    continue

                            df_m = pd.DataFrame(data, columns=["date","open","high","low","close","volume"])
                            sig_m = engine.generate_signal(df_m, sym, use_lgbm=use_lgbm)

                            action_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig_m.action, "⚪")
                            primary_sig = sig_m.signal_lgbm if (use_lgbm and sig_m.lgbm_available) else sig_m.signal_ic
                            st.markdown(f"""
                            <div style="background:#1a1d24; border-radius:10px; padding:10px; margin:4px 0;">
                                <strong>{action_icon} {sym}</strong> —
                                信号: {primary_sig:+.2f} |
                                共识: {sig_m.consensus:.0%} |
                                操作: <span style="color:{action_color}; font-weight:bold;">{sig_m.action}</span>
                                {" 🧠" if sig_m.lgbm_available else ""}
                            </div>
                            """, unsafe_allow_html=True)

                    st.success(f"✅ 已计算 {signal.feature_count} 个特征 | {signal.n_sub_signals_active}/7 子信号活跃")

                    # ── LightGBM 特征重要性 ──
                    if signal.lgbm_available and hasattr(engine, '_lgbm_feature_importance'):
                        with st.expander("🔬 LightGBM 特征重要性 Top-5 (各主题)", expanded=False):
                            imp_data = engine._lgbm_feature_importance
                            for theme_id, feats in imp_data.items():
                                if not feats:
                                    continue
                                theme_name = SubSignalBuilder.THEME_CONFIG.get(theme_id, {}).get("name", theme_id)
                                st.caption(f"**{theme_name}**")
                                feat_df = pd.DataFrame(feats[:5])
                                feat_df.columns = ["特征", "重要性"]
                                feat_df["特征"] = feat_df["特征"].apply(lambda x: f"`{x}`")
                                st.dataframe(feat_df, hide_index=True, use_container_width=True)

                except ImportError:
                    st.error("❌ ccxt 未安装 — 无法获取实时数据")
                except Exception as e:
                    st.error(f"❌ 信号生成失败: {e}")

        # 架构说明
        with st.expander("📐 架构说明"):
            st.markdown("""
            ### 🧬 ML信号引擎 v4.0 + 🌐跨市场 架构

            ```
            286个特征 (17个类别 A-T)
              ↓ FDR校正 + ICIR筛选
            28个存活特征 (p_fdr < 0.1 & |ICIR| > 0.3)
              ↓ 按主题分配 (每个特征只属于一个子信号)
            ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
            │ 趋势跟踪  │ 均值回归  │ 量价确认  │ 波动突破  │ 尾部风险  │ 动量增强  │ 跨市场联动│
            │ (动量+均线)│(振荡+形态)│(量+OBV) │(波动率)  │(偏度峰度)│(排名+Hurst)│(T类+资金费率)│
            └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
              ↓ IC加权平均
            最终信号 [-3, +3]
              ↓
            BUY (>0.5) / HOLD / SELL (<-0.5)
            ```

            ### 与旧版区别

            | 维度 | 旧版 (signals.py) | 新版 v4.0 (IC加权) | v4.0 + LightGBM + 🌐 |
            |------|:---:|:---:|:---:|
            | 特征数 | 15个人工指标 | 279个数据驱动特征 | 286个 (含46跨市场) |
            | 筛选方式 | 人类判断 | FDR + ICIR 统计检验 | FDR + ICIR 统计检验 |
            | 信号构建 | 固定阈值打分 | 子信号×IC加权组合 | **LightGBM非线性预测** |
            | 子信号 | 无 | 6个独立主题 | 7个独立LightGBM模型 |
            | 跨市场 | ❌ | ❌ | ✅ ETH/SPY/DXY/VIX/F&G |
            | 可解释性 | 黑盒总分 | 每个子信号独立可查 | 特征重要性 + 预测收益 |
            | 非线性 | 无 | LightGBM-ready | ✅ 已实现 |
            | ICIR | 未测量 | 0.3-4.77 | OOS 0.11-0.71 |
            | 过拟合防护 | 无 | PurgedKFold | PurgedKFold + EarlyStopping |
            """)

        # ── Phase 7: 策略回测结果 ──
        bt_results_path = Path(__file__).parent / "data" / "backtest_results" / "latest.json"
        if bt_results_path.exists():
            with st.expander("📈 策略回测 — ML策略历史表现", expanded=True):
                try:
                    import json
                    with open(bt_results_path) as f:
                        bt = json.load(f)

                    # 关键指标卡
                    bt_col1, bt_col2, bt_col3, bt_col4, bt_col5 = st.columns(5)
                    with bt_col1:
                        bt_total = bt.get("total_return_pct", 0)
                        st.metric("总收益", f"{bt_total:+.1f}%",
                                 delta=f"vs 买入持有 {bt.get('benchmark_return_pct', 0):+.1f}%")
                    with bt_col2:
                        st.metric("Sharpe比率", f"{bt.get('sharpe_ratio', 0):.2f}")
                    with bt_col3:
                        st.metric("最大回撤", f"-{bt.get('max_drawdown_pct', 0):.2f}%")
                    with bt_col4:
                        st.metric("胜率", f"{bt.get('win_rate_pct', 0):.0f}%")
                    with bt_col5:
                        st.metric("盈亏比", f"{bt.get('profit_factor', 0):.1f}")

                    bt_col6, bt_col7, bt_col8 = st.columns(3)
                    with bt_col6:
                        st.metric("交易笔数", bt.get("n_trades", 0))
                    with bt_col7:
                        st.metric("超额收益 α", f"{bt.get('alpha_pct', 0):+.1f}%")
                    with bt_col8:
                        st.metric("信号准确率", f"{bt.get('signal_accuracy', 0):.0%}")

                    # 权益曲线
                    equity_html = Path(__file__).parent / "data" / "backtest_results" / "equity_curve.html"
                    if equity_html.exists():
                        with open(equity_html) as f:
                            st.components.v1.html(f.read(), height=520)

                    # 交易明细
                    trades = bt.get("trades", [])
                    if trades:
                        st.caption(f"📜 最近交易明细 ({len(trades)}笔)")
                        trade_rows = []
                        for t in trades[-8:]:
                            trade_rows.append({
                                "入场": t.get("entry_time", "")[:10],
                                "出场": t.get("exit_time", "")[:10],
                                "持有时长": f"{t.get('holding_days', 0)}天",
                                "盈亏": f"{t.get('pnl_pct', 0):+.2f}%",
                                "出场原因": t.get("exit_reason", ""),
                            })
                        st.dataframe(
                            pd.DataFrame(trade_rows),
                            hide_index=True, use_container_width=True,
                            column_config={
                                "盈亏": st.column_config.NumberColumn(format="%.2f%%"),
                            }
                        )

                except Exception as e:
                    st.caption(f"⚠️ 回测结果加载失败: {e}")
        else:
            with st.expander("📈 策略回测 — ML策略历史表现", expanded=False):
                st.caption("💡 运行 `python3 strategy_backtest.py` 生成回测结果")

        # ── Phase 8: 参数优化结果 ──
        opt_results_path = Path(__file__).parent / "data" / "optimization_results" / "latest_optimize.json"
        if opt_results_path.exists():
            with st.expander("🎯 参数优化 — 最优参数 vs 默认参数", expanded=False):
                try:
                    import json
                    with open(opt_results_path) as f:
                        opt = json.load(f)

                    best = opt.get("best_params", {})
                    best_score = opt.get("best_score", 0)
                    baseline = opt.get("baseline_score", 0)
                    improvement = (best_score - baseline) / max(0.001, abs(baseline)) * 100

                    st.caption(f"🏆 最优复合得分: **{best_score:.4f}** | "
                              f"默认: {baseline:.4f} | "
                              f"提升: {improvement:+.1f}%")

                    # 最优vs默认对比
                    default_params = {
                        "entry_threshold": 0.5, "stop_loss": -0.08,
                        "take_profit": 0.15, "max_position": 0.40, "warmup_days": 200,
                    }
                    comp_rows = []
                    for k, dv in default_params.items():
                        bv = best.get(k, dv)
                        if isinstance(dv, float):
                            change = f"{(bv/dv - 1)*100:+.0f}%"
                        else:
                            change = "—"
                        comp_rows.append({
                            "参数": k,
                            "默认值": f"{dv:.3f}" if isinstance(dv, float) else str(dv),
                            "最优值": f"{bv:.3f}" if isinstance(bv, float) else str(int(bv)),
                            "变化": change,
                        })
                    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

                    # 参数重要性
                    imp = opt.get("param_importance", {})
                    if imp:
                        st.caption("📊 参数重要性 (得分方差贡献)")
                        imp_sorted = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)
                        imp_df = pd.DataFrame(imp_sorted, columns=["参数", "重要性"])
                        st.bar_chart(imp_df.set_index("参数"), use_container_width=True)

                    # 敏感性热力图
                    sens_html = Path(__file__).parent / "data" / "optimization_results" / "sensitivity_heatmap.html"
                    if sens_html.exists():
                        with open(sens_html) as f:
                            st.components.v1.html(f.read(), height=420)

                    # Top-5 参数组合
                    trials = opt.get("trials", [])
                    if trials:
                        st.caption("🔝 Top-5 参数组合")
                        top5 = sorted(trials, key=lambda t: t.get("score", -99), reverse=True)[:5]
                        top_rows = []
                        for t in top5:
                            top_rows.append({
                                "Score": f"{t.get('score', 0):.3f}",
                                "Sharpe": f"{t.get('sharpe', 0):.3f}",
                                "MaxDD": f"{t.get('max_dd', 0):.1f}%",
                                "Entry": f"{t.get('entry_threshold', 0):.2f}",
                                "Stop": f"{t.get('stop_loss', 0):.2f}",
                                "TP": f"{t.get('take_profit', 0):.2f}",
                                "Pos%": f"{t.get('max_position', 0):.0%}",
                                "Warm": int(t.get('warmup_days', 0)),
                                "#Tr": int(t.get('n_trades', 0)),
                            })
                        st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

                except Exception as e:
                    st.caption(f"⚠️ 优化结果加载失败: {e}")
        else:
            with st.expander("🎯 参数优化 — 最优参数 vs 默认参数", expanded=False):
                st.caption("💡 运行 `python3 hyperparam_optimizer.py` 生成优化结果")

        # ── Phase 8: Walk-Forward验证结果 ──
        wf_results_path = Path(__file__).parent / "data" / "optimization_results" / "walk_forward_results.json"
        if wf_results_path.exists():
            with st.expander("📊 Walk-Forward验证 — 滚动OOS表现", expanded=False):
                try:
                    import json
                    with open(wf_results_path) as f:
                        wf = json.load(f)

                    summary = wf.get("summary", {})
                    stability = wf.get("stability_score", 0)

                    wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)
                    with wf_col1:
                        st.metric("OOS Sharpe均值", f"{summary.get('avg_sharpe', 0):.3f}")
                    with wf_col2:
                        st.metric("OOS MaxDD均值", f"-{summary.get('avg_maxdd', 0):.1f}%")
                    with wf_col3:
                        st.metric("OOS 收益均值", f"{summary.get('avg_return', 0):+.1f}%")
                    with wf_col4:
                        all_pos = "✅ 全部为正" if summary.get("all_positive") else "⚠️ 有负值"
                        st.metric("OOS全正?", all_pos)

                    st.metric("参数稳定性", f"{stability:.0f}/100",
                             delta="高" if stability > 80 else ("中" if stability > 50 else "低"))

                    # W-F热力图
                    wf_html = Path(__file__).parent / "data" / "optimization_results" / "walk_forward_heatmap.html"
                    if wf_html.exists():
                        with open(wf_html) as fh:
                            st.components.v1.html(fh.read(), height=380)

                    # 各窗口详情
                    windows = wf.get("windows", [])
                    if windows:
                        st.caption("🪟 各窗口OOS详情")
                        wf_rows = []
                        for w in windows:
                            wf_rows.append({
                                "#": w.get("window_id", ""),
                                "Test区间": f"{w.get('test_start', '')}→{w.get('test_end', '')}",
                                "Sharpe": f"{w.get('test_sharpe', 0):.3f}",
                                "MaxDD": f"-{w.get('test_maxdd', 0):.1f}%",
                                "策略收益": f"{w.get('test_return', 0):+.1f}%",
                                "基准收益": f"{w.get('benchmark_return', 0):+.1f}%",
                                "交易数": w.get("n_trades", 0),
                            })
                        st.dataframe(pd.DataFrame(wf_rows), hide_index=True, use_container_width=True)

                except Exception as e:
                    st.caption(f"⚠️ W-F结果加载失败: {e}")
        else:
            with st.expander("📊 Walk-Forward验证 — 滚动OOS表现", expanded=False):
                st.caption("💡 运行 `python3 walk_forward_validator.py` 生成W-F结果")

        # ── Phase 8: 模型版本管理 ──
        try:
            from ml_lightgbm_trainer import ModelRegistry
            registry = ModelRegistry()
            model_summary = registry.get_summary()
            if model_summary:
                with st.expander("🔄 模型管理 — LightGBM模型版本", expanded=False):
                    st.caption(f"📦 {len(model_summary)}个主题模型")

                    model_rows = []
                    for s in sorted(model_summary, key=lambda x: x["age_days"], reverse=True):
                        model_rows.append({
                            "主题": s["theme_name"],
                            "训练日期": s["latest_trained_at"],
                            "最新ICIR": f"{s['latest_oos_icir']:.3f}",
                            "最佳ICIR": f"{s['best_oos_icir']:.3f}",
                            "年龄": s["age_label"],
                            "版本数": s["n_versions"],
                        })
                    st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

                    # 老化提醒
                    stale = [s for s in model_summary if s["age_days"] > 30]
                    if stale:
                        stale_names = ", ".join(s["theme_name"] for s in stale)
                        st.warning(f"⚠️ {len(stale)}个模型超30天: {stale_names} — 建议运行 `python3 ml_lightgbm_trainer.py --retrain`")
                    else:
                        st.success("✅ 所有模型年龄正常 (< 30天)")
            else:
                with st.expander("🔄 模型管理 — LightGBM模型版本", expanded=False):
                    st.caption("💡 运行 `python3 ml_lightgbm_trainer.py` 训练模型后查看版本")
        except Exception:
            pass

# ═══════════════════════════════════════════
# Tab 7: Qlib 深度学习模型
# ═══════════════════════════════════════════
with tab7:
    st.header("🧠 Qlib 深度学习模型 — Phase 9")

    if not QLIB_AVAILABLE:
        st.warning("⚠️ Qlib 模型引擎暂不可用 — 请检查 PyTorch 安装")
        st.code("pip install torch torchvision torchaudio", language="bash")
    else:
        st.markdown("""
        > 🧠 **Qlib 风格深度学习**: 将 Microsoft Qlib 的核心模型架构嫁接到我们的特征矩阵上
        >
        > ALSTM (时序注意力) | Transformer (长程依赖) | TabNet (特征选择) | GATs (资产关系图)
        >
        > 与 LightGBM 互补 — **树模型 + 深度学习 = 王炸组合**
        """)

        # 模型状态
        model_col1, model_col2, model_col3, model_col4 = st.columns(4)
        with model_col1:
            st.metric("模型架构", "4")
        with model_col2:
            try:
                import torch
                device = "🖥️ GPU" if torch.cuda.is_available() else "💻 CPU"
                st.metric("计算设备", device)
            except Exception:
                st.metric("计算设备", "❌")
        with model_col3:
            # 检查已有模型
            model_files = list(Path(__file__).parent.glob("data/models/qlib_*.pth"))
            st.metric("已训练模型", len(model_files))
        with model_col4:
            report_path = Path(__file__).parent / "data" / "qlib_reports" / "qlib_train_latest.json"
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        report = json.load(f)
                    st.metric("最新报告", report.get("generated_at", "?")[:10])
                except Exception:
                    st.metric("最新报告", "未找到")
            else:
                st.metric("最新报告", "未训练")

        st.divider()

        # 模型架构图
        with st.expander("📐 模型架构", expanded=False):
            st.markdown("""
            ### Qlib 深度学习模型架构

            ```
                        FeatureFactoryV4 (286特征 × 17类别)
                                 │
                    ┌────────────┼────────────┬──────────────┐
                    ▼            ▼            ▼              ▼
                 ALSTM       Transformer   TabNet         GATs
              "时序+注意力"   "长程Self-Attn" "特征选择"   "资产关系图"
                    │            │            │              │
                    └────────────┼────────────┴──────────────┘
                                 │
                         模型融合层
                    (ICIR加权 + 共识投票)
                                 │
                            最终信号
                    BUY (>0.5) / HOLD / SELL (<-0.5)
            ```

            | 模型 | 擅长 | 输入 | 特色 |
            |------|------|------|------|
            | **ALSTM** | 时序模式识别 | (T, 60) 窗口 | 双向LSTM + 多头注意力池化 |
            | **Transformer** | 长程依赖 | (T, 60) 窗口 | 位置编码 + Self-Attention |
            | **TabNet** | 特征重要性 | 单步特征向量 | Sparse Attention → 可解释特征选择 |
            | **GATs** | 资产关系 | 多资产特征 + 图 | 注意力聚合邻居节点 |
            """)

        # 训练控制
        st.subheader("🎮 模型训练")
        train_col1, train_col2, train_col3 = st.columns([2, 1, 1])
        with train_col1:
            qlib_model_choice = st.multiselect(
                "选择模型架构",
                options=["alstm", "transformer", "tabnet"],
                default=["alstm", "transformer"],
                help="GATs 需要预建资产关系图, 暂不默认训练",
            )
        with train_col2:
            qlib_epochs = st.slider("训练轮数", 20, 200, 50, 10,
                                    help="更多轮数=更好效果但更慢")
        with train_col3:
            qlib_double = st.checkbox("DoubleEnsemble", value=False,
                                      help="两阶段集成降低过拟合")

        if st.button("🚀 训练 Qlib 模型", type="primary", key="qlib_train_btn"):
            with st.spinner(f"正在训练 {len(qlib_model_choice)} 个模型 × 7个主题..."):
                try:
                    import ccxt
                    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
                    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
                    df_btc = pd.DataFrame(ohlcv, columns=["date","open","high","low","close","volume"])

                    trainer = QlibTrainer({
                        "n_epochs_max": qlib_epochs,
                        "batch_size": 32,
                        "seq_len": 60,
                    })

                    results = trainer.train_all(
                        df_btc,
                        models=qlib_model_choice,
                        use_double_ensemble=qlib_double,
                    )

                    if results:
                        st.success(f"✅ 训练完成! {len(results)} 个模型")
                        # 显示结果表
                        result_rows = []
                        for r in sorted(results, key=lambda x: x.oos_icir, reverse=True):
                            vs_lgbm = f"{r.improvement_vs_lgbm:+.1f}%" if r.lgbm_icir != 0 else "N/A"
                            result_rows.append({
                                "模型": r.model_name,
                                "主题": r.theme_name,
                                "ICIR": f"{r.oos_icir:.4f}",
                                "R²": f"{r.oos_r2:.4f}",
                                "Hit%": f"{r.oos_hit_rate:.1%}",
                                "vs LGBM": vs_lgbm,
                                "特征数": r.n_features_used,
                            })
                        st.dataframe(pd.DataFrame(result_rows), hide_index=True, use_container_width=True)
                    else:
                        st.warning("⚠️ 训练未产生结果, 检查数据是否充足")

                except ImportError:
                    st.error("❌ ccxt 未安装, 无法获取训练数据")
                except Exception as e:
                    st.error(f"❌ 训练失败: {e}")

        st.divider()

        # 实时预测
        st.subheader("🔮 实时预测 — Qlib vs LightGBM 对比")

        if st.button("🧠 生成 Qlib 融合信号", type="primary", key="qlib_predict_btn"):
            with st.spinner("正在运行多模型融合预测..."):
                try:
                    import ccxt
                    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
                    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=400)
                    df_btc = pd.DataFrame(ohlcv, columns=["date","open","high","low","close","volume"])

                    # v4 LightGBM 信号
                    engine_v4 = MLSignalEngineV4()
                    signal_v4 = engine_v4.generate_signal(df_btc, "BTC/USDT", use_lgbm=True)

                    # v5 融合信号
                    engine_v5 = MLSignalEngineV5(use_qlib=True, use_lgbm=True)
                    signal_v5 = engine_v5.generate_signal(df_btc, "BTC/USDT")

                    # ── 对比卡片 ──
                    col_lgbm, col_qlib, col_fusion = st.columns(3)

                    with col_lgbm:
                        lgbm_val = signal_v4.signal_lgbm if signal_v4.lgbm_available else signal_v4.signal_ic
                        lgbm_action_color = "#00ff88" if signal_v4.action == "BUY" else "#ff4444" if signal_v4.action == "SELL" else "#888"
                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:16px; padding:20px; text-align:center; border:2px solid #4488ff;">
                            <div style="font-size:14px; color:#888;">LightGBM (v4)</div>
                            <div style="font-size:32px; font-weight:bold; color:{lgbm_action_color};">{signal_v4.action}</div>
                            <div style="color:#aaa; margin-top:8px;">信号: {lgbm_val:+.4f}</div>
                            <div style="color:#888;">置信: {signal_v4.confidence:.0%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_qlib:
                        # Qlib ensemble 平均
                        qlib_preds = [p for p in signal_v5.model_predictions if p.model_name != "lgbm"]
                        qlib_mean = np.mean([p.prediction for p in qlib_preds]) if qlib_preds else 0
                        qlib_action = "BUY" if qlib_mean > 0.001 else "SELL" if qlib_mean < -0.001 else "HOLD"
                        qlib_color = "#00ff88" if qlib_action == "BUY" else "#ff4444" if qlib_action == "SELL" else "#888"
                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:16px; padding:20px; text-align:center; border:2px solid #ff8800;">
                            <div style="font-size:14px; color:#888;">Qlib Ensemble</div>
                            <div style="font-size:32px; font-weight:bold; color:{qlib_color};">{qlib_action}</div>
                            <div style="color:#aaa; margin-top:8px;">信号: {qlib_mean:+.4f}</div>
                            <div style="color:#888;">{len(qlib_preds)}个模型</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_fusion:
                        fusion_action = signal_v5.action
                        fusion_color = "#00ff88" if fusion_action == "BUY" else "#ff4444" if fusion_action == "SELL" else "#ffaa00"
                        st.markdown(f"""
                        <div style="background:#1a1d24; border-radius:16px; padding:20px; text-align:center; border:3px solid {fusion_color};">
                            <div style="font-size:14px; color:#888;">🧬 融合信号 (v5)</div>
                            <div style="font-size:36px; font-weight:bold; color:{fusion_color};">{fusion_action}</div>
                            <div style="color:#aaa; margin-top:8px;">加权: {signal_v5.signal_weighted:+.4f}</div>
                            <div style="color:#888;">置信: {signal_v5.confidence:.0%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 融合详情
                    st.divider()
                    st.subheader("🔬 融合详情")

                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                    with detail_col1:
                        st.metric("共识比例", f"{signal_v5.consensus_ratio:.0%}",
                                 delta=f"{signal_v5.n_models_active}个模型/{len(signal_v5.model_predictions)}个预测")
                    with detail_col2:
                        st.metric("模型分歧度", f"{signal_v5.divergence:.4f}",
                                 delta="低分歧✅" if signal_v5.divergence < 0.15 else "高分歧⚠️")
                    with detail_col3:
                        st.metric("建议仓位", f"{signal_v5.suggested_size_pct:.0%}",
                                 delta=f"¥{pf.total_value * signal_v5.suggested_size_pct:,.0f}")
                    with detail_col4:
                        agreement = "✅ 一致" if (signal_v4.action == signal_v5.action) else "⚠️ 分歧"
                        st.metric("v4 vs v5", agreement)

                    # 各模型预测明细
                    st.subheader("📋 各模型 × 主题预测明细")
                    pred_data = []
                    for p in signal_v5.model_predictions:
                        if p.model_name == "lgbm":
                            continue
                        pred_data.append({
                            "模型": p.model_name.replace("qlib_", ""),
                            "主题": p.theme_name,
                            "预测": f"{p.prediction:+.4f}",
                            "方向": p.direction,
                            "权重": f"{p.weight:.3f}",
                            "ICIR": f"{p.oos_icir:.3f}",
                        })
                    if pred_data:
                        st.dataframe(pd.DataFrame(pred_data), hide_index=True, use_container_width=True)
                    else:
                        st.info("📭 Qlib 模型预测数据未生成 — 请先训练模型")

                    # 对比柱状图
                    if pred_data:
                        st.subheader("📊 各模型预测信号值")
                        model_names = list(set(d["模型"] for d in pred_data))
                        model_avgs = []
                        for m in model_names:
                            m_preds = [float(d["预测"]) for d in pred_data if d["模型"] == m]
                            model_avgs.append({"模型": m, "平均信号": np.mean(m_preds)})

                        fig = go.Figure()
                        colors = ["#00ff88" if v["平均信号"] > 0 else "#ff4444" for v in model_avgs]
                        fig.add_trace(go.Bar(
                            x=[v["模型"] for v in model_avgs],
                            y=[v["平均信号"] for v in model_avgs],
                            marker_color=colors,
                            text=[f"{v['平均信号']:+.4f}" for v in model_avgs],
                            textposition="outside",
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            template="plotly_dark", height=250,
                            margin=dict(l=0, r=0, t=10, b=0),
                            title="各模型平均信号 (跨7主题)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 分歧检测
                    if signal_v5.divergence > 0.2:
                        st.warning(f"⚠️ 模型间分歧较大 ({signal_v5.divergence:.3f}) — 市场不确定性高, 建议降低仓位")
                    elif signal_v5.consensus_ratio > 0.7:
                        st.success(f"✅ 模型共识度高 ({signal_v5.consensus_ratio:.0%}) — 信号可信任")
                    else:
                        st.info(f"ℹ️ 模型意见分散 — 建议参考 LightGBM 基线")

                except ImportError:
                    st.error("❌ 依赖缺失 — 请安装 ccxt, torch 等")
                except Exception as e:
                    st.error(f"❌ 预测失败: {e}")

        # 模型管理
        st.divider()
        with st.expander("🔄 Qlib 模型管理", expanded=False):
            model_files = list(Path(__file__).parent.glob("data/models/qlib_*.pth"))
            if model_files:
                st.caption(f"📦 {len(model_files)} 个 Qlib 模型文件")
                mf_data = []
                for mf in sorted(model_files):
                    mtime = datetime.fromtimestamp(mf.stat().st_mtime)
                    size_kb = mf.stat().st_size / 1024
                    parts = mf.stem.split("_")
                    mf_data.append({
                        "文件名": mf.name,
                        "模型": parts[1] if len(parts) > 1 else "?",
                        "主题": parts[2] if len(parts) > 2 else "?",
                        "大小": f"{size_kb:.0f}KB",
                        "修改时间": mtime.strftime("%m-%d %H:%M"),
                    })
                st.dataframe(pd.DataFrame(mf_data), hide_index=True, use_container_width=True)
            else:
                st.caption("💡 尚未训练 Qlib 模型 — 点击上方训练按钮")

            # 训练报告历史
            report_files = list(Path(__file__).parent.glob("data/qlib_reports/*.json"))
            if report_files:
                st.caption(f"📄 {len(report_files)} 份训练报告")
                for rf in sorted(report_files, reverse=True)[:5]:
                    st.caption(f"  • {rf.name}")

        # ═══════════════════════════════════════════
        # Phase 10: Rolling Training (在线学习)
        # ═══════════════════════════════════════════
        st.divider()
        st.subheader("🔄 滚动在线学习 — Phase 10 🆕")

        if not ROLLING_AVAILABLE:
            st.warning("⚠️ Rolling Trainer 暂不可用")
        else:
            st.markdown("""
            > 🔄 **滚动在线学习**: 定期自动重训模型, 适应市场变化 — 解决 vs Qlib 最大劣势
            >
            > 检查模型新鲜度 → 检测特征漂移 → 自动增量训练 → 模型版本管理
            """)

            # 初始化 rolling trainer
            if "rolling_trainer" not in st.session_state:
                st.session_state.rolling_trainer = RollingTrainer({
                    "mode": "hybrid",
                    "models_to_train": ["alstm", "transformer", "tabnet"],
                    "qlib_epochs": 50,
                })

            rt = st.session_state.rolling_trainer
            rt_status = rt.status()

            # 状态卡片
            rs_col1, rs_col2, rs_col3, rs_col4, rs_col5 = st.columns(5)
            with rs_col1:
                st.metric("模型快照", rt_status["n_snapshots"])
            with rs_col2:
                st.metric("覆盖模型", rt_status["n_models"], delta="种架构")
            with rs_col3:
                st.metric("覆盖主题", rt_status["n_themes"], delta="个")
            with rs_col4:
                max_age = rt_status["max_age_days"]
                age_color = "🟢" if max_age <= 7 else "🟡" if max_age <= 21 else "🔴"
                st.metric("最大年龄", f"{age_color} {max_age}天")
            with rs_col5:
                need_retrain = "是 🔧" if rt_status["should_retrain"] else "否 ✅"
                st.metric("需重训", need_retrain)

            # 模型新鲜度表
            if rt_status["staleness_summary"]:
                st.caption(f"📋 模型新鲜度 (最新训练: {rt_status['latest_training'][:19] if rt_status['latest_training'] != 'never' else '从未'})")
                freshness_data = []
                for s in rt_status["staleness_summary"]:
                    freshness_data.append({
                        "新鲜度": s["staleness"],
                        "模型": s["model"],
                        "主题": s["theme"],
                        "版本": f"v{s['version']}",
                        "年龄": f"{s['age_days']}天",
                        "ICIR": f"{s['icir']:.3f}",
                        "趋势": s["trend"],
                        "漂移": f"{s['drift']:.2f}",
                    })
                st.dataframe(pd.DataFrame(freshness_data), hide_index=True, use_container_width=True)

            # 滚动训练操作
            st.caption("🎮 滚动训练控制")
            roll_col1, roll_col2, roll_col3, roll_col4 = st.columns([2, 1, 1, 1])
            with roll_col1:
                roll_mode = st.selectbox(
                    "窗口模式",
                    options=["hybrid", "sliding", "expanding"],
                    index=0,
                    help="hybrid=2年滑动+拐点保留 | sliding=固定窗口 | expanding=全量历史",
                )
            with roll_col2:
                roll_models = st.multiselect(
                    "模型",
                    options=["alstm", "transformer", "tabnet"],
                    default=["alstm", "transformer"],
                )
            with roll_col3:
                roll_epochs = st.slider("Epochs", 20, 200, 50, 10, key="roll_epochs")
            with roll_col4:
                roll_force = st.checkbox("强制全量", value=False,
                                        help="忽略新鲜度检查, 强制重新训练所有模型")

            if st.button("🔄 执行滚动训练", type="primary", key="roll_train_btn"):
                with st.spinner(f"滚动训练中... 模式={roll_mode} | {len(roll_models)}模型×7主题"):
                    try:
                        import ccxt
                        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
                        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=800)
                        df_btc = pd.DataFrame(ohlcv, columns=["date","open","high","low","close","volume"])

                        # 更新配置
                        rt.config["mode"] = roll_mode
                        rt.config["models_to_train"] = roll_models
                        rt.config["qlib_epochs"] = roll_epochs

                        mode = "full" if roll_force else "update"
                        result = rt.run(df_btc, force=roll_force, mode=mode)

                        if result["action"] == "skip":
                            st.success(f"✅ {result['reason']}")
                        else:
                            st.success(f"✅ 滚动训练完成!")
                            if result.get("report"):
                                r = result["report"]["summary"]
                                imp_color = "green" if r["n_improved"] > r["n_degraded"] else "red"
                                st.metric("训练模型", r["n_models_trained"], delta=f"↑{r['n_improved']} →{r['n_stable']} ↓{r['n_degraded']}")
                                st.metric("平均ICIR", f"{r['mean_icir']:.4f}")
                                st.metric("中位ICIR", f"{r['median_icir']:.4f}")

                            # 显示 Top 表现
                            if result.get("report", {}).get("top_performers"):
                                with st.expander("🏆 Top 5 模型", expanded=True):
                                    top_data = []
                                    for tp in result["report"]["top_performers"]:
                                        top_data.append({
                                            "模型": tp["model_name"],
                                            "主题": tp["theme_name"],
                                            "ICIR": f"{tp['oos_icir']:.4f}",
                                            "Hit%": f"{tp['oos_hit_rate']:.1%}",
                                            "版本": f"v{tp['version']}",
                                            "趋势": tp["icir_trend"],
                                        })
                                    st.dataframe(pd.DataFrame(top_data), hide_index=True, use_container_width=True)

                            # 刷新状态
                            st.rerun()

                    except ImportError:
                        st.error("❌ ccxt 未安装")
                    except Exception as e:
                        st.error(f"❌ 滚动训练失败: {e}")

            # 自动检查按钮
            if st.button("🔍 快速检查 (不重训)", key="roll_check_btn"):
                rt_status = rt.status()
                if rt_status["should_retrain"]:
                    st.warning(f"⚠️ 建议重训: {rt_status['retrain_reason']}")
                else:
                    st.success(f"✅ {rt_status['retrain_reason']}")

            # Cron 提示
            with st.expander("⏰ 自动定时重训 (Cron)", expanded=False):
                st.code("""
        # 每周一早上8点自动检查并滚动训练
        0 8 * * 1 cd ~/yina-app/chase-quant-web && python3 rolling_trainer.py --update

        # 或使用 Python 直接调用
        0 8 * * 1 cd ~/yina-app/chase-quant-web && python3 -c "
        from rolling_trainer import auto_rolling_check
        import ccxt, pandas as pd
        ex = ccxt.binance()
        ohlcv = ex.fetch_ohlcv('BTC/USDT', '1d', limit=800)
        df = pd.DataFrame(ohlcv, columns=['date','open','high','low','close','volume'])
        print(auto_rolling_check(df))
        "
                """, language="bash")

        # 滚动历史报告
        roll_reports = list(Path(__file__).parent.glob("data/rolling/rolling_report_*.json"))
        if roll_reports:
            with st.expander(f"📄 滚动训练历史 ({len(roll_reports)} 份报告)", expanded=False):
                for rp in sorted(roll_reports, reverse=True)[:10]:
                    try:
                        with open(rp) as f:
                            rr = json.load(f)
                        st.caption(f"📅 {rp.stem.replace('rolling_report_', '')} | "
                                  f"触发: {rr.get('trigger_reason', '?')} | "
                                  f"模型: {rr.get('summary', {}).get('n_models_trained', '?')}个 | "
                                  f"均ICIR: {rr.get('summary', {}).get('mean_icir', '?')}")
                    except Exception:
                        st.caption(f"  • {rp.name}")

        # ═══════════════════════════════════════════
        # Phase 11: Asset Graph (资产关系图)
        # ═══════════════════════════════════════════
        st.divider()
        st.subheader("🔗 资产关系图 — Phase 11 🆕")

        try:
            from asset_graph import AssetGraphBuilder, CrossAssetGATPredictor
            GRAPH_UI_AVAILABLE = True
        except ImportError:
            GRAPH_UI_AVAILABLE = False

        if not GRAPH_UI_AVAILABLE:
            st.warning("⚠️ Asset Graph 引擎暂不可用")
        else:
            st.markdown("""
            > 🔗 **资产关系图**: 6维关系矩阵 (Pearson/Spearman/dCor/协整/Granger/波动率) → 真实邻接矩阵
            >
            > GATs 不再用随机图 — 而是真正的资产间信息传递网络
            """)

            # 初始化 builder
            if "graph_builder_ui" not in st.session_state:
                st.session_state.graph_builder_ui = AssetGraphBuilder()
            builder = st.session_state.graph_builder_ui

            graph_col1, graph_col2, graph_col3, graph_col4 = st.columns(4)

            # 加载已有图快照
            existing_snapshot = builder.load_snapshot("latest")

            with graph_col1:
                if existing_snapshot:
                    st.metric("图快照", f"{existing_snapshot.n_assets} 资产",
                             delta=f"v{existing_snapshot.version}")
                else:
                    st.metric("图快照", "未构建")
            with graph_col2:
                if existing_snapshot:
                    st.metric("图密度", f"{existing_snapshot.graph_density:.4f}")
                else:
                    st.metric("图密度", "N/A")
            with graph_col3:
                if existing_snapshot:
                    st.metric("平均度", f"{existing_snapshot.avg_degree:.1f}")
                else:
                    st.metric("平均度", "N/A")
            with graph_col4:
                if existing_snapshot:
                    st.metric("社区数", existing_snapshot.n_communities)
                else:
                    st.metric("社区数", "N/A")

            # 资产选择
            graph_symbols = st.multiselect(
                "选择资产",
                options=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"],
                default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
            )

            graph_action_col1, graph_action_col2, graph_action_col3 = st.columns([1, 1, 1])

            with graph_action_col1:
                if st.button("🔨 构建资产关系图", type="primary", key="build_graph_btn"):
                    if len(graph_symbols) < 2:
                        st.warning("至少选择2个资产")
                    else:
                        with st.spinner(f"计算 {len(graph_symbols)} 资产 6维关系矩阵..."):
                            try:
                                snapshot = builder.build(graph_symbols)
                                builder.save_snapshot(snapshot, "latest")
                                st.success(f"✅ 图构建完成! {snapshot.n_assets}资产 | "
                                          f"密度={snapshot.graph_density:.3f} | "
                                          f"{snapshot.n_communities}社区")
                                # 刷新
                                st.session_state.graph_snapshot_ui = snapshot
                            except Exception as e:
                                st.error(f"❌ 构建失败: {e}")

            with graph_action_col2:
                if st.button("🔮 多资产联合预测", key="graph_predict_btn"):
                    if len(graph_symbols) < 2:
                        st.warning("至少选择2个资产")
                    else:
                        with st.spinner("图增强多资产联合预测..."):
                            try:
                                g_predictor = CrossAssetGATPredictor()
                                if existing_snapshot:
                                    g_predictor.snapshot = existing_snapshot
                                preds = g_predictor.predict(graph_symbols)
                                st.session_state.graph_preds_ui = preds
                                st.success(f"✅ {len(preds)}/{len(graph_symbols)} 资产预测完成")
                            except Exception as e:
                                st.error(f"❌ 预测失败: {e}")

            with graph_action_col3:
                if st.button("📉 检测图漂移", key="check_drift_btn"):
                    with st.spinner("对比新旧图结构..."):
                        try:
                            drift = builder.detect_graph_drift_with_build(graph_symbols)
                            st.session_state.graph_drift_ui = drift
                            if drift.get("drifted"):
                                st.warning(f"🔴 图漂移: {drift['drift_score']:.4f}")
                            else:
                                st.success(f"✅ 图稳定: {drift['drift_score']:.4f}")
                        except Exception as e:
                            st.error(f"❌ 漂移检测失败: {e}")

            # 显示图预测结果
            if "graph_preds_ui" in st.session_state and st.session_state.graph_preds_ui:
                preds = st.session_state.graph_preds_ui
                st.subheader("📊 图增强多资产预测")
                pred_data = []
                for sym, pred in sorted(preds.items(), key=lambda x: abs(x[1] or 0), reverse=True):
                    direction = "📈 BUY" if pred and pred > 0 else "📉 SELL" if pred and pred < 0 else "➡️ HOLD"
                    pred_data.append({
                        "资产": sym,
                        "预测收益": f"{pred:+.6f}" if pred else "N/A",
                        "方向": direction,
                        "强度": abs(pred) if pred else 0,
                    })
                if pred_data:
                    st.dataframe(pd.DataFrame(pred_data), hide_index=True, use_container_width=True)

            # 显示图详情
            if existing_snapshot:
                with st.expander(f"🔍 图详情 — {existing_snapshot.n_assets}资产", expanded=False):
                    # Top 连边
                    st.caption("🔗 最强连边")
                    if existing_snapshot.top_edges:
                        edges_data = []
                        for e in existing_snapshot.top_edges[:10]:
                            edges_data.append({
                                "源资产": e["source"],
                                "目标资产": e["target"],
                                "权重": f"{e['weight']:.4f}",
                            })
                        st.dataframe(pd.DataFrame(edges_data), hide_index=True, use_container_width=True)

                    # 邻接矩阵热力图
                    st.caption("🔥 邻接矩阵热力图")
                    import plotly.express as px
                    adj = existing_snapshot.adj_matrix
                    symbols = existing_snapshot.symbols
                    fig_heat = px.imshow(
                        adj,
                        x=symbols,
                        y=symbols,
                        color_continuous_scale="RdBu_r",
                        zmin=0, zmax=adj.max() if adj.max() > 0 else 1,
                        title=f"资产关系邻接矩阵 (密度={existing_snapshot.graph_density:.4f})",
                    )
                    fig_heat.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # 社区结构
                    if existing_snapshot.n_communities > 1:
                        st.caption(f"🌐 社区结构 ({existing_snapshot.n_communities} 个社区)")
                        comm_data = []
                        for i, sym in enumerate(symbols):
                            label = existing_snapshot.community_labels[i] if i < len(existing_snapshot.community_labels) else 0
                            comm_data.append({
                                "资产": sym,
                                "社区": f"社区 {label}",
                            })
                        st.dataframe(pd.DataFrame(comm_data), hide_index=True, use_container_width=True)

            # Graph Drift 结果
            if "graph_drift_ui" in st.session_state:
                drift = st.session_state.graph_drift_ui
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    st.metric("漂移分数", f"{drift.get('drift_score', 0):.4f}")
                with dcol2:
                    st.metric("共同资产", drift.get("n_common_assets", 0))
                with dcol3:
                    drift_action = "🔴 显著漂移" if drift.get("drifted") else "✅ 结构稳定"
                    st.metric("判断", drift_action)

        # Phase 12: Alpha Mining (自动Alpha挖掘)
        st.divider()
        st.subheader("🔬 自动Alpha挖掘 — Phase 12 🆕")

        # Import check
        try:
            from alpha_miner import (AlphaExpressionParser, AlphaEvaluator,
                                      AlphaTemplateLibrary, AlphaStore, ALPHA_DIR)
            ALPHA_UI_AVAILABLE = True
        except ImportError:
            ALPHA_UI_AVAILABLE = False
            st.warning("⚠️ Alpha挖掘引擎未安装")

        if ALPHA_UI_AVAILABLE:
            st.markdown("""
            > 🔬 **自动Alpha挖掘**: 表达式驱动的因子自动发现 — 3大策略 (Grid/Genetic/Random) → 批量IC评估 → FDR校正 → 入库排名
            """)

            # ── Row 1: Expression Playground ──
            col_expr, col_tmpl = st.columns([3, 2])
            with col_expr:
                expr_input = st.text_input(
                    "🧪 Alpha表达式",
                    value="ts_delta(close, 5) / ts_std(close, 20)",
                    key="alpha_expr_input",
                    help="支持变量: open/high/low/close/volume/returns/log_returns/vwap\n"
                         "函数: ts_sum/ts_mean/ts_std/ts_delta/ts_roc/ts_zscore/ts_corr/ts_rank/ts_ema/..."
                )
            with col_tmpl:
                library = AlphaTemplateLibrary()
                cat_choice = st.selectbox("📚 模板分类", ["all"] + library.get_categories(), key="alpha_cat")
                if cat_choice != "all":
                    templates = library.get_by_category(cat_choice)
                else:
                    templates = library.get_all()
                tmpl_names = ["(手动输入)"] + [t.name for t in templates]
                tmpl_choice = st.selectbox("📝 选择模板", tmpl_names, key="alpha_tmpl")
                if tmpl_choice != "(手动输入)":
                    tmpl = next(t for t in templates if t.name == tmpl_choice)
                    st.caption(f"`{tmpl.expression}` — {tmpl.description}")

            # ── Row 2: Action Buttons ──
            c_eval, c_mine_grid, c_mine_gen, c_mine_rand = st.columns(4)
            with c_eval:
                eval_clicked = st.button("🔬 评估表达式", type="primary", key="eval_alpha_btn")
            with c_mine_grid:
                mine_grid_clicked = st.button("🔍 Grid Search", key="mine_grid_btn")
            with c_mine_gen:
                mine_gen_clicked = st.button("🧬 Genetic Evolve", key="mine_gen_btn")
            with c_mine_rand:
                mine_rand_clicked = st.button("🎲 Random Explore", key="mine_rand_btn")

            # ── Evaluate Expression ──
            if eval_clicked:
                expr = expr_input
                if tmpl_choice != "(手动输入)":
                    tmpl = next(t for t in templates if t.name == tmpl_choice)
                    expr = tmpl.expression
                # Fetch some data
                try:
                    import ccxt
                    exchange = ccxt.binance()
                    ohlcv = exchange.fetch_ohlcv("BTC/USDT", '1d', limit=500)
                    edf = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    edf['timestamp'] = pd.to_datetime(edf['timestamp'], unit='ms')
                    edf.set_index('timestamp', inplace=True)
                except Exception:
                    edf = None

                if edf is not None and len(edf) > 100:
                    evaluator = AlphaEvaluator()
                    result = evaluator.evaluate(expr, edf, name="playground", category="custom")
                    st.session_state.alpha_eval_result = result
                    st.success(f"✅ 评估完成: IC={result.rank_ic:+.4f} | ICIR={result.icir:+.3f} | Sharpe={result.sharpe:+.3f}")
                else:
                    st.error("❌ 无法获取数据")

            # Show evaluation result
            if "alpha_eval_result" in st.session_state:
                r = st.session_state.alpha_eval_result
                with st.expander("📊 表达式评估详情", expanded=True):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Rank IC", f"{r.rank_ic:+.4f}")
                    m2.metric("ICIR", f"{r.icir:+.3f}")
                    m3.metric("Sharpe", f"{r.sharpe:+.3f}")
                    m4.metric("Turnover", f"{r.turnover:.3f}")
                    m5.metric("Hit Rate", f"{r.hit_rate:.1%}")
                    st.caption(f"N={r.n_obs} | FDR p={r.fdr_p_value:.4f} | Corr={r.correlation_with_existing:.3f}")

                    # IC Decay
                    if r.ic_decay:
                        import plotly.graph_objects as go
                        decays = r.ic_decay
                        days = sorted(decays.keys())
                        vals = [decays[d] for d in days]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=days, y=vals, mode='lines+markers',
                                                 line=dict(color='#4488ff', width=2)))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(title="IC Decay", xaxis_title="Forward Days",
                                          yaxis_title="Rank IC", height=300,
                                          template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)

            # ── Mine Alphas ──
            mining_clicked = mine_grid_clicked or mine_gen_clicked or mine_rand_clicked
            if mining_clicked:
                try:
                    import ccxt
                    exchange = ccxt.binance()
                    ohlcv = exchange.fetch_ohlcv("BTC/USDT", '1d', limit=500)
                    mdf = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    mdf['timestamp'] = pd.to_datetime(mdf['timestamp'], unit='ms')
                    mdf.set_index('timestamp', inplace=True)
                except Exception:
                    mdf = None
                    st.error("❌ 无法获取数据")

                if mdf is not None and len(mdf) > 100:
                    evaluator = AlphaEvaluator()
                    miner = None  # AlphaMiner will be instantiated per strategy
                    from alpha_miner import AlphaMiner

                    with st.spinner("⛏️ 挖掘Alpha中..."):
                        if mine_grid_clicked:
                            miner = AlphaMiner(evaluator=evaluator, df=mdf)
                            results = miner.mine_grid(df=mdf, n_per_template=10, max_total=150, verbose=False)
                            gen_type = "Grid Search"
                        elif mine_gen_clicked:
                            miner = AlphaMiner(evaluator=evaluator, df=mdf)
                            results = miner.mine_genetic(df=mdf, population_size=100, generations=10, verbose=False)
                            gen_type = "Genetic Evolution"
                        else:
                            miner = AlphaMiner(evaluator=evaluator, df=mdf)
                            results = miner.mine_random(df=mdf, n=200, verbose=False)
                            gen_type = "Random Exploration"

                    if results:
                        # Save
                        store = AlphaStore()
                        store.save(results)
                        st.session_state.mined_alphas = results
                        st.session_state.mine_type = gen_type
                        st.success(f"✅ {gen_type} 完成! 发现 {len(results)} 个Alpha, "
                                   f"{sum(1 for r in results if r.passed)} 通过筛选")
                    else:
                        st.warning("⚠️ 未发现有效Alpha")

            # Show mined results
            if "mined_alphas" in st.session_state:
                results = st.session_state.mined_alphas
                gen_type = st.session_state.get("mine_type", "Mining")
                with st.expander(f"🏆 {gen_type} — Top {min(20, len(results))} Alphas", expanded=True):
                    # Build table
                    rows = []
                    for i, a in enumerate(results[:20]):
                        rows.append({
                            "#": i+1,
                            "✅": "✅" if a.passed else "❌",
                            "Expression": a.expression[:55],
                            "IC": f"{a.rank_ic:+.3f}",
                            "ICIR": f"{a.icir:+.2f}",
                            "Sharpe": f"{a.sharpe:+.2f}",
                            "Turnover": f"{a.turnover:.3f}",
                            "Category": a.category,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True, height=400)

                    # Best alpha detail
                    if results:
                        best = results[0]
                        st.caption(f"🏅 Best: `{best.expression}` — "
                                   f"IC={best.rank_ic:+.4f} ICIR={best.icir:+.3f} "
                                   f"IC Decay: {best.ic_decay}")

            # ── Active Alphas (from store) ──
            with st.expander("📦 已入库Alpha", expanded=False):
                store = AlphaStore()
                saved = store.list_saved()
                if saved:
                    st.caption(f"共 {len(saved)} 个存档, 最新: {saved[0]['saved_at'][:19]}")
                    alphas = store.get_top(20)
                    if alphas:
                        rows2 = []
                        for i, a in enumerate(alphas):
                            rows2.append({
                                "#": i+1,
                                "✅": "✅" if a.passed else "❌",
                                "Name": a.name[:30],
                                "IC": f"{a.rank_ic:+.3f}",
                                "ICIR": f"{a.icir:+.2f}",
                                "Sh": f"{a.sharpe:+.2f}",
                                "Cat": a.category,
                                "Gen": a.generation,
                            })
                        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
                    else:
                        st.info("📭 暂无入库Alpha, 运行挖掘后自动保存")
                else:
                    st.info("📭 暂无存档, 点击挖掘按钮开始")

        # ── 订单执行优化 (Phase 13) ──
        st.subheader("📊 订单执行优化 — Phase 13 🆕")
        exec_col1, exec_col2 = st.columns([1, 2])

        with exec_col1:
            st.caption("🔪 拆单策略")
            exec_strategy = st.radio(
                "执行策略",
                ["smart", "twap", "vwap", "adaptive", "iceberg"],
                format_func=lambda s: {
                    "smart": "🧠 Smart Auto (推荐)",
                    "twap": "⏱️ TWAP 时间加权",
                    "vwap": "📊 VWAP 成交量加权",
                    "adaptive": "🔄 Adaptive 自适应",
                    "iceberg": "🧊 Iceberg 冰山订单",
                }.get(s, s.upper()),
                index=0, horizontal=True,
                help="Smart: 自动根据订单大小/流动性选择最优策略"
            )

            exec_horizon = st.slider("执行窗口 (分钟)", 5, 240, 60, 5,
                                     help="总执行时间，越长滑点越小但延迟风险越大")
            exec_slices = st.slider("切片数 (0=自动)", 0, 50, 0, 1,
                                    help="0=根据 Almgren-Chriss 模型自动计算最优切片数")
            exec_urgency = st.slider("紧急度", 0.0, 1.0, 0.5, 0.1,
                                     help="0=被动(省成本) → 1=激进(抢成交)")
            exec_part_rate = st.slider("最大参与率 %", 1, 20, 5, 1,
                                       help="每切片占日均量上限") / 100

            st.divider()
            st.caption("📐 预交易估算器")
            est_symbol = st.selectbox("估算交易对",
                                      ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
                                      key="exec_est_symbol")
            est_qty = st.number_input("订单数量", min_value=0.001, value=0.1, step=0.01,
                                      format="%.3f", key="exec_est_qty",
                                      help="BTC/ETH/SOL 等单位")

            if st.button("💰 估算执行成本", use_container_width=True):
                try:
                    from execution import ExecutionEngine, ExecutionConfig
                    mock_data = {
                        "BTC/USDT": {"price": 87000, "avg_daily_volume": 35000, "volatility": 0.025, "spread": 0.0002},
                        "ETH/USDT": {"price": 3200, "avg_daily_volume": 500000, "volatility": 0.030, "spread": 0.0003},
                        "SOL/USDT": {"price": 180, "avg_daily_volume": 8000000, "volatility": 0.045, "spread": 0.0005},
                        "BNB/USDT": {"price": 620, "avg_daily_volume": 500000, "volatility": 0.028, "spread": 0.0004},
                        "XRP/USDT": {"price": 2.5, "avg_daily_volume": 200000000, "volatility": 0.035, "spread": 0.0008},
                    }
                    mdata = mock_data.get(est_symbol, {"price": 100, "avg_daily_volume": 1e6, "volatility": 0.03, "spread": 0.001})
                    engine = ExecutionEngine()
                    est = engine.pre_trade_estimate(est_qty, est_symbol, mdata)

                    st.success(f"✅ 推荐策略: **{est['optimal_strategy'].upper()}** | "
                               f"推荐切片: **{est['recommended_slices']}** 片")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("预估冲击", f"{est['impact_estimate']['total_bps']:.1f} bps")
                    c2.metric("预估手续费", f"${est['est_fee']:.2f}")
                    c3.metric("预估滑点", f"${est['est_slippage']:.2f}")
                    c4.metric("总成本", f"${est['est_total_cost']:.2f}")
                    if est.get("warning_flags"):
                        st.warning(est["warning_flags"])
                except Exception as e:
                    st.error(f"估算失败: {e}")

        with exec_col2:
            st.caption("📈 执行质量仪表板")

            try:
                from execution import ExecutionStore
                estore = ExecutionStore()
                estats = estore.get_stats()
                comp = estore.get_strategy_comparison()

                if estats.get("n_reports", 0) > 0:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("总执行次数", str(estats["n_reports"]))
                    mc2.metric("平均 IS", f"{estats['avg_shortfall_bps']:.1f} bps")
                    mc3.metric("平均成交率", f"{estats['avg_fill_rate']:.1%}")
                    mc4.metric("最佳执行", f"{estats['best_execution_bps']:.1f} bps")

                    # 策略对比柱状图
                    if comp:
                        import plotly.express as px
                        df_comp = pd.DataFrame(comp)
                        fig_comp = px.bar(df_comp, x="strategy", y="avg_shortfall_bps",
                                          title="各策略平均 Implementation Shortfall (越低越好)",
                                          color="strategy",
                                          color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_comp.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                                               showlegend=False, xaxis_title="", yaxis_title="IS (bps)")
                        st.plotly_chart(fig_comp, use_container_width=True)

                    # 最近执行记录
                    st.caption("📋 最近执行")
                    reports = estore.load(5)
                    if reports:
                        rec_rows = []
                        for r in reversed(reports):
                            is_val = r["implementation_shortfall_bps"]
                            icon = "🟢" if is_val < 5 else "🟡" if is_val < 20 else "🔴"
                            rec_rows.append({
                                "时间": r["timestamp"][:19],
                                "交易对": r["symbol"],
                                "方向": r["side"].upper(),
                                "策略": r["strategy_used"].upper(),
                                "IS": f"{icon} {is_val:+.1f}bps",
                                "成交": f"{r['fill_rate']:.0%}",
                                "切片": f"{r['n_slices_filled']}/{r['n_slices_total']}",
                            })
                        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("📭 暂无执行记录。运行一次模拟执行来填充数据:")
                    st.code("python3 execution.py --simulate BTC/USDT --qty 0.1 --strategy smart")

            except ImportError:
                st.info("🔌 执行引擎模块 (execution.py) 未加载, 请确保文件存在")

            # 模拟执行
            st.divider()
            st.caption("🎮 模拟执行")
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_symbol = st.selectbox("交易对",
                                          ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                                          key="exec_sim_symbol")
            with sim_col2:
                sim_side = st.selectbox("方向", ["buy", "sell"], key="exec_sim_side")
            with sim_col3:
                sim_qty = st.number_input("数量", min_value=0.001, value=0.05, step=0.01, format="%.3f", key="exec_sim_qty")

            if st.button("🔪 模拟拆单执行", use_container_width=True):
                try:
                    from execution import ExecutionEngine, ExecutionConfig
                    mock_data = {
                        "BTC/USDT": {"price": 87000, "avg_daily_volume": 35000, "volatility": 0.025, "spread": 0.0002},
                        "ETH/USDT": {"price": 3200, "avg_daily_volume": 500000, "volatility": 0.030, "spread": 0.0003},
                        "SOL/USDT": {"price": 180, "avg_daily_volume": 8000000, "volatility": 0.045, "spread": 0.0005},
                    }
                    mdata = mock_data.get(sim_symbol, {"price": 100, "avg_daily_volume": 1e6, "volatility": 0.03, "spread": 0.001})
                    cfg = ExecutionConfig(
                        strategy=exec_strategy,
                        horizon_minutes=exec_horizon,
                        n_slices=exec_slices,
                        urgency=exec_urgency,
                        participation_rate=exec_part_rate,
                    )
                    engine = ExecutionEngine(config=cfg)
                    report = engine.execute_paper(
                        symbol=sim_symbol, side=sim_side,
                        quantity=sim_qty, price=mdata["price"],
                        market_data=mdata
                    )

                    # 显示结果
                    is_color = "green" if report.implementation_shortfall_bps < 5 else "orange" if report.implementation_shortfall_bps < 20 else "red"
                    st.success(f"✅ 执行完成! 策略: **{report.strategy_used.upper()}** | "
                               f"IS: :{is_color}[{report.implementation_shortfall_bps:+.1f} bps] | "
                               f"Fill: {report.fill_rate:.1%}")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Arrival Price", f"${report.arrival_price:,.2f}")
                    r2.metric("Avg Exec Price", f"${report.avg_execution_price:,.2f}")
                    r3.metric("成交/总切片", f"{report.n_slices_filled}/{report.n_slices_total}")
                    r4.metric("执行时间", f"{report.duration_seconds:.0f}s")

                    # 成本分解
                    st.caption("💸 成本分解 (bps)")
                    cost_data = pd.DataFrame({
                        "成本项": ["点差", "市场冲击", "延迟", "VWAP滑点"],
                        "bps": [report.spread_cost_bps, report.market_impact_bps,
                                report.delay_cost_bps, report.vwap_slippage_bps],
                    })
                    fig_cost = px.bar(cost_data, x="成本项", y="bps", color="成本项",
                                      title="执行成本分解",
                                      color_discrete_sequence=["#4488ff", "#ffaa00", "#ff4444", "#888888"])
                    fig_cost.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10),
                                           showlegend=False)
                    st.plotly_chart(fig_cost, use_container_width=True)

                    # 切片明细
                    if report.slice_details:
                        with st.expander("📋 切片成交明细", expanded=False):
                            sl_df = pd.DataFrame(report.slice_details)
                            st.dataframe(sl_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"模拟执行失败: {e}")

        # 架构对比
        st.divider()
        with st.expander("📐 与 Qlib 原框架对比", expanded=False):
            st.markdown("""
            ### Chase Quant vs Microsoft Qlib — 架构对比

            | 维度 | Chase Quant (v5) | Microsoft Qlib |
            |------|:---:|:---:|
            | **时序模型** | ALSTM (自实现) | ALSTM + GRU + LSTM |
            | **注意力模型** | Transformer (自实现) | Transformer + HIST |
            | **特征选择** | TabNet (自实现) | TabNet + 表达式引擎 |
            | **图模型** | GATs (自实现) + **真实资产关系图 🆕** | GATs + RGCN + RSRL |
            | **集成方法** | DoubleEnsemble (自实现) | DoubleEnsemble + 更多变体 |
            | **自动因子挖掘** | ✅ **NEW! 表达式引擎+遗传算法** | ✅ Alpha Mining Pipeline |
            | **订单执行优化** | ✅ **NEW! 拆单算法 (TWAP/VWAP/Adaptive/Iceberg)** | ❌ 研究为主 |
            | **在线学习** | ✅ **滚动在线学习** | ✅ Rolling Training |
            | **资产关系图** | ✅ **NEW! 6维关系+图漂移检测** | ⚠️ 研究为主 |
            | **市场覆盖** | ✅ 4市场 (Crypto+A股+美股+港股) | ❌ A股为主 |
            | **实盘交易** | ✅ 自动交易 + 五层风控 | ❌ 研究为主 |
            | **可视化** | ✅ Streamlit 仪表板 | ⚠️ Jupyter/CLI |
            | **开源** | ✅ (本仓库) | ✅ MIT License |

            > 💡 **我们的定位**: 用 Qlib 的 AI 能力武装我们的实盘系统 — 取其精华, 为我所用。
            """)

# ═══════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════
st.divider()
st.caption("🐾 Chase的量化策略 v2.4 | 由 Yina 为 Chase哥 打造 | Qlib增强 + 在线学习 + 资产关系图 + Alpha挖掘 + 订单执行优化 · 虚拟盘 · 风险自负")

# 自动快照 (每60秒)
if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = time.time()

if time.time() - st.session_state.last_snapshot > 60:
    pf.take_snapshot()
    st.session_state.last_snapshot = time.time()
