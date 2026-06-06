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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 总览", "📈 信号", "💼 持仓", "📋 交易记录", "🛡️ 风控", "🧬 ML信号"
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
        > 🧬 **西蒙斯风格**: 279个特征 → FDR筛选28个 → 6个独立子信号 → 组合决策
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
                    st.subheader("🎯 6个子信号")
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

# ═══════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════
st.divider()
st.caption("🐾 Chase的量化策略 v1.0 | 由 Yina 为 Chase哥 打造 | 虚拟盘 · 风险自负")

# 自动快照 (每60秒)
if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = time.time()

if time.time() - st.session_state.last_snapshot > 60:
    pf.take_snapshot()
    st.session_state.last_snapshot = time.time()
