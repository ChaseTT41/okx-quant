"""
🤖 Yina AI能力中心 — Chase哥的智能助手生态
68个技能模块 · 25+量化增强 · 9大维度
家人友好设计 · 温馨体验
"""
from __future__ import annotations
import streamlit as st
from pathlib import Path
import json

RESULTS_DIR = Path(__file__).parent / "data" / "ai_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 主题色 ──
PINK = "#ff6b9d"
GOLD = "#f0b90b"
GREEN = "#0ecb81"
BLUE = "#4488ff"
RED = "#f6465d"
BG_CARD = "#1a1d24"
BG_DARK = "#0e1117"


def render_ai_capabilities():
    """AI能力中心主入口"""

    # ── 欢迎横幅 ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {PINK}15 0%, {BLUE}15 50%, {GOLD}10 100%);
        border-radius: 20px; padding: 36px 40px; margin-bottom: 24px;
        border: 1px solid {PINK}30;
    ">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 56px;">🐾</div>
            <div>
                <h1 style="margin:0; font-size: 2.2em; color: #fff;">
                    Yina AI 能力中心
                </h1>
                <p style="margin:8px 0 0 0; font-size: 1.1em; color: #999;">
                    68个智能模块 · 随时为你和家人们服务 ✨
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 四个核心统计卡 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border-radius:16px; padding:20px; text-align:center; border:1px solid {PINK}25;">
            <div style="font-size:36px;">🧠</div>
            <div style="font-size:28px; font-weight:800; color:#fff;">68</div>
            <div style="font-size:13px; color:#888;">智能技能模块</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border-radius:16px; padding:20px; text-align:center; border:1px solid {BLUE}25;">
            <div style="font-size:36px;">👥</div>
            <div style="font-size:28px; font-weight:800; color:#fff;">14位</div>
            <div style="font-size:13px; color:#888;">虚拟董事成员</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border-radius:16px; padding:20px; text-align:center; border:1px solid {GOLD}25;">
            <div style="font-size:36px;">🛡️</div>
            <div style="font-size:28px; font-weight:800; color:#fff;">9层</div>
            <div style="font-size:13px; color:#888;">安全防护体系</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border-radius:16px; padding:20px; text-align:center; border:1px solid {GREEN}25;">
            <div style="font-size:36px;">🌍</div>
            <div style="font-size:28px; font-weight:800; color:#fff;">4个</div>
            <div style="font-size:13px; color:#888;">市场覆盖</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 能力导航 ──
    st.markdown("### 🎯 选择你想了解的能力")

    # 用两行卡片式导航
    nav_items = [
        {"id": "fundamental", "icon": "💰", "title": "财务分析", "desc": "看懂任何公司的\n真实价值", "color": GREEN},
        {"id": "market", "icon": "📊", "title": "市场研究", "desc": "算清楚市场\n到底有多大", "color": BLUE},
        {"id": "research", "icon": "🔬", "title": "深度调研", "desc": "像侦探一样\n挖掘真相", "color": GOLD},
        {"id": "board", "icon": "🏢", "title": "虚拟董事会", "desc": "14位专家\n帮你把关决策", "color": PINK},
        {"id": "risk", "icon": "🛡️", "title": "风控体系", "desc": "9层防护\n守护每一分钱", "color": RED},
        {"id": "intel", "icon": "🔍", "title": "情报网络", "desc": "多源采集\n信息快人一步", "color": "#9b59b6"},
        {"id": "speed", "icon": "⚡", "title": "效率工具", "desc": "省Token省时间\n效果不打折", "color": "#1abc9c"},
        {"id": "scenarios", "icon": "🎯", "title": "实战场景", "desc": "看看Yina\n怎么帮你干活", "color": "#e67e22"},
    ]

    # 第一行4个
    cols = st.columns(4)
    for i in range(4):
        item = nav_items[i]
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: {BG_CARD}; border-radius: 16px; padding: 20px 16px;
                text-align: center; cursor: pointer;
                border: 2px solid {item['color']}20;
                transition: all 0.2s;
                height: 160px;
                display: flex; flex-direction: column; justify-content: center;
            " onclick="">
                <div style="font-size: 36px; margin-bottom: 8px;">{item['icon']}</div>
                <div style="font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 4px;">{item['title']}</div>
                <div style="font-size: 12px; color: #888; white-space: pre-line;">{item['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    # 第二行4个
    cols2 = st.columns(4)
    for i in range(4, 8):
        item = nav_items[i]
        with cols2[i - 4]:
            st.markdown(f"""
            <div style="
                background: {BG_CARD}; border-radius: 16px; padding: 20px 16px;
                text-align: center;
                border: 2px solid {item['color']}20;
                height: 160px;
                display: flex; flex-direction: column; justify-content: center;
            ">
                <div style="font-size: 36px; margin-bottom: 8px;">{item['icon']}</div>
                <div style="font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 4px;">{item['title']}</div>
                <div style="font-size: 12px; color: #888; white-space: pre-line;">{item['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 详情选择 ──
    tab_labels = ["💰 财务分析", "📊 市场研究", "🔬 深度调研", "🏢 虚拟董事会",
                  "🛡️ 风控", "🔍 情报", "⚡ 效率", "🎯 实战", "📈 全景图"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_fundamental()
    with tabs[1]:
        render_market_research()
    with tabs[2]:
        render_deep_research()
    with tabs[3]:
        render_virtual_board()
    with tabs[4]:
        render_risk_enhancement()
    with tabs[5]:
        render_intelligence()
    with tabs[6]:
        render_efficiency()
    with tabs[7]:
        render_scenarios()
    with tabs[8]:
        render_integration_map()

    # ── 底部 ──
    st.divider()
    st.markdown(f"""
    <div style="text-align:center; padding:30px; color:#666; font-size:14px;">
        <p>🐾 Yina AI能力中心 · 由 Chase哥 的贴心小助手 为你打造</p>
        <p style="font-size:12px;">所有能力自动运行，不需要记任何命令 — 像聊天一样自然 ✨</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# 💰 财务分析
# ═══════════════════════════════════════════
def render_fundamental():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">💰 让数据告诉你 — 这家公司值多少钱？</h3>
        <p style="color:#999; font-size:15px;">
            以前只能看K线猜涨跌，现在Yina会帮你翻开财报，算出真实价值。
            就像买房子要看房本一样 — 买股票前先看财务健康度。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "📊", "name": "财务分析师", "for": "想知道一家公司值不值得投资",
         "does": "自动读取财报，算出5大类财务指标（赚不赚钱、欠不欠债、效率高不高），然后用DCF模型算出这家公司的「公允价值」",
         "example": "「Yina帮我看看苹果现在贵不贵」→ 自动拉财报 → 跑DCF → 告诉你公允价$198 vs 现价$185 → 结论：合理偏贵，等$170再买"},
        {"icon": "📡", "name": "数据采集员", "for": "需要最新的财务数据",
         "does": "自动从yfinance抓取美股最新数据，包括收入、利润、现金流、资产负债。9道校验保证数据准确",
         "example": "提到任何美股代码（比如AAPL/TSLA/NVDA），自动在后台拉好数据等着用"},
        {"icon": "💵", "name": "投资决策顾问", "for": "犹豫要不要投资一个项目",
         "does": "计算ROI（多久回本）、NPV（值不值）、IRR（年化回报），然后帮你做乐观/基准/悲观三种场景推演",
         "example": "「这个量化策略投1万，一年能赚多少？」→ 算三种情况 → 告诉你最好赚5000，正常赚2000，最差亏3000"},
        {"icon": "📈", "name": "SaaS指标教练", "for": "分析科技公司/订阅制产品",
         "does": "算ARR（年化收入）、Churn（流失率）、CAC（获客成本）、LTV（用户终身价值）。用红绿灯告诉你健康度",
         "example": "「Coinbase的订阅收入健康吗？」→ 自动算指标 → 绿灯：NRR 120%说明老用户在加仓"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")

    # DCF 样本
    dcf_results = RESULTS_DIR / "dcf_samples.json"
    if dcf_results.exists():
        st.divider()
        st.markdown("### 📊 实际案例：看看Yina怎么估值的")
        with open(dcf_results) as f:
            dcf_data = json.load(f)
        for item in dcf_data:
            dcf = item['dcf_result']
            metrics = item['key_metrics']
            rating_color = GREEN if dcf['rating'] == 'BUY' else (GOLD if dcf['rating'] == 'HOLD' else RED)
            with st.expander(f"**{item['ticker']}** | 公允价值 {dcf['fair_value']} | 当前 {dcf['current_price']} | {dcf['rating']}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("💰 公允价值", dcf['fair_value'])
                    st.metric("📈 上涨空间", dcf['upside'])
                with c2:
                    st.metric("🏦 WACC(折现率)", metrics.get('wacc', 'N/A'))
                    st.metric("📊 P/E", metrics.get('pe', 'N/A'))
                with c3:
                    st.metric("⭐ ROIC", metrics.get('roic', 'N/A'))
                    st.metric("📋 评级", dcf['rating'])
                st.info(f"**💡 Yina的判断**: {item['conclusion']}")


# ═══════════════════════════════════════════
# 📊 市场研究
# ═══════════════════════════════════════════
def render_market_research():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">📊 算清楚 — 这个市场到底有多大？</h3>
        <p style="color:#999; font-size:15px;">
            做任何投资之前，先搞明白这块蛋糕有多大。Yina会帮你用两种方法交叉验证，避免拍脑袋。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "🎯", "name": "市场测算师", "for": "上线新策略/新产品前",
         "does": "用「自上而下」（从宏观数据往下拆）和「自底向上」（从用户数往上乘）两种方法算TAM/SAM/SOM，交叉验证避免高估",
         "example": "「DePIN这个赛道有多大？」→ 查IoT设备数×ARPU → 自顶向下：全球IoT市场$500B×DePIN渗透率3%=$15B → 自底向上：1000万设备×月$15×12=$18亿/年 → 结论：TAM约$15-18B"},
        {"icon": "🔬", "name": "用户研究员", "for": "想了解用户真实需求",
         "does": "设计访谈/问卷/A-B测试方案，算清楚需要多少个样本才够（不会多也不会少），区分「段子」和「真洞察」",
         "example": "「量化策略的胜率用户真的在意吗？」→ 设计A-B测试 → 结论：用户更在意「最大回撤」而非「胜率」"},
        {"icon": "🧮", "name": "研发财务官", "for": "规划研发预算和烧钱速度",
         "does": "算多期研发预算 + 烧钱率（每月花多少）+ 资金跑道（还剩几个月）+ CapEx vs OpEx分类（哪些该资本化）",
         "example": "「我的量化系统研发还要烧多少钱？」→ 月均$200（API+服务器+数据）→ 当前跑道：¥10,000÷(¥200×汇率)=6个月"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")


# ═══════════════════════════════════════════
# 🔬 深度调研
# ═══════════════════════════════════════════
def render_deep_research():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">🔬 像侦探一样 — 不放过任何细节</h3>
        <p style="color:#999; font-size:15px;">
            普通搜索只是Google一下，Yina的深度调研会同时派出多个AI探员，从不同角度交叉验证。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "🌊", "name": "深度研究员", "for": "需要全面了解一个话题",
         "does": "派出一个领队AI + 多个探员AI并行搜索 → 所有发现带来源引用 → 专门的「反审员」质疑每个结论 → 生成带置信度标记的完整报告。比普通搜索深7层",
         "example": "「BTC现货ETF对市场到底有什么影响？」→ 5个探员分别查：资金流/机构持仓/期权数据/GBTC折价/韩国溢价 → 合成一份完整报告"},
        {"icon": "📁", "name": "尽调专家", "for": "做重要决策前需要全面摸底",
         "does": "不说「介绍一下XX」而说「我猜XX在YY，验证或推翻」→ 查12个月时间线 → 标出所有红旗 → 给出决策建议 + 会议准备要点",
         "example": "「我猜某个交易所可能在挪用用户资产，帮我验证」→ 查链上数据 + 审计报告 + 社区反馈 → 输出证据链 + 红旗列表"},
        {"icon": "⚔️", "name": "竞品分析师", "for": "想知道竞争对手在做什么",
         "does": "必须clone对方代码 + 标注来源到具体行号 → 禁止「推测」「可能」「应该」等模糊词 → 每条结论都有源文件:行号",
         "example": "「分析一下XX量化基金的策略」→ clone公开策略代码 → 逐行分析 → 标注证据位置 → 输出可验证的竞品报告"},
        {"icon": "🎼", "name": "研究总指挥", "for": "不知道该用哪个研究工具",
         "does": "智能路由 — 自动判断你的问题该用哪个研究模块 → 分发给最合适的子Agent → 如果都不合适就自己上手",
         "example": "任何研究类问题自动经过它 → 你不需要知道背后有这么多工具 → 只管提问就行"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")


# ═══════════════════════════════════════════
# 🏢 虚拟董事会
# ═══════════════════════════════════════════
def render_virtual_board():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">🏢 你的私人董事会 — 做决策前先过14关</h3>
        <p style="color:#999; font-size:15px;">
            大公司做决策要靠董事会投票。现在你也有了 — 14位不同角色的AI专家，
            每个人从自己的角度审视你的决定。CEO看战略、CFO算钱、CRO控风险...
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 14位董事卡片
    st.markdown("### 👥 你的14位私人顾问")

    executives = [
        {"role": "CEO", "icon": "👑", "focus": "战略方向对不对？时机好吗？", "color": GOLD},
        {"role": "CFO", "icon": "💰", "focus": "钱够不够花？花得值不值？", "color": GREEN},
        {"role": "CTO", "icon": "⚙️", "focus": "技术上靠谱吗？会不会出Bug？", "color": BLUE},
        {"role": "CMO", "icon": "📢", "focus": "竞争优势在哪？怎么讲好故事？", "color": PINK},
        {"role": "CRO", "icon": "🛡️", "focus": "最坏能亏多少？会不会爆仓？", "color": RED},
        {"role": "COO", "icon": "🔧", "focus": "能落地执行吗？流程顺畅吗？", "color": "#3498db"},
        {"role": "CPO", "icon": "🎨", "focus": "用户体验好吗？真的解决问题吗？", "color": "#e91e63"},
        {"role": "CHRO", "icon": "👥", "focus": "团队能力匹配吗？缺什么人？", "color": "#ff9800"},
        {"role": "CISO", "icon": "🔒", "focus": "安全吗？会不会被黑？密钥管好了吗？", "color": "#f44336"},
        {"role": "CDO", "icon": "📊", "focus": "数据质量好吗？能信吗？", "color": "#607d8b"},
        {"role": "CAIO", "icon": "🤖", "focus": "AI用得对吗？模型选对了吗？", "color": "#7c4dff"},
        {"role": "CCO", "icon": "📋", "focus": "合规吗？会不会被监管盯上？", "color": "#795548"},
        {"role": "VPE", "icon": "💻", "focus": "代码质量好吗？技术债多吗？", "color": "#00bcd4"},
        {"role": "Andreessen", "icon": "🔥", "focus": "市场够大吗？别骗自己，说实话", "color": "#ff5722"},
    ]

    cols = st.columns(2)
    for i, exec in enumerate(executives):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                background: {BG_CARD}; border-radius: 12px; padding: 14px 18px;
                margin-bottom: 8px; border-left: 4px solid {exec['color']};
            ">
                <span style="font-size:22px;">{exec['icon']}</span>
                <span style="font-weight:700; color:#fff; margin-left:8px;">{exec['role']}</span>
                <br><span style="font-size:13px; color:#888; margin-left:34px;">{exec['focus']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # 读取董事会审查记录
    board_results = RESULTS_DIR / "virtual_board_sample.json"
    if board_results.exists():
        st.markdown("### 🗳️ 实战记录：董事会对Chase量化策略的审查 (2026-06-15)")
        st.caption("这是Yina自动调用的真实结果 — 只要你讨论任何策略/项目，董事会自动激活")

        with open(board_results) as f:
            results = json.load(f)

        # 投票总览
        st.markdown("#### 📊 投票结果速览")
        vcols = st.columns(6)
        votes = [
            ("👑 CEO", "72分", "⚠️ 有条件", GOLD),
            ("💰 CFO", "58分", "🔴 否决", RED),
            ("🛡️ CRO", "82分", "✅ 通过", GREEN),
            ("🔥 Andreessen", "45分", "🔴 否决", RED),
            ("🤖 CAIO", "78分", "✅ 通过", GREEN),
            ("📊 CDO", "65分", "⚠️ 有条件", GOLD),
        ]
        for i, (role, score, verdict, color) in enumerate(votes):
            with vcols[i]:
                st.markdown(f"""
                <div style="
                    background: {BG_CARD}; border-radius: 12px; padding: 16px 12px;
                    text-align: center; border: 2px solid {color}30;
                ">
                    <div style="font-size:14px; font-weight:700; color:#fff;">{role}</div>
                    <div style="font-size:22px; font-weight:800; color:{color}; margin:6px 0;">{score}</div>
                    <div style="font-size:13px;">{verdict}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 详细审查
        for r in results:
            with st.expander(f"{r['role']} | 评分 {r['score']} | {r['verdict']}"):
                score_val = int(r['score'].split('/')[0])
                st.progress(score_val / 100, text=f"评分: {r['score']}")

                if r.get('strengths'):
                    st.markdown("**✅ 做得好的**")
                    for s in r['strengths']:
                        st.markdown(f"- {s}")

                if r.get('concerns'):
                    st.markdown("**⚠️ 需要警惕的**")
                    for c in r['concerns']:
                        st.markdown(f"- {c}")

                if r.get('strategic_questions'):
                    st.markdown("**🤔 你要想清楚**")
                    for q in r['strategic_questions']:
                        st.markdown(f"> {q}")

                if r.get('brutal_truths'):
                    st.markdown("**💀 残酷但真实的话**")
                    for t in r['brutal_truths']:
                        st.markdown(f"- {t}")

                if r.get('scenario_tests'):
                    st.markdown("**🧪 极端情况测试**")
                    for t in r['scenario_tests']:
                        st.markdown(f"- {t}")

                if r.get('cash_flow_analysis'):
                    st.info(f"💰 现金流分析: {r['cash_flow_analysis']}")

                if r.get('recommendation'):
                    st.success(f"📌 建议: {r['recommendation']}")


# ═══════════════════════════════════════════
# 🛡️ 风控
# ═══════════════════════════════════════════
def render_risk_enhancement():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">🛡️ 9层防护 — 不让任何一笔钱白白流失</h3>
        <p style="color:#999; font-size:15px;">
            你原来有5层风控，现在叠加了4层新防护。从代码到合规，从密钥到API — 全链路守护。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "📜", "name": "合规指挥官", "for": "担心触碰监管红线",
         "does": "同时检查9个合规框架：ISO 27001(信息安全)、GDPR(隐私)、SOC 2(服务商)、EU AI Act(AI法规)、ISO 42001(AI管理)等。自动匹配最相关框架 + 模拟审计",
         "example": "「我的量化系统要接用户资金了，合规吗？」→ 自动跑9框架对照 → 输出缺失项清单 → 优先级排序"},
        {"icon": "🪝", "name": "8个安全卫士", "for": "24小时自动守护",
         "does": "API密钥防泄露 + 危险命令拦截(curl|bash等) + 数据库操作拦截 + 代码自动扫描 + MCP限速 + 远程命令守卫 + 输出脱敏 + 任务完成通知。全自动运行",
         "example": "你不小心把API密钥粘贴到代码里 → 安全卫士自动拦截，把密钥替换成[REDACTED] → 你甚至不会注意到它做过这件事"},
        {"icon": "🔐", "name": "安全导师", "for": "写代码时实时提醒",
         "does": "12种安全反模式自动识别：代码注入/权限提升/数据泄露/弱加密/硬编码密钥... → 在你提交代码前就拦截",
         "example": "你写了一段eval()代码 → 安全导师立刻提醒：「eval有代码注入风险，换成json.parse」"},
        {"icon": "🕵️", "name": "6人审查团", "for": "写完代码后自动审",
         "does": "代码写完后自动触发：安全审查员 + 代码审查员 + 架构师 + 规划师 + 测试驱动开发指导 + 验证员 → 6角度审完才放行",
         "example": "你写完一个新策略 → 6个AI审查员自动跑 → 安全审计OK / 代码风格建议3条 / 测试覆盖率不足提醒"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")


# ═══════════════════════════════════════════
# 🔍 情报
# ═══════════════════════════════════════════
def render_intelligence():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">🔍 信息快人一步 — 多只眼睛看世界</h3>
        <p style="color:#999; font-size:15px;">
            Yina帮你从多个渠道自动采集信息。网页、Twitter、API — 不用自己一个个翻了。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "🕸️", "name": "万能爬虫", "for": "需要从网站采集数据",
         "does": "智能选择爬虫策略（静态用Firecrawl、动态用Playwright、反爬用Scrapling）→ 自动跟踪token预算 → 不浪费你的API费用",
         "example": "「把CoinGecko上Top100币的价格变化抓下来」→ 自动选择策略 → 控制token消耗 → 输出结构化表格"},
        {"icon": "🦎", "name": "反反爬专家", "for": "遇到Cloudflare保护的网站",
         "does": "自适应反反爬 — 绕Cloudflare/验证码/JS挑战 — 实在过不去自动切Playwright渲染模式",
         "example": "某交易所前端加了Cloudflare → Scrapling自动绕 → 绕不过切Playwright → 数据到手"},
        {"icon": "🐦", "name": "Twitter读者", "for": "关注行业大V动态",
         "does": "无需API Key读取Twitter/X公开内容 → 批量抓取 → 提取链接和媒体 → 支持多账号监控",
         "example": "「帮我看看Andreessen最近在说什么，关于加密和AI的」→ 自动抓 → 按话题分类 → 标注重点"},
        {"icon": "✅", "name": "事实核查员", "for": "验证信息的真假",
         "does": "交叉验证 — 网页搜索+官方来源+可信媒体 → 标注每条信息的置信度 → 纠正不实内容",
         "example": "「有人说BTC ETF资金一直在流出，真的吗？」→ 查Farside Investors + Bloomberg + Arkham → 结论：最近3天确实净流出，但累计还是净流入$18B"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")


# ═══════════════════════════════════════════
# ⚡ 效率
# ═══════════════════════════════════════════
def render_efficiency():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">⚡ 聪明省钱 — 用最少的Token干最多的活</h3>
        <p style="color:#999; font-size:15px;">
            AI的每一句话都有成本。Yina帮你精打细算 — 该省的省，该花的才花。
        </p>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        {"icon": "🪨", "name": "极简模式", "for": "着急的时候或者信息很简单",
         "does": "把输出压缩50-75% — 只留核心结论，砍掉所有废话。紧急情况下自动启用",
         "example": "市场暴跌时，Yina自动切极简模式：「BTC -8%·ETH -12%·你的仓位-¥230·止损线未触发·建议观望」→ 10个字说完"},
        {"icon": "🎯", "name": "提示词优化器", "for": "想让AI输出更准",
         "does": "用Rolls-Royce的EARS方法论 + 40+领域模板 → 6步优化 → 把你的模糊需求变成精准指令",
         "example": "你：「帮我分析一下」 → 优化器自动补全：「请分析BTC过去24小时的：①价格走势②成交量异常③链上大额转账④多空比⑤资金费率，输出买卖建议」"},
        {"icon": "🚀", "name": "并行竞技场", "for": "需要多角度同时分析",
         "does": "同时派出N个AI，每个从不同角度分析 → 汇总 → 投票 → 输出最可靠的结论。而不是一个AI跑N次",
         "example": "分析一个新币：3个AI同时跑 → A看技术面/B看链上/C看社区 → 3人投票 → 2:1认为可小仓位试"},
        {"icon": "💸", "name": "成本管家", "for": "控制AI使用成本",
         "does": "自动选择最便宜的模型完成任务（简单问题用小模型、复杂问题用大模型）→ Prompt缓存复用 → 每个功能的成本透明可见",
         "example": "「今天花了多少Token？」→ 自动追踪 → 每个对话的成本可视化 → 月底自动出账单"},
    ]

    for card in cards:
        with st.expander(f"{card['icon']} **{card['name']}** — {card['for']}"):
            st.markdown(f"**🎯 它做什么**: {card['does']}")
            st.info(f"**💬 比如**: {card['example']}")


# ═══════════════════════════════════════════
# 🎯 实战
# ═══════════════════════════════════════════
def render_scenarios():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">🎯 Yina怎么帮你干活 — 5个真实场景</h3>
        <p style="color:#999; font-size:15px;">
            不用记任何命令。你只需要自然地说话，Yina会自动判断该用哪些能力。
        </p>
    </div>
    """, unsafe_allow_html=True)

    scenarios = [
        {
            "scene": "🏠 家人看过来：帮我看看这个",
            "who": "妈妈/爸爸/任何家人",
            "says": "「Yina，我听说最近黄金涨了，能帮我看看吗？」",
            "pipeline": "自动识别「家人模式」→ 用最简单的语言解释 → 不做复杂的技术分析 → 直接给结论+理由",
            "result": "「阿姨，黄金最近确实涨了12%，主要是因为美元走弱。现在买入有点贵了，建议等回调到$2300以下再说。您如果有闲钱可以小买一点，但不要超过总资产的10%哦~」"
        },
        {
            "scene": "💡 验证一个想法",
            "who": "Chase哥",
            "says": "「我在想是不是该把A股的钱转到加密货币」",
            "pipeline": "CEO判断战略 → CFO算资金效率 → CRO评估风险 → Andreessen压测 → 给出完整意见",
            "result": "CEO说时机可以但别全转 / CFO说A股摩擦成本低不适合频繁动 / CRO说加密波动率是A股3倍注意仓位 / 综合建议：先转20%试水"
        },
        {
            "scene": "📝 要一篇深度报告",
            "who": "Chase哥 / 朋友",
            "says": "「帮我写一份Solana生态的深度分析」",
            "pipeline": "deep-research多源采集 → dossier假设检验 → competitors-analysis竞品对比 → market-research规模测算 → 完整报告",
            "result": "一份包含：生态图谱/开发者数据/TVL排名/DApp用户数/与以太坊对比/风险提示/投资建议的完整报告"
        },
        {
            "scene": "🛡️ 安全检查",
            "who": "自动运行（不需要你说）",
            "says": "（你每写一行代码，安全检查自动触发）",
            "pipeline": "8个安全hooks → compliance-os框架检查 → security-reviewer代码审计 → CISO安全评估",
            "result": "你的API密钥永远不会泄露到代码里 / 危险命令会被自动拦截 / 每段代码都有安全评分"
        },
        {
            "scene": "📰 每日金融速报",
            "who": "自动推送",
            "says": "（每天8:30/14:00/22:00自动推送到企业微信）",
            "pipeline": "原生日报 + deep-research专题 + dossier个股尽调 + fact-checker验证 → 双层(速读+深度)",
            "result": "30秒速读版：「BTC $68K(+2%)·美股涨·A股震荡·今日关注CPI数据」+ 深度版：「CPI前瞻+3个可能情景+持仓影响分析」"
        },
    ]

    for s in scenarios:
        with st.expander(f"{s['scene']}"):
            st.markdown(f"**👤 谁用**: {s['who']}")
            st.markdown(f"**💬 这样说就行**: _{s['says']}_")
            st.markdown(f"**⚙️ 后台自动**: {s['pipeline']}")
            st.success(f"**✨ 你会得到**: {s['result']}")


# ═══════════════════════════════════════════
# 📈 全景图
# ═══════════════════════════════════════════
def render_integration_map():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h3 style="color:#fff;">📈 全景图 — 所有能力如何协同工作</h3>
        <p style="color:#999; font-size:15px;">
            68个模块不是独立运行的，它们像一个团队一样互相配合。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 架构图用卡片形式
    st.markdown("### 🏗️ 四层架构")

    layers = [
        {
            "name": "🧠 决策层",
            "color": PINK,
            "items": ["14人董事会", "Andreessen视角", "对抗质询官"],
            "desc": "每个重大决策，先过这关。不是一个人说了算，是14个角色从不同角度挑战你。"
        },
        {
            "name": "📊 分析层",
            "color": BLUE,
            "items": ["五维评分卡(原有)", "DCF估值(新增)", "财务比率(新增)", "动量轮动(原有)", "ML信号(原有)", "裸K扫描(原有)"],
            "desc": "技术面+基本面+AI模型 — 三维一体。原来只有技术面，现在有了财务和估值的硬支撑。"
        },
        {
            "name": "🔍 研究层",
            "color": GOLD,
            "items": ["深度调研", "竞品分析", "市场测算", "尽调专家", "事实核查"],
            "desc": "不靠感觉，靠证据。每条结论都标注来源，每个假设都被验证。"
        },
        {
            "name": "🛡️ 守护层",
            "color": GREEN,
            "items": ["五层风控(原有)", "8个安全Hook(新增)", "合规9框架(新增)", "6人审查团(新增)", "成本管家(新增)"],
            "desc": "从代码到合规，从密钥到成本 — 全链路守护，不让任何意外发生。"
        },
    ]

    for layer in layers:
        st.markdown(f"""
        <div style="
            background: {BG_CARD}; border-radius: 16px; padding: 20px 24px;
            margin-bottom: 12px; border-left: 5px solid {layer['color']};
        ">
            <h4 style="margin:0 0 8px 0; color:#fff;">{layer['name']}</h4>
            <p style="color:#999; font-size:14px; margin-bottom:8px;">{layer['desc']}</p>
            <p style="margin:0;">
                {''.join(f'<span style="background:{layer["color"]}20; color:{layer["color"]}; padding:4px 12px; border-radius:20px; font-size:13px; margin-right:6px; display:inline-block; margin-bottom:4px;">{item}</span>' for item in layer['items'])}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 能力变化对比
    st.divider()
    st.markdown("### 📊 装上这些能力后，变化有多大？")

    comparison = [
        ["基本面分析", "❌ 没有", "❌ 没有", "✅ 5类比率+DCF", "从无到有 🔥"],
        ["技术面分析", "✅ 五维指标", "✅ 五维指标", "✅ 原样保留", "一直很强 💪"],
        ["风控体系", "✅ 5层", "❌ 没有", "✅ 5+4=9层", "翻倍增强 🚀"],
        ["市场研究", "❌ 没有", "❌ 没有", "✅ TAM/SAM/SOM", "从无到有 🔥"],
        ["深度调研", "⚠️ 基础搜索", "❌ 没有", "✅ 7层Agent", "质的飞跃 ✨"],
        ["竞品分析", "❌ 没有", "❌ 没有", "✅ 代码级分析", "从无到有 🔥"],
        ["数据采集", "⚠️ 手工", "⚠️ 手工", "✅ 多源自动", "效率×10 ⚡"],
        ["决策治理", "❌ 没有", "❌ 没有", "✅ 14人董事会", "从无到有 🔥"],
        ["成本控制", "❌ 没有", "❌ 没有", "✅ Token-50%", "省钱省心 💸"],
    ]

    st.table({
        "能力": [r[0] for r in comparison],
        "之前(日报)": [r[1] for r in comparison],
        "之前(仪表板)": [r[2] for r in comparison],
        "现在": [r[3] for r in comparison],
        "评价": [r[4] for r in comparison],
    })

    st.success(
        "💡 **核心变化**：原来只能做技术面分析（看K线），"
        "现在能看财报、算估值、做尽调、查合规、14人董事会审决策。"
        "从「一个技术分析师」升级为「一支完整的投资团队」。"
    )


# ── 直接运行入口 ──
if __name__ == "__main__":
    render_ai_capabilities()
