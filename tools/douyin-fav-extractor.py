#!/usr/bin/env python3
"""抖音收藏+喜欢提取 v4 — 双tab抓取 + 自动登录 + LLM分析就绪"""
import os, sys, json, time
from datetime import datetime
from playwright.sync_api import sync_playwright

OUTPUT_FILE = '/tmp/douyin_fav_result.json'
COOKIE_FILE = '/tmp/douyin_cookies.json'
SCREENSHOT_DIR = '/tmp/'

def log(msg):
    print(msg, flush=True)

def is_logged_in(page):
    """检测是否已登录"""
    text = page.evaluate('() => document.body.innerText')
    if '扫码登录' in text or '验证码登录' in text:
        return False
    unlogged_markers = ['登录后即可', '登录即可', '请先登录']
    for m in unlogged_markers:
        if m in text:
            return False
    return True

def ensure_login(page, context):
    """确保登录状态，未登录则等待扫码"""
    if is_logged_in(page):
        log("✅ Cookie有效，已登录！")
        return True

    log("❌ 未登录，登录面板应该已经弹出")
    log("🔐 Chase哥 请在浏览器中用抖音App扫码登录！(等2分钟)")

    for i in range(60):
        page.wait_for_timeout(2000)
        if is_logged_in(page):
            log(f"✅ 登录成功！（{i*2}秒）")
            # 保存登录状态
            state = context.storage_state()
            with open(COOKIE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False)
            log("💾 Cookie已保存")
            return True
        if i % 15 == 0 and i > 0:
            log(f"  还在等扫码... ({i*2}秒)")

    log("⏱️ 扫码超时")
    return False

def click_tab(page, tab_name):
    """点击个人主页的某个tab（收藏/喜欢/作品）"""
    log(f"🔍 点击「{tab_name}」...")
    selectors = [
        f'text={tab_name}',
        f'span:has-text("{tab_name}")',
        f'div:has-text("{tab_name}"):not(:has(div))',
        f'[class*="tab"] span:has-text("{tab_name}")',
        f'li:has-text("{tab_name}")',
    ]

    for sel in selectors:
        try:
            els = page.query_selector_all(sel)
            for el in els:
                if el.is_visible():
                    text_content = el.inner_text() if hasattr(el, 'inner_text') else ''
                    log(f"  尝试点击: '{text_content[:50]}' (selector: {sel})")
                    try:
                        el.click(force=True)
                        return True
                    except:
                        try:
                            el.evaluate('el => el.click()')
                            return True
                        except:
                            continue
        except:
            continue

    # JS fallback
    log(f"  选择器未命中，尝试JS点击...")
    try:
        result = page.evaluate(f'''(tab) => {{
            const all = document.querySelectorAll('*');
            for (const el of all) {{
                if (el.innerText?.trim() === tab && el.tagName !== 'BODY' && el.offsetHeight > 0) {{
                    el.click();
                    return true;
                }}
            }}
            return false;
        }}''', tab_name)
        if result:
            return True
    except Exception as e:
        log(f"  JS点击失败: {e}")
    return False

def extract_videos(page, tab_name):
    """滚动并提取当前页面的视频链接"""
    log(f"📜 滚动加载「{tab_name}」...")
    for i in range(15):
        page.evaluate('window.scrollBy(0, 600)')
        page.wait_for_timeout(800)

    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v4_{tab_name}.png')

    log(f"📄 提取「{tab_name}」内容...")
    items = page.evaluate('''() => {
        const results = [];
        const seen = new Set();
        const videoLinks = document.querySelectorAll('a[href*="/video/"]');
        videoLinks.forEach(a => {
            const href = a.href;
            if (!seen.has(href) && !href.includes('source=')) {
                seen.add(href);
                let card = a.closest('[class*="card"], [class*="item"], [class*="video"], li, [class*="Card"], [class*="Item"], div');
                let title = '';

                if (card) {
                    // 策略1：找标题类元素（最长文本优先）
                    const titleCandidates = card.querySelectorAll('[class*="title"], [class*="desc"], [class*="content"], [class*="text"], [class*="info"], p, span');
                    let best = '';
                    for (const el of titleCandidates) {
                        const t = el.innerText?.trim() || '';
                        // 选最长的非纯数字文本（>10字）
                        if (t.length > 10 && t.length > best.length && !/^[\\d.,万]+$/.test(t)) {
                            best = t;
                        }
                    }
                    if (best) {
                        title = best;
                    } else {
                        // 策略2：取 card 内最长的一行文本
                        const lines = card.innerText?.trim().split('\\n').filter(l => l.length > 3);
                        title = lines.reduce((a, b) => a.length > b.length ? a : b, '') || '';
                    }
                }
                if (!title || title.length < 3) {
                    title = a.innerText?.trim() || '';
                }
                // 过滤纯数字标题（点赞数）
                if (/^[\\d.,万]+$/.test(title)) {
                    title = '(数字)';
                }
                if (title && title.length > 1) {
                    results.push({title: title.slice(0, 200), link: href});
                } else {
                    results.push({title: '(无标题)', link: href});
                }
            }
        });
        return results.slice(0, 100);
    }''')

    log(f"  ✅ 「{tab_name}」找到 {len(items)} 个视频")
    return items

def extract_tab(page, tab_name):
    """完整的tab提取流程：点击→等待→滚动→提取"""
    clicked = click_tab(page, tab_name)
    if clicked:
        page.wait_for_timeout(4000)
    else:
        log(f"⚠️ 「{tab_name}」tab点击可能失败，尝试继续...")
        page.wait_for_timeout(3000)
    return extract_videos(page, tab_name)


# ==================== MAIN ====================

log("🚀 v4 — 启动浏览器（收藏+喜欢双抓取）...")

with sync_playwright() as p:
    # 加载cookie
    storage_state = None
    if os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE) as f:
            try:
                storage_state = json.load(f)
                log("📦 加载保存的cookie")
            except:
                pass

    browser = p.chromium.launch(
        headless=False,
        args=[
            '--no-sandbox',
            '--disable-blink-features=AutomationControlled',
            '--lang=zh-CN',
        ]
    )

    context_kwargs = {
        'viewport': {'width': 1440, 'height': 900},
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
        'locale': 'zh-CN',
    }
    if storage_state:
        context_kwargs['storage_state'] = storage_state

    context = browser.new_context(**context_kwargs)
    page = context.new_page()

    # === 打开抖音 ===
    page.goto('https://www.douyin.com', timeout=30000, wait_until='domcontentloaded')
    page.wait_for_timeout(5000)
    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v4_step1.png')
    log(f"当前URL: {page.url}")

    # === 登录 ===
    if not ensure_login(page, context):
        browser.close()
        sys.exit(1)

    # === 进入个人主页 ===
    log("🏠 进入个人主页...")
    page.goto('https://www.douyin.com/user/self', timeout=15000, wait_until='domcontentloaded')
    page.wait_for_timeout(5000)

    if not is_logged_in(page):
        log("⚠️ 导航到个人主页后登录状态丢失！")
        page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v4_lost_login.png')
    else:
        log("✅ 个人主页登录状态正常")

    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v4_profile.png')

    # === 提取收藏 ===
    favorites = extract_tab(page, "收藏")
    fav_url = page.url

    # === 提取喜欢 ===
    # 需要回到个人主页重新点击"喜欢"tab
    page.goto('https://www.douyin.com/user/self', timeout=15000, wait_until='domcontentloaded')
    page.wait_for_timeout(4000)
    likes = extract_tab(page, "喜欢")
    likes_url = page.url

    # === 组装结果 ===
    page_text = page.evaluate('() => document.body.innerText')[:10000]

    result = {
        'timestamp': datetime.now().isoformat(),
        'title': page.title(),
        'favorites': {
            'count': len(favorites),
            'url': fav_url,
            'items': favorites,
        },
        'likes': {
            'count': len(likes),
            'url': likes_url,
            'items': likes,
        },
        'total_items': len(favorites) + len(likes),
        'page_text': page_text,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"\n✅ 保存到 {OUTPUT_FILE}")
    log(f"   📌 收藏: {len(favorites)} 个")
    log(f"   ❤️ 喜欢: {len(likes)} 个")
    log(f"   📦 总计: {len(favorites) + len(likes)} 个")

    # 打印前20条概览
    log("\n📌 收藏预览:")
    for i, item in enumerate(favorites[:10]):
        log(f"   [{i}] {item['title'][:80]}")

    log("\n❤️ 喜欢预览:")
    for i, item in enumerate(likes[:10]):
        log(f"   [{i}] {item['title'][:80]}")

    browser.close()
    log("✅ v4 完成！")
