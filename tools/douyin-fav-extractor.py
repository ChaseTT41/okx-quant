#!/usr/bin/env python3
"""抖音收藏提取 v3 — 修复登录检测 + 自动保存状态"""
import os, sys, json, time
from datetime import datetime
from playwright.sync_api import sync_playwright

OUTPUT_FILE = '/tmp/douyin_fav_result.json'
COOKIE_FILE = '/tmp/douyin_cookies.json'
SCREENSHOT_DIR = '/tmp/'

def log(msg):
    print(msg, flush=True)

def is_logged_in(page):
    """检测是否已登录：登录弹窗是否消失"""
    text = page.evaluate('() => document.body.innerText')
    # 这些是未登录的明确标志
    if '扫码登录' in text or '验证码登录' in text:
        return False
    # 检查登录弹窗特有的文字
    unlogged_markers = ['登录后即可', '登录即可', '请先登录']
    for m in unlogged_markers:
        if m in text:
            return False
    # 再检查有没有用户信息（昵称、头像等）
    # 已登录页面一般不会有这些大块登录引导
    return True

def login_dialog_visible(page):
    """检查登录弹窗是否可见"""
    try:
        # 抖音登录面板常见选择器
        panels = [
            '#login-full-panel',
            '[id*="login"][class*="panel"]',
            '[class*="login-panel"]',
            '[class*="LoginModal"]',
            '[class*="login-modal"]',
            '.login-container',
            '[class*="LoginContainer"]',
        ]
        for sel in panels:
            el = page.query_selector(sel)
            if el and el.is_visible():
                return True
        # 通过文字判断
        if '扫码登录' in page.evaluate('() => document.body.innerText[:2000]'):
            return True
    except:
        pass
    return False

log("🚀 v3 — 启动浏览器...")

with sync_playwright() as p:
    # 尝试加载之前的cookie
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
    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_step1.png')
    log(f"当前URL: {page.url}")

    # 检查登录状态
    if not is_logged_in(page):
        log("❌ 未登录，登录面板应该已经弹出")
        log("🔐 Chase哥 请在浏览器中用抖音App扫码登录！(等2分钟)")

        # 不点击登录按钮，因为弹窗可能已经在了
        # 等用户扫码
        logged = False
        for i in range(60):
            page.wait_for_timeout(2000)
            if is_logged_in(page):
                log(f"✅ 登录成功！（{i*2}秒）")
                logged = True
                break
            if i % 15 == 0 and i > 0:
                log(f"  还在等扫码... ({i*2}秒)")

        if not logged:
            log("⏱️ 超时，退出")
            browser.close()
            sys.exit(1)

        # 保存登录状态
        state = context.storage_state()
        with open(COOKIE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
        log("💾 Cookie已保存")

    else:
        log("✅ Cookie有效，已登录！")

    page.wait_for_timeout(2000)
    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_loggedin.png')

    # === 进入个人主页 ===
    log("🏠 进入个人主页...")
    page.goto('https://www.douyin.com/user/self', timeout=15000, wait_until='domcontentloaded')
    page.wait_for_timeout(5000)

    # 确认还是登录状态
    if not is_logged_in(page):
        log("⚠️ 导航到个人主页后登录状态丢失！")
        page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_lost_login.png')
    else:
        log("✅ 个人主页登录状态正常")

    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_profile.png')

    # === 点击收藏 ===
    log("🔍 点击收藏...")
    clicked = False
    fav_selectors = [
        'text=收藏',
        'span:has-text("收藏")',
        'div:has-text("收藏"):not(:has(div))',
        '[class*="tab"] span:has-text("收藏")',
        'li:has-text("收藏")',
    ]

    for sel in fav_selectors:
        try:
            els = page.query_selector_all(sel)
            for el in els:
                if el.is_visible():
                    text_content = el.inner_text() if hasattr(el, 'inner_text') else ''
                    log(f"  尝试点击: '{text_content[:50]}' (selector: {sel})")
                    try:
                        el.click(force=True)
                        clicked = True
                        break
                    except:
                        try:
                            el.evaluate('el => el.click()')
                            clicked = True
                            break
                        except:
                            continue
        except Exception as e:
            continue
        if clicked:
            break

    if clicked:
        log("✅ 收藏标签已点击")
        page.wait_for_timeout(5000)
    else:
        log("⚠️ 没找到收藏标签，尝试用JS点击...")
        try:
            page.evaluate('''() => {
                const els = document.querySelectorAll('*');
                for (const el of els) {
                    if (el.innerText?.trim() === '收藏' && el.tagName !== 'BODY') {
                        el.click();
                        return true;
                    }
                }
                return false;
            }''')
            page.wait_for_timeout(5000)
        except Exception as e:
            log(f"JS点击也失败: {e}")

    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_favorites.png')

    # === 滚动加载 ===
    log("📜 滚动...")
    for i in range(15):
        page.evaluate('window.scrollBy(0, 600)')
        page.wait_for_timeout(1000)

    page.screenshot(path=f'{SCREENSHOT_DIR}douyin_v3_scrolled.png')

    # === 提取 ===
    log("📄 提取内容...")
    current_url = page.url

    # 提取视频链接和标题
    items = page.evaluate('''() => {
        const results = [];
        const seen = new Set();

        // 找所有视频链接
        const videoLinks = document.querySelectorAll('a[href*="/video/"]');
        videoLinks.forEach(a => {
            const href = a.href;
            if (!seen.has(href)) {
                seen.add(href);
                // 找父级卡片
                let card = a.closest('[class*="card"], [class*="item"], [class*="video"], li, [class*="Card"], [class*="Item"]');
                let title = '';
                if (card) {
                    title = card.innerText?.trim()?.split('\\n')[0] || '';
                }
                if (!title) {
                    title = a.innerText?.trim() || '';
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

    log(f"找到 {len(items)} 个视频链接")

    # 全页文字
    full_text = page.evaluate('() => document.body.innerText')
    full_text = full_text[:10000]

    result = {
        'timestamp': datetime.now().isoformat(),
        'url': current_url,
        'title': page.title(),
        'item_count': len(items),
        'items': items,
        'page_text': full_text,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log(f"\n✅ 保存到 {OUTPUT_FILE}")
    log(f"   收藏数: {len(items)}")
    log(f"   URL: {current_url}")

    for i, item in enumerate(items[:20]):
        log(f"   [{i}] {item['title'][:100]}")
        log(f"       {item['link']}")

    browser.close()
    log("✅ 完成！")
