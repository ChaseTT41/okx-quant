"""
Chase量化策略 — 安全中间件
================================
公网部署安全加固: 速率限制 · 安全头 · 输入验证 · 反爬虫 · IP黑白名单

设计原则: 不修改任何原有代码, 作为 FastAPI middleware 透明叠加
"""

from __future__ import annotations
import os
import re
import time
import hashlib
import logging
import ipaddress
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.requests import Request


# ═══════════════════════════════════════════
# 配置 (通过环境变量覆盖)
# ═══════════════════════════════════════════

# 生产模式开关
PRODUCTION_MODE = os.environ.get("PRODUCTION", "false").lower() in ("1", "true", "yes")

# API 速率限制
RATE_LIMIT_GLOBAL_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))    # 每 IP 每分钟最多请求数
RATE_LIMIT_MUTATION_PER_MIN = int(os.environ.get("RATE_LIMIT_MUTATION_PER_MIN", "10"))  # POST/PUT/DELETE 更严格
RATE_LIMIT_WINDOW_SEC = 60          # 滑动窗口秒数
RATE_LIMIT_BURST = 5               # 允许的小突发

# 白名单 (可选)
IP_WHITELIST = os.environ.get("IP_WHITELIST", "").split(",") if os.environ.get("IP_WHITELIST") else []

# 黑名单
IP_BLACKLIST = os.environ.get("IP_BLACKLIST", "").split(",") if os.environ.get("IP_BLACKLIST") else []

# 可信局域网 (这些网段的 IP 自动跳过管理员鉴权)
TRUSTED_NETWORKS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),       # 本机回环
    ipaddress.ip_network("10.0.0.0/8"),        # 私有 A 类
    ipaddress.ip_network("172.16.0.0/12"),     # 私有 B 类
    ipaddress.ip_network("192.168.0.0/16"),    # 私有 C 类
    ipaddress.ip_network("::1/128"),           # IPv6 回环
    ipaddress.ip_network("fc00::/7"),          # IPv6 本地唯一
    ipaddress.ip_network("fe80::/10"),         # IPv6 链路本地
]

# 从环境变量加载额外可信 IP/CIDR
_extra_trusted = os.environ.get("TRUSTED_IPS", "")
if _extra_trusted.strip():
    for item in _extra_trusted.split(","):
        item = item.strip()
        if item:
            try:
                TRUSTED_NETWORKS.append(ipaddress.ip_network(item))
            except ValueError:
                import sys
                print(f"⚠️ [security] 无效的 TRUSTED_IPS 条目: {item}", file=sys.stderr)


def is_trusted_ip(client_ip: str) -> bool:
    """检查客户端 IP 是否在可信网段内"""
    if not client_ip or client_ip == "unknown":
        return False
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_NETWORKS)


def get_client_ip(request: Request) -> str:
    """从请求中提取真实客户端 IP"""
    # X-Forwarded-For (取第一个, 最原始客户端)
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # X-Real-IP
    real_ip = request.headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()
    # 直连 IP
    client = request.client
    if client:
        return client.host
    return "unknown"


# 安全日志
SECURITY_LOG = logging.getLogger("chase-quant.security")
SECURITY_LOG.setLevel(logging.WARNING)

# ═══════════════════════════════════════════
# 速率限制器 (滑动窗口 + Token Bucket)
# ═══════════════════════════════════════════


class RateLimiter:
    """
    基于滑动窗口的速率限制器

    算法: Sliding Window Counter
    - 记录每个请求的时间戳
    - 在窗口中过期
    - 支持全局限制和 mutation 操作更严格的限制
    """

    def __init__(self):
        # {ip: [timestamp, ...]}
        self._global: Dict[str, List[float]] = defaultdict(list)
        self._mutation: Dict[str, List[float]] = defaultdict(list)
        # IP 违规计数 (逐步递增惩罚)
        self._violations: Dict[str, int] = defaultdict(int)

    def _prune(self, records: List[float], window: float) -> List[float]:
        """剪除过期记录"""
        cutoff = time.monotonic() - window
        # 原地剪除 (保留最近的)
        while records and records[0] < cutoff:
            records.pop(0)
        return records

    def check(self, ip: str, is_mutation: bool = False) -> tuple[bool, int, str]:
        """
        检查是否允许请求

        Returns:
            (allowed, remaining, reason)
        """
        now = time.monotonic()

        # 黑名单检查
        if IP_BLACKLIST and ip in (b.strip() for b in IP_BLACKLIST if b.strip()):
            return False, 0, "IP 已被列入黑名单"

        # 白名单检查 (白名单中的 IP 跳过速率限制)
        if IP_WHITELIST and ip in (w.strip() for w in IP_WHITELIST if w.strip()):
            return True, -1, ""  # -1 表示不限速

        # 全局限制
        global_window = self._global[ip]
        self._prune(global_window, RATE_LIMIT_WINDOW_SEC)

        # 违规惩罚: 累积违规次数越多, 限制越严格
        penalty = self._violations.get(ip, 0)
        effective_limit = max(10, RATE_LIMIT_GLOBAL_PER_MIN - penalty * 10)

        if len(global_window) >= effective_limit + RATE_LIMIT_BURST:
            self._violations[ip] = min(10, penalty + 1)
            remaining = 0
            return False, remaining, f"全局速率限制: {effective_limit}/分钟"

        # Mutation 端点更严格的限制
        if is_mutation:
            mut_window = self._mutation[ip]
            self._prune(mut_window, RATE_LIMIT_WINDOW_SEC)
            if len(mut_window) >= RATE_LIMIT_MUTATION_PER_MIN:
                self._violations[ip] = min(10, penalty + 1)
                remaining = 0
                return False, remaining, f"写操作速率限制: {RATE_LIMIT_MUTATION_PER_MIN}/分钟"

            mut_window.append(now)

        # 记录请求
        global_window.append(now)

        # 逐渐降低违规计数
        if penalty > 0 and len(global_window) < effective_limit // 2:
            self._violations[ip] = max(0, penalty - 1)

        remaining = effective_limit - len(global_window)
        return True, max(0, remaining), ""


# 全局限速器实例
_rate_limiter = RateLimiter()


# ═══════════════════════════════════════════
# 反爬虫检测
# ═══════════════════════════════════════════

# 已知爬虫/恶意 UA 特征
BOT_UA_PATTERNS = [
    r"(?i).*bot.*",           # 通用 bot
    r"(?i).*crawler.*",       # 爬虫
    r"(?i).*spider.*",        # 蜘蛛
    r"(?i).*scraper.*",       # 刮刀
    r"(?i).*curl.*",          # curl (可配置)
    r"(?i).*wget.*",          # wget
    r"(?i).*python-requests.*",
    r"(?i).*go-http-client.*",
    r"(?i).*java.*",
    r"(?i).*libwww.*",
    r"(?i).*apache-httpclient.*",
    r"(?i).*okhttp.*",
    r"(?i).*node-fetch.*",
    r"(?i).*axios.*",
    r"(?i).*zgrab.*",         # 扫描器
    r"(?i).*nmap.*",          # 扫描器
    r"(?i).*masscan.*",       # 扫描器
]
_bot_patterns_compiled = [re.compile(p) for p in BOT_UA_PATTERNS]

# 例外: 允许正常浏览器和已知合法工具
WHITELIST_UA_PATTERNS = [
    r"(?i).*mozilla.*",
    r"(?i).*chrome.*",
    r"(?i).*safari.*",
    r"(?i).*firefox.*",
    r"(?i).*edge.*",
]
_whitelist_ua_compiled = [re.compile(p) for p in WHITELIST_UA_PATTERNS]


def is_suspicious_bot(user_agent: str) -> bool:
    """
    检测可疑爬虫

    逻辑: 如果 UA 匹配 bot 特征但不在白名单浏览器中 → 标记为可疑
    """
    if not user_agent:
        return True  # 空 UA 高度可疑

    # 白名单优先
    for p in _whitelist_ua_compiled:
        if p.match(user_agent):
            return False

    # 检测机器人
    for p in _bot_patterns_compiled:
        if p.match(user_agent):
            return True

    return False


# ═══════════════════════════════════════════
# 安全头中间件 (纯 ASGI)
# ═══════════════════════════════════════════


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """注入安全响应头"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # 只在 HTTP 响应上设置 (不设置 WebSocket)
        if hasattr(response, "headers"):
            headers = response.headers
            # 内容安全策略 (CSP) — 所有资源本地化，不再依赖外部CDN
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            )
            # 禁止被嵌入 iframe (防点击劫持)
            headers.setdefault("X-Frame-Options", "DENY")
            # XSS 保护
            headers.setdefault("X-Content-Type-Options", "nosniff")
            # 引用策略
            headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
            # 权限策略
            headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()")
            # 禁止 MIME 类型嗅探
            headers.setdefault("X-DNS-Prefetch-Control", "off")
            # 跨域隔离 (如需)
            headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

            # 移除服务端信息泄露
            if "server" in headers:
                del headers["server"]
            if "x-powered-by" in headers:
                del headers["x-powered-by"]

        return response


# ═══════════════════════════════════════════
# 速率限制 + 反爬虫中间件
# ═══════════════════════════════════════════


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    综合请求防护中间件
    - 速率限制
    - UA 检测
    - 请求体大小限制
    - IP 统计
    """

    # 不需要防护的路径 (健康检查 / 监控 / 鉴权)
    EXEMPT_PATHS = {"/", "/api/chat", "/api/health", "/health", "/ping", "/api/auth/verify", "/api/auth/status"}

    # Mutation 方法
    MUTATION_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

    # 最大请求体大小 (1MB)
    MAX_BODY_SIZE = 1_048_576

    async def dispatch(self, request: Request, call_next):
        # ── 获取真实 IP ──
        client_ip = self._get_client_ip(request)

        # ── 白名单 IP 放行 ──
        if IP_WHITELIST and client_ip in (w.strip() for w in IP_WHITELIST if w.strip()):
            return await call_next(request)

        # ── 黑名单直接拒绝 ──
        if IP_BLACKLIST and client_ip in (b.strip() for b in IP_BLACKLIST if b.strip()):
            SECURITY_LOG.warning(f"BLOCKED blacklist IP: {client_ip} → {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "访问被拒绝", "code": "BLACKLISTED"},
            )

        # ── UA 检测 (豁免健康检查等路径) ──
        path = request.url.path
        is_exempt = path in self.EXEMPT_PATHS or path.startswith("/api/health")

        user_agent = request.headers.get("user-agent", "")
        if PRODUCTION_MODE and not is_exempt and is_suspicious_bot(user_agent):
            SECURITY_LOG.warning(f"BOT detected: {client_ip} UA='{user_agent[:80]}' → {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "自动化工具访问被拒绝", "code": "BOT_DETECTED"},
            )

        # ── 速率限制 ──
        path = request.url.path
        method = request.method
        is_mutation = method in self.MUTATION_METHODS

        if path not in self.EXEMPT_PATHS:
            allowed, remaining, reason = _rate_limiter.check(client_ip, is_mutation)
            if not allowed:
                SECURITY_LOG.warning(f"RATE LIMIT: {client_ip} {method} {path} → {reason}")
                retry_after = RATE_LIMIT_WINDOW_SEC
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"请求过于频繁，请 {retry_after} 秒后重试",
                        "code": "RATE_LIMITED",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # 添加速率限制信息到响应头
            response = await call_next(request)
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Limit"] = str(
                    RATE_LIMIT_MUTATION_PER_MIN if is_mutation else RATE_LIMIT_GLOBAL_PER_MIN
                )
            return response

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """获取真实客户端 IP (支持反向代理)"""
        # X-Forwarded-For (取第一个, 最原始客户端)
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # X-Real-IP
        real_ip = request.headers.get("x-real-ip", "")
        if real_ip:
            return real_ip.strip()

        # 直连 IP
        client = request.client
        if client:
            return client.host

        return "unknown"


# ═══════════════════════════════════════════
# 输入验证工具
# ═══════════════════════════════════════════

# 安全的正则模式
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_./\-]{1,128}$")
SAFE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,12}(/[A-Z0-9]{1,12})?$")  # e.g. BTC/USDT
SAFE_TIMEFRAME_PATTERN = re.compile(r"^(1m|5m|15m|30m|1h|4h|1d|1w)$")


def validate_strategy_id(strategy_id: str) -> bool:
    """验证策略 ID 格式安全"""
    if not strategy_id:
        return False
    return bool(SAFE_ID_PATTERN.match(strategy_id))


def validate_symbol(symbol: str) -> bool:
    """验证交易对符号格式安全"""
    if not symbol:
        return False
    return bool(SAFE_SYMBOL_PATTERN.match(symbol.replace("-", "/")))


def sanitize_input(value: str, max_length: int = 256) -> str:
    """
    清理输入: 去除控制字符 + 截断
    不修改原值, 只移除危险字符
    """
    if not isinstance(value, str):
        return ""
    # 移除 NULL 字节和控制字符 (保留换行和制表符)
    cleaned = value.replace("\x00", "")[:max_length]
    # 移除 ANSI 转义序列
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", cleaned)
    return cleaned.strip()


# ═══════════════════════════════════════════
# 请求日志 (可选)
# ═══════════════════════════════════════════


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """精简请求日志 — 仅记录异常和写操作"""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        # 只记录 mutation 操作和异常状态
        if request.method in ("POST", "PUT", "DELETE") or response.status_code >= 400:
            client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
                request.client.host if request.client else "?"
            )
            SECURITY_LOG.info(
                f"{client_ip} {request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f}ms)"
            )

        return response


# ═══════════════════════════════════════════
# 管理员鉴权中间件
# ═══════════════════════════════════════════

# 管理员密钥 (从环境变量读取, 未设置则鉴权不生效)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

# 需要管理员权限的 API 路径前缀
MUTATION_PATHS = [
    "/api/strategies/",     # 策略 CRUD (PUT/POST/DELETE)
    "/api/positions/close", # 手动平仓
    "/api/trade",           # 手动交易
]

# 管理员验证端点 (不需要鉴权)
AUTH_EXEMPT_PATHS = {"/api/auth/verify", "/api/auth/status"}


def verify_admin_token(token: str) -> bool:
    """
    验证管理员令牌 (恒定时间比对防时序攻击)

    如果 ADMIN_TOKEN 未配置, 始终返回 False (鉴权未启用)
    """
    if not ADMIN_TOKEN:
        return False
    if not token:
        return False
    # 使用 sha256 hash 做恒定时间比对
    expected = hashlib.sha256(ADMIN_TOKEN.encode()).digest()
    actual = hashlib.sha256(token.strip().encode()).digest()
    return hashlib.sha256(ADMIN_TOKEN.encode()).digest() == hashlib.sha256(token.strip().encode()).digest()


def is_admin_enabled() -> bool:
    """检查管理员鉴权是否已配置"""
    return bool(ADMIN_TOKEN)


def _requires_admin(path: str, method: str) -> bool:
    """判断请求是否需要管理员权限"""
    # GET/HEAD/OPTIONS 不需要
    if method in ("GET", "HEAD", "OPTIONS"):
        return False
    # 检查是否匹配 mutation 路径
    for prefix in MUTATION_PATHS:
        if path.startswith(prefix):
            return True
    # 也检查策略 toggle 端点
    if "/api/strategies/" in path and "/toggle" in path:
        return True
    return False


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """
    管理员鉴权中间件

    行为:
    - 如果 ADMIN_TOKEN 未配置 → 所有请求放行 (向后兼容)
    - 如果 ADMIN_TOKEN 已配置:
      - GET/HEAD/OPTIONS → 放行 (只读)
      - POST/PUT/DELETE on MUTATION_PATHS → 需要 X-Admin-Token header
      - /api/auth/* → 放行 (验证端点自身)
    """

    async def dispatch(self, request: Request, call_next):
        # 鉴权未启用 → 放行
        if not ADMIN_TOKEN:
            return await call_next(request)

        path = request.url.path
        method = request.method

        # 验证端点豁免
        if path in AUTH_EXEMPT_PATHS or path.startswith("/api/auth/"):
            return await call_next(request)

        # 健康检查豁免
        if path == "/api/health" or path.startswith("/api/health"):
            return await call_next(request)

        # ── 可信局域网 IP 自动放行 (无需管理员令牌) ──
        client_ip = get_client_ip(request)
        if is_trusted_ip(client_ip):
            return await call_next(request)

        # 只读请求放行
        if not _requires_admin(path, method):
            return await call_next(request)

        # ── 写操作需要管理员令牌 ──
        admin_token = request.headers.get("X-Admin-Token", "")

        if not verify_admin_token(admin_token):
            SECURITY_LOG.warning(
                f"AUTH DENIED: {client_ip} {method} {path} — 缺少有效管理员令牌"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "需要管理员权限，请提供有效的管理员令牌",
                    "code": "ADMIN_REQUIRED",
                    "hint": "在请求头中添加 X-Admin-Token",
                },
            )

        return await call_next(request)


# ═══════════════════════════════════════════
# 便捷安装函数
# ═══════════════════════════════════════════


def install_security_middleware(app, production: bool = None):
    """
    向 FastAPI app 安装所有安全中间件

    使用方式 (在 api_server.py 中):
        from security import install_security_middleware
        app = FastAPI(...)
        install_security_middleware(app, production=True)

    中间件执行顺序 (LIFO):
        1. SecurityHeadersMiddleware (最外层)
        2. RateLimitMiddleware
        3. RequestLoggingMiddleware
        4. CORS (在 app 中已配置)
    """
    if production is not None:
        global PRODUCTION_MODE
        PRODUCTION_MODE = production

    # 从内到外添加 (Starlette LIFO)
    # 最内层: 管理员鉴权 (最先执行)
    app.add_middleware(AdminAuthMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # 生产模式下禁用 API 文档
    if PRODUCTION_MODE:
        app.docs_url = None
        app.redoc_url = None
        app.openapi_url = None
        # 移除已注册的 /docs, /redoc, /openapi.json 路由，避免 500 错误
        app.router.routes = [
            r for r in app.router.routes
            if getattr(r, 'path', '') not in ('/docs', '/redoc', '/openapi.json')
        ]

    admin_status = "🔐 已配置" if ADMIN_TOKEN else "⚠️ 未配置 (公网可写)"
    print(f"🔒 安全中间件已安装 {'(生产模式)' if PRODUCTION_MODE else '(开发模式)'}")
    print(f"   🔑 管理员鉴权: {admin_status}")
    if not PRODUCTION_MODE:
        print(f"   ⚠️ API 文档仍然可访问: /docs")
        print(f"   ⚠️ 速率限制较宽松: {RATE_LIMIT_GLOBAL_PER_MIN} req/min")

    return app
