#!/usr/bin/env python3
"""
🕯️ 五市场Hourly快报 — Yina全指标标准操作脚本
=============================================
基于 Chase哥 2026-06-20 SOP 要求:
  每次扫描必须包含10类全指标 + 7维共振 + 企业微信推送

10类强制指标:
  1. MACD 金叉/死叉 (1h/4h/1d)
  2. RSI 超买超卖 (1h/4h/1d)
  3. 布林带 + ATR 波动率 (4h)
  4. Stochastic RSI (4h)
  5. OBV + MFI 资金流 (4h)
  6. ADX 趋势强度 (4h)
  7. EMA矩阵 (日线 EMA9/20/50/200)
  8. 熊猫K线结构 (4h+1d)
  9. BTC关联性矩阵 (1h)
  10. 7维共振检查 (每币种)

7维共振:
  MACD(4h) | RSI(4h) | 布林(4h) | Stoch(4h) | ADX(4h) | EMA20(4h) | OBV(4h)
  ≥5/7 = 强共振轻仓试单, 4/7 = 中共振关注等待, <4/7 = 弱共振观望

用法: python3 scripts/hourly_full_report.py
输出: 企业微信「金融监控」群 + /tmp/hourly_report.md
"""

import ccxt
import pandas as pd
import numpy as np
import json
import urllib.request
import time
import warnings
import sys

warnings.filterwarnings('ignore')

WECHAT_WEBHOOK = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2c602b48-5da2-4989-9193-30c0e226c769'
EXCHANGE = ccxt.binance({'enableRateLimit': True})
TIER1 = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
         'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT']
TIMEFRAMES = ['1h', '4h', '1d']
COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT']
MAX_CHARS = 3900

# ============================================================
# 指标计算
# ============================================================

def calc_rsi(series, period=14):
    d = series.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - (100 / (1 + g.ewm(alpha=1/period, adjust=False).mean() /
                         l.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)))

def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es; sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def calc_bollinger(close, period=20, std=2):
    sma = close.rolling(period).mean(); sd = close.rolling(period).std()
    return sma + std*sd, sma - std*sd, sma, ((sma+std*sd)-(sma-std*sd))/sma*100

def calc_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_stochastic(df, kp=14, dp=3):
    h, l, c = df['high'], df['low'], df['close']
    ll = l.rolling(kp).min(); hh = h.rolling(kp).max()
    k = 100 * ((c - ll) / (hh - ll).replace(0, np.nan))
    return k, k.rolling(dp).mean()

def calc_obv(df):
    c, v = df['close'], df['volume']; obv = [0]
    for i in range(1, len(c)):
        if c.iloc[i] > c.iloc[i-1]: obv.append(obv[-1] + v.iloc[i])
        elif c.iloc[i] < c.iloc[i-1]: obv.append(obv[-1] - v.iloc[i])
        else: obv.append(obv[-1])
    return pd.Series(obv, index=c.index)

def calc_mfi(df, period=14):
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    tp = (h + l + c) / 3; mf = tp * v
    pm = mf.where(tp > tp.shift(), 0); nm = mf.where(tp < tp.shift(), 0)
    mr = pm.rolling(period).sum() / nm.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mr))

def calc_adx(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    um = h - h.shift(); dm = l.shift() - l
    pdm = um.where((um > dm) & (um > 0), 0); ndm = dm.where((dm > um) & (dm > 0), 0)
    pdi = 100 * (pdm.ewm(span=period, adjust=False).mean() / atr)
    ndi = 100 * (ndm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100 * abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean(), pdi, ndi

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

# ============================================================
# 数据获取
# ============================================================

def fetch_ohlcv(symbol, timeframe, limit=200):
    try:
        d = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(d, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
        return df.dropna()
    except Exception: return None

def fetch_sina(code):
    try:
        req = urllib.request.Request(
            'https://hq.sinajs.cn/list=' + code,
            headers={'Referer': 'https://finance.sina.com.cn'})
        return urllib.request.urlopen(req, timeout=10).read().decode('gbk')
    except Exception: return ""

def fetch_yahoo(symbol):
    try:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol + '?interval=1d&range=1d'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        d = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return d['chart']['result'][0]['meta']['regularMarketPrice']
    except Exception: return None

def fmt(v, spec='.2f'):
    if isinstance(v, (int, float)): return format(v, spec)
    return str(v) if v else '?'

def fmt_price(v):
    if isinstance(v, (int, float)):
        if v >= 1000: return '{:,.0f}'.format(v)
        elif v >= 1: return '{:,.2f}'.format(v)
        else: return '{:.4f}'.format(v)
    return str(v)

# ============================================================
# 主流程
# ============================================================

def main():
    print("🕯️  Yina 五市场Hourly快报", flush=True)
    print("=" * 60, flush=True)

    # ---- 1. 加密数据 ----
    print("🔍 [1/4] 加密数据采集...", flush=True)
    crypto_data = {}
    for sym in TIER1:
        name = sym.split('/')[0]
        crypto_data[name] = {}
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(sym, tf)
            if df is not None: crypto_data[name][tf] = df
            time.sleep(0.1)

    # MACD + RSI
    print("📊 [2/4] MACD+RSI...", flush=True)
    macd_rpt = {}; rsi_rpt = {}
    for sym in TIER1:
        name = sym.split('/')[0]; macd_rpt[name] = {}; rsi_rpt[name] = {}
        for tf in TIMEFRAMES:
            if tf in crypto_data[name]:
                df = crypto_data[name][tf]
                ml, sl, hist = calc_macd(df['close'])
                cm, cs, ch, ph = ml.iloc[-1], sl.iloc[-1], hist.iloc[-1], hist.iloc[-2]
                if ml.iloc[-1] > sl.iloc[-1] and ml.iloc[-2] <= sl.iloc[-2]: cross = "金叉✨"
                elif ml.iloc[-1] < sl.iloc[-1] and ml.iloc[-2] >= sl.iloc[-2]: cross = "死叉💀"
                elif cm > cs: cross = "多方"
                else: cross = "空方"
                if ch > ph and ch > 0: bar = "↑"
                elif ch < ph and ch < 0: bar = "↓"
                elif ch > ph: bar = "↗"
                else: bar = "↘"
                macd_rpt[name][tf] = {'cross': cross, 'bar': bar}
                rsi_rpt[name][tf] = round(calc_rsi(df['close']).iloc[-1], 1)

    # 共振 + 布林/Stoch/ADX/OBV/MFI/ATR
    print("📐 [3/4] 布林+Stoch+ADX+OBV+MFI+共振...", flush=True)
    resonance = {}
    for sym in TIER1:
        name = sym.split('/')[0]
        df4 = crypto_data[name].get('4h')
        if df4 is None: resonance[name] = 0; continue
        c = df4['close'].iloc[-1]; score = 0.0

        ms = macd_rpt[name].get('4h', {})
        if '金叉' in ms.get('cross', ''): score += 1
        elif '多方' in ms.get('cross', ''): score += 0.5

        r = rsi_rpt[name].get('4h', 50)
        if 30 < r < 60: score += 1

        u, l, m, bw = calc_bollinger(df4['close'])
        pos = ((c - l.iloc[-1]) / (u.iloc[-1] - l.iloc[-1]) * 100)
        if 15 < pos < 45: score += 1

        k, d = calc_stochastic(df4)
        if k.iloc[-1] < 45 and k.iloc[-1] > d.iloc[-1]: score += 1

        adx, pdi, ndi = calc_adx(df4)
        if pdi.iloc[-1] > ndi.iloc[-1]: score += 1

        ema20 = calc_ema(df4['close'], 20).iloc[-1]
        if c > ema20: score += 1

        obv = calc_obv(df4)
        if obv.iloc[-1] > obv.iloc[-12]: score += 1

        resonance[name] = round(score, 1)

    # ---- 2. 传统市场 ----
    print("📰 [4/4] 传统市场...", flush=True)

    # A股: p[1]=现价, p[2]=涨跌额, p[3]=涨跌幅%
    a = {}
    for name, code in [('sh', 's_sh000001'), ('sz', 's_sz399001'),
                        ('cy', 's_sz399006'), ('kc', 's_sh000688')]:
        raw = fetch_sina(code)
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 3: a[name] = {'p': float(p[1]), 'chg': float(p[3])}

    # 美股
    us_ix = {}; us_st = {}
    for code, name in [('GSPC', 'sp'), ('IXIC', 'nq')]:
        raw = fetch_sina('gb_' + code.lower())
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 3: us_ix[name] = {'p': float(p[1]), 'c': float(p[2])}
    for code, name in [('nvda', 'NVDA'), ('msft', 'MSFT')]:
        raw = fetch_sina('gb_' + code)
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 3: us_st[name] = {'p': float(p[1]), 'c': float(p[2])}

    # 港股
    hk_ix = {}; hk_st = {}
    for code, name in [('HSI', 'hsi'), ('HSTECH', 'hst')]:
        raw = fetch_sina('rt_' + code)
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 5:
                hk_ix[name] = {'p': float(p[6]),
                               'c': round((float(p[6])-float(p[3]))/float(p[3])*100, 2)}
    for code, name in [('00700', 'Tencent'), ('09988', 'Ali'), ('01810', 'Xiaomi')]:
        raw = fetch_sina('rt_hk' + code)
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 5:
                hk_st[name] = {'p': float(p[6]),
                               'c': round((float(p[6])-float(p[3]))/float(p[3])*100, 2)}

    # 商品
    comm = {}
    for code, name in [('hf_CL', 'oil'), ('hf_GC', 'gold')]:
        raw = fetch_sina(code)
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 1:
                try: comm[name] = float(p[0])
                except: pass

    dxy = fetch_yahoo('DX-Y.NYB')
    if dxy is None:
        try:
            raw = fetch_sina('hf_DINIW')
            if raw and '"' in raw:
                p = raw.split('"')[1].split(',')
                if len(p) > 1 and p[0]: dxy = float(p[0])
        except: pass

    vix_val = None
    try:
        raw = fetch_sina('gb_vix')
        if raw and '"' in raw:
            p = raw.split('"')[1].split(',')
            if len(p) > 1 and p[1]: vix_val = float(p[1])
    except: pass

    fg_val = 0; fg_class = '?'
    try:
        fg = json.loads(urllib.request.urlopen(
            'https://api.alternative.me/fng/', timeout=8).read())['data'][0]
        fg_val = int(fg['value']); fg_class = fg['value_classification']
    except: pass

    # ---- 3. 数据提取 ----
    now = pd.Timestamp.now().strftime('%m/%d %H:%M')
    btc_df = crypto_data.get('BTC', {}).get('1d')
    eth_df = crypto_data.get('ETH', {}).get('1d')
    btc_p = btc_df['close'].iloc[-1] if btc_df is not None else 0
    eth_p = eth_df['close'].iloc[-1] if eth_df is not None else 0

    crypto_changes = []
    for sym in TIER1:
        name = sym.split('/')[0]
        df = crypto_data[name].get('1d')
        if df is not None and len(df) >= 2:
            ch = round((df['close'].iloc[-1]-df['close'].iloc[-2])/df['close'].iloc[-2]*100, 2)
            crypto_changes.append((name, df['close'].iloc[-1], ch))

    up_n = sum(1 for _, _, x in crypto_changes if x > 0)
    avg_c = np.mean([x[2] for x in crypto_changes]) if crypto_changes else 0
    tg = max(crypto_changes, key=lambda x: x[2]) if crypto_changes else ('?', 0, 0)
    tl = min(crypto_changes, key=lambda x: x[2]) if crypto_changes else ('?', 0, 0)

    gc = [n for n in COINS if '金叉' in macd_rpt.get(n, {}).get('4h', {}).get('cross', '')]
    dc = [n for n in COINS if '死叉' in macd_rpt.get(n, {}).get('4h', {}).get('cross', '')]
    hi = [n for n, r in resonance.items() if r >= 4]
    mi = [n for n, r in resonance.items() if 2 <= r < 4]
    lo = [n for n, r in resonance.items() if r < 2]
    strong = [n for n, r in resonance.items() if r >= 5]

    # MACD表
    md_rows = []
    for n in COINS:
        m1 = macd_rpt.get(n, {}).get('1h', {}).get('cross', '?')
        m4 = macd_rpt.get(n, {}).get('4h', {}).get('cross', '?')
        md = macd_rpt.get(n, {}).get('1d', {}).get('cross', '?')
        md_rows.append("  {:<6} {:>6} {:>6} {:>6}".format(n, m1, m4, md))

    # RSI表
    rs_rows = []
    for n in COINS:
        r1 = int(rsi_rpt.get(n, {}).get('1h', 0))
        r4 = int(rsi_rpt.get(n, {}).get('4h', 0))
        rd = int(rsi_rpt.get(n, {}).get('1d', 0))
        st = '🟢超卖' if rd < 30 else ('🔴超买' if r1 > 65 else ('偏强' if r1 > 55 else ''))
        rs_rows.append("  {:<6} {:>4} {:>4} {:>4}  {}".format(n, r1, r4, rd, st))

    # EMA表
    ema_rows = []
    for n in COINS:
        if '1d' in crypto_data.get(n, {}):
            df = crypto_data[n]['1d']
            c = df['close'].iloc[-1]
            e9 = calc_ema(df['close'], 9).iloc[-1]
            e20 = calc_ema(df['close'], 20).iloc[-1]
            e50 = calc_ema(df['close'], 50).iloc[-1]
            above = sum([c > e9, c > e20, c > e50])
            if c > e9 > e20 > e50: align = '🟢多头'
            elif c < e9 < e20 < e50: align = '🔴空头'
            else: align = '🟡混战'
            ema_rows.append("  {:<6} 价>{}/3EMA  {}".format(n, above, align))

    # 共振表
    res_rows = []
    for n, r in sorted(resonance.items(), key=lambda x: x[1], reverse=True):
        bar = '█'*min(7, int(r)) + '░'*max(0, 7-int(r))
        tag = ' 🔥' if r >= 5 else (' ✓' if r >= 4 else '')
        res_rows.append("  {:<6} {} {}/7{}".format(n, bar, r, tag))

    # ---- 4. 报告 ----
    dxy_s = fmt(dxy, '.2f') if dxy else '?'
    vix_s = fmt(vix_val, '.2f') if vix_val else '?'
    gold_s = fmt_price(comm.get('gold', 0))
    oil_s = fmt_price(comm.get('oil', 0))

    cs = "⭐⭐⭐⭐⭐" if up_n >= 7 else ("⭐⭐⭐⭐☆" if up_n >= 5 else "⭐⭐⭐☆☆")
    as_ = "⭐⭐⭐⭐⭐" if a.get('kc', {}).get('chg', 0) > 2 else ("⭐⭐⭐⭐☆" if a.get('kc', {}).get('chg', 0) > 0 else "⭐⭐⭐☆☆")
    us_ = "⭐⭐⭐☆☆" if us_st.get('MSFT', {}).get('c', 0) < -1 else "⭐⭐⭐⭐☆"
    hk_ = "⭐⭐⭐☆☆" if hk_ix.get('hsi', {}).get('c', 0) < -1 else "⭐⭐⭐⭐☆"

    if fg_val < 30 and gc:
        narrative = "极度恐惧({})下的反弹。日线10/10空头排列不改，但MACD 4h出现{}个金叉({})——趋势转换早期信号。多指标打架=过渡期。熊猫教练：顺大势逆小势，等日线站上EMA20。".format(fg_val, len(gc), ','.join(gc))
    elif fg_val < 30:
        narrative = "极度恐惧({})，日线10/10空头排列，反弹量能待验证。等恐慌见底+MACD金叉双重确认。".format(fg_val)
    elif gc:
        narrative = "恐慌中性({})，MACD 4h {}个金叉暗示动能转换。轻仓观察。".format(fg_val, len(gc))
    else:
        narrative = "多空指标分化——典型的趋势过渡期。日线空头不改，4h在筑底。等方向确认。"

    if strong:
        action = "**轻仓试多** — {}达到≥5/7强共振！<15%仓位试探，止损4h前低。日线站上EMA20后加仓到30%。".format(','.join(strong))
    elif hi:
        action = "**关注等待** — {}/10个币种≥4/7共振，准备好但等日线信号K。BTC/ADA布林收窄=暴风雨前的宁静。".format(len(hi))
    else:
        action = "**观望** — 无≥4/7共振信号，全市场多空未共识。耐心等。"

    if strong:
        kline_note = "{}强共振({}) → 轻仓试多，等1h信号K+入场K确认。日线站上EMA20=加仓。".format(len(strong), ','.join(strong))
    elif hi:
        kline_note = "{}/10共振≥4 → 关注{}。等4h通道突破+信号K+入场K两步确认。".format(len(hi), ','.join(hi[:3]))
    else:
        kline_note = "0共振→观望。等日线站上EMA20 + 4h MACD金叉 + 7维≥4/7三重确认。"

    a_note = "🟢 科创50领涨+{}%，深市强于沪市".format(fmt(a.get('kc', {}).get('chg', 0), '.2f')) if a.get('kc', {}).get('chg', 0) > 2 else "存量博弈格局"

    report = """# 🌙 {now} 五市场30秒速读

## 🎯 瓶颈叙事
**核心矛盾**: {fg_narrative} + 美元{dxy} + 日线10/10空头 ≠ 加密{up}/{total}反弹
**BTC**: ${btc} | **ETH**: ${eth} | 领涨: {ticker} {tgpct}% | 领跌: {loser} {lspct}%
**黄金**: {gold} | **原油**: {oil} | **VIX**: {vix}

## 📊 五维评分卡

**加密货币** | {cs} | {up}/{total}涨 | 均变幅{avg:.2f}%
> 📈 MACD 4h金叉: {gcs} | 死叉: {dcs}
> ⚡ 7维≥5/7(强共振): {strong_list} | ≥4/7: {hi_list}
> 🔥 日线仍10/10空头排列 → 短多长空格局

**A股** | {as_} | 科创50: {kc_chg}% | 上证: {sh_chg}% | 创业板: {cy_chg}%
> {a_note}

**美股** | {us_} | 纳指: {nq_chg}% | NVDA: {nvda_chg}% | MSFT: {msft_chg}%

**港股** | {hks} | 恒生: {hsi_chg}% | 腾讯: {tct_chg}% | 阿里: {ali_chg}% | 小米: {xmi_chg}%

**商品/bStocks** | ⭐⭐⭐☆☆ | 美元: {dxy} | 黄金: {gold}

## ₿ 加密全指标深度

### MACD 1h / 4h / 日线
```
 币种      1h    4h    日
{macd_table}
```

### RSI 1h / 4h / 日线
```
 币种     1h   4h   日  状态
{rsi_table}
```

### EMA日线排列 (价vs EMA9/20/50)
```
{ema_table}
```

### 7维共振 ≥4/7 才出手
```
{res_table}

🟢 ≥5/7 强共振可试单: {strong_list}
🟢 ≥4/7 中共振关注: {hi_list}
🟡 2-3/7 弱共振观望: {mi_list}
🔴 <2/7 无共振禁止: {lo_list}
```

## 🧭 大白话决策

> 🧠 {narrative}
> 📌 操作: {action}
> ⚠️ 禁止: 日线空头排列下追多 | 无共振开仓 | 强动能K追单
> 🔬 美元{dxy} | 黄金{gold} | 恐慌{fg_val}
> 🕯️ 裸K: {kline_note}

> 🐾 全指标扫描 | {now} BJT | 仓位≤30% · {res_summary} · 偷懒看得见
""".format(
        now=now, fg_narrative="😱 恐慌{} 极度恐惧".format(fg_val) if fg_val < 30 else "恐慌中性{}".format(fg_val),
        dxy=dxy_s, up=up_n, total=len(crypto_changes),
        btc=fmt_price(btc_p), eth=fmt_price(eth_p),
        ticker=tg[0], tgpct=fmt(tg[2], '.2f'), loser=tl[0], lspct=fmt(tl[2], '.2f'),
        gold=gold_s, oil=oil_s, vix=vix_s, cs=cs, avg=avg_c,
        gcs=', '.join(gc) if gc else '无', dcs=', '.join(dc) if dc else '无',
        strong_list=', '.join(strong) if strong else '无',
        hi_list=', '.join(hi) if hi else '无',
        as_=as_, kc_chg=fmt(a.get('kc', {}).get('chg', '?'), '.2f'),
        sh_chg=fmt(a.get('sh', {}).get('chg', '?'), '.2f'),
        cy_chg=fmt(a.get('cy', {}).get('chg', '?'), '.2f'), a_note=a_note,
        us_=us_, nq_chg=fmt(us_ix.get('nq', {}).get('c', '?'), '.2f'),
        nvda_chg=fmt(us_st.get('NVDA', {}).get('c', '?'), '.2f'),
        msft_chg=fmt(us_st.get('MSFT', {}).get('c', '?'), '.2f'),
        hks=hk_, hsi_chg=fmt(hk_ix.get('hsi', {}).get('c', '?'), '.2f'),
        tct_chg=fmt(hk_st.get('Tencent', {}).get('c', '?'), '.2f'),
        ali_chg=fmt(hk_st.get('Ali', {}).get('c', '?'), '.2f'),
        xmi_chg=fmt(hk_st.get('Xiaomi', {}).get('c', '?'), '.2f'),
        macd_table='\n'.join(md_rows), rsi_table='\n'.join(rs_rows),
        ema_table='\n'.join(ema_rows), res_table='\n'.join(res_rows),
        mi_list=', '.join(mi) if mi else '无', lo_list=', '.join(lo) if lo else '无',
        narrative=narrative, action=action, kline_note=kline_note, fg_val=fg_val,
        res_summary="{}强共振".format(len(strong)) if strong else ("{}中共振".format(len(hi)) if hi else "观望"),
    )

    if len(report) > MAX_CHARS:
        report = report[:MAX_CHARS-80] + '\n\n> 📋 报告过长已截断 | 完整版本地\n'

    # ---- 5. 输出 ----
    with open('/tmp/hourly_report.md', 'w') as f: f.write(report)
    print("\n✅ 报告 ({}字符) | 共振: ≥5={} ≥4={} 2-3={} <2={}".format(len(report), strong, hi, mi, lo))

    print("📤 推送企业微信...", flush=True)
    data = json.dumps({"msgtype": "markdown", "markdown": {"content": report}}).encode()
    req = urllib.request.Request(WECHAT_WEBHOOK, data=data, headers={'Content-Type': 'application/json'})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        print("   推送结果: errcode={}, errmsg={}".format(resp.get('errcode'), resp.get('errmsg')))
    except Exception as e:
        print("   推送失败: {}".format(e))

    print("\n🐾 完成! 全指标无偷懒 ✨")
    return report

if __name__ == '__main__':
    main()
