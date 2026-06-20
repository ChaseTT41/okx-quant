#!/usr/bin/env python3
"""
港股数据自动更新脚本 🐾
使用 yfinance 拉取最新日线 → 更新 all_daily.parquet + individual parquet
覆盖: 20只核心港股 (五维评分卡 watchlist)
运行: python3 update_hk_data.py
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# ── 抑制 yfinance 日志 ──
import logging
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

DATA_DIR = Path(__file__).parent / "data" / "hk_stocks"
ALL_DAILY = DATA_DIR / "all_daily.parquet"
DAILY_DIR = DATA_DIR / "daily"
STOCK_LIST = DATA_DIR / "stock_list.parquet"
UPDATE_LOG = DATA_DIR / "update_log.txt"

# 核心港股 (五维评分 watchlist + 持仓)
WATCHLIST = {
    "00700": "腾讯控股", "09988": "阿里巴巴", "03690": "美团",
    "02318": "中国平安", "00388": "港交所", "01299": "友邦保险",
    "00939": "建设银行", "01398": "工商银行", "00941": "中国移动",
    "00883": "中海油", "01211": "比亚迪股份", "01024": "快手",
    "09618": "京东", "09999": "网易", "01810": "小米集团",
    "09888": "百度集团", "02269": "药明生物", "01109": "华润置地",
    "02020": "安踏体育", "03968": "招商银行",
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(UPDATE_LOG, "a") as f:
        f.write(line + "\n")


def yf_symbol(code: str) -> str:
    """00700 → 0700.HK, 03690 → 3690.HK"""
    return f"{int(code):04d}.HK"


def fetch_latest(code: str, since_date: str) -> pd.DataFrame | None:
    """用 yfinance 拉取单只股票自 since_date 以来的日线"""
    import yfinance as yf
    try:
        ticker = yf.Ticker(yf_symbol(code))
        df = ticker.history(start=since_date, end=None)
        if df.empty:
            return None

        # 转成标准格式
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "日期", "Open": "开盘", "High": "最高",
            "Low": "最低", "Close": "收盘", "Volume": "成交量",
        })
        df["日期"] = pd.to_datetime(df["日期"]).dt.tz_localize(None)
        df["代码"] = str(code).zfill(5)
        # 只保留需要的列
        df = df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "代码"]]
        df["成交量"] = df["成交量"].fillna(0).astype(int)
        return df
    except Exception as e:
        log(f"  ⚠️ {code} 拉取失败: {e}")
        return None


def update_all_daily(new_rows: pd.DataFrame):
    """将新数据追加到 all_daily.parquet"""
    if ALL_DAILY.exists():
        existing = pd.read_parquet(ALL_DAILY)
        existing["日期"] = pd.to_datetime(existing["日期"])
        existing["代码"] = existing["代码"].astype(str).str.zfill(5)
        # 去重: 同一 (代码, 日期) 只保留新数据
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["代码", "日期"], keep="last")
        combined = combined.sort_values(["代码", "日期"]).reset_index(drop=True)
    else:
        combined = new_rows

    combined.to_parquet(ALL_DAILY, index=False)
    log(f"  ✅ all_daily.parquet: {len(combined)} 行 "
        f"({combined['日期'].min().date()} ~ {combined['日期'].max().date()})")


def update_individual(code: str, new_rows: pd.DataFrame):
    """更新单只股票的 parquet 文件"""
    fpath = DAILY_DIR / f"{str(code).zfill(5)}.parquet"
    if fpath.exists():
        existing = pd.read_parquet(fpath)
        existing["日期"] = pd.to_datetime(existing["日期"])
        existing["代码"] = existing["代码"].astype(str).str.zfill(5)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["日期"], keep="last")
        combined = combined.sort_values("日期").reset_index(drop=True)
    else:
        combined = new_rows
    combined.to_parquet(fpath, index=False)


def main():
    log("=" * 50)
    log("🔄 港股数据更新开始")

    # 读取现有数据，确定最后更新日期
    if ALL_DAILY.exists():
        existing = pd.read_parquet(ALL_DAILY)
        existing["日期"] = pd.to_datetime(existing["日期"])
        last_date = existing["日期"].max()
        log(f"📅 现有数据截止: {last_date.date()}")
    else:
        last_date = pd.Timestamp("2026-01-01")
        log("📅 无现有数据, 从2026-01-01开始拉取")

    # yfinance 需要从 last_date + 1 天开始
    since = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"🔍 拉取日期范围: {since} → 今天")

    all_new = []
    updated = 0
    skipped = 0
    failed = 0

    for code, name in WATCHLIST.items():
        print(f"  📥 {code} {name}...", end=" ")
        new_rows = fetch_latest(code, since)
        if new_rows is None or new_rows.empty:
            print("无新数据")
            skipped += 1
            continue

        try:
            # 更新 individual parquet
            update_individual(code, new_rows)
            all_new.append(new_rows)
            print(f"+{len(new_rows)} 条")
            updated += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    log(f"📊 结果: {updated} 只更新 / {skipped} 只无新数据 / {failed} 只失败")

    if all_new:
        merged = pd.concat(all_new, ignore_index=True)
        merged = merged.drop_duplicates(subset=["代码", "日期"], keep="last")
        update_all_daily(merged)
        log(f"📦 合并写入: {len(merged)} 条新记录, "
            f"日期范围: {merged['日期'].min().date()} ~ {merged['日期'].max().date()}")
    else:
        log("⚠️ 无新数据写入")

    # 验证
    verify = pd.read_parquet(ALL_DAILY)
    verify["日期"] = pd.to_datetime(verify["日期"])
    log(f"✅ 最终状态: {len(verify)} 行, {verify['代码'].nunique()} 只股票, "
        f"日期: {verify['日期'].min().date()} ~ {verify['日期'].max().date()}")
    log("=" * 50)


if __name__ == "__main__":
    main()
