from __future__ import annotations

from datetime import datetime, date
import re
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 WarrantRadar/1.0"}


def _get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def twse_stock_month(stock_no: str, month: Optional[str] = None) -> pd.DataFrame:
    if month is None:
        month = date.today().strftime("%Y%m01")
    js = _get_json("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
                   {"date": month, "stockNo": stock_no, "response": "json"})
    fields = js.get("fields", [])
    data = js.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data, columns=fields)
    mapping = {"日期":"date", "成交股數":"volume", "開盤價":"open", "最高價":"high", "最低價":"low", "收盤價":"close"}
    df = df.rename(columns=mapping)
    # ROC yy/mm/dd -> datetime
    def roc_to_date(v):
        try:
            y,m,d = [int(z) for z in str(v).split("/")]
            return pd.Timestamp(y+1911,m,d)
        except Exception:
            return pd.NaT
    df["date"] = df["date"].map(roc_to_date)
    for c in ["volume", "open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.replace("--", "", regex=False), errors="coerce")
    df["volume"] = df["volume"] / 1000.0  # shares -> lots
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def twse_stock_history(stock_no: str, months: int = 4) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    frames = []
    for i in range(months-1, -1, -1):
        d = today - pd.DateOffset(months=i)
        month = d.strftime("%Y%m01")
        try:
            frames.append(twse_stock_month(stock_no, month))
            time.sleep(0.05)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")


def twse_warrant_daily_volume() -> pd.DataFrame:
    js = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap42_L")
    df = pd.DataFrame(js)
    # Expected fields: 出表日期、交易日期、權證代號、權證名稱、成交金額、成交數量
    ren = {"權證代號":"warrant_code", "權證名稱":"warrant_name", "成交數量":"volume", "成交金額":"turnover", "交易日期":"trade_date"}
    df = df.rename(columns=ren)
    if "warrant_code" not in df.columns:
        return pd.DataFrame()
    df["warrant_code"] = df["warrant_code"].astype(str).str.strip()
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        # Dataset can be in shares/units; normalize conservatively when huge values are observed.
        med = df["volume"].dropna().median() if df["volume"].notna().any() else 0
        if med > 10000:
            df["volume"] = df["volume"] / 1000.0
    return df


def merge_warrant_volume(terms: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if terms is None or terms.empty:
        return daily if daily is not None else pd.DataFrame()
    if daily is None or daily.empty:
        return terms
    x = terms.copy()
    if "warrant_code" not in x.columns:
        return x
    x["warrant_code"] = x["warrant_code"].astype(str).str.strip()
    cols = [c for c in ["warrant_code", "volume", "turnover", "trade_date"] if c in daily.columns]
    d = daily[cols].drop_duplicates("warrant_code", keep="last")
    x = x.drop(columns=["volume"], errors="ignore").merge(d, on="warrant_code", how="left")
    return x


def demo_warrants() -> pd.DataFrame:
    # Used only to demonstrate the UI when no complete warrant terms feed is supplied.
    return pd.DataFrame(columns=["warrant_code","warrant_name","underlying","issuer","warrant_type","price","volume","strike","expiry","bid","ask","delta","iv","effective_leverage"])
