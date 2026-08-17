from __future__ import annotations
from datetime import date
from typing import Optional
import time
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 WarrantRadar/2.0"}

def _get_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def _num(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", "", regex=False).str.replace("--", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )

def twse_stock_month(stock_no: str, month: Optional[str] = None) -> pd.DataFrame:
    if month is None:
        month = date.today().strftime("%Y%m01")
    js = _get_json("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
                   {"date": month, "stockNo": stock_no, "response": "json"})
    fields, data = js.get("fields", []), js.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = pd.DataFrame(data, columns=fields).rename(columns={
        "日期":"date","成交股數":"volume","開盤價":"open","最高價":"high","最低價":"low","收盤價":"close"
    })
    def roc(v):
        try:
            y,m,d = [int(z) for z in str(v).split("/")]
            return pd.Timestamp(y+1911,m,d)
        except Exception:
            return pd.NaT
    df["date"] = df["date"].map(roc)
    for c in ["volume","open","high","low","close"]:
        df[c] = _num(df[c])
    df["volume"] = df["volume"] / 1000.0
    return df[["date","open","high","low","close","volume"]].dropna(subset=["close"])

def twse_stock_history(stock_no: str, months: int = 4) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    frames=[]
    for i in range(months-1,-1,-1):
        d=today-pd.DateOffset(months=i)
        try:
            frames.append(twse_stock_month(stock_no,d.strftime("%Y%m01")))
            time.sleep(0.03)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames,ignore_index=True).drop_duplicates("date").sort_values("date")

def twse_all_stock_daily() -> pd.DataFrame:
    """TWSE 上市個股當日/最近交易日成交資訊。用於全市場第一層快速掃描。"""
    js = _get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
    df = pd.DataFrame(js)
    aliases = {
        "code":["Code","證券代號","股票代號"],
        "name":["Name","證券名稱","股票名稱"],
        "volume":["TradeVolume","成交股數","成交量"],
        "turnover":["TradeValue","成交金額"],
        "open":["OpeningPrice","開盤價"],
        "high":["HighestPrice","最高價"],
        "low":["LowestPrice","最低價"],
        "close":["ClosingPrice","收盤價"],
        "change":["Change","漲跌價差"],
        "trade_date":["Date","交易日期","日期"],
    }
    ren={}
    for target,cands in aliases.items():
        for c in cands:
            if c in df.columns:
                ren[c]=target; break
    df=df.rename(columns=ren)
    for c in aliases:
        if c not in df.columns: df[c]=pd.NA
    df["code"]=df["code"].astype(str).str.strip()
    df["name"]=df["name"].astype(str).str.strip()
    for c in ["volume","turnover","open","high","low","close","change"]:
        df[c]=_num(df[c])
    # API volume is normally shares; normalize to lots when magnitude indicates shares.
    if df["volume"].dropna().median() > 10000:
        df["volume_lots"]=df["volume"]/1000.0
    else:
        df["volume_lots"]=df["volume"]
    return df

def twse_warrant_basic() -> pd.DataFrame:
    """TWSE 上市權證基本資料。欄名做多組 alias，避免官方中英欄名調整造成整個 App 失效。"""
    js = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap37_L")
    df = pd.DataFrame(js)
    aliases = {
        "warrant_code":["權證代號","證券代號","WarrantCode"],
        "warrant_name":["權證名稱","證券名稱","WarrantName"],
        "underlying":["標的證券代號","標的代號","標的證券","UnderlyingCode"],
        "issuer":["發行人名稱","發行券商","發行人","IssuerName"],
        "warrant_type":["認購售別","權證類型","Type"],
        "strike":["履約價格","履約價","StrikePrice"],
        "expiry":["到期日期","到期日","ExpirationDate"],
        "ratio":["行使比例","執行比例","ConversionRatio"],
    }
    ren={}
    for target,cands in aliases.items():
        for c in cands:
            if c in df.columns:
                ren[c]=target; break
    x=df.rename(columns=ren).copy()
    for c in aliases:
        if c not in x.columns: x[c]=pd.NA
    x["warrant_code"]=x["warrant_code"].astype(str).str.strip()
    x["warrant_name"]=x["warrant_name"].astype(str).str.strip()
    x["underlying"]=x["underlying"].astype(str).str.extract(r"(\d{4,6})",expand=False)
    x["strike"]=_num(x["strike"])
    x["warrant_type"]=x["warrant_type"].fillna("CALL").astype(str).apply(
        lambda v: "PUT" if ("售" in v or "PUT" in v.upper()) else "CALL")
    return x

def twse_warrant_daily_volume() -> pd.DataFrame:
    js = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap42_L")
    df = pd.DataFrame(js)
    ren={"權證代號":"warrant_code","權證名稱":"warrant_name","成交數量":"volume","成交金額":"turnover","交易日期":"trade_date"}
    df=df.rename(columns=ren)
    if "warrant_code" not in df.columns:
        return pd.DataFrame()
    df["warrant_code"]=df["warrant_code"].astype(str).str.strip()
    if "volume" in df.columns:
        df["volume"]=_num(df["volume"])
        med=df["volume"].dropna().median() if df["volume"].notna().any() else 0
        if med>10000: df["volume"]=df["volume"]/1000.0
    if "turnover" in df.columns: df["turnover"]=_num(df["turnover"])
    return df

def merge_warrant_volume(terms: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if terms is None or terms.empty:
        return daily if daily is not None else pd.DataFrame()
    if daily is None or daily.empty:
        return terms
    x=terms.copy()
    x["warrant_code"]=x["warrant_code"].astype(str).str.strip()
    cols=[c for c in ["warrant_code","volume","turnover","trade_date"] if c in daily.columns]
    d=daily[cols].drop_duplicates("warrant_code",keep="last")
    return x.drop(columns=["volume"],errors="ignore").merge(d,on="warrant_code",how="left")
