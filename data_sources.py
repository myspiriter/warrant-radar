from __future__ import annotations
from datetime import date
from typing import Optional
import time
import random
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 WarrantRadar/6.17",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.twse.com.tw/",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def _get_json(url, params=None, timeout=20, retries=3, backoff=0.7):
    """HTTP JSON with retry/backoff for TWSE transient 429/5xx failures."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt >= retries:
                raise
            time.sleep(backoff * (2 ** attempt) + random.uniform(0.05, 0.25))
    raise last_err

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
    """
    V6.17 resilient history loader.
    - Retries transient TWSE failures instead of silently returning 0-score data.
    - Records diagnostic metadata in DataFrame.attrs for the UI.
    - If too few rows are returned, performs one slower recovery pass.
    """
    stock_no = str(stock_no).strip()
    today = pd.Timestamp.today().normalize()

    def fetch_pass(delay: float, only_months=None):
        frames, ok_months, errors = [], [], []
        month_items = []
        for i in range(months-1, -1, -1):
            d = today - pd.DateOffset(months=i)
            key = d.strftime("%Y%m01")
            if only_months is None or key in only_months:
                month_items.append(key)
        for key in month_items:
            try:
                x = twse_stock_month(stock_no, key)
                if x is not None and not x.empty:
                    frames.append(x)
                    ok_months.append(key[:6])
                else:
                    errors.append(f"{key[:6]}:空資料")
            except Exception as e:
                errors.append(f"{key[:6]}:{type(e).__name__}:{str(e)[:80]}")
            time.sleep(delay + random.uniform(0.02, 0.08))
        return frames, ok_months, errors

    frames, ok_months, errors = fetch_pass(0.12)
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    else:
        out = pd.DataFrame(columns=["date","open","high","low","close","volume"])

    # A score typically needs ~20-40 sessions. If rate limiting caused holes, retry failed months slowly.
    if len(out) < 45:
        failed_keys = []
        for i in range(months-1, -1, -1):
            d = today - pd.DateOffset(months=i)
            key = d.strftime("%Y%m01")
            if key[:6] not in ok_months:
                failed_keys.append(key)
        if failed_keys:
            time.sleep(0.6)
            f2, ok2, err2 = fetch_pass(0.45, failed_keys)
            errors.extend(err2)
            ok_months.extend(ok2)
            if f2:
                out = pd.concat([out] + f2, ignore_index=True).drop_duplicates("date").sort_values("date")

    out.attrs["history_source"] = "TWSE STOCK_DAY"
    out.attrs["history_rows"] = int(len(out))
    out.attrs["months_ok"] = sorted(set(ok_months))
    out.attrs["history_errors"] = errors[-8:]
    out.attrs["history_status"] = "正常" if len(out) >= 45 else ("部分資料" if len(out) else "無資料")
    return out

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


def twse_mis_quotes(codes, market="tse") -> pd.DataFrame:
    """Best-effort quote overlay from TWSE Market Information System.
    This is used only as a quote overlay; the strategy can still run if it is unavailable.
    """
    codes = [str(c).strip() for c in codes if str(c).strip()]
    if not codes:
        return pd.DataFrame()
    # MIS accepts multiple exchange-channel symbols separated by |
    ex_ch = "|".join([f"{market}_{c}.tw" for c in codes[:80]])
    try:
        js = _get_json(
            "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
            {"ex_ch": ex_ch, "json": "1", "delay": "0", "_": str(int(time.time()*1000))},
            timeout=12,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for r in js.get("msgArray", []) or []:
        code = str(r.get("c") or "").strip()
        if not code:
            continue
        def fnum(v):
            try:
                s = str(v).replace(",", "").strip()
                if s in ("", "-", "--", "nan"):
                    return None
                return float(s)
            except Exception:
                return None
        last = fnum(r.get("z"))
        ref = fnum(r.get("y"))
        open_ = fnum(r.get("o"))
        high = fnum(r.get("h"))
        low = fnum(r.get("l"))
        volume = fnum(r.get("v"))
        bid = None
        ask = None
        try:
            b = str(r.get("b","")).split("_")[0]
            bid = fnum(b)
        except Exception:
            pass
        try:
            a = str(r.get("a","")).split("_")[0]
            ask = fnum(a)
        except Exception:
            pass
        change_pct = ((last/ref)-1)*100 if last is not None and ref not in (None,0) else None
        rows.append({
            "code": code,
            "name": str(r.get("n") or "").strip(),
            "last": last,
            "reference": ref,
            "open_live": open_,
            "high_live": high,
            "low_live": low,
            "volume_live": volume,
            "bid": bid,
            "ask": ask,
            "change_pct_live": change_pct,
            "quote_date": str(r.get("d") or ""),
            "quote_time": str(r.get("t") or ""),
            "source": "TWSE MIS",
        })
    return pd.DataFrame(rows)
