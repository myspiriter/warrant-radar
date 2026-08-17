from __future__ import annotations
from datetime import date
from typing import Optional
import time
import re
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
    """TWSE 上市權證基本資料。
    V6.2：採多欄名辨識；若官方欄名調整，至少保留代號/名稱供後續用權證名稱反推標的。
    """
    js = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap37_L")
    df = pd.DataFrame(js)

    aliases = {
        "warrant_code":[
            "權證代號","證券代號","認購（售）權證代號","認購(售)權證代號",
            "WarrantCode","Code","公司代號"
        ],
        "warrant_name":[
            "權證名稱","證券名稱","認購（售）權證名稱","認購(售)權證名稱",
            "WarrantName","Name","公司簡稱"
        ],
        "underlying":[
            "標的證券代號","標的代號","標的證券代碼","標的股票代號",
            "標的證券","UnderlyingCode","UnderlyingSecurityCode"
        ],
        "underlying_name":[
            "標的證券名稱","標的名稱","標的股票名稱","UnderlyingName"
        ],
        "issuer":[
            "發行人名稱","發行券商","發行人","發行證券商名稱",
            "IssuerName","Issuer"
        ],
        "warrant_type":[
            "認購售別","認購（售）別","認購(售)別","權證類型","種類","Type"
        ],
        "strike":[
            "履約價格","履約價","履約價格(元)","StrikePrice","ExercisePrice"
        ],
        "expiry":[
            "到期日期","到期日","最後交易日","ExpirationDate","ExpiryDate"
        ],
        "ratio":[
            "行使比例","執行比例","履約比例","ConversionRatio","ExerciseRatio"
        ],
    }

    ren = {}
    for target, cands in aliases.items():
        for c in cands:
            if c in df.columns:
                ren[c] = target
                break

    x = df.rename(columns=ren).copy()
    for c in aliases:
        if c not in x.columns:
            x[c] = pd.NA

    x["warrant_code"] = x["warrant_code"].fillna("").astype(str).str.strip()
    x["warrant_name"] = x["warrant_name"].fillna("").astype(str).str.strip()
    x["issuer"] = x["issuer"].fillna("").astype(str).str.strip()
    x["underlying_name"] = x["underlying_name"].fillna("").astype(str).str.strip()

    x["underlying"] = (
        x["underlying"].fillna("").astype(str)
        .str.extract(r"(\d{4,6})", expand=False)
        .fillna("")
    )

    x["strike"] = _num(x["strike"])

    raw_type = x["warrant_type"].fillna("").astype(str)
    nm = x["warrant_name"].fillna("").astype(str)
    x["warrant_type"] = [
        "PUT" if ("售" in t or "PUT" in t.upper() or "售" in n) else "CALL"
        for t, n in zip(raw_type, nm)
    ]

    x = x[x["warrant_code"].str.len() > 0].reset_index(drop=True)
    return x


def infer_underlying_from_warrant_name(warrants: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """V6.2 備援：由權證名稱反推標的股票，例如「技嘉富邦61購01」→ 2376。"""
    if warrants is None or warrants.empty:
        return pd.DataFrame() if warrants is None else warrants
    if stocks is None or stocks.empty or "code" not in stocks.columns or "name" not in stocks.columns:
        return warrants

    out = warrants.copy()
    if "underlying" not in out.columns:
        out["underlying"] = ""
    if "underlying_name" not in out.columns:
        out["underlying_name"] = ""

    stock_map = {}
    max_len = 0
    for _, r in stocks[["code","name"]].dropna().iterrows():
        code = str(r["code"]).strip()
        name = str(r["name"]).strip()
        if not code or not name or not re.fullmatch(r"\d{4}", code):
            continue
        stock_map[name] = code
        max_len = max(max_len, len(name))

    names = sorted(stock_map.keys(), key=len, reverse=True)
    name_set = set(names)
    min_len = min((len(n) for n in names), default=2)

    def infer_one(wname):
        s = str(wname or "").strip()
        if not s:
            return ("","")
        for L in range(min(max_len, len(s)), min_len-1, -1):
            p = s[:L]
            if p in name_set:
                return stock_map[p], p
        return ("","")

    underlying_str = out["underlying"].fillna("").astype(str).str.strip()
    missing = underlying_str.eq("") | ~underlying_str.str.fullmatch(r"\d{4}", na=False)
    if missing.any():
        inferred = out.loc[missing, "warrant_name"].map(infer_one)
        out.loc[missing, "underlying"] = inferred.map(lambda z: z[0]).values
        for idx, pair in zip(out.loc[missing].index, inferred):
            current = str(out.at[idx, "underlying_name"] if pd.notna(out.at[idx, "underlying_name"]) else "").strip()
            if not current:
                out.at[idx, "underlying_name"] = pair[1]

    return out

def twse_warrant_daily_volume() -> pd.DataFrame:
    """TWSE 上市權證每日成交資料，V6.2 多欄名辨識。"""
    js = _get_json("https://openapi.twse.com.tw/v1/opendata/t187ap42_L")
    df = pd.DataFrame(js)

    aliases = {
        "warrant_code":["權證代號","證券代號","認購（售）權證代號","認購(售)權證代號","WarrantCode","Code"],
        "warrant_name":["權證名稱","證券名稱","認購（售）權證名稱","認購(售)權證名稱","WarrantName","Name"],
        "volume":["成交數量","成交股數","成交量","成交張數","TradeVolume","Volume"],
        "turnover":["成交金額","成交值","TradeValue","Turnover"],
        "trade_date":["交易日期","日期","Date"],
    }

    ren={}
    for target,cands in aliases.items():
        for c in cands:
            if c in df.columns:
                ren[c]=target
                break

    x=df.rename(columns=ren).copy()
    for c in aliases:
        if c not in x.columns:
            x[c]=pd.NA

    x["warrant_code"]=x["warrant_code"].fillna("").astype(str).str.strip()
    x["warrant_name"]=x["warrant_name"].fillna("").astype(str).str.strip()
    x["volume"]=_num(x["volume"])
    x["turnover"]=_num(x["turnover"])

    med=x["volume"].dropna().median() if x["volume"].notna().any() else 0
    if med > 10000:
        x["volume"]=x["volume"]/1000.0

    return x[x["warrant_code"].str.len()>0].reset_index(drop=True)

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
