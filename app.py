from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import streamlit as st

from data_sources import twse_stock_history, twse_all_stock_daily, twse_warrant_basic, twse_warrant_daily_volume, merge_warrant_volume, twse_mis_quotes, infer_underlying_from_warrant_name, enrich_warrant_live
from radar import stock_score, rank_warrants, normalize_warrant_columns, entry_plan, add_indicators

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
HISTORY_PATH = APP_DIR / "signal_history.csv"

# 常用名稱作為離線備援；正常情況會優先使用 TWSE 全市場資料的公司名稱
COMPANY_NAMES = {
    "2330":"台積電","2317":"鴻海","2376":"技嘉","2382":"廣達","3231":"緯創","6669":"緯穎",
    "2308":"台達電","3017":"奇鋐","2454":"聯發科","2345":"智邦","3037":"欣興","3443":"創意",
    "3661":"世芯-KY","3653":"健策"
}
DYNAMIC_NAMES = {}

def stock_label(code: str) -> str:
    code=str(code)
    name=DYNAMIC_NAMES.get(code) or COMPANY_NAMES.get(code,"")
    return f"{code} {name}".strip()

def zh_stock_table(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "code" in x.columns:
        x["股票"] = x["code"].astype(str).map(stock_label)
    x = x.rename(columns={
        "score":"機會分數", "setup":"訊號類型", "close":"前收/日線收盤價", "data_confidence":"資料信心(%)", "score_reason":"評分重點", "candidate_status":"候選狀態",
        "ret1_pct":"日線漲跌幅(%)", "ret5_pct":"近5日漲跌幅(%)",
        "volume_ratio":"日線量比", "rsi14":"RSI強弱指標", "ma20":"20日均線", "breakout_quality":"突破品質", "support":"支撐承接", "risk":"風險控制", "consistency":"穩定性",
        "last":"盤中成交價", "bid":"盤中委買", "ask":"盤中委賣",
        "change_pct_live":"盤中漲跌幅(%)", "volume_live":"盤中累計量",
        "quote_time":"行情時間"
    })
    cols=[c for c in ["股票","機會分數","候選等級","候選狀態","資料信心(%)","盤中判斷","訊號類型","評分重點","盤中成交價","盤中漲跌幅(%)","盤中委買","盤中委賣","行情時間","前收/日線收盤價","日線漲跌幅(%)","近5日漲跌幅(%)","日線量比","RSI強弱指標","突破品質","支撐承接","風險控制","穩定性","20日均線"] if c in x.columns]
    return x[cols]

def zh_warrant_table(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "underlying" in x.columns:
        x["標的股票"] = x["underlying"].astype(str).map(stock_label)
    if "stock_code" in x.columns:
        x["標的股票"] = x["stock_code"].astype(str).map(stock_label)
    if "eligible" in x.columns:
        x["eligible"] = x["eligible"].map({True:"通過", False:"不通過"}).fillna(x["eligible"])

    x = x.rename(columns={
        "stock_score":"現股分數", "warrant_code":"權證代號", "warrant_name":"權證名稱",
        "issuer":"發行券商", "price":"權證價格", "volume":"成交量(張)",
        "strike":"履約價", "expiry":"到期日", "days_to_expiry":"剩餘天數",
        "otm_pct":"價外幅度(%)", "delta":"Delta敏感度",
        "effective_leverage":"有效槓桿(倍)", "spread_pct":"買賣價差(%)",
        "iv":"隱含波動率IV(%)", "warrant_score":"權證分數",
        "combo_score":"綜合分數", "eligible":"是否通過篩選", "filter_reason":"篩選結果",
        "warrant_quote_time":"權證行情時間",
    })

    # 不再顯示 Python None；無公開資料者用明確文字。
    public_missing = ["發行券商","權證價格","履約價","到期日","剩餘天數","價外幅度(%)","買賣價差(%)"]
    broker_missing = ["Delta敏感度","有效槓桿(倍)","隱含波動率IV(%)"]

    for c in public_missing:
        if c in x.columns:
            x[c] = x[c].where(pd.notna(x[c]), "—")
    for c in broker_missing:
        if c in x.columns:
            x[c] = x[c].where(pd.notna(x[c]), "需券商資料")

    return x

st.set_page_config(page_title="個人版權證雷達", page_icon="📡", layout="wide")

@st.cache_data(ttl=900, show_spinner=False)
def load_hist(code: str):
    return twse_stock_history(code, months=4)

@st.cache_data(ttl=900, show_spinner=False)
def load_warrant_volume():
    try:
        return twse_warrant_daily_volume()
    except Exception:
        return pd.DataFrame()

MARKET_LAST_GOOD = Path("/tmp/warrant_radar_last_good_market.csv")

@st.cache_data(ttl=300, show_spinner=False)
def load_all_stocks():
    """V6.4：多來源抓取＋本次雲端執行期間的 last-good 快取。"""
    try:
        df = twse_all_stock_daily()
        if df is not None and not df.empty:
            try:
                df.to_csv(MARKET_LAST_GOOD, index=False, encoding="utf-8-sig")
            except Exception:
                pass
            return df
    except Exception:
        pass

    # 如果官方來源暫時全部失效，使用本次 Streamlit instance 最後一次成功快取。
    try:
        if MARKET_LAST_GOOD.exists():
            cached = pd.read_csv(MARKET_LAST_GOOD, dtype={"code":str})
            if not cached.empty:
                cached["data_source"] = "本機 last-good 快取"
                cached["source_mode"] = "快取備援"
                return cached
    except Exception:
        pass

    return pd.DataFrame(columns=[
        "code","name","volume","turnover","open","high","low","close",
        "change","trade_date","volume_lots","data_source","source_mode","fetched_at"
    ])

@st.cache_data(ttl=1800, show_spinner=False)
def load_warrant_basic():
    try:
        return twse_warrant_basic()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_live_quotes(codes_tuple):
    try:
        return twse_mis_quotes(list(codes_tuple))
    except Exception:
        return pd.DataFrame()

def data_health(all_stocks, wb, wv):
    rows=[]
    rows.append(("上市股票日資料", not all_stocks.empty, len(all_stocks)))
    rows.append(("權證基本資料", not wb.empty, len(wb)))
    rows.append(("權證成交資料", not wv.empty, len(wv)))
    ok=sum(1 for _,x,_ in rows if x)
    return rows, ok

def live_signal_text(score, setup, live_pct):
    if live_pct is None or pd.isna(live_pct):
        return "⚪ 等待盤中報價"
    if score >= 75 and setup not in ["過熱・不追","弱勢觀察","資料不足","資料錯誤"]:
        if -2.5 <= live_pct <= 3.5:
            return "🟢 可觀察進場"
        if live_pct > 5:
            return "🔴 急漲不追"
        return "🟡 等止穩"
    if score >= 62:
        return "🟡 NEXT 觀察"
    return "⚪ 暫不進場"


def load_cfg():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


LAST_RANKING_PATH = Path("/tmp/warrant_radar_last_ranking.csv")

def load_last_ranking():
    try:
        if LAST_RANKING_PATH.exists():
            return pd.read_csv(LAST_RANKING_PATH, dtype={"code":str})
    except Exception:
        pass
    return pd.DataFrame()

def save_last_ranking(df):
    try:
        if df is not None and not df.empty:
            df.to_csv(LAST_RANKING_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def preserve_missing_high_score(current: pd.DataFrame, previous: pd.DataFrame,
                                prior_threshold=70, max_keep=10):
    """V6.8 候選防消失。
    前次高分股若本次因資料/預篩關聯而完全消失，保留在 ranking 內，
    但降為『等待資料確認』且不允許直接進今日推薦。
    """
    if previous is None or previous.empty:
        return current
    if current is None:
        current = pd.DataFrame()

    prev = previous.copy()
    if "score" not in prev.columns or "code" not in prev.columns:
        return current

    prev["score"] = pd.to_numeric(prev["score"], errors="coerce").fillna(0)
    prev = prev[prev["score"] >= prior_threshold].sort_values("score", ascending=False)

    cur_codes = set(current["code"].astype(str)) if (not current.empty and "code" in current.columns) else set()
    keep = prev[~prev["code"].astype(str).isin(cur_codes)].head(max_keep).copy()
    if keep.empty:
        if not current.empty:
            current["candidate_status"] = current.get("candidate_status", "本次正常")
        return current

    keep["last_score"] = keep["score"]
    keep["score"] = (keep["score"] * 0.88).clip(upper=69.0).round(1)
    keep["setup"] = "資料待確認"
    keep["candidate_status"] = "前次高分保留"
    keep["score_reason"] = "前次為高分候選，本次未進候選池；先保留觀察，等待資料/預篩確認"
    keep["data_confidence"] = 40.0

    if current.empty:
        out = keep
    else:
        cur = current.copy()
        if "candidate_status" not in cur.columns:
            cur["candidate_status"] = "本次正常"
        out = pd.concat([cur, keep], ignore_index=True, sort=False)

    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    return out.sort_values("score", ascending=False).reset_index(drop=True)

def save_signal_snapshot(ranking: pd.DataFrame):
    if ranking.empty:
        return
    snap = ranking.copy()
    snap["snapshot_date"] = pd.Timestamp.today().date().isoformat()
    if HISTORY_PATH.exists():
        old = pd.read_csv(HISTORY_PATH, dtype={"code":str})
        old = old[old["snapshot_date"] != snap["snapshot_date"].iloc[0]]
        snap = pd.concat([old, snap], ignore_index=True)
    snap.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def issuer_from_name(name: str):
    issuers = ["元大","凱基","群益","永豐","富邦","元富","統一","國票","兆豐","中信","玉山","第一金","華南永昌","康和","國泰","台新"]
    s = str(name)
    for i in issuers:
        if i in s: return i
    return ""


def stock_ranking(watchlist):
    base_cols = ["code","score","setup","close","ret1_pct","ret5_pct","volume_ratio","rsi14","ma20"]
    rows = []
    for code in watchlist:
        try:
            h = load_hist(code)
            s = stock_score(h)
            row = {"code":str(code)}
            if isinstance(s, dict):
                row.update(s)
            rows.append(row)
        except Exception as e:
            rows.append({"code":str(code), "score":0, "setup":"資料錯誤", "error":str(e)})
    if not rows:
        return pd.DataFrame(columns=base_cols)
    out = pd.DataFrame(rows)
    for c in base_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out["setup"] = out["setup"].fillna("資料不足")
    return out.sort_values("score", ascending=False).reset_index(drop=True)



def build_dynamic_pool(all_stocks: pd.DataFrame, wb: pd.DataFrame, wv: pd.DataFrame,
                       min_stock_volume=1000, min_turnover_m=50, min_liquid_warrants=3,
                       max_candidates=60):
    """全市場第一層：只保留有權證、現股流動性正常、且有足夠活躍認購權證的股票。"""
    if all_stocks.empty or wb.empty:
        return pd.DataFrame()
    s=all_stocks.copy()
    s=s[s["code"].str.fullmatch(r"\d{4}", na=False)]
    s=s[(s["volume_lots"].fillna(0)>=min_stock_volume) & (s["turnover"].fillna(0)>=min_turnover_m*1_000_000)]
    w=wb.copy()
    w=w[w["warrant_type"].eq("CALL")]
    if not wv.empty and "volume" in wv.columns:
        w=w.merge(wv[["warrant_code","volume"]].drop_duplicates("warrant_code",keep="last"),
                  on="warrant_code",how="left")
    else:
        w["volume"]=0
    liquid=w[w["volume"].fillna(0)>=cfg["min_warrant_volume"]]
    counts=liquid.groupby("underlying").agg(
        活躍權證數=("warrant_code","nunique"),
        權證總成交量=("volume","sum")
    ).reset_index().rename(columns={"underlying":"code"})
    s=s.merge(counts,on="code",how="inner")
    s=s[s["活躍權證數"]>=min_liquid_warrants]
    # 第一層優先看現股成交金額、權證活躍度，避免對全市場逐檔抓歷史造成速度過慢
    s["預篩分數"]=(s["turnover"].rank(pct=True)*55 + s["權證總成交量"].rank(pct=True)*30 +
                 s["活躍權證數"].rank(pct=True)*15)
    return s.sort_values("預篩分數",ascending=False).head(max_candidates).reset_index(drop=True)

def dynamic_ranking(pool: pd.DataFrame):
    base_cols = [
        "code","name","活躍權證數","現股成交量(張)","現股成交金額",
        "score","raw_score","setup","close","ret1_pct","ret5_pct","ret20_pct",
        "volume_ratio","rsi14","ma20","breakout_quality","support","risk","consistency",
        "data_confidence","score_reason"
    ]
    if pool is None or pool.empty:
        return pd.DataFrame(columns=base_cols)
    rows=[]
    for _,r in pool.iterrows():
        code=str(r.get("code","")).strip()
        if not code:
            continue
        try:
            h=load_hist(code)
            s=stock_score(h)
            row={"code":code,"name":r.get("name",""),"活躍權證數":r.get("活躍權證數",0),
                 "現股成交量(張)":r.get("volume_lots",0),"現股成交金額":r.get("turnover",0)}
            if isinstance(s, dict):
                row.update(s)
            rows.append(row)
        except Exception as e:
            rows.append({"code":code,"name":r.get("name",""),"score":0,"setup":"資料錯誤","error":str(e)})
    if not rows:
        return pd.DataFrame(columns=base_cols)
    out=pd.DataFrame(rows)
    for c in base_cols:
        if c not in out.columns:
            out[c]=pd.NA
    out["score"]=pd.to_numeric(out["score"],errors="coerce").fillna(0)
    out["setup"]=out["setup"].fillna("資料不足")
    return out.sort_values("score",ascending=False).reset_index(drop=True)



def fallback_market_candidates(all_stocks: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    """正式權證預篩為空時的保底候選池。
    以成交金額 70% + 成交量 30% 排序，確保畫面不會因權證資料關聯失敗而整頁空白。
    """
    required = ["code","name","close","volume_lots","turnover"]
    if all_stocks is None or all_stocks.empty:
        return pd.DataFrame(columns=required)

    x = all_stocks.copy()
    for c in required:
        if c not in x.columns:
            x[c] = pd.NA

    x["code"] = x["code"].astype(str).str.strip()
    x = x[x["code"].str.fullmatch(r"\d{4}", na=False)]
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x["volume_lots"] = pd.to_numeric(x["volume_lots"], errors="coerce").fillna(0)
    x["turnover"] = pd.to_numeric(x["turnover"], errors="coerce").fillna(0)
    x = x[(x["close"] > 0) & (x["volume_lots"] > 0)]

    if x.empty:
        return pd.DataFrame(columns=required)

    x["_成交金額排名"] = x["turnover"].rank(pct=True)
    x["_成交量排名"] = x["volume_lots"].rank(pct=True)
    x["預篩分數"] = x["_成交金額排名"] * 70 + x["_成交量排名"] * 30
    return x.sort_values("預篩分數", ascending=False).head(max_rows).reset_index(drop=True)


def screening_funnel(all_stocks, wb, wv, min_stock_volume, min_turnover_m, min_liquid_warrants, min_warrant_volume):
    """顯示每一關剩幾檔，避免只看到 0 卻不知道卡在哪。"""
    rows = []
    if all_stocks is None or all_stocks.empty:
        return pd.DataFrame([{"篩選階段":"全市場股票","剩餘檔數":0}])

    s = all_stocks.copy()
    s["code"] = s["code"].astype(str).str.strip()
    s = s[s["code"].str.fullmatch(r"\d{4}", na=False)]
    rows.append({"篩選階段":"全市場四位數上市股票","剩餘檔數":len(s)})

    s["volume_lots"] = pd.to_numeric(s["volume_lots"], errors="coerce").fillna(0)
    s["turnover"] = pd.to_numeric(s["turnover"], errors="coerce").fillna(0)

    s = s[s["volume_lots"] >= min_stock_volume]
    rows.append({"篩選階段":f"現股成交量 ≥ {min_stock_volume:,} 張","剩餘檔數":len(s)})

    s = s[s["turnover"] >= min_turnover_m * 1_000_000]
    rows.append({"篩選階段":f"現股成交金額 ≥ {min_turnover_m:,} 百萬元","剩餘檔數":len(s)})

    if wb is None or wb.empty:
        rows.append({"篩選階段":"具有權證基本資料","剩餘檔數":0})
        return pd.DataFrame(rows)

    w = wb.copy()
    if "underlying" not in w.columns:
        rows.append({"篩選階段":"權證標的代號可辨識","剩餘檔數":0})
        return pd.DataFrame(rows)

    if "warrant_type" in w.columns:
        w = w[w["warrant_type"].eq("CALL")]

    # 先看有認購權證的標的
    under = set(w["underlying"].dropna().astype(str).str.strip())
    s_has = s[s["code"].isin(under)]
    rows.append({"篩選階段":"具有認購權證","剩餘檔數":len(s_has)})

    if wv is None or wv.empty or "volume" not in wv.columns:
        rows.append({"篩選階段":"有當日權證成交資料","剩餘檔數":0})
        return pd.DataFrame(rows)

    vv = wv[["warrant_code","volume"]].copy()
    vv["volume"] = pd.to_numeric(vv["volume"], errors="coerce").fillna(0)
    w = w.merge(vv.drop_duplicates("warrant_code", keep="last"), on="warrant_code", how="left")
    liquid = w[w["volume"].fillna(0) >= min_warrant_volume]
    counts = liquid.groupby("underlying")["warrant_code"].nunique()
    valid_under = set(counts[counts >= min_liquid_warrants].index.astype(str))
    s_final = s_has[s_has["code"].isin(valid_under)]
    rows.append({"篩選階段":f"至少 {min_liquid_warrants} 張活躍認購權證","剩餘檔數":len(s_final)})
    return pd.DataFrame(rows)


def candidate_grade(score):
    score = pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0]
    if score >= 80:
        return "🔥 強力候選"
    if score >= 70:
        return "🟢 今日推薦"
    if score >= 60:
        return "🟡 NEXT 潛在"
    if score >= 50:
        return "⚪ 持續觀察"
    return "低優先"



def count_warrants_for_code(code, warrants_df, min_warrant_volume):
    code = str(code).strip()
    if warrants_df is None or warrants_df.empty or "underlying" not in warrants_df.columns:
        return 0, 0
    x = warrants_df.copy()
    x["underlying"] = x["underlying"].fillna("").astype(str).str.strip()
    x = x[x["underlying"] == code]
    if "warrant_type" in x.columns:
        calls = x[x["warrant_type"].fillna("CALL").eq("CALL")]
    else:
        calls = x
    total_calls = len(calls)
    if "volume" in calls.columns:
        vol = pd.to_numeric(calls["volume"], errors="coerce").fillna(0)
        active_calls = int((vol >= min_warrant_volume).sum())
    else:
        active_calls = 0
    return total_calls, active_calls


def diagnose_stock(code, all_stocks, ranking, pool, warrants_df,
                   min_stock_volume, min_turnover_m, min_liquid_warrants,
                   min_warrant_volume):
    code = str(code).strip()
    result = {
        "股票代號": code,
        "公司名稱": DYNAMIC_NAMES.get(code) or COMPANY_NAMES.get(code, ""),
        "現股成交量(張)": None,
        "現股成交金額(百萬元)": None,
        "成交量門檻": "未知",
        "成交金額門檻": "未知",
        "認購權證數": 0,
        "活躍認購權證數": 0,
        "權證流動性門檻": "未知",
        "正式預篩": "未進入",
        "目前排名": None,
        "機會分數": None,
        "訊號類型": "",
        "候選等級": "",
        "盤中判斷": "",
        "落選原因": "",
    }

    # 現股資料
    if all_stocks is not None and not all_stocks.empty and "code" in all_stocks.columns:
        s = all_stocks[all_stocks["code"].astype(str).str.strip() == code]
        if not s.empty:
            r = s.iloc[0]
            vol = pd.to_numeric(r.get("volume_lots"), errors="coerce")
            turn = pd.to_numeric(r.get("turnover"), errors="coerce")
            result["公司名稱"] = str(r.get("name") or result["公司名稱"])
            result["現股成交量(張)"] = None if pd.isna(vol) else float(vol)
            result["現股成交金額(百萬元)"] = None if pd.isna(turn) else float(turn)/1_000_000
            result["成交量門檻"] = "通過" if (not pd.isna(vol) and vol >= min_stock_volume) else "未通過"
            result["成交金額門檻"] = "通過" if (not pd.isna(turn) and turn >= min_turnover_m*1_000_000) else "未通過"

    # 權證資料
    total_calls, active_calls = count_warrants_for_code(code, warrants_df, min_warrant_volume)
    result["認購權證數"] = total_calls
    result["活躍認購權證數"] = active_calls
    result["權證流動性門檻"] = "通過" if active_calls >= min_liquid_warrants else "未通過"

    # 正式預篩
    if pool is not None and not pool.empty and "code" in pool.columns:
        result["正式預篩"] = "已進入" if code in set(pool["code"].astype(str)) else "未進入"

    # 排名與分數
    if ranking is not None and not ranking.empty and "code" in ranking.columns:
        rr = ranking.reset_index(drop=True)
        m = rr[rr["code"].astype(str) == code]
        if not m.empty:
            idx = int(m.index[0])
            row = m.iloc[0]
            result["目前排名"] = idx + 1
            result["機會分數"] = float(pd.to_numeric(row.get("score"), errors="coerce") or 0)
            result["訊號類型"] = str(row.get("setup") or "")
            result["候選等級"] = str(row.get("候選等級") or "")
            result["盤中判斷"] = str(row.get("盤中判斷") or "")

    # 落選原因：依優先順序組合
    reasons = []
    if result["成交量門檻"] == "未通過":
        reasons.append(f"現股成交量低於 {min_stock_volume:,} 張")
    if result["成交金額門檻"] == "未通過":
        reasons.append(f"現股成交金額低於 {min_turnover_m:,} 百萬元")
    if result["認購權證數"] == 0:
        reasons.append("目前未成功對應到認購權證")
    elif result["活躍認購權證數"] < min_liquid_warrants:
        reasons.append(f"活躍認購權證少於 {min_liquid_warrants} 張")
    if result["機會分數"] is None:
        reasons.append("未進入深度評分候選池")
    else:
        if result["機會分數"] < 70:
            reasons.append("機會分數未達今日推薦門檻 70 分")
        if result["訊號類型"] in ["過熱・不追","弱勢觀察","資料錯誤","資料不足"]:
            reasons.append(f"目前訊號為「{result['訊號類型']}」")
        if result["盤中判斷"] == "🔴 急漲不追":
            reasons.append("盤中漲幅過熱，系統判定不追價")
        elif result["盤中判斷"] == "🟡 等止穩":
            reasons.append("盤中仍偏弱，等待止穩確認")

    result["落選原因"] = "；".join(reasons) if reasons else "目前條件大致通過，若仍未進榜，可能是排名被其他高分標的擠出。"
    return result


cfg = load_cfg()
st.title("📡 個人版台股權證雷達 V6.8")
st.caption("全市場策略掃描 → 資料健康檢查 → 盤中行情覆蓋 → 今日推薦 → 最佳權證。策略底稿使用 TWSE 公開資料；盤中行情以 TWSE 市況資訊做最佳努力覆蓋，仍請下單前以券商報價確認。")

# 先抓市場與權證資料，建立每日動態股票池
with st.spinner("掃描全市場與權證流動性…"):
    all_stocks = load_all_stocks()
    wvol = load_warrant_volume()
    wbasic = load_warrant_basic()

# V6.4 市場資料來源狀態
market_source = "無可用資料"
market_mode = "異常"
market_fetched_at = "—"
if not all_stocks.empty:
    if "data_source" in all_stocks.columns and all_stocks["data_source"].notna().any():
        market_source = str(all_stocks["data_source"].dropna().iloc[0])
    if "source_mode" in all_stocks.columns and all_stocks["source_mode"].notna().any():
        market_mode = str(all_stocks["source_mode"].dropna().iloc[0])
    if "fetched_at" in all_stocks.columns and all_stocks["fetched_at"].notna().any():
        market_fetched_at = str(all_stocks["fetched_at"].dropna().iloc[0])

if not all_stocks.empty:
    for _, r in all_stocks[["code","name"]].dropna().iterrows():
        DYNAMIC_NAMES[str(r["code"])] = str(r["name"])

# V6.3：官方標的代號缺失時，由權證名稱前綴反推標的股票
if not wbasic.empty and not all_stocks.empty:
    try:
        wbasic = infer_underlying_from_warrant_name(wbasic, all_stocks)
    except Exception as _e:
        st.warning(f"權證名稱備援關聯暫時失敗：{_e}")


health_rows, health_ok = data_health(all_stocks, wbasic, wvol)
with st.expander("🩺 資料健康檢查", expanded=(health_ok < 3)):
    h1,h2,h3,h4 = st.columns(4)
    for col,(label,ok,count) in zip([h1,h2,h3],health_rows):
        col.metric(label, "正常" if ok else "異常/無資料", f"{count:,} 筆")
    h4.metric("市場資料來源", market_mode, market_source)
    if health_ok < 3:
        st.warning("部分公開資料源目前沒有回傳資料。v6 會降級運作，不會把『資料缺失』直接判定成『市場沒有機會』。")
    else:
        st.success("策略掃描所需的三組公開資料均已取得。")
    if not all_stocks.empty:
        if market_mode == "主來源":
            st.caption(f"市場資料：{market_source}｜抓取時間：{market_fetched_at}")
        else:
            st.warning(f"目前使用市場資料備援模式：{market_source}｜抓取時間：{market_fetched_at}。榜單可用，但請注意資料可能較主來源延遲。")
    if not wbasic.empty:
        _total_w = len(wbasic)
        _linked_w = int(wbasic["underlying"].astype(str).str.fullmatch(r"\d{4}", na=False).sum()) if "underlying" in wbasic.columns else 0
        _call_w = int((wbasic["warrant_type"]=="CALL").sum()) if "warrant_type" in wbasic.columns else 0
        st.caption(f"V6.3 權證關聯診斷：基本資料 {_total_w:,} 筆｜可辨識股票標的 {_linked_w:,} 筆｜認購權證 {_call_w:,} 筆")


with st.sidebar:
    st.header("V6 盤中模式")
    intraday_mode = st.toggle("啟用盤中行情覆蓋", value=True)
    live_top_n = st.slider("盤中更新前幾名候選", 3, 20, 10, step=1)
    if st.button("🔄 立即刷新盤中行情", use_container_width=True):
        load_live_quotes.clear()
        st.rerun()
    if st.button("♻️ 重新抓取全市場資料", use_container_width=True):
        load_all_stocks.clear()
        st.rerun()
    st.caption("盤中行情約每20秒可重新抓取；策略分數仍以較完整的日線資料計算，避免只看瞬間波動。")
    st.divider()
    st.header("每日動態掃描")
    scan_mode = st.toggle("啟用全市場自動掃描", value=True)
    min_stock_volume = st.number_input("現股最低成交量（張）", 0, 1000000, 1000, step=500)
    min_turnover_m = st.number_input("現股最低成交金額（百萬元）", 0, 100000, 50, step=10)
    min_liquid_warrants = st.number_input("至少活躍認購權證數", 1, 50, 3, step=1)
    max_candidates = st.slider("進入深度評分候選數", 20, 100, 60, step=10)
    fallback_count = st.slider("正式預篩為0時，保底候選數", 10, 50, 30, step=5)
    show_funnel = st.toggle("顯示篩選漏斗診斷", value=True)
    st.caption("先用流動性與權證活躍度快速預篩，再對候選股計算技術/量價分數，避免全市場逐檔抓歷史造成過慢。")

    st.divider()
    st.header("我的關注股")
    watch_txt = st.text_area("固定關注（逗號分隔）", ",".join(cfg["watchlist"]), height=80)
    watchlist = [x.strip() for x in watch_txt.replace("\\n", ",").split(",") if x.strip()]

    st.subheader("資金配置")
    first_amount = st.number_input("首筆投入金額",0,500000,int(cfg.get("first_entry_amount",50000)),step=10000)
    reserve_amount = st.number_input("預備金額",0,500000,int(cfg.get("reserve_amount",50000)),step=10000)
    capital=first_amount+reserve_amount
    if capital>500000: st.error("首筆投入＋預備金合計上限為 500,000 元。")
    else: st.caption(f"目前單一標的總配置：NT$ {capital:,.0f}／上限 NT$ 500,000")

    st.subheader("權證硬條件")
    min_vol=st.number_input("權證最低成交量（張）",0,100000,int(cfg["min_warrant_volume"]),step=50)
    min_dte=st.number_input("最低剩餘天數",0,1000,int(cfg["min_days_to_expiry"]),step=10)
    max_otm=st.number_input("最大價外 %",0.0,100.0,float(cfg["max_otm_pct"]),step=1.0)
    cfg["min_warrant_volume"]=min_vol; cfg["min_days_to_expiry"]=min_dte; cfg["max_otm_pct"]=max_otm

    st.divider()
    uploaded=st.file_uploader("補充券商權證 CSV（可選）",type=["csv"])
    st.caption("官方資料可做每日掃描；若要 Delta、IV、即時買賣價差，仍建議補充券商資料或未來串接即時行情。")

previous_ranking = load_last_ranking()

if scan_mode:
    funnel_df = screening_funnel(
        all_stocks, wbasic, wvol,
        min_stock_volume, min_turnover_m, min_liquid_warrants,
        cfg["min_warrant_volume"]
    )

    pool = build_dynamic_pool(
        all_stocks, wbasic, wvol,
        min_stock_volume, min_turnover_m,
        min_liquid_warrants, max_candidates
    )
    strict_pool = pool.copy()

    fallback_used = False
    if pool.empty:
        fallback_used = True
        pool = fallback_market_candidates(all_stocks, fallback_count)
        st.warning(
            "正式『有足夠活躍權證』預篩目前為 0 檔。"
            "V6.3 已自動啟動保底候選池，因此下方仍會列出全市場高流動性股票進行評分。"
            "保底候選不等於權證可直接買進，仍需到『權證排行』確認。"
        )

    with st.spinner(f"深度評分 {len(pool)} 檔候選股…"):
        ranking = dynamic_ranking(pool)

    if show_funnel:
        with st.expander("🔍 為什麼正式預篩會是 0？｜篩選漏斗", expanded=fallback_used):
            st.dataframe(funnel_df, width="stretch", hide_index=True)
            if not funnel_df.empty:
                st.caption("最後一關若突然歸零，通常代表權證標的代號/成交量資料關聯或門檻造成，而不是全市場真的沒有股票。")
else:
    fallback_used = False
    funnel_df = pd.DataFrame()
    strict_pool = pd.DataFrame()
    with st.spinner("更新固定關注股…"):
        ranking = stock_ranking(watchlist)

# V6.8：前次高分候選若因本次資料/預篩問題消失，保留為「資料待確認」
if scan_mode:
    ranking = preserve_missing_high_score(ranking, previous_ranking, prior_threshold=70, max_keep=10)

save_last_ranking(ranking)

# 固定關注股另外計算，不受每日 TOP 排名限制
with st.spinner("更新我的關注股…"):
    watch_ranking=stock_ranking(watchlist)

if not ranking.empty:
    save_signal_snapshot(ranking)
# Warrant data：官方基本資料 + 每日成交量為預設；上傳券商 CSV 時優先使用較完整欄位
if uploaded is not None:
    try:
        wterms=pd.read_csv(uploaded,encoding="utf-8-sig")
    except UnicodeDecodeError:
        wterms=pd.read_csv(uploaded,encoding="big5")
    wterms=normalize_warrant_columns(wterms)
    if wterms["issuer"].isna().all() or (wterms["issuer"].astype(str).str.len()==0).all():
        wterms["issuer"]=wterms["warrant_name"].map(issuer_from_name)
    warrants=merge_warrant_volume(wterms,wvol)
else:
    warrants=merge_warrant_volume(wbasic,wvol) if not wbasic.empty else pd.DataFrame()
    if not warrants.empty and not all_stocks.empty:
        warrants=infer_underlying_from_warrant_name(warrants, all_stocks)

# 若基本資料仍完全無法對應標的，但每日成交資料有名稱，建立最低限度備援關聯
if (warrants is None or warrants.empty or
    ("underlying" in warrants.columns and
     warrants["underlying"].astype(str).str.fullmatch(r"\d{4}", na=False).sum() == 0)):
    if not wvol.empty and "warrant_name" in wvol.columns:
        _synthetic = wvol.copy()
        if "warrant_code" not in _synthetic.columns:
            _synthetic["warrant_code"] = ""
        _synthetic["issuer"] = ""
        _synthetic["warrant_type"] = _synthetic["warrant_name"].fillna("").astype(str).map(
            lambda n: "PUT" if "售" in n else "CALL"
        )
        for _c in ["strike","expiry","ratio","underlying","underlying_name"]:
            if _c not in _synthetic.columns:
                _synthetic[_c] = pd.NA
        _synthetic = infer_underlying_from_warrant_name(_synthetic, all_stocks)
        warrants = _synthetic

# 資料新鮮度提示
if not all_stocks.empty:
    td = all_stocks["trade_date"].dropna().astype(str) if "trade_date" in all_stocks.columns else pd.Series(dtype=str)
    trade_text = td.iloc[0] if len(td) else "最近交易日"
    st.caption(
        f"📅 市場資料：{trade_text}｜來源：{market_source}（{market_mode}）｜"
        f"本版為公開資料掃描，並非券商逐筆即時報價。"
    )
else:
    st.error(
        "目前三組 TWSE 全市場資料來源皆無法取得，而且尚無 last-good 快取。"
        "這是『資料源異常』，不是市場沒有投資標的；固定關注股仍可使用。"
    )

# 統一資料表欄位防呆：無論資料源是否暫時回傳空值，後面都不會因缺欄位而中斷
required_ranking_cols = {
    "code": "",
    "score": 0.0,
    "setup": "資料不足",
    "close": pd.NA,
    "ret1_pct": pd.NA,
    "ret5_pct": pd.NA,
    "volume_ratio": pd.NA,
    "rsi14": pd.NA,
    "ma20": pd.NA,
    "breakout_quality": pd.NA,
    "support": pd.NA,
    "risk": pd.NA,
    "consistency": pd.NA,
    "data_confidence": pd.NA,
    "score_reason": "",
    "candidate_status": "本次正常",
    "last_score": pd.NA,
}
if ranking is None or not isinstance(ranking, pd.DataFrame):
    ranking = pd.DataFrame(columns=list(required_ranking_cols.keys()))
for _col, _default in required_ranking_cols.items():
    if _col not in ranking.columns:
        ranking[_col] = _default
ranking["score"] = pd.to_numeric(ranking["score"], errors="coerce").fillna(0)
ranking["setup"] = ranking["setup"].fillna("資料不足").astype(str)
ranking["候選等級"] = ranking["score"].map(candidate_grade)
if "candidate_status" in ranking.columns:
    _preserved = ranking["candidate_status"].eq("前次高分保留")
    ranking.loc[_preserved, "候選等級"] = "🟠 前次高分／待確認"

# V6.7 盤中行情覆蓋
# 不再只抓 ranking 前 N 名；改成：
# 1) ranking 前 N 名
# 2) NEXT 頁可能出現的前 10 名
# 3) 固定關注股
live_quotes = pd.DataFrame()

for _c in ["last","bid","ask","volume_live","change_pct_live","quote_date","quote_time"]:
    if _c not in ranking.columns:
        ranking[_c] = pd.NA

if intraday_mode and not ranking.empty:
    _codes = []

    # 全市場前 N
    _codes += ranking.head(live_top_n)["code"].astype(str).tolist()

    # NEXT 可能顯示的前 10 名：55~69 分＋指定 setup
    _next_for_quote = ranking[
        (ranking["score"] >= 55) & (ranking["score"] < 70) &
        (ranking["setup"].isin(["蓄勢/中性","趨勢回檔","突破型"]))
    ].head(10)
    _codes += _next_for_quote["code"].astype(str).tolist()

    # 固定關注股
    _codes += [str(c) for c in watchlist]

    # 去重
    _codes = list(dict.fromkeys([c for c in _codes if re.fullmatch(r"\d{4}", str(c))]))

    if _codes:
        live_quotes = load_live_quotes(tuple(_codes))
        if not live_quotes.empty:
            _live_cols = [
                "code","last","bid","ask","volume_live",
                "change_pct_live","quote_date","quote_time"
            ]
            _live_cols = [c for c in _live_cols if c in live_quotes.columns]
            ranking = ranking.drop(
                columns=[c for c in _live_cols if c != "code" and c in ranking.columns],
                errors="ignore"
            ).merge(
                live_quotes[_live_cols].drop_duplicates("code", keep="last"),
                on="code", how="left"
            )

ranking["盤中判斷"] = ranking.apply(
    lambda r: live_signal_text(
        float(r.get("score",0) or 0),
        str(r.get("setup","")),
        pd.to_numeric(r.get("change_pct_live"), errors="coerce")
    ),
    axis=1
)

# Header metrics
c1,c2,c3,c4 = st.columns(4)
valid = ranking[pd.to_numeric(ranking["score"], errors="coerce").fillna(0) > 0].copy()
c1.metric("今日候選", len(ranking))
c2.metric("80分以上強力候選", int((pd.to_numeric(valid["score"], errors="coerce").fillna(0) >= 80).sum()))
c3.metric("事件型反彈", int((valid["setup"] == "事件型反彈").sum()))
c4.metric("趨勢回檔/突破", int(valid["setup"].isin(["趨勢回檔","突破型"]).sum()))

TAB_TODAY, TAB_NEXT, TAB_WATCH, TAB_YDAY, TAB_WARRANT, TAB_SETTINGS = st.tabs(
    ["🔥 今日推薦","🔭 NEXT 潛在標的","⭐ 我的關注股","⏪ 昨日追蹤","🎯 權證排行","⚙️ 模型說明"])

with TAB_TODAY:
    st.subheader("今日推薦")
    recommended = ranking[
        (pd.to_numeric(ranking["score"], errors="coerce").fillna(0) >= 70) &
        (~ranking["setup"].isin(["過熱・不追","弱勢觀察","資料錯誤","資料不足","資料待確認"])) &
        (~ranking["candidate_status"].eq("前次高分保留"))
    ].copy()

    if recommended.empty:
        if ranking.empty:
            st.error("連保底候選池也無法建立。這代表股票市場日資料本身異常，不是『沒有投資標的』。")
            recommended = ranking.copy()
        else:
            st.info("今天沒有 ≥70 分的正式推薦；以下改顯示目前全市場最高分 TOP 10，供觀察而非直接買進。")
            recommended = ranking.head(10).copy()
    else:
        recommended = recommended.head(10).copy()
    show_df = zh_stock_table(recommended)
    st.dataframe(show_df, width="stretch", hide_index=True,
                 column_config={"機會分數":st.column_config.ProgressColumn("機會分數", min_value=0,max_value=100,format="%.1f")})

    top_codes = recommended[pd.to_numeric(recommended["score"], errors="coerce").fillna(0) > 0]["code"].astype(str).tolist() if ("score" in recommended.columns and "code" in recommended.columns) else []
    if top_codes:
        chosen = st.selectbox("查看個股", top_codes, format_func=stock_label, key="today_stock")
        sr = ranking[ranking.code == chosen].iloc[0].to_dict()
        h = add_indicators(load_hist(chosen))
        left,right = st.columns([2,1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=h.date, open=h.open, high=h.high, low=h.low, close=h.close, name=stock_label(chosen)))
            fig.add_trace(go.Scatter(x=h.date, y=h.ma20, name="20日均線"))
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown(f"### {stock_label(chosen)}｜{sr.get('setup','')}")
            st.metric("現股分數", f"{sr.get('score',0):.1f}/100")
            st.write(f"趨勢 {sr.get('trend',0)}/18｜量價 {sr.get('volume_price',0)}/14｜動能 {sr.get('momentum',0)}/12")
            st.write(f"位置 {sr.get('position',0)}/12｜突破品質 {sr.get('breakout_quality',0)}/12｜支撐承接 {sr.get('support',0)}/10")
            st.write(f"風險控制 {sr.get('risk',0)}/10｜穩定性 {sr.get('consistency',0)}/12")
            st.write(f"**資料信心：** {sr.get('data_confidence','—')}%")
            st.caption(f"評分重點：{sr.get('score_reason','—')}")
            plan = entry_plan(sr, first_amount, reserve_amount)
            st.info(plan["instruction"])
            st.write("**加碼條件：**", plan["add_rule"])
            st.write("**風控條件：**", plan["stop_rule"])

        if not warrants.empty:
            wr = rank_warrants(warrants, chosen, float(sr.get("close",0)), cfg)
            st.markdown("#### 此標的權證 TOP 5")
            if wr.empty:
                st.warning("上傳的權證資料中沒有這個標的。")
            else:
                cols = ["warrant_code","warrant_name","issuer","price","volume","strike","expiry","days_to_expiry","otm_pct","delta","effective_leverage","spread_pct","iv","warrant_score","eligible","filter_reason"]
                st.dataframe(zh_warrant_table(wr[[c for c in cols if c in wr.columns]].head(5)), width="stretch", hide_index=True)
        else:
            st.warning("目前沒有完整權證條款資料。請在左側上傳券商匯出的權證 CSV，就能直接算出『最佳券商＋最佳權證』。TWSE 每日成交量資料已可自動抓取。")

with TAB_YDAY:
    st.subheader("昨日訊號追蹤")
    if HISTORY_PATH.exists():
        hist_sig = pd.read_csv(HISTORY_PATH, dtype={"code":str})
        dates = sorted(hist_sig.snapshot_date.unique(), reverse=True)
        if len(dates) >= 2:
            yd = hist_sig[hist_sig.snapshot_date == dates[1]].copy().sort_values("score", ascending=False)
            today_map = ranking.set_index("code")
            def now_ret(r):
                try:
                    return (float(today_map.loc[r.code,"close"])/float(r.close)-1)*100
                except Exception: return np.nan
            yd["至今變化(%)"] = yd.apply(now_ret, axis=1)
            yd_show = yd[[c for c in ["code","score","setup","close","至今變化(%)"] if c in yd.columns]].head(20).copy()
            yd_show["股票"] = yd_show["code"].astype(str).map(stock_label)
            yd_show = yd_show.rename(columns={"score":"昨日機會分數","setup":"昨日訊號類型","close":"昨日收盤價"})
            yd_show = yd_show[[c for c in ["股票","昨日機會分數","昨日訊號類型","昨日收盤價","至今變化(%)"] if c in yd_show.columns]]
            st.caption(f"訊號日期：{dates[1]}")
            st.dataframe(yd_show, width="stretch", hide_index=True)
        else:
            st.info("目前只有今天的快照；明天再打開就會開始形成昨日績效。")
    else:
        st.info("尚無歷史快照。")

with TAB_NEXT:
    st.subheader("NEXT｜接近買點的潛在標的")
    if ranking.empty:
        st.info("目前沒有資料。")
    else:
        nxt = ranking[
            (ranking["score"] >= 55) & (ranking["score"] < 70) &
            (ranking["setup"].isin(["蓄勢/中性","趨勢回檔","突破型"]))
        ].copy().head(10)
        if nxt.empty:
            st.info("目前沒有明確的 NEXT 候選。")
        else:
            _nshow = zh_stock_table(nxt)
            for _c in ["盤中成交價","盤中漲跌幅(%)","盤中委買","盤中委賣","行情時間"]:
                if _c in _nshow.columns:
                    _nshow[_c] = _nshow[_c].where(pd.notna(_nshow[_c]), "等待盤中報價")
            st.dataframe(_nshow, width="stretch", hide_index=True)
            st.caption("NEXT 不是立即買進訊號；代表條件正在接近，後續若量價/突破條件改善，可能進入今日推薦。")

with TAB_WATCH:
    st.subheader("我的固定關注股")
    st.caption("這份清單不會因每日全市場排名而消失，可在左側自行修改。")
    st.dataframe(zh_stock_table(watch_ranking), width="stretch", hide_index=True)

with TAB_WARRANT:
    st.subheader("🎯 個股推薦前三名＋各自權證前五名")
    st.caption("先挑股票，再挑權證，避免單一股票的權證佔滿整張排行榜。")
    st.caption("V6.7 權證價格採：最後成交價 → 委買/委賣中間價 → 日資料價格；履約價/到期日由基本條款精準合併。Delta／IV／有效槓桿若無券商資料，維持『需券商資料』。")


    _base_stock = ranking.copy()

    if _base_stock.empty:
        st.info("目前沒有可建立個股前三名的候選股票。")
    else:
        _base_stock["score"] = pd.to_numeric(_base_stock["score"], errors="coerce").fillna(0)
        _base_stock = _base_stock.sort_values("score", ascending=False).reset_index(drop=True)

        _formal = _base_stock[
            (_base_stock["score"] >= 70) &
            (~_base_stock["setup"].isin(["過熱・不追","弱勢觀察","資料錯誤","資料不足"]))
        ].copy()

        _selected_codes = []
        for _c in _formal["code"].astype(str).tolist():
            if _c not in _selected_codes:
                _selected_codes.append(_c)
            if len(_selected_codes) >= 3:
                break

        if len(_selected_codes) < 3:
            for _c in _base_stock["code"].astype(str).tolist():
                if _c not in _selected_codes:
                    _selected_codes.append(_c)
                if len(_selected_codes) >= 3:
                    break

        st.markdown("### 🏆 個股推薦 TOP 3")
        _top3 = _base_stock[_base_stock["code"].astype(str).isin(_selected_codes)].copy()
        _top3["_順序"] = _top3["code"].astype(str).map({c:i for i,c in enumerate(_selected_codes)})
        _top3 = _top3.sort_values("_順序").drop(columns=["_順序"], errors="ignore")
        st.dataframe(
            zh_stock_table(_top3),
            width="stretch",
            hide_index=True,
            column_config={
                "機會分數": st.column_config.ProgressColumn(
                    "機會分數", min_value=0, max_value=100, format="%.1f"
                )
            }
        )

        if warrants is None or warrants.empty:
            st.warning("目前沒有權證資料，因此暫時只能顯示個股前三名。")
        else:
            for _rank, _code in enumerate(_selected_codes, start=1):
                _row = _base_stock[_base_stock["code"].astype(str) == str(_code)]
                if _row.empty:
                    continue
                _row = _row.iloc[0]

                _live = pd.to_numeric(_row.get("last"), errors="coerce")
                _close = pd.to_numeric(_row.get("close"), errors="coerce")
                _spot = float(_live if pd.notna(_live) else (_close if pd.notna(_close) else 0))

                st.markdown(
                    f"### {_rank}. {stock_label(_code)}｜現股分數 {float(_row.get('score',0)):.1f}"
                )

                # V6.6：只針對這檔股票的權證補 MIS 盤中價格/委買/委賣。
                _wbase = warrants[
                    warrants["underlying"].astype(str) == str(_code)
                ].copy() if ("underlying" in warrants.columns) else warrants.copy()
                try:
                    _wbase = enrich_warrant_live(_wbase, max_codes=60)
                except Exception:
                    pass

                _wr = rank_warrants(_wbase, str(_code), _spot, cfg)

                if _wr is None or _wr.empty:
                    st.info(f"{stock_label(_code)}：目前沒有可排名的權證資料。")
                    continue

                if "warrant_score" in _wr.columns:
                    _wr["warrant_score"] = pd.to_numeric(
                        _wr["warrant_score"], errors="coerce"
                    ).fillna(0)

                if "eligible" in _wr.columns:
                    _pass = _wr[_wr["eligible"] == True].copy()
                    _fail = _wr[_wr["eligible"] != True].copy()

                    if "warrant_score" in _pass.columns:
                        _pass = _pass.sort_values("warrant_score", ascending=False)
                    if "warrant_score" in _fail.columns:
                        _fail = _fail.sort_values("warrant_score", ascending=False)

                    _top5 = pd.concat([_pass, _fail], ignore_index=True).head(5)
                else:
                    if "warrant_score" in _wr.columns:
                        _top5 = _wr.sort_values("warrant_score", ascending=False).head(5)
                    else:
                        _top5 = _wr.head(5)

                _top5 = _top5.copy()
                _top5.insert(0, "同標的排名", range(1, len(_top5)+1))

                _cols = [
                    "同標的排名","warrant_code","warrant_name","issuer",
                    "price","volume","strike","expiry","days_to_expiry",
                    "otm_pct","delta","effective_leverage","spread_pct","iv",
                    "warrant_score","eligible","filter_reason"
                ]
                _show = _top5[[c for c in _cols if c in _top5.columns]]

                st.dataframe(
                    zh_warrant_table(_show),
                    width="stretch",
                    hide_index=True
                )

                if "eligible" in _top5.columns and (_top5["eligible"] == True).any():
                    _best = _top5[_top5["eligible"] == True].iloc[0]
                    _best_score = pd.to_numeric(_best.get("warrant_score"), errors="coerce")
                    _best_score = float(_best_score) if pd.notna(_best_score) else 0.0
                    st.success(
                        f"此標的目前首選：{_best.get('warrant_code','')} "
                        f"{_best.get('warrant_name','')}｜權證分數 {_best_score:.1f}"
                    )
                else:
                    st.warning(
                        "這檔股票目前沒有權證通過全部硬條件；表格列出相對較佳者供觀察。"
                    )

with TAB_SETTINGS:
    st.markdown("""
### V6.8 評分架構
現股機會分數改為八大因子，共100分：  
**趨勢18｜量價14｜動能12｜位置12｜突破品質12｜支撐承接10｜風險控制10｜穩定性12。**

另外加入「資料信心」與「前次高分防消失」機制。前次≥70分的股票若本次只是因資料/預篩缺失消失，
會降為 **🟠 前次高分／待確認**，不會直接從雷達消失，也不會被誤列為正式今日推薦。
""")
    st.markdown("""
### 現股 100 分
- 趨勢 20：均線排列、20日方向
- 量價 20：爆量急跌、量縮回測、放量突破
- 技術位置 15：離20日線距離、RSI
- 動能 15：5日/20日報酬與過熱扣分
- 事件 15：異常漲跌＋異常量（第一版以量價代理）
- 基本面代理 15：第一版以中期趨勢代理，**不假裝等同財報**

### 權證 100 分
- 成交量 20
- 剩餘天數 15
- 價內外 20
- Delta 15
- 有效槓桿 10
- 買賣價差 10
- IV 10

### 預設硬性淘汰
成交量 < 300 張、剩餘天數 < 120 天、價外 > 15%。

### 下一階段最值得升級
若你有券商 API，可把即時現股/權證 bid-ask、Delta、IV、有效槓桿直接餵給本 App；評分引擎不用重寫，只替換資料來源即可。
""")

st.caption("資料來源以 TWSE 公開資料為主；免費公開資料可能為盤後/延遲。投資前仍需以券商即時報價確認。")
