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
from radar import kd_decline_signal_v617

# V6.18.1：避免 Streamlit/GitHub 檔案版本不同步造成整個 App ImportError。
try:
    from radar import overheat_score_v614
except ImportError:
    def overheat_score_v614(hist):
        """內建備援過熱評分；當遠端 radar.py 尚未同步時仍可正常啟動。"""
        if hist is None or len(hist) < 22:
            return {
                "過熱分數": 0,
                "過熱等級": "⚪ 資料不足",
                "過熱原因": "歷史資料不足",
                "反轉風險": 0
            }

        x = add_indicators(hist)
        r = x.iloc[-1]

        def _s(v, d=0):
            try:
                return d if pd.isna(v) else float(v)
            except Exception:
                return d

        close = _s(r.get("close"))
        ma5 = _s(r.get("ma5"), close)
        rsi = _s(r.get("rsi14"), 50)
        dist = _s(r.get("dist_ma20"), 0)
        bb = _s(r.get("bb_pos"), .5)
        ret1 = _s(r.get("ret1"), 0)
        ret3 = _s(r.get("ret3"), 0)
        ret5 = _s(r.get("ret5"), 0)
        ret10 = _s(r.get("ret10"), 0)
        volr = _s(r.get("volume_ratio"), 1)
        atr = _s(r.get("atr_pct"), .03)
        loc = _s(r.get("close_location"), .5)
        obv = _s(r.get("obv"), 0)
        obvma = _s(r.get("obv_ma10"), obv)
        high = _s(r.get("high"), close)
        low = _s(r.get("low"), close)
        op = _s(r.get("open"), close)

        rng = max(high-low, 1e-9)
        upper = max(0, high-max(op, close))/rng

        score = 0
        reasons = []

        if rsi >= 85:
            score += 20; reasons.append(f"RSI {rsi:.0f} 極端過熱")
        elif rsi >= 78:
            score += 16; reasons.append(f"RSI {rsi:.0f} 明顯過熱")
        elif rsi >= 72:
            score += 10; reasons.append(f"RSI {rsi:.0f} 偏熱")
        elif rsi >= 68:
            score += 5

        dp = dist * 100
        if dp >= 18:
            score += 20; reasons.append(f"高於MA20約 {dp:.1f}%")
        elif dp >= 13:
            score += 16; reasons.append(f"MA20乖離 {dp:.1f}%")
        elif dp >= 9:
            score += 10
        elif dp >= 6:
            score += 5

        if bb >= 1.15:
            score += 10; reasons.append("價格明顯超出布林上緣")
        elif bb >= 1.02:
            score += 7
        elif bb >= .9:
            score += 3

        if ret5 >= .15 or ret10 >= .25:
            score += 15; reasons.append("短期漲幅過快")
        elif ret5 >= .10 or ret10 >= .18:
            score += 11
        elif ret3 >= .07 or ret5 >= .07:
            score += 6

        if volr >= 3:
            score += 10; reasons.append("高檔爆量")
        elif volr >= 2:
            score += 7
        elif volr >= 1.5:
            score += 4

        if upper >= .45 and loc <= .55:
            score += 10; reasons.append("高檔長上影、追價轉弱")
        elif upper >= .3:
            score += 6

        if atr >= .065:
            score += 5; reasons.append("波動率快速擴張")
        elif atr >= .045:
            score += 3

        if close > ma5 and obv < obvma:
            score += 6; reasons.append("股價強但OBV未同步")
        if ret1 < 0 and ret5 > .08 and volr >= 1.5:
            score += 4; reasons.append("急漲後爆量轉弱")

        score = round(min(100, max(0, score)), 1)
        if score >= 80:
            level = "🔴 極度過熱"
        elif score >= 65:
            level = "🟠 明顯過熱"
        elif score >= 50:
            level = "🟡 偏熱觀察"
        else:
            level = "🟢 尚未過熱"

        reversal = min(
            100,
            score*.65
            + (15 if upper >= .4 else 0)
            + (10 if ret1 < 0 and volr >= 1.5 else 0)
            + (10 if obv < obvma else 0)
        )

        return {
            "過熱分數": score,
            "過熱等級": level,
            "過熱原因": "；".join(reasons[:5]) if reasons else "未出現明顯過熱訊號",
            "反轉風險": round(reversal, 1),
            "RSI": round(rsi, 1),
            "MA20乖離(%)": round(dp, 1),
            "量比": round(volr, 2),
            "近5日漲幅(%)": round(ret5*100, 1),
            "近10日漲幅(%)": round(ret10*100, 1),
        }

from radar import chip_score_v611, blend_score_v611
from radar import chip_event_v613, adjust_for_chip_event_v613

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
        "score":"機會分數", "trend_score":"趨勢分數", "event_score":"事件分數", "score_mode":"評分模型", "setup":"訊號類型", "close":"前收/日線收盤價", "data_confidence":"資料信心(%)", "score_reason":"評分重點", "candidate_status":"候選狀態",
        "ret1_pct":"日線漲跌幅(%)", "ret5_pct":"近5日漲跌幅(%)",
        "volume_ratio":"日線量比", "rsi14":"RSI強弱指標", "ma20":"20日均線", "breakout_quality":"突破品質", "support":"支撐承接", "risk":"風險控制", "consistency":"穩定性",
        "last":"盤中成交價", "bid":"盤中委買", "ask":"盤中委賣",
        "change_pct_live":"盤中漲跌幅(%)", "volume_live":"盤中累計量",
        "quote_time":"行情時間"
    })
    cols=[c for c in ["股票","機會分數","趨勢分數","事件分數","評分模型","候選等級","候選狀態","資料信心(%)","盤中判斷","訊號類型","評分重點","盤中成交價","盤中漲跌幅(%)","盤中委買","盤中委賣","行情時間","前收/日線收盤價","日線漲跌幅(%)","近5日漲跌幅(%)","日線量比","RSI強弱指標","突破品質","支撐承接","風險控制","穩定性","20日均線"] if c in x.columns]
    out = x[cols].copy()
    for _c in out.columns:
        if _c != "機會分數":
            out[_c] = out[_c].where(pd.notna(out[_c]), "—")
    return out

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


LAST_RANKING_PATH = APP_DIR / "last_ranking.csv"
EVENT_TRACK_PATH = APP_DIR / "event_tracking.csv"

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


def load_event_tracking():
    try:
        if EVENT_TRACK_PATH.exists():
            return pd.read_csv(EVENT_TRACK_PATH, dtype={"code":str})
    except Exception:
        pass
    return pd.DataFrame()

def update_event_tracking(ranking: pd.DataFrame, keep_days=5):
    today = pd.Timestamp.today().normalize()
    old = load_event_tracking()

    fresh = pd.DataFrame()
    if ranking is not None and not ranking.empty:
        mode = ranking["score_mode"] if "score_mode" in ranking.columns else pd.Series("", index=ranking.index)
        setup = ranking["setup"] if "setup" in ranking.columns else pd.Series("", index=ranking.index)
        mask = mode.eq("事件型模型") | setup.isin(["事件型機會","事件型反彈"])
        fresh = ranking[mask].copy()
        if not fresh.empty:
            fresh["event_last_seen"] = today.strftime("%Y-%m-%d")

    if old is not None and not old.empty and "event_last_seen" in old.columns:
        old["event_last_seen"] = pd.to_datetime(old["event_last_seen"], errors="coerce")
        old = old[(today - old["event_last_seen"]).dt.days <= keep_days].copy()
        old["event_last_seen"] = old["event_last_seen"].dt.strftime("%Y-%m-%d")

    combined = pd.concat([old, fresh], ignore_index=True, sort=False) if old is not None and not old.empty else fresh
    if combined is None or combined.empty:
        return pd.DataFrame()

    combined["code"] = combined["code"].astype(str)
    combined = combined.sort_values("event_last_seen").drop_duplicates("code", keep="last")
    try:
        combined.to_csv(EVENT_TRACK_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return combined

def merge_event_tracking(ranking: pd.DataFrame, event_track: pd.DataFrame, keep_days=5):
    if event_track is None or event_track.empty:
        return ranking
    if ranking is None:
        ranking = pd.DataFrame()

    cur_codes = set(ranking["code"].astype(str)) if (not ranking.empty and "code" in ranking.columns) else set()
    missing = event_track[~event_track["code"].astype(str).isin(cur_codes)].copy()
    if missing.empty:
        return ranking

    today = pd.Timestamp.today().normalize()
    seen = pd.to_datetime(missing["event_last_seen"], errors="coerce")
    age = (today - seen).dt.days.fillna(keep_days)

    original = pd.to_numeric(missing["score"], errors="coerce").fillna(0)
    decay = (1 - 0.04 * age.clip(lower=1, upper=keep_days))
    missing["last_score"] = original
    missing["score"] = (original * decay).clip(upper=69).round(1)
    missing["candidate_status"] = "事件追蹤保留"
    missing["setup"] = "事件追蹤"
    missing["score_mode"] = "事件追蹤"
    missing["score_reason"] = "事件型高分股持續追蹤；本次未進候選池，等待最新資料重新確認"
    missing["data_confidence"] = 50.0

    if ranking.empty:
        out = missing
    else:
        out = pd.concat([ranking, missing], ignore_index=True, sort=False)

    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    return out.sort_values("score", ascending=False).reset_index(drop=True)

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
        "score","trend_score","event_score","score_mode","setup","close","ret1_pct","ret5_pct","ret20_pct",
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



def event_grade(score):
    s = pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0]
    if s >= 80:
        return "🔥 強力事件機會"
    if s >= 70:
        return "🟢 事件型推薦"
    if s >= 60:
        return "🟡 事件型觀察"
    return "未達事件門檻"

def build_event_pool(all_stocks: pd.DataFrame,
                     min_stock_volume=1000,
                     min_turnover_m=50,
                     max_candidates=120):
    """V6.18.1 獨立事件池：
    只要求現股流動性，不要求活躍權證數，避免事件股先被權證資料刷掉。
    """
    if all_stocks is None or all_stocks.empty:
        return pd.DataFrame()

    s = all_stocks.copy()
    s["code"] = s["code"].astype(str).str.strip()
    s = s[s["code"].str.fullmatch(r"\d{4}", na=False)].copy()

    s["volume_lots"] = pd.to_numeric(s.get("volume_lots"), errors="coerce").fillna(0)
    s["turnover"] = pd.to_numeric(s.get("turnover"), errors="coerce").fillna(0)

    s = s[
        (s["volume_lots"] >= min_stock_volume) &
        (s["turnover"] >= min_turnover_m * 1_000_000)
    ].copy()

    if s.empty:
        return s

    s["_流動性分"] = (
        s["turnover"].rank(pct=True) * 65 +
        s["volume_lots"].rank(pct=True) * 35
    )
    return s.sort_values("_流動性分", ascending=False).head(max_candidates).reset_index(drop=True)

def event_dynamic_ranking(event_pool: pd.DataFrame):
    """只做事件型評分，不依賴權證門檻。"""
    base_cols = [
        "code","name","score","trend_score","event_score","score_mode","setup",
        "data_confidence","score_reason","close","ret1_pct","ret5_pct","ret20_pct",
        "volume_ratio","rsi14","ma20",
        "event_shock","event_washout","event_stabilization","event_rebound","event_risk"
    ]
    if event_pool is None or event_pool.empty:
        return pd.DataFrame(columns=base_cols)

    rows = []
    for _, r in event_pool.iterrows():
        code = str(r.get("code","")).strip()
        if not code:
            continue
        try:
            h = load_hist(code)
            s = stock_score(h)
            row = {
                "code": code,
                "name": r.get("name",""),
                "現股成交量(張)": r.get("volume_lots", pd.NA),
                "現股成交金額": r.get("turnover", pd.NA),
            }
            if isinstance(s, dict):
                row.update(s)
            rows.append(row)
        except Exception as e:
            rows.append({
                "code":code, "name":r.get("name",""),
                "score":0, "event_score":0, "trend_score":0,
                "score_mode":"資料錯誤", "setup":"資料錯誤",
                "score_reason":str(e)
            })

    if not rows:
        return pd.DataFrame(columns=base_cols)

    out = pd.DataFrame(rows)
    for c in base_cols:
        if c not in out.columns:
            out[c] = pd.NA

    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out["event_score"] = pd.to_numeric(out["event_score"], errors="coerce").fillna(0)
    out["trend_score"] = pd.to_numeric(out["trend_score"], errors="coerce").fillna(0)

    # V6.18.1：只要事件分 >= 60 就進事件雷達，不要求事件分比趨勢分高6分。
    out["事件等級"] = out["event_score"].map(event_grade)
    out = out[out["event_score"] >= 60].copy()

    return out.sort_values(
        ["event_score","score"],
        ascending=[False,False]
    ).reset_index(drop=True)



def event_reason_text(row):
    def n(k):
        v = pd.to_numeric(row.get(k), errors="coerce")
        return float(v) if pd.notna(v) else 0.0
    shock,wash,stab,rebound,risk = [n(k) for k in ["event_shock","event_washout","event_stabilization","event_rebound","event_risk"]]
    ret1,volr = n("ret1_pct"),n("volume_ratio")
    reasons=[]
    if shock>=14: reasons.append("近期出現明顯爆量/急跌事件")
    elif shock>=8: reasons.append("近期價格與成交量出現異常波動")
    if wash>=13: reasons.append("恐慌賣壓後出現承接/洗盤跡象")
    if stab>=13: reasons.append("短線跌勢開始止穩")
    if rebound>=13: reasons.append("反彈動能已出現確認")
    if risk>=14: reasons.append("事件風險已有收斂")
    elif risk<=8: reasons.append("但目前波動風險仍偏高")
    if ret1>0 and volr>=1.1: reasons.append("今日上漲且量能配合")
    return "；".join((reasons or ["事件模型綜合分數達觀察門檻"])[:4])

def market_summary_text(ranking, event_ranking):
    ranking = pd.DataFrame() if ranking is None else ranking
    event_ranking = pd.DataFrame() if event_ranking is None else event_ranking
    scores = pd.to_numeric(ranking.get("score", pd.Series(dtype=float)), errors="coerce").dropna()
    es = pd.to_numeric(event_ranking.get("event_score", pd.Series(dtype=float)), errors="coerce").dropna()
    strong=int((scores>=80).sum()) if len(scores) else 0
    good=int(((scores>=70)&(scores<80)).sum()) if len(scores) else 0
    avg=float(scores.mean()) if len(scores) else 0
    en=int((es>=60).sum()) if len(es) else 0
    estrong=int((es>=80).sum()) if len(es) else 0
    if strong>=3 and avg>=72:
        tone="偏多且機會擴散"; action="可優先從高分股尋找量價確認後的進場點，但避免追高。"
    elif strong>=1 or good>=3:
        tone="偏多但屬選股行情"; action="強弱分化，宜集中高分且有籌碼或事件確認的標的。"
    elif en>=3:
        tone="震盪、事件型機會較多"; action="宜等待事件股止穩與反彈確認，不宜全面追價。"
    else:
        tone="中性偏保守"; action="高品質訊號不多，建議提高進場門檻並控制部位。"
    return f"今日雷達判斷：**{tone}**。一般候選 {len(ranking)} 檔，80分以上 {strong} 檔、70–79分 {good} 檔，平均 {avg:.1f} 分。事件雷達 {en} 檔達60分，其中 {estrong} 檔達80分以上。{action}"



@st.cache_data(ttl=300, show_spinner=False)
def chip_event_for_code(code):
    try:
        h=load_hist(str(code))
        return chip_event_v613(h, chip=None)
    except Exception as e:
        return {"籌碼事件":"⚪ 無法確認","判斷信心":0,"倒貨機率":0,"換手機率":0,
                "吸籌機率":0,"洗盤承接機率":0,"大量區防守":"—",
                "判斷原因":f"資料取得失敗：{e}","確認階段":"—"}


@st.cache_data(ttl=300, show_spinner=False)
def overheat_for_code(code):
    try:
        return overheat_score_v614(load_hist(str(code)))
    except Exception as e:
        return {"過熱分數":0,"過熱等級":"⚪ 資料錯誤","過熱原因":str(e),"反轉風險":0}

def build_overheat_ranking(stock_pool, max_scan=120):
    if stock_pool is None or stock_pool.empty: return pd.DataFrame()
    rows=[]
    for _,rr in stock_pool.head(max_scan).iterrows():
        code=str(rr.get("code",""))
        if not code: continue
        rows.append({"code":code,"name":rr.get("name",""),**overheat_for_code(code)})
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["過熱分數"]=pd.to_numeric(out["過熱分數"],errors="coerce").fillna(0)
    return out[out["過熱分數"]>=50].sort_values(["過熱分數","反轉風險"],ascending=False).reset_index(drop=True)


VALIDATION_HISTORY_PATH = APP_DIR / "recommendation_validation_history.csv"
VALIDATION_BACKUP_PATH = APP_DIR / "recommendation_validation_history_backup.csv"

def load_validation_history_all():
    frames=[]
    for p in [VALIDATION_HISTORY_PATH, VALIDATION_BACKUP_PATH]:
        try:
            if p.exists():
                x=pd.read_csv(p,dtype={"code":str})
                if not x.empty: frames.append(x)
        except Exception:
            pass
    if not frames: return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True,sort=False)
    if "snapshot_date" in x.columns and "code" in x.columns:
        x=x.drop_duplicates(["snapshot_date","code"],keep="first")
    return x

def save_recommendation_snapshot(ranking):
    """每天每檔只保存第一次正式推薦，避免刷新覆寫原始推薦。"""
    if ranking is None or ranking.empty: return
    today=pd.Timestamp.today().strftime("%Y-%m-%d")
    x=ranking.copy().head(20)
    x["snapshot_date"]=today
    x["code"]=x["code"].astype(str)
    keep=[c for c in ["snapshot_date","code","name","score","close","setup","score_mode","event_score"] if c in x.columns]
    x=x[keep]
    try:
        old=load_validation_history_all()
        if not old.empty:
            existing=set(zip(old["snapshot_date"].astype(str),old["code"].astype(str)))
            x=x.loc[[(today,str(c)) not in existing for c in x["code"]]]
            merged=pd.concat([old,x],ignore_index=True,sort=False)
        else:
            merged=x
        merged=merged.drop_duplicates(["snapshot_date","code"],keep="first")
        merged.to_csv(VALIDATION_HISTORY_PATH,index=False,encoding="utf-8-sig")
        merged.to_csv(VALIDATION_BACKUP_PATH,index=False,encoding="utf-8-sig")
    except Exception:
        pass

def load_previous_recommendation_snapshot():
    try:
        x=load_validation_history_all()
        if x.empty or "snapshot_date" not in x.columns:
            return pd.DataFrame(), None
        today=pd.Timestamp.today().strftime("%Y-%m-%d")
        dates=sorted([d for d in x["snapshot_date"].dropna().astype(str).unique() if d<today])
        if not dates:
            return pd.DataFrame(), None
        d=dates[-1]  # 前一個有資料的交易日，不硬抓自然日
        return x[x["snapshot_date"].astype(str)==d].copy(), d
    except Exception:
        return pd.DataFrame(), None

def validation_grade(pct, score_change):
    if pct >= 5: return "🔥 強勢符合"
    if pct >= 2: return "🟢 符合"
    if pct > -2 and score_change >= -8: return "🟡 持續觀察"
    if pct <= -5: return "🔴 明顯失敗"
    return "🟠 未符合"

def build_yesterday_validation(prev, live_quotes=None):
    if prev is None or prev.empty: return pd.DataFrame()
    out=prev.copy()
    out["昨日收盤/基準價"]=pd.to_numeric(out.get("close"),errors="coerce")
    if live_quotes is not None and not live_quotes.empty:
        q=live_quotes.copy()
        q["code"]=q["code"].astype(str)
        cols=[c for c in ["code","last","change_pct_live"] if c in q.columns]
        out=out.merge(q[cols].drop_duplicates("code",keep="last"),on="code",how="left")
    else:
        out["last"]=pd.NA
    out["今日價格"]=pd.to_numeric(out.get("last"),errors="coerce")
    out["今日漲跌幅(%)"]=(out["今日價格"]/out["昨日收盤/基準價"]-1)*100

    # 重新計算今日模型分數，才能比較昨日推薦品質是否仍成立
    today_scores=[]; today_modes=[]
    for code in out["code"].astype(str):
        try:
            s=stock_score(load_hist(code))
            today_scores.append(pd.to_numeric(s.get("score"),errors="coerce"))
            today_modes.append(s.get("score_mode",""))
        except Exception:
            today_scores.append(pd.NA); today_modes.append("")
    out["今日模型分數"]=today_scores
    out["昨日推薦分數"]=pd.to_numeric(out.get("score"),errors="coerce")
    out["分數變化"]=out["今日模型分數"]-out["昨日推薦分數"]
    out["符合結果"]=out.apply(lambda r: validation_grade(
        float(r["今日漲跌幅(%)"]) if pd.notna(r["今日漲跌幅(%)"]) else 0,
        float(r["分數變化"]) if pd.notna(r["分數變化"]) else 0),axis=1)
    return out


def forward_performance(code, base_date, base_price, horizons=(1,3,5)):
    """用歷史日線計算推薦後1/3/5個交易日績效、期間最高漲幅與最大回撤。"""
    try:
        h = load_hist(str(code)).copy()
        if h is None or h.empty:
            return {}
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        base_dt = pd.to_datetime(base_date, errors="coerce")
        if pd.isna(base_dt):
            return {}

        after = h[h["date"] > base_dt].copy()
        if after.empty:
            return {}

        base = pd.to_numeric(base_price, errors="coerce")
        if pd.isna(base) or float(base) <= 0:
            return {}
        base = float(base)

        out = {}
        closes = pd.to_numeric(after["close"], errors="coerce")
        highs = pd.to_numeric(after["high"], errors="coerce")
        lows = pd.to_numeric(after["low"], errors="coerce")

        for n in horizons:
            if len(after) >= n:
                px = closes.iloc[n-1]
                out[f"{n}日報酬(%)"] = ((float(px)/base)-1)*100 if pd.notna(px) else pd.NA
            else:
                out[f"{n}日報酬(%)"] = pd.NA

        upto = min(5, len(after))
        if upto > 0:
            hi = highs.iloc[:upto].max()
            lo = lows.iloc[:upto].min()
            out["5日內最高漲幅(%)"] = ((float(hi)/base)-1)*100 if pd.notna(hi) else pd.NA
            out["5日內最大回撤(%)"] = ((float(lo)/base)-1)*100 if pd.notna(lo) else pd.NA
        else:
            out["5日內最高漲幅(%)"] = pd.NA
            out["5日內最大回撤(%)"] = pd.NA

        return out
    except Exception:
        return {}


def multiday_validation_grade(r1, r3, r5):
    vals = [v for v in [r1,r3,r5] if pd.notna(v)]
    if not vals:
        return "⚪ 尚未完成"
    best = max(vals)
    latest = vals[-1]
    if best >= 8 and latest > 0:
        return "🔥 高度符合"
    if best >= 5:
        return "🟢 符合"
    if latest >= 2:
        return "🟢 符合"
    if latest > -2:
        return "🟡 持續觀察"
    if latest <= -5:
        return "🔴 明顯失敗"
    return "🟠 未符合"


def build_multiday_validation(history_df):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in history_df.iterrows():
        code = str(r.get("code",""))
        base_date = r.get("snapshot_date")
        base_price = r.get("close")
        perf = forward_performance(code, base_date, base_price, horizons=(1,3,5))
        row = r.to_dict()
        row.update(perf)
        row["驗證結果"] = multiday_validation_grade(
            row.get("1日報酬(%)"),
            row.get("3日報酬(%)"),
            row.get("5日報酬(%)"),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def validation_summary(df, horizon_col):
    if df is None or df.empty or horizon_col not in df.columns:
        return {"樣本數":0,"命中率":0,"平均報酬":0,"中位數報酬":0}
    s = pd.to_numeric(df[horizon_col], errors="coerce").dropna()
    if s.empty:
        return {"樣本數":0,"命中率":0,"平均報酬":0,"中位數報酬":0}
    # 命中定義：該觀察期報酬 >= 2%
    hit = float((s >= 2).mean() * 100)
    return {
        "樣本數": int(len(s)),
        "命中率": round(hit,1),
        "平均報酬": round(float(s.mean()),2),
        "中位數報酬": round(float(s.median()),2),
    }



@st.cache_data(ttl=1800, show_spinner=False)
def kd_decline_for_code(code):
    try:
        return kd_decline_signal_v617(load_hist(str(code)))
    except Exception:
        return {
            "連跌日數":0, "K值":pd.NA, "D值":pd.NA, "3K-2D":pd.NA,
            "KD狀態":"⚪ 資料錯誤", "符合條件":False, "KD超賣":False,
            "最新收盤":pd.NA
        }

@st.cache_data(ttl=1800, show_spinner=False)
def build_kd_decline_ranking(stock_records):
    rows = []
    for rr in stock_records:
        code = str(rr.get("code","")).strip()
        if not code or not re.fullmatch(r"\d{4}", code):
            continue
        d = kd_decline_for_code(code)
        if d.get("符合條件", False):
            rows.append({"code":code, "name":rr.get("name",""), **d})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["連跌日數"] = pd.to_numeric(out["連跌日數"], errors="coerce").fillna(0)
    out["K值"] = pd.to_numeric(out["K值"], errors="coerce")
    out["D值"] = pd.to_numeric(out["D值"], errors="coerce")
    out["3K-2D"] = pd.to_numeric(out["3K-2D"], errors="coerce")

    # V6.18.1：3K-2D 越負排名越前；連跌日數作為次排序
    out["連跌3日加強"] = out["連跌日數"] >= 3
    return out.sort_values(
        ["3K-2D","連跌日數","K值"], ascending=[True,False,True]
    ).reset_index(drop=True)


KD_CACHE_PATH = Path("/tmp/warrant_radar_kd_cache.csv")
VALIDATION_CACHE_PATH = Path("/tmp/warrant_radar_validation_cache.csv")

def load_kd_cached_result():
    x = st.session_state.get("kd_decline_cached")
    if isinstance(x, pd.DataFrame):
        return x
    try:
        if KD_CACHE_PATH.exists():
            x = pd.read_csv(KD_CACHE_PATH, dtype={"code":str})
            st.session_state["kd_decline_cached"] = x
            return x
    except Exception:
        pass
    return pd.DataFrame()

def save_kd_cached_result(df):
    st.session_state["kd_decline_cached"] = df if df is not None else pd.DataFrame()
    st.session_state["kd_decline_cached_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        st.session_state["kd_decline_cached"].to_csv(KD_CACHE_PATH,index=False,encoding="utf-8-sig")
    except Exception:
        pass

def fast_kd_prefilter(all_stocks, max_candidates=350):
    if all_stocks is None or all_stocks.empty:
        return []
    x = all_stocks.copy()
    x["code"] = x["code"].astype(str).str.strip()
    x = x[x["code"].str.fullmatch(r"\d{4}",na=False)].copy()
    for c in ["volume_lots","turnover","change"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c],errors="coerce")
    if "change" in x.columns and x["change"].notna().any():
        x = x[x["change"].fillna(0) <= 0].copy()
    x["_a"] = x["turnover"].fillna(0).rank(pct=True) if "turnover" in x.columns else 0
    x["_b"] = x["volume_lots"].fillna(0).rank(pct=True) if "volume_lots" in x.columns else 0
    x["_s"] = x["_a"]*.65 + x["_b"]*.35
    return x.sort_values("_s",ascending=False).head(max_candidates)[["code","name"]].to_dict("records")

def run_kd_scan_fast(all_stocks, max_candidates=350):
    return build_kd_decline_ranking(fast_kd_prefilter(all_stocks,max_candidates))

def load_validation_cached_result():
    x = st.session_state.get("validation_cached")
    if isinstance(x,pd.DataFrame):
        return x
    try:
        if VALIDATION_CACHE_PATH.exists():
            x = pd.read_csv(VALIDATION_CACHE_PATH,dtype={"code":str})
            st.session_state["validation_cached"] = x
            return x
    except Exception:
        pass
    return pd.DataFrame()

def save_validation_cached_result(df):
    st.session_state["validation_cached"] = df if df is not None else pd.DataFrame()
    try:
        st.session_state["validation_cached"].to_csv(VALIDATION_CACHE_PATH,index=False,encoding="utf-8-sig")
    except Exception:
        pass


cfg = load_cfg()
st.title("📡 個人版台股權證雷達 V6.18.1")
st.caption("V6.18.1｜已加入模組版本不同步保護，避免單一功能匯入失敗造成整個 App 無法啟動。")
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
    _event_track = update_event_tracking(ranking, keep_days=5)
    ranking = merge_event_tracking(ranking, _event_track, keep_days=5)

save_last_ranking(ranking)

# V6.18.1：獨立事件掃描，不經過權證活躍度預篩
event_pool = build_event_pool(
    all_stocks,
    min_stock_volume=min_stock_volume,
    min_turnover_m=min_turnover_m,
    max_candidates=120
)
with st.spinner("掃描全市場事件型機會…"):
    event_ranking = event_dynamic_ranking(event_pool)

with st.spinner("掃描市場過熱股票…"):
    overheat_ranking = build_overheat_ranking(event_pool, max_scan=120)

# V6.18.1：KD 改為手動觸發。

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
    "trend_score": pd.NA,
    "event_score": pd.NA,
    "score_mode": "",
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
    _evt_saved = ranking["candidate_status"].eq("事件追蹤保留")
    ranking.loc[_evt_saved, "候選等級"] = "🔥 事件追蹤"

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

    # V6.18.1 事件雷達前10名
    if "event_ranking" in globals() and event_ranking is not None and not event_ranking.empty:
        _codes += event_ranking.head(10)["code"].astype(str).tolist()

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
_event80 = int((pd.to_numeric(event_ranking["event_score"], errors="coerce").fillna(0) >= 80).sum()) if event_ranking is not None and not event_ranking.empty else 0
c3.metric("80分以上事件機會", _event80)
c4.metric("趨勢回檔/突破", int(valid["setup"].isin(["趨勢回檔","突破型"]).sum()))

st.markdown("### 🧭 今日市場分析")
st.info(market_summary_text(ranking, event_ranking))

# V6.18.1 每日推薦驗證快照
_previous_recs, _previous_rec_date = load_previous_recommendation_snapshot()
save_recommendation_snapshot(ranking)

TAB_TODAY, TAB_EVENT, TAB_OVERHEAT, TAB_KD, TAB_VALIDATE, TAB_NEXT, TAB_WATCH, TAB_YDAY, TAB_WARRANT, TAB_SETTINGS = st.tabs(
    ["🔥 今日推薦","⚡ 事件型機會","🌡️ 過熱雷達","📉 連跌＋3K-2D","📊 昨日推薦驗證","🔭 NEXT 潛在標的","⭐ 我的關注股","⏪ 昨日追蹤","🎯 權證排行","⚙️ 模型說明"])

with TAB_TODAY:
    st.subheader("今日推薦")
    recommended = ranking[
        (pd.to_numeric(ranking["score"], errors="coerce").fillna(0) >= 70) &
        (~ranking["setup"].isin(["過熱・不追","弱勢觀察","資料錯誤","資料不足","資料待確認"])) &
        (~ranking["candidate_status"].isin(["前次高分保留","事件追蹤保留"]))
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

with TAB_EVENT:
    st.subheader("⚡ 事件型機會")
    st.caption(
        "V6.18.1 事件雷達獨立掃描全市場高流動性股票，不先經過權證活躍度門檻。"
        "事件分數 ≥60 即列入：60–69觀察、70–79推薦、80+強力事件機會。"
    )

    if event_ranking is None or event_ranking.empty:
        st.info("目前沒有事件分數達 60 分以上的股票。")
    else:
        _evt = event_ranking.copy().head(15)

        # 補盤中報價
        if intraday_mode:
            try:
                _ecodes = _evt["code"].astype(str).tolist()
                _elive = load_live_quotes(tuple(_ecodes))
                if _elive is not None and not _elive.empty:
                    _cols = [c for c in [
                        "code","last","bid","ask","volume_live",
                        "change_pct_live","quote_date","quote_time"
                    ] if c in _elive.columns]
                    _evt = _evt.merge(
                        _elive[_cols].drop_duplicates("code", keep="last"),
                        on="code", how="left"
                    )
            except Exception:
                pass

        _evt["候選等級"] = _evt["event_score"].map(event_grade)
        _evt["candidate_status"] = "事件獨立掃描"
        _evt["盤中判斷"] = _evt.apply(
            lambda r: live_signal_text(
                float(pd.to_numeric(r.get("event_score"), errors="coerce") or 0),
                "事件型機會",
                pd.to_numeric(r.get("change_pct_live"), errors="coerce")
            ),
            axis=1
        )

        # V6.18.1 籌碼事件辨識
        _chip_events = _evt["code"].astype(str).map(chip_event_for_code)
        _evt["籌碼事件"] = _chip_events.map(lambda d: d.get("籌碼事件","⚪ 無法確認"))
        _evt["籌碼判斷信心"] = _chip_events.map(lambda d: d.get("判斷信心",0))
        _evt["大量區防守"] = _chip_events.map(lambda d: d.get("大量區防守","—"))
        _evt["籌碼事件原因"] = _chip_events.map(lambda d: d.get("判斷原因","—"))

        # 自訂事件表
        _eshow = pd.DataFrame({
            "股票": _evt["code"].astype(str).map(stock_label),
            "事件分數": _evt["event_score"],
            "趨勢分數": _evt["trend_score"],
            "事件等級": _evt["event_score"].map(event_grade),
            "資料信心(%)": _evt.get("data_confidence", pd.NA),
            "盤中判斷": _evt.get("盤中判斷", "—"),
            "盤中成交價": _evt.get("last", pd.NA),
            "盤中漲跌幅(%)": _evt.get("change_pct_live", pd.NA),
            "事件衝擊": _evt.get("event_shock", pd.NA),
            "洗盤品質": _evt.get("event_washout", pd.NA),
            "止穩": _evt.get("event_stabilization", pd.NA),
            "反彈確認": _evt.get("event_rebound", pd.NA),
            "事件風險": _evt.get("event_risk", pd.NA),
            "事件型原因": _evt.apply(event_reason_text, axis=1),
            "籌碼事件": _evt["籌碼事件"],
            "籌碼判斷信心(%)": _evt["籌碼判斷信心"],
            "大量區防守": _evt["大量區防守"],
            "籌碼事件原因": _evt["籌碼事件原因"],
            "評分重點": _evt.get("score_reason", "—"),
        })

        for _c in _eshow.columns:
            if _c != "事件分數":
                _eshow[_c] = _eshow[_c].where(pd.notna(_eshow[_c]), "—")

        st.dataframe(
            _eshow,
            width="stretch",
            hide_index=True,
            column_config={
                "事件分數": st.column_config.ProgressColumn(
                    "事件分數", min_value=0, max_value=100, format="%.1f"
                )
            }
        )

        _ecodes = _evt["code"].astype(str).tolist()
        _chosen_evt = st.selectbox(
            "查看事件型個股",
            _ecodes,
            format_func=stock_label,
            key="event_stock_v610"
        )
        _er = _evt[_evt["code"].astype(str) == str(_chosen_evt)].iloc[0]

        def _num0(v):
            vv = pd.to_numeric(v, errors="coerce")
            return float(vv) if pd.notna(vv) else 0.0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("事件分數", f"{_num0(_er.get('event_score')):.1f}")
        c2.metric("趨勢分數", f"{_num0(_er.get('trend_score')):.1f}")
        c3.metric("事件等級", event_grade(_num0(_er.get("event_score"))))
        c4.metric("資料信心", f"{_num0(_er.get('data_confidence')):.0f}%")

        st.success(f"**為什麼是事件型機會：** {event_reason_text(_er)}")
        _ce = chip_event_for_code(str(_chosen_evt))
        st.markdown("#### 🧠 籌碼事件辨識")
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("籌碼事件", _ce.get("籌碼事件","—"))
        cc2.metric("判斷信心", f"{float(_ce.get('判斷信心',0)):.0f}%")
        cc3.metric("倒貨機率", f"{float(_ce.get('倒貨機率',0)):.0f}%")
        cc4.metric("換手機率", f"{float(_ce.get('換手機率',0)):.0f}%")
        cc5,cc6,cc7 = st.columns(3)
        cc5.metric("吸籌機率", f"{float(_ce.get('吸籌機率',0)):.0f}%")
        cc6.metric("洗盤承接機率", f"{float(_ce.get('洗盤承接機率',0)):.0f}%")
        cc7.metric("大量區防守", _ce.get("大量區防守","—"))
        st.caption(f"判斷原因：{_ce.get('判斷原因','—')}｜{_ce.get('確認階段','—')}")
        st.info(f"評分重點：{_er.get('score_reason','—')}")

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("事件衝擊", f"{_num0(_er.get('event_shock')):.0f}/20")
        d2.metric("洗盤品質", f"{_num0(_er.get('event_washout')):.0f}/20")
        d3.metric("止穩", f"{_num0(_er.get('event_stabilization')):.0f}/20")
        d4.metric("反彈確認", f"{_num0(_er.get('event_rebound')):.0f}/20")
        d5.metric("事件風險", f"{_num0(_er.get('event_risk')):.0f}/20")

with TAB_KD:
    st.subheader("📉 全市場 3K-2D 負值排行")
    st.caption("效能模式：不在App啟動時掃全市場。按按鈕才掃描，結果會快取。")

    _k1,_k2 = st.columns([2,1])
    with _k1:
        kd_scan_size = st.slider("本次掃描候選數",150,600,350,50)
    with _k2:
        st.write("")
        st.write("")
        kd_run = st.button("🔎 掃描 3K-2D",use_container_width=True)

    if kd_run:
        with st.spinner(f"正在計算約 {kd_scan_size} 檔候選的3K-2D…"):
            _res = run_kd_scan_fast(all_stocks,kd_scan_size)
            save_kd_cached_result(_res)

    kd_decline_ranking = load_kd_cached_result()
    if kd_decline_ranking is None or kd_decline_ranking.empty:
        st.info("尚未有3K-2D快取結果，請按『🔎 掃描 3K-2D』。")
    else:
        _kd=kd_decline_ranking.copy()
        _kdshow=pd.DataFrame({
            "股票":_kd["code"].astype(str).map(stock_label),
            "3K-2D":pd.to_numeric(_kd["3K-2D"],errors="coerce"),
            "K值":pd.to_numeric(_kd["K值"],errors="coerce"),
            "D值":pd.to_numeric(_kd["D值"],errors="coerce"),
            "連跌日數":pd.to_numeric(_kd["連跌日數"],errors="coerce"),
            "連跌≥3日":pd.to_numeric(_kd["連跌日數"],errors="coerce").map(lambda x:"⭐ 是" if pd.notna(x) and x>=3 else "—"),
            "KD狀態":_kd.get("KD狀態","—"),
            "最新收盤":pd.to_numeric(_kd.get("最新收盤"),errors="coerce")
        })
        st.dataframe(_kdshow,width="stretch",hide_index=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("負值股票數",len(_kd))
        c2.metric("3K-2D≤-10",int((pd.to_numeric(_kd["3K-2D"],errors="coerce")<=-10).sum()))
        c3.metric("連跌≥3日",int((pd.to_numeric(_kd["連跌日數"],errors="coerce")>=3).sum()))
        c4.metric("連跌≥5日",int((pd.to_numeric(_kd["連跌日數"],errors="coerce")>=5).sum()))

with TAB_VALIDATE:
    st.subheader("📊 推薦績效驗證｜1日・3日・5日")
    st.caption("效能模式：只有按『更新績效驗證』才重新計算歷史資料。")

    if st.button("🔄 更新績效驗證",use_container_width=True):
        with st.spinner("正在更新推薦績效…"):
            _vhist=load_validation_history_all()
            _new=build_multiday_validation(_vhist) if not _vhist.empty else pd.DataFrame()
            save_validation_cached_result(_new)

    _mv=load_validation_cached_result()
    if _mv is None or _mv.empty:
        st.info("尚未有績效快取，請按『🔄 更新績效驗證』。")
    else:
        _mv["snapshot_date"]=_mv["snapshot_date"].astype(str)
        _dates=sorted(_mv["snapshot_date"].dropna().unique(),reverse=True)
        _latest_date=_dates[0] if _dates else ""
        _latest=_mv[_mv["snapshot_date"]==_latest_date].copy()
        st.markdown(f"### 最近一批推薦：{_latest_date}")

        def _col(df,name):
            return df[name] if name in df.columns else pd.Series(pd.NA,index=df.index)

        _show=pd.DataFrame({
            "股票":_latest["code"].astype(str).map(stock_label),
            "推薦分數":pd.to_numeric(_col(_latest,"score"),errors="coerce"),
            "基準價":pd.to_numeric(_col(_latest,"close"),errors="coerce"),
            "1日報酬(%)":pd.to_numeric(_col(_latest,"1日報酬(%)"),errors="coerce"),
            "3日報酬(%)":pd.to_numeric(_col(_latest,"3日報酬(%)"),errors="coerce"),
            "5日報酬(%)":pd.to_numeric(_col(_latest,"5日報酬(%)"),errors="coerce"),
            "5日內最高漲幅(%)":pd.to_numeric(_col(_latest,"5日內最高漲幅(%)"),errors="coerce"),
            "5日內最大回撤(%)":pd.to_numeric(_col(_latest,"5日內最大回撤(%)"),errors="coerce"),
            "驗證結果":_col(_latest,"驗證結果")
        })
        st.dataframe(_show,width="stretch",hide_index=True)
        _s1=validation_summary(_mv,"1日報酬(%)")
        _s3=validation_summary(_mv,"3日報酬(%)")
        _s5=validation_summary(_mv,"5日報酬(%)")
        a,b,c=st.columns(3)
        a.metric("1日命中率",f"{_s1['命中率']:.1f}%")
        b.metric("3日命中率",f"{_s3['命中率']:.1f}%")
        c.metric("5日命中率",f"{_s5['命中率']:.1f}%")

with TAB_OVERHEAT:
    st.subheader("🌡️ 過熱股票雷達")
    st.caption("不是單看RSI；同時計算乖離、布林位置、短期漲速、爆量、K棒疲態、波動與OBV背離。50分以上列入。")
    if overheat_ranking is None or overheat_ranking.empty:
        st.info("目前掃描範圍沒有過熱分數達50分的股票。")
    else:
        _oh=overheat_ranking.head(20).copy()
        _ohshow=pd.DataFrame({
            "股票":_oh["code"].astype(str).map(stock_label),"過熱分數":_oh["過熱分數"],
            "過熱等級":_oh["過熱等級"],"反轉風險(%)":_oh["反轉風險"],
            "RSI":_oh.get("RSI",pd.NA),"MA20乖離(%)":_oh.get("MA20乖離(%)",pd.NA),
            "量比":_oh.get("量比",pd.NA),"近5日漲幅(%)":_oh.get("近5日漲幅(%)",pd.NA),
            "近10日漲幅(%)":_oh.get("近10日漲幅(%)",pd.NA),"為什麼過熱":_oh["過熱原因"]})
        st.dataframe(_ohshow,width="stretch",hide_index=True)
        _ohcodes=_oh["code"].astype(str).tolist()
        _ohc=st.selectbox("查看過熱股票",_ohcodes,format_func=stock_label,key="overheat_stock")
        _ohr=_oh[_oh["code"].astype(str)==str(_ohc)].iloc[0]
        a,b,c,d=st.columns(4)
        a.metric("過熱分數",f"{float(_ohr['過熱分數']):.0f}")
        b.metric("過熱等級",_ohr["過熱等級"])
        c.metric("反轉風險",f"{float(_ohr['反轉風險']):.0f}%")
        d.metric("RSI",f"{float(_ohr.get('RSI',0)):.1f}")
        st.warning(f"**過熱原因：** {_ohr['過熱原因']}")
        st.caption("過熱 ≠ 一定下跌。強勢股可長時間維持過熱，因此反轉風險會另外評估高檔爆量轉弱、長上影與OBV背離。")

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
### V6.18.1 雙軌＋獨立事件雷達
每檔股票同時計算 **趨勢分數** 與 **事件分數**。一般突破／回檔使用趨勢模型；
爆量急跌、恐慌洗盤、重大事件後止穩反彈，則由事件模型接手。

事件型模型包含：事件衝擊、洗盤品質、止穩、反彈確認、事件風險五部分，共100分。
事件雷達現在獨立掃描全市場高流動性股票，不先經過權證門檻。事件分數≥60就顯示；80分以上為強力事件機會。
""")
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


# ===== V6.18.1 籌碼評分說明 =====
with st.expander("🏦 V6.18.1 法人／大戶籌碼評分", expanded=False):
    st.markdown("""
**新增籌碼觀測（最高 20 分）**

- 外資：近 1 / 3 / 5 日買賣超與連續買賣方向
- 投信：近 1 / 3 / 5 日買賣超，連續買超權重較高
- 自營商：作為輔助訊號
- 大戶：公開資料可取得時納入持股集中度變化
- 籌碼背離：股價下跌但外資＋投信買超加分；股價上漲但法人賣超扣分

**綜合分數原則：原策略最高約 80%＋籌碼確認最高約 20%。**
若當日法人/大戶資料尚未取得，系統不會把缺值當成賣超，也不會因此錯誤扣分。
""")


with st.expander("🧠 V6.18.1 籌碼事件辨識說明", expanded=False):
    st.markdown("""
系統會把爆量事件分成 **疑似大戶倒貨、籌碼換手、疑似大戶吸籌、洗盤後承接、無法確認**。
判斷依據包含爆量程度、K棒收盤位置、上下影線、OBV、MA20、大量區是否守住，以及可取得時的法人方向。

**重要：這是機率式判讀，不代表能直接知道真正下單者身分。**
T日屬初判，後續 T+1～T+3 若大量區守住、量能收斂或法人方向確認，可信度才會提高。
""")

with st.expander("🌡️ V6.18.1 過熱判斷說明", expanded=False):
    st.markdown("""
V6.18.1 使用 **RSI、MA20乖離、布林帶位置、3/5/10日漲速、成交量異常、K棒長上影、ATR波動擴張、OBV量價背離** 交叉判斷。
分級：🔴80+極度過熱、🟠65–79明顯過熱、🟡50–64偏熱觀察、🟢0–49尚未過熱。

另外獨立計算「反轉風險」，因為過熱不等於立即反轉；真正需要提高警戒的是過熱同時出現爆量滯漲、長上影、OBV背離或急漲後轉弱。
""")

with st.expander("📊 V6.18.1 昨日推薦驗證說明", expanded=False):
    st.markdown("""
系統從 V6.18.1 起每天保存推薦快照，下一個有資料的交易日自動比對：
**昨日推薦分數、今日模型分數、分數變化、昨日基準價、今日價格、今日漲跌幅與符合結果。**

目前「今日符合值」定義為：**昨日推薦股中，今日相對昨日基準價上漲至少 2% 的比例**。
這個值主要用來驗證短線推薦的命中情況；後續累積資料後，可再增加 3日、5日最高報酬、最大回撤與整體歷史勝率，會比只看隔日漲跌更公平。
""")


with st.expander("📊 V6.18.1 多日績效驗證說明", expanded=False):
    st.markdown("""
### 為什麼要看 1 / 3 / 5 日？
單看隔日漲跌很容易錯判模型。例如事件股可能隔日整理，但第3～5日才真正反彈。

因此 V6.18.1 同時追蹤：
- **1日報酬**
- **3日報酬**
- **5日報酬**
- **5日內最高漲幅**
- **5日內最大回撤**
- **1 / 3 / 5 日命中率**
- **平均與中位數報酬**

目前命中定義為該觀察期報酬 **≥ +2%**。後續累積樣本後，可以再依「事件型、趨勢型、過熱型」拆開勝率，進一步反向優化評分權重。
""")


with st.expander("📉 V6.18.1 3K-2D弱勢雷達說明", expanded=False):
    st.markdown("""
### 正式條件
- **連續收盤下跌 ≥ 3 個交易日**
- **3K－2D < 0**

其中 **3K－2D** 是把KD進一步放大的短線動能值，因此確實可能出現負數。
數值越低，代表短線弱勢程度越深。

另外標示：
- K、D 是否同時 ≤20
- 連跌日數
- 3K－2D 是否低於 -10

這個雷達是用來找「已經連跌且短線動能非常弱」的股票，並不是直接買進訊號。
若後續再搭配 **止跌、量縮、長下影、KD翻揚、法人回補或事件型承接**，才更適合觀察反彈機會。
""")

with st.expander("💾 V6.18.1 推薦歷史保存說明", expanded=False):
    st.markdown("""
V6.18.1 改為每天每檔保留**第一次正式推薦**，盤中刷新不覆寫原始分數與基準價，並同步寫入主檔與備援檔。

注意：Streamlit Cloud 的執行磁碟不是永久資料庫。若重新部署，要真正延續舊歷史，需把既有的
`recommendation_validation_history.csv` 一併保留在專案中；本版會自動合併主檔與備援檔並去重。
""")
