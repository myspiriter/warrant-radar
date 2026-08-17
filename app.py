from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import twse_stock_history, twse_warrant_daily_volume, merge_warrant_volume
from radar import stock_score, rank_warrants, normalize_warrant_columns, entry_plan, add_indicators

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
HISTORY_PATH = APP_DIR / "signal_history.csv"

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


def load_cfg():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
    rows = []
    for code in watchlist:
        try:
            h = load_hist(code)
            s = stock_score(h)
            rows.append({"code":str(code), **s})
        except Exception as e:
            rows.append({"code":str(code), "score":0, "setup":"資料錯誤", "error":str(e)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


cfg = load_cfg()
st.title("📡 個人版台股權證雷達")
st.caption("先選現股，再選權證。評分是決策輔助，不是保證獲利或自動下單。")

with st.sidebar:
    st.header("篩選設定")
    watch_txt = st.text_area("股票追蹤清單（逗號分隔）", ",".join(cfg["watchlist"]), height=90)
    watchlist = [x.strip() for x in watch_txt.replace("\n", ",").split(",") if x.strip()]
    st.subheader("資金配置")
    first_amount = st.number_input(
        "首筆投入金額", min_value=0, max_value=500000,
        value=int(cfg.get("first_entry_amount", 50000)), step=10000
    )
    reserve_amount = st.number_input(
        "預備金額", min_value=0, max_value=500000,
        value=int(cfg.get("reserve_amount", 50000)), step=10000
    )
    capital = first_amount + reserve_amount
    if capital > 500000:
        st.error("首筆投入＋預備金合計上限為 500,000 元，請調低金額。")
    else:
        st.caption(f"目前單一標的總配置：NT$ {capital:,.0f}／上限 NT$ 500,000")
    min_vol = st.number_input("權證最低成交量（張）", 0, 100000, int(cfg["min_warrant_volume"]), step=50)
    min_dte = st.number_input("最低剩餘天數", 0, 1000, int(cfg["min_days_to_expiry"]), step=10)
    max_otm = st.number_input("最大價外 %", 0.0, 100.0, float(cfg["max_otm_pct"]), step=1.0)
    cfg["min_warrant_volume"] = min_vol
    cfg["min_days_to_expiry"] = min_dte
    cfg["max_otm_pct"] = max_otm
    st.divider()
    st.subheader("權證資料")
    uploaded = st.file_uploader("上傳券商權證 CSV（可選，但強烈建議）", type=["csv"])
    st.caption("若沒有上傳，仍會抓 TWSE 每日權證成交量；但履約價/到期/Delta/IV 等可能不完整。")

with st.spinner("更新追蹤標的資料…"):
    ranking = stock_ranking(watchlist)

save_signal_snapshot(ranking)

# Warrant data
wvol = load_warrant_volume()
if uploaded is not None:
    try:
        wterms = pd.read_csv(uploaded, encoding="utf-8-sig")
    except UnicodeDecodeError:
        wterms = pd.read_csv(uploaded, encoding="big5")
    wterms = normalize_warrant_columns(wterms)
    if wterms["issuer"].isna().all() or (wterms["issuer"].astype(str).str.len() == 0).all():
        wterms["issuer"] = wterms["warrant_name"].map(issuer_from_name)
    warrants = merge_warrant_volume(wterms, wvol)
else:
    warrants = pd.DataFrame()

# Header metrics
c1,c2,c3,c4 = st.columns(4)
valid = ranking[ranking.score > 0]
c1.metric("追蹤標的", len(watchlist))
c2.metric("80分以上", int((valid.score >= 80).sum()))
c3.metric("事件型反彈", int((valid.setup == "事件型反彈").sum()))
c4.metric("趨勢回檔/突破", int(valid.setup.isin(["趨勢回檔","突破型"]).sum()))

TAB_TODAY, TAB_YDAY, TAB_NEXT, TAB_WARRANT, TAB_SETTINGS = st.tabs(["🔥 TODAY", "⏪ YESTERDAY", "🔭 NEXT", "🎯 權證排行", "⚙️ 模型說明"])

with TAB_TODAY:
    st.subheader("今日標的排行")
    show_cols = [c for c in ["code","score","setup","close","ret1_pct","ret5_pct","volume_ratio","rsi14","ma20"] if c in ranking.columns]
    st.dataframe(ranking[show_cols], use_container_width=True, hide_index=True,
                 column_config={"score":st.column_config.ProgressColumn("機會分數", min_value=0,max_value=100,format="%.1f")})

    top_codes = ranking[ranking.score > 0].code.tolist()
    if top_codes:
        chosen = st.selectbox("查看個股", top_codes, key="today_stock")
        sr = ranking[ranking.code == chosen].iloc[0].to_dict()
        h = add_indicators(load_hist(chosen))
        left,right = st.columns([2,1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=h.date, open=h.open, high=h.high, low=h.low, close=h.close, name=chosen))
            fig.add_trace(go.Scatter(x=h.date, y=h.ma20, name="MA20"))
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=35,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(f"### {chosen}｜{sr.get('setup','')}")
            st.metric("現股分數", f"{sr.get('score',0):.1f}/100")
            st.write(f"趨勢 {sr.get('trend',0)}/20｜量價 {sr.get('volume_price',0)}/20")
            st.write(f"位置 {sr.get('position',0)}/15｜動能 {sr.get('momentum',0)}/15")
            st.write(f"事件 {sr.get('event',0)}/15｜基本面代理 {sr.get('fundamental_proxy',0)}/15")
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
                st.dataframe(wr[[c for c in cols if c in wr.columns]].head(5), use_container_width=True, hide_index=True)
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
            yd["至今變化%"] = yd.apply(now_ret, axis=1)
            st.caption(f"訊號日期：{dates[1]}")
            st.dataframe(yd[[c for c in ["code","score","setup","close","至今變化%"] if c in yd.columns]].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("目前只有今天的快照；明天再打開就會開始形成昨日績效。")
    else:
        st.info("尚無歷史快照。")

with TAB_NEXT:
    st.subheader("NEXT｜接近買點但尚未完全觸發")
    nx = ranking[(ranking.score >= 55) & (~ranking.setup.isin(["過熱・不追","弱勢觀察","事件型反彈","突破型"]))].copy()
    if nx.empty:
        st.info("目前追蹤清單沒有明顯 NEXT 標的。")
    else:
        nx["預期觸發"] = nx.apply(lambda r: "放量突破近期高點" if r.setup == "蓄勢/中性" else "回測支撐不破後轉強", axis=1)
        st.dataframe(nx[["code","score","setup","close","ret5_pct","volume_ratio","rsi14","預期觸發"]], use_container_width=True, hide_index=True)

with TAB_WARRANT:
    st.subheader("全權證排行")
    if warrants.empty:
        st.info("請上傳一份券商權證 CSV。只要有代號、標的、成交量、履約價、到期日，就能先做核心排行；Delta/IV/價差越完整越準。")
        if not wvol.empty:
            st.success(f"TWSE 每日權證成交量已成功抓到 {len(wvol):,} 筆。")
            st.dataframe(wvol.head(20), use_container_width=True, hide_index=True)
    else:
        all_ranked=[]
        pmap = ranking.set_index("code")["close"].to_dict()
        for code, p in pmap.items():
            r=rank_warrants(warrants, code, float(p), cfg)
            if not r.empty:
                r["stock_code"] = code
                r["stock_score"] = float(ranking.loc[ranking.code==code,"score"].iloc[0])
                all_ranked.append(r)
        if all_ranked:
            aw=pd.concat(all_ranked, ignore_index=True)
            aw["combo_score"] = aw["stock_score"]*0.45 + aw["warrant_score"]*0.55
            aw=aw.sort_values(["eligible","combo_score"], ascending=[False,False])
            cols=["stock_code","stock_score","warrant_code","warrant_name","issuer","volume","strike","days_to_expiry","otm_pct","delta","effective_leverage","spread_pct","iv","warrant_score","combo_score","eligible","filter_reason"]
            st.dataframe(aw[[c for c in cols if c in aw.columns]].head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("目前權證資料沒有對應追蹤清單內的標的。")

with TAB_SETTINGS:
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
