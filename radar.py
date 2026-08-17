from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("date")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = _num(x[c])
    for n in [5, 10, 20, 60]:
        x[f"ma{n}"] = x["close"].rolling(n).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["ret20"] = x["close"].pct_change(20)
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))
    x["dist_ma20"] = x["close"] / x["ma20"] - 1
    x["range20_high"] = x["high"].rolling(20).max()
    x["range20_low"] = x["low"].rolling(20).min()
    x["volume_ratio"] = x["volume"] / x["vol_ma20"]
    return x


def _clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


def stock_score(hist: pd.DataFrame) -> Dict[str, float | str]:
    if hist is None or len(hist) < 22:
        return {"score": 0, "setup": "資料不足", "trend": 0, "volume_price": 0,
                "position": 0, "momentum": 0, "event": 0, "fundamental_proxy": 0}
    x = add_indicators(hist)
    r = x.iloc[-1]
    prev = x.iloc[-2]

    close = float(r.close)
    ma5, ma10, ma20 = r.ma5, r.ma10, r.ma20
    ma60 = r.ma60 if not pd.isna(r.ma60) else ma20
    volr = 1 if pd.isna(r.volume_ratio) else float(r.volume_ratio)
    ret1 = 0 if pd.isna(r.ret1) else float(r.ret1)
    ret5 = 0 if pd.isna(r.ret5) else float(r.ret5)
    ret20 = 0 if pd.isna(r.ret20) else float(r.ret20)
    rsi = 50 if pd.isna(r.rsi14) else float(r.rsi14)
    dist20 = 0 if pd.isna(r.dist_ma20) else float(r.dist_ma20)

    trend = 5.0
    trend += 4 if close > ma20 else 0
    trend += 3 if ma5 > ma10 else 0
    trend += 3 if ma10 > ma20 else 0
    trend += 3 if ma20 > ma60 else 0
    trend += 2 if ret20 > 0 else 0
    trend = _clamp(trend, 0, 20)

    volume_price = 7.0
    if volr >= 2.5:
        volume_price += 7
    elif volr >= 1.5:
        volume_price += 5
    elif volr >= 1.1:
        volume_price += 2
    if ret1 <= -0.07 and volr >= 1.8:
        volume_price += 6  # event washout candidate
    elif -0.04 <= ret1 <= 0.04 and volr < 0.9 and close >= ma20:
        volume_price += 5  # quiet pullback
    elif ret1 > 0.03 and volr >= 1.5:
        volume_price += 5  # breakout
    volume_price = _clamp(volume_price, 0, 20)

    position = 8.0
    if -0.04 <= dist20 <= 0.03:
        position += 7
    elif -0.08 <= dist20 < -0.04:
        position += 4
    elif dist20 > 0.12:
        position -= 6
    if 35 <= rsi <= 65:
        position += 2
    elif rsi > 78:
        position -= 4
    position = _clamp(position, 0, 15)

    momentum = 7.0
    if ret5 > 0 and ret20 > 0:
        momentum += 5
    if ret1 < 0 and ret5 > 0:
        momentum += 2
    if rsi > 80:
        momentum -= 5
    momentum = _clamp(momentum, 0, 15)

    event = 5.0
    if abs(ret1) >= 0.07 and volr >= 1.8:
        event += 10
    elif abs(ret1) >= 0.04 and volr >= 1.4:
        event += 7
    elif volr >= 2:
        event += 5
    event = _clamp(event, 0, 15)

    # Without a paid fundamental feed, this rewards sustained medium-term trend and avoids pretending
    # that price data equals financial statements. Users can later replace this module.
    fundamental_proxy = 7.0
    if ret20 > 0:
        fundamental_proxy += 4
    if ma20 > ma60:
        fundamental_proxy += 3
    if ret20 < -0.20:
        fundamental_proxy -= 4
    fundamental_proxy = _clamp(fundamental_proxy, 0, 15)

    score = trend + volume_price + position + momentum + event + fundamental_proxy

    # Setup classifier
    recent_high = float(x["high"].iloc[-21:-1].max())
    if ret1 <= -0.07 and volr >= 1.8:
        setup = "事件型反彈"
    elif close > recent_high and volr >= 1.4:
        setup = "突破型"
    elif close >= ma20 * 0.97 and close <= ma20 * 1.035 and volr <= 1.1 and ma20 > ma60:
        setup = "趨勢回檔"
    elif rsi >= 75 or dist20 >= 0.12:
        setup = "過熱・不追"
        score -= 12
    elif close < ma20 and ret20 < 0:
        setup = "弱勢觀察"
        score -= 8
    else:
        setup = "蓄勢/中性"

    score = _clamp(score, 0, 100)
    return {
        "score": round(score, 1), "setup": setup,
        "trend": round(trend, 1), "volume_price": round(volume_price, 1),
        "position": round(position, 1), "momentum": round(momentum, 1),
        "event": round(event, 1), "fundamental_proxy": round(fundamental_proxy, 1),
        "close": round(close, 2), "ret1_pct": round(ret1 * 100, 2),
        "ret5_pct": round(ret5 * 100, 2), "volume_ratio": round(volr, 2),
        "rsi14": round(rsi, 1), "ma20": round(float(ma20), 2),
    }


def moneyness_pct(underlying_price: float, strike: float, warrant_type: str = "CALL") -> float:
    if not underlying_price or not strike:
        return np.nan
    t = str(warrant_type).upper()
    if "PUT" in t or "售" in t:
        return (underlying_price - strike) / strike * 100  # positive means OTM for put
    return (strike - underlying_price) / underlying_price * 100  # positive means OTM for call


def warrant_score(row: pd.Series, underlying_price: float, cfg: Dict) -> Dict[str, float | str | bool]:
    volume = float(row.get("volume", 0) or 0)
    expiry = pd.to_datetime(row.get("expiry"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    dte = (expiry.normalize() - today).days if not pd.isna(expiry) else np.nan
    strike = pd.to_numeric(row.get("strike"), errors="coerce")
    wtype = str(row.get("warrant_type", "CALL")).upper()
    otm = moneyness_pct(underlying_price, strike, wtype) if not pd.isna(strike) else np.nan
    delta = pd.to_numeric(row.get("delta"), errors="coerce")
    eff = pd.to_numeric(row.get("effective_leverage"), errors="coerce")
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    iv = pd.to_numeric(row.get("iv"), errors="coerce")
    price = pd.to_numeric(row.get("price"), errors="coerce")

    hard_ok = True
    reasons = []
    if volume < cfg["min_warrant_volume"]:
        hard_ok = False; reasons.append("成交量不足")
    if not pd.isna(dte) and dte < cfg["min_days_to_expiry"]:
        hard_ok = False; reasons.append("剩餘天數不足")
    if not pd.isna(otm) and otm > cfg["max_otm_pct"]:
        hard_ok = False; reasons.append("價外過深")

    if volume >= 1000: vol_s = 20
    elif volume >= 500: vol_s = 18
    elif volume >= 300: vol_s = 15
    elif volume >= 100: vol_s = 8
    else: vol_s = 2

    if pd.isna(dte): dte_s = 7
    elif dte >= 180: dte_s = 15
    elif dte >= 150: dte_s = 14
    elif dte >= 120: dte_s = 12
    elif dte >= 90: dte_s = 7
    else: dte_s = 2

    if pd.isna(otm): money_s = 8
    elif -5 <= otm <= 5: money_s = 20
    elif 5 < otm <= 10: money_s = 18
    elif 10 < otm <= 15: money_s = 13
    elif -15 <= otm < -5: money_s = 17
    elif otm > 20: money_s = 2
    else: money_s = 8

    if pd.isna(delta): delta_s = 7
    elif cfg["preferred_delta_min"] <= abs(delta) <= cfg["preferred_delta_max"]: delta_s = 15
    elif 0.25 <= abs(delta) <= 0.75: delta_s = 10
    else: delta_s = 4

    if pd.isna(eff): lev_s = 5
    elif cfg["preferred_effective_leverage_min"] <= eff <= cfg["preferred_effective_leverage_max"]: lev_s = 10
    elif 2 <= eff <= 8: lev_s = 7
    else: lev_s = 3

    spread_pct = np.nan
    if not pd.isna(bid) and not pd.isna(ask) and ask > 0:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else np.nan
    if pd.isna(spread_pct): spread_s = 5
    elif spread_pct <= 2: spread_s = 10
    elif spread_pct <= 4: spread_s = 8
    elif spread_pct <= 7: spread_s = 5
    else: spread_s = 1

    if pd.isna(iv): iv_s = 5
    elif iv <= 45: iv_s = 10
    elif iv <= 65: iv_s = 8
    elif iv <= 85: iv_s = 5
    else: iv_s = 2

    total = vol_s + dte_s + money_s + delta_s + lev_s + spread_s + iv_s
    if not hard_ok:
        total = min(total, 59)
    return {
        "warrant_score": round(total, 1), "eligible": hard_ok,
        "days_to_expiry": None if pd.isna(dte) else int(dte),
        "otm_pct": None if pd.isna(otm) else round(float(otm), 2),
        "spread_pct": None if pd.isna(spread_pct) else round(float(spread_pct), 2),
        "filter_reason": "、".join(reasons) if reasons else "通過",
    }


def normalize_warrant_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "warrant_code": ["warrant_code", "權證代號", "代號", "證券代號"],
        "warrant_name": ["warrant_name", "權證名稱", "名稱", "證券名稱"],
        "underlying": ["underlying", "標的代號", "標的證券代號", "標的"],
        "issuer": ["issuer", "發行商", "發行人", "券商"],
        "warrant_type": ["warrant_type", "類型", "權證類型", "認購售"],
        "price": ["price", "成交價", "收盤價", "權證價格"],
        "volume": ["volume", "成交量", "成交數量", "成交量(張)"],
        "strike": ["strike", "履約價", "履約價格"],
        "expiry": ["expiry", "到期日", "最後交易日"],
        "bid": ["bid", "委買", "最佳買價"],
        "ask": ["ask", "委賣", "最佳賣價"],
        "delta": ["delta", "Delta", "DELTA"],
        "iv": ["iv", "IV", "隱含波動率"],
        "effective_leverage": ["effective_leverage", "有效槓桿", "實質槓桿"],
    }
    ren = {}
    for target, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                ren[c] = target
                break
    x = df.rename(columns=ren).copy()
    for c in aliases:
        if c not in x.columns:
            x[c] = np.nan
    x["warrant_code"] = x["warrant_code"].astype(str).str.strip()
    x["underlying"] = x["underlying"].astype(str).str.extract(r"(\d{4,6})", expand=False).fillna(x["underlying"].astype(str))
    for c in ["price", "volume", "strike", "bid", "ask", "delta", "iv", "effective_leverage"]:
        x[c] = pd.to_numeric(x[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
    # Treat Chinese type names
    x["warrant_type"] = x["warrant_type"].fillna("CALL").astype(str).apply(
        lambda v: "PUT" if ("售" in v or "PUT" in v.upper()) else "CALL")
    return x


def rank_warrants(df: pd.DataFrame, underlying: str, underlying_price: float, cfg: Dict,
                  call_only: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = normalize_warrant_columns(df)
    x = x[x["underlying"].astype(str) == str(underlying)].copy()
    if call_only:
        x = x[x["warrant_type"] == "CALL"]
    if x.empty:
        return x
    scored = x.apply(lambda r: warrant_score(r, underlying_price, cfg), axis=1, result_type="expand")
    x = pd.concat([x.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    return x.sort_values(["eligible", "warrant_score", "volume"], ascending=[False, False, False])


def entry_plan(stock_row: Dict, first_amount: float, reserve_amount: float) -> Dict[str, str | float]:
    """Create an entry plan using user-editable first-entry and reserve amounts.

    Total allocation is capped in the UI at NT$500,000. The reserve is not
    automatically deployed just because price falls; deployment rules depend
    on the detected setup.
    """
    p = float(stock_row.get("close", 0) or 0)
    setup = stock_row.get("setup", "")
    first = round(max(0, first_amount))
    reserve = round(max(0, reserve_amount))
    capital = first + reserve
    if capital > 500000:
        return {
            "first": 0, "reserve": 0,
            "instruction": "資金設定超過單一標的 50 萬元上限，請先調整金額。",
            "add_rule": "首筆投入＋預備金合計不得超過 NT$500,000。",
            "stop_rule": "調整完成後才建立交易計畫。"
        }
    if setup == "事件型反彈":
        return {
            "first": first, "reserve": reserve,
            "instruction": f"首筆 {first:,.0f}；預備 {reserve:,.0f}，不因單純下跌一次補滿。",
            "add_rule": f"優先等約 {p*0.96:.1f}～{p*0.98:.1f} 出現止跌，再考慮動用約一半預備金（約 {reserve/2:,.0f}）；更深支撐約 {p*0.92:.1f}～{p*0.95:.1f} 再評估剩餘預備金。",
            "stop_rule": f"若跌破約 {p*0.90:.1f} 且仍放量，停止攤平，重新判斷基本面/事件。"
        }
    if setup == "趨勢回檔":
        return {
            "first": first, "reserve": reserve,
            "instruction": f"首筆 {first:,.0f}；保留 {reserve:,.0f}。",
            "add_rule": "站回短均線或回測20日線不破後再加碼，不用向下攤平；預備金可依確認程度分2次使用。",
            "stop_rule": "有效跌破20日線且量增時降低部位。"
        }
    if setup == "突破型":
        return {
            "first": first, "reserve": reserve,
            "instruction": f"突破確認先投入 {first:,.0f}；保留 {reserve:,.0f}。",
            "add_rule": "突破後回測前高不破再動用預備金；不要在長紅末端追滿。",
            "stop_rule": "跌回突破區並放量時視為假突破。"
        }
    return {
        "first": 0, "reserve": capital,
        "instruction": f"目前不建議主動建立大部位；保留設定資金 {capital:,.0f}。",
        "add_rule": "等待型態轉成趨勢回檔、突破或事件型止跌再進。",
        "stop_rule": "無持倉則不需停損；已有持倉以20日線/前低為風控參考。"
    }

