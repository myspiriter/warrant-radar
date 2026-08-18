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
    """V6.8 多因子技術指標。只使用 OHLCV，避免把價格資料假裝成基本面。"""
    x = df.copy().sort_values("date")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = _num(x[c])

    for n in [5, 10, 20, 60]:
        x[f"ma{n}"] = x["close"].rolling(n).mean()

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)

    # RSI
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    # ATR / volatility
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["atr_pct"] = x["atr14"] / x["close"]
    x["volatility20"] = x["ret1"].rolling(20).std()

    # Bollinger position
    std20 = x["close"].rolling(20).std()
    x["bb_upper"] = x["ma20"] + 2 * std20
    x["bb_lower"] = x["ma20"] - 2 * std20
    width = (x["bb_upper"] - x["bb_lower"]).replace(0, np.nan)
    x["bb_pos"] = (x["close"] - x["bb_lower"]) / width

    # Range / distance
    x["dist_ma20"] = x["close"] / x["ma20"] - 1
    x["range20_high"] = x["high"].rolling(20).max()
    x["range20_low"] = x["low"].rolling(20).min()
    x["range60_high"] = x["high"].rolling(60).max()
    x["range60_low"] = x["low"].rolling(60).min()
    x["from_20_high"] = x["close"] / x["range20_high"] - 1
    x["from_20_low"] = x["close"] / x["range20_low"] - 1

    # Volume / money-flow proxies
    x["volume_ratio"] = x["volume"] / x["vol_ma20"]
    x["volume_ratio5"] = x["volume"] / x["vol_ma5"]
    direction = np.sign(x["close"].diff()).fillna(0)
    x["obv"] = (direction * x["volume"].fillna(0)).cumsum()
    x["obv_ma10"] = x["obv"].rolling(10).mean()

    # Candle quality
    body = (x["close"] - x["open"]).abs()
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = body / rng
    x["close_location"] = (x["close"] - x["low"]) / rng

    return x


def _clamp(v, lo, hi):
    try:
        return float(max(lo, min(hi, v)))
    except Exception:
        return float(lo)


def _safe(v, default=0.0):
    return default if pd.isna(v) else float(v)


def stock_score(hist: pd.DataFrame) -> Dict[str, float | str]:
    """V6.9 雙軌評分：一般趨勢模型 + 事件型模型。"""
    if hist is None or len(hist) < 22:
        return {"score":0, "trend_score":0, "event_score":0, "score_mode":"資料不足",
                "setup":"資料不足", "data_confidence":0, "score_reason":"歷史資料不足"}

    x = add_indicators(hist)
    r, prev = x.iloc[-1], x.iloc[-2]
    close = _safe(r.close); open_ = _safe(r.open, close)
    ma5 = _safe(r.ma5, close); ma10 = _safe(r.ma10, close)
    ma20 = _safe(r.ma20, close); ma60 = _safe(r.ma60, ma20)
    volr = _safe(r.volume_ratio, 1); ret1 = _safe(r.ret1, 0)
    ret3 = _safe(r.ret3, 0); ret5 = _safe(r.ret5, 0)
    ret10 = _safe(r.ret10, 0); ret20 = _safe(r.ret20, 0)
    rsi = _safe(r.rsi14, 50); dist20 = _safe(r.dist_ma20, 0)
    macd_hist = _safe(r.macd_hist, 0); prev_macd_hist = _safe(prev.macd_hist, 0)
    atr_pct = _safe(r.atr_pct, 0.03); vol20 = _safe(r.volatility20, 0.02)
    bb_pos = _safe(r.bb_pos, 0.5); close_loc = _safe(r.close_location, 0.5)
    body_ratio = _safe(r.body_ratio, 0.5); obv = _safe(r.obv, 0)
    obv_ma10 = _safe(r.obv_ma10, obv)
    from20high = _safe(r.from_20_high, 0); from20low = _safe(r.from_20_low, 0)
    recent_high = _safe(x["high"].iloc[-21:-1].max(), close)

    reasons = []
    trend = 2.0 + (4 if close > ma20 else 0) + (3 if ma5 > ma10 else 0) + (3 if ma10 > ma20 else 0) + (3 if ma20 > ma60 else 0) + (2 if ret20 > 0 else 0) + (1 if ma20 > _safe(x["ma20"].iloc[-6], ma20) else 0)
    trend = _clamp(trend, 0, 18)

    volume_price = 4.0
    if 1.1 <= volr < 1.5: volume_price += 2
    elif 1.5 <= volr < 2.5: volume_price += 4
    elif volr >= 2.5: volume_price += 5
    if ret1 > 0 and volr >= 1.2: volume_price += 3
    if abs(ret1) <= 0.02 and volr <= 0.85 and close >= ma20: volume_price += 2
    if ret1 < -0.04 and volr >= 1.8: volume_price -= 2
    if obv > obv_ma10: volume_price += 1
    volume_price = _clamp(volume_price, 0, 14)

    momentum = 4.0 + (1 if ret3 > 0 else 0) + (2 if ret5 > 0 else 0) + (1 if ret10 > 0 else 0) + (2 if macd_hist > 0 else 0) + (1 if macd_hist > prev_macd_hist else 0) + (1 if 45 <= rsi <= 68 else 0) - (3 if rsi > 78 else 0)
    momentum = _clamp(momentum, 0, 12)

    position = 4.0
    if -0.035 <= dist20 <= 0.035: position += 4
    elif -0.07 <= dist20 < -0.035: position += 2
    elif 0.035 < dist20 <= 0.08: position += 2
    if 0.25 <= bb_pos <= 0.75: position += 2
    if from20high > -0.08: position += 1
    if dist20 > 0.14 or bb_pos > 1.05: position -= 4
    if rsi > 80: position -= 2
    position = _clamp(position, 0, 12)

    breakout_quality = 2.0 + (4 if close > recent_high else 0) + (2 if close >= recent_high * 0.985 else 0) + (2 if volr >= 1.4 else 0) + (1 if close_loc >= 0.70 else 0) + (1 if body_ratio >= 0.55 and close >= open_ else 0)
    if close > recent_high and close_loc < 0.45: breakout_quality -= 3
    breakout_quality = _clamp(breakout_quality, 0, 12)

    support = 3.0 + (2 if close >= ma20 * 0.98 else 0) + (1 if close >= ma10 * 0.985 else 0) + (1 if close_loc >= 0.55 else 0) + (2 if ret1 < 0 and close_loc >= 0.65 else 0) + (1 if from20low > 0.08 else 0)
    if close < ma20 * 0.94: support -= 3
    support = _clamp(support, 0, 10)

    risk = 8.0
    if atr_pct > 0.06: risk -= 3
    elif atr_pct > 0.045: risk -= 1
    if vol20 > 0.045: risk -= 2
    if abs(ret1) > 0.08: risk -= 2
    if dist20 > 0.15: risk -= 2
    if rsi > 82: risk -= 2
    risk = _clamp(risk, 0, 10)

    consistency = 4.0
    last10 = x["ret1"].iloc[-10:].dropna()
    if len(last10) >= 8:
        positive_ratio = float((last10 > 0).mean())
        if 0.45 <= positive_ratio <= 0.75: consistency += 2
        elif positive_ratio > 0.80: consistency -= 1
    if ret5 > -0.03: consistency += 1
    if ret20 > -0.08: consistency += 1
    if ma20 > ma60: consistency += 2
    if obv > obv_ma10: consistency += 1
    if abs(ret1) < 0.07: consistency += 1
    consistency = _clamp(consistency, 0, 12)

    trend_score = trend + volume_price + momentum + position + breakout_quality + support + risk + consistency

    # 事件型模型
    last5_ret = x["ret1"].iloc[-5:].dropna()
    last5_volr = x["volume_ratio"].iloc[-5:].dropna()
    max_drop = abs(float(last5_ret.min())) if len(last5_ret) else 0
    max_volr = float(last5_volr.max()) if len(last5_volr) else 1

    shock = 0
    if max_drop >= 0.09: shock += 12
    elif max_drop >= 0.07: shock += 10
    elif max_drop >= 0.05: shock += 7
    elif max_drop >= 0.035: shock += 4
    if max_volr >= 3: shock += 8
    elif max_volr >= 2: shock += 6
    elif max_volr >= 1.5: shock += 4
    shock = _clamp(shock, 0, 20)

    washout = 0
    if close_loc >= 0.65: washout += 7
    elif close_loc >= 0.5: washout += 4
    if ret1 < 0 and close_loc >= 0.65: washout += 5
    if max_drop >= 0.05 and ret1 > -0.03: washout += 4
    if max_volr >= 1.8 and volr < max_volr * 0.7: washout += 4
    washout = _clamp(washout, 0, 20)

    stabilization = 0
    if ret3 > -0.02: stabilization += 5
    if ret5 > -0.06: stabilization += 4
    if close >= ma5 * 0.98: stabilization += 4
    if macd_hist > prev_macd_hist: stabilization += 3
    if 30 <= rsi <= 60: stabilization += 2
    if close_loc >= 0.55: stabilization += 2
    stabilization = _clamp(stabilization, 0, 20)

    rebound = 0
    if ret1 > 0: rebound += 5
    if ret3 > 0: rebound += 4
    if close > ma5: rebound += 3
    if close > ma10: rebound += 2
    if volr >= 1.1 and ret1 > 0: rebound += 3
    if obv > obv_ma10: rebound += 3
    rebound = _clamp(rebound, 0, 20)

    event_risk = 12
    if ret1 < -0.07: event_risk -= 5
    if close < ma20 * 0.9: event_risk -= 3
    if atr_pct > 0.07: event_risk -= 3
    if vol20 > 0.06: event_risk -= 2
    if rsi < 20: event_risk -= 2
    if ret3 > -0.03: event_risk += 3
    if close_loc >= 0.55: event_risk += 2
    event_risk = _clamp(event_risk, 0, 20)

    event_score = shock + washout + stabilization + rebound + event_risk
    event_active = ((max_drop >= 0.045 and max_volr >= 1.4) or (shock >= 12 and stabilization >= 10))

    if close > recent_high and volr >= 1.35 and breakout_quality >= 8:
        normal_setup = "突破型"
    elif close >= ma20 * 0.97 and close <= ma20 * 1.04 and volr <= 1.15 and ma20 >= ma60:
        normal_setup = "趨勢回檔"
    elif rsi >= 78 or dist20 >= 0.14:
        normal_setup = "過熱・不追"; trend_score -= 10
    elif close < ma20 * 0.96 and ret20 < 0:
        normal_setup = "弱勢觀察"; trend_score -= 8
    else:
        normal_setup = "蓄勢/中性"

    trend_score = _clamp(trend_score, 0, 100)
    event_score = _clamp(event_score, 0, 100)

    if event_active and event_score >= trend_score + 6:
        score = event_score; score_mode = "事件型模型"; setup = "事件型機會"
        reason = f"事件衝擊{shock:.0f}/20｜洗盤{washout:.0f}/20｜止穩{stabilization:.0f}/20｜反彈{rebound:.0f}/20"
    else:
        score = trend_score; score_mode = "趨勢型模型"; setup = normal_setup
        reason = f"趨勢{trend:.0f}/18｜量價{volume_price:.0f}/14｜動能{momentum:.0f}/12｜突破{breakout_quality:.0f}/12"

    critical = [r.ma20, r.rsi14, r.volume_ratio, r.macd_hist, r.atr_pct, r.bb_pos]
    known = sum(0 if pd.isna(v) else 1 for v in critical)
    data_confidence = round(known / len(critical) * 100, 1)

    return {
        "score":round(score,1), "trend_score":round(trend_score,1), "event_score":round(event_score,1),
        "score_mode":score_mode, "setup":setup, "trend":round(trend,1),
        "volume_price":round(volume_price,1), "momentum":round(momentum,1), "position":round(position,1),
        "breakout_quality":round(breakout_quality,1), "support":round(support,1),
        "risk":round(risk,1), "consistency":round(consistency,1),
        "event_shock":round(shock,1), "event_washout":round(washout,1),
        "event_stabilization":round(stabilization,1), "event_rebound":round(rebound,1),
        "event_risk":round(event_risk,1), "event_active":bool(event_active),
        "data_confidence":data_confidence, "score_reason":reason,
        "close":round(close,4), "ret1_pct":round(ret1*100,4), "ret5_pct":round(ret5*100,4),
        "ret20_pct":round(ret20*100,4), "volume_ratio":round(volr,3),
        "rsi14":round(rsi,1), "ma20":round(ma20,4), "atr_pct":round(atr_pct*100,3),
        "macd_hist":round(macd_hist,4),
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
        "issuer": ["issuer", "發行券商", "發行商", "發行人", "發行證券商名稱", "券商"],
        "warrant_type": ["warrant_type", "類型", "權證類型", "認購售", "認購（售）別"],
        "price": ["price", "成交價", "收盤價", "最後成交價", "權證價格"],
        "volume": ["volume", "成交量", "成交數量", "成交量(張)"],
        "strike": ["strike", "履約價", "履約價格", "最新履約價格"],
        "expiry": ["expiry", "到期日", "到期日期", "權證到期日", "最後交易日"],
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
    x["warrant_name"] = x["warrant_name"].fillna("").astype(str).str.strip()
    x["underlying"] = x["underlying"].astype(str).str.extract(
        r"(\d{4,6})", expand=False
    ).fillna(x["underlying"].astype(str))

    for c in ["price","volume","strike","bid","ask","delta","iv","effective_leverage"]:
        x[c] = pd.to_numeric(
            x[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce"
        )

    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce")
    x["warrant_type"] = x["warrant_type"].fillna("CALL").astype(str).apply(
        lambda v: "PUT" if ("售" in v or "PUT" in v.upper()) else "CALL"
    )

    issuers = ["元大","凱基","群益","永豐","富邦","元富","統一","國票","兆豐","中信",
               "玉山","第一金","華南永昌","康和","國泰","台新","合庫","宏遠","亞東"]
    def infer_issuer(row):
        current = str(row.get("issuer") or "").strip()
        if current and current.lower() not in ("nan","none","<na>"):
            return current
        name = str(row.get("warrant_name") or "")
        for i in issuers:
            if i in name:
                return i
        return ""
    x["issuer"] = x.apply(infer_issuer, axis=1)
    return x

def rank_warrants(df: pd.DataFrame, underlying: str, spot: float, cfg: dict) -> pd.DataFrame:
    x = normalize_warrant_columns(df)
    underlying = str(underlying)
    x = x[x["underlying"].astype(str) == underlying].copy()
    if x.empty:
        return x

    # 認購為主
    x = x[x["warrant_type"].fillna("CALL").eq("CALL")].copy()
    if x.empty:
        return x

    today = pd.Timestamp.today().normalize()
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce")
    x["days_to_expiry"] = (x["expiry"] - today).dt.days

    x["strike"] = pd.to_numeric(x["strike"], errors="coerce")
    if spot and spot > 0:
        x["otm_pct"] = (x["strike"] / float(spot) - 1.0) * 100.0
    else:
        x["otm_pct"] = np.nan

    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0)
    x["bid"] = pd.to_numeric(x["bid"], errors="coerce")
    x["ask"] = pd.to_numeric(x["ask"], errors="coerce")
    x["delta"] = pd.to_numeric(x["delta"], errors="coerce")
    x["iv"] = pd.to_numeric(x["iv"], errors="coerce")
    x["effective_leverage"] = pd.to_numeric(x["effective_leverage"], errors="coerce")

    # 買賣價差
    mid = (x["bid"] + x["ask"]) / 2
    x["spread_pct"] = np.where(
        (x["bid"].notna()) & (x["ask"].notna()) & (mid > 0),
        (x["ask"] - x["bid"]) / mid * 100,
        np.nan
    )

    min_vol = float(cfg.get("min_warrant_volume", 300))
    min_dte = float(cfg.get("min_days_to_expiry", 120))
    max_otm = float(cfg.get("max_otm_pct", 15.0))

    # 分數：公開資料可得欄位優先，缺資料不直接判死刑
    vol_score = np.clip(np.log10(x["volume"].fillna(0) + 1) / 5 * 35, 0, 35)

    dte_score = pd.Series(8.0, index=x.index)
    dte_known = x["days_to_expiry"].notna()
    dte_score.loc[dte_known] = np.clip((x.loc[dte_known, "days_to_expiry"] / max(min_dte,1)) * 15, 0, 15)

    otm_score = pd.Series(8.0, index=x.index)
    otm_known = x["otm_pct"].notna()
    otm_abs = x.loc[otm_known, "otm_pct"].abs()
    otm_score.loc[otm_known] = np.clip(20 - (otm_abs / max(max_otm,1) * 20), 0, 20)

    spread_score = pd.Series(8.0, index=x.index)
    spread_known = x["spread_pct"].notna()
    spread_score.loc[spread_known] = np.clip(15 - x.loc[spread_known, "spread_pct"] * 1.2, 0, 15)

    price_score = pd.Series(5.0, index=x.index)
    price_known = x["price"].notna() & (x["price"] > 0)
    price_score.loc[price_known] = 10.0

    issuer_score = x["issuer"].fillna("").astype(str).str.len().gt(0).astype(float) * 5

    x["warrant_score"] = (
        vol_score + dte_score + otm_score + spread_score + price_score + issuer_score
    ).round(1)

    # 硬條件只對「已知」欄位判斷；未知欄位不因資料缺失直接淘汰。
    eligible = x["volume"] >= min_vol
    reasons = pd.Series("", index=x.index, dtype=object)

    fail_vol = x["volume"] < min_vol
    reasons.loc[fail_vol] += "成交量不足；"

    fail_dte = x["days_to_expiry"].notna() & (x["days_to_expiry"] < min_dte)
    eligible = eligible & ~fail_dte
    reasons.loc[fail_dte] += "剩餘天數不足；"

    fail_otm = x["otm_pct"].notna() & (x["otm_pct"] > max_otm)
    eligible = eligible & ~fail_otm
    reasons.loc[fail_otm] += "價外過深；"

    x["eligible"] = eligible
    x["filter_reason"] = np.where(
        x["eligible"],
        "通過目前可驗證條件",
        reasons.str.rstrip("；")
    )

    return x.sort_values(
        ["eligible","warrant_score","volume"],
        ascending=[False,False,False]
    ).reset_index(drop=True)

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
    if setup in ["事件型反彈","事件型機會","事件追蹤"]:
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



# ===== V6.11 法人／大戶籌碼評分 =====
def chip_score_v611(
    foreign_1d=0, foreign_3d=0, foreign_5d=0,
    trust_1d=0, trust_3d=0, trust_5d=0,
    dealer_1d=0, dealer_3d=0, dealer_5d=0,
    foreign_streak=0, trust_streak=0,
    large_holder_change=None, price_5d=None
):
    """
    回傳 0~20 分的法人/大戶籌碼分數與中文判斷。
    金額/張數可使用同一方向性的任意單位；重點是正負與相對強弱。
    large_holder_change: 大戶持股比例變化(百分點)，無資料可為 None。
    price_5d: 近5日漲跌幅(%)，用於法人/股價背離判斷。
    """
    vals = [foreign_1d, foreign_3d, foreign_5d, trust_1d, trust_3d, trust_5d,
            dealer_1d, dealer_3d, dealer_5d]
    vals = [0 if v is None else float(v) for v in vals]
    (f1,f3,f5,t1,t3,t5,d1,d3,d5) = vals

    score = 10.0  # 中性起點

    # 外資：最高 4 分影響
    score += 1.0 if f1 > 0 else (-1.0 if f1 < 0 else 0)
    score += 1.2 if f3 > 0 else (-1.2 if f3 < 0 else 0)
    score += 1.8 if f5 > 0 else (-1.8 if f5 < 0 else 0)

    # 投信：最高 4.5 分影響，較重視連續性
    score += 1.0 if t1 > 0 else (-1.0 if t1 < 0 else 0)
    score += 1.5 if t3 > 0 else (-1.5 if t3 < 0 else 0)
    score += 2.0 if t5 > 0 else (-2.0 if t5 < 0 else 0)

    # 自營商：輔助訊號
    score += 0.4 if d3 > 0 else (-0.4 if d3 < 0 else 0)
    score += 0.6 if d5 > 0 else (-0.6 if d5 < 0 else 0)

    # 連買/連賣
    score += max(-1.5, min(1.5, float(foreign_streak or 0) * 0.3))
    score += max(-2.0, min(2.0, float(trust_streak or 0) * 0.4))

    # 大戶持股比例變化（資料存在才計分）
    if large_holder_change is not None:
        c = float(large_holder_change)
        score += max(-2.0, min(2.0, c * 2.0))

    # 法人與股價背離：跌價法人買 → 加分；漲價法人賣 → 扣分
    if price_5d is not None:
        institutional_5d = f5 + t5
        p = float(price_5d)
        if p < 0 and institutional_5d > 0:
            score += 1.5
        elif p > 0 and institutional_5d < 0:
            score -= 1.5

    score = round(max(0.0, min(20.0, score)), 1)
    if score >= 15:
        label = "🟢 法人偏多"
    elif score >= 8:
        label = "🟡 籌碼中性"
    else:
        label = "🔴 法人偏空"

    return {
        "籌碼分數": score,
        "籌碼判斷": label,
        "外資1日": f1, "外資3日": f3, "外資5日": f5,
        "投信1日": t1, "投信3日": t3, "投信5日": t5,
        "自營商1日": d1, "自營商3日": d3, "自營商5日": d5,
        "外資連買天數": foreign_streak or 0,
        "投信連買天數": trust_streak or 0,
        "大戶趨勢": "資料待取得" if large_holder_change is None else large_holder_change,
    }


def blend_score_v611(base_score, chip_score=None, chip_confidence=1.0):
    """
    V6.11 綜合分數：原模型 80% + 籌碼面 20%。
    若籌碼資料缺漏，依 confidence 降低籌碼影響，避免缺資料被誤判為空頭。
    """
    base = float(base_score or 0)
    if chip_score is None:
        return round(base, 1)
    conf = max(0.0, min(1.0, float(chip_confidence)))
    chip100 = float(chip_score) * 5.0
    w = 0.20 * conf
    return round(base * (1.0 - w) + chip100 * w, 1)
