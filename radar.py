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

