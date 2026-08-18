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
    """V6.8 多因子評分（100分）。

    八大因子：
    趨勢18、量價14、動能12、位置12、突破品質12、
    支撐/承接10、波動風險10、穩定性12。

    另外輸出資料完整度/信心值與加扣分摘要。
    """
    if hist is None or len(hist) < 22:
        return {
            "score": 0, "raw_score": 0, "setup": "資料不足",
            "trend": 0, "volume_price": 0, "momentum": 0, "position": 0,
            "breakout_quality": 0, "support": 0, "risk": 0, "consistency": 0,
            "data_confidence": 0, "score_reason": "歷史資料不足"
        }

    x = add_indicators(hist)
    r = x.iloc[-1]
    prev = x.iloc[-2]

    close = _safe(r.close)
    ma5 = _safe(r.ma5, close)
    ma10 = _safe(r.ma10, close)
    ma20 = _safe(r.ma20, close)
    ma60 = _safe(r.ma60, ma20)
    volr = _safe(r.volume_ratio, 1)
    ret1 = _safe(r.ret1, 0)
    ret3 = _safe(r.ret3, 0)
    ret5 = _safe(r.ret5, 0)
    ret10 = _safe(r.ret10, 0)
    ret20 = _safe(r.ret20, 0)
    rsi = _safe(r.rsi14, 50)
    dist20 = _safe(r.dist_ma20, 0)
    macd_hist = _safe(r.macd_hist, 0)
    prev_macd_hist = _safe(prev.macd_hist, 0)
    atr_pct = _safe(r.atr_pct, 0.03)
    vol20 = _safe(r.volatility20, 0.02)
    bb_pos = _safe(r.bb_pos, 0.5)
    close_loc = _safe(r.close_location, 0.5)
    body_ratio = _safe(r.body_ratio, 0.5)
    obv = _safe(r.obv, 0)
    obv_ma10 = _safe(r.obv_ma10, obv)
    from20high = _safe(r.from_20_high, 0)
    from20low = _safe(r.from_20_low, 0)

    reasons = []

    # 1) 趨勢 18
    trend = 2.0
    trend += 4 if close > ma20 else 0
    trend += 3 if ma5 > ma10 else 0
    trend += 3 if ma10 > ma20 else 0
    trend += 3 if ma20 > ma60 else 0
    trend += 2 if ret20 > 0 else 0
    trend += 1 if ma20 > _safe(x["ma20"].iloc[-6], ma20) else 0
    trend = _clamp(trend, 0, 18)
    if trend >= 14: reasons.append("中期趨勢偏多")
    elif trend <= 6: reasons.append("趨勢偏弱")

    # 2) 量價 14
    volume_price = 4.0
    if 1.1 <= volr < 1.5: volume_price += 2
    elif 1.5 <= volr < 2.5: volume_price += 4
    elif volr >= 2.5: volume_price += 5
    if ret1 > 0 and volr >= 1.2: volume_price += 3
    if abs(ret1) <= 0.02 and volr <= 0.85 and close >= ma20: volume_price += 2
    if ret1 < -0.04 and volr >= 1.8: volume_price -= 2
    if obv > obv_ma10: volume_price += 1
    volume_price = _clamp(volume_price, 0, 14)
    if volume_price >= 11: reasons.append("量價配合佳")

    # 3) 動能 12
    momentum = 4.0
    if ret3 > 0: momentum += 1
    if ret5 > 0: momentum += 2
    if ret10 > 0: momentum += 1
    if macd_hist > 0: momentum += 2
    if macd_hist > prev_macd_hist: momentum += 1
    if 45 <= rsi <= 68: momentum += 1
    if rsi > 78: momentum -= 3
    momentum = _clamp(momentum, 0, 12)
    if momentum >= 9: reasons.append("動能正在改善")

    # 4) 位置 12
    position = 4.0
    if -0.035 <= dist20 <= 0.035: position += 4
    elif -0.07 <= dist20 < -0.035: position += 2
    elif 0.035 < dist20 <= 0.08: position += 2
    if 0.25 <= bb_pos <= 0.75: position += 2
    if from20high > -0.08: position += 1
    if dist20 > 0.14 or bb_pos > 1.05: position -= 4
    if rsi > 80: position -= 2
    position = _clamp(position, 0, 12)
    if position >= 9: reasons.append("價格位置尚佳")
    elif position <= 4: reasons.append("價格位置不利")

    # 5) 突破品質 12
    recent_high = _safe(x["high"].iloc[-21:-1].max(), close)
    breakout_quality = 2.0
    if close > recent_high: breakout_quality += 4
    if close >= recent_high * 0.985: breakout_quality += 2
    if volr >= 1.4: breakout_quality += 2
    if close_loc >= 0.70: breakout_quality += 1
    if body_ratio >= 0.55 and close >= r.open: breakout_quality += 1
    # 假突破風險
    if close > recent_high and close_loc < 0.45: breakout_quality -= 3
    breakout_quality = _clamp(breakout_quality, 0, 12)
    if breakout_quality >= 9: reasons.append("突破品質佳")

    # 6) 支撐 / 承接 10
    support = 3.0
    if close >= ma20 * 0.98: support += 2
    if close >= ma10 * 0.985: support += 1
    if close_loc >= 0.55: support += 1
    if ret1 < 0 and close_loc >= 0.65: support += 2  # 下跌但收高
    if from20low > 0.08: support += 1
    if close < ma20 * 0.94: support -= 3
    support = _clamp(support, 0, 10)
    if support >= 8: reasons.append("支撐承接明顯")

    # 7) 波動/風險 10：分數越高代表風險越可控
    risk = 8.0
    if atr_pct > 0.06: risk -= 3
    elif atr_pct > 0.045: risk -= 1
    if vol20 > 0.045: risk -= 2
    if abs(ret1) > 0.08: risk -= 2
    if dist20 > 0.15: risk -= 2
    if rsi > 82: risk -= 2
    risk = _clamp(risk, 0, 10)
    if risk <= 5: reasons.append("短線波動風險偏高")

    # 8) 穩定性 12：避免只因單日爆量/長紅就衝高分
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

    raw_score = trend + volume_price + momentum + position + breakout_quality + support + risk + consistency

    # 型態分類與額外懲罰
    if ret1 <= -0.07 and volr >= 1.8 and close_loc >= 0.55:
        setup = "事件型反彈"
    elif close > recent_high and volr >= 1.35 and breakout_quality >= 8:
        setup = "突破型"
    elif close >= ma20 * 0.97 and close <= ma20 * 1.04 and volr <= 1.15 and ma20 >= ma60:
        setup = "趨勢回檔"
    elif rsi >= 78 or dist20 >= 0.14:
        setup = "過熱・不追"
        raw_score -= 10
    elif close < ma20 * 0.96 and ret20 < 0:
        setup = "弱勢觀察"
        raw_score -= 8
    else:
        setup = "蓄勢/中性"

    # 資料完整度：未知不等於壞，但降低信心。
    critical = [r.ma20, r.rsi14, r.volume_ratio, r.macd_hist, r.atr_pct, r.bb_pos]
    known = sum(0 if pd.isna(v) else 1 for v in critical)
    data_confidence = round(known / len(critical) * 100, 1)

    # 極端單日訊號不能讓總分失真
    if abs(ret1) >= 0.09 and volr >= 2.5:
        raw_score -= 4
        reasons.append("單日極端波動扣分")

    score = _clamp(raw_score, 0, 100)

    return {
        "score": round(score, 1),
        "raw_score": round(raw_score, 1),
        "setup": setup,
        "trend": round(trend, 1),
        "volume_price": round(volume_price, 1),
        "momentum": round(momentum, 1),
        "position": round(position, 1),
        "breakout_quality": round(breakout_quality, 1),
        "support": round(support, 1),
        "risk": round(risk, 1),
        "consistency": round(consistency, 1),
        "data_confidence": data_confidence,
        "score_reason": "；".join(reasons[:5]) if reasons else "條件中性",
        "close": round(close, 4),
        "ret1_pct": round(ret1 * 100, 4),
        "ret5_pct": round(ret5 * 100, 4),
        "ret20_pct": round(ret20 * 100, 4),
        "volume_ratio": round(volr, 3),
        "rsi14": round(rsi, 1),
        "ma20": round(ma20, 4),
        "atr_pct": round(atr_pct * 100, 3),
        "macd_hist": round(macd_hist, 4),
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

