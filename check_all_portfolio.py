from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

JST = pytz.timezone("Asia/Tokyo")

# ==========================================
# 監視対象銘柄リスト（相場流スイング枠）
# ==========================================
UNIVERSE = {
    "8306.T": "三菱UFJ",
    "6326.T": "クボタ",
    "7269.T": "スズキ",
    "7011.T": "三菱重工",
    "7012.T": "川崎重工",
    "9107.T": "川崎汽船",
    "8002.T": "丸紅",
}

# ==========================================
# 現在の保有ポジション管理
# ==========================================
HOLDINGS = {
    "9107": {
        "side": "BUY",
        "entry_price": 3500.40,
        "shares": 200,
        "stop_loss": 3490.0,
        "target_profit": 3530.0,
        "entry_date": "2026-09-17",
    },
    "7011": {
        "side": "BUY",
        "entry_price": 3891.40,
        "shares": 100,
        "stop_loss": 3830.0,
        "target_profit": 3895.0,
        "entry_date": "2026-09-18",
    },
}


# ==========================================
# 評価・スコアリング判定ロジック
# ==========================================
def evaluate_stock(df, code_clean):
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    c_open, c_close = float(curr["Open"]), float(curr["Close"])
    c_high, c_low = float(curr["High"]), float(curr["Low"])
    p_open, p_close = float(prev["Open"]), float(prev["Close"])

    ma5, ma20 = float(curr["MA5"]), float(curr["MA20"])
    p_ma5, p_ma20 = float(prev["MA5"]), float(prev["MA20"])

    # 移動平均線の傾き (%)
    ma5_slope = ((ma5 - p_ma5) / p_ma5) * 100 if p_ma5 > 0 else 0.0
    ma20_slope = ((ma20 - p_ma20) / p_ma20) * 100 if p_ma20 > 0 else 0.0

    is_ma5_up_or_flat = ma5 >= p_ma5
    is_ma5_down_or_flat = ma5 <= p_ma5
    is_ma20_up = ma20 >= p_ma20
    is_ma20_down = ma20 <= p_ma20

    # 20日線乖離率
    bias_20 = ((c_close - ma20) / ma20) * 100 if ma20 > 0 else 0.0
    bias_str = f"{bias_20:+.1f}%"

    candle_body_mid = (c_open + c_close) / 2.0
    is_yang = c_close >= c_open
    is_yin = c_close < c_open

    # 相場流：基本シグナル判定
    is_shitahanshin = (
        is_yang
        and (candle_body_mid > ma5)
        and (c_close > ma5)
        and (p_close <= p_ma5 or (p_open + p_close) / 2.0 <= p_ma5)
        and is_ma5_up_or_flat
    )

    is_gyaku_shitahanshin = (
        is_yin
        and (candle_body_mid < ma5)
        and (c_close < ma5)
        and (p_close >= p_ma5 or (p_open + p_close) / 2.0 >= p_ma5)
        and is_ma5_down_or_flat
    )

    is_monowakare = (
        (c_low <= ma20 * 1.015) and (c_close > ma20) and is_yang and is_ma20_up
    )

    # ----------------------------------------------------
    # 1. 保有銘柄のエグジット判定（優先処理）
    # ----------------------------------------------------
    if code_clean in HOLDINGS:
        h = HOLDINGS[code_clean]
        side = h.get("side", "BUY")
        entry_date = pd.to_datetime(h.get("entry_date", df.index[-1]))
        candles_since_entry = int(len(df[df.index >= entry_date]))

        if side == "BUY":
            pnl = ((c_close - h["entry_price"]) / h["entry_price"]) * 100

            if c_close <= h["stop_loss"] or c_low <= h["stop_loss"]:
                return {
                    "status": "ロスカット撤退",
                    "badge": "EXIT",
                    "bias": bias_str,
                    "score": -99,
                }
            elif is_gyaku_shitahanshin:
                return {
                    "status": "手仕舞い(逆下半身)",
                    "badge": "EXIT",
                    "bias": bias_str,
                    "score": -99,
                }
            elif c_close >= h["target_profit"]:
                return {
                    "status": "利確指値到達",
                    "badge": "EXIT",
                    "bias": bias_str,
                    "score": -99,
                }
            elif candles_since_entry >= 14 and is_yin:
                return {
                    "status": f"手仕舞い(日柄{candles_since_entry}本・陰線)",
                    "badge": "EXIT",
                    "bias": bias_str,
                    "score": -99,
                }
            else:
                return {
                    "status": f"保有継続({pnl:+.1f}%/日柄{candles_since_entry}本)",
                    "badge": "HOLD",
                    "bias": bias_str,
                    "score": 0,
                }

    # ----------------------------------------------------
    # 2. 未保有銘柄のスコアリングロジック
    # ----------------------------------------------------
    score = 0

    # 【基本点】
    if is_shitahanshin and is_monowakare:
        score += 80
        status = "下半身+ものわかれ(強買)"
        badge = "BUY"
    elif is_shitahanshin:
        score += 50
        status = "下半身(買い)"
        badge = "BUY"
    elif is_gyaku_shitahanshin:
        score += 50
        status = "逆下半身(空売り)"
        badge = "SHORT"
    elif is_monowakare:
        score += 30
        status = "ものわかれ初動"
        badge = "BUY"
    else:
        return {"status": "様子見", "badge": "NONE", "bias": bias_str, "score": 0}

    # 【加点1：20日線の傾き（トレンドの向き）】最大+20点
    if badge == "BUY" and is_ma20_up:
        score += min(max(int(ma20_slope * 20), 0), 20)
    elif badge == "SHORT" and is_ma20_down:
        score += min(max(int(abs(ma20_slope) * 20), 0), 20)

    # 【加点2：実体比率（ローソク足の推進力）】最大+10点
    high_low_range = c_high - c_low
    body_range = abs(c_close - c_open)
    body_ratio = (body_range / high_low_range) if high_low_range > 0 else 0.0
    score += int(body_ratio * 10)

    # 【減点：過熱感（20日線乖離率 8%超）】-15点
    if abs(bias_20) > 8.0:
        score -= 15

    return {
        "status": status,
        "badge": badge,
        "bias": bias_str,
        "score": int(score),
    }


# ==========================================
# メイン処理（データ取得・ソート・JSON保存）
# ==========================================
def main():
    now_jst = datetime.now(JST).strftime("%m/%d %H:%M JST")
    stock_results = []

    for symbol, name in UNIVERSE.items():
        code_clean = symbol.replace(".T", "")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
        except Exception as e:
            print(f"データ取得エラー ({symbol}): {e}")
            continue

        if df.empty or len(df) < 25:
            continue

        # 移動平均線
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()

        res = evaluate_stock(df, code_clean)
        c_price = float(df.iloc[-1]["Close"])
        price_str = (
            f"{c_price:,.1f}円" if c_price < 1000 else f"{int(c_price):,}円"
        )

        stock_results.append(
            {
                "code": code_clean,
                "name": name,
                "price": price_str,
                "raw_price": c_price,
                "bias": res["bias"],
                "status": res["status"],
                "badge": res["badge"],
                "score": res["score"],
            }
        )

    # ----------------------------------------------------
    # 並び替え：シグナル成立(BUY/SHORT)を上位に、スコア降順でソート
    # ----------------------------------------------------
    stock_results.sort(
        key=lambda x: (
            x["badge"] in ["BUY", "SHORT"],
            x["score"],
            -x["raw_price"],
        ),
        reverse=True,
    )

    top_stock = stock_results[0] if stock_results else None

    output_data = {
        "updated_at": now_jst,
        "top_stock": top_stock,
        "stocks": stock_results,
    }

    output_path = "result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[{now_jst}] 判定完了 -> result.json を出力しました。")
    if top_stock:
        print(
            f"★ 最優先銘柄: {top_stock['name']} ({top_stock['code']}) | 判定: {top_stock['status']} | スコア: {top_stock['score']}点"
        )


if __name__ == "__main__":
    main()
