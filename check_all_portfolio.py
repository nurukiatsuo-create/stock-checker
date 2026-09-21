from datetime import datetime
import json
import os
import matplotlib

matplotlib.use("Agg")  # GitHub Actions等のCUI環境で描画するためのヘッドレス設定
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

JST = pytz.timezone("Asia/Tokyo")

# ==========================================
# 1. 監視対象銘柄リスト（相場流スイング枠）
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
# 2. 現在の保有ポジション管理
# ==========================================
HOLDINGS = {
    "9107": {
        "name": "川崎汽船",
        "side": "BUY",
        "entry_price": 3500.40,
        "shares": 200,
        "stop_loss": 3490.0,
        "target_profit": 3530.0,
        "entry_date": "2026-09-17",
    },
    "7011": {
        "name": "三菱重工",
        "side": "BUY",
        "entry_price": 3891.40,
        "shares": 100,
        "stop_loss": 3830.0,
        "target_profit": 3895.0,
        "entry_date": "2026-09-18",
    },
}


# ==========================================
# 3. 判定 & スコアリングエンジン
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

    # 相場流：シグナル判定
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
    # A. 保有銘柄のエグジット判定（優先処理）
    # ----------------------------------------------------
    if code_clean in HOLDINGS:
        h = HOLDINGS[code_clean]
        side = h.get("side", "BUY")

        # タイムゾーン不整合を防止した安全な日付比較
        raw_date = h.get("entry_date", df.index[-1])
        entry_date = pd.to_datetime(raw_date)
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)

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
    # B. 未保有銘柄の優先度スコアリング
    # ----------------------------------------------------
    score = 0

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

    # 【加点1】20日線の傾き（トレンドの勢い）最大+20点
    if badge == "BUY" and is_ma20_up:
        score += min(max(int(ma20_slope * 20), 0), 20)
    elif badge == "SHORT" and is_ma20_down:
        score += min(max(int(abs(ma20_slope) * 20), 0), 20)

    # 【加点2】ローソク足の実体比率（推進力）最大+10点
    hl_range = c_high - c_low
    body_ratio = (abs(c_close - c_open) / hl_range) if hl_range > 0 else 0.0
    score += int(body_ratio * 10)

    # 【減点】過熱感（20日線乖離率 8%超）
    if abs(bias_20) > 8.0:
        score -= 15

    return {
        "status": status,
        "badge": badge,
        "bias": bias_str,
        "score": int(score),
    }


# ==========================================
# 4. チャート画像生成（Scriptableウィジェット用）
# ==========================================
def generate_chart(df, name, code, output_path="chart.png"):
    plot_df = df.tail(40).copy()  # 直近40日分を描画

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#141414")
    ax.set_facecolor("#141414")

    # グリッド線
    ax.grid(True, linestyle=":", alpha=0.3, color="#555555")

    # 移動平均線
    ax.plot(
        plot_df.index,
        plot_df["MA5"],
        color="#ff4444",
        linewidth=1.8,
        label="5MA",
        alpha=0.9,
    )
    ax.plot(
        plot_df.index,
        plot_df["MA20"],
        color="#3399ff",
        linewidth=1.8,
        label="20MA",
        alpha=0.9,
    )

    # ローソク足の描画
    width = 0.6
    for idx, row in plot_df.iterrows():
        o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
        color = "#ff4444" if c >= o else "#3399ff"

        # ヒゲ
        ax.vlines(idx, l, h, color=color, linewidth=1.0, alpha=0.8)
        # 実体
        lower = min(o, c)
        height = abs(c - o)
        if height == 0:
            height = 0.5
        ax.bar(
            idx,
            height,
            bottom=lower,
            color=color,
            width=width,
            align="center",
            alpha=0.9,
        )

    # 直近高値ライン（黄色破線）
    recent_high = plot_df["High"].max()
    ax.axhline(
        recent_high, color="#f1c40f", linestyle="--", linewidth=1.0, alpha=0.7
    )

    # 軸設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    plt.title(f"{name} ({code})", color="#ffffff", fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()


# ==========================================
# 5. メイン処理
# ==========================================
def main():
    now_jst = datetime.now(JST).strftime("%m/%d %H:%M JST")
    stock_results = []
    chart_targets = {}

    for symbol, name in UNIVERSE.items():
        code_clean = symbol.replace(".T", "")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
        except Exception as e:
            print(f"取得エラー ({symbol}): {e}")
            continue

        if df.empty or len(df) < 25:
            continue

        # --- タイムゾーンエラーの根本解消処理 ---
        if df.index.tz is not None:
            df.index = df.index.tz_convert(JST).tz_localize(None)

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
        chart_targets[code_clean] = (df, name)

    # ----------------------------------------------------
    # 並び替え（シグナル点灯を上位、スコア降順）
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

    # チャート画像の出力（最優先銘柄、または保有銘柄を描画）
    if top_stock and top_stock["code"] in chart_targets:
        df_top, name_top = chart_targets[top_stock["code"]]
        generate_chart(df_top, name_top, top_stock["code"], "chart.png")
        print(f"チャート生成完了: chart.png -> {name_top}")

    # JSON書き出し
    output_data = {
        "updated_at": now_jst,
        "top_stock": top_stock,
        "stocks": stock_results,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[{now_jst}] 処理完了 -> result.json 出力完了")
    if top_stock:
        print(
            f"★ 最優先銘柄: {top_stock['name']} ({top_stock['code']}) | 状態: {top_stock['status']} | スコア: {top_stock['score']}点"
        )


if __name__ == "__main__":
    main()
