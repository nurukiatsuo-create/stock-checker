from datetime import datetime
import json
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# ==========================================
# 1. 設定 & ポートフォリオ管理
# ==========================================
JST = pytz.timezone("Asia/Tokyo")

# 監視対象ユニバース
UNIVERSE = {
    "swing": {
        "8306.T": "三菱UFJ",
        "7012.T": "川崎重工",
        "7011.T": "三菱重工",
        "9107.T": "川崎汽船",
        "6758.T": "ソニーG",
    }
}

# 保有ポジション設定（約定値・損切りライン）
HOLDINGS = {
    "8306": {
        "side": "BUY",
        "entry_price": 3661.0,  # 本日の約定価格
        "stop_loss": 3570.0,  # 本日安値割れ（撤退ライン）
        "target_profit": 3780.0,  # 直近高値圏ターゲット
    }
}


# ==========================================
# 2. 相場流判定ロジック
# ==========================================
def evaluate_stock(df, code_clean):
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    c_open, c_close, c_high, c_low = (
        curr["Open"],
        curr["Close"],
        curr["High"],
        curr["Low"],
    )
    p_open, p_close = prev["Open"], prev["Close"]

    ma5 = curr["MA5"]
    ma20 = curr["MA20"]
    ma60 = curr["MA60"]
    p_ma5 = prev["MA5"]

    bias_20 = ((c_close - ma20) / ma20) * 100
    bias_str = f"{bias_20:+.1f}%"

    candle_body_mid = (c_open + c_close) / 2.0
    is_yang = c_close >= c_open
    is_yin = c_close < c_open

    # 相場流シグナル判定
    is_shitahanshin = (
        is_yang
        and (candle_body_mid > ma5)
        and (c_close > ma5)
        and (p_close <= p_ma5 or (p_open + p_close) / 2.0 <= p_ma5)
    )
    is_gyaku_shitahanshin = (
        is_yin and (candle_body_mid < ma5) and (c_close < ma5)
    )
    is_monowakare = (c_low <= ma20 * 1.015) and (c_close > ma20) and is_yang
    is_ppp = (ma5 > ma20) and (ma20 > ma60)
    is_reverse_ppp = (ma5 < ma20) and (ma20 < ma60)

    # 保有中銘柄の判定
    if code_clean in HOLDINGS:
        h = HOLDINGS[code_clean]
        sl = h["stop_loss"]
        tp = h["target_profit"]
        pnl = ((c_close - h["entry_price"]) / h["entry_price"]) * 100

        if c_close <= sl or c_low <= sl:
            return {
                "status": "ロスカット撤退",
                "badge": "EXIT",
                "bias": bias_str,
            }
        elif is_gyaku_shitahanshin:
            return {
                "status": "手仕舞い(逆下半身)",
                "badge": "EXIT",
                "bias": bias_str,
            }
        elif c_close >= tp:
            return {
                "status": "利確目安到達",
                "badge": "EXIT",
                "bias": bias_str,
            }
        else:
            return {
                "status": f"継続保有({pnl:+.1f}%)",
                "badge": "HOLD",
                "bias": bias_str,
            }

    # 未保有銘柄の判定
    if is_shitahanshin and is_monowakare:
        return {
            "status": "下半身+ものわかれ(買)",
            "badge": "BUY",
            "bias": bias_str,
        }
    elif is_shitahanshin:
        return {"status": "下半身(打診買)", "badge": "BUY", "bias": bias_str}
    elif is_monowakare:
        return {"status": "ものわかれ初動", "badge": "BUY", "bias": bias_str}
    elif is_reverse_ppp:
        return {"status": "待機(手出し無用)", "badge": "NONE", "bias": bias_str}
    elif is_ppp:
        if c_close > ma5:
            return {"status": "押し目待ち", "badge": "WAIT", "bias": bias_str}
        else:
            return {
                "status": "調整中(押し目監視)",
                "badge": "WAIT",
                "bias": bias_str,
            }
    return {"status": "保ち合い(様子見)", "badge": "NONE", "bias": bias_str}


# ==========================================
# 3. ウィジェット用ダークテーマチャート生成
# ==========================================
def draw_chart(df, code_clean, stock_name, status_text):
    plot_df = df.tail(35).copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="#161618")
    ax.set_facecolor("#161618")

    # 移動平均線
    dates = [mdates.date2num(d) for d in plot_df.index]
    ax.plot(
        dates,
        plot_df["MA5"],
        color="#30d158",
        linewidth=1.2,
        label="5MA",
        alpha=0.9,
    )
    ax.plot(
        dates,
        plot_df["MA20"],
        color="#ff9f0a",
        linewidth=1.4,
        label="20MA",
        alpha=0.9,
    )
    ax.plot(
        dates,
        plot_df["MA60"],
        color="#64d2ff",
        linewidth=1.2,
        label="60MA",
        alpha=0.8,
    )

    # ローソク足描画
    width = 0.55
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        d = dates[i]
        o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
        color = "#ff453a" if c >= o else "#0a84ff"  # 日本株仕様：陽線=赤, 陰線=青
        ax.plot([d, d], [l, h], color=color, linewidth=1.0)
        ax.bar(
            d,
            abs(c - o),
            bottom=min(o, c),
            width=width,
            color=color,
            edgecolor=color,
            linewidth=0.5,
        )

    # 相場流シグナルマーカー
    for i in range(1, len(plot_df)):
        curr_row = plot_df.iloc[i]
        prev_row = plot_df.iloc[i - 1]
        d = dates[i]
        mid = (curr_row["Open"] + curr_row["Close"]) / 2.0
        # 下半身シグナル（上向き三角）
        if (
            curr_row["Close"] >= curr_row["Open"]
            and mid > curr_row["MA5"]
            and prev_row["Close"] <= prev_row["MA5"]
        ):
            ax.scatter(
                d,
                curr_row["Low"] * 0.992,
                color="#ffd60a",
                marker="^",
                s=40,
                zorder=5,
            )
        # 逆下半身シグナル（下向き三角）
        elif (
            curr_row["Close"] < curr_row["Open"]
            and mid < curr_row["MA5"]
            and prev_row["Close"] >= prev_row["MA5"]
        ):
            ax.scatter(
                d,
                curr_row["High"] * 1.008,
                color="#64d2ff",
                marker="v",
                s=40,
                zorder=5,
            )

    # 軸・グリッド設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(colors="#8e8e93", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#2c2c2e", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#2c2c2e")

    plt.tight_layout()
    fig.savefig(
        f"chart_{code_clean}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#161618",
    )
    if code_clean == "8306":
        fig.savefig(
            "chart.png", dpi=150, bbox_inches="tight", facecolor="#161618"
        )
    plt.close(fig)


# ==========================================
# 4. メイン処理（実行 & 出力）
# ==========================================
def main():
    now_jst = datetime.now(JST).strftime("%m/%d %H:%M JST")
    stock_results = []
    swing_universe = UNIVERSE["swing"]

    for symbol, name in swing_universe.items():
        code_clean = symbol.replace(".T", "")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")

        if df.empty or len(df) < 60:
            continue

        # 移動平均線の算出
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()

        # 判定
        res = evaluate_stock(df, code_clean)
        c_price = df.iloc[-1]["Close"]
        price_str = (
            f"{c_price:,.1f}円" if c_price < 1000 else f"{int(c_price):,}円"
        )

        stock_results.append(
            {
                "code": code_clean,
                "name": name,
                "price": price_str,
                "bias": res["bias"],
                "status": res["status"],
                "badge": res["badge"],
            }
        )

        # チャート画像を出力
        draw_chart(df, code_clean, name, res["status"])

    # 三菱UFJを優先してトップに配置
    top_stock = next(
        (s for s in stock_results if s["code"] == "8306"), stock_results[0]
    )

    output_data = {
        "updated_at": now_jst,
        "top_stock": top_stock,
        "stocks": stock_results,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[{now_jst}] 更新完了: 三菱UFJステータス -> {top_stock['status']}")


if __name__ == "__main__":
    main()
