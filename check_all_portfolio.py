from datetime import datetime
import json
import os
import shutil
import matplotlib

matplotlib.use("Agg")  # GitHub Actions(Linux)用ヘッドレス描画設定
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

JST = pytz.timezone("Asia/Tokyo")

# ==========================================
# 1. 監視対象銘柄リスト（バックテスト検証済みの厳選銘柄）
# ==========================================
UNIVERSE = {
    "8306.T": "三菱UFJ",
    "8002.T": "丸紅",
    "9107.T": "川崎汽船",
    "7012.T": "川崎重工",
    "7011.T": "三菱重工",  # ※木曜日の手仕舞い完了まで監視維持
}

# ==========================================
# 2. 現在の保有ポジション管理
# ==========================================
HOLDINGS = {
    "9107": {
        "name": "川崎汽船",
        "side": "buy",
        "entry_price": 3500.40,
        "shares": 200,
        "stop_loss": 3490.0,
        "target_profit": 3530.0,
        "entry_date": "2026-09-17",
    },
    "7011": {
        "name": "三菱重工",
        "side": "buy",
        "entry_price": 3891.40,
        "shares": 100,
        "stop_loss": 3830.0,
        "target_profit": 3895.0,
        "entry_date": "2026-09-18",
    },
}


# ==========================================
# 3. 相場流・高精度判定 & スコアリング
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

    # 相場流：初動シグナル
    is_shitahanshin = (
        is_yang
        and (candle_body_mid > ma5)
        and (c_close > ma5)
        and (p_close <= p_ma5)
        and is_ma5_up_or_flat
    )

    is_gyaku_shitahanshin = (
        is_yin
        and (candle_body_mid < ma5)
        and (c_close < ma5)
        and (p_close >= p_ma5)
        and is_ma5_down_or_flat
    )

    is_monowakare = (
        (c_low <= ma20 * 1.015) and (c_close > ma20) and is_yang and is_ma20_up
    )

    # ----------------------------------------------------
    # A. 保有ポジションのエグジット判定
    # ----------------------------------------------------
    if code_clean in HOLDINGS:
        h = HOLDINGS[code_clean]
        pos_type = h.get("side", "buy")
        entry_price = float(h.get("entry_price", c_close))

        raw_date = h.get("entry_date", df.index[-1])
        entry_date = pd.to_datetime(raw_date)
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_localize(None)

        candles_since_entry = int(len(df[df.index >= entry_date]))
        pnl = ((c_close - entry_price) / entry_price) * 100

        # 手仕舞い判定（利確指値・損切り・相場流手仕舞い）
        if c_close >= h["target_profit"] or c_high >= h["target_profit"]:
            status = "手仕舞(利確指値)"
            badge = "EXIT"
        elif c_close <= h["stop_loss"] or c_low <= h["stop_loss"]:
            status = "手仕舞(ロスカット)"
            badge = "EXIT"
        elif (c_close < ma5) and is_yin:
            status = "手仕舞(5MA割れ陰線)"
            badge = "EXIT"
        elif candles_since_entry >= 10 and is_yin:
            status = f"手仕舞(日柄{candles_since_entry}本)"
            badge = "EXIT"
        else:
            status = f"保有中({pnl:+.1f}%)"
            badge = "HOLD"

        return {
            "status": status,
            "badge": badge,
            "bias": bias_str,
            "score": 0,
            "is_holding": True,
            "pos_type": pos_type,
            "entry_price": entry_price,
        }

    # ----------------------------------------------------
    # B. 未保有銘柄の判定（20MA順張り・安全フィルター準拠）
    # ----------------------------------------------------
    score = 0
    status = "待機"
    badge = "NONE"

    # 【買い条件】20MA上向き ＋ 株価が20MA以上 ＋ 初動下半身
    if is_shitahanshin and is_ma20_up and (c_close >= ma20):
        if is_monowakare:
            score = 80
            status = "下半身+ものわかれ(現買)"
        else:
            score = 50
            status = "下半身(現買)"
        badge = "BUY"

    # 【空売り条件】20MA下向き ＋ 株価が20MA以下 ＋ 乖離-5%以内の安全初動
    elif (
        is_gyaku_shitahanshin
        and is_ma20_down
        and (c_close <= ma20)
        and (bias_20 > -5.0)
    ):
        score = 50
        status = "逆下半身(空売)"
        badge = "SHORT"

    # 有効シグナル時のみ加減点を計算
    if badge != "NONE":
        # 傾き加点（最大+20点）
        if badge == "BUY":
            score += min(max(int(ma20_slope * 20), 0), 20)
        elif badge == "SHORT":
            score += min(max(int(abs(ma20_slope) * 20), 0), 20)

        # 実体推進力加点（最大+10点）
        hl_range = c_high - c_low
        if hl_range > 0:
            body_ratio = abs(c_close - c_open) / hl_range
            score += int(body_ratio * 10)

        # 乖離過熱ペナルティ
        if abs(bias_20) > 8.0:
            score -= 15

    return {
        "status": status,
        "badge": badge,
        "bias": bias_str,
        "score": int(score),
        "is_holding": False,
        "pos_type": None,
        "entry_price": None,
    }


# ==========================================
# 4. チャート画像生成（文字化け防止版）
# ==========================================
def generate_chart(df, code, output_path):
    plot_df = df.tail(40).copy()

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#141414")
    ax.set_facecolor("#141414")
    ax.grid(True, linestyle=":", alpha=0.3, color="#555555")

    # 5日線（赤） / 20日線（青）
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

    # ローソク足
    width = 0.6
    for idx, row in plot_df.iterrows():
        o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
        color = "#ff4444" if c >= o else "#3399ff"
        ax.vlines(idx, l, h, color=color, linewidth=1.0, alpha=0.8)
        lower = min(o, c)
        height = max(abs(c - o), 0.5)
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
    ax.axhline(
        plot_df["High"].max(),
        color="#f1c40f",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    # 【文字化け防止】英字とコードのみでタイトルを構成
    plt.title(
        f"STOCK CODE: {code} (5MA / 20MA)", color="#ffffff", fontsize=10, pad=8
    )
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

        # タイムゾーンの正規化（不整合エラー防止）
        if df.index.tz is not None:
            df.index = df.index.tz_convert(JST).tz_localize(None)

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
                "is_holding": res["is_holding"],
                "pos_type": res["pos_type"],
                "entry_price": res["entry_price"],
            }
        )
        chart_targets[code_clean] = df

        # 個別銘柄チャート画像の出力 (chart_{code}.png)
        generate_chart(df, code_clean, f"chart_{code_clean}.png")

    # ソート順：保有銘柄 ＞ 有効シグナル(BUY/SHORT) ＞ スコア順
    stock_results.sort(
        key=lambda x: (
            x["is_holding"],
            x["badge"] in ["BUY", "SHORT"],
            x["score"],
            -x["raw_price"],
        ),
        reverse=True,
    )

    top_stock = stock_results[0] if stock_results else None

    # 共通チャート（chart.png）の更新
    if top_stock and top_stock["code"] in chart_targets:
        shutil.copy(f"chart_{top_stock['code']}.png", "chart.png")

    output_data = {
        "updated_at": now_jst,
        "top_stock": top_stock,
        "stocks": stock_results,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[{now_jst}] 更新完了: 厳選ユニバースの判定およびチャート出力完了")


if __name__ == "__main__":
    main()
