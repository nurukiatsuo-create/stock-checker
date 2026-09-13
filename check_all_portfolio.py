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
# 1. 設定 & ポートフォリオ管理（厳選7銘柄）
# ==========================================
JST = pytz.timezone("Asia/Tokyo")

UNIVERSE = {
    "swing": {
        "8306.T": "三菱UFJ",
        "6326.T": "クボタ",
        "7269.T": "スズキ",
        "7011.T": "三菱重工",
        "7012.T": "川崎重工",
        "9107.T": "川崎汽船",
        "8002.T": "丸紅",
    }
}

# 保有ポジション設定（約定値・損切り・利確ターゲット）
HOLDINGS = {
    "8306": {
        "side": "BUY",
        "entry_price": 3661.0,  # 約定価格
        "stop_loss": 3570.0,  # 撤退ライン（20日線割れ）
        "target_profit": 3780.0,  # 利確ターゲット
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
  p_ma5 = prev["MA5"]
  p_ma20 = prev["MA20"]

  is_ma5_up_or_flat = ma5 >= p_ma5
  is_ma5_down = ma5 < p_ma5
  is_ma20_up = ma20 >= p_ma20

  bias_20 = ((c_close - ma20) / ma20) * 100
  bias_str = f"{bias_20:+.1f}%"

  candle_body_mid = (c_open + c_close) / 2.0
  is_yang = c_close >= c_open
  is_yin = c_close < c_open

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
      and is_ma5_down
  )
  is_monowakare = (
      (c_low <= ma20 * 1.015) and (c_close > ma20) and is_yang and is_ma20_up
  )
  is_gyaku_monowakare = (
      (c_high >= ma20 * 0.985)
      and (c_close < ma20)
      and is_yin
      and (not is_ma20_up)
  )

  is_trend_up = (ma5 > ma20) and is_ma20_up
  is_trend_down = (ma5 < ma20) and (not is_ma20_up)

  # A. 保有銘柄
  if code_clean in HOLDINGS:
    h = HOLDINGS[code_clean]
    sl = h.get("stop_loss", h["entry_price"] * 0.97)
    tp = h.get("target_profit", h["entry_price"] * 1.05)
    pnl = ((c_close - h["entry_price"]) / h["entry_price"]) * 100

    if c_close <= sl or c_low <= sl:
      return {"status": "ロスカット撤退", "badge": "EXIT", "bias": bias_str}
    elif is_gyaku_shitahanshin or (c_close < ma5 and is_yin):
      return {
          "status": "手仕舞い(5日線割れ)",
          "badge": "EXIT",
          "bias": bias_str,
      }
    elif c_close >= tp:
      return {"status": "利確目安到達", "badge": "EXIT", "bias": bias_str}
    else:
      return {
          "status": f"継続保有({pnl:+.1f}%)",
          "badge": "HOLD",
          "bias": bias_str,
      }

  # B. 監視銘柄
  if is_shitahanshin and is_monowakare:
    return {"status": "下半身+ものわかれ(買)", "badge": "BUY", "bias": bias_str}
  elif is_shitahanshin:
    return {"status": "下半身(現買確定)", "badge": "BUY", "bias": bias_str}
  elif is_gyaku_shitahanshin and is_gyaku_monowakare:
    return {
        "status": "逆下半身+逆ものわかれ",
        "badge": "SELL",
        "bias": bias_str,
    }
  elif is_gyaku_shitahanshin:
    return {"status": "逆下半身(空売)", "badge": "SELL", "bias": bias_str}
  elif is_gyaku_monowakare:
    return {"status": "逆ものわかれ(空売)", "badge": "SELL", "bias": bias_str}
  elif is_monowakare:
    return {"status": "ものわかれ(反発待)", "badge": "WAIT", "bias": bias_str}
  elif is_trend_up:
    if c_close > ma5:
      return {"status": "押し目待ち", "badge": "WAIT", "bias": bias_str}
    else:
      return {"status": "調整中(押し目監視)", "badge": "WAIT", "bias": bias_str}
  elif is_trend_down:
    return {"status": "下落トレンド(待機)", "badge": "NONE", "bias": bias_str}

  return {"status": "保ち合い(様子見)", "badge": "NONE", "bias": bias_str}


# ==========================================
# 3. チャート生成（文字拡大・見切れ防止版）
# ==========================================
def draw_chart(df, code_clean, stock_name, status_text, is_top=False):
  plot_df = df.tail(35).copy()
  fig, ax = plt.subplots(figsize=(6.5, 3.3), facecolor="#161618")
  ax.set_facecolor("#161618")

  dates = [mdates.date2num(d) for d in plot_df.index]

  # 5日線＝赤、20日線＝青
  ax.plot(
      dates,
      plot_df["MA5"],
      color="#ff3b30",
      linewidth=1.6,
      label="5MA",
      alpha=0.95,
  )
  ax.plot(
      dates,
      plot_df["MA20"],
      color="#007aff",
      linewidth=1.6,
      label="20MA",
      alpha=0.95,
  )

  width = 0.55
  for i, (idx, row) in enumerate(plot_df.iterrows()):
    d = dates[i]
    o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
    color = "#ff453a" if c >= o else "#0a84ff"
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

  # シグナルマーカー
  for i in range(1, len(plot_df)):
    curr_row = plot_df.iloc[i]
    prev_row = plot_df.iloc[i - 1]
    d = dates[i]
    mid = (curr_row["Open"] + curr_row["Close"]) / 2.0
    is_ma5_up = curr_row["MA5"] >= prev_row["MA5"]

    if (
        curr_row["Close"] >= curr_row["Open"]
        and mid > curr_row["MA5"]
        and prev_row["Close"] <= prev_row["MA5"]
        and is_ma5_up
    ):
      ax.scatter(
          d, curr_row["Low"] * 0.992, color="#ffd60a", marker="^", s=45, zorder=5
      )
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
          s=45,
          zorder=5,
      )

  # ★保有中の場合：利確・買値・撤退ラインを描画（文字サイズ拡大）
  if code_clean in HOLDINGS:
    h = HOLDINGS[code_clean]
    entry = h["entry_price"]
    sl = h.get("stop_loss", entry * 0.97)
    tp = h.get("target_profit", entry * 1.05)

    ax.axhline(tp, color="#af52de", linestyle="--", linewidth=1.3, alpha=0.9)
    ax.axhline(entry, color="#ffd60a", linestyle=":", linewidth=1.3, alpha=0.9)
    ax.axhline(sl, color="#ff9500", linestyle="--", linewidth=1.3, alpha=0.9)

    last_d = dates[-1]
    ax.text(
        last_d + 0.3,
        tp,
        f" 利確 {int(tp):,}",
        color="#af52de",
        fontsize=11,
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        last_d + 0.3,
        entry,
        f" 買値 {int(entry):,}",
        color="#ffd60a",
        fontsize=11,
        va="center",
        fontweight="bold",
    )
    ax.text(
        last_d + 0.3,
        sl,
        f" 撤退 {int(sl):,}",
        color="#ff9500",
        fontsize=11,
        va="top",
        fontweight="bold",
    )

    # 右端の余白を確保して文字欠けを防止
    ax.set_xlim(dates[0] - 0.5, dates[-1] + 5.5)

    y_min = min(plot_df["Low"].min(), sl) * 0.985
    y_max = max(plot_df["High"].max(), tp) * 1.015
    ax.set_ylim(y_min, y_max)

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
  if is_top:
    fig.savefig("chart.png", dpi=150, bbox_inches="tight", facecolor="#161618")
  plt.close(fig)


# ==========================================
# 4. メイン処理
# ==========================================
def main():
  now_jst = datetime.now(JST).strftime("%m/%d %H:%M JST")
  stock_results = []
  stock_dfs = {}
  swing_universe = UNIVERSE["swing"]

  for symbol, name in swing_universe.items():
    code_clean = symbol.replace(".T", "")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")

    if df.empty or len(df) < 30:
      continue

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    res = evaluate_stock(df, code_clean)
    c_price = df.iloc[-1]["Close"]
    price_str = (
        f"{c_price:,.1f}円" if c_price < 1000 else f"{int(c_price):,}円"
    )

    is_holding = code_clean in HOLDINGS
    entry_price_val = (
        HOLDINGS[code_clean]["entry_price"] if is_holding else None
    )

    stock_results.append({
        "code": code_clean,
        "name": name,
        "price": price_str,
        "bias": res["bias"],
        "status": res["status"],
        "badge": res["badge"],
        "is_holding": is_holding,
        "entry_price": entry_price_val,
    })
    stock_dfs[code_clean] = (df, name, res["status"])

  exit_stock = next((s for s in stock_results if s["badge"] == "EXIT"), None)
  signal_stock = next(
      (s for s in stock_results if s["badge"] in ["BUY", "SELL"]), None
  )
  hold_stock = next((s for s in stock_results if s["badge"] == "HOLD"), None)
  top_stock = exit_stock or signal_stock or hold_stock or stock_results[0]

  for code, (df, name, status) in stock_dfs.items():
    is_top = code == top_stock["code"]
    draw_chart(df, code, name, status, is_top=is_top)

  output_data = {
      "updated_at": now_jst,
      "top_stock": top_stock,
      "stocks": stock_results,
  }

  with open("result.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
  with open("result_all.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

  print(
      f"[{now_jst}] 7銘柄更新完了 | トップ: {top_stock['name']} ->"
      f" {top_stock['status']}"
  )


if __name__ == "__main__":
  main()
