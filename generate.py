import base64
import io
import json
import time
import matplotlib

matplotlib.use("Agg")  # サーバー用描画モード
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 新8銘柄・枠厳守配分
UNIVERSE = [
    {"code": "8306.T", "num": "8306", "name": "三菱UFJ", "target": 300},
    {"code": "7182.T", "num": "7182", "name": "ゆうちょ銀行", "target": 400},
    {"code": "8393.T", "num": "8393", "name": "宮崎銀行", "target": 200},
    {"code": "8058.T", "num": "8058", "name": "三菱商事", "target": 200},
    {"code": "6981.T", "num": "6981", "name": "村田製作所", "target": 200},
    {"code": "4502.T", "num": "4502", "name": "武田薬品", "target": 100},
    {"code": "2768.T", "num": "2768", "name": "双日", "target": 100},
    {"code": "6324.T", "num": "6324", "name": "ハーモニック", "target": 100},
]

MIN_SCORE = 60
output_data = {"updated_at": time.strftime("%m/%d %H:%M JST"), "stocks": {}}


def make_chart_png(df):
  recent = df.iloc[-35:].copy()
  fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140, facecolor="#1c1c1e")
  ax.set_facecolor("#1c1c1e")

  dates = [d.strftime("%m/%d") for d in recent.index]
  x = np.arange(len(dates))

  # ローソク足
  for i in range(len(recent)):
    bar = recent.iloc[i]
    o, c, h, l = bar["Open"], bar["Close"], bar["High"], bar["Low"]
    color = "#ff453a" if c >= o else "#0a84ff"

    ax.vlines(x[i], l, h, color=color, linewidth=1.0)
    bot = min(o, c)
    height = max(abs(c - o), (h - l) * 0.02)
    ax.add_patch(
        plt.Rectangle(
            (x[i] - 0.32, bot),
            0.64,
            height,
            color=color,
            fill=True,
            zorder=3,
        )
    )

  # 移動平均線
  ax.plot(
      x,
      recent["MA5"],
      color="#ff3b30",
      linewidth=2.0,
      label="5MA",
      zorder=4,
  )
  ax.plot(
      x,
      recent["MA20"],
      color="#0a84ff",
      linewidth=2.0,
      label="20MA",
      zorder=4,
  )

  ax.grid(True, linestyle=":", color="#ffffff", alpha=0.15)
  ax.tick_params(colors="#8e8e93", labelsize=8)
  for spine in ax.spines.values():
    spine.set_color("#2c2c2e")

  step = max(len(dates) // 6, 1)
  ax.set_xticks(x[::step])
  ax.set_xticklabels(dates[::step])
  plt.tight_layout()

  buf = io.BytesIO()
  plt.savefig(
      buf,
      format="png",
      bbox_inches="tight",
      facecolor=fig.get_facecolor(),
      edgecolor="none",
  )
  plt.close(fig)
  buf.seek(0)
  return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


for item in UNIVERSE:
  sym = item["code"]
  try:
    d = yf.download(
        sym, period="3mo", interval="1d", progress=False, auto_adjust=True
    )
    if d.empty or len(d) < 25:
      continue
    if isinstance(d.columns, pd.MultiIndex):
      d.columns = d.columns.get_level_values(0)

    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()

    curr, prev = d.iloc[-1], d.iloc[-2]
    co, cc = float(curr["Open"]), float(curr["Close"])
    ch, cl = float(curr["High"]), float(curr["Low"])
    pc = float(prev["Close"])
    m5, m20 = float(curr["MA5"]), float(curr["MA20"])
    pm5, pm20 = float(prev["MA5"]), float(prev["MA20"])

    is_yang = cc >= co
    mid = (co + cc) / 2.0
    m20_slope = ((m20 - pm20) / pm20) * 100 if pm20 > 0 else 0.0
    bias_20 = ((cc - m20) / m20) * 100 if m20 > 0 else 0.0

    is_shita = (
        is_yang
        and (mid > m5)
        and (cc > m5)
        and (pc <= pm5)
        and (m5 >= pm5)
    )
    is_mono = (
        (cl <= m20 * 1.015) and (cc > m20) and is_yang and (m20 >= pm20)
    )

    signal = "待機"
    score = 0
    if is_shita and (m20 >= pm20) and (cc >= m20):
      score = 80 if is_mono else 50
      score += min(max(int(m20_slope * 20), 0), 20)
      hl = ch - cl
      if hl > 0:
        score += int((abs(cc - co) / hl) * 10)
      if abs(bias_20) > 8.0:
        score -= 15
      if score >= MIN_SCORE:
        signal = "現買"

    png_data = make_chart_png(d)

    output_data["stocks"][sym] = {
        "num": item["num"],
        "name": item["name"],
        "target": item["target"],
        "price": int(cc),
        "score": f"{score}点" if score >= 50 else "--",
        "signal": signal,
        "chart_png": png_data,
    }
  except Exception as e:
    print(f"Error {sym}: {e}")

with open("stock_data.json", "w", encoding="utf-8") as f:
  json.dump(output_data, f, ensure_ascii=False, indent=2)

print("stock_data.json 生成完了")
