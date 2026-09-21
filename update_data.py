import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

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
    is_yin = cc < co
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
    is_gyaku = (
        is_yin
        and (mid < m5)
        and (cc < m5)
        and (pc >= pm5)
        and (m5 <= pm5)
    )
    is_mono = (
        (cl <= m20 * 1.015) and (cc > m20) and is_yang and (m20 >= pm20)
    )

    signal = None
    score = 0

    if is_shita and (m20 >= pm20) and (cc >= m20):
      signal = "BUY"
      score = 80 if is_mono else 50
    elif is_gyaku and (m20 <= pm20) and (cc <= m20):
      signal = "SHORT"
      score = 50

    if signal == "BUY":
      score += min(max(int(m20_slope * 20), 0), 20)
      hl = ch - cl
      if hl > 0:
        score += int((abs(cc - co) / hl) * 10)
      if abs(bias_20) > 8.0:
        score -= 15

      if score < MIN_SCORE:
        signal = None

    # チャート用データ（直近40本）
    recent = d.iloc[-40:]
    bars = []
    for idx_dt, row in recent.iterrows():
      bars.append({
          "date": idx_dt.strftime("%m/%d"),
          "open": float(row["Open"]),
          "high": float(row["High"]),
          "low": float(row["Low"]),
          "close": float(row["Close"]),
          "ma5": float(row["MA5"]) if not np.isnan(row["MA5"]) else None,
          "ma20": float(row["MA20"]) if not np.isnan(row["MA20"]) else None,
      })

    output_data["stocks"][sym] = {
        "num": item["num"],
        "name": item["name"],
        "target": item["target"],
        "price": cc,
        "score": score,
        "signal": signal,
        "bars": bars,
    }
  except Exception as e:
    print(f"Error {sym}: {e}")

with open("stock_data.json", "w", encoding="utf-8") as f:
  json.dump(output_data, f, ensure_ascii=False, indent=2)

print("stock_data.json の生成が完了しました。")
