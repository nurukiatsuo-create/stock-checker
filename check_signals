import json
from datetime import datetime
import pytz
import yfinance as yf

# 監視ユニバース定義（優先順位順）
TARGET_UNIVERSE = {
    "7012.T": {"name": "川崎重工", "sec": "重工", "rank": 1},
    "8306.T": {"name": "三菱UFJ", "sec": "銀行", "rank": 2},
    "7011.T": {"name": "三菱重工", "sec": "重工", "rank": 3},
    "9107.T": {"name": "川崎汽船", "sec": "海運", "rank": 4},
    "6758.T": {"name": "ソニーG", "sec": "電機", "rank": 5},
}


def analyze_market():
  jst = pytz.timezone("Asia/Tokyo")
  now_jst = datetime.now(jst)

  results = []

  for code, meta in TARGET_UNIVERSE.items():
    try:
      df = yf.Ticker(code).history(period="1y")
      if len(df) < 100:
        continue
      df.index = df.index.tz_localize(None)

      for p in [5, 10, 20, 50, 100]:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p).mean()
        df[f"Slope_{p}"] = df[f"SMA_{p}"].diff()

      df["Body_Center"] = (df["Open"] + df["Close"]) / 2
      df["Is_Yang"] = df["Close"] > df["Open"]
      df["Is_Yin"] = df["Close"] < df["Open"]
      df["Bias_20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]) * 100

      ma_max = df[["SMA_5", "SMA_10", "SMA_20"]].max(axis=1)
      ma_min = df[["SMA_5", "SMA_10", "SMA_20"]].min(axis=1)
      is_box = (df["SMA_20"].pct_change(5).abs().iloc[-1] < 0.005) and (
          (ma_max.iloc[-1] - ma_min.iloc[-1]) / df["SMA_20"].iloc[-1] < 0.02
      )

      latest = df.iloc[-1]
      price = float(latest["Close"])

      sub_ppp = bool(
          (latest["SMA_5"] > latest["SMA_10"])
          and (latest["SMA_10"] > latest["SMA_20"])
          and (latest["Slope_20"] > 0)
      )
      above_100 = bool(latest["Close"] > latest["SMA_100"])
      bias_ok = bool(latest["Bias_20"] <= 4.0)
      kahanshin = bool(
          (latest["Slope_5"] >= 0)
          and latest["Is_Yang"]
          and (latest["Body_Center"] > latest["SMA_5"])
      )

      buy_signal = bool(
          sub_ppp and above_100 and bias_ok and kahanshin and (not is_box)
      )
      exit_signal = bool(
          latest["Is_Yin"] and (latest["Body_Center"] < latest["SMA_5"])
      )

      if buy_signal:
        status = "★買いシグナル"
        badge = "BUY"
      elif exit_signal:
        status = "▼手じまい警告"
        badge = "EXIT"
      elif sub_ppp and (latest["Bias_20"] > 4.0):
        status = "高値圏(待機)"
        badge = "WAIT"
      elif sub_ppp:
        status = "押し目待ち"
        badge = "WAIT"
      else:
        status = "手出し無用"
        badge = "NONE"

      results.append({
          "rank": meta["rank"],
          "code": code.replace(".T", ""),
          "name": meta["name"],
          "price": f"{int(price):,}円",
          "bias_20": f"{latest['Bias_20']:+.1f}%",
          "status": status,
          "badge": badge,
      })

    except Exception as e:
      print(f"Error {code}: {e}")

  output_data = {
      "updated_at": now_jst.strftime("%m/%d %H:%M"),
      "stocks": results,
  }

  with open("result.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

  print("result.json を正常に生成しました。")


if __name__ == "__main__":
  analyze_market()
