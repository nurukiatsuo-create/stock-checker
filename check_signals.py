import json
from datetime import datetime
import pandas as pd
import pytz
import yfinance as yf

# 監視対象銘柄と現在の保有状況（Colabと同一）
# 保有している銘柄は True、監視のみは False
PORTFOLIO = {
    "6758.T": {"name": "ソニーG",   "holding": True},   # 保有中
    "7012.T": {"name": "川崎重工", "holding": False},  # 監視中
    "8306.T": {"name": "三菱UFJ",  "holding": False},  # 監視中
    "7011.T": {"name": "三菱重工", "holding": False},  # 監視中
    "9107.T": {"name": "川崎汽船", "holding": False}   # 監視中
}

def analyze_stock(df, is_holding):
    # 移動平均線
    for p in [5, 20, 100]:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
        df[f"Slope_{p}"] = df[f"SMA_{p}"].diff().fillna(0)

    df["Body_Center"] = (df["Open"] + df["Close"]) / 2
    df["Is_Yang"] = df["Close"] >= df["Open"]
    df["Is_Yin"] = df["Close"] < df["Open"]
    df["Bias_20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]) * 100

    latest = df.iloc[-1]
    
    # 判定フラグ
    sub_ppp = (latest["SMA_5"] > latest["SMA_20"]) and (latest["Slope_20"] > 0)
    above_100 = latest["Close"] > latest["SMA_100"]
    bias_ok = latest["Bias_20"] <= 4.0
    kahanshin = (latest["Slope_5"] >= 0) and latest["Is_Yang"] and (latest["Body_Center"] > latest["SMA_5"])
    gyaku_kahanshin = latest["Is_Yin"] and (latest["Body_Center"] < latest["SMA_5"])

    bias_val = f"{latest['Bias_20']:+.1f}%" if pd.notnull(latest['Bias_20']) else "-"

    # --- Colab完全準拠判定ロジック ---
    if is_holding:
        # 【保有中の銘柄】
        if gyaku_kahanshin or latest["Close"] < latest["SMA_5"]:
            return {
                "action": "【★成行売り決済】",
                "badge": "EXIT",
                "reason": "逆下半身(手じまい)",
                "priority": 1,
                "bias": bias_val
            }
        else:
            return {
                "action": "【保有継続】",
                "badge": "WAIT",
                "reason": "上昇維持",
                "priority": 4,
                "bias": bias_val
            }
    else:
        # 【未保有（監視）の銘柄】
        if sub_ppp and above_100 and bias_ok and kahanshin:
            return {
                "action": "【★成行買い新規】",
                "badge": "BUY",
                "reason": "下半身(買いシグナル)",
                "priority": 2,
                "bias": bias_val
            }
        elif sub_ppp and (latest["Bias_20"] > 4.0):
            return {
                "action": "【見送り(高値圏)】",
                "badge": "WAIT",
                "reason": "20日線乖離4%超(押し目待ち)",
                "priority": 3,
                "bias": bias_val
            }
        elif sub_ppp:
            return {
                "action": "【押し目待ち】",
                "badge": "WAIT",
                "reason": "下半身形成待ち",
                "priority": 3,
                "bias": bias_val
            }
        else:
            return {
                "action": "【待機(手出し無用)】",
                "badge": "NONE",
                "reason": "下降または保ち合い",
                "priority": 5,
                "bias": bias_val
            }

def main():
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst)
    results = []

    for code, info in PORTFOLIO.items():
        try:
            ticker = yf.Ticker(code)
            df = ticker.history(period="1y")
            if df.empty or len(df) < 5:
                continue

            price_val = int(df.iloc[-1]["Close"])
            analysis = analyze_stock(df, info["holding"])

            results.append({
                "code": code.replace(".T", ""),
                "name": info["name"],
                "price": f"{price_val:,}円",
                "bias": analysis["bias"],
                "status": analysis["action"].replace("【", "").replace("】", ""),
                "badge": analysis["badge"],
                "priority": analysis["priority"],
                "holding": info["holding"]
            })
        except Exception as e:
            print(f"Error {code}: {e}")

    # アクションが必要な銘柄（決済・買い）を最優先にソート
    results.sort(key=lambda x: x["priority"])

    output_data = {
        "updated_at": now_jst.strftime("%m/%d %H:%M JST"),
        "stocks": results
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("result.json をColab完全準拠で正常に生成しました。")

if __name__ == "__main__":
    main()
