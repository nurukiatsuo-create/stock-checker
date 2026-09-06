import json
from datetime import datetime
import pandas as pd
import pytz
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 監視対象銘柄と保有状況
PORTFOLIO = {
    "6758.T": {"name": "ソニーG",   "holding": True},   # 保有中
    "7012.T": {"name": "川崎重工", "holding": False},  # 監視中
    "8306.T": {"name": "三菱UFJ",  "holding": False},  # 監視中
    "7011.T": {"name": "三菱重工", "holding": False},  # 監視中
    "9107.T": {"name": "川崎汽船", "holding": False}   # 監視中
}

def analyze_stock(df, is_holding):
    for p in [5, 20, 100]:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
        df[f"Slope_{p}"] = df[f"SMA_{p}"].diff().fillna(0)

    df["Body_Center"] = (df["Open"] + df["Close"]) / 2
    df["Is_Yang"] = df["Close"] >= df["Open"]
    df["Is_Yin"] = df["Close"] < df["Open"]
    df["Bias_20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]) * 100

    latest = df.iloc[-1]
    
    sub_ppp = (latest["SMA_5"] > latest["SMA_20"]) and (latest["Slope_20"] > 0)
    above_100 = latest["Close"] > latest["SMA_100"]
    bias_ok = latest["Bias_20"] <= 4.0
    kahanshin = (latest["Slope_5"] >= 0) and latest["Is_Yang"] and (latest["Body_Center"] > latest["SMA_5"])
    gyaku_kahanshin = latest["Is_Yin"] and (latest["Body_Center"] < latest["SMA_5"])

    bias_val = f"{latest['Bias_20']:+.1f}%" if pd.notnull(latest['Bias_20']) else "-"

    if is_holding:
        if gyaku_kahanshin or latest["Close"] < latest["SMA_5"]:
            return {"action": "【★成行売り決済】", "badge": "EXIT", "reason": "逆下半身(手じまい)", "priority": 1, "bias": bias_val}
        else:
            return {"action": "【保有継続】", "badge": "WAIT", "reason": "上昇維持", "priority": 4, "bias": bias_val}
    else:
        if sub_ppp and above_100 and bias_ok and kahanshin:
            return {"action": "【★成行買い新規】", "badge": "BUY", "reason": "下半身(買いシグナル)", "priority": 2, "bias": bias_val}
        elif sub_ppp and (latest["Bias_20"] > 4.0):
            return {"action": "【見送り(高値圏)】", "badge": "WAIT", "reason": "20日線乖離4%超", "priority": 3, "bias": bias_val}
        elif sub_ppp:
            return {"action": "【押し目待ち】", "badge": "WAIT", "reason": "下半身待ち", "priority": 3, "bias": bias_val}
        else:
            return {"action": "【待機(手出し無用)】", "badge": "NONE", "reason": "下降または保ち合い", "priority": 5, "bias": bias_val}

def generate_chart(df, save_path="chart.png"):
    """最優先銘柄の相場流チャート（直近35日分）を画像化"""
    sub = df.tail(35).copy().reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#161618")
    ax.set_facecolor("#161618")

    # 移動平均線（赤: 5日線, 緑: 20日線, 青: 100日線）
    ax.plot(sub.index, sub["SMA_5"], color="#ff3b30", label="5MA", linewidth=2.0)
    ax.plot(sub.index, sub["SMA_20"], color="#34c759", label="20MA", linewidth=2.0)
    if "SMA_100" in sub.columns and sub["SMA_100"].notna().any():
        ax.plot(sub.index, sub["SMA_100"], color="#0a84ff", label="100MA", linewidth=1.5)

    # ローソク足（赤: 陽線, 水色: 陰線）
    width = 0.6
    up = sub[sub["Close"] >= sub["Open"]]
    down = sub[sub["Close"] < sub["Open"]]

    ax.vlines(up.index, up["Low"], up["High"], color="#ff453a", linewidth=1.2)
    ax.bar(up.index, up["Close"] - up["Open"], bottom=up["Open"], width=width, color="#ff453a", edgecolor="#ff453a")

    ax.vlines(down.index, down["Low"], down["High"], color="#64d2ff", linewidth=1.2)
    ax.bar(down.index, down["Open"] - down["Close"], bottom=down["Close"], width=width, color="#64d2ff", edgecolor="#64d2ff")

    # 軸・グリッド
    ax.tick_params(colors="#8e8e93", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2c2c2e")
    ax.grid(True, color="#2c2c2e", linestyle="--", linewidth=0.5)

    # 日付ラベル
    date_col = sub.columns[0]
    step = max(1, len(sub) // 5)
    ticks = list(range(0, len(sub), step))
    tick_labels = [sub.loc[i, date_col].strftime('%m/%d') if hasattr(sub.loc[i, date_col], 'strftime') else str(sub.loc[i, date_col])[:5] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def main():
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst)
    results = []
    df_dict = {}

    for code, info in PORTFOLIO.items():
        try:
            ticker = yf.Ticker(code)
            df = ticker.history(period="1y")
            if df.empty or len(df) < 5:
                continue

            price_val = int(df.iloc[-1]["Close"])
            analysis = analyze_stock(df, info["holding"])
            df_dict[code] = df

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

    # 最優先銘柄（ソニーGなど決済・新規買い対象）を先頭にソート
    results.sort(key=lambda x: x["priority"])

    # 優先度1位の銘柄のチャート画像を生成
    if results:
        top_code = results[0]["code"] + ".T"
        if top_code in df_dict:
            generate_chart(df_dict[top_code])

    output_data = {
        "updated_at": now_jst.strftime("%m/%d %H:%M JST"),
        "top_stock": results[0] if results else None,
        "stocks": results
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("result.json と chart.png を正常に生成しました。")

if __name__ == "__main__":
    main()
