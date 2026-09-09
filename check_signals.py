import json
from datetime import datetime
import pandas as pd
import pytz
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 監視対象銘柄と保有状況（Colab完全準拠）
PORTFOLIO = {
    "6758.T": {"name": "ソニーG",   "holding": False},  # 監視中
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
    """相場流チャート（直近35営業日）を描画し、下半身・逆下半身マークをプロット"""
    for p in [5, 20, 100]:
        if f"SMA_{p}" not in df.columns:
            df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
            df[f"Slope_{p}"] = df[f"SMA_{p}"].diff().fillna(0)

    df["Body_Center"] = (df["Open"] + df["Close"]) / 2
    df["Is_Yang"] = df["Close"] >= df["Open"]
    df["Is_Yin"] = df["Close"] < df["Open"]

    # 下半身・逆下半身の判定
    kahanshin = df["Is_Yang"] & (df["Body_Center"] > df["SMA_5"]) & (df["Slope_5"] >= 0)
    gyaku_kahanshin = df["Is_Yin"] & (df["Body_Center"] < df["SMA_5"])

    # 転換初日フラグ
    buy_turn = kahanshin & (~kahanshin.shift(1).fillna(False))
    exit_turn = gyaku_kahanshin & (~gyaku_kahanshin.shift(1).fillna(False))

    # 20日線跨ぎ（突破 / 割り込み）
    cross_20_buy = buy_turn & (df["Low"] <= df["SMA_20"]) & (df["Close"] >= df["SMA_20"])
    cross_20_exit = exit_turn & (df["High"] >= df["SMA_20"]) & (df["Close"] <= df["SMA_20"])

    df["Buy_Turn"] = buy_turn
    df["Exit_Turn"] = exit_turn
    df["Cross_20_Buy"] = cross_20_buy
    df["Cross_20_Exit"] = cross_20_exit

    sub = df.tail(35).copy().reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#161618")
    ax.set_facecolor("#161618")

    # 移動平均線（赤: 5日線, 緑: 20日線, 青: 100日線）
    ax.plot(sub.index, sub["SMA_5"], color="#ff3b30", linewidth=1.8, alpha=0.9)
    ax.plot(sub.index, sub["SMA_20"], color="#34c759", linewidth=2.0, alpha=0.95)
    if "SMA_100" in sub.columns and sub["SMA_100"].notna().any():
        ax.plot(sub.index, sub["SMA_100"], color="#0a84ff", linewidth=1.5, alpha=0.8)

    # ローソク足
    width = 0.6
    up = sub[sub["Close"] >= sub["Open"]]
    down = sub[sub["Close"] < sub["Open"]]

    ax.vlines(up.index, up["Low"], up["High"], color="#ff453a", linewidth=1.2)
    ax.bar(up.index, up["Close"] - up["Open"], bottom=up["Open"], width=width, color="#ff453a", edgecolor="#ff453a")

    ax.vlines(down.index, down["Low"], down["High"], color="#64d2ff", linewidth=1.2)
    ax.bar(down.index, down["Open"] - down["Close"], bottom=down["Close"], width=width, color="#64d2ff", edgecolor="#64d2ff")

    # シグナルマークのプロット
    price_range = max(1.0, sub["High"].max() - sub["Low"].min())
    offset = price_range * 0.04

    # ① 通常の下半身（赤▲）
    normal_buys = sub[sub["Buy_Turn"] & (~sub["Cross_20_Buy"])]
    if not normal_buys.empty:
        ax.scatter(normal_buys.index, normal_buys["Low"] - offset, marker="^", color="#ff3b30", s=60, zorder=5)

    # ② 20日線突破・重要下半身（黄縁の特大赤▲）
    key_buys = sub[sub["Cross_20_Buy"]]
    if not key_buys.empty:
        ax.scatter(key_buys.index, key_buys["Low"] - offset, marker="^", color="#ff3b30", edgecolors="#ffd60a", linewidths=1.5, s=110, zorder=6)

    # ③ 通常の逆下半身（水色▼）
    normal_exits = sub[sub["Exit_Turn"] & (~sub["Cross_20_Exit"])]
    if not normal_exits.empty:
        ax.scatter(normal_exits.index, normal_exits["High"] + offset, marker="v", color="#64d2ff", s=60, zorder=5)

    # ④ 20日線割れ・重要手じまい（黄縁の特大青▼）
    key_exits = sub[sub["Cross_20_Exit"]]
    if not key_exits.empty:
        ax.scatter(key_exits.index, key_exits["High"] + offset, marker="v", color="#007aff", edgecolors="#ffd60a", linewidths=1.5, s=110, zorder=6)

    # 軸・余白設定
    ax.set_ylim(sub["Low"].min() - offset * 2.5, sub["High"].max() + offset * 2.5)
    ax.tick_params(colors="#8e8e93", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2c2c2e")
    ax.grid(True, color="#2c2c2e", linestyle="--", linewidth=0.5)

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

            code_num = code.replace(".T", "")
            results.append({
                "code": code_num,
                "name": info["name"],
                "price": f"{price_val:,}円",
                "bias": analysis["bias"],
                "status": analysis["action"].replace("【", "").replace("】", ""),
                "badge": analysis["badge"],
                "priority": analysis["priority"],
                "holding": info["holding"]
            })

            # 各銘柄ごとの個別チャート画像を生成
            generate_chart(df, save_path=f"chart_{code_num}.png")
        except Exception as e:
            print(f"Error {code}: {e}")

    results.sort(key=lambda x: x["priority"])

    # 最優先銘柄のチャート（デフォルト用）
    if results:
        top_code_num = results[0]["code"]
        generate_chart(df_dict[top_code_num + ".T"], save_path="chart.png")

    output_data = {
        "updated_at": now_jst.strftime("%m/%d %H:%M JST"),
        "top_stock": results[0] if results else None,
        "stocks": results
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("5銘柄のシグナルマーク付きチャートと result.json を正常に生成しました。")

if __name__ == "__main__":
    main()
