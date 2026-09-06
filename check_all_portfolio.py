import json
from datetime import datetime
import pandas as pd
import pytz
import yfinance as yf

# 監視ユニバース定義
UNIVERSE = {
    # 1. 相場流スイング（大型株）
    "swing": {
        "7012.T": "川崎重工",
        "7011.T": "三菱重工",
        "8306.T": "三菱UFJ"
    },
    # 2. シクリカルバリュー（景気循環・割安）
    "value": {
        "5386.T": "鶴弥",
        "5446.T": "北越メタル",
        "5923.T": "高田機工",
        "5928.T": "アルメタックス",
        "6022.T": "赤阪鉄",
        "8144.T": "電響社"
    },
    # 3. 成長株・フィジカルAI
    "growth": {
        "215A.T": "タイミー",
        "4377.T": "ワンキャリア",
        "4419.T": "フィナテキスト",
        "5253.T": "カバー",
        "5016.T": "ＪＸ金属",
        "6954.T": "ファナック"
    }
}

def analyze_swing(df):
    for p in [5, 20, 100]:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
        df[f"Slope_{p}"] = df[f"SMA_{p}"].diff().fillna(0)
    df["Body_Center"] = (df["Open"] + df["Close"]) / 2
    df["Is_Yang"] = df["Close"] >= df["Open"]
    df["Is_Yin"] = df["Close"] < df["Open"]
    df["Bias_20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]) * 100
    
    latest = df.iloc[-1]
    sub_ppp = (latest["SMA_5"] > latest["SMA_20"]) and (latest["Slope_20"] > 0)
    above_100 = (latest["Close"] > latest["SMA_100"])
    bias_ok = (latest["Bias_20"] <= 4.0)
    kahanshin = (latest["Slope_5"] >= 0) and latest["Is_Yang"] and (latest["Body_Center"] > latest["SMA_5"])
    buy_signal = sub_ppp and above_100 and bias_ok and kahanshin
    exit_signal = latest["Is_Yin"] and (latest["Body_Center"] < latest["SMA_5"])
    
    bias_val = f"{latest['Bias_20']:+.1f}%" if pd.notnull(latest['Bias_20']) else "-"
    if buy_signal:
        return {"text": "★買いシグナル", "badge": "BUY", "val": bias_val}
    elif exit_signal:
        return {"text": "▼手じまい警告", "badge": "EXIT", "val": bias_val}
    elif sub_ppp and (latest["Bias_20"] > 4.0):
        return {"text": "高値圏(待機)", "badge": "WAIT", "val": bias_val}
    elif sub_ppp:
        return {"text": "押し目待ち", "badge": "WAIT", "val": bias_val}
    else:
        return {"text": "手出し無用", "badge": "NONE", "val": bias_val}

def analyze_value(df, ticker):
    try:
        df_w = df.resample('W').last().dropna(subset=['Close'])
        for p in [13, 26]:
            df_w[f"SMA_W{p}"] = df_w["Close"].rolling(window=min(p, len(df_w)), min_periods=1).mean()
        latest_w = df_w.iloc[-1]
        prev_w = df_w.iloc[-2] if len(df_w) > 1 else latest_w
        bottoming = (latest_w["Close"] >= latest_w.get("SMA_W13", 0)) and \
                    (latest_w.get("SMA_W13", 0) >= prev_w.get("SMA_W13", 0))
    except Exception:
        bottoming = False

    pbr_val = None
    try:
        info = ticker.info or {}
        pbr_raw = info.get('priceToBook')
        if pbr_raw is not None:
            pbr_val = float(pbr_raw)
    except Exception:
        pass

    pbr_str = f"PBR {pbr_val:.2f}倍" if pbr_val is not None else "PBR -"

    if pbr_val is not None and pbr_val <= 0.8:
        if bottoming:
            return {"text": "★週足底打ち反転", "badge": "BUY", "val": pbr_str}
        else:
            return {"text": "底練り中(待機)", "badge": "WAIT", "val": pbr_str}
    else:
        if bottoming:
            return {"text": "週足反転の兆し", "badge": "WATCH", "val": pbr_str}
        return {"text": "様子見", "badge": "NONE", "val": pbr_str}

def analyze_growth(df):
    for p in [50, 200]:
        df[f"SMA_{p}"] = df["Close"].rolling(window=min(p, len(df)), min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Volume_SMA5"] = df["Volume"].rolling(window=min(5, len(df)), min_periods=1).mean()
    
    window_high = min(250, len(df))
    high_n = df["High"].tail(window_high).max()
    latest = df.iloc[-1]
    
    trend_up = (latest["SMA_50"] >= latest["SMA_200"]) if "SMA_200" in latest else True
    is_near_high = (latest["Close"] / high_n >= 0.95) if (high_n and high_n > 0) else False
    is_volume_up = (latest["Volume"] > latest["Volume_SMA5"] * 1.3) if latest["Volume_SMA5"] > 0 else False
    
    high_str = f"高値比 {latest['Close']/high_n*100:.1f}%" if (high_n and high_n > 0) else "-"
    
    if trend_up and is_near_high and is_volume_up:
        return {"text": "★新高値ブレイク", "badge": "BUY", "val": high_str}
    elif trend_up and (latest["Close"] >= latest["EMA_20"]):
        return {"text": "上昇トレンド中", "badge": "BUY", "val": high_str}
    else:
        return {"text": "押し目・様子見", "badge": "WAIT", "val": high_str}

def analyze_market_all():
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst)
    all_results = {
        "updated_at": now_jst.strftime("%m/%d %H:%M"),
        "swing": [],
        "value": [],
        "growth": []
    }
    
    print(f"[{all_results['updated_at']}] ポートフォリオ解析を開始します...")

    for category, stocks in UNIVERSE.items():
        print(f"\n--- カテゴリー: {category} ---")
        category_results = []
        for code, meta_name in stocks.items():
            try:
                ticker = yf.Ticker(code)
                df = ticker.history(period="1y")
                if df.empty or len(df) < 5:
                    print(f"スキップ: {code} ({meta_name}) - データ不足")
                    continue
                
                price_val = int(df.iloc[-1]['Close'])
                price_str = f"{price_val:,}円"
                
                if category == "swing":
                    sig = analyze_swing(df)
                elif category == "value":
                    sig = analyze_value(df, ticker)
                elif category == "growth":
                    sig = analyze_growth(df)
                
                category_results.append({
                    "code": code.replace(".T", ""),
                    "name": meta_name,
                    "price": price_str,
                    "sig_val": sig["val"],
                    "status": sig["text"],
                    "badge": sig["badge"]
                })
                print(f"完了: {code} ({meta_name}) -> {sig['text']} ({sig['badge']})")
            except Exception as e:
                print(f"エラー発生 {code} ({meta_name}): {e}")
        
        all_results[category] = category_results

    with open("result_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\nresult_all.json を正常に生成しました。")

if __name__ == "__main__":
    analyze_market_all()

