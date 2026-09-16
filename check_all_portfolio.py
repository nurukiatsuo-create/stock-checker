import os
import json
import datetime
import shutil
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

# ==========================================
# 1. 監視・保有銘柄設定
# ==========================================
HOLDINGS = {
    "9107": {"entry_price": 3661, "type": "buy", "shares": 100},   # 三菱UFJ（買）
}

WATCH_LIST = [
    {"code": "8306", "name": "三菱UFJ"},
    {"code": "6326", "name": "クボタ"},
    {"code": "7269", "name": "スズキ"},
    {"code": "7011", "name": "三菱重工"},
    {"code": "7012", "name": "川崎重工"},
    {"code": "9107", "name": "川崎汽船"},
    {"code": "8002", "name": "丸紅"},
    {"code": "1326", "name": "SPDRゴールド"},
]

def analyze_stock(ticker_code, name):
    try:
        symbol = f"{ticker_code}.T"
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        if len(df) < 25:
            return None

        close = df['Close'].squeeze()
        df['SMA5'] = close.rolling(window=5).mean()
        df['SMA20'] = close.rolling(window=20).mean()

        valid_df = df.dropna(subset=['SMA5', 'SMA20']).copy()
        if len(valid_df) < 5:
            return None

        valid_close = valid_df['Close'].squeeze()
        curr_close = float(valid_close.iloc[-1])
        prev_close = float(valid_close.iloc[-2])
        sma5_curr = float(valid_df['SMA5'].iloc[-1])
        sma5_prev = float(valid_df['SMA5'].iloc[-2])
        sma20_curr = float(valid_df['SMA20'].iloc[-1])

        is_holding = ticker_code in HOLDINGS
        entry_info = HOLDINGS.get(ticker_code, {})
        entry_price = entry_info.get("entry_price")
        pos_type = entry_info.get("type", "buy")

        pnl_per_share = None
        pnl_rate = None

        if is_holding and entry_price:
            if pos_type == "buy":
                pnl_per_share = round(curr_close - entry_price, 2)
                pnl_rate = round((pnl_per_share / entry_price) * 100, 2)
            else:
                pnl_per_share = round(entry_price - curr_close, 2)
                pnl_rate = round((pnl_per_share / entry_price) * 100, 2)

        recent_high = float(valid_close.tail(20).max())
        recent_low = float(valid_close.tail(20).min())

        status = "待機"
        tp_price = None
        sl_price = None

        if is_holding:
            if pos_type == "buy":
                tp_price = recent_high
                sl_price = round(entry_price * 0.975, 1)
                if curr_close < sma5_curr and curr_close < prev_close:
                    status = "手仕舞"
                elif curr_close >= tp_price * 0.995:
                    status = "高値警戒"
                else:
                    status = "継続保有"
            else:
                tp_price = recent_low
                sl_price = round(entry_price * 1.025, 1)
                if curr_close > sma5_curr and curr_close > prev_close:
                    status = "返済買"
                elif curr_close <= tp_price * 1.005:
                    status = "底値警戒"
                else:
                    status = "空売保有"
        else:
            if curr_close > sma5_curr and sma5_curr > sma5_prev and curr_close > sma20_curr:
                status = "下半身(買)"
            elif curr_close < sma5_curr and sma5_curr < sma5_prev and curr_close < sma20_curr:
                status = "逆下半身(空売)"
            elif abs(curr_close - sma20_curr) / sma20_curr < 0.015:
                status = "反発待"

        plot_df = valid_df.tail(40).copy()
        mc = mpf.make_marketcolors(up='#ff453a', down='#0a84ff', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#27272a', facecolor='#141416')

        addplots = [
            mpf.make_addplot(plot_df['SMA5'], color='#ff453a', width=1.5),
            mpf.make_addplot(plot_df['SMA20'], color='#0a84ff', width=2.0)
        ]

        chart_filename = f"chart_{ticker_code}.png"

        if is_holding and entry_price:
            if pos_type == "buy":
                h_lines = [tp_price, entry_price, sl_price]
                h_colors = ['#bf5af2', '#ffd60a', '#ff9f0a']
            else:
                h_lines = [sl_price, entry_price, tp_price]
                h_colors = ['#ff9f0a', '#ffd60a', '#bf5af2']
            
            mpf.plot(plot_df, type='candle', style=s, addplot=addplots,
                     hlines=dict(hlines=h_lines, colors=h_colors, linestyle='--'),
                     savefig=chart_filename, figsize=(7.2, 4.2), tight_layout=True)
        else:
            mpf.plot(plot_df, type='candle', style=s, addplot=addplots,
                     savefig=chart_filename, figsize=(7.2, 4.2), tight_layout=True)

        return {
            "code": ticker_code,
            "name": name,
            "price": f"{int(curr_close):,}円" if not np.isnan(curr_close) else "---円",
            "price_num": curr_close,
            "status": status,
            "is_holding": is_holding,
            "pos_type": pos_type if is_holding else None,
            "entry_price": entry_price,
            "pnl_per_share": pnl_per_share,
            "pnl_rate": pnl_rate,
            "chart_file": chart_filename
        }
    except Exception as e:
        print(f"Error processing {ticker_code}: {e}")
        return None

def main():
    results = []
    for item in WATCH_LIST:
        data = analyze_stock(item["code"], item["name"])
        if data:
            results.append(data)

    # 日本時間 (JST: UTC+9) で現在時刻を記録
    jst = datetime.timezone(datetime.timedelta(hours=9))
    output = {
        "updated_at": datetime.datetime.now(jst).strftime("%m/%d %H:%M JST"),
        "stocks": results
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if os.path.exists("chart_8306.png"):
        shutil.copy("chart_8306.png", "chart.png")
    elif results and os.path.exists(results[0]["chart_file"]):
        shutil.copy(results[0]["chart_file"], "chart.png")

if __name__ == "__main__":
    main()
