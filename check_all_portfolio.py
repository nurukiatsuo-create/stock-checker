import os
import json
import datetime
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
# 空売りの場合は "type": "sell" を指定します
HOLDINGS = {
    "8306": {"entry_price": 3661, "type": "buy", "shares": 100},   # 三菱UFJ（買）
    # 例：スズキを空売りした場合
    # "7269": {"entry_price": 2050, "type": "sell", "shares": 100},
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
    symbol = f"{ticker_code}.T"
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    if df.empty or len(df) < 25:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close']
    df['SMA5'] = close.rolling(window=5).mean()
    df['SMA20'] = close.rolling(window=20).mean()

    curr_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    sma5_curr = float(df['SMA5'].iloc[-1])
    sma5_prev = float(df['SMA5'].iloc[-2])
    sma20_curr = float(df['SMA20'].iloc[-1])

    is_holding = ticker_code in HOLDINGS
    entry_info = HOLDINGS.get(ticker_code, {})
    entry_price = entry_info.get("entry_price")
    pos_type = entry_info.get("type", "buy")  # "buy" or "sell"

    pnl_per_share = None
    pnl_rate = None

    # --- 損益計算 ---
    if is_holding and entry_price:
        if pos_type == "buy":
            pnl_per_share = round(curr_close - entry_price, 2)
            pnl_rate = round((pnl_per_share / entry_price) * 100, 2)
        else: # sell (空売り)
            pnl_per_share = round(entry_price - curr_close, 2)
            pnl_rate = round((pnl_per_share / entry_price) * 100, 2)

    # --- 相場流シグナル判定 ---
    recent_high = float(close.tail(20).max())
    recent_low = float(close.tail(20).min())

    status = "待機"
    tp_price = None
    sl_price = None

    if is_holding:
        if pos_type == "buy":
            tp_price = recent_high
            sl_price = entry_price * 0.975  # 撤退目安
            if curr_close < sma5_curr and curr_close < prev_close:
                status = "手仕舞"
            elif curr_close >= tp_price * 0.995:
                status = "高値警戒"
            else:
                status = "継続保有"
        else: # 空売り保有
            tp_price = recent_low
            sl_price = entry_price * 1.025  # 撤退目安
            if curr_close > sma5_curr and curr_close > prev_close:
                status = "返済買"
            elif curr_close <= tp_price * 1.005:
                status = "底値警戒"
            else:
                status = "空売保有"
    else:
        # 新規エントリー判定
        if curr_close > sma5_curr and sma5_curr > sma5_prev and curr_close > sma20_curr:
            status = "下半身(買)"
        elif curr_close < sma5_curr and sma5_curr < sma5_prev and curr_close < sma20_curr:
            status = "逆下半身(空売)"
        elif abs(curr_close - sma20_curr) / sma20_curr < 0.015:
            status = "反発待"

    # --- チャート描画 ---
    plot_df = df.tail(40).copy()
    mc = mpf.make_marketcolors(up='#ff453a', down='#0a84ff', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#27272a', facecolor='#141416')

    addplots = [
        mpf.make_addplot(plot_df['SMA5'], color='#ff453a', width=1.5),
        mpf.make_addplot(plot_df['SMA20'], color='#0a84ff', width=2.0)
    ]

    hlines_dict = None
    if is_holding and entry_price:
        if pos_type == "buy":
            # 買い：紫(目標高値), 黄(買値), 橙(撤退安値)
            hlines_dict = dict(hlines=[tp_price, entry_price, sl_price],
                               colors=['#bf5af2', '#ffd60a', '#ff9f0a'],
                               linestyle='--', linewidths=[1.2, 1.5, 1.2])
        else:
            # 空売り：橙(撤退高値), 黄(売値), 紫(返済目標安値)
            hlines_dict = dict(hlines=[sl_price, entry_price, tp_price],
                               colors=['#ff9f0a', '#ffd60a', '#bf5af2'],
                               linestyle='--', linewidths=[1.2, 1.5, 1.2])

    chart_filename = f"chart_{ticker_code}.png"
    if hlines_dict:
        mpf.plot(plot_df, type='candle', style=s, addplot=addplots, hlines=hlines_dict,
                 savefig=chart_filename, figsize=(7.2, 4.2), tight_layout=True)
    else:
        mpf.plot(plot_df, type='candle', style=s, addplot=addplots,
                 savefig=chart_filename, figsize=(7.2, 4.2), tight_layout=True)

    return {
        "code": ticker_code,
        "name": name,
        "price": f"{int(curr_close):,}円",
        "price_num": curr_close,
        "status": status,
        "is_holding": is_holding,
        "pos_type": pos_type if is_holding else None,
        "entry_price": entry_price,
        "pnl_per_share": pnl_per_share,
        "pnl_rate": pnl_rate,
        "chart_file": chart_filename
    }

def main():
    results = []
    for item in WATCH_LIST:
        data = analyze_stock(item["code"], item["name"])
        if data:
            results.append(data)

    output = {
        "updated_at": datetime.datetime.now().strftime("%m/%d %H:%M JST"),
        "stocks": results
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
