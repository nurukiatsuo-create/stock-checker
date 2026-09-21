import yfinance as yf
import pandas as pd
import numpy as np

UNIVERSE = {
    "8306.T": {"name": "三菱UFJ", "target": 300},
    "7182.T": {"name": "ゆうちょ銀行", "target": 500},
    "8393.T": {"name": "宮崎銀行", "target": 200},
    "8058.T": {"name": "三菱商事", "target": 300},
    "6981.T": {"name": "村田製作所", "target": 300},
    "4502.T": {"name": "武田薬品", "target": 200}
}

# 木曜指値約定後は空 {} にしてください
CURRENT_HOLDINGS = {}

print("=================================================================")
print("【相場流・エリート6銘柄 最新シグナルボード (Colab版)】")
print("=================================================================")
print(f"{'銘柄名':<10} {'終値':>8} {'5MA':>8} {'20MA':>8} {'スコア':>6} {'シグナル':>8} {'推奨アクション'}")
print("-" * 65)

for sym, info in UNIVERSE.items():
    df = yf.Ticker(sym).history(period="2mo")
    if len(df) < 25:
        continue
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_open, c_close = curr['Open'], curr['Close']
    ma5, ma20 = curr['MA5'], curr['MA20']
    p_ma5, p_ma20 = prev['MA5'], prev['MA20']
    
    is_yang = c_close >= c_open
    is_yin = c_close < c_open
    mid = (c_open + c_close) / 2.0
    
    is_shita = is_yang and (mid > ma5) and (c_close > ma5) and (prev['Close'] <= p_ma5) and (ma5 >= p_ma5)
    is_mono = (curr['Low'] <= ma20 * 1.015) and (c_close > ma20) and is_yang and (ma20 >= p_ma20)
    is_exit = is_yin and (c_close < ma5)
    
    sig = "待機"
    score = 0
    action = "様子見"
    
    if is_exit and sym in CURRENT_HOLDINGS:
        sig = "手仕舞"
        action = "【全株決済】翌朝寄り付きで成行売り"
    elif is_shita and (ma20 >= p_ma20) and (c_close >= ma20):
        sig = "現買"
        score = 80 if is_mono else 50
        action = f"【新規打診】100株エントリー (本命目標: {info['target']}株)"
    
    print(f"{info['name']:<10} {c_close:>8.0f}円 {ma5:>8.1f} {ma20:>8.1f} {score:>5}点 {sig:>8}   {action}")
print("=================================================================")
