import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. 検証設定（150万円予算・厳選4銘柄）
# ==========================================
INITIAL_CAPITAL = 1500000  # 予算150万円

UNIVERSE = {
    "8306.T": "三菱UFJ",
    "8002.T": "丸紅",
    "9107.T": "川崎汽船",
    "7012.T": "川崎重工",
}

MIN_SCORE = 50


# スコア連動ロット設定（株価帯を考慮し、建玉70万〜100万円規模に調整）
def get_trade_shares(symbol, score):
  if score >= 70:
    if "7012" in symbol:  # 川崎重工（株価2,400円前後）
      return 300
    else:  # 三菱UFJ, 川崎汽船, 丸紅（株価3,500〜5,000円前後）
      return 200
  else:
    return 100  # 通常シグナル（50〜69点）は一律100株打診


# ==========================================
# 2. 相場流判定ロジック（20MA順張り＋安全空売り）
# ==========================================
def evaluate_bar(df_slice):
  if len(df_slice) < 25:
    return None, 0
  curr, prev = df_slice.iloc[-1], df_slice.iloc[-2]

  c_open, c_close = curr["Open"], curr["Close"]
  c_high, c_low = curr["High"], curr["Low"]
  p_open, p_close = prev["Open"], prev["Close"]
  ma5, ma20 = curr["MA5"], curr["MA20"]
  p_ma5, p_ma20 = prev["MA5"], prev["MA20"]

  if np.isnan(ma5) or np.isnan(ma20) or np.isnan(p_ma5) or np.isnan(p_ma20):
    return None, 0

  ma5_slope = ((ma5 - p_ma5) / p_ma5) * 100 if p_ma5 > 0 else 0.0
  ma20_slope = ((ma20 - p_ma20) / p_ma20) * 100 if p_ma20 > 0 else 0.0
  bias_20 = ((c_close - ma20) / ma20) * 100 if ma20 > 0 else 0.0
  candle_body_mid = (c_open + c_close) / 2.0
  is_yang = c_close >= c_open
  is_yin = c_close < c_open

  is_shitahanshin = (
      is_yang
      and (candle_body_mid > ma5)
      and (c_close > ma5)
      and (p_close <= p_ma5)
      and (ma5 >= p_ma5)
  )

  is_gyaku_shitahanshin = (
      is_yin
      and (candle_body_mid < ma5)
      and (c_close < ma5)
      and (p_close >= p_ma5)
      and (ma5 <= p_ma5)
  )

  is_monowakare = (
      (c_low <= ma20 * 1.015)
      and (c_close > ma20)
      and is_yang
      and (ma20 >= p_ma20)
  )

  signal = None
  score = 0

  # 買い：20MA上向き＋株価20MA以上
  if is_shitahanshin and (ma20 >= p_ma20) and (c_close >= ma20):
    signal = "BUY"
    score = 80 if is_monowakare else 50

  # 売り：20MA下向き＋株価20MA以下＋乖離安全圏
  elif (
      is_gyaku_shitahanshin
      and (ma20 <= p_ma20)
      and (c_close <= ma20)
      and (bias_20 > -5.0)
  ):
    signal = "SHORT"
    score = 50

  if signal is None:
    return None, 0

  # 加点
  if signal == "BUY":
    score += min(max(int(ma20_slope * 20), 0), 20)
  elif signal == "SHORT":
    score += min(max(int(abs(ma20_slope) * 20), 0), 20)

  hl_range = c_high - c_low
  if hl_range > 0:
    score += int((abs(c_close - c_open) / hl_range) * 10)

  if abs(bias_20) > 8.0:
    score -= 15

  return signal, score


# ==========================================
# 3. バックテスト集計処理
# ==========================================
all_trades = []

for symbol, name in UNIVERSE.items():
  ticker = yf.Ticker(symbol)
  df = ticker.history(period="1y")
  if df.empty or len(df) < 50:
    continue
  if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

  df["MA5"] = df["Close"].rolling(5).mean()
  df["MA20"] = df["Close"].rolling(20).mean()

  position = None
  for i in range(25, len(df) - 1):
    df_slice = df.iloc[: i + 1]
    curr_bar, next_bar = df.iloc[i], df.iloc[i + 1]

    # 手仕舞い判定
    if position is not None:
      hold_days = i - position["entry_idx"]
      entry_p = position["entry_price"]
      c_close = curr_bar["Close"]
      c_open = curr_bar["Open"]
      shares = position["shares"]

      exit_reason = None
      if position["side"] == "BUY":
        if (c_close < curr_bar["MA5"]) and (c_close < c_open):
          exit_reason = "5MA割れ陰線"
        elif hold_days >= 10:
          exit_reason = "日柄手仕舞(10日)"

      elif position["side"] == "SHORT":
        if (c_close > curr_bar["MA5"]) and (c_close >= c_open):
          exit_reason = "5MA超え陽線"
        elif hold_days >= 10:
          exit_reason = "日柄手仕舞(10日)"

      if exit_reason:
        exit_p = next_bar["Open"]
        pnl = (
            (exit_p - entry_p) * shares
            if position["side"] == "BUY"
            else (entry_p - exit_p) * shares
        )
        all_trades.append({
            "name": name,
            "side": position["side"],
            "score": position["score"],
            "shares": shares,
            "exit_date": next_bar.name,
            "pnl": int(pnl),
            "reason": exit_reason,
        })
        position = None
        continue

    # 新規エントリー
    if position is None:
      sig, score = evaluate_bar(df_slice)
      if sig in ["BUY", "SHORT"] and score >= MIN_SCORE:
        shares = get_trade_shares(symbol, score)
        position = {
            "side": sig,
            "shares": shares,
            "score": score,
            "entry_price": next_bar["Open"],
            "entry_idx": i + 1,
            "entry_date": next_bar.name,
        }

# ==========================================
# 4. 結果レポート出力
# ==========================================
if all_trades:
  trade_df = pd.DataFrame(all_trades).sort_values("exit_date")
  trade_df["cum_pnl"] = trade_df["pnl"].cumsum()

  wins = trade_df[trade_df["pnl"] > 0]
  losses = trade_df[trade_df["pnl"] <= 0]
  win_rate = (len(wins) / len(trade_df)) * 100
  total_pnl = trade_df["pnl"].sum()
  annual_yield = (total_pnl / INITIAL_CAPITAL) * 100
  pf = (
      wins["pnl"].sum() / abs(losses["pnl"].sum())
      if len(losses) > 0
      else float("inf")
  )

  buys = trade_df[trade_df["side"] == "BUY"]
  shorts = trade_df[trade_df["side"] == "SHORT"]

  high_score_trades = trade_df[trade_df["score"] >= 70]

  print("==================================================")
  print(f"【予算150万円・ロット可変モデル実績】")
  print(f" 取引数: {len(trade_df)}回 | 勝率: {win_rate:.1f}% | PF: {pf:.2f}")
  print(f" 年間総損益: {total_pnl:+d} 円")
  print(f" 年利換算:   {annual_yield:+.1f} % (予算150万円基準)")
  print("--------------------------------------------------")
  print(
      f" └ 高得点勝負玉 (70点以上/200-300株): {len(high_score_trades)}回中"
      f" {len(high_score_trades[high_score_trades['pnl']>0])}勝 | 損益:"
      f" {high_score_trades['pnl'].sum():+d}円"
  )
  if len(buys) > 0:
    print(
        f" └ 買い損益: {buys['pnl'].sum():+d}円 (勝率"
        f" {len(buys[buys['pnl']>0])/len(buys)*100:.1f}%)"
    )
  if len(shorts) > 0:
    print(
        f" └ 売り損益: {shorts['pnl'].sum():+d}円 (勝率"
        f" {len(shorts[shorts['pnl']>0])/len(shorts)*100:.1f}%)"
    )
  print("==================================================\n")

  # 銘柄別ランキング
  print("【銘柄別 パフォーマンス内訳】")
  print(
      f"{'銘柄名':<10} {'取引数':>6} {'勝率':>8} {'総損益(円)':>14} {'PF':>6}"
  )
  print("-" * 50)
  for name, grp in trade_df.groupby("name"):
    g_win = grp[grp["pnl"] > 0]
    g_loss = grp[grp["pnl"] <= 0]
    g_rate = (len(g_win) / len(grp)) * 100
    g_pnl = grp["pnl"].sum()
    g_pf = (
        (g_win["pnl"].sum() / abs(g_loss["pnl"].sum()))
        if len(g_loss) > 0 and g_loss["pnl"].sum() != 0
        else 9.99
    )
    print(
        f"{name:<10} {len(grp):>6}回 {g_rate:>7.1f}% {g_pnl:>+14,d}円 {g_pf:>6.2f}"
    )
  print("==================================================")

  # チャート描画
  plt.figure(figsize=(10, 4.5))
  plt.plot(
      trade_df["exit_date"],
      trade_df["cum_pnl"],
      marker=".",
      color="#0a84ff",
      linewidth=2,
  )
  plt.title("【予算150万・ロット可変レバレッジ】過去1年 累積損益推移 (円)")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.ylabel("累積損益 (円)")
  plt.show()
