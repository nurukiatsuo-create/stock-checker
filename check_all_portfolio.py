import json
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# 1. 運用パラメータ設定（150万円・エリート6銘柄）
# ==========================================
MAX_HOLDINGS = 2  # 同時保有上限枠（最大2銘柄）
INITIAL_SHARES = 100  # 初動打診エントリー株数
MIN_SCORE = 50  # エントリー最低スコア
PYRAMID_SCORE = 70  # 増し玉対象スコア

# 監視対象ユニバース（目標ロット設計済み）
WATCH_UNIVERSE = {
    "8306.T": {
        "name": "三菱UFJ",
        "target_shares": 300,
        "sector": "メガバンク",
    },
    "7182.T": {
        "name": "ゆうちょ銀行",
        "target_shares": 500,
        "sector": "国策金融",
    },
    "8393.T": {
        "name": "宮崎銀行",
        "target_shares": 200,
        "sector": "地方銀行",
    },
    "8058.T": {
        "name": "三菱商事",
        "target_shares": 300,
        "sector": "総合商社",
    },
    "6981.T": {
        "name": "村田製作所",
        "target_shares": 300,
        "sector": "電子部品",
    },
    "4502.T": {
        "name": "武田薬品",
        "target_shares": 200,
        "sector": "医薬品",
    },
}

# ==========================================
# 2. 現在の保有状況管理（木曜決済後は空 {} で待機）
# ==========================================
# 保有が発生した場合は以下のように更新:
# "8306.T": {"side": "BUY", "shares": 100, "entry_price": 1850.0, "pyramided": False, "score": 80, "hold_days": 1}
HOLDINGS = {}


# ==========================================
# 3. 相場流テクニカル判定ロジック
# ==========================================
def evaluate_stock(df):
  if len(df) < 25:
    return None, 0, {}

  curr = df.iloc[-1]
  prev = df.iloc[-2]

  c_open, c_close = curr["Open"], curr["Close"]
  c_high, c_low = curr["High"], curr["Low"]
  p_open, p_close = prev["Open"], prev["Close"]
  ma5, ma20 = curr["MA5"], curr["MA20"]
  p_ma5, p_ma20 = prev["MA5"], prev["MA20"]

  if np.isnan(ma5) or np.isnan(ma20) or np.isnan(p_ma5) or np.isnan(p_ma20):
    return None, 0, {}

  ma5_slope = ((ma5 - p_ma5) / p_ma5) * 100 if p_ma5 > 0 else 0.0
  ma20_slope = ((ma20 - p_ma20) / p_ma20) * 100 if p_ma20 > 0 else 0.0
  bias_20 = ((c_close - ma20) / ma20) * 100 if ma20 > 0 else 0.0
  candle_body_mid = (c_open + c_close) / 2.0
  is_yang = c_close >= c_open
  is_yin = c_close < c_open

  # 下半身（買い）
  is_shitahanshin = (
      is_yang
      and (candle_body_mid > ma5)
      and (c_close > ma5)
      and (p_close <= p_ma5)
      and (ma5 >= p_ma5)
  )

  # 逆下半身（空売り）
  is_gyaku_shitahanshin = (
      is_yin
      and (candle_body_mid < ma5)
      and (c_close < ma5)
      and (p_close >= p_ma5)
      and (ma5 <= p_ma5)
  )

  # ものわかれ
  is_monowakare = (
      (c_low <= ma20 * 1.015)
      and (c_close > ma20)
      and is_yang
      and (ma20 >= p_ma20)
  )

  signal = None
  score = 0

  if is_shitahanshin and (ma20 >= p_ma20) and (c_close >= ma20):
    signal = "BUY"
    score = 80 if is_monowakare else 50
  elif (
      is_gyaku_shitahanshin
      and (ma20 <= p_ma20)
      and (c_close <= ma20)
      and (bias_20 > -5.0)
  ):
    signal = "SHORT"
    score = 50

  if signal is None:
    return None, 0, {"ma5": ma5, "ma20": ma20, "bias_20": bias_20}

  # 加点・減点
  if signal == "BUY":
    score += min(max(int(ma20_slope * 20), 0), 20)
  elif signal == "SHORT":
    score += min(max(int(abs(ma20_slope) * 20), 0), 20)

  hl_range = c_high - c_low
  if hl_range > 0:
    score += int((abs(c_close - c_open) / hl_range) * 10)

  if abs(bias_20) > 8.0:
    score -= 15

  return signal, score, {"ma5": ma5, "ma20": ma20, "bias_20": bias_20}


# ==========================================
# 4. メイン監視実行ループ
# ==========================================
def main():
  print("=" * 65)
  print("【相場流・エリート6銘柄 本番運用監視シグナル】")
  print(
      f" 運用資金枠: 150万円 | 同時保有上限: {MAX_HOLDINGS}枠 | 現在保有:"
      f" {len(HOLDINGS)}枠"
  )
  print("=" * 65)

  data_map = {}
  for sym in WATCH_UNIVERSE:
    try:
      t = yf.Ticker(sym)
      df = t.history(period="3mo")
      if df.empty or len(df) < 25:
        continue
      if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
      df["MA5"] = df["Close"].rolling(5).mean()
      df["MA20"] = df["Close"].rolling(20).mean()
      data_map[sym] = df
    except Exception as e:
      print(f"データ取得エラー ({sym}): {e}")

  # --- A. 保有銘柄の決済・増し玉判定 ---
  if HOLDINGS:
    print("\n▼ 【保有銘柄 ステータス＆アクション】")
    print("-" * 65)
    for sym, pos in HOLDINGS.items():
      if sym not in data_map:
        continue
      df = data_map[sym]
      curr = df.iloc[-1]
      info = WATCH_UNIVERSE[sym]
      c_close, c_open, ma5 = curr["Close"], curr["Open"], curr["MA5"]
      hold_days = pos.get("hold_days", 1)

      exit_signal = None
      if pos["side"] == "BUY":
        if (c_close < ma5) and (c_close < c_open):
          exit_signal = "5MA割れ陰線（手仕舞い推奨）"
        elif hold_days >= 10:
          exit_signal = "日柄10本到達（手仕舞い推奨）"
      elif pos["side"] == "SHORT":
        if (c_close > ma5) and (c_close >= c_open):
          exit_signal = "5MA超え陽線（手仕舞い推奨）"
        elif hold_days >= 10:
          exit_signal = "日柄10本到達（手仕舞い推奨）"

      if exit_signal:
        print(f"■ 【手仕舞い】{info['name']} ({sym})")
        print(f"   理由: {exit_signal} | 翌朝寄り付きで全株成行決済")
        continue

      # 増し玉チェック
      if not pos.get("pyramided", False):
        can_pyramid = False
        if (
            pos["side"] == "BUY"
            and (c_close >= c_open)
            and (c_close > ma5)
            and pos.get("score", 0) >= PYRAMID_SCORE
        ):
          can_pyramid = True
        elif (
            pos["side"] == "SHORT"
            and (c_close <= c_open)
            and (c_close < ma5)
            and pos.get("score", 0) >= PYRAMID_SCORE
        ):
          can_pyramid = True

        if can_pyramid:
          add_qty = info["target_shares"] - pos["shares"]
          print(f"★ 【増し玉推奨】{info['name']} ({sym})")
          print(f"   状態: 5日線上を陽線キープ（波に乗った本命展開）")
          print(
              f"   発注: 翌朝寄り付きで +{add_qty}株 追加（計"
              f" {info['target_shares']}株へ増量）"
          )
        else:
          print(
              f"● 【キープ】{info['name']} ({sym}): 打診{pos['shares']}株のまま継続"
          )
      else:
        print(
            f"● 【フル保有中】{info['name']} ({sym}):"
            f" {pos['shares']}株（利大伸ばし中）"
        )

  # --- B. 新規エントリー候補選抜 ---
  available_slots = MAX_HOLDINGS - len(HOLDINGS)
  print(f"\n▼ 【監視銘柄シグナルスキャン】(空き枠: {available_slots}枠)")
  print("-" * 65)

  candidates = []
  for sym, info in WATCH_UNIVERSE.items():
    if sym in HOLDINGS or sym not in data_map:
      continue
    df = data_map[sym]
    sig, score, meta = evaluate_stock(df)
    last_p = df.iloc[-1]["Close"]

    status_str = f"スコア: {score:>2}点 | 終値: {last_p:>6.0f}円"
    if sig:
      status_str += f" | シグナル: {sig}"
      if score >= MIN_SCORE:
        candidates.append((sym, sig, score, last_p))
    else:
      status_str += " | シグナル: なし"
    print(f" {info['name']:<8} ({sym}): {status_str}")

  # スコア順にソートして選抜発注の指示
  candidates.sort(key=lambda x: x[2], reverse=True)

  print("\n" + "=" * 65)
  print("【明日の実戦発注アクション】")
  print("=" * 65)
  if available_slots <= 0:
    print("現在、保有上限枠（2銘柄）に達しているため新規発注は見送ります。")
  elif not candidates:
    print("現在、50点以上のエントリーシグナルは点灯していません（待機）。")
  else:
    selected = candidates[:available_slots]
    for sym, sig, score, last_p in selected:
      info = WATCH_UNIVERSE[sym]
      action_type = "現物買い/信用買い" if sig == "BUY" else "信用新規売り"
      print(f"★ 【新規打診エントリー】: {info['name']} ({sym})")
      print(f"   スコア: {score}点 ({info['sector']})")
      print(f"   発注: 翌朝寄り付き成行で 『{INITIAL_SHARES}株』 {action_type}")
      print(
          f"   増し玉目安: 明日も5MAキープなら +{info['target_shares'] - 100}株"
          f" 追加予定（目標計{info['target_shares']}株）"
      )
      print("-" * 65)


if __name__ == "__main__":
  main()
