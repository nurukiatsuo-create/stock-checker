import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. 運用設定 (新8銘柄体制・スコア60点基準)
# ==========================================
PORTFOLIO_CONFIG = {
    # 既存エリート6銘柄
    "8306.T": {"name": "三菱UFJ", "target": 300},
    "7182.T": {"name": "ゆうちょ銀行", "target": 500},
    "8393.T": {"name": "宮崎銀行", "target": 200},
    "8058.T": {"name": "三菱商事", "target": 300},
    "6981.T": {"name": "村田製作所", "target": 300},
    "4502.T": {"name": "武田薬品", "target": 200},
    # 新規追加2銘柄
    "2768.T": {"name": "双日", "target": 100},
    "6324.T": {"name": "ハーモニック", "target": 100},
}

MIN_SCORE = 60  # 実証済み最適化スコア基準

# 現在の保有ポジション（保有がない待機時は空 {}）
# 例: {"8306.T": {"shares": 100, "entry_price": 1850, "days": 3}}
CURRENT_POSITIONS = {}


# ==========================================
# 2. 株価取得 & 相場流判定ロジック
# ==========================================
def fetch_and_evaluate(sym, meta):
  try:
    d = yf.download(
        sym, period="3mo", interval="1d", progress=False, auto_adjust=True
    )
    if d.empty or len(d) < 25:
      return None

    if isinstance(d.columns, pd.MultiIndex):
      d.columns = d.columns.get_level_values(0)

    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()

    curr, prev = d.iloc[-1], d.iloc[-2]
    co, cc = float(curr["Open"]), float(curr["Close"])
    ch, cl = float(curr["High"]), float(curr["Low"])
    pc = float(prev["Close"])
    m5, m20 = float(curr["MA5"]), float(curr["MA20"])
    pm5, pm20 = float(prev["MA5"]), float(prev["MA20"])

    is_yang = cc >= co
    is_yin = cc < co
    mid = (co + cc) / 2.0
    m20_slope = ((m20 - pm20) / pm20) * 100 if pm20 > 0 else 0.0
    bias_20 = ((cc - m20) / m20) * 100 if m20 > 0 else 0.0

    # 相場流「下半身」&「ものわかれ」
    is_shita = (
        is_yang
        and (mid > m5)
        and (cc > m5)
        and (pc <= pm5)
        and (m5 >= pm5)
    )
    is_mono = (
        (cl <= m20 * 1.015) and (cc > m20) and is_yang and (m20 >= pm20)
    )

    action = "待機"
    score = 0
    reason = "--"

    # A. 保有銘柄の手仕舞い判定
    if sym in CURRENT_POSITIONS:
      pos = CURRENT_POSITIONS[sym]
      if (cc < m5) and is_yin:
        action = "【手仕舞】"
        reason = "5MA割れ陰線（全決済）"
      elif pos.get("days", 0) >= 9:
        action = "【手仕舞】"
        reason = "日柄10本到達（明日寄付き決済）"
      elif is_yang and (cc > m5) and (pos.get("shares", 100) < meta["target"]):
        action = "【増し玉】"
        reason = f"続伸確認（+{meta['target'] - 100}株 追加）"
      else:
        action = "キープ"
        reason = f"保有継続中 ({pos.get('shares', 100)}株)"
      return {
          "code": sym.replace(".T", ""),
          "name": meta["name"],
          "price": cc,
          "score": "--",
          "action": action,
          "reason": reason,
          "target": meta["target"],
      }

    # B. 新規エントリー判定
    if is_shita and (m20 >= pm20) and (cc >= m20):
      score = 80 if is_mono else 50
      score += min(max(int(m20_slope * 20), 0), 20)
      hl = ch - cl
      if hl > 0:
        score += int((abs(cc - co) / hl) * 10)
      if abs(bias_20) > 8.0:
        score -= 15

      if score >= MIN_SCORE:
        action = "★【買エントリー】"
        reason = (
            f"下半身+ものわかれ（打診100株）"
            if is_mono
            else f"下半身成立（打診100株）"
        )

    return {
        "code": sym.replace(".T", ""),
        "name": meta["name"],
        "price": cc,
        "score": f"{score}点" if score > 0 else "--",
        "action": action,
        "reason": reason,
        "target": meta["target"],
    }
  except Exception as e:
    return None


# ==========================================
# 3. 監視実行 & レポート表示
# ==========================================
print("=====================================================================")
print(
    f"【相場流】新8銘柄 日次シグナル監視モニター ({time.strftime('%Y-%m-%d %H:%M')})"
)
print(" 判定基準: スコア60点以上 | 同時保有最大2枠 | 打診100株")
print("=====================================================================")
print(
    f"{'コード':<6} {'銘柄名':<11} {'終値':>8} {'スコア':>7} {'指示アクション':<16} {'判定理由'}"
)
print("-" * 69)

entries = []
for sym, meta in PORTFOLIO_CONFIG.items():
  res = fetch_and_evaluate(sym, meta)
  if res:
    print(
        f"{res['code']:<6} {res['name']:<11} {int(res['price']):>7,d}円"
        f" {res['score']:>7} {res['action']:<14} {res['reason']}"
    )
    if "買エントリー" in res["action"]:
      entries.append(res)
  time.sleep(0.2)

print("=" * 69)

# 明朝のアクション指示
if entries:
  print("\n【明朝 9:00 の発注指示】")
  entries.sort(key=lambda x: int(x["score"].replace("点", "")), reverse=True)
  for e in entries:
    print(
        f" ▶ [{e['code']} {e['name']}] スコア{e['score']} $\\rightarrow$"
        " 翌朝寄付き『成行買い 100株』を発注"
    )
else:
  print("\n【明朝の発注指示】")
  print(" ▶ 新規エントリーシグナル（60点以上）はありません。待機を継続します。")
