# pixabay_diag.py
# ---------------------------------------------
# ・環境変数 PIXABAY_API_KEY から API キーを読む
# ・ステータス・ヘッダー・本文を全部出力
# ---------------------------------------------
import os
import textwrap
import requests
from dotenv import load_dotenv

def diag():
    load_dotenv()
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        print("❌ PIXABAY_API_KEY が未設定")
        return

    endpoint = "https://pixabay.com/api/"
    params = {
        "key": key.strip(),   # 前後の空白・改行を除去
        "q": "nature",
        "per_page": 3,
        "safesearch": "true",
    }

    print("▶ リクエスト先:", endpoint)
    print("▶ クエリパラメータ:", params)

    try:
        resp = requests.get(endpoint, params=params, timeout=5)
    except requests.RequestException as e:
        print("❌ ネットワーク例外:", e)
        return

    print("\n=== HTTP ステータス ===")
    print(resp.status_code, resp.reason)

    print("\n=== レスポンスヘッダー ===")
    for k, v in resp.headers.items():
        print(f"{k}: {v}")

    print("\n=== レスポンス本文 (先頭 500 文字) ===")
    body = resp.text.strip()
    print(textwrap.shorten(body, width=500, placeholder=" …"))

if __name__ == "__main__":
    diag()
