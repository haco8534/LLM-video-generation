"""
image_set.py
~~~~~~~~~~~~
構造化台本からセグメント単位で
Pixabay に適した画像を取得するユーティリティ

- 1) Pixabay へ疎通確認（ping）
- 2) OK なら GPT でキーワード生成
- 3) Pixabay で画像 URL を取得
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence

from dotenv import load_dotenv
from openai import OpenAI
import requests

# ------------------------------------------------------------------------------
# シナリオ関連
# ------------------------------------------------------------------------------


def extract_segment_prompts(scenario: dict) -> List[str]:
    """
    dialogue なら script.text, topic なら title を
    id 昇順 (≒時系列) に返す純粋関数
    """
    prompts: List[str] = []
    for seg in sorted(scenario["segments"], key=lambda s: s["id"]):
        if seg["type"] == "dialogue":
            prompts.append(seg["script"]["text"])
        elif seg["type"] == "topic":
            prompts.append(seg["title"])
    return prompts


# ------------------------------------------------------------------------------
# LLM でキーワード生成
# ------------------------------------------------------------------------------


class KeywordGenerator:
    """OpenAI を用いて英語キーワードを生成"""

    _SYSTEM_PROMPT = '''
    # 役割
    あなたは「Pixabay 用キーワード生成 AI」です。

    # 目的
    与えられた日本語テキストの配列を読み取り、各要素に対応する **Pixabay の画像検索に適した英語キーワード** を生成してください。

    # 出力形式
    - 一次元配列で出力（例: ["keyword one", "another example", "last one"]）
    - 前後に余計な文字列・コードブロック記号は禁止。
    - 入力配列と同じ要素数
    - 各要素は英単語1〜3語（スペース区切り）、**すべて小文字**
    - 記号、句読点、固有名詞は含めない

    # キーワード生成ルール
    1. **Pixabayでヒットしやすい**、抽象的・汎用的な語を選ぶ  
    2. 固有名詞・登録商標・人物名・サービス名などは避ける  
    3. 文の意味を要約し、**視覚的に連想しやすい概念**に変換する  
    4. 抽象語と具体語を組み合わせ、**写真・イラストの両方にマッチするキーワード**を作る  
    5. 同義語がある場合は、**Pixabayで一般的な語**を選ぶ  
    6. セグメント内に複数の話題がある場合、**最も代表的なイメージ**を優先する
    '''

    def __init__(self, client: OpenAI, *, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, texts: Sequence[str], *, max_retry: int = 3) -> List[str]:
        """texts と同数のキーワード配列を返す。数が合わなければリトライ"""
        n = len(texts)
        user_prompt = (
            f"要素数 {n} のリストです。必ず同じ数で返してください。\n"
            f"{json.dumps(list(texts), ensure_ascii=False)}"
        )

        for _ in range(max_retry):
            reply = self._chat(user_prompt)
            try:
                reply = self._chat(user_prompt)
                # print("[DEBUG] GPT raw reply:\n", reply[:300])
                keywords = json.loads(reply)
            except json.JSONDecodeError:
                continue

            if isinstance(keywords, list) and len(keywords) == n:
                return keywords

            user_prompt = (
                "⚠️ 要素数が違いました。もう一度 "
                f"{n} 要素で返してください。\n"
                f"リストは同じです: {json.dumps(list(texts), ensure_ascii=False)}"
            )

        raise RuntimeError("Failed to obtain keyword list with correct length.")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _chat(self, user_content: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            top_p=0.9,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------------------
# Pixabay API 呼び出し
# ------------------------------------------------------------------------------


class PixabayFetcher:
    """Pixabay から画像 URL を取得"""

    _ENDPOINT = "https://pixabay.com/api/"

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ---------- 追加 ----------
    def ping(self, timeout: int = 5) -> bool:
        """
        Pixabay API への疎通確認。
        200 OK が返れば True。失敗は False。
        """
        try:
            params = {
                "key": self.api_key,
                "q": "test",
                "per_page": 3,
                "safesearch": "true",
            }
            r = requests.get(self._ENDPOINT, params=params, timeout=timeout)
            print("[DEBUG] ping status:", r.status_code, r.text[:120])
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            print("[DEBUG] ping exception:", e)
            return False
    # --------------------------

    def search_first_url(self, idx: int, query: str) -> str | None:
        """query で最初にヒットした画像 URL (webformatURL) を返す"""
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": 3,
            "safesearch": "true",
        }
        r = requests.get(self._ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        return hits[0]["webformatURL"] if hits else None


# ------------------------------------------------------------------------------
# Facade
# ------------------------------------------------------------------------------


class ImageSetService:
    """
    台本 → 画像 URL 一覧 をワンショットで行う
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        pixabay_api_key: str | None = None,
    ):
        load_dotenv()
        self.openai_client = OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.keyword_gen = KeywordGenerator(self.openai_client)
        self.pixabay = PixabayFetcher(pixabay_api_key or os.getenv("PIXABAY_API_KEY"))

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def scenario_to_images(self, scenario: dict) -> List[str | None]:
        # 1) Pixabay に接続できるかテスト
        if not self.pixabay.ping():
            raise ConnectionError(
                "Pixabay API に接続できません。APIキーまたはネットワークを確認してください。"
            )

        # 2) GPT でキーワード生成（ここで初めてトークン消費）
        prompts = extract_segment_prompts(scenario)
        keywords = self.keyword_gen.generate(prompts)

        # 3) Pixabay で画像 URL を取得
        return [self.pixabay.search_first_url(idx, k) for idx, k in enumerate(keywords, 1)]


# ------------------------------------------------------------------------------
# CLI / サンプル実行
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # サンプルシナリオを読み込む
    scenario = json.load(open("./llm_video_generation/src/s.txt", encoding="utf-8"))

    service = ImageSetService()

    try:
        urls = service.scenario_to_images(scenario)
    except ConnectionError as e:
        print(f"[Pixabay エラー] {e}")
        exit(1)

    # 取得した URL を pickle 保存（デバッグ用）
    import pickle
    with open("./llm_video_generation/src/i.pkl", "wb") as f:
        pickle.dump(urls, f)

    print("画像 URL:", urls)
