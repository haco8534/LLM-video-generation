"""
image_set.py
~~~~~~~~~~~~
構造化台本からセグメント単位で
Pixabay に適した画像を取得するためのユーティリティ群
"""

import json
import os
from typing import List, Sequence

from dotenv import load_dotenv
from openai import OpenAI
import requests

#------------------------------------------------------------------------------
# シナリオ関連
#------------------------------------------------------------------------------


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


#------------------------------------------------------------------------------
# LLM でキーワード生成
#------------------------------------------------------------------------------


class KeywordGenerator:
    """OpenAI を用いて英語キーワードを生成"""

    _SYSTEM_PROMPT = f'''
            # 役割
            あなたは「Pixabay 用キーワード生成 AI」です。

            # 目的
            入力される配列の各要素ごとの日本語テキストを読み取り、
            Pixabay での画像検索に適した **英語キーワード** を、
            要素順に一次元配列で返してください。

            # 出力形式
            - 配列は JSON の一次元配列（例: ["first keywords", "second keywords", ...]）とする。
            - 要素数は入力セグメントと同じだけ。
            - 各要素はスペース区切り 1〜3 語の英語キーワード。
            - すべて **小文字**、記号・句読点なし。

            # キーワード生成ルール
            1. **Pixabay でヒットしやすい** 抽象的・汎用的な単語を選ぶ  
            - 例: “computer”, “coding”, “memory”, “education” など  
            2. 固有名詞・登録商標・人物名・サービス名など  
            **検索結果が極端に絞られる語は避ける**  
            - “Python”, “Javascript”, “Pixabay” などは使わない  
            3. 台本の文脈を要約し、視覚的に表現しやすい概念へ変換する  
            - 「ビット演算とは？」→ “bitwise operation” ではなく  
                “binary code” / “digital logic” など  
            4. 抽象語と具体語を 1〜3 語組み合わせ、  
            **写真・イラストどちらにも合う** 検索語句にする  
            5. 同義語が複数ある場合は、より一般的で  
            画像数が多い語を選択（例: “computer” > “workstation”）  
            6. 同じセグメント内で複数トピックがあるときは  
            もっとも主要なイメージを優先  
            7. 入力に HTML や注釈が混在していても無視する

            # 手順
            - あなたに渡されるのは  
            Python リスト形式の文字列例:  
            ["テキスト1", "テキスト2", …]  
            - ルールに従って各テキストから  
            1〜3 語の英語キーワードを抽出・要約し、  
            セグメント順に並べて JSON 配列として出力。

            # 例 (参考・出力には含めない)
            入力: ["私はコーヒーが好きです", "空を飛びたい"]  
            出力: ["coffee cup", "blue sky freedom"]

            # 準備ができたら、入力リストを受け取りしだい  
            ただちにキーワード配列のみを JSON 形式で返してください。
        '''

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1-mini"):
        self.client = client
        self.model = model

    def generate(self, texts: Sequence[str], *, max_retry: int = 3) -> List[str]:
        """texts と同数のキーワード配列を返す。数が合わなければリトライ"""
        n = len(texts)
        user_prompt = (
            f"要素数 {n} のリストです。必ず同じ数で返してください。\n"
            f"{json.dumps(list(texts), ensure_ascii=False)}"
        )

        for attempt in range(1, max_retry + 1):
            reply = self._chat(user_prompt)
            try:
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
            top_p=0.95,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()


#------------------------------------------------------------------------------
# Pixabay API 呼び出し
#------------------------------------------------------------------------------


class PixabayFetcher:
    """Pixabay から画像 URL を取得"""

    _ENDPOINT = "https://pixabay.com/api/"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_first_url(self,idx: int, query: str) -> str | None:
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


#------------------------------------------------------------------------------
# Facade
#------------------------------------------------------------------------------


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
        prompts = extract_segment_prompts(scenario)
        keywords = self.keyword_gen.generate(prompts)
        return [self.pixabay.search_first_url(idx,k) for idx, k in enumerate(keywords,1)]


if __name__ == "__main__":
    scenario = json.load(open("./modules/a.txt", encoding="utf-8"))
    service = ImageSetService()
    urls = service.scenario_to_images(scenario)

    print(urls)