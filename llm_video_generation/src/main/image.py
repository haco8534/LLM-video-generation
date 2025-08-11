"""
image_set.py
~~~~~~~~~~~~
構造化台本からセグメント単位で
Pixabay に適した画像を取得するユーティリティ

- 1) Pixabay へ疎通確認（ping）
- 2) OK なら GPT でキーワード生成（50件ごとに分割）
-   2a) 失敗バッチは単発生成でフォールバック
- 3) Pixabay で画像 URL を取得
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import List, Sequence, Iterable, Any

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


_SANITIZE = re.compile(r'[^a-z ]+')

def sanitize_kw(kw: str, limit: int = 80) -> str:
    kw = kw.lower()
    kw = _SANITIZE.sub(' ', kw)       # 記号・数字を除去
    kw = ' '.join(kw.split())         # 連続空白を1つに
    return urllib.parse.quote_plus(kw[:limit])


# ------------------------------------------------------------------------------
# LLM でキーワード生成
# ------------------------------------------------------------------------------

class KeywordGenerator:
    """OpenAI を用いて英語キーワードを生成（堅牢化版）"""

    _SYSTEM_PROMPT = '''
    # 役割
    あなたは「Pixabay 用キーワード生成 AI」です。

    # 目的
    与えられた日本語テキストの配列を読み取り、各要素に対応する **Pixabay の画像検索に適した英語キーワード** を生成してください。

    # 出力形式（厳守）
    - 一次元配列 JSON のみを返す（例: ["keyword one", "another example", "last one"]）
    - 入力配列と**同じ要素数**
    - 各要素は英単語1〜3語（スペース区切り）、**すべて小文字**
    - 記号、句読点、固有名詞は含めない
    - 余計な文字列・コードフェンス・説明文は一切禁止
    '''

    _SYSTEM_PROMPT_ONE = '''
    あなたは「Pixabay 用キーワード生成 AI」です。
    次の日本語テキストを、Pixabay検索に適した**英語キーワード（小文字・1〜3語・記号なし）**で1つだけ返してください。
    返答はプレーンテキストでキーワードのみ。説明・引用・句読点・コードフェンスは禁止。
    '''

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1"):
        self.client = client
        self.model = model

    # ---------- public ----------

    def generate(self, texts: Sequence[str], *, max_retry: int = 2) -> List[str]:
        """
        texts と同数のキーワード配列を返す。失敗時は
        - JSON抽出のリカバリ
        - ネスト配列の平坦化
        - 長さ調整（>n は切り詰め、<n は単発生成で補完）
        を行う。
        """
        n = len(texts)
        user_prompt = (
            f"要素数 {n} のリストです。必ず **同じ数** のJSON配列で返してください。\n"
            f"{json.dumps(list(texts), ensure_ascii=False)}"
        )

        for _ in range(max_retry):
            reply = self._chat(user_prompt)
            arr = self._extract_array(reply)

            if arr is None:
                continue

            arr = self._flatten_once(arr)  # [[...], "..."] → ["...", "...", ...]
            if len(arr) == n and all(isinstance(x, str) for x in arr):
                return [self._post_sanitize(x) for x in arr]

            if len(arr) > n:
                return [self._post_sanitize(x) for x in arr[:n]]

            if 0 < len(arr) < n:
                # 足りないぶんは単発生成で補完
                missing = [self.generate_one(t) for t in texts[len(arr):]]
                merged = [self._post_sanitize(x) for x in (arr + missing)]
                if len(merged) == n:
                    return merged

            # リトライ用メッセージ
            user_prompt = (
                f"⚠️ 要素数が {len(arr) if arr is not None else '不明'} でした。"
                f"必ず {n} 要素の **一次元JSON配列**で返してください。\n"
                f"リストは同じです: {json.dumps(list(texts), ensure_ascii=False)}"
            )

        # ここまでで合わなければ、全件を単発生成で確定させる
        return [self.generate_one(t) for t in texts]

    def generate_one(self, text: str) -> str:
        """単発生成（最後の砦）。必ず1語〜3語の小文字英語に整形。"""
        reply = self._chat_one(text)
        # 行・カンマ・セミコロンで最初のトークン候補を取る
        token = re.split(r'[\n,;]| +', reply.strip())[0:3]
        guess = " ".join([t for t in token if t]).strip()
        return self._post_sanitize(guess or "concept")

    # ---------- internal ----------

    def _post_sanitize(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^a-z ]+', ' ', s)
        s = ' '.join(s.split())
        # 1〜3語に丸める
        parts = s.split(' ')[:3]
        return ' '.join(parts) if parts else "concept"

    def _extract_array(self, reply: str) -> List[Any] | None:
        """JSON配列を頑健に抽出。コードフェンス除去 + 最初の[]をパース。"""
        if not reply:
            return None
        txt = reply.strip()
        # コードフェンス除去
        txt = re.sub(r"^```.*?\n", "", txt, flags=re.S).strip()
        txt = re.sub(r"\n```$", "", txt).strip()

        # 1) そのままJSON
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "keywords" in obj and isinstance(obj["keywords"], list):
                return obj["keywords"]
        except Exception:
            pass

        # 2) 最初の [ ... ] を抜き出して再トライ
        m = re.search(r"\[[\s\S]*\]", txt)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return obj
            except Exception:
                return None
        return None

    def _flatten_once(self, arr: List[Any]) -> List[str]:
        out: List[str] = []
        for x in arr:
            if isinstance(x, list):
                out.extend([str(y) for y in x])
            else:
                out.append(str(x))
        return out

    def _chat(self, user_content: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,   # 安定性重視
            top_p=0.9,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()

    def _chat_one(self, text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            top_p=0.9,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT_ONE},
                {"role": "user", "content": text},
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
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            print("[DEBUG] ping exception:", e)
            return False

    def search_first_url(self, idx: int, query: str) -> str | None:
        """query で最初にヒットした画像 URL (webformatURL) を返す"""
        query = sanitize_kw(query)
        if not query:
            return None
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

CHUNK_SIZE = 50  # 50件ごとに分割

def _chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """単純な固定長分割ジェネレータ（最後は size 未満になりうる）"""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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

        # 2) GPT でキーワード生成（50件ごとに分割 → 結合）
        prompts = extract_segment_prompts(scenario)

        keywords: List[str] = []
        for batch in _chunked(prompts, CHUNK_SIZE):
            part = self.keyword_gen.generate(batch)
            keywords.extend(part)

        # 念のため総数を検証（理論上ここは常に一致する）
        if len(keywords) != len(prompts):
            # 万一ズレたら、最終フォールバック：全件単発生成で揃える
            keywords = [self.keyword_gen.generate_one(t) for t in prompts]

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
