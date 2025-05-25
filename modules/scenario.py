"""
1. 指定テーマ＆長さで日本語対話シナリオを生成
2. 生成された台本を JSON 構造化スキーマへ変換

外部依存
--------
- openai>=1.0.0
- python-dotenv>=1.0.0   # .env から API キー取得用
"""

from __future__ import annotations

import json
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 低レベル : プロンプトテンプレート
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT_SCENARIO = "\n".join(
    [
        "あなたは受賞歴のある脚本家です。",
        "## 出力規則",
        "1. 台本のみを出力し、余計なメタ情報は書かない。",
        "2. 登場人物:",
        "   - **アオイ** : 好奇心旺盛な学習者（質問役）",
        "   - **カイ**   : 親しみやすい専門家（解説役）",
        "3. 構成:",
        "   a. 【導入】全体の 5-10 % : テーマにまつわる雑談から自然に話題へ。",
        "   b. 【本編】以下を必ず守る:",
        "      • 本文は互いに独立したサブトピックを順番に扱う。",
        "      • サブトピックは最低でも{min_subtopics}個必ず作る",
        "      • 各サブトピックで《質問→解説→相づち》を 1 サイクル完結させる。",
        "      • 広すぎる一般論は避け、具体例や最新トレンドを 1 つ盛り込む。",
        "      • アオイは疑問→カイが回答、を明確にする。",
        "      • 生成する文章は会話形式のみで、jsonやソースコードは生成しない",
        "      • 一回のセリフは「60文字を絶対に超えてはならない」。超える場合はもう一度そのキャラターのセリフとして、新しく台詞を書く",
        "   c. 【まとめ】全体の 5 % : カイが要点整理、アオイが学びを言語化。",
        "4. 文体: やさしい日本語＋必要に応じて専門用語。",
        "5. 長さ目標: 約 {word_target} 語（±5 %）。",
        "6. 思考過程は出力に含めない。",
    ]
)

_SYSTEM_PROMPT_STRUCT = r"""
    あなたは動画制作パイプラインの「構造化エンジン」です。  
    入力された台本を、下記 JSON スキーマに従って厳密に変換してください。

    ================== JSON スキーマ ==================
    {
    "segments": [
        {
        "id":    <int, 1からの連番>,
        "type":  "topic",  // サブトピックの見出し
        "title": <string>
        },
        {
        "id":    <int>,
        "type":  "dialogue",  // 会話
        "script": {
            "speaker": <"1" or "2">,  // "1"=アオイ, "2"=カイ
            "face":    <"normal"|"surprised"|"irritated"|"rage"|"worry">,
            "text":    <string（20文字ごとに \n を挿入）>
        }
        }
    ]
    }
    ===================================================

    【変換ルール】

    1. 「（サブトピックX：...）」という行を見つけたら、
    その直後に type:"topic" セグメントを追加し、title にその文を入れる。

    2. 台詞は「アオイ：」「カイ：」から話者を判定し、speaker に "1" または "2" を設定する。

    3. 各台詞や疑問符や感嘆符から表情を推測し、face に以下の8つの選択肢から最も適切なものを選ぶ(normal1~4はどそれぞれ違う表情。前回選んだnormal表情と違うものを選ぶ)：  
    → "normal1", normal2", normal3", normal4", "surprised", "annoy", "rage", "worry"

    4. 各セグメントの script.text は約30文字以内ごとに改行コード `\\n` を「一度だけ」挿入する（3行にはならない）。挿入は適当に行わずに句点、読点や文脈から適切に判断する。

    5. image や design などこのスキーマに含まれない情報は一切含めてはいけない。

    6. 出力は整形された **JSONオブジェクト1つのみ**。前後に説明文・コードブロック記号（```など）を付けてはいけない。

    【入力例】
    （サブトピック1：Pythonの書きやすさ）  
    アオイ：Pythonってどうして初心者に人気なの？  
    カイ：文法がシンプルで読みやすいからだよ。

    【期待される出力】
    {
    "segments": [
        { "id": 1, "type": "topic", "title": "Pythonの書きやすさ" },
        { "id": 2, "type": "dialogue", "script": {
            "speaker": "1", "face": "surprised",
            "text": "Pythonってどうして\n初心者に人気なの？"
        }},
        { "id": 3, "type": "dialogue", "script": {
            "speaker": "2", "face": "normal",
            "text": "文法がシンプルで\n読みやすいからだよ。"
        }}
    ]
    }

    """


# --------------------------------------------------------------------------- #
# 高レベルクラス
# --------------------------------------------------------------------------- #


class ScenarioGenerator:
    """テーマ＆分数 → 台本文字列"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1-mini"):
        self._client = client
        self._model = model

    def generate(self, theme: str, minutes: int) -> str:
        words_target = minutes * 250
        min_subtopics = max(1, int(words_target / 300))

        sys_prompt = _SYSTEM_PROMPT_SCENARIO.format(
            min_subtopics=min_subtopics, word_target=words_target
        )
        user_prompt = f"【テーマ】{theme}\n【所要時間】{minutes}分"

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


class ScenarioStructurer:
    """台本文字列 → JSON 構造化文字列"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1-mini"):
        self._client = client
        self._model = model

    def to_json(self, script: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=1,
            top_p=0.95,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT},
                {"role": "user", "content": script},
            ],
        )
        # 念のため JSON 検証
        json.loads(resp.choices[0].message.content)
        return resp.choices[0].message.content


class ScenarioService:
    """
    Facade：テーマ/分数を渡すだけで
    ① 台本生成 → ② 構造化 JSON まで一気に返す
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        model_gen: str = "gpt-4.1-mini",
        model_struct: str = "gpt-4.1-mini",
    ):
        load_dotenv()
        self._client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self._generator = ScenarioGenerator(self._client, model=model_gen)
        self._structurer = ScenarioStructurer(self._client, model=model_struct)

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def run(self, theme: str, minutes: int) -> Dict:
        """dict 型の構造化データを返す"""
        print("台本生成中・・・")
        script = self._generator.generate(theme, minutes)
        print(script,"\n\n","台本構造化中・・・")
        structured = self._structurer.to_json(script)
        return json.loads(structured)


if __name__ == "__main__":
    
    from pprint import pprint
    
    theme = "宇宙はどんな形？数学で紐解く宇宙の果てとポアンカレ予想"
    minutes = 3

    svc = ScenarioService()
    structured = svc.run(theme, minutes)
    
    pprint(structured)
    
    with open('b.txt', 'w', encoding='utf-8') as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)