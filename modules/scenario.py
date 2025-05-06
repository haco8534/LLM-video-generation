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
        "   c. 【まとめ】全体の 5 % : カイが要点整理、アオイが学びを言語化。",
        "4. 文体: やさしい日本語＋必要に応じて専門用語。",
        "5. 長さ目標: 約 {word_target} 語（±5 %）。",
        "6. 思考過程は出力に含めない。",
    ]
)

_SYSTEM_PROMPT_STRUCT = """
あなたは動画制作パイプラインの「構造化エンジン」です。  
入力された台本を、下記 JSON スキーマに沿って正確に変換してください。

================== JSON スキーマ ==================
{
    "segments": [
        {
            "id":    <int 1 から連番>,
            "type":  "topic",
            "title": <サブトピックの見出し>
        },
        {
            "id":    <int>,
            "type":  "dialogue",
            "script": {
                "speaker": <"アオイ" | "カイ">,
                "text":    <台詞>
        }
        }
    ]
}
===================================================

### 変換ルール
1. 台本中の「サブトピック n：〜」を検出し type:"topic" として追加。  
2. 台詞行は「アオイ：」「カイ：」で判定し type:"dialogue"。  
3. id は 1 から挿入順で +1。  
4. 不要項目を出力しない。  
5. **整形済み JSON のみ** を返し、前後に説明やコードブロックを付けない。
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
    
    theme = "【ORMの定番】SQLAlchemyを分かりやすく解説！PythonからDBを簡単操作〜初心者向け〜"
    minutes = 3

    svc = ScenarioService()
    structured = svc.run(theme, minutes)
    
    pprint(structured)
