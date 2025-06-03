"""
1. 指定テーマ＆長さで日本語対話シナリオを生成
2. 生成された台本を JSON 構造化スキーマへ変換

外部依存
--------
- openai>=1.0.0
- python-dotenv>=1.0.0   # .env から API キー取得用
"""

import json
import os
import itertools
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from rich import print

# --------------------------------------------------------------------------- #
# プロンプトテンプレート
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT_TOPICS =  """
        あなたは受賞歴のある脚本家です。
        ユーザーから提示された「テーマ」に対して、
        その内容のエンタメとしての解説動画を作るために適切な**トピック（小見出し）**を複数提案し、
        それぞれのトピックについて話すべき要点を箇条書きで整理してください。

        ## 出力形式（JSON）
        {{
        "introduction": {{
            "title": "<タイトル>",
            "points": [
                "<要点（話すべき問い・内容）>",
            ]
        }},
        "topics": [
            {{
            "title": "<サブトピックのタイトル>",
            "points": [
                "<要点1（話すべき問い・内容）>",
                "<要点2>",
                "<要点3（例・背景・誤解されがちな点などでもOK）>"
            ]
            }},
            // ... 他のトピックも同様に ...
        ],
        "conclusion": {{
            "title": "<タイトル>",
            "points": [
                "<要点1（topicのまとめで必要な分だけ生成）>",
            ]
        }}

        ## 制約・ルール
        - introductionではトピックに入る前の自然な話題を提示する。ユニークであるほどよい
        - トピック数は {min_subtopics} 個を目安とする
        - 各トピックは、面白みのない広すぎる一般論は避け、具体例や最新トレンドを盛り込んでオリジナリティを出す。
        - 硬い表現はあまり用いず、キャッチーで取っつきやすい印象にする。
        - 広すぎる抽象語（例: 「技術の進化」や「社会の変化」など）だけで終わらせず、**具体的な観点・視点・事例**を入れる
        - 「初心者にもわかるが、深掘りできる」レベルを意識する
        - 出力は整形された **JSONオブジェクト1つのみ**。前後に説明文・コードブロック記号（```など）を付けてはいけない。
        """

_SYSTEM_PROMPT_SCENARIO = """
    あなたは対話型台本を作るプロフェッショナルです。  
    以下のルールに従って、「キャラクター2人の対話形式」で分かりやすく構成された台本を生成してください。

    # キャラクター設定
    ■ 四国めたん（解説役）
        - 一人称は「わたくし」を使用してください。
        - 語尾には「〜かしら」「〜わよ」「〜ですわね」「〜ですよ」などを使用し、落ち着いた親しみやすい口調で話してください。
        - クールで落ち着いた性格。丁寧で知的な語り口。お嬢様。
        - 難しい内容でも理解を助けるためにわかりやすい例を提示たりする。
    - ■ ずんだもん（質問役）
        - 一人称は「ボク」を使用。
        - 語尾には「〜なのだ」「〜のだ」を多用し、柔らかく優しい口調で話す。
        - 疑問形では「〜のだ？」を使用し、「〜のだろうか？」のような硬い表現は使わない。
        - 明るく元気な性格で、少し調子に乗りやすいが、不幸体質な一面もあるキャラクター。
        - シュールな状況では「そんなことあるのだ？」と冷静にツッコむこともある。

    # 出力ルール
    - 台本は完全にプレーンテキストで出力すること。
    - キャラクターが一度に喋る文字数は50文字以内にする
    - 全文字数は200文字(±5%)にする。
    - キャラクター名「ずんだもん：」「四国めたん：」を台詞の先頭につけること。
    - 1つのポイントにつき最低2往復以上（質問→解説→感想→補足など）を含めて会話を展開すること。
    - 話の流れが自然になるよう、必ずタイトルに対応する導入の一言または要約を最初に入れてから本題に入ること。
    - 難解すぎる表現は避け、例や比喩を交えて内容に踏み込んで説明すること。
    - 会話はすべて「本文部分」のみとし、**導入（きっかけ）やまとめ（結論や振り返り）は一切含めない**。
    - 淡白な台本にならないようにする。ユニークであるほどよい。

    # 禁止事項
    - 導入の雑談やテーマ紹介を書かない。
    - 会話の終わりに「なるほど」や「勉強になった」などで締めくくらない。
    - 「今日は〜について話そう」などトピック全体の振り返りをしない。
"""

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
            "speaker": <"1" or "2">,  // "1"=解説役, "2"=質問役
            "face":    <"normal"|"surprised"|"irritated"|"rage"|"worry">,
            "text":    <string（約30文字ごとに \n を挿入）>
        }
        }
    ]
    }
    ===================================================

    【変換ルール】

    1. 「（サブトピックX：...）」という行を見つけたら、
    その直後に type:"topic" セグメントを追加し、title にその文を入れる。

    2. 台詞は「解説役」「質問役」から話者を判定し、speaker に "1" または "2" を設定する。

    3. 各台詞や疑問符や感嘆符から表情を推測し、face に以下の8つの選択肢から最も適切なものを選ぶ(normal1~4はどそれぞれ違う表情。前回選んだnormal表情と違うものを選ぶ)：  
    → "normal1", normal2", normal3", normal4", "surprised", "annoy", "rage", "worry"

    4. 各セグメントの script.text は約30文字を超える場合は「必ず」改行コード `\\n` を「一度だけ」挿入する（3行にはならない）。挿入は文章の半分あたりで行い、句点、読点や文脈から適切に判断する。

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
# クラス 1/3 台本のタイトルと要点リスト
# --------------------------------------------------------------------------- #
class ScenarioTopicGenerator:
    """テーマ → トピック + 要点（JSON）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1-mini"):
        self._client = client
        self._model = model

    def generate(self, theme: str, minutes: int) -> Dict:
        # 分数 ≒ トピック数の簡易基準
        min_subtopics = max(1, minutes)

        sys_prompt = _SYSTEM_PROMPT_TOPICS.format(min_subtopics=min_subtopics)
        user_prompt = f"【テーマ】{theme}"

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # JSON を返している想定なのでバリデーションして dict に
        return json.loads(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# クラス 2/3  台本生成（トピックごと）
# --------------------------------------------------------------------------- #
class ScenarioGenerator:
    """タイトル+ポイント → 対話台本（プレーンテキスト）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1-mini"):
        self._client = client
        self._model = model

    def generate(self, title: str, points: List[str]) -> str:
        user_prompt = f"【タイトル】{title}\n【ポイント】\n" + "\n".join(
            [f"- {p}" for p in points]
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_SCENARIO},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# クラス 3/3  構造化（台本 → JSON segments）
# --------------------------------------------------------------------------- #
class ScenarioStructurer:
    """台本文字列 → JSON 構造化（segments）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1"):
        self._client = client
        self._model = model

    def to_segments(self, script: str) -> List[Dict]:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT},
                {"role": "user", "content": script},
            ],
        )
        segments_json = resp.choices[0].message.content
        data = json.loads(segments_json)
        return data["segments"]


# --------------------------------------------------------------------------- #
# Facade  全処理パイプライン
# --------------------------------------------------------------------------- #
class ScenarioService:
    """
    Facade：テーマ/分数を渡すだけで
        ① トピック構成 → ② 各トピック台本 → ③ 各台本の構造化 →
        ④ すべて統合した segments JSON を返す
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
        self._topic_gen = ScenarioTopicGenerator(self._client, model=model_gen)
        self._script_gen = ScenarioGenerator(self._client, model=model_gen)
        self._structurer = ScenarioStructurer(self._client, model=model_struct)

    # ------------------------------------------------------------------ #
    def _flatten_topic_dict(self, raw: Dict) -> List[Dict]:
        """introduction / topics[] / conclusion を 1 直列リストに"""
        out: List[Dict] = []
        out.append({"title": raw["introduction"]["title"], "points": raw["introduction"]["points"]})
        out.extend({"title": t["title"], "points": t["points"]} for t in raw["topics"])
        out.append({"title": raw["conclusion"]["title"], "points": raw["conclusion"]["points"]})
        return out

    # ------------------------------------------------------------------ #
    def run(self, theme: str, minutes: int) -> Dict:
        """全セグメントを 1 つの dict にまとめて返す"""
        print("📝 1) トピック構成を生成中…")
        topic_dict = self._topic_gen.generate(theme, minutes)
        print(topic_dict)
        topic_list = self._flatten_topic_dict(topic_dict)

        all_segments: List[Dict] = []
        id_counter = itertools.count(1)  # 連番ジェネレータ

        # --- 各トピックを処理 -----------------------------------------
        for idx, t in enumerate(topic_list,1):
            print(f"🎬 2-{idx: 04d}) 台本生成中: {t['title']}")
            script = self._script_gen.generate(t["title"], t["points"])

            print(f"📑 3-{idx: 04d}) 構造化中: {t['title']}")
            segs = self._structurer.to_segments(script)

            # id を全体でユニークになるよう振り直す
            for seg in segs:
                seg["id"] = next(id_counter)
                all_segments.append(seg)

        print("✅ 4) 完了 ー セグメント統合")
        return {"segments": all_segments}


# --------------------------------------------------------------------------- #
# 動作確認
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    theme = "高速逆平方根アルゴリズム〜たった数行に込められた驚異の数学〜【数値計算法】"
    minutes = 3

    svc = ScenarioService()
    result = svc.run(theme, minutes)
    print(result)

    with open('./llm_video_generation/src/a.txt', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)