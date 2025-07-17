from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from rich import print


_SYSTEM_PROMPT_TOPICS = """
    あなたは受賞歴のある脚本家です。
    ユーザーから提示された「テーマ」に対して、
    その内容を解説動画に仕立てるための**トピック（小見出し）**を複数提案し、各トピックで語るべき要点を箇条書きしてください。

    # 出力形式（JSON）

    {{
        "introduction": {{
            "title": "<タイトル>",
            "points": [
                "<要点（問い・フック・導入）>" // 要点は1つだけ
            ]
        }},
        "topics": [
            {{
                "title": "<サブトピックタイトル>",
                "points": [
                    "<要点1>" // 要点は1つだけ
                ]
            }}
        // ... 他トピックも同様 ...
        ],
        "conclusion": {{
            "title": "<タイトル>",
            "points": [
                "<まとめ要点>" // 要点は1つだけ
            ]
        }}
    }}

    # 制約・ルール
    - **introduction** では本題に入る前のワクワクするフックを置く。
    - **conclusion** では視聴後の余韻と次への好奇心を残すように。
    - タイトルは15文字以内。
    - 要点は文字数の制限は考えず可能な限り細かく具体的に記述をする。
    - **トピック数** は `{min_subtopics}` 個とする。
    - 各トピックは幅広い一般論を避け、**具体例**を交える。
    - 各トピックはテーマから外れた内容にしない。
    - 「上級者が見て学術的で面白い」レベル感を意識する。
    - 出力は **整形済みJSONオブジェクトのみ**。前後に余計な文字やコードブロック記号を付けない。
"""

_SYSTEM_PROMPT_INTRO = """
    あなたは台本を作るプロフェッショナルです。
    台本のイントロダクショントピックのパートを作成します。
    
    # 前提
    ユーザーからは、台本全体の構成要約（他トピックも含む全体の流れ）と、
    現在注力すべきトピックのタイトルおよびポイントが与えられます。
    全体の流れを踏まえたうえで、解説動画におけるイントロダクションの台本を作成してください。

    # 出力ルール
    1. **台本はプレーンテキストのみ**。
    2. 各台詞は1フレーズごとに改行する。
    3. **1発話60文字以内**、**文章数は6～8文**。
    4. メインパートへの自然な導入を意識する
    5. 挨拶などは入れず、すぐにイントロダクションの本題に入る
    6. テーマ的に可能であれば、読者に対して引きや新たな視点を与えるような問いかけを作る

    # 入力形式（ユーザーから与えられる情報）
    【台本全体の構成（要約）】
    1. <タイトル1> – 要点1 / 要点2 / ...
    2. <タイトル2> – 要点1 / 要点2 / ...
    ...

    【現在のトピック】<現在のタイトル>
    【ポイント】
    - 要点1
    - 要点2
    ...
"""

_SYSTEM_PROMPT_SCENARIO = """
    あなたは対話型台本を作るプロフェッショナルです。
    台本全体を構成する一つのトピックの対話パートを作成します。

    # 前提
    ユーザーからは、台本全体の構成要約（他トピックも含む全体の流れ）と、
    現在注力すべきトピックのタイトルおよびポイントが与えられます。
    全体の流れを踏まえたうえで、現在のトピックの内容だけを丁寧に会話化してください。

    # キャラクター設定
    ## 四国めたん（解説役）
    * 一人称は**「わたくし」**。
    * 二人称は「ずんだもん」。
    * 語尾は「〜かしら」「〜わよ」「〜ですわね」「〜ですのよ」。
    * クールで落ち着いたお嬢様口調。例え話や比喩を使うこともある。

    ## ずんだもん（質問役）
    * 一人称は**「ボク」**。
    * 二人称は「めたん」。
    * 語尾は「〜なのだ」「〜のだ」。疑問形は「〜のだ？」のみ。
    * 明るく元気。ボケ担当。

    # 出力ルール
    1. **台本はプレーンテキストのみ**。
    2. 各台詞は「キャラ名：本文」の形式。
    3. **1発話70文字以内**、**総文字数800文字±5%**。
    4. **1ポイントにつき最低2往復以上**。
    5. 与えられた現在のトピックにのみ焦点を当てること。他の話題に脱線しない。
    6. 会話の自然さを重視する。
    7. 「上級者が見て面白い」レベル感を意識する。

    # 入力形式（ユーザーから与えられる情報）
    【台本全体の構成（要約）】
    1. <タイトル1> – 要点1 / 要点2 / ...
    2. <タイトル2> – 要点1 / 要点2 / ...
    ...

    【現在のトピック】<現在のタイトル>
    【ポイント】
    - 要点1
    - 要点2
    ...
"""

_SYSTEM_PROMPT_STRUCT_INTRO = r"""
    あなたは動画制作パイプラインの「構造化エンジン」です。
    入力された台本を、下記 JSON スキーマに従って厳密に変換してください。

    {
        "introduction": {
            "title": <string>,
            "text": [
                <string>,
                <string>
            ]
        },
    }

    # 変換ルール
    * titleにはテキスト全体を要約した15文字程度の見出し
    (例)「〇〇は〇〇なのか？」「なぜ〇〇は〇〇になるのか？」などキャッチーな見出しに
    * textには時系列順にテキストを1フレーズごとに挿入
    * 前後に余計な文字列・コードブロック記号は禁止。
"""

_SYSTEM_PROMPT_STRUCT = r"""
    あなたは動画制作パイプラインの「構造化エンジン」です。
    入力された台本を、下記 JSON スキーマに従って厳密に変換してください。

    {
        "segments": [
            { "id": <int>, "type": "topic",    "title": <string> },
            { "id": <int>, "type": "dialogue", "script": {
                "speaker": <"1"|"2">,
                "face": <"normal1"|"normal2"|"normal3"|"normal4"|"surprised"|"annoy"|"rage"|"worry">,
                "text": <string>
            }}
        ]
    }

    # 変換ルール
    * トピック行が来たら type:"topic"。
    * 行頭のキャラクター名で speaker 判定。
    * 「？」や「！」を含む → surprised / rage / worry を優先。
    * 連続する normal 表情は 1〜4 をラウンドロビンで変化させる。
    * 前後に余計な文字列・コードブロック記号は禁止。
"""

# ───────────────────────────────────────────────────────────────
# 📝 ScenarioPreprocessor
#    - 台本ポイント（箇条書き）を “問い” 形式に変換するヘルパ
#    - 動画中のダイアログが単調な説明にならないよう、
#      指標単語 → 疑問文 へのヒューリスティックを適用
#    - *副作用なし*・純粋関数的に points → new_points を返す
# ───────────────────────────────────────────────────────────────
class ScenarioPreprocessor:
    clue_words = (
        "なぜ", "どうして", "どうやって", "本当", "マジ", "意味", "原因", "理由",
        "仕組み", "裏側", "怖い", "驚き"
    )

    def convert(self, points: List[str]) -> List[str]:
        out: List[str] = []
        for p in points:
            p = p.strip()
            # Already looks like a question → leave as-is.
            if p.endswith("？") or p.endswith("?"):
                out.append(p)
                continue
            # Heuristics: try to transform statement → question.
            if any(k in p for k in self.clue_words):
                out.append(p + "？")
            else:
                out.append(f"なんで{p}の？")
        return out

# ───────────────────────────────────────────────────────────────
# 🏗️ ScenarioTopicGenerator
#    - テーマ文字列を GPT へ投げ、解説動画の全体構成案(JSON)を生成
#    - 引数 minutes を “最低トピック数” としてプロンプトに組み込む
#    - 戻り値: introduction / topics / conclusion を含む dict
# ───────────────────────────────────────────────────────────────
class ScenarioTopicGenerator:
    """テーマ → トピック + 要点（JSON）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    def generate(self, theme: str, minutes: int) -> Dict:
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
        return json.loads(resp.choices[0].message.content)


# ───────────────────────────────────────────────────────────────
# 🎤 ScenarioGenerator
#    - 1トピック分の *対話台本* を生成
#    - outline 全体を与えることで “前後の流れ” を踏まえた応答を誘導
#    - points は ScenarioPreprocessor で質問化済みのリストを想定
# ───────────────────────────────────────────────────────────────
class ScenarioGenerator:
    """タイトル+ポイント(+全体概要) → 対話台本（プレーンテキスト）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4.1"):
        self._client = client
        self._model = model

    def generate(
        self,
        title: str,
        points: List[str],
        outline: List[Dict[str, List[str]]],   # ★ 追加
    ) -> str:
        # 台本全体の概要を 1 行ずつ整形
        outline_str = "\n".join(
            f"{idx+1}. {t['title']} – {' / '.join(t['points'])}"
            for idx, t in enumerate(outline)
        )

        user_prompt = (
            "【台本全体の構成（要約）】\n"
            f"{outline_str}\n\n"
            f"【現在のトピック】{title}\n"
            "【ポイント】\n" + "\n".join(f"- {p}" for p in points)
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

# ───────────────────────────────────────────────────────────────
# 🎬 IntroductionGenerator
#    - イントロダクション専用の台本を生成（短尺・ナレーション形式）
# ───────────────────────────────────────────────────────────────
class IntroductionGenerator:
    """イントロダクションタイトル+ポイント(+全体概要) → イントロ台本（テキスト）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    def generate(
        self,
        title: str,
        points: List[str],
        outline: List[Dict[str, List[str]]],
    ) -> str:
        outline_str = "\n".join(
            f"{idx+1}. {t['title']} – {' / '.join(t['points'])}"
            for idx, t in enumerate(outline)
        )
        user_prompt = (
            "【台本全体の構成（要約）】\n"
            f"{outline_str}\n\n"
            f"【現在のトピック】{title}\n"
            "【ポイント】\n" + "\n".join(f"- {p}" for p in points)
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_INTRO},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

# ───────────────────────────────────────────────────────────────
# 🗂️ ScenarioStructurer / ScenarioIntroStructurer
#    - LLM 出力テキスト → JSON へのパース専用クラス
#    - *LLM に JSON を吐かせる* アプローチなので実装は薄いが、
#      プロンプト変更時の影響範囲が集中するハブ
# ───────────────────────────────────────────────────────────────
class ScenarioStructurer:
    """台本文字列 → JSON 構造化（segments）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4o-mini"):
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
        data = json.loads(resp.choices[0].message.content)
        return data["segments"]
    

class ScenarioIntroStructurer:
    """イントロ台本文字列 → JSON 構造化（introduction.text[]）"""

    def __init__(self, client: OpenAI, *, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    def to_intro(self, script: str) -> Dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT_INTRO},
                {"role": "user", "content": script},
            ],
        )
        # 生成物は {"introduction": { "text": [...] }}
        return json.loads(resp.choices[0].message.content)["introduction"]

# ───────────────────────────────────────────────────────────────
# 🚀 ScenarioService
#    - 外部 API として “theme → 最終 JSON” を一気通貫で提供する Facade
#    - 主な流れ
#        1. トピック一覧生成
#        2. イントロ生成 → 構造化
#        3. 各サブトピックをループ処理（質問化 → 台本生成 → 構造化）
#        4. 後処理で台詞を自動折り返し
# ───────────────────────────────────────────────────────────────
class ScenarioService:
    """High-level orchestration: theme → structured segments（+ intro）"""

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        model_topic: str = "gpt-4o-mini",
        model_dialogue: str = "gpt-4o",
        model_struct: str = "gpt-4o-mini",
    ) -> None:
        load_dotenv()
        self._client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self._topic_gen   = ScenarioTopicGenerator(self._client, model=model_topic)
        self._intro_gen   = IntroductionGenerator(self._client, model=model_dialogue)
        self._dialogue_gen = ScenarioGenerator(self._client, model=model_dialogue)
        self._intro_struct = ScenarioIntroStructurer(self._client, model=model_struct)
        self._structurer   = ScenarioStructurer(self._client, model=model_struct)
        self._pre = ScenarioPreprocessor()

    # ------------------------------------------------------------------
    def _iter_topics(self, topic_json: Dict) -> List[Dict]:
        """introduction / conclusion を除いた純粋なトピック部分だけを返す"""
        return [{"title": t["title"], "points": t["points"]}
                for t in topic_json["topics"]]

    """35 字以上の台詞を読点・句点を優先して強制改行するユーティリティ"""
    def _wrap_text(self, text: str, *, max_len: int = 35) -> str:
        BREAK_CHARS = "、。.,!?！？…「」『』"
        lines, buf = [], ""
        for ch in text:
            buf += ch
            if len(buf) >= max_len:
                # 直近 5 文字以内に句読点があればそこを優先
                split_pos = next(
                    (len(buf) - i + 1
                    for i in range(1, min(5, len(buf)) + 1)
                    if buf[-i] in BREAK_CHARS),
                    max_len
                )
                lines.append(buf[:split_pos].rstrip())
                buf = buf[split_pos:]
        if buf:
            lines.append(buf)
        return "\n".join(lines)

    def _postprocess_intro(self, intro: Dict,
                           *, max_len: int = 35) -> Dict:
        intro["text"] = [
            self._wrap_text(t, max_len=max_len) for t in intro["text"]
        ]
        return intro

    def _postprocess_segments(self, segments: List[Dict],
                              *, max_len: int = 35) -> List[Dict]:

        for seg in segments:
            if seg.get("type") == "dialogue":
                txt = seg["script"]["text"]
                if len(txt) > max_len:
                    seg["script"]["text"] = self._wrap_text(txt, max_len=max_len)
        return segments

    # ------------------------------------------------------------------
    def run(self, theme: str, minutes: int) -> Dict:
        print("📝 Generating topic list …")
        topic_dict = self._topic_gen.generate(theme, minutes)

        intro_meta = topic_dict["introduction"]  # {'title', 'points'}
        concl_meta = topic_dict["conclusion"]

        # 全体の流れを組む
        outline_all = (
            [intro_meta] + topic_dict["topics"] + [concl_meta]
        )

        # ★ 1) イントロ台本生成 → 構造化
        print("🎬 Generating introduction script …")
        intro_script = self._intro_gen.generate(
            intro_meta["title"], intro_meta["points"], outline_all
        )
        intro_struct = self._intro_struct.to_intro(intro_script)  # {"text":[...]}
        intro_struct = self._postprocess_intro(intro_struct, max_len=30)

        # ★ 2) 各サブトピックを従来通り処理
        all_segments: List[Dict] = []
        id_counter = itertools.count(1)

        for idx, t in enumerate(self._iter_topics(topic_dict), 1):
            print(f"🎬 Topic {idx}: {t['title']}")
            conv_points = self._pre.convert(t["points"])
            script = self._dialogue_gen.generate(t["title"], conv_points, outline_all)
            segments = self._structurer.to_segments(script)
            for seg in segments:
                seg["id"] = next(id_counter)
            all_segments.extend(segments)

        all_segments = self._postprocess_segments(all_segments, max_len=35)

        # ★ 3) 最終 JSON
        return {
            "introduction": intro_struct,
            "segments": all_segments,
            "conclusion": concl_meta,   
        }


# ────────────────────────────────────────────────────────────────────────────────
# 動作テスト
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import format

    THEME = "【雑学】生物に必ず炭素が含まれている理由"
    MINUTES = 2

    ssv = ScenarioService()
    result = ssv.run(THEME, MINUTES)

    result = format.insert_sound_info(
        result,
        intro_bgm="llm_video_generation/assets/bgm/Mineral.mp3",
        intro_se="llm_video_generation/assets/se/5.mp3",
        body_bgm="llm_video_generation/assets/bgm/Voice.mp3",
        body_se="llm_video_generation/assets/se/3.mp3"
    )

    with open('./llm_video_generation/src/s.txt', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
