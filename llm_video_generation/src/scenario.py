from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI


_SYSTEM_PROMPT_TOPICS = """
    あなたは受賞歴のある脚本家です。
    ユーザーから提示された「テーマ」に対して、
    その内容をエンタメ解説動画に仕立てるための**トピック（小見出し）**を複数提案し、各トピックで語るべき要点をユーモラスに箇条書きしてください。

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
                    "<要点1>",
                    "<要点2>",
                    "<要点3>"
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
    1. **introduction** では本題に入る前のワクワクするフックを置く。突飛な比喩やワンライナー歓迎。
    2. **conclusion** では視聴後の余韻と次への好奇心を残すように。
    3. **トピック数** は `{min_subtopics}` 個を目安とする。
    4. 各トピックは幅広い一般論を避け、**具体例・最新トレンド・意外な裏話**を交える。
    5. **硬い表現NG**。キャッチーで取っつきやすい言い回しに。
    6. 広すぎる抽象語で終わらず、**ピンポイントな視点・事例**を必ず含める。
    7. 「初心者にも分かるが深掘りできる」レベル感を意識する。
    8. 出力は **整形済みJSONオブジェクトのみ**。前後に余計な文字やコードブロック記号を付けない。
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
    * クールで落ち着いたお嬢様口調。例え話と比喩を多用。

    ## ずんだもん（質問役）
    * 一人称は**「ボク」**。
    * 二人称は「めたん」。
    * 語尾は「〜なのだ」「〜のだ」。疑問形は「〜のだ？」のみ。
    * 明るく元気でテンポ良くツッコミを入れる。

    # 出力ルール
    1. **台本はプレーンテキストのみ**。
    2. 各台詞は「キャラ名：本文」の形式。
    3. **1発話60文字以内**、**総文字数500文字±5%**。
    4. **1ポイントにつき最低2往復以上**。
    5. 難解表現は避け、例え・比喩を織り交ぜる。
    6. 与えられた現在のトピックにのみ焦点を当てること。他の話題に脱線しない。
    7. 会話の自然さとテンポの良さを重視する。

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

# ────────────────────────────────────────────────────────────────────────────────
# ❶ Pre-processor : abstract → conversational questions
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# ❷ テーマから、小見出しとそれぞれの要点をに出力
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# ❸ Dialogue generator – points are pre-processed before use
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# ❹ Structurer (unchanged logic)
# ────────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────────
# ❺ Facade – rewired to (a) preprocess points and (b) skip intro/conclusion
# ────────────────────────────────────────────────────────────────────────────────

class ScenarioService:
    """High-level orchestration: theme → structured segments."""

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
        self._topic_gen = ScenarioTopicGenerator(self._client, model=model_topic)
        self._dialogue_gen = ScenarioGenerator(self._client, model=model_dialogue)
        self._structurer = ScenarioStructurer(self._client, model=model_struct)
        self._pre = ScenarioPreprocessor()

    # ────────────────────────────────────────────────────────────────────
    def _iter_topics(self, topic_json: Dict) -> List[Dict]:
        """Yield only real sub-topics; intro/conclusion kept aside."""
        return [{"title": t["title"], "points": t["points"]} for t in topic_json["topics"]]

    def _postprocess_segments(self, segments: List[Dict], *, max_len: int = 35) -> List[Dict]:
        """dialogueセグメントの台本に対して、文字数制限に応じた改行を挿入"""
        def wrap_text(text: str) -> str:
            BREAK_CHARS = "、。.,!?！？…「」『』"
            lines, buf = [], ""
            for ch in text:
                buf += ch
                if len(buf) >= max_len:
                    split_pos = None
                    for i in range(1, min(5, len(buf)) + 1):
                        if buf[-i] in BREAK_CHARS:
                            split_pos = len(buf) - i + 1
                            break
                    if split_pos is None:
                        split_pos = max_len
                    lines.append(buf[:split_pos].rstrip())
                    buf = buf[split_pos:]
            if buf:
                lines.append(buf)
            return "\n".join(lines)

        for seg in segments:
            if seg.get("type") == "dialogue":
                txt = seg["script"]["text"]
                if len(txt) > max_len:
                    seg["script"]["text"] = wrap_text(txt)
        return segments

    # ────────────────────────────────────────────────────────────────────
    def run(self, theme: str, minutes: int) -> Dict:
        print("📝 Generating topic list …")
        topic_dict = self._topic_gen.generate(theme, minutes)

        intro = topic_dict["introduction"]
        concl = topic_dict["conclusion"]

        # 台本全体の概要 (= intro + topics + concl) を 1 つのリストにまとめる ★
        outline_all = (
            [{"title": intro["title"], "points": intro["points"]}]
            + topic_dict["topics"]
            + [{"title": concl["title"], "points": concl["points"]}]
        )

        all_segments: List[Dict] = []
        id_counter = itertools.count(1)

        for idx, t in enumerate(self._iter_topics(topic_dict), 1):
            print(f"🎬 Topic {idx}: {t['title']}")
            conv_points = self._pre.convert(t["points"])
            # ★ 追加パラメータ outline_all を渡す
            script = self._dialogue_gen.generate(t["title"], conv_points, outline_all)
            segments = self._structurer.to_segments(script)
            for seg in segments:
                seg["id"] = next(id_counter)
            all_segments.extend(segments)

        all_segments = self._postprocess_segments(all_segments, max_len=35)

        return {
            "introduction": intro,
            "segments": all_segments,
            "conclusion": concl,
        }


# ────────────────────────────────────────────────────────────────────────────────
# Quick CLI test (will run only if this file is executed directly)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    THEME = "存在しない漢字を、なぜ入力できるのか？ 世にも恐ろしい技術的負債の話。"
    MINUTES = 2

    svc = ScenarioService()
    result = svc.run(THEME, MINUTES)
    with open('./llm_video_generation/src/s.txt', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
