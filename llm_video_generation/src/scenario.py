"""
scenario_refactored.py
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ===================== 外部ライブラリ =====================
import os
import json
import itertools
from typing import Dict, List, Sequence
from dotenv import load_dotenv
from openai import OpenAI
# =========================================================

# ===================== 共通定数 ===========================
# OpenAI モデル名
MODEL_TOPIC   = "gpt-5"
MODEL_DIALOG  = "gpt-5"
MODEL_STRUCT  = "gpt-5"

# 台詞の自動折り返し長
WRAP_LEN_DIALOG   = 35  # メイン・結論
WRAP_LEN_INTRO    = 30  # イントロ

# システムプロンプト
# --- 要点リスト生成用
_SYSTEM_PROMPT_TOPICS = """
    あなたは受賞歴のある脚本家です。
    ユーザーから提示された「テーマ」と「参考資料」に基づき、
    解説動画に仕立てるための **トピック（小見出し）** と各トピックで語るべき要点を抽出してください。

    # 出力形式（JSON）

    {
        "introduction": {
            "title": "<タイトル>",
            "points": ["<要点（問い・フック・導入）>"]
        },
        "topics": [
            {
                "title": "<サブトピックタイトル>",
                "points": ["<要点1>"]
            }
        ],
        "conclusion": {
            "title": "<タイトル>",
            "points": ["<まとめ要点>"]
        }
    }

    # 制約・ルール
    - 必ず【参考資料】の内容に基づいて要点を抽出すること。
    - 想像や補完は禁止。資料に書かれていないことは言及しない。
    - 各要点は資料の内容に忠実に、できるだけ具体的に記述する。(何年の、誰の、どのような研究か)
    - 要点はどれだけ長くなっても良いので可能な限り細かく記述する
    - タイトルは15文字以内。
    - **トピック数** は 参考資料のトピックの数とする。
    - 出力は整形済みJSONオブジェクトのみ。前後に余計な文字列は禁止。
"""

# --- イントロ台本生成用
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

# --- メイン・結論台本生成用
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

    ## ずんだもん（聞き手役）
    * 一人称は**「ボク」**。
    * 二人称は「めたん」。
    * 語尾は「〜なのだ」「〜のだ」。疑問形は「〜のだ？」のみ。

    # 出力ルール
    1. 「ファクトベース」で厳密で論理的な台本。要点に書かれているファクトを論理的に考察する。
    2. **台本はプレーンテキストのみ**。
    3. 各台詞は「キャラ名：本文」の形式。
    4. **1発話70文字以内**、**総文字数800文字±5%**。
    5. 与えられた現在のトピックにのみ焦点を当てること。他の話題に脱線しない。
    6. 会話の自然さを重視する。

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

_SYSTEM_PROMPT_OUTTRO = """
    あなたは台本を作るプロフェッショナルです。
    台本の **結論（アウトロ）** パートを作成します。

    # 前提
    ユーザーからは、台本全体の構成要約と
    現在注力すべき「結論」トピックのタイトル・ポイントが与えられます。
    視聴後余韻とリキャップを意識しつつ、次のアクションや問い掛けで締めてください。

    # キャラクター設定
    ## 四国めたん（解説役）
    * 一人称は**「わたくし」**。
    * 二人称は「ずんだもん」。
    * 語尾は「〜かしら」「〜わよ」「〜ですわね」「〜ですのよ」。
    * クールで落ち着いたお嬢様口調。例え話や比喩を使うこともある。

    ## ずんだもん（聞き手役）
    * 一人称は**「ボク」**。
    * 二人称は「めたん」。
    * 語尾は「〜なのだ」「〜のだ」。疑問形は「〜のだ？」のみ。

    # 出力ルール
    1. **台本はプレーンテキストのみ**。
    2. 各台詞は「キャラ名：本文」の形式。
    3. **1発話70文字以内**、**総文字数200文字±5%**。
    4. 与えられた現在のトピックにのみ焦点を当てること。他の話題に脱線しない。
    5. 最初の文章はメインの解説が終わったかのように「四国めたん:いかかだったかしら」「ずんだもん：〇〇についてよくわかったのだ」のように結論に入る流れを作る。
    6. 最後の文章は解説動画の最後の文章としてふさわしいように締めくくる。
"""

# --- 台本構造化用プロンプト
_SYSTEM_PROMPT_STRUCT_INTRO = r"""
    あなたは動画制作パイプラインの「構造化エンジン」です。
    入力された台本を、下記 JSON スキーマに従って厳密に変換してください。

    {
        "introduction": {
            "title": <string>,
            "text": [
                {
                "id": <int>,
                "script": <string>
                },
                {
                "id": <int>,
                "script": <string>
                },
            ]
        },
    }

    # 変換ルール
    * titleにはテキスト全体を要約した15文字程度の見出し
    (例)「〇〇は〇〇なのか？」「なぜ〇〇は〇〇になるのか？」などキャッチーな見出しに
    * textsのscriptには時系列順にテキストを1フレーズごとに挿入
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

_SYSTEM_PROMPT_STRUCT_OUTRO = r"""
    あなたは動画制作パイプラインの「構造化エンジン」です。
    入力された台本を下記 JSON スキーマに従って厳密に変換してください。

    {
        "conclusion": {
            "title": <string>,
            "text": [
                { "id": <int>, "type": "topic",    "title": <string> },
                { "id": <int>, "type": "dialogue", "script": {
                    "speaker": <"1"|"2">,
                    "face": <"normal1"|"normal2"|"normal3"|"normal4"|"surprised"|"annoy"|"rage"|"worry">,
                    "text": <string>
                }}
            ]
        }
    }

    # 変換ルール
    * トピック行が来たら type:"topic"。
    * 行頭のキャラクター名で speaker 判定。
    * 「？」や「！」を含む → surprised / rage / worry を優先。
    * 連続する normal 表情は 1〜4 をラウンドロビンで変化させる。
    * 前後に余計な文字列・コードブロック記号は禁止。
"""

# =========================================================

# ===================== ユーティリティ ====================
_BREAK_CHARS = "、。.,!?！？…『』「」"

def _wrap_text(text: str, max_len: int) -> str:
    """指定長を超えたら句読点優先で改行する"""
    buf, lines = "", []
    for ch in text:
        buf += ch
        if len(buf) >= max_len:
            # 直近 5 文字以内の句読点優先
            split_pos = next(
                (len(buf) - i + 1 for i in range(1, min(5, len(buf)) + 1)
                    if buf[-i] in _BREAK_CHARS),
                max_len
            )
            lines.append(buf[:split_pos].rstrip())
            buf = buf[split_pos:]
    if buf:
        lines.append(buf)
    return "\n".join(lines)

def _sanitize_chat_opts(model: str, **opts):
    """
    モデルごとに未サポートな生成パラメータを落とす。
    例: gpt-5 は temperature/top_p 等の任意値を受け付けない。
    """
    if model.startswith("gpt-5"):
        # gpt-5 系はデフォルト以外を受け付けないので全部落とす
        return {}
    # None を除去してクリーンに
    return {k: v for k, v in opts.items() if v is not None}
# =========================================================

# ===================== OpenAI ラッパ ======================
class OpenAIClient:
    """最小限のラッパ。chat(...) でテキストを直接返す"""

    def __init__(self, api_key: str | None = None):
        load_dotenv()
        self._cli = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def chat(self, model: str, messages: Sequence[Dict], **opts) -> str:
        clean_opts = _sanitize_chat_opts(model, **opts)
        res = self._cli.chat.completions.create(
            model=model,
            messages=messages,
            **clean_opts,
        )
        return res.choices[0].message.content.strip()
# =========================================================

# ===================== シナリオ生成 =======================
class ScenarioBuilder:
    """テーマ + 参考資料 から構造化済みシナリオ JSON を生成する高レベル API"""

    def __init__(self, client: OpenAIClient):
        self.ai = client

    # ---------- ステップ 1: 要点リスト ----------
    def _make_topic_outline(self, theme: str, minutes: int, reference: str = "") -> Dict:
        sys_p = _SYSTEM_PROMPT_TOPICS
        user_p = f"【テーマ】{theme}\n\n【参考資料】\n{reference}"
        content = self.ai.chat(
            MODEL_TOPIC,
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.2,
        )
        return json.loads(content)

    # ---------- ステップ 2: イントロ ----------
    def _build_intro(self, intro_meta: Dict, outline_all: List[Dict], theme: str) -> Dict:
        script = self._generate_intro_script(intro_meta, outline_all, theme)
        intro_json = self._structure_intro(script)
        # 折り返し
        for t in intro_json["text"]:
            t["script"] = _wrap_text(t["script"], WRAP_LEN_INTRO)
        return intro_json

    def _generate_intro_script(self, meta: Dict, outline: List[Dict], theme: str) -> str:
        outline_str = "\n".join(
            f"{i+1}. {o['title']} – {' / '.join(o['points'])}"
            for i, o in enumerate(outline)
        )
        user_p = (
            f"【テーマ】{theme}\n"                                 # ← ★追加
            "【台本全体の構成（要約）】\n" + outline_str + "\n\n" +
            f"【現在のトピック】{meta['title']}\n" +
            "【ポイント】\n" + "\n".join(f"- {p}" for p in meta["points"])
        )
        return self.ai.chat(
            MODEL_DIALOG,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_INTRO},
                {"role": "user",   "content": user_p},
            ],
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.2,
        )

    def _structure_intro(self, script: str) -> Dict:
        content = self.ai.chat(
            MODEL_STRUCT,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT_INTRO},
                {"role": "user",   "content": script},
            ],
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
        )
        return json.loads(content)["introduction"]

    # ---------- ステップ 3: メイントピック（ループ） ----------
    def _build_main_segments(self, topics: List[Dict], outline_all: List[Dict], theme: str) -> List[Dict]:
        segments: List[Dict] = []
        counter = itertools.count(1)
        for idx, t in enumerate(topics, 1):
            print(f"▶ メイントピック {idx}: {t['title']}")
            # 質問化
            conv_points = self._convert_to_questions(t["points"])
            # 台本生成 → 構造化
            script = self._generate_dialogue_script(t["title"], conv_points, outline_all, theme)
            segs   = self._structure_dialogue(script)
            # id 付与 & 折り返し
            for s in segs:
                s["id"] = next(counter)
                if s.get("type") == "dialogue":
                    s["script"]["text"] = _wrap_text(s["script"]["text"], WRAP_LEN_DIALOG)
            segments.extend(segs)
        return segments

    def _convert_to_questions(self, points: List[str]) -> List[str]:
        """ヒューリスティックに陳述 → 疑問文へ"""
        clue = ("なぜ", "どうして", "どうやって", "本当", "原因", "理由", "仕組み", "裏側")
        out: List[str] = []
        for p in points:
            p = p.strip()
            if p.endswith("？") or p.endswith("?"):
                out.append(p)
            elif any(k in p for k in clue):
                out.append(p + "？")
            else:
                out.append(f"なんで{p}の？")
        return out

    def _generate_dialogue_script(
        self, title: str, points_q: List[str],
        outline: List[Dict], theme: str, *, mode: str="body"
    ) -> str:
        outline_str = "\n".join(
            f"{i+1}. {o['title']} – {' / '.join(o['points'])}"
            for i, o in enumerate(outline)
        )
        user_p = (
            f"【テーマ】{theme}\n"
            "【台本全体の構成（要約）】\n" + outline_str + "\n\n" +
            f"【現在のトピック】{title}\n" +
            "【ポイント】\n" + "\n".join(f"- {p}" for p in points_q)
        )
        sys_p = _SYSTEM_PROMPT_OUTTRO if mode == "conclusion" else _SYSTEM_PROMPT_SCENARIO
        return self.ai.chat(
            MODEL_DIALOG,
            [
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": user_p},
            ],
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.2,
        )

    def _structure_dialogue(self, script: str) -> List[Dict]:
        content = self.ai.chat(
            MODEL_STRUCT,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT},
                {"role": "user",   "content": script},
            ],
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
        )
        return json.loads(content)["segments"]

    # ---------- ステップ 4: 結論 ----------
    def _build_conclusion(self, concl_meta: Dict, outline_all: List[Dict], theme: str) -> Dict:
        script = self._generate_dialogue_script(
            concl_meta["title"],
            self._convert_to_questions(concl_meta["points"]),
            outline_all,
            theme,
            mode="conclusion",
        )
        concl_json = self._structure_conclusion(script)
        # id 付与 & 折り返し
        counter = itertools.count(1)
        for seg in concl_json["text"]:
            seg["id"] = next(counter)
            if seg.get("type") == "dialogue":
                seg["script"]["text"] = _wrap_text(seg["script"]["text"], WRAP_LEN_DIALOG)
        return concl_json

    def _structure_conclusion(self, script: str) -> Dict:
        content = self.ai.chat(
            MODEL_STRUCT,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT_OUTRO},
                {"role": "user",   "content": script},
            ],
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
        )
        return json.loads(content)["conclusion"]

    # ---------- Public API ----------
    def run(self, theme: str, minutes: int, reference: str = "") -> Dict:
        """テーマ + 参考資料 → 完全構造化 JSON"""
        # 1) 要点リスト（アウトライン）
        outline = self._make_topic_outline(theme, minutes, reference)
        intro_meta = outline["introduction"]
        topics_meta = outline["topics"]
        concl_meta = outline["conclusion"]
        outline_all = [intro_meta] + topics_meta + [concl_meta]

        # 2) イントロ
        intro_json    = self._build_intro(intro_meta, outline_all, theme)

        # 3) メインループ
        main_segments = self._build_main_segments(topics_meta, outline_all, theme)

        # 4) 結論
        concl_json    = self._build_conclusion(concl_meta, outline_all, theme)

        # 5) 連結して返す
        return {
            "theme": theme,                     # ← ★ここを追加
            "introduction": intro_json,
            "segments": main_segments,
            "conclusion": concl_json,
        }
# =========================================================

# ===================== 動作テスト =========================
if __name__ == "__main__":
    from rich import print
    import format

    THEME = "あなたはなぜ“つい後回し”してしまうのか？"
    MINUTES = 2


    with open(r"llm_video_generation\data\1\ref.txt", 'r', encoding='utf-8') as f:
        REFERENCE = f.readlines()

    builder = ScenarioBuilder(OpenAIClient())
    result = builder.run(THEME, MINUTES, reference=REFERENCE)

    result = format.insert_sound_info(
        result,
        intro_bgm="llm_video_generation/assets/bgm/Mineral.mp3",
        intro_se="llm_video_generation/assets/se/5.mp3",
        body_bgm="llm_video_generation/assets/bgm/Voice.mp3",
        body_se="llm_video_generation/assets/se/3.mp3",
    )

    with open("./llm_video_generation/src/s.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))

