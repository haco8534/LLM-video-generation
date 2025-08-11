"""
scenario.py (refactored)
────────────────────────────────────────────────────────────
- 定数・プロンプト・ユーティリティを集約
- OpenAI呼び出しは単一点 (OpenAIClient) に集約
- すべての生成→構造化のフローは ScenarioBuilder から呼ぶだけ
- __main__ 直実行時のみダンプをONにできる（環境変数でも切替可）
"""
from __future__ import annotations

# ===================== 標準ライブラリ =====================
import os
import json
import re
import itertools
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Sequence, Any

# ===================== 外部ライブラリ =====================
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================================
# =============== 定数（モデル / 体裁 / 環境） ===============
# OpenAI モデル名
MODEL_TOPIC: str  = "gpt-4.1"
MODEL_DIALOG: str = "gpt-5-mini"
MODEL_STRUCT: str = "gpt-4.1-mini"

# 自動折返し長
WRAP_LEN_DIALOG: int = 35  # メイン・結論
WRAP_LEN_INTRO: int  = 30  # イントロ

# ダンプ（デバッグ用）
DEBUG_DUMP: bool = os.getenv("SCENARIO_DEBUG_DUMP", "0") in {"1", "true", "True"}
_DUMP_DIR: Path = Path(os.getenv("SCENARIO_DUMP_DIR", ".ai_dumps"))

# ==========================================================
# ====================== プロンプト群 =======================
# 可能な限り一元管理。ScenarioBuilder 内では参照のみ。

_SYSTEM_PROMPT_TOPICS = r"""
あなたは受賞歴のある脚本家です。
ユーザーから提示された「テーマ」と「参考資料」に基づき、
解説動画に仕立てるための **トピック（小見出し）** と各トピックで語るべき要点を抽出してください。

# 出力形式（JSON）
{
    "introduction": {"title": "<タイトル>", "points": ["<要点>"]},
    "topics": [{"title": "<サブトピックタイトル>", "points": ["<要点1>"]}],
    "conclusion": {"title": "<タイトル>", "points": ["<まとめ要点>"]}
}

# 制約・ルール
- 必ず【参考資料】の内容に基づいて要点を抽出すること。
- 想像や補完は禁止。資料に書かれていないことは言及しない。
- 各要点は資料の内容に忠実に、できるだけ具体的に記述する。(何年/誰/どの研究か)
- 要点は長くなっても良いので可能な限り細かく記述する。
- タイトルは15文字以内。
- **トピック数** は参考資料のトピック数とする。
- 出力は整形済みJSONオブジェクトのみ。前後に余計な文字列は禁止。
"""

_SYSTEM_PROMPT_INTRO = r"""
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

_SYSTEM_PROMPT_SCENARIO = r"""
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

_SYSTEM_PROMPT_OUTTRO = r"""
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

# ==========================================================
# ====================== ユーティリティ ====================
_BREAK_CHARS = "、。.,!?！？…『』「」"


def _wrap_text(text: str, max_len: int) -> str:
    """指定長を超えたら句読点優先で改行する。"""
    buf, lines = "", []
    for ch in text:
        buf += ch
        if len(buf) >= max_len:
            split_pos = next(
                (len(buf) - i + 1 for i in range(1, min(5, len(buf)) + 1)
                    if buf[-i] in _BREAK_CHARS),
                max_len,
            )
            lines.append(buf[:split_pos].rstrip())
            buf = buf[split_pos:]
    if buf:
        lines.append(buf)
    return "\n".join(lines)


def _slug(s: str, max_len: int = 64) -> str:
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r"[^\w\-_.]", "", s)
    return s[:max_len] or "untitled"


def _set_dump_dir(base: Path | None = None) -> None:
    """実行ごとにタイムスタンプ付きの保存フォルダを作成する。"""
    global _DUMP_DIR
    base = base or Path(".ai_dumps")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    _DUMP_DIR = base / ts
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)


def _dump(name: str, data: Any, ext: str = ".txt") -> None:
    """DEBUG_DUMP が True のときだけ保存。dict/list は JSON 整形。"""
    if not DEBUG_DUMP:
        return
    p = _DUMP_DIR / f"{name}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        p.write_text(str(data), encoding="utf-8")


def _outline_str(outline: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{i+1}. {o['title']} – {' / '.join(o['points'])}" for i, o in enumerate(outline)
    )


def _safe_json_loads(s: str, ctx_name: str) -> Any:
    """JSON パースに失敗した場合、raw をダンプして例外に文脈を載せる。"""
    try:
        return json.loads(s)
    except Exception as e:
        _dump(f"error_{_slug(ctx_name)}_raw", s, ".json")
        raise ValueError(f"Invalid JSON returned at {ctx_name}: {e}")

# ==========================================================
# ===================== OpenAI クライアント =================
class OpenAIClient:
    """OpenAI Chat API を最小限で叩くラッパ。chat(...) が str を返す。"""

    def __init__(self, api_key: str | None = None):
        load_dotenv()
        self._cli = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def chat(self, model: str, messages: Sequence[Dict[str, Any]], **opts) -> str:
        # gpt-5 系の追加オプション（互換のため任意）
        gpt5_reasoning_effort = opts.pop("gpt5_reasoning_effort", None)
        gpt5_verbosity        = opts.pop("gpt5_verbosity", None)

        if str(model).startswith("gpt-5"):
            # 明示的に使わないパラメータを除去
            for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                opts.pop(k, None)
            # 既定の gpt-5 オプション（上書き可）
            opts["reasoning_effort"] = gpt5_reasoning_effort or "minimal"
            opts["verbosity"]        = gpt5_verbosity        or "low"

        res = self._cli.chat.completions.create(model=model, messages=messages, **opts)
        return res.choices[0].message.content.strip()


class DumpingAI:
    """OpenAIClient を包んで、chat() 入出力を .ai_dumps/ に保存する薄いラッパ。"""

    def __init__(self, base_ai: OpenAIClient):
        self.base = base_ai
        self._n = itertools.count(1)

    def chat(self, model: str, messages: Sequence[Dict[str, Any]], **kwargs) -> str:
        idx = next(self._n)
        _dump(f"{idx:03d}_prompt_model-{_slug(model)}", {
            "model": model,
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k not in {"api_key", "organization"}},
        }, ".json")

        content = self.base.chat(model, messages, **kwargs)
        _dump(f"{idx:03d}_response_model-{_slug(model)}", content, ".txt")
        return content

# ==========================================================
# ===================== シナリオ生成本体 ====================
@dataclass
class ScenarioConfig:
    theme: str
    minutes: int
    reference: str | List[str] | None = None


class ScenarioBuilder:
    """テーマ + 参考資料 から構造化済みシナリオ JSON を生成する高レベル API"""

    def __init__(self, client: OpenAIClient, *, gpt5_reasoning_effort: str | None = None, gpt5_verbosity: str | None = None):
        self.ai = client
        self.gpt5_reasoning_effort = gpt5_reasoning_effort
        self.gpt5_verbosity = gpt5_verbosity

    # ---- 内部: gpt-5 拡張オプション
    def _gpt5_opts(self) -> Dict[str, Any]:
        return {
            "gpt5_reasoning_effort": self.gpt5_reasoning_effort,
            "gpt5_verbosity": self.gpt5_verbosity,
        }

    # ---------- ステップ 1: 要点リスト ----------
    def _make_topic_outline(self, theme: str, minutes: int, reference: str = "") -> Dict[str, Any]:
        user_p = f"【テーマ】{theme}\n\n【参考資料】\n{reference}"
        content = self.ai.chat(
            MODEL_TOPIC,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_TOPICS},
                {"role": "user", "content": user_p},
            ],
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.2,
        )
        return _safe_json_loads(content, "outline")

    # ---------- ステップ 2: イントロ ----------
    def _build_intro(self, intro_meta: Dict[str, Any], outline_all: List[Dict[str, Any]], theme: str) -> Dict[str, Any]:
        script = self._generate_intro_script(intro_meta, outline_all, theme)
        intro_json = self._structure_intro(script)
        for t in intro_json.get("text", []):
            t["script"] = _wrap_text(t["script"], WRAP_LEN_INTRO)
        return intro_json

    def _generate_intro_script(self, meta: Dict[str, Any], outline: List[Dict[str, Any]], theme: str) -> str:
        outline_str = _outline_str(outline)
        user_p = (
            f"【テーマ】{theme}\n"
            "【台本全体の構成（要約）】\n" + outline_str + "\n\n"
            f"【現在のトピック】{meta['title']}\n"
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

    def _structure_intro(self, script: str) -> Dict[str, Any]:
        content = self.ai.chat(
            MODEL_STRUCT,
            [
                {"role": "system", "content": _SYSTEM_PROMPT_STRUCT_INTRO},
                {"role": "user",   "content": script},
            ],
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.2,
            **self._gpt5_opts(),
        )
        return _safe_json_loads(content, "intro")["introduction"]

    # ---------- ステップ 3: メイントピック（ループ） ----------
    def _build_main_segments(self, topics: List[Dict[str, Any]], outline_all: List[Dict[str, Any]], theme: str) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        counter = itertools.count(1)
        for idx, t in enumerate(topics, 1):
            print(f"▶ メイントピック {idx}: {t['title']}")
            conv_points = self._convert_to_questions(t["points"])  # 質問化
            script = self._generate_dialogue_script(t["title"], conv_points, outline_all, theme)
            segs   = self._structure_dialogue(script)
            for s in segs:  # id 付与 & 折返し
                s["id"] = next(counter)
                if s.get("type") == "dialogue":
                    s["script"]["text"] = _wrap_text(s["script"]["text"], WRAP_LEN_DIALOG)
            segments.extend(segs)
        return segments

    @staticmethod
    def _convert_to_questions(points: List[str]) -> List[str]:
        """ヒューリスティックに陳述→疑問文へ変換。"""
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

    def _generate_dialogue_script(self, title: str, points_q: List[str], outline: List[Dict[str, Any]], theme: str, *, mode: str = "body") -> str:
        outline_str = _outline_str(outline)
        user_p = (
            f"【テーマ】{theme}\n"
            "【台本全体の構成（要約）】\n" + outline_str + "\n\n"
            f"【現在のトピック】{title}\n"
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

    def _structure_dialogue(self, script: str) -> List[Dict[str, Any]]:
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
        return _safe_json_loads(content, "body")["segments"]

    # ---------- ステップ 4: 結論 ----------
    def _build_conclusion(self, concl_meta: Dict[str, Any], outline_all: List[Dict[str, Any]], theme: str) -> Dict[str, Any]:
        script = self._generate_dialogue_script(
            concl_meta["title"],
            self._convert_to_questions(concl_meta["points"]),
            outline_all,
            theme,
            mode="conclusion",
        )
        concl_json = self._structure_conclusion(script)
        counter = itertools.count(1)
        for seg in concl_json.get("text", []):
            seg["id"] = next(counter)
            if seg.get("type") == "dialogue":
                seg["script"]["text"] = _wrap_text(seg["script"]["text"], WRAP_LEN_DIALOG)
        return concl_json

    def _structure_conclusion(self, script: str) -> Dict[str, Any]:
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
        return _safe_json_loads(content, "outro")["conclusion"]

    # ---------- Public API ----------
    def run(self, theme: str, minutes: int, reference: str | List[str] | None = None) -> Dict[str, Any]:
        """テーマ + 参考資料 → 完全構造化 JSON"""
        ref_text = "\n".join(reference) if isinstance(reference, list) else (reference or "")
        outline = self._make_topic_outline(theme, minutes, ref_text)

        intro_meta  = outline["introduction"]
        topics_meta = outline["topics"]
        concl_meta  = outline["conclusion"]
        outline_all = [intro_meta] + topics_meta + [concl_meta]

        intro_json    = self._build_intro(intro_meta, outline_all, theme)
        main_segments = self._build_main_segments(topics_meta, outline_all, theme)
        concl_json    = self._build_conclusion(concl_meta, outline_all, theme)

        return {
            "theme": theme,
            "introduction": intro_json,
            "segments": main_segments,
            "conclusion": concl_json,
        }


# ==========================================================
# ======================== 実行例 ===========================
if __name__ == "__main__":
    # 直実行時のみダンプON（環境変数でも可）
    if DEBUG_DUMP:
        _set_dump_dir(_DUMP_DIR)

    # 例: 既存の format モジュールに渡す想定の処理（必要に応じて調整）
    from rich import print
    import format  # プロジェクト内モジュールを想定

    THEME = "あなたはなぜ“つい後回し”してしまうのか？"
    MINUTES = 2

    with open(r"llm_video_generation/data/1/ref.txt", "r", encoding="utf-8") as f:
        REFERENCE = f.readlines()

    # DumpingAI で包むと自動で .ai_dumps/ に入出力が保存される
    base_ai = OpenAIClient()
    ai = DumpingAI(base_ai) if DEBUG_DUMP else base_ai

    builder = ScenarioBuilder(
        ai,
        gpt5_reasoning_effort="minimal",
        gpt5_verbosity="low",
    )

    result = builder.run(THEME, MINUTES, reference=REFERENCE)

    # 以降は既存のパイプラインに合流
    result = format.insert_sound_info(
        result,
        intro_bgm="llm_video_generation/assets/bgm/Mineral.mp3",
        intro_se="llm_video_generation/assets/se/5.mp3",
        body_bgm="llm_video_generation/assets/bgm/Voice.mp3",
        body_se="llm_video_generation/assets/se/3.mp3",
    )

    out_path = Path("./llm_video_generation/src/s.txt")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
