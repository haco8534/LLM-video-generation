"""
intro_tts.py  –  動画イントロダクション用 TTS パイプライン
─────────────────────────────────────────────
JSON シナリオ → (タイトル + 本文) → 読み仮名生成(OpenAI) → VoiceVox 合成
・話者は 1 人のみ（key="1"）
・キャラクター／スタイル、音声パラメータ(speedScale 等) を指定可
・multiprocessing で高速化（音声bytes リストを返す）

依存:
    pip install openai python-dotenv requests
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from multiprocessing import Pool, cpu_count

import requests
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 1. ひらがな読み生成 (LLM)
# --------------------------------------------------------------------------- #

class ReadGenerator:
    """OpenAI で漢字かな混在文をひらがなに変換"""

    _PROMPT = r'''
        以下の要件に厳密に従い、与えられた日本語セリフの配列を同じ順序・要素数の配列に変換してください。

        1. 出力は JSON 配列（[…]）のみとし、説明文や余計な文字は一切含めない。
        2. 機械的な TTS で誤読が起こりそうな箇所（英字、略語、固有名詞など）はひらがなに変換する。
        3. 誤読リスクが低い漢字はそのまま維持する。
        4. ひらがな・カタカナ部分はそのまま維持する。
        5. 文末の「。」は入力の有無にかかわらずすべて削除する。
        6. 文字数やトークン数を制限せず、正確な読み仮名を最優先する。
    '''

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _chat(self, user_content: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=1.0,
            top_p=0.95,
            messages=[
                {"role": "system", "content": self._PROMPT.strip()},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()

    def generate(self, texts: Sequence[str], max_retry: int = 3) -> List[str]:
        payload = json.dumps(list(texts), ensure_ascii=False)
        user_msg = f"要素数 {len(texts)} のリストです。同じ数で返してください。\n{payload}"

        for _ in range(max_retry):
            reply = self._chat(user_msg)
            try:
                data = json.loads(reply)
                if isinstance(data, list) and len(data) == len(texts):
                    return data
            except json.JSONDecodeError:
                pass
        raise RuntimeError("読み生成に失敗しました")

# --------------------------------------------------------------------------- #
# 2. VoiceVox 合成
# --------------------------------------------------------------------------- #

class VoiceVoxTTS:
    """VoiceVox API ラッパ – 単文を wav bytes へ"""

    def __init__(
        self,
        host: str = "http://localhost:50021",
        char_style: Dict[str, str] | None = None,
        default_style: str = "ノーマル",
        params: Dict[str, float] | None = None,
    ):
        self.host = host.rstrip("/")
        self.session = requests.Session()

        # スタイル名→ID 変換表
        self.style_id_map: Dict[str, int] = {}
        for c in self.session.get(f"{self.host}/speakers").json():
            for st in c["styles"]:
                key = f"{c['name']}/{st['name']}"
                self.style_id_map[key] = st["id"]

        # ユーザ入力 (キャラクター/スタイル) → speaker_id
        self.speaker_map: Dict[str, int] = {}
        if char_style:
            for name, ns in char_style.items():
                if ns in self.style_id_map:
                    self.speaker_map[name] = self.style_id_map[ns]
                else:
                    alt = f"{ns}/{default_style}"
                    self.speaker_map[name] = self.style_id_map.get(alt, 1)

        self.params = params or {}

    def synthesize(self, text: str, speaker_name: str) -> bytes:
        speaker_id = self.speaker_map.get(speaker_name, 1)

        query = self.session.post(
            f"{self.host}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=30,
        ).json()

        query.update(self.params)

        wav = self.session.post(
            f"{self.host}/synthesis",
            params={"speaker": speaker_id},
            data=json.dumps(query),
            timeout=30,
        ).content

        return wav

# --------------------------------------------------------------------------- #
# 3. multiprocessing ユーティリティ
# --------------------------------------------------------------------------- #

def _worker(args: Tuple[str, str, Dict[str, str], Dict[str, float]]) -> bytes:
    text, speaker, char_style, params = args
    tts = VoiceVoxTTS(char_style=char_style, params=params)
    return tts.synthesize(text, speaker)

# --------------------------------------------------------------------------- #
# 4. Introduction TTS パイプライン
# --------------------------------------------------------------------------- #

class IntroductionTTSPipeline:
    """シナリオJSON → イントロ用 音声bytes のリスト"""

    def __init__(
        self,
        char_style: Dict[str, str] | None = None,
        tts_params: Dict[str, float] | None = None,
        processes: int | None = None,
    ):
        self.char_style = char_style or {}
        self.tts_params = tts_params or {}
        self.processes = processes or cpu_count()

    def _extract_intro_texts(self, scenario: dict) -> List[str]:
        intro = scenario.get("introduction", {})

        # タイトル
        title = intro.get("title", "")

        # 本文 ― script フィールドだけを取り出す
        lines = [
            item.get("script", "")
            for item in intro.get("text", [])
            if isinstance(item, dict)
        ]

        return [title, *lines] if title else lines

    def run(self, scenario: dict, speaker: str = "1") -> List[bytes]:
        texts = self._extract_intro_texts(scenario)
        readings = ReadGenerator().generate(texts)

        tasks: List[Tuple[str, str, Dict, Dict]] = [
            (reading, speaker, self.char_style, self.tts_params) for reading in readings
        ]

        with Pool(self.processes) as pool:
            audio_bytes_list = pool.map(_worker, tasks)

        return audio_bytes_list

# --------------------------------------------------------------------------- #
# 5. main (動作テスト)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    scenario_path = Path("./llm_video_generation/src/s.txt")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    # ---- 設定 ----
    char_style = {
        "1": "もち子さん/ノーマル",
    }
    tts_params = {"speedScale": 1.1, "intonationScale": 1.1}

    pipeline = IntroductionTTSPipeline(
        char_style=char_style,
        tts_params=tts_params,
        processes=3,
    )

    audio_bytes_list = pipeline.run(scenario, speaker="1")

    # ===== 出力保存 =====
    out_dir = Path("intro_audio_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, wav_bytes in enumerate(audio_bytes_list):
        (out_dir / f"intro_{i:02}.wav").write_bytes(wav_bytes)