"""
tts_pipeline.py
────────────────────────────────────────────────────────
構造化台本(json) → ひらがな読み生成(OpenAI) → VoiceVox で音声合成
・話者名ごとにキャラクター／スタイルを切替
・speedScale など任意パラメータを上書き
・multiprocessing で高速合成（出力は音声bytes）

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
# 1. 台本ユーティリティ
# --------------------------------------------------------------------------- #

def extract_dialogues_with_speaker(scenario: dict) -> List[Tuple[str, str]]:
    """(text, speaker) を id 昇順で返す"""
    return [
        (seg["script"]["text"], seg["script"]["speaker"])
        for seg in sorted(scenario["segments"], key=lambda s: s["id"])
        if seg["type"] == "dialogue"
    ]

# --------------------------------------------------------------------------- #
# 2. ひらがな読み生成 (LLM)
# --------------------------------------------------------------------------- #

class ReadGenerator:
    """OpenAI で漢字かな混在文をひらがなに変換"""

    _PROMPT =  '''
        以下の要件に厳密に従い、与えられた日本語セリフの配列を同じ順序・要素数の配列に変換してください。

        1. 出力は JSON 配列（[…]）のみとし、説明文や余計な文字は一切含めない。  
        2. 機械的な TTS で誤読が起こりそうな箇所（英字、略語、固有名詞など）はひらがなに変換する。  
        3. 誤読リスクが低い漢字はそのまま維持する。  
        4. ひらがな・カタカナ部分はそのまま維持する。  
        5. 文末の「。」は入力の有無にかかわらずすべて削除する。  
        6. 文字数やトークン数を制限せず、正確な読み仮名を最優先する。  

        ――――――  
        【入力例】  
        [
        "こんにちは、今日のテーマはAIです。",
        "次に、最適化の話をしましょう。"
        ]

        【期待する出力】
        [
        "こんにちは、今日のテーマはえーあいです",
        "次に、最適化の話をしましょう"
        ]
    '''

    def __init__(self, model: str = "gpt-4.1"):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

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

# --------------------------------------------------------------------------- #
# 3. VoiceVox 合成
# --------------------------------------------------------------------------- #

class VoiceVoxTTS:
    def __init__(
        self,
        host: str = "http://localhost:50021",
        char_style: Dict[str, str] | None = None,
        default_style: str = "ノーマル",
        params: Dict[str, float] | None = None,
    ):
        self.host = host.rstrip("/")
        self.session = requests.Session()

        self.style_id_map: Dict[str, int] = {}
        for c in self.session.get(f"{self.host}/speakers").json():
            for st in c["styles"]:
                key = f"{c['name']}/{st['name']}"
                self.style_id_map[key] = st["id"]

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
# 4. Multiprocess パイプライン
# --------------------------------------------------------------------------- #

def _worker(
    args: Tuple[str, str, Dict[str, str], Dict[str, float]]
) -> bytes:
    text, speaker, char_style, params = args
    tts = VoiceVoxTTS(char_style=char_style, params=params)
    return tts.synthesize(text, speaker)

class TTSPipeline:
    """
    構造化台本 dict → 音声bytesのリスト
    """

    def __init__(
        self,
        char_style: Dict[str, str] | None = None,
        tts_params: Dict[str, float] | None = None,
        processes: int | None = None,
    ):
        self.char_style = char_style or {}
        self.tts_params = tts_params or {}
        self.processes = processes or cpu_count()

    def run(self, scenario: dict) -> List[bytes]:
        dialogs = extract_dialogues_with_speaker(scenario)
        texts = [d[0] for d in dialogs]
        readings = ReadGenerator().generate(texts)

        tasks: List[Tuple[str, str, Dict, Dict]] = [
            (reading, speaker, self.char_style, self.tts_params)
            for reading, (_, speaker) in zip(readings, dialogs)
        ]

        with Pool(self.processes) as pool:
            audio_bytes_list = pool.map(_worker, tasks)

        return audio_bytes_list

# --------------------------------------------------------------------------- #
# 5. main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    scenario_path = Path("./src/llm_video_generation/a.txt")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    char_style = {
        "1": "四国めたん/ノーマル",
        "2": "ずんだもん/ノーマル",
    }

    tts_params = {"speedScale": 1.1, "intonationScale": 1.1}

    pipeline = TTSPipeline(
        char_style=char_style,
        tts_params=tts_params,
        processes=3,
    )

    wav_bytes_list = pipeline.run(scenario)

    print(f"{len(wav_bytes_list)}個の音声データを取得しました")
