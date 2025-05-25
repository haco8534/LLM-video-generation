"""
tts_pipeline.py
────────────────────────────────────────────────────────
構造化台本(json) → ひらがな読み生成(OpenAI) → VoiceVox で音声合成
・話者名ごとにキャラクター／スタイルを切替
・speedScale など任意パラメータを上書き
・multiprocessing で高速合成

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
            - 例：「AI」→「えーあい」「Tokyo」→「とうきょう」「Python」→「ぱいそん」 「12:30」→ 「じゅうにじさんじゅっぷん」 「A/Bテスト」→「えーびーてすと」
            3. 誤読リスクが低い漢字はそのまま維持する。  
            4  ひらがな・カタカナ部分は読み間違いが起こることはないため一切変更せずそのまま維持する。
            - 例: 「プログラミング」 → 「プログラミング」
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

    # main --------------------------------------------------------------

    def generate(self, texts: Sequence[str], max_retry: int = 3) -> List[str]:
        payload = json.dumps(list(texts), ensure_ascii=False)
        user_msg = f"要素数 {len(texts)} のリストです。同じ数で返してください。\n{payload}"

        for _ in range(max_retry):
            reply = self._chat(user_msg)
            try:
                data = json.loads(reply)
                if isinstance(data, list) and len(data) == len(texts):
                    print(data)
                    return data
            except json.JSONDecodeError:
                pass  # retry
        raise RuntimeError("読み生成に失敗しました")

    # internal ----------------------------------------------------------

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
    """
    char_style 形式:
        {
            "アオイ": "四国めたん/ノーマル",
            "カイ":   "ずんだもん/ノーマル"
        }
    params:
        {"speedScale": 1.2, "pitchScale": 0.0, ...}
        -> audio_query JSON に上書き
    """

    def __init__(
        self,
        host: str = "http://localhost:50021",
        char_style: Dict[str, str] | None = None,
        default_style: str = "ノーマル",
        params: Dict[str, float] | None = None,
    ):
        self.host = host.rstrip("/")
        self.session = requests.Session()

        # スタイル一覧を取得
        self.style_id_map: Dict[str, int] = {}
        for c in self.session.get(f"{self.host}/speakers").json():
            for st in c["styles"]:
                key = f"{c['name']}/{st['name']}"
                self.style_id_map[key] = st["id"]

        # name -> style_id 変換
        self.speaker_map: Dict[str, int] = {}
        if char_style:
            for name, ns in char_style.items():
                if ns in self.style_id_map:
                    self.speaker_map[name] = self.style_id_map[ns]
                else:
                    # 補完：キャラ名だけの場合は default_style を当てる
                    alt = f"{ns}/{default_style}"
                    self.speaker_map[name] = self.style_id_map.get(alt, 1)

        self.params = params or {}

    # ------------------------------------------------------------------

    def synthesize(self, text: str, speaker_name: str, out_path: Path) -> Path:
        speaker_id = self.speaker_map.get(speaker_name, 1)

        # audio_query
        query = self.session.post(
            f"{self.host}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=10,
        ).json()

        # パラメータ上書き
        query.update(self.params)

        # synthesis
        wav = self.session.post(
            f"{self.host}/synthesis",
            params={"speaker": speaker_id},
            data=json.dumps(query),
            timeout=30,
        ).content

        out_path.write_bytes(wav)
        return out_path


# --------------------------------------------------------------------------- #
# 4. Multiprocess パイプライン
# --------------------------------------------------------------------------- #

#タプルで引数を受け取ることで並列処理に引数をまとめて渡す
def _worker(
    args: Tuple[str, str, Path, Dict[str, str], Dict[str, float]]
) -> str:
    text, speaker, path, char_style, params = args
    tts = VoiceVoxTTS(char_style=char_style, params=params)
    tts.synthesize(text, speaker, path)
    return str(path)


class TTSPipeline:
    """
    構造化台本 dict → wav ファイル群
    """

    def __init__(
        self,
        out_dir: str = "./voice",
        char_style: Dict[str, str] | None = None,
        tts_params: Dict[str, float] | None = None,
        processes: int | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.char_style = char_style or {}
        self.tts_params = tts_params or {}
        self.processes = processes or cpu_count()

    # ------------------------------------------------------------------

    def run(self, scenario: dict) -> List[str]:
        # 1) セリフ取得
        dialogs = extract_dialogues_with_speaker(scenario)  # (text, speaker)
        texts = [d[0] for d in dialogs]

        # 2) 読み生成
        readings = ReadGenerator().generate(texts)

        # 3) タスク作成
        tasks: List[Tuple[str, str, Path, Dict, Dict]] = []
        for idx, (reading, (_, speaker)) in enumerate(zip(readings, dialogs), 1):
            wav_path = self.out_dir / f"{idx:03d}.wav"
            tasks.append(
                (reading, speaker, wav_path, self.char_style, self.tts_params)
            )

        # 4) 並列合成
        with Pool(self.processes) as pool:
            results = pool.map(_worker, tasks)

        return results

if __name__ == "__main__":
    # ① 入力シナリオ（構造化 JSON）をロード
    scenario_path = Path("./modules/a.txt")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    # ② キャラクター → VoiceVox スタイル対応
    char_style = {
        "1": "四国めたん/ノーマル",
        "2": "ずんだもん/ノーマル",
    }

    # ③ 共通 TTS パラメータ
    tts_params = {"speedScale": 1.2, "intonationScale": 1.1}

    # ④ パイプライン実行
    pipeline = TTSPipeline(
        out_dir="./voice",
        char_style=char_style,
        tts_params=tts_params,
        processes=None, #<-Noneだとcpuコア数になる
    )
    wav_files = pipeline.run(scenario)