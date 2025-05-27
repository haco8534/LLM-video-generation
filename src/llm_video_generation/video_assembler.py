"""
video_assembler.py
────────────────────────────────────────────────────────
・構造化台本(segments) と合成済み音声ファイルを入力
・キャラクター画像／字幕付き 720p MP4 をセグメントごとに生成
・最後にすべて連結して 1 本の動画に出力
"""

import json
from pathlib import Path
from typing import List, Dict

import ffmpeg

# ──────────────────────────────
# グローバル設定
# ──────────────────────────────
W, H = 1280, 720
FPS = 30
SEG_DUR = 1          # 秒
FONT = "C:/Windows/Fonts/meiryo.ttc"


# ──────────────────────────────
# 低レイヤ helper
# ──────────────────────────────
def _mk_background():
    return ffmpeg.input(f"color=c=white:s={W}x{H}:d={SEG_DUR}:r={FPS}", f="lavfi")


def _overlay_characters(base, faces: Dict[str, str]):
    # faces = {"1":"normal1", "2":"normal2"}
    zunda = (
        ffmpeg.input(f"assets/character/ずんだもん/{faces['2']}.png")
        .filter("scale", 400, -1)
        .filter("hflip")
    )
    base = ffmpeg.overlay(base, zunda, x=-50, y=250)

    metan = (
        ffmpeg.input(f"assets/character/四国めたん/{faces['1']}.png")
        .filter("scale", 400, -1)
    )
    return ffmpeg.overlay(base, metan, x=950, y=250)


def _subtitle_box(base):
    return base.drawbox(
        x="(iw-w)/2",
        y=f"{H-180}",
        width=W,
        height=200,
        color="black@0.6",
        thickness="fill",
    )


def _subtitle_text(base, text: str, speaker: str):
    border = "#E7609E" if speaker == "1" else "#6CBB5A"
    return base.drawtext(
        text=text,
        fontfile=FONT,
        fontsize=35,
        fontcolor="white",
        x="(w-text_w)/2",
        y="h-140",
        borderw=2,
        bordercolor=border,
        shadowcolor="black",
        shadowx=2,
        shadowy=2,
        line_spacing=10,
        enable="between(t,0,5)",
    )


def _segment_ffmpeg_graph(voice_path: Path, text: str, speaker: str, faces: Dict[str, str]):
    bg = _mk_background()
    bg = _overlay_characters(bg, faces)
    bg = _subtitle_box(bg)
    bg = _subtitle_text(bg, text, speaker)
    audio = ffmpeg.input(str(voice_path))
    return bg, audio


# ──────────────────────────────
# メインクラス
# ──────────────────────────────
class VideoAssembler:
    """
    1. `build_segments()` で temp/*.mp4 を作成
    2. `concat()`       で連結
    """

    def __init__(
        self,
        temp_dir: str | Path = "./temp",
        assets_voice: str | Path = "./assets/voice",
    ):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.assets_voice = Path(assets_voice)

    # --------------------------------------------------------

    def build_segments(self, scenario: dict) -> List[Path]:
        """セグメント mp4 を生成してファイルパスを返す"""
        segments = _extract_dialogue_segments(scenario)
        current_face = {"1": "normal1", "2": "normal1"}

        paths: List[Path] = []
        for idx, seg in enumerate(segments, 1):
            text, speaker, face = seg["text"], seg["speaker"], seg["face"]
            if idx != 1:
                current_face[speaker] = face  # 表情更新

            voice = self.assets_voice / f"{idx:03}.wav"
            out = self.temp_dir / f"segment_{idx:03}.mp4"

            video_stream, audio_stream = _segment_ffmpeg_graph(
                voice, text, speaker, current_face
            )

            (
                ffmpeg.output(
                    video_stream,
                    audio_stream,
                    str(out),
                    vcodec="libx264",
                    acodec="aac",
                    pix_fmt="yuv420p",
                    movflags="faststart",
                    loglevel="error",
                )
                .overwrite_output()
                .run()
            )
            paths.append(out)
        return paths

    # --------------------------------------------------------

    def concat(self, segment_files: List[Path], output: str | Path):
        list_file = self.temp_dir / "concat_list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for p in segment_files:
                f.write(f"file '{p.resolve()}'\n")

        (
            ffmpeg.input(str(list_file), format="concat", safe=0)
            .output(str(output), c="copy", loglevel="error")
            .overwrite_output()
            .run()
        )

    # --------------------------------------------------------

    def build_full_video(self, scenario: dict, output_path: str | Path = "output.mp4"):
        segs = self.build_segments(scenario)
        self.concat(segs, output_path)
        return output_path


# ──────────────────────────────
# 内部 util
# ──────────────────────────────
def _extract_dialogue_segments(scenario: dict):
    """
    {text, speaker, face} を順番に返す
    """
    out = []
    for seg in scenario["segments"]:
        if seg["type"] == "dialogue":
            out.append(
                {
                    "text": seg["script"]["text"],
                    "speaker": seg["script"]["speaker"],
                    "face": seg["script"]["face"],
                }
            )
    return out


# ──────────────────────────────
# サンプル実行
# ──────────────────────────────
if __name__ == "__main__":
    # 台本例を読み込み
    scenario = json.loads(Path("./modules/a.txt").read_text(encoding="utf-8"))

    assembler = VideoAssembler(temp_dir="./temp", assets_voice="./assets/voice")
    final_mp4 = assembler.build_full_video(scenario, output_path="output.mp4")
    print("✅ 完成:", final_mp4)
