"""
video.py
────────────────────────────────────────────────────────
・構造化台本(dict) と音声(bytes)リストを入力
・キャラクター画像／字幕付き 720p MP4 をセグメントごとに生成
・最後に連結して 1 本の動画に出力
"""

from __future__ import annotations

import mimetypes
import tempfile
from pathlib import Path
from typing import List, Dict, Sequence

import requests
import ffmpeg
from rich import print

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
    return (
        ffmpeg.input(
            "llm_video_generation/assets/background/1.png",
            loop=1,
            t=SEG_DUR,
            framerate=FPS,
        )
        .filter("scale", W, H)
    )


def _overlay_characters(base, faces: Dict[str, str]):
    zunda = (
        ffmpeg.input(
            f"llm_video_generation/assets/character/ずんだもん/{faces['2']}.png"
        )
        .filter("scale", 400, -1)
        .filter("hflip")
    )
    base = ffmpeg.overlay(base, zunda, x=-50, y=250)

    metan = (
        ffmpeg.input(f"llm_video_generation/assets/character/四国めたん/{faces['1']}.png")
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


def _image_asset_box(base):
    return base.drawbox(
        x="(iw-w)/2",
        y=f"{H-700}",
        width=W - 400,
        height=480,
        color="white@0.3",
        thickness="fill",
    )


def _overlay_image_asset(base, url: str | Path):
    """
    アスペクト比を保ったまま 880×480 に収まるようにリサイズ→中央配置
    """
    img = (
        ffmpeg.input(str(url), loop=1, t=SEG_DUR, framerate=FPS)
        .filter("scale", "if(gt(a,720/400),720,-1)", "if(gt(a,720/400),-1,400)")
        .filter("pad", 880, 480, "(ow-iw)/2", "(oh-ih)/2", "black@0.0")
    )
    return ffmpeg.overlay(base, img, x=200, y=0)


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


def _segment_ffmpeg_graph(
    wav_path: Path,
    text: str,
    speaker: str,
    faces: Dict[str, str],
    img_url: str | Path | None = None,
):
    bg = _mk_background()
    bg = _image_asset_box(bg)
    if img_url:
        bg = _overlay_image_asset(bg, img_url)
    bg = _overlay_characters(bg, faces)
    bg = _subtitle_box(bg)
    bg = _subtitle_text(bg, text, speaker)
    audio = ffmpeg.input(str(wav_path))
    return bg, audio


# ──────────────────────────────
# 画像キャッシュ helper
# ──────────────────────────────
def _cache_images(urls: Sequence[str], work_dir: Path) -> List[Path]:
    """
    各 URL を work_dir に asset_001.jpg のような名前で保存し、
    ローカル Path のリストを返す。既にファイルがあれば再ダウンロードしない。
    """
    cached: List[Path] = []
    for idx, url in enumerate(urls, 1):
        # 拡張子を推定（URL に無い場合は Content-Type から）
        ext = Path(url).suffix
        if not ext:
            ct = requests.head(url, timeout=5).headers.get("content-type", "")
            ext = mimetypes.guess_extension(ct) or ".jpg"
        path = work_dir / f"asset_{idx:03}{ext}"
        if not path.exists():
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            path.write_bytes(r.content)
        cached.append(path)
    return cached


# ──────────────────────────────
# メインクラス
# ──────────────────────────────
class VideoAssembler:
    """
    音声ファイルを書き出さなくても bytes → wav の一時化で処理できる。

    Typical flow
    ------------
        scenario = ScenarioService(...).run(...)
        audio_bytes = TTSPipeline(...).run(scenario)
        assembler = VideoAssembler()
        assembler.build_full_video(scenario, audio_bytes, image_urls, "output.mp4")
    """

    def __init__(self, temp_dir: str | Path | None = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    def build_segments(
        self,
        scenario: Dict,
        audio_bytes: Sequence[bytes],
        image_paths: Sequence[Path],
    ) -> List[Path]:
        """
        各 dialogue セグメントごとに MP4 を作成し、一時ディレクトリに保存。
        """
        segments = _extract_dialogue_segments(scenario)
        if audio_bytes is not None and len(audio_bytes) != len(segments):
            raise ValueError("audio_bytes の個数が dialogue セグメント数と一致しません")

        current_face = {"1": "normal1", "2": "normal1"}
        paths: List[Path] = []

        for idx, seg in enumerate(segments, 1):
            text, speaker, face = seg["text"], seg["speaker"], seg["face"]
            if idx != 1:
                current_face[speaker] = face  # 表情更新

            # --- 音声ファイルを用意 ----------------------------------
            if audio_bytes is not None:
                wav_path = self.temp_dir / f"{idx:03}.wav"
                wav_path.write_bytes(audio_bytes[idx - 1])
            else:
                wav_path = Path(f"./assets/voice/{idx:03}.wav")

            out = self.temp_dir / f"segment_{idx:03}.mp4"
            video_stream, audio_stream = _segment_ffmpeg_graph(
                wav_path,
                text,
                speaker,
                current_face,
                image_paths[idx - 1] if image_paths else None,
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
    def build_full_video(
        self,
        scenario: Dict,
        audio_bytes: Sequence[bytes],
        image_urls: Sequence[str],
        output_path: str | Path = "output.mp4",
    ) -> Path:
        # ① ネット上の画像を一度だけローカルに保存
        local_images = _cache_images(image_urls, self.temp_dir)

        # ② セグメント動画生成 → ③ 連結
        segs = self.build_segments(scenario, audio_bytes, local_images)
        self.concat(segs, output_path)
        return Path(output_path)


# ──────────────────────────────
# 内部 util
# ──────────────────────────────
def _extract_dialogue_segments(scenario: Dict):
    """
    {text, speaker, face} を順番に返す
    """
    return [
        {
            "text": seg["script"]["text"],
            "speaker": seg["script"]["speaker"],
            "face": seg["script"]["face"],
        }
        for seg in scenario["segments"]
        if seg["type"] == "dialogue"
    ]
