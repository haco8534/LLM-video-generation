from __future__ import annotations

"""
動画生成パイプライン（dialogue + topic 転換クリップ対応）
------------------------------------------------------
- 入力 : 構造化シナリオ(dict) と dialogue セグメント数分の音声 bytes[]
- 出力 : 1280×720 / 30 fps / AAC 48 kHz stereo / H.264 MP4

シナリオ中に `{"type": "topic", "title": "..."}` が現れたら
その場で 3 秒の場面転換クリップ (背景 2.png + タイトル文字) を挿入する。

依存:
- FFmpeg (ffmpeg-python ラッパ)
- requests, rich
"""

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
DIALOGUE_DUR = 1      # dialogue セグメント長 (秒)
TOPIC_DUR    = 3      # topic   セグメント長 (秒)

BGM_PATH    = "llm_video_generation/assets/bgm/Voice.mp3"
BGM_VOLUME  = 0.1
SE_TOPIC_PATH = "llm_video_generation/assets/SE/3.mp3"
SE_VOLUME     = 0.6

FONT = "C:/Windows/Fonts/meiryo.ttc"

# concat で -c copy するため、すべての音声ストリーム仕様を統一
SAMPLE_RATE   = 48_000    # Hz
CHANNEL_LAYOUT = "stereo"

# 背景画像パス
BG_DIALOGUE = "llm_video_generation/assets/background/3.png"
BG_TOPIC    = "llm_video_generation/assets/background/2.png"

# キャラクター画像ベースパス
CHAR_ROOT = "llm_video_generation/assets/character"

# ──────────────────────────────
# 場面転換スタイルプリセット
# ──────────────────────────────

TOPIC_STYLES: dict[str, dict] = {
    "1": {
        "bg": "llm_video_generation/assets/background/2.png",
        "rect_outer": "#7281E3@0.80",
        "rect_inner": "#9DA7EB",
        "text_color": "orange",
        "char": f"{CHAR_ROOT}/ずんだもん/think.png",
        "char_scale": 700,
        "char_pos": (780, 50),
    },
    "2": {
        "bg": "llm_video_generation/assets/background/4.png",
        "rect_outer": "#8D006E@0.80",
        "rect_inner": "#EC9CDB",
        "text_color": "#63C8FF",
        "char": f"{CHAR_ROOT}/四国めたん/whisper.png",
        "char_scale": 650,
        "char_pos": (800, 100),
    },
}

# ──────────────────────────────
# 低レイヤ helper
# ──────────────────────────────

def _mk_background(path: str, dur: int):
    """静止画を指定秒数ループさせる背景 video stream"""
    return (
        ffmpeg.input(path, loop=1, t=dur, framerate=FPS)
        .filter("scale", W, H)
    )


def _overlay_characters(base, faces: Dict[str, str]):
    """ずんだもん & 四国めたん を所定位置に合成"""
    zunda = (
        ffmpeg.input(f"{CHAR_ROOT}/ずんだもん/{faces['2']}.png")
        .filter("scale", 400, -1)
        .filter("hflip")
    )
    base = ffmpeg.overlay(base, zunda, x=-50, y=250)

    metan = (
        ffmpeg.input(f"{CHAR_ROOT}/四国めたん/{faces['1']}.png")
        .filter("scale", 400, -1)
    )
    return ffmpeg.overlay(base, metan, x=950, y=250)


def _subtitle_box(base):
    return base.drawbox(
        x="(iw-w)/2",
        y=str(H - 180),
        width=W,
        height=200,
        color="black@0.6",
        thickness="fill",
    )


def _image_asset_box(base):
    return base.drawbox(
        x="(iw-w)/2",
        y=str(H - 700),
        width=W - 400,
        height=480,
        color="white@0.3",
        thickness="fill",
    )


def _overlay_image_asset(base, url: str | Path):
    """外部画像を 880×480 に収め中央配置"""
    img = (
        ffmpeg.input(str(url), loop=1, t=DIALOGUE_DUR, framerate=FPS)
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
    )


def _topic_text_overlay(base, topic: str):
    """dialogue セグメント右上に現在のトピックを表示"""
    if not topic:
        return base
    return base.drawtext(
        text=topic,
        fontfile=FONT,
        fontsize=40,
        fontcolor="white",
        x="w-text_w-40",
        y="40",
        borderw=1,
        bordercolor="white",
        shadowcolor="black",
        shadowx=2,
        shadowy=2,
    )

# --------------------------------------
# FFmpeg graph builders
# --------------------------------------

def _build_dialogue_graph(
    wav_path: Path,
    text: str,
    speaker: str,
    faces: Dict[str, str],
    topic: str = "",
    img_url: str | Path | None = None,
):
    """音声付き dialogue セグメント"""
    bg = _mk_background(BG_DIALOGUE, DIALOGUE_DUR)
    bg = _image_asset_box(bg)
    if img_url:
        bg = _overlay_image_asset(bg, img_url)
    bg = _overlay_characters(bg, faces)
    bg = _subtitle_box(bg)
    bg = _subtitle_text(bg, text, speaker)
    bg = _topic_text_overlay(bg, topic)

    audio = (
        ffmpeg.input(str(wav_path))
        .filter("aresample", SAMPLE_RATE)
        .filter("aformat", channel_layouts=CHANNEL_LAYOUT)
    )
    return bg, audio

def _build_topic_graph(title: str, design: str = "1"):
    s = TOPIC_STYLES.get(design, TOPIC_STYLES["1"]) 

    rect_w, rect_h = 950, 570
    bg = _mk_background(s["bg"], TOPIC_DUR)

    # 文字サイズ計算
    max_fs, min_fs, step, base_len = 75, 55, 10, 11
    dec = max(0, ((len(title) - base_len) // 1) * step)
    fontsize = max(min_fs, max_fs - dec)

    # ----- 矩形 3 枚 -----
    bg = bg.drawbox(
        x=f"(iw-{rect_w})/2 - 80",  y=f"(ih-{rect_h})/2 + 20",
        width=rect_w, height=rect_h, color=s["rect_inner"], thickness="fill",
    ).drawbox(
        x=f"(iw-{rect_w})/2 - 100", y=f"(ih-{rect_h})/2",
        width=rect_w, height=rect_h, color=s["rect_outer"], thickness="fill",
    ).drawbox(
        x=f"(iw-{rect_w})/2 - 80",  y=f"(ih-{rect_h})/2 + 340",
        width=rect_w - 35, height=3, color="white", thickness="fill",
    )

    # ----- タイトル -----
    bg = bg.drawtext(
        text=title, fontfile="C:/Windows/Fonts/meiryob.ttc",
        fontsize=fontsize, fontcolor=s["text_color"],
        x="(w-text_w)/2 - 100", y="(h-text_h)/2",
        borderw=4, bordercolor="white",
        shadowcolor="black", shadowx=2, shadowy=2,
    )

    # ----- キャラクター -----
    char = (
        ffmpeg.input(s["char"], loop=1, t=TOPIC_DUR, framerate=FPS)
        .filter("scale", s["char_scale"], -1)
    )
    cx, cy = s["char_pos"]
    bg = ffmpeg.overlay(bg, char, x=cx, y=cy)

    # ----- 効果音 -----
    audio = (
        ffmpeg.input(SE_TOPIC_PATH)
        .filter("apad", pad_dur=TOPIC_DUR)
        .filter("atrim", duration=TOPIC_DUR)
        .filter("aresample", SAMPLE_RATE)
        .filter("aformat", channel_layouts=CHANNEL_LAYOUT)
        .filter("volume", SE_VOLUME)
    )

    return bg, audio


# ──────────────────────────────
# 画像キャッシュ helper
# ──────────────────────────────

def _cache_images(urls: Sequence[str], work_dir: Path) -> List[Path]:
    """URL 画像を work_dir に保存し Path のリストを返す"""
    cached: List[Path] = []
    for idx, url in enumerate(urls, 1):
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
# VideoAssembler
# ──────────────────────────────

class VideoAssembler:
    """Scenario + 音声 bytes[] からフル動画を組み立てる"""

    def __init__(self, temp_dir: str | Path | None = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    def _write_wav(self, data: bytes, idx: int) -> Path:
        path = self.temp_dir / f"voice_{idx:03}.wav"
        path.write_bytes(data)
        return path
    
    # --------------------------------------------------------
    def _add_bgm(self, src: Path, dst: str | Path):
        """
        src の映像ストリームはコピーし、
        src の音声 + ループさせた BGM を amix で重ねて dst に保存
        """
        v_in  = ffmpeg.input(str(src))
        bgm_delay_ms = 2900

        bgm_in = (
            ffmpeg
            .input(BGM_PATH, stream_loop=-1)
            .filter("aresample", SAMPLE_RATE)
            .filter("aformat", channel_layouts=CHANNEL_LAYOUT)
            .filter("volume", BGM_VOLUME)
            .filter("adelay", f"{bgm_delay_ms}|{bgm_delay_ms}")  # L/R 両チャンネルに適用
        )

        mixed = ffmpeg.filter(
            [v_in.audio, bgm_in],
            "amix",
            inputs=2,
            duration="first",           # 映像の長さに合わせて BGM を切る
            dropout_transition=3,
        )

        (
            ffmpeg
            .output(
                v_in.video, mixed, str(dst),
                vcodec="copy",          # 映像はコピー
                acodec="aac",
                movflags="faststart",
                loglevel="error",
            )
            .overwrite_output()
            .run()
        )


    # --------------------------------------------------------
    def build_segments(
        self,
        scenario: Dict,
        audio_bytes: Sequence[bytes],
        image_paths: Sequence[Path],
    ) -> List[Path]:
        """シナリオを逐次走査し、各 type に応じて MP4 セグメントを書き出す"""
        paths: List[Path] = []
        current_face = {"1": "normal1", "2": "normal1"}
        current_topic = ""
        audio_idx = 0  # dialogue ごとに消費
        img_idx = 0    # dialogue ごとに消費

        for seq_idx, seg in enumerate(scenario["segments"], 1):
            # topic --------------------------------------------------
            if seg["type"] == "topic":
                current_topic = seg.get("title", "")
                design_no = seg.get("design", "1") 
                v, a = _build_topic_graph(current_topic, design_no)
            # dialogue ----------------------------------------------
            elif seg["type"] == "dialogue":
                sc = seg["script"]
                text, speaker, face = sc["text"], sc["speaker"], sc["face"]
                if paths:
                    current_face[speaker] = face

                wav_path = self._write_wav(audio_bytes[audio_idx], audio_idx) if audio_bytes else Path(f"./assets/voice/{audio_idx:03}.wav")
                img_path = image_paths[img_idx] if img_idx < len(image_paths) else None

                v, a = _build_dialogue_graph(
                    wav_path, text, speaker, current_face.copy(), current_topic, img_path
                )

                audio_idx += 1
                img_idx += 1
            else:
                continue  # 不明タイプはスキップ

            out = self.temp_dir / f"seg_{seq_idx:03}.mp4"
            (
                ffmpeg.output(
                    v, a, str(out),
                    vcodec="libx264", acodec="aac",
                    pix_fmt="yuv420p", movflags="faststart",
                    loglevel="error",
                ).overwrite_output().run()
            )
            paths.append(out)
        return paths

    # --------------------------------------------------------
    def concat(self, seg_files: List[Path], output: str | Path):
        """concat demuxer (-c copy) で結合"""
        list_file = self.temp_dir / "concat_list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for p in seg_files:
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
        """シナリオ + 音声 + 画像 URL から output_path に MP4 を生成"""
        local_images = _cache_images(image_urls, self.temp_dir)

        # ② セグメント生成 → ③ 連結
        segs = self.build_segments(scenario, audio_bytes, local_images)

        concat_path = self.temp_dir / "concat.mp4"
        self.concat(segs, concat_path)

        # ④ BGM を重ねて最終出力
        self._add_bgm(concat_path, output_path)
        return Path(output_path)