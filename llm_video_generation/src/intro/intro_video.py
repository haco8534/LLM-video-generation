"""
intro_video_builder.py  – タイトル中央表示版
────────────────────────────────────────────────────────────
- audio_bytes は TTS から渡される bytes[] （タイトル＋本文）
- タイトル行は中央に大きく表示（フェードイン）
- 本文行は従来どおり下部字幕
"""

from __future__ import annotations
import subprocess, tempfile
from itertools import accumulate
from pathlib import Path
from typing import List
import ffmpeg

# -------------- 画面設定 -----------------
W, H          = 1280, 720
FPS           = 30
FONT_PATH     = "C:/Windows/Fonts/meiryo.ttc"
BASE_FONT_SIZE = 100
SUB_FONT_SIZE  = 45
BG_COLOR      = "black@0.4"
BG_PATH       = r"llm_video_generation/assets/background/8.png"

FADE_DURATION = 1        # タイトルのフェード秒
SUB_Y         = H - 140
BOX_H         = 200

SAMPLE_RATE    = 48_000
CHANNEL_LAYOUT = "stereo"
# ----------------------------------------

# ────────────────────────────
# 台本ヘルパ
# ────────────────────────────
def _extract_intro_texts(sce: dict) -> List[str]:
    intro = sce.get("introduction", {})
    title = intro.get("title", "")
    lines = intro.get("text", [])
    return [title, *lines] if title else list(lines)


# ────────────────────────────
# ffmpeg ヘルパ
# ────────────────────────────
def _write_wavs(tmp: Path, audios: List[bytes]) -> List[Path]:
    paths = []
    for i, b in enumerate(audios):
        p = tmp / f"voice_{i:03}.wav"
        p.write_bytes(b); paths.append(p)
    return paths


def _probe_durations(wavs: List[Path]) -> List[float]:
    return [float(ffmpeg.probe(str(w))["format"]["duration"]) for w in wavs]


def _concat_audio(wavs: List[Path], dst: Path):
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs), encoding="utf-8")
    subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(dst)],
        check=True
    )


# ────────────────────────────
# 背景 + 字幕 / タイトル合成
# ────────────────────────────
def _build_video_bg(
    duration: float,
    title: str, t_start: float, t_end: float,
    lines: List[str], starts: List[float], ends: List[float],
):
    v = (
        ffmpeg
        .input(BG_PATH, loop=1, t=duration, framerate=FPS)
        .filter("scale", W, H)
        .filter("setsar", "1")
        .drawbox(x="(iw-w)/2", y=str(H-BOX_H), width=W, height=BOX_H,
                color=BG_COLOR, thickness="fill")
    )

    # --- タイトル中央表示 ---
    fade_expr = f"if(lt(t,{t_start+FADE_DURATION}), (t-{t_start})/{FADE_DURATION}, 1)"
    v = v.drawtext(
        text=title,
        fontfile=FONT_PATH,
        fontsize=str(BASE_FONT_SIZE),
        fontcolor="white",
        alpha=fade_expr,
        x="(w-text_w)/2",
        y="(h-text_h)/2 - 100",
        borderw=6, bordercolor="black",
        shadowx=2, shadowy=2, shadowcolor="black@0.5",
        enable=f"gte(t,{t_start:.3f})",
    )

    # --- 本文字幕（タイトル以外） ---
    for txt, st, ed in zip(lines, starts, ends):
        v = v.drawtext(
            text=txt,
            fontfile=FONT_PATH,
            fontsize=SUB_FONT_SIZE,
            fontcolor="white",
            x="(w-text_w)/2",
            y=str(SUB_Y),
            borderw=2, bordercolor="black",
            shadowx=2, shadowy=2, shadowcolor="black",
            line_spacing=8,
            enable=f"between(t,{st:.3f},{ed:.3f})",
        )
    return v


# ────────────────────────────
# パブリック API
# ────────────────────────────
def build_intro_video(scenario: dict, audio_bytes: List[bytes],
                        output_path: str | Path = "intro.mp4"):
    texts = _extract_intro_texts(scenario)
    if len(texts) != len(audio_bytes):
        raise ValueError("台本行数と音声数が一致しません")

    tmp = Path(tempfile.mkdtemp())
    wavs = _write_wavs(tmp, audio_bytes)

    durs      = _probe_durations(wavs)
    cum       = list(accumulate(durs))
    starts    = [0.0, *cum[:-1]]   # 各行開始秒
    ends      = cum                # 各行終了秒
    total_sec = cum[-1]

    # ---- タイトルを分離 ----
    title, lines          = texts[0], texts[1:]
    t_start, t_end        = starts[0], ends[0]
    sub_starts, sub_ends  = starts[1:], ends[1:]
    wav_lines             = wavs      # 音声は全部まとめるので分離不要

    # ---- 音声一本化 ----
    full_wav = tmp / "full.wav"
    _concat_audio(wav_lines, full_wav)

    # ---- 映像ストリーム ----
    v = _build_video_bg(
        total_sec, title, t_start, t_end,
        lines, sub_starts, sub_ends
    )

    # ---- 音声ストリーム ----
    a = (
        ffmpeg
        .input(str(full_wav))
        .filter("aresample", SAMPLE_RATE)
        .filter("aformat", channel_layouts=CHANNEL_LAYOUT)
    )

    # ---- 出力 ----
    (ffmpeg
        .output(v, a, str(output_path),
                vcodec="libx264", acodec="aac",
                r=FPS, pix_fmt="yuv420p", movflags="faststart",
                loglevel="error")
        .overwrite_output()
        .run())
    print(f"[OK] intro saved → {output_path}")


# ────────────────────────────
# CLI テスト（VoiceVox が起動している前提）
# ────────────────────────────
if __name__ == "__main__":
    import json
    from intro_tts import IntroductionTTSPipeline

    scenario_json = r'''{
    "introduction": {
        "title": "炭素の生物的重要性",
        "text": [
        "炭素は、なぜすべての生物に欠かせないのでしょうか。",
        "宇宙にはたくさんの元素がありますが、",
        "生物にとって特に重要な位置を占めるのが炭素です。",
        "その理由を探ることは、生命の神秘を解き明かす鍵となります。",
        "炭素はどのようにして私たちを形作り、",
        "生物の多様性に貢献しているのでしょうか。",
        "この動画では、その秘密に迫ってみましょう。",
        "炭素の化学的特性について、次に詳しくお話しします。"
        ]
    }}'''
    sce = json.loads(scenario_json)

    # ① 音声を生成
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

    voices = pipeline.run(sce, speaker="1")

    # ② 動画を生成
    build_intro_video(sce, voices, "intro.mp4")
