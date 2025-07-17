"""
intro_video_builder.py
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import subprocess, tempfile
from itertools import accumulate
from pathlib import Path
from typing import List, Sequence, Optional
import ffmpeg

# -------------- 画面設定 -----------------
W, H          = 1280, 720
FPS           = 30
FONT_PATH     = "C:/Windows/Fonts/meiryo.ttc"
BASE_FONT_SIZE = 100
SUB_FONT_SIZE  = 40
BG_COLOR      = "black@0.4"
BG_PATH       = r"llm_video_generation/assets/background/8.png"
CHAR_DIR = Path("llm_video_generation/assets/character/冥鳴ひまり") 

FADE_DURATION = 1        # タイトルのフェード秒
SUB_Y         = H - 140
BOX_H         = 200

CHAR_BASE_X      = 1000   # もともと指定していた位置
CHAR_SLIDE_OFFSET = 300   # 右 (+X) にどれだけ余分に置いておくか
SLIDE_DURATION    = 0.8   # スライドにかける秒数

# -------------- 音声設定 -----------------
SAMPLE_RATE    = 48_000
CHANNEL_LAYOUT = "stereo"

DEFAULT_BGM_VOLUME = 0.1
DEFAULT_SE_VOLUME  = 0.6     
# ----------------------------------------


# ────────────────────────────
# 台本ヘルパ
# ────────────────────────────
def _extract_intro_lines(sce: dict):
    """タイトル / 本文 / 顔 ID をまとめて取得"""
    intro = sce.get("introduction", {})
    title = intro.get("title", "")
    lines_info = intro.get("text", [])
    lines  = [li["script"] for li in lines_info]
    faces  = [int(li.get("face", 1)) for li in lines_info]   # face 無し → 1
    return title, lines, faces


# ────────────────────────────
# WAV 出力 & probe
# ────────────────────────────
def _write_wavs(tmp: Path, audios: Sequence[bytes]) -> List[Path]:
    paths = []
    for i, b in enumerate(audios):
        p = tmp / f"voice_{i:03}.wav"
        p.write_bytes(b); paths.append(p)
    return paths


def _probe_durations(wavs: Sequence[Path]) -> List[float]:
    return [float(ffmpeg.probe(str(w))["format"]["duration"]) for w in wavs]


def _concat_audio(wavs: Sequence[Path], dst: Path):
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs), encoding="utf-8")
    subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(dst)],
        check=True
    )


# ────────────────────────────
# 背景 + 字幕 / タイトル合成
# ────────────────────────────
def _build_video_bg(duration: float,
                    title: str, t_start: float,
                    lines: List[str], starts: List[float], ends: List[float],
                    faces: List[int]):
    v = (ffmpeg
        .input(BG_PATH, loop=1, t=duration, framerate=FPS)
        .filter("scale", W, H)
        .filter("setsar", "1")
        .drawbox(x="(iw-w)/2", y=str(H-BOX_H), width=W, height=BOX_H,
                color=BG_COLOR, thickness="fill"))

    # ─ 立ち絵 ─  face 番号ごとに 1 回だけ overlay
    face_intervals: dict[int, list[tuple[float, float]]] = {}
    for st, ed, fc in zip(starts, ends, faces):
        face_intervals.setdefault(fc, []).append((st, ed))

    for fc, ivals in face_intervals.items():
        # 同じ PNG を scale したストリームは 1 つだけ作る
        char_path = CHAR_DIR / f"{fc}.png"
        ch = (
            ffmpeg.input(str(char_path))
                .filter("scale", -1, 650)
        )

        # ───── ① 各セリフの再生区間 ─────
        first_st = ivals[0][0]                 # このキャラが初めて出る時刻
        enable_expr = f"gte(t,{first_st:.3f})"

        # ───── ② スライドインの X 座標式 ─────
        x_expr = (
            f"if(lte(t,{first_st:.3f}),"                                    # まだ登場前
            f"{CHAR_BASE_X + CHAR_SLIDE_OFFSET},"
            f"if(lt(t,{first_st + SLIDE_DURATION:.3f}),"                    # スライド中
            f"{CHAR_BASE_X + CHAR_SLIDE_OFFSET} - "
            f"{CHAR_SLIDE_OFFSET}*(t-{first_st:.3f})/{SLIDE_DURATION},"
            f"{CHAR_BASE_X}))"                                              # スライド後
        )

        # ───── ③ overlay ─────
        v = ffmpeg.overlay(
            v, ch,
            x=x_expr,
            y=f"{H-BOX_H-400}",
            enable=enable_expr,
        )

    # ─ タイトル（フェード） ─
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

    # ─ 字幕 ─
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
# オーディオ合成
# ────────────────────────────
def _build_audio_mix(
    tts_wav: Path,
    total_sec: float,
    starts: List[float],
    bgm_path: Optional[str|Path],
    se_paths: Optional[Sequence[Optional[str|Path]]],
    bgm_vol: float,
    se_vol: float,
):
    """tts_wav + (bgm) + (SEs) -> mixed audio stream"""
    streams = []

    # ① TTS (メイン)
    tts = ffmpeg.input(str(tts_wav))
    streams.append(tts)

    # ② BGM ループ
    if bgm_path:
        bgm = (
            ffmpeg
            .input(str(bgm_path), stream_loop=-1)
            .filter("atrim", duration=total_sec)
            .filter("asetpts", "N/SR/TB")
            .filter("volume", bgm_vol)
        )
        streams.append(bgm)

    # ③ SE（タイトル＋各セリフ）
    if se_paths:
        if len(se_paths) != len(starts):
            raise ValueError("se_paths の要素数は台本行数と同じにしてください")
        for se_path, st in zip(se_paths, starts):
            if se_path is None:
                continue
            delay_ms = int(st * 1000)
            se = (
                ffmpeg
                .input(str(se_path))
                .filter("adelay", f"{delay_ms}|{delay_ms}")
                .filter("volume", se_vol)
            )
            streams.append(se)

    # ④ mix
    if len(streams) == 1:
        return streams[0]       # BGM/SE 無し
    return ffmpeg.filter(streams, "amix",
                        inputs=len(streams),
                        dropout_transition=0,
                        normalize=0)


# ────────────────────────────
# パブリック API
# ────────────────────────────
def build_intro_video(
    scenario: dict,
    audio_bytes: List[bytes],
    output_path: str | Path = "intro.mp4",
    *,
    bgm_path: str | Path | None = None,
    se_paths: Sequence[Optional[str | Path]] | None = None,
    bgm_volume: float = DEFAULT_BGM_VOLUME,
    se_volume: float = DEFAULT_SE_VOLUME,
):
    
    # ▼ ❶ ここで JSON 中の "sound" からパスを拾う ──────────
    sound_cfg = scenario.get("sound", {})
    if bgm_path is None:
        bgm_path = sound_cfg.get("intro_bgm")

    if se_paths is None:
        se_path_title = sound_cfg.get("intro_se")
        # タイトルだけ鳴らし、それ以外は None
        if se_path_title:
            se_paths = [se_path_title] + [None] * len(scenario.get("introduction", {}).get("text", []))
        else:
            se_paths = None
    # ────────────────────────────────────────────────

    # --- 台本 / 尺 ---
    title, lines, faces = _extract_intro_lines(scenario)
    texts = [title, *lines]
    if len(texts) != len(audio_bytes):
        raise ValueError("台本行数と音声数が一致しません")

    tmp = Path(tempfile.mkdtemp())
    wavs = _write_wavs(tmp, audio_bytes)       # 各行 WAV

    durs   = _probe_durations(wavs)
    cum    = list(accumulate(durs))
    starts = [0.0, *cum[:-1]]
    ends   = cum
    total  = cum[-1]

    # タイトル / 本文 分離
    title, lines      = texts[0], texts[1:]
    sub_starts, sub_ends = starts[1:], ends[1:]

    # --- TTS 一本化 ---
    full_wav = tmp / "tts_full.wav"
    _concat_audio(wavs, full_wav)

    # --- 映像ストリーム ---
    v_stream = _build_video_bg(
        total, title, starts[0],
        lines, sub_starts, sub_ends,
        faces                           # ★ 追加
    )

    # --- 音声ストリーム (TTS + BGM + SE) ---
    a_stream = _build_audio_mix(
        full_wav, total, starts,
        bgm_path, se_paths,
        bgm_volume, se_volume,
    ).filter("aresample", SAMPLE_RATE)\
    .filter("aformat", channel_layouts=CHANNEL_LAYOUT)

    # --- 出力 ---
    (ffmpeg
        .output(v_stream, a_stream, str(output_path),
                vcodec="libx264", acodec="aac",
                r=FPS, pix_fmt="yuv420p",
                movflags="faststart",
                loglevel="error")
        .overwrite_output()
        .run())
    out_path = Path(output_path).resolve()
    print(f"[OK] intro saved → {out_path}")
    return out_path 


# ────────────────────────────
# CLI テスト例
# ────────────────────────────
if __name__ == "__main__":
    import json
    from intro_tts import IntroductionTTSPipeline

    # ▶ JSON ファイルからシナリオ読み込み
    with open("llm_video_generation/src/s.txt", "r", encoding="utf-8") as f:
        scenario = json.load(f)

    # ▶ 音声を生成（title + text）
    pipeline = IntroductionTTSPipeline(
        char_style={"1": "冥鳴ひまり/ノーマル"},
        tts_params={"speedScale": 1.05},
        processes=3,
    )
    voices = pipeline.run(scenario, speaker="1")

    # ▶ 動画を生成（BGM/SE は JSON 中の sound に従う）
    build_intro_video(scenario, voices, output_path="intro.mp4")

