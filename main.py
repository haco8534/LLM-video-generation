# main.py
# ──────────────────────────────────────────────────────────
# 1) 台本生成（scenario.ScenarioBuilder）
# 2) 画像収集（Pixabay）
# 3) イントロ動画作成 → メイン動画作成
# 4) FFmpeg concat demuxer (-c copy) で連結して final.mp4
# SCENARIO_DEBUG_DUMP=1 を指定すると LLM の入出力が .ai_dumps/ に保存
# ──────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
import os
import json
import tempfile
import ffmpeg

# プロジェクト内モジュール
from llm_video_generation.src import scenario, format
from llm_video_generation.src.intro import intro_tts, intro_video
from llm_video_generation.src.main import image, main_tts, main_video

# ===== 設定 =====
THEME = "あなたはなぜ“つい後回し”してしまうのか？"
VIDEO_LENGTH_MINUTES = 4

INTRO_CHAR_STYLE = {"1": "もち子さん/ノーマル"}
MAIN_CHAR_STYLE  = {"1": "四国めたん/ノーマル", "2": "ずんだもん/ノーマル"}
TTS_PARAMS       = {"speedScale": 1.1, "intonationScale": 1.1}

INTRO_BGM_PATH = "llm_video_generation/assets/bgm/Mineral.mp3"
INTRO_SE_PATH  = "llm_video_generation/assets/se/5.mp3"
BODY_BGM_PATH  = "llm_video_generation/assets/bgm/Voice.mp3"
BODY_SE_PATH   = "llm_video_generation/assets/se/3.mp3"

TMP_DIR = Path(tempfile.gettempdir()) / "llm_concat_tmp"
TMP_DIR.mkdir(exist_ok=True)


# ===== FFmpegユーティリティ =====
def _reencode(src: Path, dst: Path) -> None:
    """src を libx264/aac/yuv420p/30fps に再エンコード"""
    (
        ffmpeg
        .input(str(src))
        .output(
            str(dst),
            vcodec="libx264",
            acodec="aac",
            pix_fmt="yuv420p",
            r=30,
            movflags="+faststart",
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )


def concat_videos(intro_path: Path, body_path: Path, output_path: Path) -> Path:
    """イントロ→ボディの順で厳密連結 (-c copy)"""
    intro_path = Path(intro_path)
    body_path  = Path(body_path)

    intro_tmp = TMP_DIR / "intro_prepared.mp4"
    body_tmp  = TMP_DIR / "body_prepared.mp4"

    _reencode(intro_path, intro_tmp)
    _reencode(body_path, body_tmp)

    list_file = TMP_DIR / "list.txt"
    list_file.write_text(
        f"file '{intro_tmp.as_posix()}'\nfile '{body_tmp.as_posix()}'\n",
        encoding="utf-8",
    )

    (
        ffmpeg
        .input(str(list_file), format="concat", safe=0)
        .output(str(output_path), c="copy", movflags="+faststart", loglevel="error")
        .overwrite_output()
        .run()
    )

    # 後始末
    for f in (intro_tmp, body_tmp, list_file):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

    return output_path


# ===== 台本生成 =====
def _make_ai_client():
    """
    scenario.OpenAIClient を基本に、SCENARIO_DEBUG_DUMP=1 なら DumpingAI で包む。
    scenario.py 側の DEBUG_DUMP は環境変数で自動反映される設計。
    """
    base = scenario.OpenAIClient()
    if getattr(scenario, "DEBUG_DUMP", False):
        # ダンプ先ディレクトリの初期化（scenario 側のヘルパを利用）
        if hasattr(scenario, "_set_dump_dir") and hasattr(scenario, "_DUMP_DIR"):
            scenario._set_dump_dir(scenario._DUMP_DIR)
        return scenario.DumpingAI(base)
    return base


def generate_script(theme: str, minutes: int, reference: str | list[str] | None = None) -> dict:
    ai_client = _make_ai_client()
    builder   = scenario.ScenarioBuilder(
        ai_client,
        gpt5_reasoning_effort="minimal",
        gpt5_verbosity="low",
    )
    raw_script    = builder.run(theme, minutes, reference=reference)
    styled_script = format.add_design_to_topics(raw_script)
    faced_script  = format.add_random_face(styled_script)
    final_script  = format.insert_sound_info(
        faced_script,
        intro_bgm=INTRO_BGM_PATH, intro_se=INTRO_SE_PATH,
        body_bgm=BODY_BGM_PATH,  body_se=BODY_SE_PATH,
    )
    print("✅ 台本作成完了")
    return final_script


# ===== 画像収集 =====
def collect_images(script: dict) -> list[str | None]:
    image_service = image.ImageSetService()
    urls          = image_service.scenario_to_images(script)
    print("✅ 画像収集完了")
    return urls


# ===== イントロ動画生成 =====
def create_intro_video(script: dict) -> Path:
    tts_pipeline = intro_tts.IntroductionTTSPipeline(
        char_style=INTRO_CHAR_STYLE, tts_params=TTS_PARAMS, processes=3
    )
    audio_bytes = tts_pipeline.run(script, speaker="1")
    path        = intro_video.build_intro_video(
        script, audio_bytes, output_path="intro.mp4"
    )
    print(f"✅ イントロ動画生成完了: {path}")
    return Path(path)


# ===== メイン動画生成 =====
def create_main_video(script: dict, image_urls: list[str | None]) -> Path:
    tts_pipeline = main_tts.TTSPipeline(
        char_style=MAIN_CHAR_STYLE, tts_params=TTS_PARAMS, processes=3
    )
    audio_bytes = tts_pipeline.run(script)
    assembler   = main_video.VideoAssembler()
    path        = assembler.build_full_video(
        script, audio_bytes, image_urls, "body.mp4"
    )
    print(f"✅ メイン動画生成完了: {path}")
    return Path(path)


# ===== エントリポイント =====
def main() -> None:
    from pathlib import Path

    CACHE_PATH = Path("llm_video_generation/src/main/s.json")

    # 参考資料を使うならここで読み込んで渡せる 無しなら reference = None
    ref_path = Path("llm_video_generation/data/1/ref.txt")
    reference = ref_path.read_text(encoding="utf-8")

    #script      = generate_script(THEME, VIDEO_LENGTH_MINUTES, reference=reference)
    #CACHE_PATH.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    
    script = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    image_urls  = collect_images(script)
    intro_path  = create_intro_video(script)
    body_path   = create_main_video(script, image_urls)
    final_path  = concat_videos(intro_path, body_path, Path("final.mp4"))
    print(f"🎉 完成動画: {final_path.resolve()}")


if __name__ == "__main__":
    main()
