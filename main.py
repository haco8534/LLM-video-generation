from llm_video_generation.src import scenario, format
from llm_video_generation.src.intro import intro_tts, intro_video
from llm_video_generation.src.main import image, main_tts, main_video
import json

# ====== 定数定義 ======
THEME = "【楽して学ぶ】「趣味でプログラミングをやる」というのは成立するのか？"
VIDEO_LENGTH_MINUTES = 4

INTRO_CHAR_STYLE = {
    "1": "もち子さん/ノーマル",
}
MAIN_CHAR_STYLE = {
    "1": "四国めたん/ノーマル",
    "2": "ずんだもん/ノーマル",
}
TTS_PARAMS = {
    "speedScale": 1.1,
    "intonationScale": 1.1,
}

INTRO_BGM_PATH = "llm_video_generation/assets/bgm/Mineral.mp3"
INTRO_SE_PATH = "llm_video_generation/assets/se/5.mp3"
BODY_BGM_PATH = "llm_video_generation/assets/bgm/Voice.mp3"
BODY_SE_PATH = "llm_video_generation/assets/se/3.mp3"

# ====== 各処理 ======
def generate_script(theme: str, minutes: int) -> dict:
    scenario_service = scenario.ScenarioService()
    raw_script = scenario_service.run(theme, minutes)
    styled_script = format.add_design_to_topics(raw_script)
    faced_script = format.add_random_face(styled_script)
    final_script = format.insert_sound_info(
        faced_script,
        intro_bgm=INTRO_BGM_PATH,
        intro_se=INTRO_SE_PATH,
        body_bgm=BODY_BGM_PATH,
        body_se=BODY_SE_PATH,
    )
    print("✅ 台本作成完了")
    return final_script

def collect_images(script: dict) -> list:
    image_service = image.ImageSetService()
    image_urls = image_service.scenario_to_images(script)
    print("✅ 画像収集完了")
    return image_urls

def create_intro_video(script: dict):
    tts_pipeline = intro_tts.IntroductionTTSPipeline(
        char_style=INTRO_CHAR_STYLE,
        tts_params=TTS_PARAMS,
        processes=3,
    )
    audio_bytes = tts_pipeline.run(script, speaker="1")
    intro_video.build_intro_video(script, audio_bytes, output_path="intro.mp4")
    print("✅ イントロ動画生成完了")

def create_main_video(script: dict, image_urls: list):
    tts_pipeline = main_tts.TTSPipeline(
        char_style=MAIN_CHAR_STYLE,
        tts_params=TTS_PARAMS,
        processes=3,
    )
    audio_bytes = tts_pipeline.run(script)
    assembler = main_video.VideoAssembler()
    output = assembler.build_full_video(script, audio_bytes, image_urls, "output.mp4")
    print(f"✅ メイン動画完成: {output.resolve()}")

# ====== 実行エントリーポイント ======
def main():
    #script = generate_script(THEME, VIDEO_LENGTH_MINUTES)
    path = r"llm_video_generation\src\s.txt"
    with open(path, "r", encoding="utf-8") as f:
        script = json.load(f)

    image_urls = collect_images(script)
    create_intro_video(script)
    create_main_video(script, image_urls)

if __name__ == "__main__":
    main()
