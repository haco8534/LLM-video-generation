from llm_video_generation.src import scenario, image, voice, video
# ① シナリオ生成
svc = scenario.ScenarioService()
script = svc.run(
    "高速逆平方根アルゴリズム〜たった数行に込められた驚異の数学〜", minutes=2
)

# ② TTS 生成（bytes, ファイルは書かない）
char_style = {"1": "四国めたん/ノーマル", "2": "ずんだもん/ノーマル"}
tts_params = {"speedScale": 1.1, "intonationScale": 1.1}
audio_bytes = voice.TTSPipeline(char_style, tts_params, processes=3).run(script)

# ③ 動画組み立て
assembler = video.VideoAssembler()
mp4 = assembler.build_full_video(script, audio_bytes, "output.mp4")
print(f"✅ 完成: {mp4.resolve()}")