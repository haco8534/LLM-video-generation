def main():
    from llm_video_generation.src import scenario, image, voice, video

    '''
    ss = scenario.ScenarioService()
    script = ss.run("国語の授業って必要？？本読めばよくない？？", minutes=1)
    print(script)
    '''

    with open(r"llm_video_generation\src\s.txt", "r", encoding="utf-8") as f:
        import json
        script = json.load(f)
    '''
    char_style = {"1": "四国めたん/ノーマル", "2": "ずんだもん/ノーマル"}
    tts_params = {"speedScale": 1.1, "intonationScale": 1.1}
    tts = voice.TTSPipeline(char_style, tts_params, processes=5)
    audio_bytes = tts.run(script)
    '''

    with open(r"llm_video_generation\src\v.pkl", "rb") as f:
        import pickle
        audio_bytes = pickle.load(f)


    with open(r"llm_video_generation\src\i.pkl", "rb") as f:
        import pickle
        urls = pickle.load(f)

    assembler = video.VideoAssembler()
    mp4 = assembler.build_full_video(script, audio_bytes, urls, "output.mp4")
    print(f"✅ 完成: {mp4.resolve()}")

if __name__ == "__main__":
    main()