with open(r'llm_video_generation\src\a.txt', 'r', encoding='UTF-8') as f:
    a = f.readlines()

import json
a = "".join([f"{b}" for b in a])
a = json.loads(a)

segs = a['segments']

for seg in segs:
    speaker = "四国めたん" if seg['script']['speaker'] == "1" else "ずんだもん"
    print(f"{speaker}: {seg['script']['text']}")

'''from llm_video_generation.src import scenario, image, voice, video
import pickle

with open(r"llm_video_generation\src\s.txt", "r", encoding="utf-8") as f:
    import json
    script = json.load(f)

char_style = {"1": "四国めたん/ノーマル", "2": "ずんだもん/ノーマル"}
tts_params = {"speedScale": 1.1, "intonationScale": 1.1}
tts = voice.TTSPipeline(char_style, tts_params, processes=5)
audio_bytes = tts.run(script)

with open(r'llm_video_generation\src\v.pkl', 'wb',) as f:
    pickle.dump(audio_bytes, f)'''