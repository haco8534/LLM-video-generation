import ffmpeg
from typing import List
import json
import os

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
DURATION = 1  # 秒
FONT_PATH = 'C:/Windows/Fonts/meiryo.ttc'
OUTPUT_FILE = 'a.mp4'

# === 映像セグメント作成関数 ===
def create_video_segment(voice_path: str, text: str, speaker: str,face: dict, index: int):
    output_path = f"./temp/segment_{index:03}.mp4"
    background = create_background()
    characters = overlay_characters(background,face)
    boxed = draw_subtitle_box(characters)
    texted = draw_subtitle_text(boxed, text, speaker)
    audio = ffmpeg.input(voice_path)

    (
        ffmpeg.output(
        texted, audio,
        output_path,
        vcodec='libx264',
        acodec='aac',
        pix_fmt='yuv420p',
        movflags='faststart'
        )
        .run(overwrite_output=True)
    )

    return output_path

# === 背景生成 ===
def create_background():
    return ffmpeg.input(
        f'color=c=white:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:d={DURATION}:r={FPS}',
        f='lavfi'
    )

# === キャラクター画像の合成 ===
def overlay_characters(base,face):
    chara1 = (
        ffmpeg.input(f'assets/character/ずんだもん/{face["2"]}.png')
        .filter('scale', 400, -1)
        .filter('hflip')
    )
    result = ffmpeg.overlay(base, chara1, x=-50, y=250)

    chara2 = (
        ffmpeg.input(f'assets/character/四国めたん/{face["1"]}.png')
        .filter('scale', 400, -1)
    )
    result = ffmpeg.overlay(result, chara2, x=950, y=250)

    return result

# === 字幕背景ボックス描画 ===
def draw_subtitle_box(base):
    return base.drawbox(
        x=f'(iw-{VIDEO_WIDTH})/2',
        y=f'{VIDEO_HEIGHT - 180}',
        width=VIDEO_WIDTH,
        height=200,
        color='black@0.6',
        thickness='fill'
    )

# === 字幕テキスト描画 ===
def draw_subtitle_text(base, text, speaker):
    if speaker == "1":
        return base.drawtext(
        text=text,
        fontfile=FONT_PATH,
        fontsize=35,
        fontcolor='white',
        x='(w-text_w)/2',
        y='h-140',
        borderw=2,
        bordercolor='#E7609E',
        shadowcolor='black',
        shadowx=2,
        shadowy=2,
        line_spacing=10,
        enable='between(t,0,5)'
        )
    elif speaker == "2":
        return base.drawtext(
        text=text,
        fontfile=FONT_PATH,
        fontsize=35,
        fontcolor='white',
        x='(w-text_w)/2',
        y='h-140',
        borderw=2,
        bordercolor='#6CBB5A',
        shadowcolor='black',
        shadowx=2,
        shadowy=2,
        line_spacing=10,
        enable='between(t,0,5)'
        )

# ===連結用のリストファイルを作成===
def create_concat_list_file(file_paths: List[str], list_path: str = "./temp/concat_list.txt"):
    with open(list_path, 'w', encoding='utf-8') as f:
        for path in file_paths:
            f.write(f"file '{path}'\n")
    return list_path

def create_concat_list_file(file_paths: List[str], list_path: str = "./temp/concat_list.txt"):
    with open(list_path, 'w', encoding='utf-8') as f:
        for path in file_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    return list_path


def concat_segments(list_file_path: str, output_path: str = "final_output.mp4"):
    (
        ffmpeg
        .input(list_file_path, format='concat', safe=0)
        .output(output_path, c='copy')
        .run(overwrite_output=True)
    )


def extract_segment_prompts(scenario: dict):
    """
    dialogue なら script.textを
    id 昇順 (≒時系列) に返す純粋関数
    """
    prompts = []
    for seg in scenario["segments"]:
        if seg["type"] == "dialogue":
            prompts.append({"text":seg["script"]["text"],"type":seg["script"]["speaker"],"face": seg["script"]["face"]})
    return prompts

# === 実行ブロック ===
if __name__ == "__main__":
    with open('./modules/a.txt', 'r', encoding='utf-8') as f:
        scenario = json.load(f)
        scenario_segments = extract_segment_prompts(scenario)

    # セグメント生成＆保存
    segment_paths = []
    current_face = {"1": "normal1", "2": "normal1"}
    for i, scenario_segment in enumerate(scenario_segments,1):
        text = scenario_segment["text"]
        speaker = scenario_segment["type"]

        current_face[speaker] = scenario_segment["face"]

        voice_path = f"./assets/voice/{i:03}.wav"
        segment_path = create_video_segment(voice_path, text, speaker,current_face, i)
        segment_paths.append(segment_path)

    # concat 用リストを作成
    concat_list_path = create_concat_list_file(segment_paths)

    # セグメントを連結
    concat_segments(concat_list_path, output_path="output.mp4")


