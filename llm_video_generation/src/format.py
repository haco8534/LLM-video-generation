import json
from typing import Dict, Any

def add_design_to_topics(data: dict) -> dict:
    design_counter = 1
    for segment in data.get("segments", []):
        if segment.get("type") == "topic":
            segment["design"] = str(design_counter)
            design_counter += 1
    return data

def insert_sound_info(
    json_data: Dict[str, Any],
    intro_bgm: str,
    intro_se: str,
    body_bgm: str,
    body_se: str
) -> Dict[str, Any]:
    """既存のJSONデータに'sound'キーを追加して返す"""

    # soundセクションを構築
    sound_info = {
        "intro_bgm": intro_bgm,
        "intro_se": intro_se,
        "body_bgm": body_bgm,
        "body_se": body_se
    }

    # 元のJSONに追加（元データを直接変更せずにコピーして返す）
    new_json = {"sound": sound_info, **json_data}
    return new_json

if __name__ == "__main__":
    path = r"llm_video_generation\src\s.txt"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = insert_sound_info(
    data,
    intro_bgm="llm_video_generation/assets/bgm/Mineral.mp3",
    intro_se="llm_video_generation/assets/se/5.mp3",
    body_bgm="llm_video_generation/assets/bgm/Voice.mp3",
    body_se="llm_video_generation/assets/se/3.mp3"
)
    from rich import print
    print(data)