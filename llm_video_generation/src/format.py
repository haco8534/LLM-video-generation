import json

def add_design_to_topics(data: dict) -> dict:
    design_counter = 1
    for segment in data.get("segments", []):
        if segment.get("type") == "topic":
            segment["design"] = str(design_counter)
            design_counter += 1
    return data



if __name__ == "__main__":
    path = r"llm_video_generation\src\s.txt"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    add_design_to_topics(data)

    from rich import print
    print(data)