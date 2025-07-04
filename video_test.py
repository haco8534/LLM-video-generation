import ffmpeg
from pathlib import Path

# 環境に応じて調整
FONT = "C:/Windows/Fonts/meiryo.ttc"
BG_TOPIC = "llm_video_generation/assets/background/2.png"
CHAR_ROOT = "llm_video_generation/assets/character"
TOPIC_DUR = 3
W, H = 1280, 720
FPS = 30
SAMPLE_RATE = 48000
CHANNEL_LAYOUT = "stereo"

def _mk_background(path: str, dur: int):
    return (
        ffmpeg.input(path, loop=1, t=dur, framerate=FPS)
        .filter("scale", W, H)
    )

def _build_topic_graph(title: str):
    rect_w, rect_h = 950, 570
    bg = _mk_background(BG_TOPIC, TOPIC_DUR)

    # 動的フォントサイズ（2文字ごとに10下げる）
    max_fontsize = 80
    min_fontsize = 55
    step = 1
    step_size = 10
    base_length = 11

    length = len(title)
    decrement = ((length - base_length) // step) * step_size
    fontsize = max(min_fontsize, max_fontsize - decrement)

    # --- 背景・矩形・下線 -----------------------
    bg = bg.drawbox(
        x=f"(iw-{rect_w})/2 - 80",
        y=f"(ih-{rect_h})/2 + 20",
        width=rect_w,
        height=rect_h,
        color="#9DA7EB",
        thickness="fill",
    )
    bg = bg.drawbox(
        x=f"(iw-{rect_w})/2 - 100",
        y=f"(ih-{rect_h})/2",
        width=rect_w,
        height=rect_h,
        color="#7281E3@0.8",
        thickness="fill",
    )
    bg = bg.drawbox(
        x=f"(iw-{rect_w})/2 - 80",
        y=f"(ih-{rect_h})/2 + 340",
        width=rect_w - 30,
        height=3,
        color="white",
        thickness="fill",
    )

    # --- タイトルテキスト -----------------
    bg = bg.drawtext(
        text=title,
        fontfile="C:/Windows/Fonts/meiryob.ttc",
        fontsize=fontsize,
        fontcolor="orange",
        x="(w-text_w)/2 - 100",
        y="(h-text_h)/2",
        borderw=4,
        bordercolor="white",
        shadowcolor="black",
        shadowx=2,
        shadowy=2,
    )

    # --- キャラクター ----------------------
    char = (
        ffmpeg.input(f"{CHAR_ROOT}/ずんだもん/think.png", loop=1, t=TOPIC_DUR, framerate=FPS)
        .filter("scale", 700, -1)
    )
    bg = ffmpeg.overlay(bg, char, x=780, y=50)

    # --- 無音音声 --------------------------
    audio = ffmpeg.input(
        f"anullsrc=channel_layout={CHANNEL_LAYOUT}:sample_rate={SAMPLE_RATE}",
        format="lavfi",
        t=TOPIC_DUR,
    )

    return bg, audio



# 実行用関数
def main():
    out_path = "test_topic.mp4"
    title = "存在しない漢字について"  # お好きなタイトルに変更可
    video, audio = _build_topic_graph(title)
    (
        ffmpeg.output(
            video, audio, out_path,
            vcodec="libx264", acodec="aac",
            pix_fmt="yuv420p", movflags="faststart",
            loglevel="error"
        )
        .overwrite_output()
        .run()
    )
    print(f"[OK] {out_path} に出力しました")

if __name__ == "__main__":
    main()
