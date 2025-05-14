import ffmpeg

def create_video_segment(voice_path):
    # 黒背景映像（音声なし）
    seg_vid = ffmpeg.input('color=c=white:s=1280x720:d=5:r=30', f='lavfi')
    # 音声
    speak = ffmpeg.input(voice_path)

    seg_vid = seg_vid.drawbox(
    x='(1280-1000)/2',
    y='720 - 120',
    width=1000,
    height=80,
    color='black@0.5',
    thickness='fill'
)

    # 映像＋音声を結合
    return ffmpeg.output(seg_vid, speak, 'a.mp4', vcodec='libx264', acodec='aac')

if __name__ == "__main__":
    voice_path = "./assets/voice/001.wav"
    vid = create_video_segment(voice_path)
    vid.run(overwrite_output=True)