import ffmpeg

W, H = 1280, 720
FPS = 30

def _mk_background(path: str, dur: int):
    """静止画を指定秒数ループさせる背景 video stream"""
    return (
        ffmpeg.input(path, loop=1, t=dur, framerate=FPS)
        .filter("scale", W, H)
    )


