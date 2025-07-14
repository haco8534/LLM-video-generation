import ffmpeg

W, H = 1280, 720

def add_popin_text(base, text: str):

    # ------- パラメータ -------
    FONT        = "C:/Windows/Fonts/meiryo.ttc" # ←あなたの環境に合わせて
    BASE_SIZE   = 100
    SCALE_MIN   = 0.3
    SCALE_MAX   = 1.2
    SCALE_END   = 1.0
    T_PEAK      = 0.3
    T_END       = 0.5

    fontsize_expr = f'''
    if(
        lt(t,{T_PEAK}),
        {BASE_SIZE}*({SCALE_MIN} + ({SCALE_MAX}-{SCALE_MIN})*(t/{T_PEAK})),
        if(
            lt(t,{T_END}),
            {BASE_SIZE}*({SCALE_END} + ({SCALE_MAX}-{SCALE_END}) * (1 - (t-{T_PEAK})/({T_END}-{T_PEAK})) * cos(3.1415*(t-{T_PEAK})/({T_END}-{T_PEAK}))),
            {BASE_SIZE}
        )
    )
    '''.replace('\n', '')

    # ------- drawtext 合成 -------
    return base.filter(
        'drawtext',
        text=text,
        fontfile=FONT,
        fontsize=fontsize_expr,
        fontcolor='white',
        x='(w-text_w)/2',
        y='(h-text_h)/2 - 150',
        borderw=6, bordercolor='black',
        shadowx=2, shadowy=2, shadowcolor='black@0.5'
    )

if __name__ == '__main__':
    # 黒背景 3 秒の映像に「Hello Pop!」を表示
    path = r"llm_video_generation\assets\background\5.png"
    bg = (ffmpeg
            .input(path, loop=1, t=2, framerate=30)
            .filter("scale", W, H)
            .filter("setsar", "1") 
        )
    pop = add_popin_text(bg, '生物と炭素の関係')

    # 出力
    (
        ffmpeg
        .output(pop, 'pop_out.mp4', r=30, vcodec='libx264', pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )
