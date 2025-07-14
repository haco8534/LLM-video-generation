import ffmpeg

def add_popin_text(base, text: str):

    # ------- パラメータ -------
    FONT        = '/path/to/font.ttf'  # ←あなたの環境に合わせて
    BASE_SIZE   = 100
    SCALE_MIN   = 0.3
    SCALE_MAX   = 1.2
    SCALE_END   = 1.0
    T_PEAK      = 0.4
    T_END       = 1.0

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
        y='(h-text_h)/2',
        borderw=4, bordercolor='black',
        shadowx=2, shadowy=2, shadowcolor='black@0.5'
    )
