from llm_video_generation.src import scenario

def test_generate_scenario():
    theme = "コンピュータ科学の失敗たとえで笑い転げるよ！"
    minute = 3
    
    scenario_svc = scenario.ScenarioService()
    result = scenario_svc.run(theme, minute)

    print(result)

    assert "segments" in result

if __name__ == '__main__':
    test_generate_scenario()