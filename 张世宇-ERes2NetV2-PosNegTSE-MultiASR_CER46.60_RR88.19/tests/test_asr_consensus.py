from asr_consensus import select_consensus_text, should_select_tse_text


def test_consensus_prefers_majority_text():
    text, index, _ = select_consensus_text(
        ["打开客厅空调", "打开客厅空调", "打开空调"]
    )
    assert text == "打开客厅空调"
    assert index == 0


def test_tse_route_rejects_identity_collapse():
    assert not should_select_tse_text(
        baseline_text="打开客厅空调",
        tse_text="打开客厅空调",
        raw_candidates=["打开客厅空调"],
        similarity_gain=-0.20,
    )


def test_tse_route_accepts_consistent_correction():
    assert should_select_tse_text(
        baseline_text="空调调到百分之三",
        tse_text="空调调到百分之三十",
        raw_candidates=["空调调到百分之三", "空调调到百分之三十"],
        similarity_gain=0.03,
    )
