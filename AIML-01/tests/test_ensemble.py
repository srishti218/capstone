from src.models.ensemble import blend_scores

def test_blend_scores():
    assert abs(blend_scores(0.8, 0.4, 0.5) - 0.6) < 1e-9
