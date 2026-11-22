from dual_annotator import box_iou, filter_dual_consensus, filter_missing_non_overlapping

def test_dual_consensus_agreement():
    t = [{"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "conf": 0.82, "label": "car"}]
    v = [{"x1": 0.11, "y1": 0.11, "x2": 0.49, "y2": 0.51, "conf": 0.64, "label": "car"}]
    res = filter_dual_consensus(t, v, min_iou=0.5, min_teacher_conf=0.5, min_vlm_conf=0.3)
    assert len(res) == 1
    assert res[0]["vlm_conf"] == 0.64

def test_missing_non_overlapping():
    cands = [{"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}, {"x1": 0.6, "y1": 0.6, "x2": 0.8, "y2": 0.8}]
    existing = [(0.1, 0.1, 0.3, 0.3)]
    kept = filter_missing_non_overlapping(cands, existing, max_existing_iou=0.15)
    assert len(kept) == 1 and kept[0]["x1"] == 0.6
