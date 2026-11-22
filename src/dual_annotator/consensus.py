from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]

def box_iou(b1: Box, b2: Box) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)

def filter_dual_consensus(
    teacher_preds: List[Dict],
    vlm_preds: List[Dict],
    min_iou: float = 0.48,
    min_teacher_conf: float = 0.50,
    min_vlm_conf: float = 0.28,
) -> List[Dict]:
    """Keep only bounding boxes independently confirmed by both the domain YOLO Teacher and zero-shot VLM."""
    accepted = []
    for t in teacher_preds:
        if t["conf"] < min_teacher_conf:
            continue
        tb = (t["x1"], t["y1"], t["x2"], t["y2"])
        best_vlm = max(
            (v["conf"] for v in vlm_preds if v.get("label") == t.get("label") and box_iou(tb, (v["x1"], v["y1"], v["x2"], v["y2"])) >= min_iou),
            default=0.0,
        )
        if best_vlm >= min_vlm_conf:
            out = dict(t)
            out["vlm_conf"] = best_vlm
            out["consensus_score"] = round((t["conf"] + best_vlm) / 2.0, 4)
            accepted.append(out)
    return accepted

def filter_missing_non_overlapping(candidates: List[Dict], existing_boxes: List[Box], max_existing_iou: float = 0.15) -> List[Dict]:
    """Filter candidate detections so only genuinely missing objects not overlapping existing annotations are proposed."""
    return [
        c for c in candidates
        if all(box_iou((c["x1"], c["y1"], c["x2"], c["y2"]), eb) < max_existing_iou for eb in existing_boxes)
    ]
