from vision.evaluation import (
    evaluate_counting,
    evaluate_detection,
    evaluate_matching,
    evaluate_tracking,
)


def test_detection_perfect():
    boxes = [[[0, 0, 10, 10], [20, 20, 30, 30]]]
    m = evaluate_detection(boxes, boxes)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_detection_with_fp_and_fn():
    pred = [[[0, 0, 10, 10], [50, 50, 60, 60]]]      # 1 correct, 1 spurious
    gt = [[[0, 0, 10, 10], [100, 100, 110, 110]]]    # 1 matched, 1 missed
    m = evaluate_detection(pred, gt)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.precision == 0.5 and m.recall == 0.5


def test_tracking_no_switches():
    frames_pred = [[(1, [0, 0, 10, 10])], [(1, [1, 0, 11, 10])], [(1, [2, 0, 12, 10])]]
    frames_gt = [[(7, [0, 0, 10, 10])], [(7, [1, 0, 11, 10])], [(7, [2, 0, 12, 10])]]
    m = evaluate_tracking(frames_pred, frames_gt)
    assert m.id_switches == 0
    assert m.mota == 1.0


def test_tracking_counts_id_switch():
    frames_pred = [[(1, [0, 0, 10, 10])], [(2, [1, 0, 11, 10])]]  # same person, pred id flips 1->2
    frames_gt = [[(7, [0, 0, 10, 10])], [(7, [1, 0, 11, 10])]]
    m = evaluate_tracking(frames_pred, frames_gt)
    assert m.id_switches == 1
    assert m.mota < 1.0


def test_counting_metrics():
    m = evaluate_counting(predicted_unique=95, true_unique=100)
    assert m.absolute_error == 5
    assert m.percent_error == 5.0


def test_matching_metrics_detects_wrong_and_missed_merges():
    gt = {("CAM01", 1): "P1", ("CAM02", 5): "P1", ("CAM01", 2): "P2", ("CAM02", 6): "P2"}
    pred = {
        ("CAM01", 1): "G1", ("CAM02", 5): "G1",   # correct merge (P1)
        ("CAM01", 2): "G2", ("CAM02", 6): "G3",   # missed merge (P2 split)
    }
    m = evaluate_matching(pred, gt)
    assert m.correct_merges == 1
    assert m.missed_merges == 1
    assert m.wrong_merges == 0

    pred_wrong = {("CAM01", 1): "G1", ("CAM02", 5): "G1", ("CAM01", 2): "G1", ("CAM02", 6): "G1"}
    m2 = evaluate_matching(pred_wrong, gt)
    assert m2.wrong_merges > 0
    assert m2.false_match_rate > 0
