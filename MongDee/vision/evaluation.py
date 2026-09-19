"""Evaluation metrics (MongDee_Master_Prompt.md section 37).

These compute what the master prompt asks to *report* — detection
precision/recall, tracking ID-switches / MOTA, cross-camera counting
error — from predictions + ground truth. There is no bundled labelled
dataset; scripts/evaluate.py runs these against an annotation file you
provide from real footage.

Section 37 rule: never claim "100% accuracy" — report the measured number.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision.tracking.matching import greedy_match, iou_matrix


@dataclasses.dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def evaluate_detection(
    predictions: list[list[list[float]]],
    ground_truth: list[list[list[float]]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """predictions / ground_truth: one list of [x1,y1,x2,y2] boxes per frame."""
    tp = fp = fn = 0
    for pred_boxes, gt_boxes in zip(predictions, ground_truth):
        if not gt_boxes:
            fp += len(pred_boxes)
            continue
        if not pred_boxes:
            fn += len(gt_boxes)
            continue
        iou = iou_matrix(gt_boxes, pred_boxes)
        matches, unmatched_gt, unmatched_pred = greedy_match(iou, iou_threshold)
        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionMetrics(round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn)


@dataclasses.dataclass
class TrackingMetrics:
    mota: float          # Multi-Object Tracking Accuracy (CLEAR-MOT, simplified)
    id_switches: int
    false_positives: int
    misses: int
    num_gt: int
    precision: float
    recall: float


def evaluate_tracking(
    pred_tracks: list[list[tuple[int, list[float]]]],
    gt_tracks: list[list[tuple[int, list[float]]]],
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """Per frame: list of (track_id, [x1,y1,x2,y2]). CLEAR-MOT style:
    MOTA = 1 - (misses + false_positives + id_switches) / num_gt."""
    id_switches = fp = misses = num_gt = matched = 0
    gt_to_pred: dict[int, int] = {}  # last frame's gt_id -> pred_id assignment

    for pred_frame, gt_frame in zip(pred_tracks, gt_tracks):
        num_gt += len(gt_frame)
        if not gt_frame:
            fp += len(pred_frame)
            continue
        if not pred_frame:
            misses += len(gt_frame)
            continue

        gt_ids = [g[0] for g in gt_frame]
        pred_ids = [p[0] for p in pred_frame]
        iou = iou_matrix([g[1] for g in gt_frame], [p[1] for p in pred_frame])
        pairs, unmatched_gt, unmatched_pred = greedy_match(iou, iou_threshold)

        matched += len(pairs)
        fp += len(unmatched_pred)
        misses += len(unmatched_gt)

        seen_now: dict[int, int] = {}
        for gi, pi in pairs:
            gid, pid = gt_ids[gi], pred_ids[pi]
            if gid in gt_to_pred and gt_to_pred[gid] != pid:
                id_switches += 1
            gt_to_pred[gid] = pid
            seen_now[gid] = pid

    mota = 1.0 - (misses + fp + id_switches) / num_gt if num_gt else 0.0
    precision = matched / (matched + fp) if (matched + fp) else 0.0
    recall = matched / num_gt if num_gt else 0.0
    return TrackingMetrics(
        round(mota, 4), id_switches, fp, misses, num_gt, round(precision, 4), round(recall, 4)
    )


@dataclasses.dataclass
class CountingMetrics:
    predicted_unique: int
    true_unique: int
    absolute_error: int
    percent_error: float


def evaluate_counting(predicted_unique: int, true_unique: int) -> CountingMetrics:
    err = abs(predicted_unique - true_unique)
    pct = err / true_unique * 100.0 if true_unique else 0.0
    return CountingMetrics(predicted_unique, true_unique, err, round(pct, 2))


@dataclasses.dataclass
class MatchingMetrics:
    """Cross-camera identity linking quality vs a ground-truth mapping
    of (camera, local_track_id) -> true person id."""

    correct_merges: int
    wrong_merges: int          # two tracks of DIFFERENT people put in one global id
    missed_merges: int         # two tracks of the SAME person left in different global ids
    false_match_rate: float
    missed_match_rate: float


def evaluate_matching(
    predicted: dict[tuple[str, int], str],
    ground_truth: dict[tuple[str, int], str],
) -> MatchingMetrics:
    keys = [k for k in predicted if k in ground_truth]
    correct = wrong = missed = 0
    total_same_person_pairs = 0
    total_pred_merge_pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            same_true = ground_truth[a] == ground_truth[b]
            same_pred = predicted[a] == predicted[b]
            if same_true:
                total_same_person_pairs += 1
            if same_pred:
                total_pred_merge_pairs += 1
            if same_true and same_pred:
                correct += 1
            elif same_pred and not same_true:
                wrong += 1
            elif same_true and not same_pred:
                missed += 1
    fmr = wrong / total_pred_merge_pairs if total_pred_merge_pairs else 0.0
    mmr = missed / total_same_person_pairs if total_same_person_pairs else 0.0
    return MatchingMetrics(correct, wrong, missed, round(fmr, 4), round(mmr, 4))
