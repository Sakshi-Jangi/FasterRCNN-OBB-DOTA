from evaluation_utils import compute_iou

IOU_THRESHOLD = 0.5


def evaluate_image(
    pred_boxes,
    gt_boxes
):

    tp = 0
    fp = 0

    matched_gt = set()

    for pred_box in pred_boxes:

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(
            gt_boxes
        ):

            iou = compute_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_iou >= IOU_THRESHOLD
            and
            best_gt_idx not in matched_gt
        ):

            tp += 1

            matched_gt.add(
                best_gt_idx
            )

        else:

            fp += 1

    fn = (
        len(gt_boxes)
        -
        len(matched_gt)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    accuracy = (
        tp /
        (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )

    return {

        "tp": tp,

        "fp": fp,

        "fn": fn,

        "precision": precision,

        "recall": recall,

        "accuracy": accuracy

    }