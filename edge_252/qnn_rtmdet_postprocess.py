"""Dependency-light RTMDet COCO post-processing used on the QCS6490.

Changes vs. the original:
  * Each anchor point keeps only its best class (argmax) instead of emitting a
    box for every class above the threshold, so one point can no longer produce
    "car 0.33" + "bus 0.36" + "truck 0.45" for the same vehicle.
  * NMS is class-agnostic (one NMS over all classes) and also removes boxes that
    lie mostly inside a higher-score box. Result: one box per object, labelled
    with the highest-confidence class.
"""
import cv2
import numpy as np

COCO_CLASSES = '''person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush'''.split()
STRIDES = (8, 16, 32)


def nms_agnostic(boxes, scores, iou_threshold, contain_threshold, max_det):
    """Class-agnostic NMS. Returns indices of kept boxes, highest score first.

    A lower-score box is suppressed when either
      * IoU with a kept box > iou_threshold, or
      * (intersection / its own area, vs. the smaller box) > contain_threshold,
        which removes small boxes nested inside a larger, better-scored one.
    Set contain_threshold >= 1.0 to disable the containment rule.
    """
    if boxes.shape[0] == 0:
        return np.empty(0, np.int64)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        cur = order[0]
        keep.append(cur)
        rest = order[1:]
        if rest.size == 0:
            break
        w = np.maximum(0, np.minimum(boxes[cur, 2], boxes[rest, 2]) - np.maximum(boxes[cur, 0], boxes[rest, 0]))
        h = np.maximum(0, np.minimum(boxes[cur, 3], boxes[rest, 3]) - np.maximum(boxes[cur, 1], boxes[rest, 1]))
        inter = w * h
        iou = inter / np.maximum(areas[cur] + areas[rest] - inter, 1e-9)
        contain = inter / np.maximum(np.minimum(areas[cur], areas[rest]), 1e-9)
        order = rest[(iou <= iou_threshold) & (contain <= contain_threshold)]
    return np.asarray(keep, np.int64)


def postprocess(outputs, scale, original_hw, input_size, score_threshold, iou_threshold, max_det,
                contain_threshold=0.85):
    cls_by_hw, box_by_hw = {}, {}
    for output in outputs:
        output = np.asarray(output)
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"Unexpected output {output.shape}")
        (cls_by_hw if output.shape[1] == 80 else box_by_hw)[tuple(output.shape[2:])] = output[0]

    all_boxes, all_scores, all_labels = [], [], []
    for stride in STRIDES:
        h = input_size // stride
        key = (h, h)
        logits, distances = cls_by_hw[key], box_by_hw[key]
        scores = 1 / (1 + np.exp(-logits.transpose(1, 2, 0).reshape(-1, 80)))
        ys, xs = np.meshgrid(np.arange(h, dtype=np.float32) * stride,
                             np.arange(h, dtype=np.float32) * stride, indexing="ij")
        points = np.stack((xs.ravel(), ys.ravel()), axis=1)
        distances = distances.transpose(1, 2, 0).reshape(-1, 4)
        boxes = np.column_stack((points[:, 0] - distances[:, 0], points[:, 1] - distances[:, 1],
                                 points[:, 0] + distances[:, 2], points[:, 1] + distances[:, 3]))
        # One class per anchor point: the highest-scoring one.
        best_labels = scores.argmax(axis=1)
        best_scores = scores[np.arange(scores.shape[0]), best_labels]
        mask = best_scores > score_threshold
        if mask.any():
            all_boxes.append(boxes[mask])
            all_scores.append(best_scores[mask])
            all_labels.append(best_labels[mask])

    if not all_scores:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)

    boxes = np.concatenate(all_boxes) / scale
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_hw[1])
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_hw[0])
    scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
    keep = nms_agnostic(boxes, scores, iou_threshold, contain_threshold, max_det)
    return boxes[keep], scores[keep], labels[keep]


def draw(frame, boxes, scores, labels, fps):
    out = frame.copy()
    for (x1, y1, x2, y2), score, label in zip(boxes.astype(int), scores, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{COCO_CLASSES[label]} {score:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
    return out