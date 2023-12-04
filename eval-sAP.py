import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from models.build import MODELS


def calculate_tp(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1, 2, 2)
    lines_gt = lines_gt.reshape(-1, 2, 2)
    euid = lambda x, y: ((x - y) ** 2).sum(axis=-1)
    diff = np.minimum(
        euid(lines_dt[:, None, 0], lines_gt[None, :, 0]) + euid(lines_dt[:, None, 1], lines_gt[None, :, 1]),
        euid(lines_dt[:, None, 1], lines_gt[None, :, 0]) + euid(lines_dt[:, None, 0], lines_gt[None, :, 1])
    )
    choice = np.argmin(diff, axis=1)
    dist = diff[np.arange(len(lines_dt)), choice]
    hit = np.zeros(len(lines_gt), dtype=bool)
    tp = np.zeros(len(lines_dt), dtype=bool)
    for i in range(len(lines_dt)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = True
    return tp


def average_precision(y_true, y_score, n_gt: int):
    ind = np.argsort(-y_score)
    tp = y_true[ind].astype(np.float32)
    fp = 1 - tp
    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt

    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["wireframe", "yorkurban"])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[5, 10, 15],
                        help="List of thresholds to use for evaluation")
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, args.dataset, args.model)

    for threshold in args.thresholds:
        with open(f"data/{args.dataset}/test.json") as f:
            dataset = json.load(f)

        images = dataset["images"]
        annotations = dataset["annotations"]
        tp_list, scores_list, n_gt = [], [], 0

        for anno in tqdm(annotations, leave=False):
            # Normalize ground truth lines
            lines_gt = np.array(anno["points"])[np.array(anno["lines"])].reshape(-1, 4)
            image = images[anno["image_id"] - 1]
            h, w = image["height"], image["width"]
            lines_gt = lines_gt / [w, h, w, h] * 128

            file_name = image["file_name"]
            with np.load(os.path.join(output_dir, file_name.replace(".jpg", ".npz"))) as npz:
                lines_dt, scores_dt = npz["lines"], npz["scores"]
            ind = np.argsort(-scores_dt)
            lines_dt, scores_dt = lines_dt[ind], scores_dt[ind]

            tp = calculate_tp(lines_dt, lines_gt, threshold)

            tp_list.append(tp)
            scores_list.append(scores_dt)
            n_gt += len(lines_gt)

        y_true = np.concatenate(tp_list, dtype=np.float32)
        y_score = np.concatenate(scores_list)

        ap = average_precision(y_true, y_score, n_gt)
        print(f"sAP{threshold:<2} = {ap * 100:.1f}")
