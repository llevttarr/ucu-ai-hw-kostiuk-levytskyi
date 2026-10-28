#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
[ -f splits/holdout.json ] || $PY scripts/split.py

$PY scripts/train.py configs/frcnn.yaml --name frcnn_base
$PY scripts/train.py configs/frcnn.yaml --name frcnn_anchors --set anchors=fit
$PY scripts/train.py configs/frcnn.yaml --name frcnn_proposals --set anchors=fit \
    params.rpn_pre_nms_top_n_test=3000 params.rpn_post_nms_top_n_test=2000
$PY scripts/train.py configs/frcnn.yaml --name frcnn_roi --set anchors=fit \
    params.box_batch_size_per_image=256 params.box_positive_fraction=0.5

$PY scripts/train.py configs/retinanet.yaml --name retinanet_base
$PY scripts/train.py configs/retinanet.yaml --name retinanet_anchors --set anchors=fit
$PY scripts/train.py configs/retinanet.yaml --name retinanet_assign --set anchors=fit \
    params.fg_iou_thresh=0.4 params.bg_iou_thresh=0.3

$PY scripts/train.py configs/yolo.yaml --name yolo_base --set train_size=640 size=640
$PY scripts/train.py configs/yolo.yaml --name yolo_1280
$PY scripts/train.py configs/yolo_p2.yaml --name yolo_p2

$PY scripts/train.py configs/rtdetr.yaml --name rtdetr

$PY scripts/evaluate.py runs/yolo_p2 --temporal
$PY scripts/evaluate.py configs/ensemble.yaml --temporal
$PY scripts/predict.py configs/ensemble.yaml --temporal
