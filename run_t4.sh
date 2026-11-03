#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
LIMIT=${LIMIT:-645m}
WORKERS=${WORKERS:-3}
OUT=${OUT:-runs}
mkdir -p $OUT

$PY scripts/prepare.py configs/frcnn.yaml configs/retinanet.yaml configs/yolo.yaml configs/yolo_p2.yaml \
    configs/rtdetr.yaml --set $EXTRA

train() {
    $PY scripts/train.py "$@" --out $OUT --set workers=$WORKERS $EXTRA || {
        s=$?
        [ $s -ge 128 ] && exit $s
        echo "FAILED: $*"
    }
}

gpu0() {
    export CUDA_VISIBLE_DEVICES=${GPU0:-0}
    train configs/frcnn.yaml --name frcnn_base
    train configs/frcnn.yaml --name frcnn_anchors --set anchors=fit
    train configs/retinanet.yaml --name retinanet_base
    train configs/retinanet.yaml --name retinanet_anchors --set anchors=fit
    train configs/frcnn.yaml --name frcnn_roi --set anchors=fit \
        params.box_batch_size_per_image=256 params.box_positive_fraction=0.5
    train configs/retinanet.yaml --name retinanet_assign --set anchors=fit \
        params.fg_iou_thresh=0.4 params.bg_iou_thresh=0.3
    train configs/yolo.yaml --name yolo_noaug --set train_size=640 size=640 export_degrade=0 \
        train.mosaic=0 train.close_mosaic=0 train.scale=0 train.translate=0 train.fliplr=0 \
        train.hsv_h=0 train.hsv_s=0 train.hsv_v=0
}

gpu1() {
    export CUDA_VISIBLE_DEVICES=${GPU1:-1}
    train configs/yolo.yaml --name yolo_base --set train_size=640 size=640
    train configs/yolo.yaml --name yolo_1280
    train configs/yolo_p2.yaml --name yolo_p2
    train configs/rtdetr.yaml --name rtdetr
    train configs/yolo.yaml --name yolo_nodegrade --set train_size=640 size=640 export_degrade=0
}

export -f train gpu0 gpu1
export PY WORKERS GPU0 GPU1 OUT EXTRA
timeout -s INT -k 5m $LIMIT bash -c gpu0 >> $OUT/gpu0.log 2>&1 &
timeout -s INT -k 5m $LIMIT bash -c gpu1 >> $OUT/gpu1.log 2>&1 &
wait || true
tail -n 3 $OUT/gpu0.log $OUT/gpu1.log
