# YOLOv5 NCNN Implementation

This repo provides C++ implementation of [YOLOv5 model](https://github.com/ultralytics/yolov5) using
Tencent's NCNN framework.

# Notes

Currently NCNN does not support Slice operations with steps, therefore I removed the Slice operation
and replaced the input with a downscaled image and stacked it to match the channel number. This
may slightly reduce the accuracy.

# Credits 

* [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5) 
* [NCNN by Tencent](https://github.com/tencent/ncnn)

仅供学习。
