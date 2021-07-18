# NanoDet NCNN Android Demo

This repo is an Android object detection demo of NanoDet using
[Tencent's NCNN framework](https://github.com/Tencent/ncnn).

# Tutorial

## Step1.
Download ncnn-android-vulkan.zip from ncnn repo or build ncnn-android from source.

- [ncnn-android-vulkan.zip download link](https://github.com/Tencent/ncnn/releases)

## Step2.
Unzip ncnn-android-vulkan.zip into demo_android_ncnn/app/src/main/cpp or change the ncnn_DIR path to yours in demo_android_ncnn/app/src/main/cpp/CMakeLists.txt

## Step3.
Copy the NanoDet ncnn model file (nanodet_m.param and nanodet_m.bin) from models folder into demo_android_ncnn/app/src/main/assets

* [NanoDet ncnn model download link](https://github.com/RangiLyu/nanodet/releases/download/v0.3.0/nanodet_m_ncnn_model.zip)

If you want to run yolov4-tiny and yolov5s, download them and also put in demo_android_ncnn/app/src/main/assets.

* [Yolov4 and v5 ncnn model download link](https://drive.google.com/file/d/1Qk_1fDvOcFmNppDnaMFW-xFpMgLDyeAs/view?usp=sharing)

## Step4.
Open demo_android_ncnn folder with Android Studio and then build it.

# Screenshot
![](Android_demo.jpg)


# Reference

* [ncnn](https://github.com/tencent/ncnn)
* [YOLOv5_NCNN](https://github.com/WZTENG/YOLOv5_NCNN)
