# NanoDet NCNN Android Demo

This repo is an Android object detection demo of NanoDet using
[Tencent's NCNN framework](https://github.com/Tencent/ncnn).

# Tutorial

## Step1.
Download ncnn-android-vulkan.zip from ncnn repo or build ncnn-android from source.

- [ncnn-android-vulkan.zip download link](https://github.com/Tencent/ncnn/releases)

## Step2.
Unzip ncnn-android-vulkan.zip into demo_android_ncnn/app/src/main/cpp or change the ncnn_DIR path to yours in demo_android_ncnn/app/src/main/cpp/CMakeLists.txt

```bash
# e.g. change to `ncnn-20211208-android-vulkan` if download version 200211208
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20211208-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
```

## Step3.
Copy the NanoDet ncnn model file and rename to nanodet.param and nanodet.bin from models folder into demo_android_ncnn/app/src/main/assets

* [NanoDet ncnn model download link](https://drive.google.com/file/d/1cuVBJiFKwyq1-l3AwHoP2boTesUQP-6K/view?usp=sharing)

If you want to run yolov4-tiny and yolov5s, download them and also put in demo_android_ncnn/app/src/main/assets.

* [Yolov4 and v5 ncnn model download link](https://drive.google.com/file/d/1Qk_1fDvOcFmNppDnaMFW-xFpMgLDyeAs/view?usp=sharing)

## Step4.
Open demo_android_ncnn folder with Android Studio and then build it.

# Screenshot
![](Android_demo.jpg)

# Notice

* The FPS in the app includes pre-process, post-process and visualization, not equal to the model inference time.

* If meet error like `No version of NDK matched the requested version`, set `android { ndkVersion` to your ndk version.

* If you want to use custom model, remember to change the hyperparams in `demo_android_ncnn/app/src/main/cpp/NanoDet.h` the same with your training config.

# Reference

* [ncnn](https://github.com/tencent/ncnn)
* [YOLOv5_NCNN](https://github.com/WZTENG/YOLOv5_NCNN)
