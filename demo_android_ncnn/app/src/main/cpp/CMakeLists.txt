cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20211208-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov5 SHARED
    jni_interface.cpp
    YoloV5.cpp
    YoloV4.cpp
    NanoDet.cpp
)

target_link_libraries(yolov5 ncnn jnigraphics)
