# NanoDet MNN Demo

This fold provides NanoDet inference code using
[Alibaba's MNN framework](https://github.com/alibaba/MNN). Most of the implements in
this fold are same as *demo_ncnn*.

## Install MNN

### Python library

Just run:

``` shell
pip install MNN
```

### C++ library

Please follow the [official document](https://www.yuque.com/mnn/en/build_linux) to build MNN engine.

## Convert model

1. Export ONNX model

   ```shell
    python tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
   ```

2. Convert to MNN

   ``` shell
   python -m MNN.tools.mnnconvert -f ONNX --modelFile sim.onnx --MNNModel nanodet.mnn
   ```

It should be note that the input size does not have to be fixed, it can be any integer multiple of strides,
since NanoDet is anchor free. We can adapt the shape of `dummy_input` in *./tools/export_onnx.py* to get ONNX and MNN models
with different input sizes.

Here are converted model
[Download Link](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_mnn.mnn).

## Build

For C++ code, replace `libMNN.so` under *./mnn/lib* with the one you just compiled, modify OpenCV path at CMake file,
and run

``` shell
mkdir build && cd build
cmake ..
make
```

Note that a flag at `main.cpp` is used to control whether to show the detection result or save it into a fold.

``` c++
#define __SAVE_RESULT__ // if defined save drawed results to ../results, else show it in windows
```

## Run

### Python

The multi-backend python demo is still working in progress.

### C++

C++ inference interface is same with NCNN code, to detect images in a fold, run:

``` shell
./nanodet-mnn "1" "../imgs/*.jpg"
```

For speed benchmark

``` shell
./nanodet-mnn "3" "0"
```

## Custom model

If you want to use custom model, please make sure the hyperparameters
in `nanodet_mnn.h` are the same with your training config file.

```cpp
int input_size[2] = {416, 416}; // input height and width
int num_class = 80; // number of classes. 80 for COCO
int reg_max = 7; // `reg_max` set in the training config. Default: 7.
std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
```

## Reference

[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN)

[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)

[NanoDet NCNN](https://github.com/RangiLyu/nanodet/tree/main/demo_ncnn)

[MNN](https://github.com/alibaba/MNN)

## Example results

![screenshot](./results/000252.jpg?raw=true)
![screenshot](./results/000258.jpg?raw=true)
