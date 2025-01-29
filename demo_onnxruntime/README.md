# NanoDet ONNX Runtime Demo

This project provides NanoDet inference using [ONNX Runtime](https://onnxruntime.ai/docs/api/c/).

# How to build

## Linux

### Step1.
Build and install OpenCV from https://github.com/opencv/opencv

### Step2.
Build and install ONNX Runtime and link it to the project.

For easy installation and integration with CMake follow [this tutorial](https://medium.com/@massimilianoriva96/onnxruntime-integration-with-ubuntu-and-cmake-5d7af482136a). It will allow you to directly find the onnx_runtime package in CMake.
	
Alternatively, follow [the official build and installation path](https://onnxruntime.ai/docs/build/inferencing.html#linux). Then, link onnx_runtime to the project manually. 

### Step3.

Build project

``` shell script
mkdir build
cd build
cmake ..
make
```

# Export your model to the ONNX format

Use the script `tool/export_onnx.py` to convert PyTorch model in `.pth` or `.ckpt` format to ONNX.

```shell script
python tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
```

Copy the model to the demo program folder (`demo_onnxruntime/build`).

## Modify hyperparameters

If you want to use a custom model, please make sure the hyperparameters
in `nanodet.h` are the same with your training config file.

```cpp
int input_size[2] = {192, 192};   // input height and width
int num_class = 1;                // number of classes
int reg_max = 7;                  // `reg_max` set in the training config. Default: 7.
std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
```

# Run demo

## Webcam

```shell script
./onnxruntime_demo 0 0
```

## Inference images

```shell script
./onnxruntime_demo 1 ${IMAGE_FOLDER}/*.jpg
```
