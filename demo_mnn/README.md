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
   python ./tools/export_onnx.py
   ```

2. Use *onnx-simplifier* to simplify it

   ``` shell
   python -m onnxsim ./output.onnx sim.onnx
   ```

3. Convert to MNN

   ``` shell
   python -m MNN.tools.mnnconvert -f ONNX --modelFile sim.onnx --MNNModel nanodet-320.mnn
   ```

It should be note that the input size does not have to be 320, it can be any integer multiple of strides,
since NanoDet is anchor free. We can adapt the shape of `dummy_input` in *./tools/export_onnx.py* to get ONNX and MNN models
with different input sizes.

Here are converted model [Baidu Disk](https://pan.baidu.com/s/1DE4_yo0xez6Wd95xv7NnDQ)(extra code: *5mfa*),
[Google Drive](https://drive.google.com/drive/folders/1dEdAXkof_lCusYBNrgbGzdLFZbDPMiFn?usp=sharing).

## Build

The python code *demo_mnn.py* can run directly and independently without main NanoDet repo.
`NanoDetONNX` and `NanoDetTorch` are two classes used to check the similarity of MNN inference results
with ONNX model and Pytorch model. They can be remove with no side effects.

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

`demo_mnn.py` provide an inference class `NanoDetMNN` that combines preprocess, post process, visualization.
Besides it can be used in command line with the form:

```shell
demo_mnn.py [-h] [--model_path MODEL_PATH] [--cfg_path CFG_PATH]
    [--img_fold IMG_FOLD] [--result_fold RESULT_FOLD]
    [--input_shape INPUT_SHAPE INPUT_SHAPE]
    [--backend {MNN,ONNX,torch}]
```

For example:

``` shell
# run MNN 320 model
python ./demo_mnn.py --model_path ../model/nanodet-320.mnn --img_fold ../imgs --result_fold ../results
# run MNN 160 model
python ./demo_mnn.py --model_path ../model/nanodet-160.mnn --input_shape 160 160 --backend MNN
# run onnx model
python ./demo_mnn.py --model_path ../model/sim.onnx --backend ONNX
# run Pytorch model
python ./demo_mnn.py --model_path ../model/nanodet_m.pth ../../config/nanodet-m.yml --backend torch
```

### C++

C++ inference interface is same with NCNN code, to detect images in a fold, run:

``` shell
./nanodet-mnn "1" "../imgs/*.jpg"
```

For speed benchmark

``` shell
./nanodet-mnn "3" "0"
```

## Reference

[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/MNN)

[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)

[NanoDet NCNN](https://github.com/RangiLyu/nanodet/tree/main/demo_ncnn)

[MNN](https://github.com/alibaba/MNN)

## Example results

![screenshot](./results/000252.jpg?raw=true)
![screenshot](./results/000258.jpg?raw=true)
