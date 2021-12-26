# NanoDet OpenVINO Demo

This fold provides NanoDet inference code using
[Intel's OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). Most of the implements in this fold are same as *demo_ncnn*.

## Install OpenVINO Toolkit

Go to [OpenVINO HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Download a suitable version and install.

Follow the official Get Started Guides: https://docs.openvinotoolkit.org/latest/get_started_guides.html

## Set the Environment Variables

### Windows:

Run this command in cmd. (Every time before using OpenVINO)
```cmd
<INSTSLL_DIR>\openvino_2021\bin\setupvars.bat
```


Or set the system environment variables once for all:

Name                  |Value
:--------------------:|:--------:
INTEL_OPENVINO_DIR | <INSTSLL_DIR>\openvino_2021
INTEL_CVSDK_DIR | %INTEL_OPENVINO_DIR%
InferenceEngine_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\share
HDDL_INSTALL_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\hddl
ngraph_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\ngraph\cmake

And add this to ```Path```
```
%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release;%HDDL_INSTALL_DIR%\bin;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\bin;%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\lib
```

### Linux

Run this command in shell. (Every time before using OpenVINO)

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Or edit .bashrc

```shell
vi ~/.bashrc
```

Add this line to the end of the file

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

## Convert model

1. Export ONNX model

   ```shell
   python ./tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
   ```

2. Convert to OpenVINO

   ``` shell
   cd <INSTSLL_DIR>/openvino_2021/deployment_tools/model_optimizer
   ```

   Install requirements for convert tool

   ```shell
   sudo ./install_prerequisites/install_prerequisites_onnx.sh
   ```

   Then convert model. Notice: mean_values and scale_values should be the same with your training settings in YAML config file.
   ```shell
   python3 mo.py --input_model ${ONNX_MODEL} --mean_values [103.53,116.28,123.675] --scale_values [57.375,57.12,58.395] --output output --data_type FP32 --output_dir ${OUTPUT_DIR}
   ```

## Build

### Windows

```cmd
<OPENVINO_INSTSLL_DIR>\openvino_2021\bin\setupvars.bat
mkdir -p build
cd build
cmake ..
msbuild nanodet_demo.vcxproj /p:configuration=release /p:platform=x64
```

### Linux
```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
mkdir build
cd build
cmake ..
make
```


## Run demo

You can convert the model to openvino or use the [converted model](https://drive.google.com/file/d/1dAwIA2pMkSetPEcvB0dvmLaOAK-9h-Lm/view?usp=sharing)

First, move nanodet openvino model files to the `build` folder and rename the files to `nanodet.xml`, `nanodet.mapping`, `nanodet.bin`.

Then run these commands:

### Webcam

```shell
./nanodet_demo 0 0
```

### Inference images

```shell
./nanodet_demo 1 ${IMAGE_FOLDER}/*.jpg
```

### Inference video

```shell
./nanodet_demo 2 ${VIDEO_PATH}
```

### Benchmark

```shell
./nanodet_demo 3 0
```

Model               |Resolution|COCO mAP  | CPU Latency (i7-8700) |
:------------------:|:--------:|:--------:|:---------------------:|
NanoDet-Plus-m      | 320*320  |   27.0   | 5.25ms / 190FPS       |
NanoDet-Plus-m      | 416*416  |   30.4   | 8.32ms / 120FPS       |
NanoDet-Plus-m-1.5x | 320*320  |   29.9   | 7.21ms / 139FPS       |
NanoDet-Plus-m-1.5x | 416*416  |   34.1   | 11.50ms / 87FPS       |

## Custom model

If you want to use custom model, please make sure the hyperparameters
in `nanodet_openvino.h` are the same with your training config file.

```cpp
int input_size[2] = {416, 416}; // input height and width
int num_class = 80; // number of classes. 80 for COCO
int reg_max = 7; // `reg_max` set in the training config. Default: 7.
std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.
```
