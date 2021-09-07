# NanoDet TorchScript / LibTorch Demo

This folder provides NanoDet inference code using for LibTorch.

## Install dependencies

This project needs OpenCV and CMake to work.

Install CMake using a package manager of your choice. For example, the following command will install CMake on Ubuntu:

```bash
sudo apt install cmake libopencv-dev
```

Also, you'll need to download LibTorch. Refer to [this page](https://pytorch.org/cppdocs/installing.html) for more info.

## Convert model

Export TorchScript model using `tools/export_torchscript.py`:

```shell
python ./tools/export_torchscript.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH} --input_shape ${MO}
```
## Build

### Linux
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```
