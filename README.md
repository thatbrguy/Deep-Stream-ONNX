# Deep-Stream-ONNX

How to deploy ONNX models using DeepStream on Jetson Nano. [[Blog](#)] [[Performance](https://www.youtube.com/watch?v=beX7RqX_FFo)]

This repository provides complementary material to this blog post about deploying an ONNX object detection model using the DeepStream SDK on Jetson Nano. Various experiments were designed to test the features and performance of DeepStream. 

## Setup

### Step 1: Setting up Jetson Nano and DeepStream.

- Follow the instructions in the [blog](#) to setup your Jetson Nano and to install the DeepStream SDK.

### Step 2: Clone this repository.

- Use the bellow commands to clone and move into the repository.

```bash
git clone https://github.com/thatbrguy/Deep-Stream-ONNX.git
cd Deep-Stream-ONNX
```

### Step 3: Download the Tiny YOLOv2 ONNX model.

- Download the Tiny YOLOv2 ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models). We used this [model](https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz) in our experiments.

### Step 4: Compiling the custom bounding box parser.

- A custom bounding box parser function is written in `nvdsparsebbox_tiny_yolo.cpp` inside the `custom_bbox_parser` folder.
- A `Makefile` is configured to compile the custom bounding box parsing function into a shared library (.so) file. It is also available inside the same folder.
- The below variables may need to be set by the user in the `Makefile`  before compiling:

```makefile
# Set the CUDA version.
CUDA_VER:=10 
# Name of the file with the custom bounding box parser function.
SRCFILES:=nvdsparsebbox_tiny_yolo.cpp
# Name of the shared library file to be created after compilation.
TARGET_LIB:=libnvdsinfer_custom_bbox_tiny_yolo.so
# Path to the DeepStream SDK. REPLACE /path/to with the location in your Jetson Nano.
DEEPSTREAM_PATH:=/path/to/deepstream_sdk_v4.0_jetson
```

> Note: If no changes were made to the code by the user, and the blog was followed to set up Jetson Nano and DeepStream, then only the **DEEPSTREAM_PATH** variable may need to be set before compilation. Default values can be used for the other three variables.

- Once the variables are set, save the `Makefile`. Compile the custom bounding box parsing function using: `make -C custom_bbox_parser`.

### Step 5: Launching DeepStream.

- Download the `sample.tar.gz` from this [drive link](https://drive.google.com/open?id=1kZERLw2y9ig9nVwvTPrFOrI5VOTri3d7). Extract the `vids` directory into the `Deep-Stream-ONNX` directory.
- You can launch DeepStream using the following command:

```bash
deepstream-app -c ./config/deepstream_app_custom_yolo.txt
```

- You can edit the config files inside the `config` to alter various settings. You can refer to the [blog](#) for resources on understanding the various properties inside the config files.

## Notes

- Methods for quickly verifying if an ONNX model will be accepted by DeepStream (v4.0):
  - Check if the opset version used is `<= 9`.
  - You can use [onnx2trt](https://github.com/onnx/onnx-tensorrt) to convert an ONNX file into a `.trt` file. I have noticed that if this conversion works, then DeepStream tends to accept the ONNX file. You can refer to the [FAQ](/FAQ.md) section for tips on setting up onnx2trt on the Jetson Nano.