# Deep-Stream-ONNX

How to deploy ONNX models using DeepStream on Jetson Nano. [[Blog](#)] [[Performance](#)]

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

### Step 5: Setting the Configuration files.

### Step 6: Launching DeepStream.

## Notes

- Methods for quickly verifying if an ONNX model will be accepted by DeepStream:
  - Check if the Opset version used is `<= 9`.
  - You can use ONNX2TRT to convert a onnx file into 