# Deep-Stream-ONNX
How to deploy ONNX models using DeepStream on Jetson Nano. [[Blog](#)] [[Performance](#)]

This repository provides complementary material to this blog post about deploying an ONNX object detection model using the DeepStream SDK on Jetson Nano. Various experiments were designed to test the features and performance of DeepStream. 

## Setup

Step 1: Setting up Jetson Nano and DeepStream.

Step 2: Clone this repository.

Step 3: Download the Tiny YOLOv2 ONNX model.

Step 4: Compiling the custom bounding box parser.

Step 5: Setting the Configuration files.

Step 6: Launching DeepStream.

## Notes

- Methods for quickly verifying if an ONNX model will be accepted by DeepStream:
  - Check if the Opset version used is `<= 9`.
  - You can use ONNX2TRT to convert a onnx file into 