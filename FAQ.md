# Resolving Issues

This document discusses some of the issues that I faced while trying to deploy a ONNX object detection model on DeepStream. It also provides some suggestions and solutions for some issues.

### 1. Why did I not use SSD and YOLOv3?

- In the blog, I mentioned that the ONNX model zoo has the SSD and YOLOv3 models. However, since I faced some issues, I used the Tiny YOLOv2 model instead.
- The SSD model was unable to be converted to a TensorRT engine due to the presence of "view" layers in the original PyTorch model. You can refer to this [issue](https://github.com/onnx/onnx-tensorrt/issues/125) for more details.

- How to use onnx2trt?