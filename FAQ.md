# Resolving Issues

This document discusses some of the issues that I faced while trying to deploy a ONNX object detection model on DeepStream. It also provides some suggestions and solutions for some issues.

### 1. Why did you not use SSD and YOLOv3?

- In the blog, I mentioned that the [ONNX model zoo](https://github.com/onnx/models) has the SSD and YOLOv3 models. However, since I faced some issues while trying to use them.
- The SSD model was unable to be converted to a TensorRT engine due to the presence of "view" layers in the original PyTorch model. You can refer to this [issue](https://github.com/onnx/onnx-tensorrt/issues/125) for more details.
- The opset version of YOLOv3 in the model zoo was 10. DeepStream v4.0 supports opset versions <=9.
- To circumvent these issues, I used the Tiny YOLOv2 model instead. This model was compatible with DeepStream.

### 2. How did you setup onnx2trt for Jetson Nano?

- This [branch](https://github.com/onnx/onnx-tensorrt/tree/5.1) (5.1) of the onnx2trt repository must be used for building the library from source. You can clone the branch using the following command:

```bash
git clone --recursive --branch 5.1 https://github.com/onnx/onnx-tensorrt.git
```

- Now, you can build onnx2trt using the following commands. Note that the cmake command is broken into multiple lines.

```bash
cd onnx-tensorrt
mkdir build
cd build
cmake .. \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc \
-DTENSORRT_ROOT=/ust/src/tensorrt \
-DGPU_ARCHS="53"
make -j2
sudo make install
```

- Once built, you can run the following command to convert an onnx model to a `.trt` file. 

```bash
onnx2trt my_model.onnx -o my_engine.trt
```



