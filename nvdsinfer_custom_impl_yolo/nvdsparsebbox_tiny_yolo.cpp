/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

static const int NUM_CLASSES_YOLO = 20;

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static unsigned clamp(const uint val, const uint minVal, const uint maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.left = x - bw / 2;
    b.width = bw;

    b.top = y - bh / 2;
    b.height = bh;

    b.left = clamp(b.left, 0, netW);
    b.width = clamp(b.width, 0, netW);
    b.top = clamp(b.top, 0, netH);
    b.height = clamp(b.height, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (((bbi.left + bbi.width) > netW) || ((bbi.top + bbi.height) > netH)) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });
    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

static std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(const float nmsThresh,
        std::vector<NvDsInferParseObjectInfo>& binfo,
        const uint numClasses)
{
    std::vector<NvDsInferParseObjectInfo> result;
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const std::vector<float> &anchors,
    const uint gridSize, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const float probThresh, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
        
                const float tx 
                    = 1 / (1 + exp (-detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)]));
                const float ty 
                    = 1 / (1 + exp (-detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)]));
                const float bx
                    = x + tx;
                const float by 
                    = y + ty;
/*
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
*/
                const float bw
                    = pw * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)]);
                const float bh
                    = ph * exp (detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > probThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}

/* C-linkage to prevent name-mangling */

static bool NvDsInferParseYoloV2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const float nmsThreshold, const float probthreshold)
{

/*
    static const std::vector<float> kANCHORS = {
        18.3273602, 21.6763191, 59.9827194, 66.0009613,
        106.829758, 175.178879, 252.250244, 112.888962,
        312.656647, 293.384949 };
    static const std::vector<float> kANCHORS = {
        1.08, 1.19, 3.42, 4.41,
        6.63, 11.38, 9.42, 5.11,
        16.62, 10.52 };
*/
    static const std::vector<float> kANCHORS = {
        34.56, 38.08, 109.44, 141.12,
        212.16, 364.16, 301.44, 163.52,
        531.84, 336.64 };

    const uint kNUM_BBOXES = 5;

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }
    
    // std::cout << "Hi!";
    assert (layer.dims.numDims == 3);
    const uint gridSize = layer.dims.d[1];
    const uint stride = networkInfo.width / gridSize;
    // std::cout << "Stride: " << stride << "\n";
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor((const float*)(layer.buffer), kANCHORS, gridSize, stride, kNUM_BBOXES,
                   NUM_CLASSES_YOLO, probthreshold, networkInfo.width, networkInfo.height);

    objectList.clear();
    objectList = nmsAllClasses(nmsThreshold, objects, NUM_CLASSES_YOLO);

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    static const float kNMS_THRESH = 0.2f;
    static const float kPROB_THRESH = 0.6f;
    return NvDsInferParseYoloV2 (
        outputLayersInfo, networkInfo, detectionParams, objectList,
        kNMS_THRESH, kPROB_THRESH);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
