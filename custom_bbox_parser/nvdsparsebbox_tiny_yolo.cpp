#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

/**
 * Function expected by DeepStream for decoding the TinyYOLOv2 output.
 *  
 * C-linkage [extern "C"] was written to prevent name-mangling. This function must return true after
 * adding all bounding boxes to the objectList vector.
 * 
 * @param [outputLayersInfo] std::vector of NvDsInferLayerInfo objects with information about the output layer.
 * @param [networkInfo] NvDsInferNetworkInfo object with information about the TinyYOLOv2 network.
 * @param [detectionParams] NvDsInferParseDetectionParams with information about some config params.
 * @param [objectList] std::vector of NvDsInferParseObjectInfo objects to which bounding box information must
 * be stored.
 * 
 * @return true
 */

// This is just the function prototype. The definition is written at the end of the file.
extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/**
 * Bounds values between the range [minVal, maxVal].
 * 
 * Values that are out of bounds are set to their boundary values. 
 * For example, consider the following: clamp(bbox.left, 0, netW). This 
 * translates to min(netW, max(0, bbox.left)). Hence, if bbox.left was 
 * negative, it is set to 0. If bbox.left is greater than netW, it is set 
 * to netW.
 * 
 * @param [val] Value to be bound.
 * @param [minVal] Lower bound.
 * @param [maxVal] Upper bound.
 * 
 * @return A value that is bound in the range [minVal, maxVal].
 */
static unsigned clamp(const uint val, const uint minVal, const uint maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

/**
 * Computes the overlap between two lines along a single axis (1D).
 * 
 * Overlap is computed along either the X or Y axis as specified by the
 * user. If no overlap is available, 0 is returned.
 * 
 * @param [x1min] Minimum coordinate of the 1st line along a single axis.
 * @param [x1max] Maximum coordinate of the 1st line along a single axis.
 * @param [x2min] Minimum coordinate of the 2nd line along a single axis.
 * @param [x2max] Maximum coordinate of the 2nd line along a single axis.
 * 
 * @return Overlap between the two lines as a float value.
 */
static float overlap1D(float x1min, float x1max, float x2min, float x2max)
{   
    if (x1min > x2min)
    {
        std::swap(x1min, x2min);
        std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
}

/**
 * Computes the Intersection over Union (IoU) of two rectangular bounding boxes.
 * 
 * IoU is computed as follows. First, the 1D overlap of the two boxes along the
 * X and Y axis is computed. They are multiplied to get the 2D overlap of the two
 * boxes. The areas of both boxes are also computed. The 2D overlap gives us the
 * intersection. The sum of the areas of both boxes minus the intersection gives us
 * the union. IoU is just the ratio of the intersection to the union. A 0 is returned
 * if there is union is 0.
 * 
 * @param [bbox1] NvDsInferParseObjectInfo object containing the 1st bounding box info.
 * @param [bbox2] NvDsInferParseObjectInfo object containing the 2nd bounding box info.
 * 
 * @return IoU between the two bounding boxes as a float value.
 */
static float computeIoU(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2)
{
    float overlapX
        = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
    float overlapY
        = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
    float area1 = (bbox1.width) * (bbox1.height);
    float area2 = (bbox2.width) * (bbox2.height);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
}

/**
 * Returns a boolean value indicating whether a bounding box confidence value is greater.
 * 
 * Given two NvDsInferParseObjectInfo bounding box object, we compare the confidence values
 * and return true if the 1st bounding box confidence value is greater than 2nd bounding box 
 * confidence value. This function is passed as an argument for the sorting function.
 * 
 * @param [bbox1] NvDsInferParseObjectInfo object containing the 1st bounding box info.
 * @param [bbox2] NvDsInferParseObjectInfo object containing the 2nd bounding box info.
 * 
 * @return Boolean value indicating whether confidence of bbox1 is greater than confidence of bbox2.
 */
static bool compareBBoxConfidence(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2)
{
    return bbox1.detectionConfidence > bbox2.detectionConfidence;
}

/**
 * Creates the NvDsInferParseObjectInfo Bounding Box object given attributes.
 * 
 * @param [bx] Bounding Box center X-Coordinate.
 * @param [by] Bounding Box center Y-Coordinate.
 * @param [bw] Bounding Box width.
 * @param [bh] Bounding Box height.
 * @param [stride] Ratio of the image width to the grid size.
 * @param [netW] Width of the image.
 * @param [netH] Height of the image.
 * 
 * @return NvDsInferParseObjectInfo Bounding Box object.
 */
static NvDsInferParseObjectInfo createBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo bbox;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    bbox.left = x - bw / 2;
    bbox.width = bw;

    bbox.top = y - bh / 2;
    bbox.height = bh;

    // Bounds bbox values between [minVal, maxVal]
    bbox.left = clamp(bbox.left, 0, netW);
    bbox.width = clamp(bbox.width, 0, netW);
    bbox.top = clamp(bbox.top, 0, netH);
    bbox.height = clamp(bbox.height, 0, netH);

    return bbox;
}

/**
 * Adds an NvDsInferParseObjectInfo Bounding Box object to a vector.
 * 
 * This function is used to accumulate all the bounding boxes present in a single
 * frame of the video into a vector called bboxInfo.
 * 
 * @param [bx] Bounding Box center X-Coordinate.
 * @param [by] Bounding Box center Y-Coordinate.
 * @param [bw] Bounding Box width.
 * @param [bh] Bounding Box height.
 * @param [stride] Ratio of the image width to the grid size.
 * @param [netW] Width of the image.
 * @param [netH] Height of the image.
 * @param [maxIndex] Class ID of the detected bounding box.
 * @param [maxProb] Confidence of the detected bounding box
 * @param [bboxInfo] std::vector of bounding boxes.
 */
static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& bboxInfo)
{
    NvDsInferParseObjectInfo bbox = createBBox(bx, by, bw, bh, stride, netW, netH);
    if (((bbox.left + bbox.width) > netW) || ((bbox.top + bbox.height) > netH)) return;

    bbox.detectionConfidence = maxProb;
    bbox.classId = maxIndex;
    bboxInfo.push_back(bbox);
}

/**
 * Performs Non Maximum Suppression (NMS) over all bounding boxes of a single class.
 * 
 * The bounding boxes are first sorted by their confidence values in decending order. Then,
 * the bounding boxes are iterated over to remove multiple detections of the same object
 * and only retain the bounding box of the highest confidence of that object. Two bounding
 * boxes are considered to belong to the same object if they have a large overlap.
 * 
 * @param [inputBBoxInfo] std::vector of bounding boxes belonging to a single class.
 * @param [nmsThresh] Overlap threshold for NMS.
 * 
 * @return std::vector of bounding boxes of a single class after NMS is applied.
 */
static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(std::vector<NvDsInferParseObjectInfo> inputBBoxInfo, const float nmsThresh)
{
    std::stable_sort(inputBBoxInfo.begin(), inputBBoxInfo.end(), compareBBoxConfidence);
    std::vector<NvDsInferParseObjectInfo> outputBBoxInfo;

    for (auto bbox1 : inputBBoxInfo)
    {
        bool keep = true;
        for (auto bbox2 : outputBBoxInfo)
        {
            if (keep)
            {
                float overlap = computeIoU(bbox1, bbox2);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) outputBBoxInfo.push_back(bbox1);
    }
    return outputBBoxInfo;
}

/**
 * Performs Non Maximum Suppression (NMS) over all classes.
 * 
 * Interatively performs NMS over all classes using the single class NMS function.
 * 
 * @param [bboxInfo] [description]
 * @param [numClasses] Number of classes.
 * @param [nmsThresh] Overlap threshold for NMS.
 * 
 * @return std::vector of bounding boxes of all classes after NMS is applied.
 */

static std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(std::vector<NvDsInferParseObjectInfo>& bboxInfo, const uint numClasses, const float nmsThresh)
{
    std::vector<NvDsInferParseObjectInfo> resultBBoxes;
    
    // std::vector of std::vector (of size numClasses) to hold classwise bounding boxes.
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);

    // Bounding box with attribute "classID" is pushed into the index "classID" of "splitBoxes". This
    // way, splitBoxes will have bounding boxes belonging to the same class at each index.
    for (auto &bbox : bboxInfo)
    {
        splitBoxes.at(bbox.classId).push_back(bbox);
    }

    // Applying NMS for bounding boxes belonging to the same class and collecting the resultant
    // bounding boxes in resultBBoxes.
    for (auto &bboxesPerClass : splitBoxes)
    {
        bboxesPerClass = nonMaximumSuppression(bboxesPerClass, nmsThresh);
        resultBBoxes.insert(resultBBoxes.end(), bboxesPerClass.begin(), bboxesPerClass.end());
    }
    return resultBBoxes;
}

/**
 * Deocodes the output of TinyYOLOv2 and outputs a vector of NvDsInferParseObjectInfo bounding boxes.
 * 
 * Loops over the output of TinyYOLOv2 and calculates the bounding box locations. Ignores bounding
 * boxes with detection probability less than the probability threshold.
 * 
 * @param [detections] Output of TinyYOLOv2 as a flattened object.
 * @param [netW] Width of the image.
 * @param [netH] Height of the image.
 * @param [anchors] XY locations of the Anchor Boxes multiplied by the stride value provided as a vector.
 * The anchor box vector is of the form [x_1, y_1, x_2, y_2, ..., x_5, y_5].
 * @param [numBBoxes] Number of Anchors. This is equal to the number of predicted bounding boxes per grid cell.
 * @param [gridSize] Size of the grid.
 * @param [stride] Ratio of the image width to the grid size.
 * @param [probThresh] Bounding box probability threshold.
 * @param [numOutputClasses] Overlap threshold for NMS.
 * 
 * @return std::vector of bounding boxes of all classes obtained by decoding the TinyYOLOv2 output.
 */

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
    const float* detections, const uint& netW, const uint& netH, 
    const std::vector<float> &anchors, const uint numBBoxes,
    const uint gridSize, const uint stride, const float probThresh,
    const uint numOutputClasses)
{
    std::vector<NvDsInferParseObjectInfo> bboxInfo;
    const int b_offset = gridSize * gridSize;

    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
	    const int xy_offset = (y * gridSize + x);

            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[b * 2];
                const float ph = anchors[b * 2 + 1];

                const int start_idx = xy_offset + b_offset * b * (5 + numOutputClasses);
        
                const float sigmoid_tx 
                    = 1 / (1 + exp (-detections[start_idx + 0 * b_offset]));
                const float sigmoid_ty 
                    = 1 / (1 + exp (-detections[start_idx + 1 * b_offset]));
                const float bx
                    = x + sigmoid_tx;
                const float by 
                    = y + sigmoid_ty;
                const float bw
                    = pw * exp (detections[start_idx + 2 * b_offset]);
                const float bh
                    = ph * exp (detections[start_idx + 3 * b_offset]);
                const float objectness
                    = 1 / (1 + exp(-detections[start_idx + 4 * b_offset]));

                int maxIndex = -1;
                float maxProb = 0.0f;
                float max_class_val = 0.0f;

                // Finding the maximum value and well as the index with maximum value 
                // prior to applying softmax. Since softmax is monotonically increasing,
                // maxIndex can be calculated here itself.
                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float class_val = detections[start_idx + (5 + i) * b_offset];

                    if (class_val > max_class_val)
                    {
                        max_class_val = class_val;
                        maxIndex = i;
                    }
                }

                float sum_exp = 0.0f;
                // Calculating the denominator of the softmax function. Note that, we are 
                // performing softmax(x - max(x)) where x is the list of class outputs. 
                // Note that softmax(x + a) gives the same result as softmax(x) where, a is 
                // a constant value. By replacing a with -max(x) softmax becomes more 
                // stable since exp does not have to deal with large numbers.
                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float class_val = detections[start_idx + (5 + i) * b_offset];
                    float class_exp = exp(class_val - max_class_val); 
                    sum_exp = sum_exp + class_exp;
                }

                // The largest softmax probability among all x values will be softmax(max(x)) 
                // since softmax is monotonically increasing. Since we are actually calculating 
                // softmax(x_i - max(x)), when x_i = max(x), we get softmax(max(x) - max(x)), 
                // which is just 1 / sum_exp.
                float max_softmax_prob = 1 / sum_exp;
                maxProb = objectness * max_softmax_prob;

                if (maxProb > probThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, bboxInfo);
                }
            }
        }
    }
    return bboxInfo;
}


/**
 * Function expected by DeepStream for decoding the TinyYOLOv2 output.
 *  
 * C-linkage [extern "C"] was written to prevent name-mangling. This function must return true after
 * adding all bounding boxes to the objectList vector.
 * 
 * @param [outputLayersInfo] std::vector of NvDsInferLayerInfo objects with information about the output layer.
 * @param [networkInfo] NvDsInferNetworkInfo object with information about the TinyYOLOv2 network.
 * @param [detectionParams] NvDsInferParseDetectionParams with information about some config params.
 * @param [objectList] std::vector of NvDsInferParseObjectInfo objects to which bounding box information must
 * be stored.
 * 
 * @return true
 */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV2Tiny(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    
    // Initializing some parameters.

    /**
     * In our case, we know stride and gridSize beforehand. If this is
     * not the case, they can be calculated using the following formulae:
     * 
     * const uint gridSize = layer.dims.d[1];
     * const uint stride = networkInfo.width / gridSize;
     */

    static const uint kSTRIDE           = 32;
    static const uint kGRID_SIZE        = 13;
    static const uint kNUM_ANCHORS      = 5;
    static const float kNMS_THRESH      = 0.2f;
    static const float kPROB_THRESH     = 0.6f;
    static const uint kNUM_CLASSES_YOLO = 20;

    /**
     * The vector kANCHORS is actually the anchor box coordinates
     * multiplied by the stride variable. Since we know the stride
     *  value before hand, we store the multiplied values as it saves
     *  some computation. [For our case, stride = 32]
     */

    static const std::vector<float> kANCHORS = {
        34.56, 38.08, 109.44, 141.12,
        212.16, 364.16, 301.44, 163.52,
        531.84, 336.64 };

    // Some assertions and error checking.
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }

    if (kNUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << kNUM_CLASSES_YOLO << std::endl;
    }
    
    // Obtaining the output layer.
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    assert (layer.dims.numDims == 3);
	
    // Decoding the output tensor of TinyYOLOv2 to the NvDsInferParseObjectInfo format.
    std::vector<NvDsInferParseObjectInfo> objects =
        decodeYoloV2Tensor(
        (const float*)(layer.buffer), networkInfo.width, networkInfo.height, 
        kANCHORS, kNUM_ANCHORS, kGRID_SIZE, kSTRIDE, kPROB_THRESH,
        kNUM_CLASSES_YOLO  
        );

    // Applying Non Maximum Suppression to remove multiple detections of the same object.
    objectList.clear();
    objectList = nmsAllClasses(objects, kNUM_CLASSES_YOLO, kNMS_THRESH);

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);
