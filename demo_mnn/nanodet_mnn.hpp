#ifndef __NanoDet_H__
#define __NanoDet_H__

#pragma once

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo_
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct CenterPrior_
{
    int x;
    int y;
    int stride;
} CenterPrior;

class NanoDet {
public:
    NanoDet(const std::string &mnn_path,
            int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.5, float nms_threshold_ = 0.3);

    ~NanoDet();

    int detect(cv::Mat &img, std::vector<BoxInfo> &result_list);
    std::string get_label_str(int label);

    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 80; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

    std::string input_name = "data";
    std::string output_name = "output";

private:
    void decode_infer(MNN::Tensor *pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results);
    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

private:

    std::shared_ptr<MNN::Interpreter> NanoDet_interpreter;
    MNN::Session *NanoDet_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

    int num_thread;
    int image_w;
    int image_h;

    float score_threshold;
    float nms_threshold;

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };

    std::vector<std::string>
    labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush"};
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
inline float sigmoid(float x);

#endif // __NanoDet_H__
