//
// Created by Borzyszkowski
// 2024 / 02 / 29
//

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>
#include <onnxruntime_cxx_api.h>

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDetONNX
{
public:
    NanoDetONNX();
    ~NanoDetONNX();

    static NanoDetONNX* detector;
    Ort::Session* session;

    std::vector<int64_t>* inputDims;
    std::vector<int64_t>* outputDims;

    // modify these parameters according to the model config
    std::string onnx_model_path = "model_path.onnx";
    const int batch_size = 1;
    const int input_size[2] = {192, 192};  // input height and width
    const int num_class = 1;               // number of classes
    const int reg_max = 7;                 // `reg_max` set in the training config. Default: 7.

    const float score_threshold = 0.4;
    const float nms_threshold = 0.5;

    std::vector<int> strides = { 8, 16, 32, 64 };  // strides of the multi-level feature.
    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush" };
    std::vector<BoxInfo> detect(cv::Mat image);
    void preprocess(cv::Mat& image);

private:
    void VectorProduct();
    void decode_infer(std::vector<float>& feats, std::vector<CenterPrior>& center_priors, std::vector<std::vector<BoxInfo>>& results, int64_t num_scores);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
};

#endif //NANODET_H
