//
// Created by Borzyszkowski
// 2025 / 01 / 29
//

#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>

#include "nanodet.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

BoxInfo NanoDetONNX::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    int reg_max = 7;
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        // std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], 192.0f);
    float ymax = (std::min)(ct_y + dis_pred[3], 192.0f);
    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void NanoDetONNX::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}

NanoDetONNX* NanoDetONNX::detector = nullptr;
NanoDetONNX::NanoDetONNX()
{
    // specify the ONNX model location
    std::string modelFilepath{this->onnx_model_path};

    // define the ONNX Runtime Session Options
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    std::string instanceName{"ONNX-demo"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    // Load the model to session
    this->session = new Ort::Session(env, modelFilepath.c_str(), sessionOptions);
    std::cout << "Loaded the model to the ONNX Session!" <<std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = this->session->GetInputCount();
    size_t numOutputNodes = this->session->GetOutputCount();
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    // TODO: Input and output names are hardcoded. They can be checked in Netron. Ultimately they should be parametizable.
    const auto inputName = "data";  //session.GetInputName(0, allocator);
    const auto outputName = "output";  //session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo inputTypeInfo = this->session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputShape = inputTensorInfo.GetShape();
    this->inputDims = new std::vector<int64_t>(inputShape);
    std::cout << "Input Dimensions: " << *inputDims << std::endl;
    int width = inputShape[2];
    int height = inputShape[3];

    Ort::TypeInfo outputTypeInfo = this->session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = outputTensorInfo.GetShape();
    this->outputDims = new std::vector<int64_t>(outputShape);
    std::cout << "Output Dimensions: " << *outputDims << std::endl;
    const int64_t scores_n = outputShape[2];

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
}

NanoDetONNX::~NanoDetONNX()
{
    delete this->session;
}

void NanoDetONNX::preprocess(cv::Mat& image)
{
    // Convert the data type of the image to CV_32F
    image.convertTo(image, CV_32F);

    // Define mean and normalization values exactly as in the training code
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 57.375f, 57.12f, 58.395f };

    // Split the channels
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Substract mean and normalize each channel
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_vals[i]) / norm_vals[i];
    }

    // Merge the channels back into the original image
    cv::merge(channels, image);

    // HWC to CHW
    cv::dnn::blobFromImage(image, image);
}

std::vector<BoxInfo> NanoDetONNX::detect(cv::Mat image)
{
    // TODO: Input and output names are hardcoded. They can be checked in Netron. Ultimately they should be parametizable.
    const auto inputName = "data";     //session.GetInputName(0, allocator);
    const auto outputName = "output";  //session.GetInputName(0, allocator);

    size_t inputTensorSize = vectorProduct(*inputDims);
    size_t outputTensorSize = vectorProduct(*outputDims);
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<float> inputTensorValues(inputTensorSize);
    std::vector<float> outputTensorValues(outputTensorSize);

    // Make copies of the same image input, depending on the batch size
    for (int64_t i = 0; i < this->batch_size; ++i)
    {
        std::copy(image.begin<float>(),
                    image.end<float>(),
                    inputTensorValues.begin() + i * inputTensorSize / this->batch_size);
    }

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, 
        inputTensorValues.data(), 
        inputTensorSize, 
        this->inputDims->data(),
        this->inputDims->size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, 
        outputTensorValues.data(), 
        outputTensorSize,
        this->outputDims->data(), 
        this->outputDims->size()));

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    this->session->Run(Ort::RunOptions{nullptr}, 
        inputNames.data(),
        inputTensors.data(), 1  /*Number of inputs*/, 
        outputNames.data(),
        outputTensors.data(), 1 /*Number of outputs*/);

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(this->input_size[0], this->input_size[1], strides, center_priors);

    // define the vector for results
    std::vector<std::vector<BoxInfo>> results;
    results.resize(1);

    // number of scores to readout is equivalent to the model output dimension
    int64_t scores_n = (*outputDims)[2];
    decode_infer(outputTensorValues, center_priors, results, scores_n);

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++)
    {
        nms(results[i], this->nms_threshold);

        for (auto box : results[i])
        {
            dets.push_back(box);
        }
    }
    return dets;
}

void NanoDetONNX::decode_infer(std::vector<float>& feats, std::vector<CenterPrior>& center_priors, std::vector<std::vector<BoxInfo>>& results, int64_t num_scores)
{
    const int num_points = center_priors.size();
    printf("num_points:%d\n", num_points);

    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        // read the scores at the desired position
        const float* scores = &feats[(idx) * num_scores];
      
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < this->num_class; label++)
        {
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > this->score_threshold)
        {
            const float* bbox_pred = scores + this->num_class;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
        }
    }
}
