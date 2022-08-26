#include <torch/script.h>
#include <iostream>
#include "nanodet_libtorch.h"


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

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}


NanoDet::NanoDet(const char* model_path)
{
    std::cout<<"load model start"<<std::endl;
    this->Net = torch::jit::load(model_path);
    this->Net.eval();
    std::cout<<"load model finished"<<std::endl;
    // this->Net->to(at::kCUDA);
}

NanoDet::~NanoDet()
{
}

torch::Tensor NanoDet::preprocess(cv::Mat& image)
{
    int img_w = image.cols;
    int img_h = image.rows;
    torch::Tensor tensor_image = torch::from_blob(image.data, {1,img_h, img_w,3}, torch::kByte);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
    // TODO: mean std per channel
    tensor_image = tensor_image.add(-116.28f);
    tensor_image = tensor_image.mul(0.017429f);
    return tensor_image;
}

std::vector<BoxInfo> NanoDet::detect(cv::Mat image, float score_threshold, float nms_threshold)
{
    auto input = preprocess(image);
    auto outputs = this->Net.forward({input}).toTensor();

    torch::Tensor cls_preds = outputs.index({ "...",torch::indexing::Slice(0,this->num_class_) });
    torch::Tensor box_preds = outputs.index({ "...",torch::indexing::Slice(this->num_class_ , torch::indexing::None) });

    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class_);

    this->decode_infer(cls_preds, box_preds, score_threshold, results);

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++)
    {
        this->nms(results[i], nms_threshold);

        for (auto box : results[i])
        {
            dets.push_back(box);
        }
    }
    return dets;
}

void NanoDet::decode_infer(torch::Tensor& cls_pred, torch::Tensor& dis_pred, float threshold, std::vector<std::vector<BoxInfo>>& results)
{
    int total_idx = 0;
    for (int stage_idx = 0; stage_idx < (int)strides_.size(); stage_idx++)
    {
        int stride = this->strides_[stage_idx];
        int feature_h = ceil(double(this->input_size_) / stride);
        int feature_w = ceil(double(this->input_size_) / stride);
        // cv::Mat debug_heatmap = cv::Mat::zeros(feature_h, feature_w, CV_8UC3);

        for (int idx = total_idx; idx < feature_h * feature_w + total_idx; idx++)
        {
            int row = (idx - total_idx) / feature_w;
            int col = (idx - total_idx) % feature_w;
            float score = -0.0f;
            int cur_label = 0;
            for (int label = 0; label < this->num_class_; label++)
            {
                float cur_score = cls_pred[0][idx][label].item<float>();
                if (cur_score > score)
                {
                    score = cur_score;
                    cur_label = label;
                }
            }
            if (score > threshold)
            {
                //std::cout << "label:" << cur_label << " score:" << score << std::endl;
                auto cur_dis = dis_pred[0][idx].contiguous();
                const float* bbox_pred = cur_dis.data<float>();
                results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
                // debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
                // cv::imshow("debug", debug_heatmap);
            }
        }
        total_idx += feature_h * feature_w;
    }
    // cv::waitKey(0);
}

BoxInfo NanoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[this->reg_max_ + 1];
        activation_function_softmax(dfl_det + i * (this->reg_max_ + 1), dis_after_sm, this->reg_max_ + 1);
        for (int j = 0; j < this->reg_max_ + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size_);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size_);

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void NanoDet::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
