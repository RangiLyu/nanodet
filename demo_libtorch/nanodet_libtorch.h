#include <torch/torch.h>
#include <opencv2/core/core.hpp>


typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class NanoDet
{
public:
    NanoDet(const char* model_path);
    ~NanoDet();
    torch::jit::script::Module Net;
    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

private:
    torch::Tensor preprocess(cv::Mat& image);
    void decode_infer(torch::Tensor& cls_pred, torch::Tensor& dis_pred, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    std::vector<int> strides_{ 8, 16, 32, 64 };
    int input_size_ = 416;
    int num_class_ = 80;
    int reg_max_ = 7;

};
