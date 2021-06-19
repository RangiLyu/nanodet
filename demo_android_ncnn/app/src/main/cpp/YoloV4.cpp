#include "YoloV4.h"

bool YoloV4::hasGPU = true;
YoloV4 *YoloV4::detector = nullptr;

YoloV4::YoloV4(AAssetManager *mgr, const char *param, const char *bin, bool useGPU) {
    Net = new ncnn::Net();
    // opt 需要在加载前设置
    hasGPU = ncnn::get_gpu_count() > 0;
    Net->opt.use_vulkan_compute = hasGPU && useGPU;  // gpu
    Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}

YoloV4::~YoloV4() {
    delete Net;
}

std::vector<BoxInfo> YoloV4::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, input_size,
                                                             input_size);
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    hasGPU = ncnn::get_gpu_count() > 0;
    ex.set_vulkan_compute(hasGPU);
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    ncnn::Mat blob;
    ex.extract("output", blob);
    auto boxes = decode_infer(blob, {(int) img_size.width, (int) img_size.height}, input_size, num_class, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());
//    nms(result,nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
YoloV4::decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold) {
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
        box.x1 = values[2] * (float) frame_size.width;
        box.y1 = values[3] * (float) frame_size.height;
        box.x2 = values[4] * (float) frame_size.width;
        box.y2 = values[5] * (float) frame_size.height;
        result.push_back(box);
    }
    return result;
}
