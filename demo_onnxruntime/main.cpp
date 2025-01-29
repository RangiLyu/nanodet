//
// Created by Borzyszkowski
// 2025 / 01 / 29
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include "nanodet.h"

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
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


struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}


const int color_list[1][3] =
{
    {216 , 82 , 24},
};


void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    // cv::imshow("image", image);
    // cv::imwrite("image.jpg", image);
}

int image_demo(NanoDetONNX &detector, const char* imagepath)
{
    std::vector<cv::String> filenames;
    cv::glob(imagepath, filenames, false);
    std::cout << filenames << std::endl;
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    for (auto img_name : filenames)
    {
        cv::Mat image;
        image = cv::imread(img_name);

        if (image.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name);
            return -1;
        }

        object_rect effect_roi;
        cv::Mat resized_img;

        // resize the image to the 192x192 as a cv::Mat with int values in range (0;255)
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        // run the pre-processing
        detector.preprocess(resized_img);

        // run the detection
        auto results = detector.detect(resized_img);

        draw_bboxes(image, results, effect_roi);
        cv::waitKey(0);
    }
    return 0;
}

int webcam_demo(NanoDetONNX &detector, int cam_id)
{
    cv::Mat image;
    cv::VideoCapture cap(cam_id);
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    while (true)
    {
        cap >> image;
        object_rect effect_roi;
        cv::Mat resized_img;

        // resize the image to the 192x192 as a cv::Mat with int values in range (0;255)
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        // run the pre-processing
        detector.preprocess(resized_img);

        // run the detection
        auto results = detector.detect(resized_img);

        draw_bboxes(image, results, effect_roi);
        cv::waitKey(1);
    }
    return 0;
}

int main(int argc, char** argv)
{

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id;\n", argv[0]);
        return -1;
    }

    // Initialize the ONNX Runtime Session with a detector
    NanoDetONNX detector = NanoDetONNX();

    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:{
        int cam_id = atoi(argv[2]);
        std::cout << "Webcam selected!" <<std::endl;
        webcam_demo(detector, cam_id);
        break;
        }
    case 1:{
        const char* images = argv[2];
        std::cout << "Image selected!" <<std::endl;
        image_demo(detector, images);
        break;
        }
    default:{
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n", argv[0]);
        break;
        }
    }
}
