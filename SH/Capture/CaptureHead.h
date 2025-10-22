#pragma once

#include <opencv2/core/mat.hpp>

namespace MLB
{
    struct Detection {
        cv::Rect box;
        float conf;
        int class_id;
    };

    class CaptureHead
    {
    public:
        static  cv::Mat CaptureScreenMat();
        static std::vector<Detection> ParseYoloOutputs(const cv::Mat& outs, float confThreshold, int inputW, int inputH, double scale, int padW, int padH, int origW, int origH, int targetClassId = 0);
        static cv::Mat Letterbox(const cv::Mat& src, const cv::Size& newShape, cv::Scalar color,double& scale, int& padW, int& padH);
    };
}
