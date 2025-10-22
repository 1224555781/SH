#include <windows.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include "CaptureHead.h"
#include "../MouseHelp.h"
#include <opencv2/opencv.hpp>

cv::Mat MLB::CaptureHead::CaptureScreenMat()
{
    FScopeTime scope{ "CaptureScreenMat" };
    HDC hScreenDC = GetDC(NULL);
    HDC hMemDC = CreateCompatibleDC(hScreenDC);

    int width = GetSystemMetrics(SM_CXSCREEN);
    int height = GetSystemMetrics(SM_CYSCREEN);

    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
    HGDIOBJ oldBitmap = SelectObject(hMemDC, hBitmap);

    if (!BitBlt(hMemDC, 0, 0, width, height, hScreenDC, 0, 0, SRCCOPY | CAPTUREBLT)) {
        SelectObject(hMemDC, oldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);
        return cv::Mat();
    }

    BITMAPINFOHEADER bi;
    ZeroMemory(&bi, sizeof(bi));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height; // top-down
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    cv::Mat mat(height, width, CV_8UC3);
    if (!GetDIBits(hMemDC, hBitmap, 0, height, mat.data, reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS)) {
        SelectObject(hMemDC, oldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);
        return cv::Mat();
    }

    SelectObject(hMemDC, oldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemDC);
    ReleaseDC(NULL, hScreenDC);

    // 保存截图到 S:\\test.png（每次捕获都重复写入）
    if (!mat.empty()) {
        try {
            cv::imwrite("S:\\test.png", mat);
        }
        catch (const cv::Exception& e) {
            std::cerr << "Failed to save screenshot: " << e.what() << "\n";
        }
    }

    return mat; // BGR
}

// Letterbox (resize while keeping aspect ratio), returns resized image and sets scale/pad for mapping back
cv::Mat MLB::CaptureHead::Letterbox(const cv::Mat& src, const cv::Size& newShape, cv::Scalar color,
    double& scale, int& padW, int& padH)
{
    FScopeTime scope{ "Letterbox" };
    int srcW = src.cols, srcH = src.rows;
    double r = std::min((double)newShape.width / srcW, (double)newShape.height / srcH);
    int newUnpadW = (int)std::round(srcW * r);
    int newUnpadH = (int)std::round(srcH * r);
    padW = (newShape.width - newUnpadW) / 2;
    padH = (newShape.height - newUnpadH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newUnpadW, newUnpadH));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, padH, newShape.height - newUnpadH - padH,
        padW, newShape.width - newUnpadW - padW,
        cv::BORDER_CONSTANT, color);
    scale = r;
    return out;
}


// Parse ONNX outputs. Supported formats:
// 1) [1, N, 6] or [N, 6] with rows = [x1, y1, x2, y2, score, classId]
// 2) YOLOv8 raw: [1, C, N] or [1, N, C] where C = 4 + num_classes, row = [cx, cy, w, h, class_scores...]
// 3) YOLOv5-style [1, N, 85] or [N,85] where row = [cx,cy,w,h,obj_conf,class_scores...]
std::vector<MLB::Detection> MLB::CaptureHead::ParseYoloOutputs(const cv::Mat& outs, float confThreshold, int inputW, int inputH, double scale, int padW, int padH, int origW, int origH, int targetClassId )
{
    
    FScopeTime time{ "ParseYoloOutputs" };
    
    std::vector<Detection> dets;

    // Case YOLOv8 raw outputs: 3D tensor
    if (outs.dims == 3 && outs.size[0] == 1) {
        // We accept either [1, C, N] or [1, N, C]
        const int dim1 = outs.size[1];
        const int dim2 = outs.size[2];
        const bool channelFirst = (dim1 <= dim2); // heuristic: when C(=84) <= N(=8400)
        int C = channelFirst ? dim1 : dim2;
        int N = channelFirst ? dim2 : dim1;
        if (C >= 5 && N > 0) {
            for (int i = 0; i < N; ++i) {
                // read bbox center in pixels
                float cx, cy, w, h;
                if (channelFirst) {
                    cx = outs.at<float>(0, 0, i);
                    cy = outs.at<float>(0, 1, i);
                    w  = outs.at<float>(0, 2, i);
                    h  = outs.at<float>(0, 3, i);
                } else {
                    cx = outs.at<float>(0, i, 0);
                    cy = outs.at<float>(0, i, 1);
                    w  = outs.at<float>(0, i, 2);
                    h  = outs.at<float>(0, i, 3);
                }

                // best class score (no objectness in YOLOv8)
                float bestClassScore = 0.f;
                int bestClass = -1;
                for (int c = 4; c < C; ++c) {
                    float cs = channelFirst ? outs.at<float>(0, c, i) : outs.at<float>(0, i, c);
                    if (cs > bestClassScore) { bestClassScore = cs; bestClass = c - 4; }
                }
                float score = bestClassScore;
                if (score <= confThreshold) continue;
                if (targetClassId >= 0 && bestClass != targetClassId) continue;

                // Convert from cx,cy,w,h (pixels in letterboxed space) to original image
                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;

                // remove letterbox padding and scale back
                x1 = (x1 - padW) / scale;
                x2 = (x2 - padW) / scale;
                y1 = (y1 - padH) / scale;
                y2 = (y2 - padH) / scale;

                int ix1 = std::max(0, std::min(origW - 1, (int)std::round(x1)));
                int iy1 = std::max(0, std::min(origH - 1, (int)std::round(y1)));
                int ix2 = std::max(0, std::min(origW - 1, (int)std::round(x2)));
                int iy2 = std::max(0, std::min(origH - 1, (int)std::round(y2)));

                if (ix2 > ix1 && iy2 > iy1) {
                    cv::Rect r(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
                    dets.push_back({ r, score, bestClass });
                }
            }
            return dets;
        }
    }

    // Flatten other outputs to [N, C]
    cv::Mat outMat = outs.reshape(1, (int)(outs.total() / outs.size[outs.dims - 1]));
    int rows = outMat.rows;
    int cols = outMat.cols;

    // Case NMS embedded: [x1,y1,x2,y2,score,classId]
    if (cols == 6) {
        for (int i = 0; i < rows; ++i) {
            const float* row = outMat.ptr<float>(i);
            float x1 = row[0];
            float y1 = row[1];
            float x2 = row[2];
            float y2 = row[3];
            float score = row[4];
            int cls = (int)std::round(row[5]);

            if (score <= confThreshold) continue;
            if (targetClassId >= 0 && cls != targetClassId) continue;

            x1 = (x1 - padW) / scale;
            x2 = (x2 - padW) / scale;
            y1 = (y1 - padH) / scale;
            y2 = (y2 - padH) / scale;

            int ix1 = std::max(0, std::min(origW - 1, (int)std::round(x1)));
            int iy1 = std::max(0, std::min(origH - 1, (int)std::round(y1)));
            int ix2 = std::max(0, std::min(origW - 1, (int)std::round(x2)));
            int iy2 = std::max(0, std::min(origH - 1, (int)std::round(y2)));

            if (ix2 > ix1 && iy2 > iy1) {
                cv::Rect r(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
                dets.push_back({ r, score, cls });
            }
        }
        return dets;
    }

    return dets;
}
