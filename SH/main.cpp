#include <chrono>
#include <filesystem>
#include <iostream>
#include <format>
#include <thread>
#include <windows.h>
#include "Capture/CaptureHead.h"
#include "gbil.h"
#include <opencv2/opencv.hpp>
#include "MouseHelp.h"
#include "Config.h"


int main()
{
    const std::string modelPath = "yolo_head.onnx";
    auto open_stat = std::format("open: {}", opendevice(0));
    namespace fs = std::filesystem;
    if (!fs::exists(modelPath)) {
        std::cerr << "ONNX model file not found at: " << modelPath << "\n";
        std::cerr << "Current working directory: " << fs::current_path().string() << "\n";
        return -1;
    }

    auto fileSize = fs::file_size(modelPath);
    std::cout << "Model file size: " << fileSize / 1024 << " KB\n";
    if (fileSize < 1024) {
        std::cerr << "Warning: ONNX file seems too small, may be corrupted.\n";
    }
    cv::dnn::Net net;
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Failed to load model at " << modelPath << "\n";
        return -1;
    }

    try {
#ifdef CV__VERSION_MAJOR
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
#else
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    }
    catch (...) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::cout << "Starting screen capture + detection. Press Esc to stop.\n";

    // Create display window for overlay
    cv::namedWindow("Detections", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detections", 960, 540);

    bool printedShape = false;

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        constexpr int inputSize = 640;
        constexpr float confThreshold = 0.35f;
        constexpr float nmsThreshold = 0.45f;
        constexpr int personClassId = 0; // COCO person
        // If ONNX is a head detector, set this to true and set headClassId accordingly.
        constexpr bool modelOutputsHeads = true;
        constexpr int headClassId = 0; // head class id in your model (0 for single-class head models)
        // When using person detector only, estimate head from the top portion:
        constexpr bool estimateHeadFromPerson = true; // use top of person box to estimate head
        constexpr float headTopRatio = 0.15f; // head height as a fraction of person bbox height
        constexpr float headWidthRatio = 0.6f; // narrow head box width around center (0~1)
      

        // If you want continuous detection, remove this click gate
        if ((GetAsyncKeyState(VK_LBUTTON) & 0x0001) == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        cv::Mat frame = MLB::CaptureHead::CaptureScreenMat();
        if (frame.empty()) {
            std::cerr << "Capture failed.\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        int origW = frame.cols, origH = frame.rows;

        double scale = 1.0;
        int padW = 0, padH = 0;
        cv::Mat blobImg;
        cv::Mat blob;
         blobImg = MLB::CaptureHead::Letterbox(frame, cv::Size(inputSize, inputSize), cv::Scalar(114, 114, 114), scale, padW, padH);
        {
             FScopeTime st("blobFromImage");
             blob = cv::dnn::blobFromImage(blobImg, 1.0f / 255.0f, cv::Size(inputSize, inputSize), cv::Scalar(), true, false);
        }
        cv::Mat outputs;
        {
            FScopeTime st("net.forward");
            net.setInput(blob);
            outputs = net.forward();
        }


        if (!printedShape) {
            std::cout << "DNN output dims=" << outputs.dims << " sizes:";
            for (int d = 0; d < outputs.dims; ++d) std::cout << " " << outputs.size[d];
            std::cout << " total=" << outputs.total() << "\n";
            printedShape = true;
        }

        // -1: keep all, we'll filter below for head/person logic
        std::vector<MLB::Detection> dets = MLB::CaptureHead::ParseYoloOutputs(
            outputs, confThreshold, inputSize, inputSize, scale, padW, padH, origW, origH, -1);

        // Apply NMS
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;
        boxes.reserve(dets.size()); scores.reserve(dets.size()); classIds.reserve(dets.size());
        for (auto& d : dets) { boxes.push_back(d.box); scores.push_back(d.conf); classIds.push_back(d.class_id); }
        std::vector<int> idxs;
        if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, idxs);

        // Draw results on frame and show
        cv::Mat vis = frame.clone();

        // Track best head region (either direct head detection or estimated from person)
        float bestHeadScore = 0.f;
        cv::Rect bestHeadRect;

        for (int id : idxs) {
            const auto& b = boxes[id];
            const float sc = scores[id];
            const int cls = classIds[id];

            // 先画原始检测框
            cv::rectangle(vis, b, cv::Scalar(0, 255, 0), 2);
            std::string label = std::format("c{} {:.2f}", cls, sc);
            int base = 0; cv::Size t = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
            cv::rectangle(vis, {b.x, std::max(0, b.y - t.height - 4)}, {b.x + t.width + 4, b.y}, cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(vis, label, {b.x + 2, std::max(0, b.y - 2)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1);

            // 直接“头部”类别（若模型直接输出头部）
            if (modelOutputsHeads && cls == headClassId) {
                if (sc > bestHeadScore) {
                    bestHeadScore = sc;
                    bestHeadRect = b;
                }
            }

            // 通过人形框顶部估计头部
            if (estimateHeadFromPerson) {
                int hH = std::max(1, static_cast<int>(b.height * headTopRatio));
                int hW = std::max(1, static_cast<int>(b.width * headWidthRatio));
                int hx = b.x + (b.width - hW) / 2;
                int hy = b.y;
                cv::Rect headRect(hx, hy, hW, hH);
                headRect &= cv::Rect(0, 0, origW, origH); // clamp
                // choose the highest score person as best head proxy
                if (sc > bestHeadScore && headRect.area() > 0) {
                    bestHeadScore = sc;
                    bestHeadRect = headRect;
                }
                // Optional: draw estimated head for each person
                //cv::rectangle(vis, headRect, cv::Scalar(255, 128, 0), 2);
            }
        }
        if (bestHeadScore > 0.f && bestHeadRect.area() > 0) {
            cv::rectangle(vis, bestHeadRect, cv::Scalar(255, 0, 0), 2);
            cv::Point headCenter(bestHeadRect.x + bestHeadRect.width / 2,
                                 bestHeadRect.y + bestHeadRect.height / 2);
           cv::drawMarker(vis, headCenter, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2);
            auto info = std::format("head {:.2f} ({},{} {}x{})",
                                    bestHeadScore, bestHeadRect.x, bestHeadRect.y,
                                    bestHeadRect.width, bestHeadRect.height);
           cv::putText(vis, info, {std::max(0, bestHeadRect.x), std::max(0, bestHeadRect.y - 6)},cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 255}, 1);

        }

        cv::imshow("Detections", vis);
        int k = cv::waitKey(1);
        if (k == 27) break;

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        std::cout << "cost time" << elapsedMs << "\n";
        int sleepMs = std::max(1, (int)(MLB::Frame - static_cast<int>(elapsedMs)));

        //
        std::cout << ismousebuttonpressed(3) << "\n";
        bool is_right_button_press = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        if (isconnected() && is_right_button_press)
        {
            if (bestHeadScore > 0.f && bestHeadRect.area() > 0) {
                POINT headCenter = MouseHelp::ToScreenCenter(bestHeadRect);
                RECT headScreenRect = MouseHelp::ToScreenRect(bestHeadRect);
                setmousemovementmode(2);
                movemouseto(headCenter.x, headCenter.y);

            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }

    std::cout << "\nStopped.\n";
    return 0;
}