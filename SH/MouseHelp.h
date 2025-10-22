#pragma once

#include <algorithm> // add to your includes
#include <windows.h>
#include <opencv2/core/types.hpp>
#include "gbil.h"

struct FScopeTime
{
    FScopeTime(const char* InScopeName)
        : ScopeName(InScopeName)
    {
        StartTime = std::chrono::high_resolution_clock::now();
    }
    ~FScopeTime()
    {
        EndTime = std::chrono::high_resolution_clock::now();
        double elapsedMs = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count());
        std::cout << ScopeName << " cost time: " << elapsedMs << " ms\n";
    }
    const char* ScopeName;
    std::chrono::high_resolution_clock::time_point StartTime;
    std::chrono::high_resolution_clock::time_point EndTime;
};

class MouseHelp
{
public:
    // Convert a cv::Rect (in captured frame) to a Win32 screen RECT
    static inline RECT ToScreenRect(const cv::Rect& r)
    {
        const int scrW = GetSystemMetrics(SM_CXSCREEN);
        const int scrH = GetSystemMetrics(SM_CYSCREEN);
        const int left = std::clamp(r.x, 0, std::max(0, scrW - 1));
        const int top = std::clamp(r.y, 0, std::max(0, scrH - 1));
        const int right = std::clamp(r.x + r.width, 0, scrW);
        const int bottom = std::clamp(r.y + r.height, 0, scrH);
        return RECT{ left, top, right, bottom };
    }

    // Get the screen center point of the rect
    static inline POINT ToScreenCenter(const cv::Rect& r)
    {
        const int scrW = GetSystemMetrics(SM_CXSCREEN);
        const int scrH = GetSystemMetrics(SM_CYSCREEN);
        const int cx = std::clamp(r.x + r.width / 2, 0, std::max(0, scrW - 1));
        const int cy = std::clamp(r.y + r.height / 2, 0, std::max(0, scrH - 1));
        return POINT{ cx, cy };
    }
};
