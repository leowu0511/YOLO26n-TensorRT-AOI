#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// TensorRT 日誌紀錄器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// 偵測結果結構
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Scalar color;
};

class YoloDetector {
public:
    YoloDetector(const std::string& enginePath);
    ~YoloDetector();

    bool init();
    std::vector<Detection> detect(const cv::Mat& img);

private:
    void preprocessGPU(const cv::Mat& img);
    // 優化：直接傳入 float 指標，減少一次 memcpy
    std::vector<Detection> postprocess(const float* data, const cv::Size& originalSize);

    std::string mEnginePath;
    Logger mLogger;

    // TensorRT 核心指標
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // GPU 記憶體指標
    void* mInputBuffer = nullptr;
    void* mOutputBuffer = nullptr;

    // CUDA 串流
    cudaStream_t mStream = nullptr;
    cv::cuda::Stream mCvStream; // OpenCV CUDA Stream 包裝器

    // 緩衝區大小
    size_t mInputSize;
    size_t mOutputSize;

    // 模型輸入尺寸
    const int mInputW = 640;
    const int mInputH = 640;

    // ========== 🚀 終極優化：預分配空間 ==========

    // [1] GPU 預處理中間變數 (避免每次重新 malloc 顯存)
    cv::cuda::GpuMat m_d_img;
    cv::cuda::GpuMat m_d_resized;
    cv::cuda::GpuMat m_d_rgb;
    cv::cuda::GpuMat m_d_float;
    std::vector<cv::cuda::GpuMat> m_chw_channels;

    // [2] Host 端緩衝
    std::vector<float> mOutputHostBuffer; // 一般記憶體備援
    float* mPinnedOutputBuffer = nullptr; // Pinned Memory (加速 PCIe)
    bool mUsePinnedMemory = true;

    // [3] NMS 中間變數快取 (避免 NMS 時反覆分配 vector)
    std::vector<int> mClassIds;
    std::vector<float> mConfidences;
    std::vector<cv::Rect> mBoxes;
    std::vector<int> mNmsIndices;
};
