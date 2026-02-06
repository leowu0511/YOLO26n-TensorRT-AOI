#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm> // 為了 std::min
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp> // OpenCV 支援

// TensorRT 日誌紀錄器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};
// [新增] 1. 定義偵測結果結構
struct Detection {
    int classId;      // 類別 ID (0:人, 1:腳踏車...)
    float confidence; // 信心度 (0.0 ~ 1.0)
    cv::Rect box;     // 框框位置 (x, y, w, h)
    cv::Scalar color; // (選做) 每個類別給不同顏色
};
class YoloDetector {
public:
    YoloDetector(const std::string& enginePath);
    ~YoloDetector();

    // 整合階段 4 & 5
    bool init();

    // [階段 6] 外部呼叫的偵測介面
    void detect(const std::string& imagePath);

private:
    // [階段 6] 內部預處理：縮放、正規化、HWC -> CHW
    std::vector<float> preprocess(const cv::Mat& img);
    
    // [新增] 2. 後處理函式宣告
    std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Size& originalSize);

    std::string mEnginePath;
    Logger mLogger;

    // TensorRT 核心指標
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // GPU 記憶體指標
    void* mInputBuffer = nullptr;
    void* mOutputBuffer = nullptr;

    // CUDA 串流：GPU 的任務排隊通道
    cudaStream_t mStream = nullptr;

    // 緩衝區大小
    size_t mInputSize;
    size_t mOutputSize;

    // 模型輸入尺寸 (YOLO26n 預設)
    const int mInputW = 640;
    const int mInputH = 640;
};
