#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

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
    int classId;      // 類別 ID
    float confidence; // 信心度 (0.0 ~ 1.0)
    cv::Rect box;     // 框框位置 (x, y, w, h)
    cv::Scalar color; // 顯示用的顏色
};

class YoloDetector {
public:
    YoloDetector(const std::string& enginePath);
    ~YoloDetector();

    // 初始化：載入引擎與分配顯存
    bool init();

    /**
     * [重大修改] 核心偵測介面
     * @param img 輸入的 OpenCV 矩陣 (BGR 格式)
     * @return 偵測到的物體清單
     * 說明：此函式現在只負責運算，不包含顯示與存檔。
     */
    std::vector<Detection> detect(const cv::Mat& img);

private:
    // 內部預處理：縮放、正規化、HWC -> CHW
    std::vector<float> preprocess(const cv::Mat& img);

    // 後處理：將 Tensor 轉回 Detection 結構
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

    // CUDA 串流
    cudaStream_t mStream = nullptr;

    // 緩衝區大小
    size_t mInputSize;
    size_t mOutputSize;

    // 模型預設輸入尺寸
    const int mInputW = 640;
    const int mInputH = 640;
};
