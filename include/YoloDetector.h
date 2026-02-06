#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp> // 💡 新增：支援 GPU 矩陣操作

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

    // 初始化：載入引擎與分配顯存
    bool init();

    /**
     * [GPU 優化版] 核心偵測介面
     * @param img 輸入的 OpenCV 矩陣 (BGR 格式)
     * @return 偵測到的物體清單
     */
    std::vector<Detection> detect(const cv::Mat& img);

private:
    /**
     * 💡 [重大修改] 內部 GPU 預處理
     * 直接將處理後的資料寫入 mInputBuffer，不再透過 CPU 中轉，節省大量記憶體拷貝時間。
     */
    void preprocessGPU(const cv::Mat& img);

    // 後處理：將 Tensor 轉回 Detection 結構
    std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Size& originalSize);

    std::string mEnginePath;
    Logger mLogger;

    // TensorRT 核心指標
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // GPU 記憶體指標 (顯存位址)
    void* mInputBuffer = nullptr;
    void* mOutputBuffer = nullptr;

    // CUDA 串流
    cudaStream_t mStream = nullptr;

    // 緩衝區大小
    size_t mInputSize;
    size_t mOutputSize;

    // 模型預設輸入尺寸 (YOLO26n 標準)
    const int mInputW = 640;
    const int mInputH = 640;
};
