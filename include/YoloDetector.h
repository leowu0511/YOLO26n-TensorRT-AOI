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

/**
 * TensorRT 系統日誌紀錄器
 */
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

/**
 * 偵測結果數據結構
 */
struct Detection {
    int classId;       // 類別索引
    float confidence;  // 置信度
    cv::Rect box;      // 偵測框座標
    cv::Scalar color;  // 繪製顏色
};

/**
 * YOLO 推論引擎類別 (支援 RTX 4060 硬體加速)
 */
class YoloDetector {
public:
    YoloDetector(const std::string& enginePath);
    ~YoloDetector();

    /**
     * 初始化推論引擎、分配 GPU 顯存與串流資源
     */
    bool init();

    /**
     * 標準推論介面 (Production Mode)：接收 CPU 端影像，包含 H2D 傳輸耗時
     */
    std::vector<Detection> detect(const cv::Mat& img);

    /**
     * 高吞吐推論介面 (Zero-Copy Mode)：直接接收 GPU 端影像，排除傳輸開銷
     */
    std::vector<Detection> detectGpu(const cv::cuda::GpuMat& d_img);

private:
    /**
     * 影像預處理：執行上傳、縮放、色域轉換與正規化
     */
    void preprocessGPU(const cv::Mat& img);

    /**
     * GPU 直接預處理：針對已存在於顯存的影像進行加速處理
     */
    void preprocessGpuDirect(const cv::cuda::GpuMat& d_img);

    /**
     * 後處理：解析輸出 Tensor 並執行非極大值抑制 (NMS)
     */
    std::vector<Detection> postprocess(const float* data, const cv::Size& originalSize);

    // 引擎核心成員
    std::string mEnginePath;
    Logger mLogger;
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // 記憶體緩衝區
    void* mInputBuffer = nullptr;
    void* mOutputBuffer = nullptr;

    // CUDA 運算串流
    cudaStream_t mStream = nullptr;
    cv::cuda::Stream mCvStream;

    size_t mInputSize;
    size_t mOutputSize;

    // 模型定義尺寸
    const int mInputW = 640;
    const int mInputH = 640;

    // 預分配之 GPU 顯存中間緩衝區 (預防 Runtime 記憶體抖動)
    cv::cuda::GpuMat m_d_img;
    cv::cuda::GpuMat m_d_resized;
    cv::cuda::GpuMat m_d_rgb;
    cv::cuda::GpuMat m_d_float;
    std::vector<cv::cuda::GpuMat> m_chw_channels;

    // 主機端緩衝區與 Pinned Memory 標記
    std::vector<float> mOutputHostBuffer;
    float* mPinnedOutputBuffer = nullptr;
    bool mUsePinnedMemory = true;

    // 後處理快取空間
    std::vector<int> mClassIds;
    std::vector<float> mConfidences;
    std::vector<cv::Rect> mBoxes;
    std::vector<int> mNmsIndices;
};
