#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// TensorRT 日誌紀錄器->報錯
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloDetector {
public:
    YoloDetector(const std::string& enginePath);
    ~YoloDetector();

    // 整合 4 & 5
    bool init();

private:
    std::string mEnginePath;
    Logger mLogger;

    // TensorRT 核心指標 (10.x 用 delete 釋放)
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    // GPU 記憶體指標 (Step 5 核心)
    void* mInputBuffer = nullptr;
    void* mOutputBuffer = nullptr;

    // 緩衝區大小
    size_t mInputSize;
    size_t mOutputSize;
};
