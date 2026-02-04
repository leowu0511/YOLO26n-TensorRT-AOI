#include "YoloDetector.h"

YoloDetector::YoloDetector(const std::string& enginePath) : mEnginePath(enginePath) {}

YoloDetector::~YoloDetector() {
    // 釋放 GPU 顯存
    if (mInputBuffer) cudaFree(mInputBuffer);
    if (mOutputBuffer) cudaFree(mOutputBuffer);

    // TensorRT 10.x 使用標準 delete 釋放物件
    if (mContext) delete mContext;
    if (mEngine) delete mEngine;
    if (mRuntime) delete mRuntime;
}

bool YoloDetector::init() {
    // step4
    std::ifstream file(mEnginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "找不到引擎檔案 " << mEnginePath << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    mRuntime = nvinfer1::createInferRuntime(mLogger);
    if (!mRuntime) return false;

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!mEngine) return false;

    mContext = mEngine->createExecutionContext();
    if (!mContext) return false;

    // step 5 GPU 記憶體分配 ---
    // 取得輸入輸出 Tensor 名稱
    const char* inputName = mEngine->getIOTensorName(0);
    const char* outputName = mEngine->getIOTensorName(1);

    // 計算 YOLO26n 所需空間 (1x3x640x640 & 1x84x8400)
    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 84 * 8400 * sizeof(float);

    // 分配顯存
    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    // 建立 Tensor 地址連結 (綁定 GPU 地址與模型的 Port)
    mContext->setTensorAddress(inputName, mInputBuffer);
    mContext->setTensorAddress(outputName, mOutputBuffer);

    std::cout << ">>> YoloDetector 初始化成功" << std::endl;
    std::cout << ">>> 硬體已分配：RTX 4060 GPU Memory" << std::endl;
    return true;
}