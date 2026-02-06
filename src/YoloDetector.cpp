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
    // --- [階段 4] 載入引擎 ---
    std::ifstream file(mEnginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "錯誤：找不到引擎檔案 " << mEnginePath << std::endl;
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

    // --- [階段 5] GPU 記憶體分配 ---
    const char* inputName = mEngine->getIOTensorName(0);
    const char* outputName = mEngine->getIOTensorName(1);

    // 計算 YOLO26n 所需空間
    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 84 * 8400 * sizeof(float);

    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    mContext->setTensorAddress(inputName, mInputBuffer);
    mContext->setTensorAddress(outputName, mOutputBuffer);

    std::cout << ">>> YoloDetector 初始化成功" << std::endl;
    std::cout << ">>> 硬體已分配：RTX 4060 GPU Memory" << std::endl;
    return true;
}

// --- [階段 6] 預處理實作 ---
std::vector<float> YoloDetector::preprocess(const cv::Mat& img) {
    int w = img.cols;
    int h = img.rows;

    // 1. Letterbox 縮放計算
    float scale = std::min((float)mInputW / w, (float)mInputH / h);
    int newW = (int)(w * scale);
    int newH = (int)(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));

    // 準備灰色畫布
    cv::Mat canvas(mInputH, mInputW, CV_8UC3, cv::Scalar(114, 114, 114));

    // 貼圖
    int xOffset = (mInputW - newW) / 2;
    int yOffset = (mInputH - newH) / 2;
    resized.copyTo(canvas(cv::Rect(xOffset, yOffset, newW, newH)));

    // 2. 轉換為 TensorRT 需要的格式 (HWC -> CHW, BGR -> RGB, 0~255 -> 0~1)
    std::vector<float> data(mInputSize / sizeof(float));

    for (int r = 0; r < mInputH; ++r) {
        for (int c = 0; c < mInputW; ++c) {
            cv::Vec3b pixel = canvas.at<cv::Vec3b>(r, c);

            // 注意這裡的順序：RGB
            data[0 * mInputH * mInputW + r * mInputW + c] = pixel[2] / 255.0f; // R
            data[1 * mInputH * mInputW + r * mInputW + c] = pixel[1] / 255.0f; // G
            data[2 * mInputH * mInputW + r * mInputW + c] = pixel[0] / 255.0f; // B
        }
    }
    return data;
}

void YoloDetector::detect(const std::string& imagePath) {
    // 1. 讀取圖片
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "錯誤：無法讀取圖片 " << imagePath << std::endl;
        return;
    }
    std::cout << "1. 讀圖成功: " << img.cols << "x" << img.rows << std::endl;

    // 2. 預處理
    std::vector<float> inputData = preprocess(img);
    std::cout << "2. 預處理完成 (Letterbox + 正規化)" << std::endl;

    // 3. 搬運到 GPU (Host -> Device)
    cudaError_t status = cudaMemcpy(mInputBuffer, inputData.data(), mInputSize, cudaMemcpyHostToDevice);

    if (status != cudaSuccess) {
        std::cerr << "CUDA Memcpy 錯誤: " << cudaGetErrorString(status) << std::endl;
        return;
    }

    std::cout << "3. 資料已上傳至 GPU (Ready for Inference)" << std::endl;
}
