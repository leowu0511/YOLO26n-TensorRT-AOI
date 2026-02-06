#include "YoloDetector.h"

YoloDetector::YoloDetector(const std::string& enginePath) : mEnginePath(enginePath) {}

YoloDetector::~YoloDetector() {
    // 釋放 GPU 顯存
    if (mInputBuffer) cudaFree(mInputBuffer);
    if (mOutputBuffer) cudaFree(mOutputBuffer);
    
    // [新增] 銷毀 Stream
    if (mStream) cudaStreamDestroy(mStream);

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

    // [新增] 建立 CUDA Stream
    if (cudaStreamCreate(&mStream) != cudaSuccess) {
        std::cerr << "CUDA Stream 建立失敗" << std::endl;
        return false;
    }
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
    std::cout << "2. 預處理完成" << std::endl;

    // 3. 上傳資料 (Host -> Device) [使用非同步 + Stream]
    cudaMemcpyAsync(mInputBuffer, inputData.data(), mInputSize, cudaMemcpyHostToDevice, mStream);

    // 4. [新增] 執行推論 (Inference)
    // 這行指令就是讓 RTX 4060 開始運算的關鍵！
    bool status = mContext->enqueueV3(mStream);
    if (!status) {
        std::cerr << "推論執行失敗！" << std::endl;
        return;
    }

    // 5. [新增] 取回結果 (Device -> Host)
    // 準備一個 CPU 陣列來接結果
    std::vector<float> outputData(mOutputSize / sizeof(float));

    // 把 GPU 算好的結果搬回來
    cudaMemcpyAsync(outputData.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);

    // 6. [新增] 等待 GPU 打卡下班
    // 因為上面都是 Async，這裡要強制等待所有任務完成
    cudaStreamSynchronize(mStream);

    std::cout << "3. 推論完成 (Inference Done)" << std::endl;

    // --- 驗證一下有沒有拿到數據 ---
    std::cout << ">>> 輸出 Tensor 前 10 個值: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << "\n>>> 總資料量: " << outputData.size() << std::endl;
}
