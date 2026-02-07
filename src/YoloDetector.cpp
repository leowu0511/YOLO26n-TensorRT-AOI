#include "YoloDetector.h"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

/**
 * 建構子：初始化成員變數並預分配 NMS 工作緩衝區
 */
YoloDetector::YoloDetector(const std::string& enginePath) : mEnginePath(enginePath) {
    // 預分配 NMS 空間以避免推論過程中的記憶體抖動
    mClassIds.reserve(1000);
    mConfidences.reserve(1000);
    mBoxes.reserve(1000);
    mNmsIndices.reserve(1000);
    mOutputHostBuffer.resize(50400); // 針對單 batch 輸出 (1 * 6 * 8400)
}

/**
 * 解構子：確保所有 CUDA 資源與 TensorRT 物件正確釋放
 */
YoloDetector::~YoloDetector() {
    if (mInputBuffer) cudaFree(mInputBuffer);
    if (mOutputBuffer) cudaFree(mOutputBuffer);
    if (mPinnedOutputBuffer) cudaFreeHost(mPinnedOutputBuffer);
    if (mStream) cudaStreamDestroy(mStream);

    if (mContext) delete mContext;
    if (mEngine) delete mEngine;
    if (mRuntime) delete mRuntime;
}

/**
 * 初始化引擎：載入模型、分配顯存與配置 Tensor 地址
 */
bool YoloDetector::init() {
    std::ifstream file(mEnginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[Error] Cannot find engine file: " << mEnginePath << std::endl;
        return false;
    }

    // 讀取模型二進位資料
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 建立 TensorRT Runtime 與 Engine
    mRuntime = nvinfer1::createInferRuntime(mLogger);
    if (!mRuntime) { std::cerr << "[Error] Failed to create Runtime"; return false; }

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!mEngine) { std::cerr << "[Error] Failed to deserialize Engine"; return false; }

    mContext = mEngine->createExecutionContext();
    if (!mContext) { std::cerr << "[Error] Failed to create Context"; return false; }

    // 初始化 CUDA 串流
    if (cudaStreamCreate(&mStream) != cudaSuccess) return false;
    mCvStream = cv::cuda::StreamAccessor::wrapStream(mStream);

    // 計算緩衝區大小
    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 6 * 8400 * sizeof(float);

    // 分配 GPU 顯存
    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    // 分配 Pinned Memory 以加速 PCIe 數據傳輸 (H2D/D2H)
    if (cudaMallocHost(&mPinnedOutputBuffer, mOutputSize) != cudaSuccess) {
        std::cerr << "[Warning] Pinned Memory allocation failed, using standard host memory." << std::endl;
        mUsePinnedMemory = false;
    }

    // 💡 針對 TensorRT 10 C++ API：綁定 Tensor 地址
    // 雖然 Python API 已更新，但在 C++ ICudaEngine 中 getIOTensorName 仍為標準用法
    mContext->setTensorAddress(mEngine->getIOTensorName(0), mInputBuffer);
    mContext->setTensorAddress(mEngine->getIOTensorName(1), mOutputBuffer);

    // 預分配 GPU 預處理中間緩衝區，確保零 malloc 延遲
    m_d_img.create(mInputH, mInputW, CV_8UC3);
    m_d_resized.create(mInputH, mInputW, CV_8UC3);
    m_d_rgb.create(mInputH, mInputW, CV_8UC3);
    m_d_float.create(mInputH, mInputW, CV_32FC3);

    // 配置影像平面 (CHW 格式)
    m_chw_channels.clear();
    for (int i = 0; i < 3; ++i) {
        m_chw_channels.push_back(cv::cuda::GpuMat(mInputH, mInputW, CV_32FC1,
            (float*)mInputBuffer + i * mInputW * mInputH));
    }

    std::cout << "[System] YoloDetector Engine verified and ready [RTX 4060]" << std::endl;
    return true;
}

/**
 * 實戰模式：接收 CPU Mat 並執行完整的上傳與推論流程
 */
std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    // 1. 上傳至 GPU 並執行預處理
    preprocessGPU(img);

    // 2. 執行推論
    mContext->enqueueV3(mStream);

    // 3. 非同步拷貝結果回 Host
    const float* dataPtr;
    if (mUsePinnedMemory) {
        cudaMemcpyAsync(mPinnedOutputBuffer, mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
        cudaStreamSynchronize(mStream); // 唯一同步點
        dataPtr = mPinnedOutputBuffer;
    }
    else {
        cudaMemcpyAsync(mOutputHostBuffer.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
        cudaStreamSynchronize(mStream);
        dataPtr = mOutputHostBuffer.data();
    }

    return postprocess(dataPtr, img.size());
}

/**
 * 零拷貝模式：直接處理顯存中的影像 (測試算力極限)
 */
std::vector<Detection> YoloDetector::detectGpu(const cv::cuda::GpuMat& d_img) {
    if (d_img.empty()) return {};

    // 1. 內部 GPU 預處理 (零 CPU 參與)
    preprocessGpuDirect(d_img);

    // 2. 執行推論
    mContext->enqueueV3(mStream);

    // 3. 傳輸偵測結果
    cudaMemcpyAsync(mPinnedOutputBuffer, mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    return postprocess(mPinnedOutputBuffer, d_img.size());
}

/**
 * 封裝影像上傳與處理
 */
void YoloDetector::preprocessGPU(const cv::Mat& img) {
    m_d_img.upload(img, mCvStream);
    preprocessGpuDirect(m_d_img);
}

/**
 * GPU 直接預處理：縮放、色彩空間轉換與歸一化 (CHW)
 */
void YoloDetector::preprocessGpuDirect(const cv::cuda::GpuMat& d_img) {
    cv::cuda::resize(d_img, m_d_resized, cv::Size(mInputW, mInputH), 0, 0, cv::INTER_LINEAR, mCvStream);
    cv::cuda::cvtColor(m_d_resized, m_d_rgb, cv::COLOR_BGR2RGB, 3, mCvStream);
    m_d_rgb.convertTo(m_d_float, CV_32FC3, 1.0 / 255.0, 0.0, mCvStream);
    cv::cuda::split(m_d_float, m_chw_channels, mCvStream);
}

/**
 * 後處理：解析 Tensor 並執行 NMS (已優化內存分配)
 */
std::vector<Detection> YoloDetector::postprocess(const float* data, const cv::Size& originalSize) {
    mClassIds.clear();
    mConfidences.clear();
    mBoxes.clear();
    mNmsIndices.clear();

    const float scale = std::min((float)mInputW / originalSize.width, (float)mInputH / originalSize.height);
    const float xOffset = (mInputW - (originalSize.width * scale)) / 2.0f;
    const float yOffset = (mInputH - (originalSize.height * scale)) / 2.0f;
    const float invScale = 1.0f / scale;

    for (int i = 0; i < 8400; ++i) {
        const float* row = data + (i * 6);
        if (row[4] > 0.25f) {
            int left = static_cast<int>((row[0] - row[2] * 0.5f - xOffset) * invScale);
            int top = static_cast<int>((row[1] - row[3] * 0.5f - yOffset) * invScale);
            int width = static_cast<int>(row[2] * invScale);
            int height = static_cast<int>(row[3] * invScale);

            // 邊界安全檢查
            left = std::max(0, std::min(left, originalSize.width - 1));
            top = std::max(0, std::min(top, originalSize.height - 1));
            width = std::max(1, std::min(width, originalSize.width - left));
            height = std::max(1, std::min(height, originalSize.height - top));

            mClassIds.push_back(static_cast<int>(row[5]));
            mConfidences.push_back(row[4]);
            mBoxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // 執行 NMS
    cv::dnn::NMSBoxes(mBoxes, mConfidences, 0.25f, 0.45f, mNmsIndices);

    std::vector<Detection> detections;
    detections.reserve(mNmsIndices.size());
    for (int idx : mNmsIndices) {
        Detection det;
        det.classId = mClassIds[idx];
        det.confidence = mConfidences[idx];
        det.box = mBoxes[idx];
        detections.push_back(det);
    }
    return detections;
}
