#include "YoloDetector.h"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

YoloDetector::YoloDetector(const std::string& enginePath) : mEnginePath(enginePath) {
    // 💡 預分配 NMS 工作區與 Host Buffer (安全第一)
    mClassIds.reserve(1000);
    mConfidences.reserve(1000);
    mBoxes.reserve(1000);
    mNmsIndices.reserve(1000);
    mOutputHostBuffer.resize(50400); // 1 * 6 * 8400
}

YoloDetector::~YoloDetector() {
    if (mInputBuffer) cudaFree(mInputBuffer);
    if (mOutputBuffer) cudaFree(mOutputBuffer);
    if (mPinnedOutputBuffer) cudaFreeHost(mPinnedOutputBuffer);
    if (mStream) cudaStreamDestroy(mStream);

    if (mContext) delete mContext;
    if (mEngine) delete mEngine;
    if (mRuntime) delete mRuntime;
}

bool YoloDetector::init() {
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

    // 💡 安全檢查：確保 TensorRT 組件正確載入
    mRuntime = nvinfer1::createInferRuntime(mLogger);
    if (!mRuntime) { std::cerr << "無法建立 Runtime"; return false; }

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!mEngine) { std::cerr << "無法載入 Engine"; return false; }

    mContext = mEngine->createExecutionContext();
    if (!mContext) { std::cerr << "無法建立 Context"; return false; }

    if (cudaStreamCreate(&mStream) != cudaSuccess) return false;
    mCvStream = cv::cuda::StreamAccessor::wrapStream(mStream);

    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 6 * 8400 * sizeof(float);

    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    // 💡 Pinned Memory 優化與退路
    if (cudaMallocHost(&mPinnedOutputBuffer, mOutputSize) != cudaSuccess) {
        std::cerr << "警告：Pinned Memory 分配失敗，使用一般記憶體" << std::endl;
        mUsePinnedMemory = false;
    }

    mContext->setTensorAddress(mEngine->getIOTensorName(0), mInputBuffer);
    mContext->setTensorAddress(mEngine->getIOTensorName(1), mOutputBuffer);

    // 預分配 GPU 緩衝區
    m_d_img.create(mInputH, mInputW, CV_8UC3);
    m_d_resized.create(mInputH, mInputW, CV_8UC3);
    m_d_rgb.create(mInputH, mInputW, CV_8UC3);
    m_d_float.create(mInputH, mInputW, CV_32FC3);

    m_chw_channels.clear();
    for (int i = 0; i < 3; ++i) {
        m_chw_channels.push_back(cv::cuda::GpuMat(mInputH, mInputW, CV_32FC1, (float*)mInputBuffer + i * mInputW * mInputH));
    }

    std::cout << ">>> YoloDetector 最終加固版已就緒 [RTX 4060]" << std::endl;
    return true;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    preprocessGPU(img);
    mContext->enqueueV3(mStream);

    const float* dataPtr;
    if (mUsePinnedMemory) {
        cudaMemcpyAsync(mPinnedOutputBuffer, mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
        cudaStreamSynchronize(mStream);
        dataPtr = mPinnedOutputBuffer;
    }
    else {
        cudaMemcpyAsync(mOutputHostBuffer.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
        cudaStreamSynchronize(mStream);
        dataPtr = mOutputHostBuffer.data();
    }

    return postprocess(dataPtr, img.size());
}

void YoloDetector::preprocessGPU(const cv::Mat& img) {
    m_d_img.upload(img, mCvStream);
    cv::cuda::resize(m_d_img, m_d_resized, cv::Size(mInputW, mInputH), 0, 0, cv::INTER_LINEAR, mCvStream);
    cv::cuda::cvtColor(m_d_resized, m_d_rgb, cv::COLOR_BGR2RGB, 3, mCvStream);
    m_d_rgb.convertTo(m_d_float, CV_32FC3, 1.0 / 255.0, 0.0, mCvStream);
    cv::cuda::split(m_d_float, m_chw_channels, mCvStream);
}

std::vector<Detection> YoloDetector::postprocess(const float* data, const cv::Size& originalSize) {
    mClassIds.clear(); mConfidences.clear(); mBoxes.clear(); mNmsIndices.clear();

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

            left = std::max(0, std::min(left, originalSize.width - 1));
            top = std::max(0, std::min(top, originalSize.height - 1));
            width = std::max(1, std::min(width, originalSize.width - left));
            height = std::max(1, std::min(height, originalSize.height - top));

            mClassIds.push_back(static_cast<int>(row[5]));
            mConfidences.push_back(row[4]);
            mBoxes.push_back(cv::Rect(left, top, width, height));
        }
    }

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
