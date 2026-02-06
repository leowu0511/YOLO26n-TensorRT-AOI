#include "YoloDetector.h"
#include <opencv2/cudawarping.hpp>  // GPU 縮放
#include <opencv2/cudaarithm.hpp>   // GPU 矩陣運算
#include <opencv2/cudaimgproc.hpp>  // GPU 色彩轉換

YoloDetector::YoloDetector(const std::string& enginePath) : mEnginePath(enginePath) {}

YoloDetector::~YoloDetector() {
    if (mInputBuffer) cudaFree(mInputBuffer);
    if (mOutputBuffer) cudaFree(mOutputBuffer);
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

    mRuntime = nvinfer1::createInferRuntime(mLogger);
    if (!mRuntime) return false;

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!mEngine) return false;

    mContext = mEngine->createExecutionContext();
    if (!mContext) return false;

    if (cudaStreamCreate(&mStream) != cudaSuccess) {
        std::cerr << "CUDA Stream 建立失敗" << std::endl;
        return false;
    }

    const char* inputName = mEngine->getIOTensorName(0);
    const char* outputName = mEngine->getIOTensorName(1);

    // 💡 這裡是固定輸入輸出大小 (與 YOLO26n 結構對應)
    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 6 * 8400 * sizeof(float);

    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    mContext->setTensorAddress(inputName, mInputBuffer);
    mContext->setTensorAddress(outputName, mOutputBuffer);

    std::cout << ">>> YoloDetector 初始化成功 [RTX 4060 硬體加速啟動]" << std::endl;
    return true;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    // 1. [極速優化] 全 GPU 預處理 (取代原本的 CPU 迴圈)
    preprocessGPU(img);

    // 2. 執行非同步推論
    mContext->enqueueV3(mStream);

    // 3. 取回資料 (Host 記憶體分配一次即可優化，這裡先維持簡單版)
    std::vector<float> outputData(mOutputSize / sizeof(float));
    cudaMemcpyAsync(outputData.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    // 4. 後處理 (這部分通常在 CPU 跑)
    return postprocess(outputData, img.size());
}

/**
 * 💡 GPU 預處理函式：將所有計算鎖死在 RTX 4060 內
 */
void YoloDetector::preprocessGPU(const cv::Mat& img) {
    cv::cuda::GpuMat d_img, d_resized, d_rgb, d_float;

    // 1. 上傳圖片到 GPU
    d_img.upload(img, cv::cuda::Stream::Null());

    // 2. GPU 縮放 (保持長寬比的邏輯可整合或簡化)
    cv::cuda::resize(d_img, d_resized, cv::Size(mInputW, mInputH), 0, 0, cv::INTER_LINEAR, cv::cuda::Stream::Null());

    // 3. 色彩轉換 BGR -> RGB 並轉為浮點數
    cv::cuda::cvtColor(d_resized, d_rgb, cv::COLOR_BGR2RGB, 3, cv::cuda::Stream::Null());

    // 4. 正規化 (0-255 -> 0.0-1.0)
    d_rgb.convertTo(d_float, CV_32FC3, 1.0 / 255.0, cv::cuda::Stream::Null());

    // 💡 5. 維度轉換 (HWC -> CHW)：這是提速關鍵
    // 將交錯的 RGB 像素分拆成三個獨立平面，直接寫入 TensorRT 的 Input Buffer
    std::vector<cv::cuda::GpuMat> chw_channels(3);
    for (int i = 0; i < 3; ++i) {
        chw_channels[i] = cv::cuda::GpuMat(mInputH, mInputW, CV_32FC1, (float*)mInputBuffer + i * mInputW * mInputH);
    }
    cv::cuda::split(d_float, chw_channels, cv::cuda::Stream::Null());
}

std::vector<Detection> YoloDetector::postprocess(const std::vector<float>& output, const cv::Size& originalSize) {
    std::vector<Detection> detections;
    const float* data = output.data();
    int numAnchors = 8400; // YOLOv8/YOLO26 標準輸出數量
    int stride = 6;        // x, y, w, h, confidence, classId

    float scale = std::min((float)mInputW / originalSize.width, (float)mInputH / originalSize.height);
    float xOffset = (mInputW - (originalSize.width * scale)) / 2;
    float yOffset = (mInputH - (originalSize.height * scale)) / 2;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numAnchors; ++i) {
        const float* row = data + (i * stride);
        float confidence = row[4];

        if (confidence > 0.25f) {
            int classId = (int)row[5];

            // 💡 這裡將座標還原回原始圖片尺寸
            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            int left = (int)((cx - w / 2 - xOffset) / scale);
            int top = (int)((cy - h / 2 - yOffset) / scale);
            int width = (int)(w / scale);
            int height = (int)(h / scale);

            classIds.push_back(classId);
            confidences.push_back(confidence);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indices);

    for (int idx : indices) {
        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        detections.push_back(det);
    }
    return detections;
}
