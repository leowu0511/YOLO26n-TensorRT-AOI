#include "YoloDetector.h"

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

    mInputSize = 1 * 3 * 640 * 640 * sizeof(float);
    mOutputSize = 1 * 6 * 8400 * sizeof(float);

    if (cudaMalloc(&mInputBuffer, mInputSize) != cudaSuccess) return false;
    if (cudaMalloc(&mOutputBuffer, mOutputSize) != cudaSuccess) return false;

    mContext->setTensorAddress(inputName, mInputBuffer);
    mContext->setTensorAddress(outputName, mOutputBuffer);

    std::cout << ">>> YoloDetector 初始化成功 (RTX 4060)" << std::endl;
    return true;
}

// [純淨版偵測介面]
std::vector<Detection> YoloDetector::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    // 1. 預處理
    std::vector<float> inputData = preprocess(img);
    cudaMemcpyAsync(mInputBuffer, inputData.data(), mInputSize, cudaMemcpyHostToDevice, mStream);

    // 2. 推論
    mContext->enqueueV3(mStream);

    // 3. 取回資料
    std::vector<float> outputData(mOutputSize / sizeof(float));
    cudaMemcpyAsync(outputData.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    // 4. 後處理並回傳結果
    return postprocess(outputData, img.size());
}

std::vector<float> YoloDetector::preprocess(const cv::Mat& img) {
    int w = img.cols;
    int h = img.rows;

    float scale = std::min((float)mInputW / w, (float)mInputH / h);
    int newW = (int)(w * scale);
    int newH = (int)(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));

    cv::Mat canvas(mInputH, mInputW, CV_8UC3, cv::Scalar(114, 114, 114));
    int xOffset = (mInputW - newW) / 2;
    int yOffset = (mInputH - newH) / 2;
    resized.copyTo(canvas(cv::Rect(xOffset, yOffset, newW, newH)));

    std::vector<float> data(mInputSize / sizeof(float));
    for (int r = 0; r < mInputH; ++r) {
        for (int c = 0; c < mInputW; ++c) {
            cv::Vec3b pixel = canvas.at<cv::Vec3b>(r, c);
            data[0 * mInputH * mInputW + r * mInputW + c] = pixel[2] / 255.0f;
            data[1 * mInputH * mInputW + r * mInputW + c] = pixel[1] / 255.0f;
            data[2 * mInputH * mInputW + r * mInputW + c] = pixel[0] / 255.0f;
        }
    }
    return data;
}

std::vector<Detection> YoloDetector::postprocess(const std::vector<float>& output, const cv::Size& originalSize) {
    std::vector<Detection> detections;
    const float* data = output.data();
    int numAnchors = 8400;
    int stride = 6;

    float scale = std::min((float)mInputW / originalSize.width, (float)mInputH / originalSize.height);
    int xOffset = (mInputW - (int)(originalSize.width * scale)) / 2;
    int yOffset = (mInputH - (int)(originalSize.height * scale)) / 2;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numAnchors; ++i) {
        const float* row = data + (i * stride);
        float confidence = row[4];

        if (confidence > 0.25f) {
            int classId = (int)row[5];
            float r_x1 = (row[0] - xOffset) / scale;
            float r_y1 = (row[1] - yOffset) / scale;
            float r_x2 = (row[2] - xOffset) / scale;
            float r_y2 = (row[3] - yOffset) / scale;

            int left = (int)r_x1;
            int top = (int)r_y1;
            int width = std::max(0, (int)(r_x2 - r_x1));
            int height = std::max(0, (int)(r_y2 - r_y1));

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
        cv::RNG rng(det.classId * 12345);
        det.color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        detections.push_back(det);
    }
    return detections;
}
