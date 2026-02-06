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
    mOutputSize = 1 * 6 * 8400 * sizeof(float);

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

    std::cout << ">>> 原圖尺寸: " << img.cols << " x " << img.rows << std::endl;

    // 2. 預處理
    std::vector<float> inputData = preprocess(img);
    cudaMemcpyAsync(mInputBuffer, inputData.data(), mInputSize, cudaMemcpyHostToDevice, mStream);

    // 3. 推論
    mContext->enqueueV3(mStream);

    // 4. 取回資料
    std::vector<float> outputData(mOutputSize / sizeof(float));
    cudaMemcpyAsync(outputData.data(), mOutputBuffer, mOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    std::cout << ">>> 推論完成，開始後處理..." << std::endl;

    // 5. [新增] 呼叫後處理
    std::vector<Detection> results = postprocess(outputData, img.size());
    std::cout << ">>> 偵測到 " << results.size() << " 個物體！" << std::endl;

    // 6. [修正] 畫圖 (Visualization) - 強制可見版
    for (const auto& det : results) {
        // 畫矩形框
        cv::rectangle(img, det.box, det.color, 3);

        // 準備標籤文字
        std::string classString = std::to_string(det.classId);
        if (det.classId == 0) classString = "Person";
        else if (det.classId == 1) classString = "Bicycle";
        else if (det.classId == 2) classString = "Car";
        else if (det.classId == 3) classString = "Motorcycle";
        else if (det.classId == 5) classString = "Bus";
        else if (det.classId == 7) classString = "Truck";

        std::string label = classString + " " + std::to_string((int)(det.confidence * 100)) + "%";

        // 計算文字背景大小
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);

        // --- [關鍵修正] 強制把字限制在圖片範圍內 ---
        int labelY = det.box.y - 10;
        int labelX = det.box.x;

        // 1. 如果 Y 座標跑出去了 (負數)，強制固定在頂端
        if (labelY < 20) {
            labelY = 20;
        }

        // 2. 如果 X 座標跑出去了 (負數)，強制固定在左邊
        if (labelX < 0) {
            labelX = 0;
        }
        // ------------------------------------------

        // 畫文字背景 (黑底)
        cv::rectangle(img, cv::Point(labelX, labelY - labelSize.height),
            cv::Point(labelX + labelSize.width, labelY + baseLine),
            det.color, cv::FILLED);

        // 寫字 (白色)
        cv::putText(img, label, cv::Point(labelX, labelY),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        std::cout << "   -> [偵測結果] " << label << " at " << det.box << std::endl;
    }

    // 7. 顯示結果並存檔
    if (img.cols > 1000) {
        cv::resize(img, img, cv::Size(img.cols * 0.5, img.rows * 0.5));
    }

    cv::imshow("YOLO Result", img);
    cv::imwrite("result_output.jpg", img);

    std::cout << ">>> 結果已存檔至 result_output.jpg" << std::endl;
    std::cout << ">>> 按任意鍵關閉視窗..." << std::endl;
    cv::waitKey(0);
}

// [最終修正版 - Final] 針對 End-to-End 模型的 [x1, y1, x2, y2] 格式
std::vector<Detection> YoloDetector::postprocess(const std::vector<float>& output, const cv::Size& originalSize) {
    std::vector<Detection> detections;
    const float* data = output.data();
    int numAnchors = 8400;
    int stride = 6;

    // 計算 Letterbox 縮放參數
    float scale = std::min((float)mInputW / originalSize.width, (float)mInputH / originalSize.height);
    int newW = (int)(originalSize.width * scale);
    int newH = (int)(originalSize.height * scale);
    int xOffset = (mInputW - newW) / 2;
    int yOffset = (mInputH - newH) / 2;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numAnchors; ++i) {
        const float* row = data + (i * stride);

        float confidence = row[4];

        if (confidence > 0.25f) {
            int classId = (int)row[5];

            // [關鍵修正] 讀取 x1, y1, x2, y2 (而不是 cx, cy, w, h)
            float x1 = row[0];
            float y1 = row[1];
            float x2 = row[2];
            float y2 = row[3];

            // 座標還原 (逆向操作 Letterbox)
            // 公式：(座標 - 偏移量) / 縮放倍率
            float r_x1 = (x1 - xOffset) / scale;
            float r_y1 = (y1 - yOffset) / scale;
            float r_x2 = (x2 - xOffset) / scale;
            float r_y2 = (y2 - yOffset) / scale;

            // 轉換成 OpenCV 的 Rect (左上角 x, 左上角 y, 寬, 高)
            int left = (int)r_x1;
            int top = (int)r_y1;
            int width = (int)(r_x2 - r_x1);
            int height = (int)(r_y2 - r_y1);

            // 防呆機制：避免寬高變成負的 (雖然理論上不應該發生)
            if (width < 0) width = 0;
            if (height < 0) height = 0;

            classIds.push_back(classId);
            confidences.push_back(confidence);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // NMS
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
