#include "YoloDetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp> 
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

/**
 * [輔助函式] 繪製偵測結果 (DeepPCB 類別對應)
 */
void drawResults(cv::Mat& img, const std::vector<Detection>& results) {
    const std::vector<std::string> classNames = {
        "open", "short", "mousebite", "spur", "copper", "pin-hole"
    };

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),   // open: 紅色
        cv::Scalar(0, 255, 255), // short: 黃色
        cv::Scalar(255, 0, 0),   // mousebite: 藍色
        cv::Scalar(0, 255, 0),   // spur: 綠色
        cv::Scalar(255, 0, 255), // copper: 紫色
        cv::Scalar(255, 165, 0)  // pin-hole: 橘色
    };

    for (const auto& det : results) {
        cv::Scalar color = (det.classId < colors.size()) ? colors[det.classId] : cv::Scalar(255, 255, 255);
        cv::rectangle(img, det.box, color, 3);

        std::string classString = (det.classId < classNames.size()) ? classNames[det.classId] : "Unknown";
        std::string label = classString + " " + std::to_string((int)(det.confidence * 100)) + "%";

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        int labelX = std::max(det.box.x, 0);
        int labelY = std::max(labelSize.height, det.box.y - 10);

        cv::rectangle(img, cv::Point(labelX, labelY - labelSize.height),
            cv::Point(labelX + labelSize.width, labelY + baseLine),
            color, cv::FILLED);

        cv::putText(img, label, cv::Point(labelX, labelY),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        std::cout << "   -> [偵測到瑕疵] " << label << " 座標: " << det.box << std::endl;
    }
}

int main() {
    // ==========================================
    // 0. 硬體環境驗證 (確認 OpenCV 4.13.0 + CUDA 是否生效)
    // ==========================================
    std::cout << ">>> [硬體檢查] 正在初始化 MSI RTX 4060 環境..." << std::endl;
    int cuda_count = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_count < 1) {
        std::cerr << "!!! 錯誤：找不到支援 CUDA 的 GPU，請檢查 OpenCV 連結是否正確。" << std::endl;
        system("pause");
        return -1;
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // ==========================================
    // 1. 設定相對路徑 (指向上一層資料夾 ../)
    // ==========================================
    std::string enginePath = "../pcb_aoi.engine";
    std::string testImagePath = "../test.jpg";

    // ==========================================
    // 2. 初始化 TensorRT 引擎
    // ==========================================
    YoloDetector detector(enginePath);
    if (!detector.init()) {
        std::cerr << ">>> 引擎初始化失敗！請確認檔案是否在上一層目錄：" << enginePath << std::endl;
        system("pause");
        return -1;
    }

    // ==========================================
    // 3. 讀取測試圖片
    // ==========================================
    cv::Mat img = cv::imread(testImagePath);
    if (img.empty()) {
        std::cerr << ">>> 無法讀取圖片，請確認檔案是否在上一層目錄：" << testImagePath << std::endl;
        system("pause");
        return -1;
    }

    // ==========================================
    // 4. [模式一] 視覺化測試
    // ==========================================
    std::cout << "\n>>> 啟動模式一：視覺化驗證..." << std::endl;
    cv::Mat visualImg = img.clone();
    auto results = detector.detect(visualImg);

    std::cout << ">>> 偵測到 " << results.size() << " 個瑕疵。" << std::endl;
    drawResults(visualImg, results);

    cv::Mat displayImg;
    double displayScale = (visualImg.cols > 1280) ? 0.5 : 1.0;
    cv::resize(visualImg, displayImg, cv::Size(), displayScale, displayScale);

    cv::imshow("AOI - YOLO26 RTX 4060", displayImg);
    cv::imwrite("../pcb_defect_result.jpg", visualImg); // 結果也存到上一層
    std::cout << ">>> 按任意鍵開始 1000 次效能跑分..." << std::endl;
    cv::waitKey(0);

    // ==========================================
    // 5. [模式二] 效能跑分測試
    // ==========================================
    std::cout << "\n>>> 啟動模式二：1000 次 End-to-End 效能跑分..." << std::endl;

    for (int i = 0; i < 50; i++) detector.detect(img);

    std::vector<double> latencies;
    latencies.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector.detect(img);
        cv::cuda::Stream::Null().waitForCompletion();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);

        if ((i + 1) % 100 == 0) std::cout << "已完成 " << (i + 1) << " 次推論..." << std::endl;
    }

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::sort(latencies.begin(), latencies.end());
    double minVal = latencies.front();
    double maxVal = latencies.back();
    double p95 = latencies[950];

    std::cout << "\n========================================" << std::endl;
    std::cout << "AOI 效能報告 (RTX 4060 + CV 4.13.0)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "平均延遲 (Mean) : " << avg << " ms" << std::endl;
    std::cout << "95% 延遲 (P95)  : " << p95 << " ms" << std::endl;
    std::cout << "極速 (Min)      : " << minVal << " ms" << std::endl;
    std::cout << "每秒幀數 (FPS)  : " << 1000.0 / avg << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n按任意鍵退出..." << std::endl;
    cv::waitKey(0);
    return 0;
}
