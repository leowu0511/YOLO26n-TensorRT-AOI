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
 * [輔助函式] 在影像上繪製偵測結果 (PCB 六大瑕疵類別)
 */
void drawResults(cv::Mat& img, const std::vector<Detection>& results) {
    const std::vector<std::string> classNames = { "open", "short", "mousebite", "spur", "copper", "pin-hole" };
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0,0,255), cv::Scalar(0,255,255), cv::Scalar(255,0,0),
        cv::Scalar(0,255,0), cv::Scalar(255,0,255), cv::Scalar(255,165,0)
    };

    for (const auto& det : results) {
        if (det.classId < 0 || det.classId >= classNames.size()) continue;
        cv::Scalar color = colors[det.classId];
        cv::rectangle(img, det.box, color, 3);

        // 建立標籤文字與背景
        std::string label = classNames[det.classId] + " " + std::to_string((int)(det.confidence * 100)) + "%";
        cv::putText(img, label, cv::Point(det.box.x, std::max(20, det.box.y - 10)),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

int main() {
    std::cout << "[System] Initializing AOI Inference Engine (Target: RTX 4060)..." << std::endl;

    // 硬體檢查
    if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        std::cerr << "[Error] No CUDA-capable device detected." << std::endl;
        return -1;
    }

    // 路徑配置
    std::string enginePath = "../pcb_aoi.engine";
    std::string testSetPath = "../Test_Set_Raw/images/";

    // 初始化推論引擎
    YoloDetector detector(enginePath);
    if (!detector.init()) {
        std::cerr << "[Error] Failed to initialize inference engine." << std::endl;
        system("pause"); return -1;
    }

    // 掃描測試資料集
    std::vector<std::string> imagePaths;
    cv::glob(testSetPath + "*.jpg", imagePaths);
    if (imagePaths.empty()) {
        std::cerr << "[Error] Test dataset not found at: " << testSetPath << std::endl;
        system("pause"); return -1;
    }

    // 影像預載入程序 (同時載入至系統記憶體 RAM 與顯示卡顯存 VRAM)
    std::vector<cv::Mat> preloadedCpuImages;
    std::vector<cv::cuda::GpuMat> preloadedGpuImages;

    size_t loadCount = std::min((size_t)100, imagePaths.size());
    std::cout << "[Storage] Pre-loading " << loadCount << " samples to RAM/VRAM..." << std::endl;

    for (size_t i = 0; i < loadCount; ++i) {
        cv::Mat m = cv::imread(imagePaths[i]);
        if (!m.empty()) {
            preloadedCpuImages.push_back(m);
            cv::cuda::GpuMat d_m;
            d_m.upload(m); // 預先上傳至 GPU 供零拷貝模式測試
            preloadedGpuImages.push_back(d_m);
        }
    }

    // 模式一：實戰模式 (Production Mode)
    // 流程：包含資料從 CPU 傳輸至 GPU 的完整耗時，對應真實產線情境
    std::cout << "\n[Mode] Launching Production Mode (Host-to-Device transfer included)..." << std::endl;
    std::vector<double> prodLatencies;
    prodLatencies.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        const cv::Mat& frame = preloadedCpuImages[i % preloadedCpuImages.size()];
        auto start = std::chrono::high_resolution_clock::now();

        detector.detect(frame);

        auto end = std::chrono::high_resolution_clock::now();
        prodLatencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // 模式二：零拷貝模式 (Zero-Copy Mode)
    // 流程：影像已存在於顯存，純粹測試 GPU 預處理與模型推論的算力極限
    std::cout << "[Mode] Launching Zero-Copy Mode (On-Device Inference)..." << std::endl;
    std::vector<double> turboLatencies;
    turboLatencies.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        const cv::cuda::GpuMat& d_frame = preloadedGpuImages[i % preloadedGpuImages.size()];
        auto start = std::chrono::high_resolution_clock::now();

        detector.detectGpu(d_frame);

        auto end = std::chrono::high_resolution_clock::now();
        turboLatencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // 效能數據統計彙整
    auto getStats = [](std::vector<double>& v) {
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::sort(v.begin(), v.end());
        return std::make_pair(avg, v[950]); // 返回平均值與 P95
        };

    auto prodStats = getStats(prodLatencies);
    auto turboStats = getStats(turboLatencies);

    // 輸出最終效能報告
    std::cout << "\n========================================" << std::endl;
    std::cout << "  AOI ENGINE PERFORMANCE REPORT" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[Production] Mean: " << prodStats.first << " ms | FPS: " << 1000.0 / prodStats.first << std::endl;
    std::cout << "[Production] P95:  " << prodStats.second << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "[Zero-Copy]  Mean: " << turboStats.first << " ms | FPS: " << 1000.0 / turboStats.first << std::endl;
    std::cout << "[Zero-Copy]  P95:  " << turboStats.second << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // 視覺化驗證結果輸出
    cv::Mat resImg = preloadedCpuImages[0].clone();
    auto dets = detector.detect(resImg);
    drawResults(resImg, dets);
    cv::imshow("Inference Output Validation", resImg);

    std::cout << "\nBenchmark complete. Results visualized." << std::endl;
    cv::waitKey(0);
    return 0;
}
