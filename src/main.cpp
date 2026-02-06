#include "YoloDetector.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
/**
 * [輔助函式] 專門負責將 Detection 結果畫在圖片上
 * 修正版：對應 DeepPCB 訓練類別 (open, short, mousebite, spur, copper, pin-hole)
 */
void drawResults(cv::Mat& img, const std::vector<Detection>& results) {
    // 💡 定義 PCB 瑕疵類別名稱 (必須與訓練時的順序完全一致)
    // 來源：NDHU_AOI_2026/PCB_Standard_v1 訓練日誌
    const std::vector<std::string> classNames = {
        "open", "short", "mousebite", "spur", "copper", "pin-hole"
    };

    // 💡 為不同類別設定不同顏色 (B, G, R)
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),   // open: 紅色
        cv::Scalar(0, 255, 255), // short: 黃色
        cv::Scalar(255, 0, 0),   // mousebite: 藍色
        cv::Scalar(0, 255, 0),   // spur: 綠色
        cv::Scalar(255, 0, 255), // copper: 紫色
        cv::Scalar(255, 165, 0)  // pin-hole: 橘色
    };

    for (const auto& det : results) {
        // 確保 classId 在合法範圍內
        cv::Scalar color = (det.classId < colors.size()) ? colors[det.classId] : cv::Scalar(255, 255, 255);

        // 1. 畫矩形框
        cv::rectangle(img, det.box, color, 3);

        // 2. 獲取類別名稱
        std::string classString = (det.classId < classNames.size()) ? classNames[det.classId] : "Unknown";
        std::string label = classString + " " + std::to_string((int)(det.confidence * 100)) + "%";

        // 3. 計算文字背景位置
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);

        int labelX = std::max(det.box.x, 0);
        int labelY = det.box.y - 10;
        if (labelY < 20) labelY = std::min(img.rows - 10, det.box.y + 25);

        // 畫文字底色塊
        cv::rectangle(img, cv::Point(labelX, labelY - labelSize.height),
            cv::Point(labelX + labelSize.width, labelY + baseLine),
            color, cv::FILLED);

        // 寫入白色文字
        cv::putText(img, label, cv::Point(labelX, labelY),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        std::cout << "   -> [偵測到瑕疵] " << label << " 座標: " << det.box << std::endl;
    }
}

int main() {
    // ==========================================
    // 1. 設定絕對路徑 (已根據你的下載路徑更新)
    // ==========================================
    std::string enginePath = "C:/Users/wu096/source/repos/testcudnddcudnn/pcb_aoi.engine";
    std::string testImagePath = "C:/Users/wu096/source/repos/testcudnddcudnn/test.jpg";

    // ==========================================
    // 2. 初始化 TensorRT 引擎
    // ==========================================
    YoloDetector detector(enginePath);
    if (!detector.init()) {
        std::cerr << ">>> 引擎初始化失敗！請確認 C++ 專案與 TensorRT 版本是否一致。" << std::endl;
        system("pause");
        return -1;
    }

    // ==========================================
    // 3. 讀取測試圖片 (Zenfone 8 實拍圖)
    // ==========================================
    cv::Mat img = cv::imread(testImagePath);
    if (img.empty()) {
        std::cerr << ">>> 無法讀取圖片，請檢查路徑：" << testImagePath << std::endl;
        system("pause");
        return -1;
    }
    // ==========================================
    // 4. [模式一] 視覺化測試 (驗證標籤正確性)
    // ==========================================
    std::cout << "\n>>> 啟動模式一：視覺化驗證 (DeepPCB 類別對應)..." << std::endl;
    cv::Mat visualImg = img.clone();
    auto results = detector.detect(visualImg); // 執行推論

    std::cout << ">>> 偵測到 " << results.size() << " 個疑似瑕疵處。" << std::endl;
    drawResults(visualImg, results);

    // 縮放顯示 (避免 4K 圖片塞不下螢幕)
    cv::Mat displayImg;
    double displayScale = (visualImg.cols > 1280) ? 0.5 : 1.0;
    cv::resize(visualImg, displayImg, cv::Size(), displayScale, displayScale);

    cv::imshow("NDHU AOI - YOLO26 TensorRT Result", displayImg);
    cv::imwrite("pcb_defect_result.jpg", visualImg);
    std::cout << ">>> 結果已存檔至 pcb_defect_result.jpg，按任意鍵開始 1000 次效能跑分..." << std::endl;
    cv::waitKey(0);

    // ==========================================
    // 5. [模式二] 效能跑分測試 (MSI RTX 4060 實力展示)
    // ==========================================
    std::cout << "\n>>> 啟動模式二：1000 次 End-to-End 跑分測試..." << std::endl;

    // 暖身 (Warm-up) 以穩定 GPU 時脈
    for (int i = 0; i < 20; i++) detector.detect(img);

    std::vector<double> latencies;
    latencies.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // 執行完整的預處理 + 推論 + 後處理
        detector.detect(img);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);

        if ((i + 1) % 100 == 0) std::cout << "已完成 " << (i + 1) << " 次測試..." << std::endl;
    }

    // 計算統計數據
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();

    std::sort(latencies.begin(), latencies.end());
    double minVal = latencies.front();
    double maxVal = latencies.back();
    double p95 = latencies[950];

    std::cout << "\n========================================" << std::endl;
    std::cout << "🏎️  NDHU AOI 效能報告 (MSI RTX 4060)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "平均延遲 (Mean) : " << avg << " ms" << std::endl;
    std::cout << "95% 延遲 (P95)  : " << p95 << " ms" << std::endl;
    std::cout << "極速 / 最慢     : " << minVal << " / " << maxVal << " ms" << std::endl;
    std::cout << "每秒幀數 (FPS)  : " << 1000.0 / avg << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n測試結束，這份數據可用於 2026 智慧創新大賞報告。" << std::endl;
    std::cout << "按任意鍵退出..." << std::endl;
    cv::waitKey(0);
    return 0;
}
