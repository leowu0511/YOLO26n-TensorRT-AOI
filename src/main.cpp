#include "YoloDetector.h"

int main() {
    // 設定引擎路徑 (請用你的路徑)
    std::string enginePath = R"(C:\Users\wu096\OneDrive\Desktop\YOLO_cpp\yolo26n.engine)";
    // 設定測試圖片路徑 (請準備一張圖片)
    std::string imagePath = R"(C:\Users\wu096\OneDrive\Desktop\YOLO_cpp\test.jpg)";

    YoloDetector detector(enginePath);

    if (detector.init()) {
        std::cout << "--- 系統初始化完畢，開始處理圖片 ---" << std::endl;
        detector.detect(imagePath);
    }
    else {
        std::cerr << "初始化失敗" << std::endl;
        return -1;
    }

    system("pause");
    return 0;
}
//暫時完成
