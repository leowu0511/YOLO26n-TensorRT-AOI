#include "YoloDetector.h"

int main() {
    std::string enginePath = R"(C:\Users\wu096\OneDrive\Desktop\YOLO_cpp\yolo26n.engine)";

    YoloDetector detector(enginePath);

    if (detector.init()) {
        std::cout << "AOI start success" << std::endl;
        std::cout << "階段 5 完成 (GPU 緩衝區已建立)" << std::endl;
    }
    else {
        std::cerr << "fail" << std::endl;
        return -1;
    }

    // 這裡暫時暫停，讓你看清楚控制台輸出
    system("pause");
    return 0;
}
