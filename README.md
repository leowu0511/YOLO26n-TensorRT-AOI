# YOLO26n-TensorRT-AOI

本專案旨在開發一個極致效能的 **PCB 自動光學檢測 (AOI) 推論引擎**，針對 **2026 智慧創新大賞** 開發。透過深度結合 TensorRT 與 OpenCV CUDA 模組，在 MSI RTX 4060 平台上實現了超越工業標準的 **440+ FPS** 處理速度。

## 📊 效能表現 (Performance Results)

經 1000 次 End-to-End 連續推論測試，檢測 YOLO26n 模型（輸入尺寸 640x640）：

| 統計指標 | 數值 | 單位 |
| --- | --- | --- |
| **平均延遲 (Mean)** | **2.2705** | ms |
| **95% 延遲 (P95)** | **2.7864** | ms |
| **極速 (Min)** | **1.7618** | ms |
| **吞吐量 (Throughput)** | **440.42** | **FPS** |

> **優化成就：** 相比於傳統 CPU 預處理方案 (約 10ms)，效能提升了 **340%**。

---

## 技術核心與優化實施

### 1. 全 GPU 影像預處理鏈 (Hardware-Accelerated Pipeline)

徹底消滅 CPU 瓶頸。使用 `cv::cuda::GpuMat` 在顯存中直接完成：

* **並行縮放與色彩轉換**：利用 CUDA 核心進行線性插值縮放。
* **Zero-copy 維度變換**：透過 `cv::cuda::split` 實現 HWC 轉 CHW，並直接映射至 TensorRT 輸入指針，達成零拷貝傳輸。

### 2. 記憶體管理優化 (Advanced Memory Management)

* **Pinned Memory (Host-Locked)**：使用分頁鎖定記憶體優化 PCIe 頻寬利用率，降低 D2H 回傳延遲。
* **顯存池化 (Memory Pooling)**：所有預處理緩衝區與推論 Tensor 均在初始化時預分配，避免運行時 `cudaMalloc` 造成的延遲抖動（Jitter）。

### 3. 非同步串流同步 (Unified Async Streams)

利用 `cv::cuda::StreamAccessor` 將 TensorRT 與 OpenCV 的 CUDA 串流統一，確保「上傳-預處理-推論-下載」在單一指令流水線中非同步執行，極大化 GPU 利用率。

---

## 系統環境要求

> **警告：** 本專案針對特定硬體架構進行高度優化，可能難以在非指定環境直接運行。

* **OS:** Windows 11 (2026 Version)
* **GPU:** NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, SM 8.9)
* **Compiler:** MSVC 14.4+ (Visual Studio 2026)
* **Libraries:** * CUDA Toolkit 13.1 / cuDNN 9.18
* TensorRT 10.x
* OpenCV 4.13.0 (Self-compiled with CUDA & cuDNN support)



---

## 📂 專案結構

* `src/` - 核心推論邏輯與 `YoloDetector` 實作。
* `include/` - 高效能封裝標頭檔。
* `models/` - YOLO26n ONNX 模型與引擎生成腳本。
* `docs/` - `value.md` (詳細優化數值日誌) 與技術升級說明。

## 👤 作者

** leowu

---

### 💡 寫給評審/開發者的話

本引擎的 2.27ms 延遲不僅是數據，更是對 C++/CUDA 執行流深度調優的結果。如果你在嘗試運行時遇到 `Compatibility Error`，那是因為這台 laptop 的性能已被鎖定在 Sm 8.9 架構。

---
