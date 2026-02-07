import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
import glob

class YoloDetectorPy:
    """
    Python 版 YOLO 推論引擎 (對標 C++ Production Mode)
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        # 載入並反序列化引擎
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 分配記憶體緩衝區
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        # TensorRT 10.x 語法：遍歷所有 Tensor
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # 計算所需空間並分配 Pinned Memory
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        return inputs, outputs, bindings, stream

    def detect(self, img):
        # 1. 影像預處理 (CPU 端執行)
        # 此部分模擬標準 Python 開發流程，用於對比 C++ GPU 預處理的效能提升
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1)).copy()
        
        # 2. 資料傳輸：Host to Device (H2D)
        self.inputs[0]['host'][:] = img_chw.ravel()
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. 執行推論
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for outp in self.outputs:
            self.context.set_tensor_address(outp['name'], int(outp['device']))
            
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 4. 資料傳輸：Device to Host (D2H)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

# --- 測試主程式 ---
def main():
    print("[System] Initializing Python Inference Baseline (Target: RTX 4060)...")
    
    # 路徑設定 (與 C++ 版對齊)
    engine_path = "../pcb_aoi.engine"
    image_dir = "../Test_Set_Raw/images/"
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))[:100]

    if not image_files:
        print("[Error] Test dataset not found.")
        return

    # 預載入影像至 RAM
    print(f"[Storage] Pre-loading {len(image_files)} samples to RAM...")
    preloaded_images = [cv2.imread(f) for f in image_files]
    
    try:
        detector = YoloDetectorPy(engine_path)
    except Exception as e:
        print(f"[Error] Engine initialization failed: {e}")
        return

    # 執行熱機 (Warm-up)
    for _ in range(50):
        detector.detect(preloaded_images[0])

    # 執行 1000 次壓力測試
    print("\n[Mode] Launching Python Baseline (CPU Pre-processing)...")
    latencies = []
    for i in range(1000):
        start = time.perf_counter()
        detector.detect(preloaded_images[i % len(preloaded_images)])
        latencies.append((time.perf_counter() - start) * 1000)

    # 效能數據統計
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\n========================================")
    print("  PYTHON BASELINE PERFORMANCE REPORT")
    print("----------------------------------------")
    print(f"[Python] Mean: {avg_latency:.5f} ms | FPS: {1000/avg_latency:.3f}")
    print(f"[Python] P95:  {p95_latency:.5f} ms")
    print("========================================")
    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()
