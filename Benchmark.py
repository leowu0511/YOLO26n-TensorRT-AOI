import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ===== 設定 =====
MODEL_PATH = "yolo26n.pt"
IMAGE_PATH = "test.jpg"
IMG_SIZE = 640
WARMUP = 20
RUNS = 1000 

assert torch.cuda.is_available()
device = "cuda"

# ===== 載入模型 =====
model = YOLO(MODEL_PATH)
model.to(device)
model.model.eval()

# ===== 固定影像 =====
img = cv2.imread(IMAGE_PATH)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# ===== 關閉梯度 =====
torch.set_grad_enabled(False)

# ===== Warm-up =====
for _ in range(WARMUP):
    model(img_resized, verbose=False, device=device)
torch.cuda.synchronize()

# ===== 正式測試 =====
latencies = []
for i in range(RUNS):
    t0 = time.perf_counter()
    model(img_resized, verbose=False, device=device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies = np.array(latencies)
mean_latency = latencies.mean()
p95_latency = np.percentile(latencies, 95)
fps = 1000.0 / mean_latency

# ===== 格式對齊輸出 =====
print("\n========================================")
print("Python Ultralytics 效能報告 (End-to-End)")
print("----------------------------------------")
print(f"Mean Latency   : {mean_latency:.5f} ms")
print(f"P95 Latency    : {p95_latency:.4f} ms")
print(f"Min / Max      : {latencies.min():.4f} / {latencies.max():.4f} ms")
print(f"FPS (Average)  : {fps:.2f}")
print("========================================")
