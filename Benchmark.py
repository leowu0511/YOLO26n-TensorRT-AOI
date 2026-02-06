import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ===== 設定 =====
MODEL_PATH = "yolo26n.pt"
IMAGE_PATH = "test.jpg"      # 任一固定圖片
IMG_SIZE = 640
WARMUP = 20
RUNS = 100

assert torch.cuda.is_available()
device = "cuda"

# ===== 載入模型 =====
model = YOLO(MODEL_PATH)
model.to(device)
model.model.eval()

# ===== 固定影像 =====
img = cv2.imread(IMAGE_PATH)
assert img is not None
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# ===== 關閉梯度 =====
torch.set_grad_enabled(False)

# ===== Warm-up =====
for _ in range(WARMUP):
    model(img, verbose=False)

torch.cuda.synchronize()

# ===== 正式測試 =====
latencies = []

for _ in range(RUNS):
    t0 = time.perf_counter()
    model(img, verbose=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies = np.array(latencies)

# ===== 統計 =====
print("=== Python End-to-End Latency (Ultralytics) ===")
print(f"Mean    : {latencies.mean():.2f} ms")
print(f"Median  : {np.median(latencies):.2f} ms")
print(f"P90     : {np.percentile(latencies, 90):.2f} ms")
print(f"P95     : {np.percentile(latencies, 95):.2f} ms")
print(f"P99     : {np.percentile(latencies, 99):.2f} ms")
print(f"Min/Max : {latencies.min():.2f} / {latencies.max():.2f} ms")
