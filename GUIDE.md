# Hướng dẫn Build, Tạo Data và Train BEVFusion

## Mục lục
1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Build Docker Image](#2-build-docker-image)
3. [Chạy Container](#3-chạy-container)
4. [Chuẩn bị Dataset NuScenes](#4-chuẩn-bị-dataset-nuscenes)
5. [Tạo Data Info (create_data)](#5-tạo-data-info-create_data)
6. [Tải Pretrained Models](#6-tải-pretrained-models)
7. [Train](#7-train)
8. [Đánh giá (Evaluation)](#8-đánh-giá-evaluation)
9. [Cấu trúc thư mục tham khảo](#9-cấu-trúc-thư-mục-tham-khảo)

---

## 1. Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| OS | Linux (Ubuntu 20.04) |
| CUDA | 12.1 |
| cuDNN | 8.9 |
| Docker | ≥ 20.10 |
| NVIDIA Container Toolkit | bắt buộc |
| RAM | ≥ 32 GB |
| GPU VRAM | ≥ 16 GB (train đầy đủ cần ≥ 32 GB) |
| Dung lượng ổ đĩa | ≥ 200 GB (NuScenes full ~530 GB) |

---

## 2. Build Docker Image

Chạy lệnh sau từ thư mục gốc của dự án (nơi có `Dockerfile`):

```bash
cd /path/to/test-bevfusion

docker build -t bevfusion:cu121 .
```

> **Lưu ý:** Quá trình build sẽ mất **30–60 phút** lần đầu do cần tải và biên dịch nhiều thư viện CUDA (mmcv, spconv, cumm, các CUDA extension của dự án).

Build lại từ đầu (bỏ cache — dùng khi cần reinstall sạch):

```bash
docker build --no-cache -t bevfusion:cu121 .
```

---

## 3. Chạy Container

Thay `/path/to/nuscenes` bằng đường dẫn thực tế đến thư mục NuScenes trên máy host:

```bash
docker run --gpus all -it --rm \
  -v /path/to/nuscenes:/workspace/bevfusion/data/nuscenes \
  -v /path/to/pretrained:/workspace/bevfusion/pretrained \
  bevfusion:cu121
```

Sau khi vào container, môi trường conda `bevfusion` đã được tự động activate và thư mục làm việc là `/workspace`:

```bash
# Di chuyển vào thư mục dự án
cd /workspace/bevfusion
```

---

## 4. Chuẩn bị Dataset NuScenes

Cấu trúc thư mục dữ liệu **bắt buộc** phải như sau trước khi chạy `create_data`:

```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/   ← full dataset
│   (hoặc v1.0-mini/ ← mini dataset để test nhanh)
```

Mount dữ liệu vào đúng đường dẫn khi chạy container (xem bước 3), hoặc copy thủ công vào trong container.

---

## 5. Tạo Data Info (`create_data`)

Lệnh này tạo các file `.pkl` chứa annotation info và groundtruth database — bắt buộc phải chạy trước khi train.

Chạy bên trong container, từ thư mục `/workspace/bevfusion`:

### Full dataset (v1.0-trainval)

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval
```

### Mini dataset (để test nhanh)

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini
```

### Kết quả tạo ra

Sau khi chạy xong, thư mục `data/nuscenes/` sẽ có thêm:

```
data/nuscenes/
├── nuscenes_dbinfos_train.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_gt_database/
    └── *.bin
```

---

## 6. Tải Pretrained Models

Chạy script tải tự động (cần kết nối internet trong container):

```bash
bash tools/download_pretrained.sh
```

Hoặc tải thủ công và đặt vào thư mục `pretrained/`:

| File | Dùng cho |
|---|---|
| `swint-nuimages-pretrained.pth` | Camera backbone (bắt buộc khi train camera/fusion) |
| `lidar-only-det.pth` | Pretrained LiDAR (dùng khi finetune Camera+LiDAR fusion) |
| `bevfusion-det.pth` | Đánh giá detection (Camera+LiDAR) |
| `bevfusion-seg.pth` | Đánh giá segmentation |

---

## 7. Train

> Cú pháp chung:
> ```bash
> torchpack dist-run -np <số GPU> python tools/train.py <config> [tùy chọn]
> ```

### 7.1 LiDAR-only Detection

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

### 7.2 Camera-only Detection

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth
```

### 7.3 BEVFusion Detection (Camera + LiDAR) ⭐ — mô hình chính

Khuyến nghị: train LiDAR-only trước (7.1), rồi dùng checkpoint đó để finetune fusion:

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

### 7.4 BEVFusion Detection (Camera + RADAR)

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/dlss.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth
```

### 7.5 BEVFusion Detection (Camera + LiDAR + RADAR)

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

### 7.6 LiDAR-only Segmentation

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

### 7.7 BEVFusion Segmentation (Camera + LiDAR)

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth
```

### Multi-GPU (ví dụ 2 GPU)

Thay `-np 1` thành `-np 2` (hoặc số GPU có sẵn):

```bash
torchpack dist-run -np 2 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
    pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth
```

### Resume training bị gián đoạn

Thêm `--load_from` trỏ vào checkpoint epoch cuối cùng trong thư mục `runs/`:

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from runs/<run_dir>/epoch_<N>.pth
```

---

## 8. Đánh giá (Evaluation)

> Cú pháp chung:
> ```bash
> torchpack dist-run -np <số GPU> python tools/test.py <config> <checkpoint> --eval <loại>
> ```
> - `--eval bbox` : đánh giá detection
> - `--eval map`  : đánh giá segmentation

### Detection (Camera + LiDAR)

```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --eval bbox
```

### Detection (Camera + RADAR)

```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/nuscenes/det/centerhead/lssfpn/camera+radar/resnet50/dlss.yaml \
    pretrained/bevfusion-det-radar.pth \
    --eval bbox
```

### Segmentation

```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
    pretrained/bevfusion-seg.pth \
    --eval map
```

---

## 9. Cấu trúc thư mục tham khảo

```
test-bevfusion/
├── Dockerfile                  ← Build image
├── configs/
│   └── nuscenes/
│       ├── det/
│       │   ├── transfusion/secfpn/
│       │   │   ├── lidar/voxelnet_0p075.yaml         ← LiDAR-only det
│       │   │   ├── camera+lidar/swint_v0p075/
│       │   │   │   └── convfuser.yaml                ← Camera+LiDAR fusion det ⭐
│       │   │   └── camera+lidar+radar/swint_v0p075/
│       │   │       └── convfuser.yaml                ← Trimodal fusion det
│       │   └── centerhead/lssfpn/
│       │       ├── camera/256x704/swint/default.yaml ← Camera-only det
│       │       └── camera+radar/resnet50/dlss.yaml   ← Camera+RADAR det
│       └── seg/
│           ├── lidar-centerpoint-bev128.yaml         ← LiDAR-only seg
│           ├── camera-bev256d2.yaml                  ← Camera-only seg
│           └── fusion-bev256d2-lss.yaml              ← Camera+LiDAR fusion seg
├── data/
│   └── nuscenes/               ← Dataset + các file .pkl sau create_data
├── pretrained/                 ← Các file .pth pretrained
├── tools/
│   ├── create_data.py          ← Tạo data info
│   ├── train.py                ← Train
│   ├── test.py                 ← Evaluate
│   └── download_pretrained.sh  ← Tải pretrained models
└── mmdet3d/                    ← Source code custom mmdet3d
```
