# RTMDet trên thiết bị Edge (Qualcomm) – `edge_246` và `edge_252`
Model đã có sẵn trong folder """rtmdet""" trên cả 2 thiết bị

## 1. Cài đặt
Trên thiết bị `edge_252`, khởi động môi trường ảo
```bash
conda activate py312
```
Trên thiết bị `edge_252`, khởi động môi trường ảo
```bash
conda activate tracking
```

## 2. Chạy chương trình
Mỗi folder đã được cài đặt đường dẫn qnn cho riêng từng máy nên chỉ cần chạy file tương ứng với từng phiên bản model
```bash
bash run_rtmdet_tiny.sh #Model RTMDet bản tiny
bash run_rtmdet_s.sh #Model RTMDet bản s
bash run_rtmdet_m.sh #Model RTMDet bản m
```

## 3. Thay đổi video đầu vào và đầu ra

Khi có video mới, bạn cần chỉ cho chương trình biết đường dẫn của video đó. 
