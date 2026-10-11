# ---------- Cấu hình QAIRT ----------
export QAIRT=/home/ubuntu/edge_artifacts/qairt_versions/2.50.0.260828/runtime-only/qairt/2.50.0.260828
T=aarch64-ubuntu-gcc9.4

export LD_LIBRARY_PATH=$QAIRT/lib/$T:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="$QAIRT/lib/hexagon-v68/unsigned"

# ---------- Đường dẫn ----------
MODEL="rtmdet_s_qcs6490.bin"
IO_JSON="rtmdet_s_qcs6490_io.json"
VIDEO="/home/ubuntu/rtmdet/data/video.mp4"
OUT="output_s.mp4"
QNN_NET_RUN="/home/ubuntu/edge_artifacts/qairt_versions/2.50.0.260828/runtime-only/qairt/2.50.0.260828/bin/aarch64-ubuntu-gcc9.4/qnn-net-run"
BACKEND="/home/ubuntu/edge_artifacts/qairt_versions/2.50.0.260828/runtime-only/qairt/2.50.0.260828/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so"
WARMUP=30

# ---------- Chạy ----------
python qnn_net_run_rtmdet_video.py \
    --model "$MODEL" \
    --io-json "$IO_JSON" \
    --video "$VIDEO" \
    --out "$OUT" \
    --qnn-net-run "$QNN_NET_RUN" \
    --backend "$BACKEND" \
    --warmup "$WARMUP"