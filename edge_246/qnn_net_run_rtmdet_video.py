"""Run RTMDet QNN context binary on QCS6490 without ONNX Runtime.

The only accelerator call is QAIRT's qnn-net-run --retrieve_context. Inputs
and outputs use native INT8/UINT8 IO. Frames are processed in bounded chunks
so a long video does not exhaust board RAM.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from qnn_rtmdet_postprocess import draw, postprocess

MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)


def preprocess(image, size=640):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = ((canvas.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), scale


def numpy_dtype(name):
    # ONNX reports e.g. 'int8' and 'uint8'. Do not silently accept float IO.
    if name not in ("int8", "uint8"):
        raise ValueError(f"Expected INT8/UINT8 native QNN IO, got {name}")
    return np.dtype(name)


def quantize(x, encoding):
    dtype = numpy_dtype(encoding["dtype"])
    info = np.iinfo(dtype)
    q = np.rint(x / encoding["scale"] + encoding["zero_point"])
    return np.ascontiguousarray(np.clip(q, info.min, info.max).astype(dtype))


def read_heads(result: Path, encodings):
    """Read native INT8/UINT8 QNN outputs and dequantize them for NumPy NMS."""
    outputs = []
    for raw in result.glob("*.raw"):
        # qnn-net-run normally preserves the ONNX output name. Match the
        # filename first; use the raw byte size only as a portable fallback.
        name = raw.stem
        encoding = encodings.get(name)
        if encoding is None:
            matches = [e for e in encodings.values()
                       if raw.stat().st_size == np.prod(e["shape"]) * numpy_dtype(e["dtype"]).itemsize]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot identify QNN output {raw}; expected names {list(encodings)}")
            encoding = matches[0]
        dtype = numpy_dtype(encoding["dtype"])
        q = np.fromfile(raw, dtype=dtype)
        if q.size != np.prod(encoding["shape"]):
            raise RuntimeError(f"Wrong byte count for {raw}: got {q.size}, expected {np.prod(encoding['shape'])}")
        outputs.append(((q.astype(np.float32) - encoding["zero_point"]) * encoding["scale"]).reshape(encoding["shape"]))
    if len(outputs) != 6:
        raise RuntimeError(f"Expected 6 RTMDet head files in {result}; got {len(outputs)}.")
    return outputs


def run_net_run(args, input_list: Path, output_dir: Path):
    command = [args.qnn_net_run, "--backend", args.backend, "--retrieve_context", args.model,
               "--input_list", str(input_list), "--output_dir", str(output_dir),
               "--profiling_level", "basic", "--use_native_input_files", "--use_native_output_files"]
    print("QNN command:", " ".join(command))
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    print(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"qnn-net-run failed ({completed.returncode})")
    # Versions differ. Keep the raw log and extract a printed average if available.
    (output_dir / "qnn-net-run.log").write_text(completed.stdout, encoding="utf-8")
    found = re.search(r"(?:average|avg).*?(?:inference|execute).*?([0-9.]+)\s*(?:us|ms)", completed.stdout, re.I)
    return wall, (float(found.group(1)) / (1000 if "ms" in found.group(0).lower() else 1) if found else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="QNN context binary (.bin), not ONNX")
    p.add_argument("--io-json", required=True, help="IO encoding JSON generated beside the .bin by export script")
    p.add_argument("--video", required=True); p.add_argument("--out", default="rtmdet_qnn_int8.mp4")
    p.add_argument("--qnn-net-run", default="qnn-net-run"); p.add_argument("--backend", default="libQnnHtp.so")
    p.add_argument("--work-dir", default="qnn_video_work"); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--chunk-frames", type=int, default=30,
                   help="Frames per qnn-net-run invocation; lower this on a memory-constrained board.")
    p.add_argument("--max-frames", type=int, default=0); p.add_argument("--score-thr", type=float, default=.30)
    p.add_argument("--iou-thr", type=float, default=.50,
                   help="Class-agnostic NMS IoU threshold (lower = fewer overlapping boxes).")
    p.add_argument("--contain-thr", type=float, default=.85,
                   help="Drop a box if this fraction of it lies inside a higher-score box. "
                        "Use 1.0 to disable (e.g. if small objects in front of big ones get lost).")
    args = p.parse_args()
    qnn_executable = shutil.which(args.qnn_net_run) or (str(Path(args.qnn_net_run).resolve())
                                                        if Path(args.qnn_net_run).is_file() else None)
    if qnn_executable is None:
        raise FileNotFoundError(
            f"qnn-net-run not found: argument={args.qnn_net_run!r}, QAIRT={os.environ.get('QAIRT', '<unset>')!r}. "
            "Run: find \"${QAIRT:?set QAIRT first}\" -type f -name qnn-net-run -executable -print")
    args.qnn_net_run = qnn_executable
    if not Path(args.backend).is_file():
        raise FileNotFoundError(f"HTP backend not found: {args.backend!r}; pass the libQnnHtp.so found under QAIRT/lib.")
    io = json.loads(Path(args.io_json).read_text(encoding="utf-8"))
    input_encoding = io["input"]; output_encodings = io["outputs"]
    if args.chunk_frames < 1:
        raise ValueError("--chunk-frames must be at least 1")
    work = Path(args.work_dir)
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise FileNotFoundError(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
    total_frames = total_wall = total_post_seconds = 0.; execute_us_sum = 0.; execute_count = 0; warmed = False
    while not args.max_frames or total_frames < args.max_frames:
        frames, scales = [], []
        for _ in range(args.chunk_frames):
            if args.max_frames and total_frames + len(frames) >= args.max_frames: break
            ok, frame = cap.read()
            if not ok: break
            frames.append(frame); scales.append(preprocess(frame)[1])
        if not frames: break
        chunk = work / f"chunk_{int(total_frames):06d}"; inputs = chunk / "inputs"; results = chunk / "results"; inputs.mkdir(parents=True)
        raw_paths = []
        for i, frame in enumerate(frames):
            tensor, _ = preprocess(frame); raw = inputs / f"image_{i:06d}.raw"; quantize(tensor, input_encoding).tofile(raw); raw_paths.append(raw.resolve())
        input_list = chunk / "inputs.txt"; input_list.write_text("\n".join(map(str, raw_paths)) + "\n")
        # Warm once after the first preprocessed frame; exclude it from metrics.
        if args.warmup and not warmed:
            warmup_list = chunk / "warmup.txt"; warmup_list.write_text("\n".join([str(raw_paths[0])] * args.warmup) + "\n")
            run_net_run(args, warmup_list, chunk / "warmup_output"); warmed = True
        wall, average_us = run_net_run(args, input_list, results); total_wall += wall
        if average_us is not None: execute_us_sum += average_us * len(frames); execute_count += len(frames)
        started = time.perf_counter()
        # If qnn-net-run did not print an average time, fall back to wall-clock FPS
        # (includes process start-up and raw file IO, so it is lower than pure NPU FPS).
        label_fps = 1e6 / average_us if average_us else len(frames) / wall
        for i, (frame, scale) in enumerate(zip(frames, scales)):
            heads = read_heads(results / f"Result_{i}", output_encodings)
            boxes, scores, labels = postprocess(heads, scale, frame.shape[:2], 640,
                                                args.score_thr, args.iou_thr, 300, args.contain_thr)
            writer.write(draw(frame, boxes, scores, labels, label_fps))
        total_post_seconds += time.perf_counter() - started; total_frames += len(frames)
        shutil.rmtree(chunk)  # raw inputs and six output heads are no longer needed
    cap.release(); writer.release()
    if not total_frames: raise RuntimeError("No frames decoded")
    print(f"frames={int(total_frames)}; qnn-net-run wall FPS (includes raw IO)={total_frames / total_wall:.2f}")
    if execute_count:
        print(f"QNN reported execute FPS (pure NPU)={1e6 / (execute_us_sum / execute_count):.2f}")
    else:
        print("QNN did not print an average execute time; use qnn-profile-viewer for pure NPU latency.")
    print(f"postprocess+encode FPS={total_frames / total_post_seconds:.2f}; annotated video={args.out}")


if __name__ == "__main__":
    main()