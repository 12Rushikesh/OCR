# Container OCR System (YOLO + Qwen3-VL + OpenVINO)

This project performs container number OCR using:
- YOLO for container detection
- Qwen3-VL-2B for OCR
- OpenVINO (CPU) for optimized inference

## Features
- Accurate container OCR
- OpenVINO FP16 inference
- FastAPI backend
- Production-safe pipeline

## Model Download
Models are NOT included in this repo.

### Qwen3-VL-2B (OpenVINO)
Convert using:
```bash
optimum-cli export openvino \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --task image-text-to-text \
  --weight-format fp16 \
  --trust-remote-code \
  qwen3-vl-2b-fp16-ov
