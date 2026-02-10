# containerNoRead_GLM_OCR_Quantized_fixed.py
# Fast CPU pipeline: Dynamic Quantization + Fixed Resolution
# YOLO -> GLM-OCR (INT8) -> ISO 6346 validation
# ==========================================================

# CRITICAL: Set thread limits FIRST (before heavy imports)
import os
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))  # default 2 for laptops; try 4 for high-core CPUs
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)

import re
import time
import json
import base64
import threading
import numpy as np
from datetime import datetime
from typing import Optional, List

import cv2
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from huggingface_hub import snapshot_download

# ==========================================================
# WINDOWS / CPU PATCH FOR GLM-OCR
# ==========================================================
def fixed_get_imports(filename):
    """Prevents flash_attn errors on Windows/CPU by removing optional import."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# ==========================================================
# CPU CONFIGURATION
# ==========================================================
DEVICE = "cpu"
DTYPE = torch.float32  # use dtype arg when loading model

# ==========================================================
# CONFIG (QUANTIZATION-OPTIMIZED)
# ==========================================================
GLM_MODEL_ID = "zai-org/GLM-OCR"
GLM_MODEL_DIR = r"E:\ocr\flocr\models\GLM-OCR"
YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

# CRITICAL OPTIMIZATION SETTINGS
GLM_IMAGE_SIZE = 640          # Fixed resolution
OCR_MAX_TOKENS = 12           # max tokens to generate (container = 11 chars)
CONCURRENT_OCR_CALLS = 1      # single-threaded OCR generation
YOLO_BATCH_SIZE = 1
MAX_IMAGE_SIZE = 1280         # For YOLO preprocessing only

# Quantization & precision flags
ENABLE_QUANTIZATION = True    # Enable INT8 dynamic quantization (recommended)
ENABLE_BFLOAT16_AMP = False   # Disabled when quantization is used (they conflict)

# ==========================================================
# ENSURE GLM-OCR MODEL IS DOWNLOADED
# ==========================================================
def ensure_glm():
    if not os.path.exists(GLM_MODEL_DIR) or not os.listdir(GLM_MODEL_DIR):
        print(f"[GLM] ⬇️ Downloading GLM-OCR from {GLM_MODEL_ID}...")
        snapshot_download(repo_id=GLM_MODEL_ID, local_dir=GLM_MODEL_DIR)
        print(f"[GLM] ✅ Download complete")

print(f"\n{'='*60}")
print(f"[SYSTEM] 🚀 Initializing Quantized CPU pipeline (fixed)")
print(f"{'='*60}\n")

ensure_glm()

glm_processor = None
glm_model = None

# ==========================================================
# LOAD AND QUANTIZE GLM-OCR
# ==========================================================
print("[GLM] 📦 Loading processor with CPU-safe patch...")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    glm_processor = AutoProcessor.from_pretrained(
        GLM_MODEL_DIR,
        trust_remote_code=True
    )

print("[GLM] 🧠 Loading model to CPU...")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    # use 'dtype' to avoid deprecation warning
    glm_model = AutoModelForImageTextToText.from_pretrained(
        GLM_MODEL_DIR,
        dtype=DTYPE,
        trust_remote_code=True
    )
    glm_model = glm_model.to(DEVICE)
    glm_model.eval()

# If quantization is enabled, ensure BF16 is disabled (they conflict)
if ENABLE_QUANTIZATION and ENABLE_BFLOAT16_AMP:
    print("[GLM] ⚠️ INT8 quantization enabled -> disabling BF16 autocast (incompatible)")
    ENABLE_BFLOAT16_AMP = False

# ==========================================================
# APPLY DYNAMIC QUANTIZATION (INT8)
# ==========================================================
if ENABLE_QUANTIZATION:
    try:
        print("[GLM] ⚡ Applying INT8 dynamic quantization to Linear layers...")
        print("[GLM] 🔧 This will take 20-60 seconds (one-time optimization)...")
        quantization_start = time.time()
        glm_model = torch.quantization.quantize_dynamic(
            glm_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        quantization_time = round(time.time() - quantization_start, 2)
        print(f"[GLM] ✅ Quantization complete in {quantization_time}s")
        print("[GLM] 💾 Quantized model ready for inference")
    except Exception as e:
        print(f"[GLM] ⚠️ Quantization failed: {e}")
        print("[GLM] ⚠️ Continuing with FP32 model")
        ENABLE_QUANTIZATION = False
else:
    print("[GLM] ⚠️ Quantization disabled - using standard FP32")

# ==========================================================
# BF16 support check (only relevant if BF16 requested and quantization disabled)
# ==========================================================
if ENABLE_BFLOAT16_AMP:
    try:
        # quick check using context manager (no ops inside) to ensure autocast exists
        from contextlib import nullcontext
        _ = torch.amp.autocast  # existence check
        # Leave ENABLE_BFLOAT16_AMP as True; any real incompatibility will be caught at generation
    except Exception:
        print("[GLM] ⚠️ CPU bfloat16 autocast not available; disabling BF16")
        ENABLE_BFLOAT16_AMP = False

# ==========================================================
# SYSTEM INFO
# ==========================================================
print(f"\n{'='*60}")
print(f"[SYSTEM] Configuration Summary")
print(f"{'='*60}")
print(f"Device: {DEVICE}")
print(f"Threads (NUM_THREADS env): {NUM_THREADS}")
print(f"GLM input size: {GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE}px (fixed)")
print(f"Quantization: {'INT8 enabled' if ENABLE_QUANTIZATION else 'Disabled'}")
print(f"BF16 autocast: {'Enabled' if ENABLE_BFLOAT16_AMP else 'Disabled'}")
print(f"Max tokens: {OCR_MAX_TOKENS}")
print(f"Cache: Disabled (use_cache=False)")
print(f"Sampling: Greedy (do_sample=False)")
print(f"{'='*60}\n")

# Thread-safe semaphore
glm_semaphore = threading.Semaphore(CONCURRENT_OCR_CALLS)

# ==========================================================
# DIRECTORY SETUP
# ==========================================================
def today_folder(base: str) -> str:
    folder_path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

for directory in [
    "container_results/received_frames",
    "container_results/yolo_detections",
    "container_results/glm_images",
    "container_results/success"
]:
    os.makedirs(directory, exist_ok=True)

# ==========================================================
# PYDANTIC MODELS
# ==========================================================
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# ==========================================================
# LOAD YOLO MODELS
# ==========================================================
def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        try:
            model.fuse()
        except Exception:
            pass
        try:
            model.to("cpu")
        except Exception:
            pass
        print(f"[{tag}] ✅ Loaded on CPU: {path}")
        return model
    except Exception as e:
        print(f"[{tag}] ❌ Failed to load ({path}): {e}")
        return None

yolo_primary = load_yolo(YOLO_PRIMARY, "YOLO-PRIMARY")
yolo_secondary = load_yolo(YOLO_SECONDARY, "YOLO-SECONDARY")

# ==========================================================
# UTILS: base64 -> cv2
# ==========================================================
def base64_to_cv2(b64_string: str) -> np.ndarray:
    try:
        if b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]
        image_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None - invalid image data")
        return frame
    except Exception as e:
        print(f"[ERROR] base64_to_cv2 failed: {e}")
        raise

# ==========================================================
# SAVE HELPERS
# ==========================================================
def save_received_frame(frame: np.ndarray, kalmar_id: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(
        today_folder("container_results/received_frames"),
        f"{kalmar_id}_{timestamp}.jpg"
    )
    cv2.imwrite(path, frame)
    print(f"[SAVE] 📷 Received -> {path}")
    return path

def save_yolo_detection(frame: np.ndarray, kalmar_id: str, detected: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    status = "detected" if detected else "no_detection"
    path = os.path.join(
        today_folder("container_results/yolo_detections"),
        f"{kalmar_id}_{timestamp}_{status}.jpg"
    )
    cv2.imwrite(path, frame)
    print(f"[SAVE] 🎯 YOLO -> {path}")
    return path

def save_glm_crop(pil_image: Image.Image, kalmar_id: str, region_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(
        today_folder("container_results/glm_images"),
        f"{kalmar_id}_{timestamp}_r{region_idx}.jpg"
    )
    pil_image.save(path, "JPEG", quality=95)
    print(f"[SAVE] 🧠 GLM input -> {path}")
    return path

def save_success_result(kalmar_id: str, container_number: str, frame: np.ndarray):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = today_folder("container_results/success")

    image_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)

    json_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "kalmar_id": kalmar_id,
            "container_number": container_number,
            "iso6346_valid": iso6346_checksum(container_number),
            "timestamp": timestamp
        }, f, indent=2)

    print(f"[SAVE] ✅ SUCCESS -> {image_path} + {json_path}")
    return image_path, json_path

# ==========================================================
# ISO 6346 CHECKSUM VALIDATION
# ==========================================================
ISO_CHAR_MAP = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19,
    'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
    'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
}

def iso6346_checksum(container: str) -> bool:
    if not container or len(container) != 11:
        return False
    if not (container[:4].isalpha() and container[4:].isdigit()):
        return False
    total = 0
    try:
        for i in range(10):
            ch = container[i]
            val = int(ch) if ch.isdigit() else ISO_CHAR_MAP.get(ch, 0)
            total += val * (2 ** i)
        remainder = total % 11
        check_digit = 0 if remainder == 10 else remainder
        return check_digit == int(container[10])
    except Exception:
        return False

def extract_container_from_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    candidates = re.findall(r"[A-Z]{4}\d{7}", cleaned)
    for candidate in candidates:
        if iso6346_checksum(candidate):
            print(f"[VALIDATION] ✅ Valid container found: {candidate}")
            return candidate
    print(f"[VALIDATION] ❌ No valid container in: {repr(text)}")
    return ""

# ==========================================================
# ASSISTANT RESPONSE EXTRACTION
# ==========================================================
def extract_assistant_response(full_text: str) -> str:
    if full_text is None:
        return ""
    if "<|im_start|>assistant" in full_text:
        parts = full_text.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[-1].split("<|im_end|>")[0].strip()
            return response
    if "assistant\n" in full_text.lower():
        parts = full_text.lower().split("assistant\n")
        if len(parts) > 1:
            response = full_text.split("assistant\n")[-1].strip()
            return response
    patterns_to_remove = [
        r"<\|im_start\|>.*?<\|im_end\|>",
        r"system:.*?assistant:.*?",
    ]
    cleaned = full_text
    for p in patterns_to_remove:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

# ==========================================================
# GLM-OCR (QUANTIZED + OPTIMIZED)
# ==========================================================
@torch.inference_mode()
def glm_ocr_raw(pil_image: Image.Image) -> str:
    """
    Quantized GLM-OCR with optimized settings.
    - Fixed resolution
    - INT8 quantized Linear layers (if enabled)
    - Greedy decoding with cache disabled
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize to fixed resolution
    if pil_image.size != (GLM_IMAGE_SIZE, GLM_IMAGE_SIZE):
        pil_image = pil_image.resize((GLM_IMAGE_SIZE, GLM_IMAGE_SIZE), Image.BILINEAR)
        print(f"[GLM] ⚡ Resized to {GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE} (fixed)")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract the container number from this image. Output only the 11-character code without spaces or markdown."}
            ]
        }
    ]

    prompt_text = glm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = glm_processor(
        images=pil_image,
        text=prompt_text,
        return_tensors="pt"
    ).to(DEVICE)

    inputs.pop("token_type_ids", None)

    start_time = time.time()
    try:
        # Choose generation path: BF16 autocast only if explicitly enabled AND quantization is disabled
        if ENABLE_BFLOAT16_AMP and not ENABLE_QUANTIZATION:
            # use recommended torch.amp.autocast interface
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                generated_ids = glm_model.generate(
                    **inputs,
                    max_new_tokens=OCR_MAX_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                    pad_token_id=glm_processor.tokenizer.pad_token_id,
                    eos_token_id=glm_processor.tokenizer.eos_token_id
                )
        else:
            # standard FP32 / INT8 (quantized) path
            generated_ids = glm_model.generate(
                **inputs,
                max_new_tokens=OCR_MAX_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=False,
                pad_token_id=glm_processor.tokenizer.pad_token_id,
                eos_token_id=glm_processor.tokenizer.eos_token_id
            )

        elapsed = round(time.time() - start_time, 3)

        input_len = inputs.input_ids.shape[1]
        full_text = glm_processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
        )

        raw_extracted = extract_assistant_response(full_text)

        optimization_status = []
        if ENABLE_QUANTIZATION:
            optimization_status.append("INT8")
        if ENABLE_BFLOAT16_AMP:
            optimization_status.append("BF16")
        opt_str = "+".join(optimization_status) if optimization_status else "FP32"

        print(f"[GLM] ⚡ Inference time: {elapsed}s ({opt_str} optimizations)")
        print(f"[GLM] 📊 Resolution: {GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE} | Threads: {NUM_THREADS}")
        print(f"[GLM] RAW-DECODE: {repr(full_text)}")
        print(f"[GLM] EXTRACTED: {repr(raw_extracted)}")

        validated_container = extract_container_from_text(raw_extracted)

        if validated_container:
            print(f"[GLM] ✅ VALIDATED: {validated_container}")
            return validated_container
        else:
            print(f"[GLM] ❌ No valid container found")
            return ""
    except Exception as e:
        print(f"[GLM] ❌ Generation Error: {e}")
        return ""

def safe_glm_ocr_raw(pil_image: Image.Image) -> str:
    with glm_semaphore:
        return glm_ocr_raw(pil_image)

# ==========================================================
# YOLO DETECTION (CPU)
# ==========================================================
def detect_with_yolo(model, frame: np.ndarray):
    annotated = frame.copy()
    regions = []

    if model is None:
        return False, [], annotated

    try:
        results = model(frame, conf=0.15, verbose=False)[0]
    except Exception as e:
        print(f"[YOLO] ❌ Inference error: {e}")
        return False, [], annotated

    if not results.boxes or len(results.boxes) == 0:
        return False, [], annotated

    best_box = max(results.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf)

    crop = frame[y1:y2, x1:x2]

    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return bool(regions), regions, annotated

# ==========================================================
# MAIN PROCESSING PIPELINE
# ==========================================================
def process_container_image(base64_image: str, kalmar_id: str) -> dict:
    start_time = time.time()

    try:
        frame = base64_to_cv2(base64_image)
        save_received_frame(frame, kalmar_id)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return {
            "success": False,
            "error": "Invalid image data",
            "processing_time": round(time.time() - start_time, 3)
        }

    detected, regions, annotated = detect_with_yolo(yolo_primary, frame)

    if not detected and yolo_secondary is not None:
        print("[PIPELINE] 🔄 Trying secondary YOLO...")
        detected, regions, annotated = detect_with_yolo(yolo_secondary, frame)

    save_yolo_detection(annotated, kalmar_id, detected)

    if not detected:
        print("[PIPELINE] ❌ No YOLO detection")
        return {
            "success": False,
            "processing_time": round(time.time() - start_time, 3)
        }

    for idx, region_pil in enumerate(regions, 1):
        save_glm_crop(region_pil, kalmar_id, idx)

        validated_container = safe_glm_ocr_raw(region_pil)
        print(f"[OCR] Region {idx} => VALIDATED: {repr(validated_container)}")

        if validated_container and len(validated_container) == 11:
            save_success_result(kalmar_id, validated_container, frame)

            return {
                "success": True,
                "container_number": validated_container,
                "iso6346_valid": True,
                "device": f"CPU (Quantized INT8)" if ENABLE_QUANTIZATION else "CPU (FP32)",
                "processing_time": round(time.time() - start_time, 3)
            }

    return {
        "success": False,
        "processing_time": round(time.time() - start_time, 3)
    }

# ==========================================================
# FASTAPI APP
# ==========================================================
app = FastAPI(
    title="Container OCR GLM API (Quantized Fixed)",
    description="YOLO + GLM-OCR with ISO 6346 validation — INT8 quantization + optimized inference",
    version="2.3-Quantized-fixed"
)

@app.get("/")
def root():
    opt_features = []
    if ENABLE_QUANTIZATION:
        opt_features.append("INT8 quantized Linear layers")
    if ENABLE_BFLOAT16_AMP:
        opt_features.append("BF16 CPU autocast")

    return {
        "name": "Container OCR GLM API (Quantized Fixed)",
        "version": "2.3-Quantized-fixed",
        "model": "GLM-OCR",
        "device": DEVICE,
        "optimizations": opt_features,
        "features": [
            "Anti-hallucination prompt",
            f"Limited token generation ({OCR_MAX_TOKENS})",
            "ISO 6346 checksum validation",
            "Regex extraction [A-Z]{4}\\d{7}",
            f"Fixed {GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE} resolution",
            "Greedy decoding (no sampling)",
            "Cache disabled for speed",
            f"{NUM_THREADS}-thread CPU execution"
        ],
        "performance": {
            "quantization": "INT8 dynamic" if ENABLE_QUANTIZATION else "None",
            "autocast": "BF16" if ENABLE_BFLOAT16_AMP else "Disabled",
            "image_size": f"{GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE}",
            "max_tokens": OCR_MAX_TOKENS,
            "threads": NUM_THREADS,
            "cache": "Disabled",
            "sampling": "Greedy"
        },
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding",
            "/api/benchmark": "POST - Benchmark GLM-OCR inference speed"
        }
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "running",
        "device": DEVICE,
        "quantization": ENABLE_QUANTIZATION,
        "autocast": ENABLE_BFLOAT16_AMP,
        "models": {
            "glm_ocr": glm_model is not None,
            "yolo_primary": yolo_primary is not None,
            "yolo_secondary": yolo_secondary is not None
        }
    }

@app.post("/api/test-decode")
async def test_decode(data: dict):
    try:
        b64 = data.get("image_base64", "")
        frame = base64_to_cv2(b64)
        return {
            "status": "success",
            "message": "Image decoded successfully",
            "shape": frame.shape,
            "dtype": str(frame.dtype),
            "device": DEVICE
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/benchmark")
async def benchmark_glm(data: dict):
    """Benchmark GLM-OCR inference speed with a test image."""
    try:
        b64 = data.get("image_base64", "")
        frame = base64_to_cv2(b64)
        crop_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)

        # Warm up (small) to stabilize first-call overhead
        for _ in range(1):
            _ = glm_ocr_raw(pil_image)

        start_time = time.time()
        result = glm_ocr_raw(pil_image)
        inference_time = round(time.time() - start_time, 3)

        opt_status = []
        if ENABLE_QUANTIZATION:
            opt_status.append("INT8")
        if ENABLE_BFLOAT16_AMP:
            opt_status.append("BF16")

        return {
            "status": "success",
            "inference_time_seconds": inference_time,
            "result": result,
            "optimizations": "+".join(opt_status) if opt_status else "FP32",
            "image_size": f"{GLM_IMAGE_SIZE}x{GLM_IMAGE_SIZE}",
            "threads": NUM_THREADS
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    print(f"\n{'='*60}")
    print(f"[API] 📥 Kalmar ID: {event.kalmar_id}")
    print(f"[API] 🕐 Time: {event.timestamp}")
    print(f"[API] 🖥️ Backend: {'Quantized CPU (INT8)' if ENABLE_QUANTIZATION else 'CPU (FP32)'}")
    print(f"{'='*60}")

    images = event.images or ([event.image_base64] if event.image_base64 else [])

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images provided. Include 'images' array or 'image_base64' field."
        )

    print(f"[API] 📷 Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] 🔍 Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id)

        if result.get("success"):
            print(f"[API] ✅ Container: {result['container_number']}")
            print(f"[API] ⏱️ Total time: {result['processing_time']}s")

            return {
                "status": "container_found",
                "container_number": result["container_number"],
                "iso6346_valid": result["iso6346_valid"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"],
                "device": result.get("device", "CPU")
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {
        "status": "no_container",
        "kalmar_id": event.kalmar_id,
        "device": "CPU"
    }

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Container OCR GLM API — Quantized CPU Mode (fixed)")
    print(f"⚡ INT8 quantization={'enabled' if ENABLE_QUANTIZATION else 'disabled'} | BF16={'enabled' if ENABLE_BFLOAT16_AMP else 'disabled'}")
    print(f"⚡ NUM_THREADS={NUM_THREADS} | GLM_IMAGE_SIZE={GLM_IMAGE_SIZE}")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
