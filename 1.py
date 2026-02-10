# ==========================================================
# containerNoRead_GLM_OCR_CPU.py
# CPU-only pipeline: YOLO -> GLM-OCR -> ISO 6346 validation
# Optimized for laptops / Windows / CPU-only environments
# ==========================================================

import os
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

# Force CPU dtype
DTYPE = torch.float32

# Set safe CPU thread limits (Windows-friendly)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

# ==========================================================
# CONFIG (CPU-ADAPTED)
# ==========================================================
GLM_MODEL_ID = "zai-org/GLM-OCR"
GLM_MODEL_DIR = r"E:\ocr\flocr\models\GLM-OCR"
YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

# CPU-friendly performance settings
OCR_MAX_TOKENS = 12
MAX_IMAGE_SIZE = 1280         # smaller for CPU memory
CONCURRENT_OCR_CALLS = 1      # single threaded CPU inference
ENABLE_TORCH_COMPILE = False  # not used for CPU in this config
YOLO_BATCH_SIZE = 1

# ==========================================================
# ENSURE GLM-OCR MODEL IS DOWNLOADED (local cache)
# ==========================================================
def ensure_glm():
    if not os.path.exists(GLM_MODEL_DIR) or not os.listdir(GLM_MODEL_DIR):
        print(f"[GLM] ⬇️ Downloading GLM-OCR from {GLM_MODEL_ID}...")
        snapshot_download(repo_id=GLM_MODEL_ID, local_dir=GLM_MODEL_DIR)
        print(f"[GLM] ✅ Download complete")

print(f"\n{'='*60}")
print(f"[SYSTEM] 🚀 Initializing CPU-only pipeline")
print(f"{'='*60}\n")

ensure_glm()

glm_processor = None
glm_model = None

# Apply CPU-safe patch and load model
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    print("[GLM] 📦 Loading processor (CPU)...")
    glm_processor = AutoProcessor.from_pretrained(
        GLM_MODEL_DIR,
        trust_remote_code=True
    )

    print("[GLM] 🧠 Loading model to CPU (this may take a moment)...")
    glm_model = AutoModelForImageTextToText.from_pretrained(
        GLM_MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True
    )

    # Move to CPU and set eval mode
    glm_model = glm_model.to(DEVICE)
    glm_model.eval()

    # Try enabling gradient checkpointing if available (can save memory)
    try:
        if hasattr(glm_model, "gradient_checkpointing_enable"):
            glm_model.gradient_checkpointing_enable()
            print("[GLM] ✅ Gradient checkpointing enabled (if supported)")
    except Exception:
        pass

print("[GLM] ✅ GLM-OCR loaded (CPU)")

# No torch.compile for CPU path
if ENABLE_TORCH_COMPILE:
    print("[GLM] ⚠️ Torch compile disabled for CPU configuration.")
    ENABLE_TORCH_COMPILE = False

# ==========================================================
# SYSTEM INFO (CPU)
# ==========================================================
print(f"\n{'='*60}")
print(f"[SYSTEM] Configuration Summary")
print(f"{'='*60}")
print(f"Device: {DEVICE}")
print(f"Dtype: {DTYPE}")
print(f"Max tokens: {OCR_MAX_TOKENS}")
print(f"Concurrent calls: {CONCURRENT_OCR_CALLS}")
print(f"Max image size: {MAX_IMAGE_SIZE}px")
print(f"YOLO batch size: {YOLO_BATCH_SIZE}")
print(f"Torch compile: {ENABLE_TORCH_COMPILE}")
print(f"{'='*60}\n")

# Thread-safe semaphore for concurrent OCR calls (CPU single-threaded by default)
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
# LOAD YOLO MODELS (CPU)
# ==========================================================
def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        # fuse may optimize but is safe on CPU
        try:
            model.fuse()
        except Exception:
            pass

        # Ensure model uses CPU
        try:
            model.to("cpu")
        except Exception:
            # some ultralytics versions don't require explicit to('cpu')
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
# GLM-OCR (CPU) - single-threaded by default
# ==========================================================
def glm_ocr_raw(pil_image: Image.Image) -> str:
    """
    CPU GLM-OCR. Returns validated 11-character container code or empty string.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize for CPU memory limits
    if max(pil_image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print(f"[GLM] ⚡ Resized to {new_size}")

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
        with torch.no_grad():
            # CPU generation (no mixed precision)
            generated_ids = glm_model.generate(
                **inputs,
                max_new_tokens=OCR_MAX_TOKENS,
                min_new_tokens=11,
                do_sample=False,
                num_beams=1,
                pad_token_id=glm_processor.tokenizer.pad_token_id,
                eos_token_id=glm_processor.tokenizer.eos_token_id,
                use_cache=True
            )

        input_len = inputs.input_ids.shape[1]
        full_text = glm_processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
        )

        elapsed = round(time.time() - start_time, 3)

        raw_extracted = extract_assistant_response(full_text)

        print(f"[GLM] ⏱️ Time: {elapsed}s (CPU)")
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
    """YOLO detection on CPU."""
    annotated = frame.copy()
    regions = []

    if model is None:
        return False, [], annotated

    # YOLO inference (CPU)
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
                "device": DEVICE,
                "processing_time": round(time.time() - start_time, 3)
            }

    return {
        "success": False,
        "processing_time": round(time.time() - start_time, 3)
    }

# ==========================================================
# FASTAPI APP (CPU)
# ==========================================================
app = FastAPI(
    title="Container OCR GLM API (CPU)",
    description="YOLO + GLM-OCR with ISO 6346 validation — CPU-only adaptive",
    version="2.1-CPU"
)

@app.get("/")
def root():
    return {
        "name": "Container OCR GLM API (CPU)",
        "version": "2.1-CPU",
        "model": "GLM-OCR",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "gpu_info": "No GPU available (CPU-only)",
        "features": [
            "Anti-hallucination prompt",
            "Limited token generation (12)",
            "ISO 6346 checksum validation",
            "Regex extraction [A-Z]{4}\\d{7}",
            "CPU-optimized inference"
        ],
        "optimizations": {
            "torch_compile": ENABLE_TORCH_COMPILE,
            "max_image_size": MAX_IMAGE_SIZE,
            "max_tokens": OCR_MAX_TOKENS,
            "concurrent_calls": CONCURRENT_OCR_CALLS,
            "yolo_batch_size": YOLO_BATCH_SIZE,
            "mixed_precision": False
        },
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding",
            "/api/gpu-status": "GET - GPU status (will return N/A on CPU)"
        }
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "running",
        "device": DEVICE,
        "cuda_available": False,
        "gpu_memory": "N/A",
        "models": {
            "glm_ocr": glm_model is not None,
            "yolo_primary": yolo_primary is not None,
            "yolo_secondary": yolo_secondary is not None
        }
    }

@app.get("/api/gpu-status")
def gpu_status():
    return {"status": "No GPU available", "device": "cpu"}

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

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    print(f"\n{'='*60}")
    print(f"[API] 📥 Kalmar ID: {event.kalmar_id}")
    print(f"[API] 🕐 Time: {event.timestamp}")
    print(f"[API] 🖥️ Device: {DEVICE.upper()}")
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
            print(f"[API] ⏱️ Time: {result['processing_time']}s on {result.get('device', DEVICE).upper()}")

            return {
                "status": "container_found",
                "container_number": result["container_number"],
                "iso6346_valid": result["iso6346_valid"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"],
                "device": result.get("device", DEVICE)
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {
        "status": "no_container",
        "kalmar_id": event.kalmar_id,
        "device": DEVICE
    }

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Container OCR GLM API — CPU Mode")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
