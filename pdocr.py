# pdocr.py
# Container OCR API: YOLO -> PaddleOCR-VL (raw text)
# Final version: global model init (works with `uvicorn pdocr:app`)

import os
import sys
import importlib
import types
import cv2
import base64
import json
import re
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import threading
import time
from datetime import datetime
import torch
from huggingface_hub import snapshot_download

# ---------------------------
# TRANSFORMERS COMPAT PATCH
# ---------------------------
# Provide safe fallbacks for HF internals that third-party remote code sometimes expects.
# This prevents import-time crashes when remote trust_remote_code modules rely on
# HF internals that vary across transformer versions.

# Patch cache_utils.SlidingWindowCache if not present
try:
    cache_utils = importlib.import_module("transformers.cache_utils")
except Exception:
    cache_utils = types.ModuleType("transformers.cache_utils")
    sys.modules.setdefault("transformers.cache_utils", cache_utils)

if not hasattr(cache_utils, "SlidingWindowCache"):
    class SlidingWindowCache:
        """Minimal shim for SlidingWindowCache expected by some dynamic modules.
           This is intentionally minimal — it prevents import errors and is harmless
           for normal model behavior (no acceleration provided)."""
        def __init__(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return None
        def set(self, *args, **kwargs):
            return None

    cache_utils.SlidingWindowCache = SlidingWindowCache

# Patch integrations.use_kernel_forward_from_hub if not present
try:
    integrations = importlib.import_module("transformers.integrations")
except Exception:
    integrations = types.ModuleType("transformers.integrations")
    sys.modules.setdefault("transformers.integrations", integrations)

if not hasattr(integrations, "use_kernel_forward_from_hub"):
    def use_kernel_forward_from_hub(*args, **kwargs):
        # Return a falsey value to indicate "do not use any special kernel"
        return False
    integrations.use_kernel_forward_from_hub = use_kernel_forward_from_hub

# Now import HF processor/model APIs (after patching)
from transformers import AutoProcessor, AutoModel

# ==========================================================
# THREAD LIMITS (WINDOWS SAFE)
# ==========================================================
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

# Disable gradients globally (saves memory / tiny perf)
torch.set_grad_enabled(False)

# ==========================================================
# CONFIG (edit these paths to match your environment)
# ==========================================================
PADDLE_MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
PADDLE_MODEL_PATH = r"E:\ocr\models\PaddleOCR-VL"  # change if needed

YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Limit concurrent Paddle calls (safe)
paddle_semaphore = threading.Semaphore(2)

# ==========================================================
# DOWNLOAD PADDLE MODEL (if missing)
# ==========================================================
def download_paddle_model():
    if not os.path.exists(PADDLE_MODEL_PATH) or not os.listdir(PADDLE_MODEL_PATH):
        print("⬇️ Downloading PaddleOCR-VL model from Hugging Face...")
        snapshot_download(
            repo_id=PADDLE_MODEL_ID,
            local_dir=PADDLE_MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ PaddleOCR-VL download complete")
    else:
        print("✅ PaddleOCR-VL already present")

# ==========================================================
# LOAD PADDLE MODEL
# ==========================================================
def load_paddle_model():
    print("[PADDLE] 🔥 Loading PaddleOCR-VL with v5.x compatibility...")

    # 1. Force the 'Slow' processor to ensure accuracy on metal containers
    processor = AutoProcessor.from_pretrained(
        PADDLE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )

    # 2. Use 'dtype' instead of the deprecated 'torch_dtype'
    model = AutoModel.from_pretrained(
        PADDLE_MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)

    model.eval()
    return model, processor

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
    "container_results/paddle_images",
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
            # fuse() optional; ignore if not supported
            pass
        print(f"[{tag}] ✅ Loaded: {path}")
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
        if not isinstance(b64_string, str):
            raise ValueError("base64 input must be a string")
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

def save_paddle_crop(pil_image: Image.Image, kalmar_id: str, region_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(
        today_folder("container_results/paddle_images"),
        f"{kalmar_id}_{timestamp}_r{region_idx}.jpg"
    )
    pil_image.save(path, "JPEG", quality=95)
    print(f"[SAVE] 🧠 Paddle input -> {path}")
    return path

def save_success_result(kalmar_id: str, raw_text: str, frame: np.ndarray):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = today_folder("container_results/success")

    image_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)

    json_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "kalmar_id": kalmar_id,
            "raw_ocr_text": raw_text,
            "timestamp": timestamp
        }, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] ✅ SUCCESS -> {image_path} + {json_path}")
    return image_path, json_path

# ==========================================================
# Minimal container-like validation (ISO-like)
# ==========================================================
ISO_CONTAINER_REGEX = re.compile(r"[A-Z]{4}\s*\d{6,7}")

def looks_like_container(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(ISO_CONTAINER_REGEX.search(text.upper()))

# ==========================================================
# Helper: prepare inputs robustly across processor versions
# ==========================================================
def prepare_processor_inputs(processor: Any, pil_image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Return dict of inputs in torch tensors (if return_tensors='pt' supported).
    Handles processors that accept images only, text only or both.
    """
    # Try image processing first
    try:
        inputs = processor(images=pil_image, return_tensors="pt")
    except TypeError:
        # Some processors might not accept PIL directly; convert to np array
        img_arr = np.array(pil_image.convert("RGB"))
        inputs = processor(images=img_arr, return_tensors="pt")

    # If prompt provided and processor supports text, attempt to get text inputs
    if prompt:
        try:
            text_inputs = processor(text=prompt, return_tensors="pt")
            # Merge keys that are not present already
            for k, v in text_inputs.items():
                if k not in inputs:
                    inputs[k] = v
        except Exception:
            # processor may not support text=... gracefully; ignore
            pass

    # Ensure all torch tensors remain torch tensors, convert numpy if necessary
    for k, v in list(inputs.items()):
        if isinstance(v, np.ndarray):
            inputs[k] = torch.from_numpy(v)

    return inputs

# ==========================================================
# PADDLE OCR (raw decoded text) with robust generate fallback
# ==========================================================
def paddle_ocr_raw(pil_image: Image.Image, model: Any, processor: Any) -> str:
    """
    Run PaddleOCR-VL on a PIL image and return the raw decoded string.
    Robust across different processor / model API shapes.
    """
    prompt = "Recognize all text in the image."

    inputs = prepare_processor_inputs(processor, pil_image, prompt)

    # Move tensors to device but only torch tensors
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(DEVICE)

    start_time = time.time()
    try:
        with torch.no_grad():
            # Preferred: use generate if available
            if hasattr(model, "generate"):
                output = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
                token_ids = output  # expect tensor or list-like
            else:
                # Fallback: call forward and derive token ids from logits
                outputs = model(**inputs)
                # Many sequence models return 'sequences' or 'logits'
                if hasattr(outputs, "sequences"):
                    token_ids = outputs.sequences
                elif isinstance(outputs, dict) and "sequences" in outputs:
                    token_ids = outputs["sequences"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits  # shape (batch, seq_len, vocab)
                    token_ids = torch.argmax(logits, dim=-1)
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                    token_ids = torch.argmax(logits, dim=-1)
                else:
                    # As a last resort try to see if model returned text directly
                    if isinstance(outputs, str):
                        decoded = outputs.strip()
                        elapsed = round(time.time() - start_time, 3)
                        print(f"[PADDLE] ⏱️ Time: {elapsed}s")
                        print(f"[PADDLE] RAW-DECODE (direct str): {repr(decoded)}")
                        return decoded
                    raise RuntimeError("Model did not return logits/sequences and has no generate()")
    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        print(f"[PADDLE] ❌ Inference error after {elapsed}s: {e}")
        raise

    elapsed = round(time.time() - start_time, 3)

    # Convert token_ids -> list of lists (CPU ints)
    if isinstance(token_ids, torch.Tensor):
        token_ids_cpu = token_ids.detach().cpu().numpy()
        # Handle different dims robustly
        if token_ids_cpu.ndim == 1:
            token_seq_list = [token_ids_cpu.tolist()]
        elif token_ids_cpu.ndim == 2:
            token_seq_list = token_ids_cpu.tolist()
        else:
            # reshape into (batch, seq) if possible
            token_seq_list = token_ids_cpu.reshape(token_ids_cpu.shape[0], -1).tolist()
    elif isinstance(token_ids, list):
        token_seq_list = token_ids
    else:
        # Unexpected type
        raise RuntimeError("Unexpected token_ids type: " + str(type(token_ids)))

    # Decode using processor (batch_decode preferred with fallback)
    try:
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(token_seq_list, skip_special_tokens=True)[0]
        else:
            decoded = processor.decode(token_seq_list[0], skip_special_tokens=True)
    except Exception:
        try:
            decoded = processor.decode(token_seq_list[0], skip_special_tokens=True)
        except Exception:
            # Fallback: join token ids as string
            decoded = " ".join(map(str, token_seq_list[0]))

    decoded = decoded.strip()
    print(f"[PADDLE] ⏱️ Time: {elapsed}s")
    print(f"[PADDLE] RAW-DECODE: {repr(decoded)}")
    return decoded

def safe_paddle_ocr_raw(pil_image: Image.Image, model: Any, processor: Any) -> str:
    with paddle_semaphore:
        return paddle_ocr_raw(pil_image, model, processor)

# ==========================================================
# YOLO DETECTION (best box only)
# ==========================================================
def detect_with_yolo(model, frame: np.ndarray, conf_threshold: float = 0.25):
    annotated = frame.copy()
    regions: List[Image.Image] = []

    if model is None:
        return False, [], annotated

    results = model(frame, conf=conf_threshold, verbose=False)[0]

    if not hasattr(results, "boxes") or not results.boxes:
        return False, [], annotated

    # select highest-confidence box
    try:
        best_box = max(results.boxes, key=lambda b: float(b.conf))
    except Exception:
        return False, [], annotated

    xyxy = best_box.xyxy[0]
    x1, y1, x2, y2 = map(int, xyxy)
    conf = float(best_box.conf)

    # safe crop bounds
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return False, [], annotated

    crop = frame[y1c:y2c, x1c:x2c]

    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

        cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{conf:.2f}",
            (x1c, max(0, y1c - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return bool(regions), regions, annotated

# ==========================================================
# MAIN PROCESSING PIPELINE
# ==========================================================
def process_container_image(base64_image: str, kalmar_id: str, model: Any, processor: Any) -> dict:
    start_time = time.time()

    try:
        frame = base64_to_cv2(base64_image)
        save_received_frame(frame, kalmar_id)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return {"success": False, "error": "Invalid image data", "processing_time": round(time.time() - start_time, 3)}

    detected, regions, annotated = detect_with_yolo(yolo_primary, frame)

    if not detected and yolo_secondary is not None:
        print("[PIPELINE] 🔄 Trying secondary YOLO...")
        detected, regions, annotated = detect_with_yolo(yolo_secondary, frame)

    save_yolo_detection(annotated, kalmar_id, detected)

    if not detected:
        print("[PIPELINE] ❌ No YOLO detection")
        return {"success": False, "processing_time": round(time.time() - start_time, 3)}

    # For each detected region, call PaddleOCR-VL and return RAW text if any
    for idx, region_pil in enumerate(regions, 1):
        save_paddle_crop(region_pil, kalmar_id, idx)

        try:
            raw_text = safe_paddle_ocr_raw(region_pil, model, processor)
        except Exception as e:
            print(f"[OCR] Region {idx} => inference error: {e}")
            raw_text = ""

        print(f"[OCR] Region {idx} => RAW: {repr(raw_text)}")

        # Minimal validation: looks like a container number (ISO-like)
        if looks_like_container(raw_text):
            # Save success (image + raw text json)
            save_success_result(kalmar_id, raw_text, frame)

            return {
                "success": True,
                "raw_text": raw_text,
                "processing_time": round(time.time() - start_time, 3)
            }
        else:
            print(f"[OCR] Region {idx} => did not pass container-like validation")

    return {"success": False, "processing_time": round(time.time() - start_time, 3)}

# ==========================================================
# FASTAPI APP
# ==========================================================
app = FastAPI(
    title="Container OCR RAW API (PaddleOCR-VL)",
    description="YOLO -> PaddleOCR-VL — returns RAW decoded text (minimal validation)",
    version="1.0"
)

@app.get("/")
def root():
    return {
        "name": "Container OCR RAW API (PaddleOCR-VL)",
        "version": "1.0",
        "philosophy": "YOLO -> PaddleOCR-VL (raw with minimal validation)",
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding"
        }
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "running",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "paddle_vl": globals().get("paddle_model") is not None,
            "yolo_primary": yolo_primary is not None,
            "yolo_secondary": yolo_secondary is not None
        }
    }

@app.post("/api/test-decode")
async def test_decode(data: dict):
    try:
        b64 = data.get("image_base64", "")
        frame = base64_to_cv2(b64)
        return {"status": "success", "message": "Image decoded successfully", "shape": frame.shape, "dtype": str(frame.dtype)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    print(f"\n{'='*60}")
    print(f"[API] 📥 Kalmar ID: {event.kalmar_id}")
    print(f"[API] 🕐 Time: {event.timestamp}")
    print(f"{'='*60}")

    images = event.images or ([event.image_base64] if event.image_base64 else [])

    if not images:
        raise HTTPException(status_code=400, detail="No images provided. Include 'images' array or 'image_base64' field.")

    print(f"[API] 📷 Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] 🔍 Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id, paddle_model, paddle_processor)

        if result.get("success"):
            print(f"[API] ✅ Found RAW: {result['raw_text']}")
            print(f"[API] ⏱️ Time: {result['processing_time']}s")
            return {
                "status": "container_found",
                "raw_text": result["raw_text"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"]
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {"status": "no_container", "kalmar_id": event.kalmar_id}

# ==========================================================
# GLOBAL MODEL LOAD (for uvicorn import-time)
# ==========================================================
print("[SYSTEM] 🔄 Initializing PaddleOCR-VL (global load)")
try:
    download_paddle_model()
    paddle_model, paddle_processor = load_paddle_model()
    print("[SYSTEM] ✅ PaddleOCR-VL ready")
except Exception as e:
    print(f"[SYSTEM] ❌ Failed to initialize PaddleOCR-VL at import: {e}")
    # re-raise so uvicorn fails fast and you can see the error
    raise

# ==========================================================
# RUN SERVER (only if executed directly)
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Container OCR RAW API (PaddleOCR-VL) — starting")
    print("="*60 + "\n")
    uvicorn.run("pdocr:app", host="0.0.0.0", port=8082, log_level="info")
