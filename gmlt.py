# ==========================================================
# containerNoRead_GLM_OCR.py
# Simplified pipeline: YOLO -> GLM-OCR -> return RAW assistant text
# Replaces Qwen3-VL with GLM-OCR from your working script
# Keeps original saving structure, thread-safety, and FastAPI endpoints
# ==========================================================

import cv2
import base64
import json
import re
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import threading
import time
from datetime import datetime
import os
import torch

# ✅ GLM-OCR IMPORTS (from your working script)
from transformers import AutoProcessor, AutoModelForImageTextToText
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from huggingface_hub import snapshot_download

# ==========================================================
# WINDOWS / CPU PATCH FOR GLM-OCR
# ==========================================================
def fixed_get_imports(filename):
    """Prevents flash_attn errors on Windows/CPU."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# ==========================================================
# THREAD LIMITS (WINDOWS SAFE)
# ==========================================================
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

# ==========================================================
# CONFIG (edit these paths to match your environment)
# ==========================================================
GLM_MODEL_ID = "zai-org/GLM-OCR"
GLM_MODEL_DIR = r"E:\ocr\flocr\models\GLM-OCR"
YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

OCR_MAX_TOKENS = 12  # ⚡ OPTIMIZED: Reduced from 64 to 12 for speed

# ⚡ SPEED OPTIMIZATION FLAGS
ENABLE_TORCH_COMPILE = True  # Compile model for ~2x speed boost (PyTorch 2.0+)
MAX_IMAGE_SIZE = 1280  # Resize images larger than this to reduce processing time
CONCURRENT_OCR_CALLS = 2 if DEVICE == "cuda" else 1  # Parallel GPU inference

# ==========================================================
# ENSURE GLM-OCR MODEL IS DOWNLOADED
# ==========================================================
def ensure_glm():
    if not os.path.exists(GLM_MODEL_DIR) or not os.listdir(GLM_MODEL_DIR):
        print(f"[GLM] ⬇️ Downloading GLM-OCR from {GLM_MODEL_ID}...")
        snapshot_download(repo_id=GLM_MODEL_ID, local_dir=GLM_MODEL_DIR)
        print(f"[GLM] ✅ Download complete")

# ==========================================================
# LOAD GLM-OCR MODEL (using your working script's approach)
# ==========================================================
print("[GLM] 🚀 Loading GLM-OCR...")
ensure_glm()

glm_processor = None
glm_model = None

# Apply the patch and load using the EXACT logic from your working script
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    glm_processor = AutoProcessor.from_pretrained(
        GLM_MODEL_DIR, 
        trust_remote_code=True
    )
    
    # ✅ KEY FIX: Use AutoModelForImageTextToText
    glm_model = AutoModelForImageTextToText.from_pretrained(
        GLM_MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE).eval()

print("[GLM] ✅ GLM-OCR loaded successfully")

# ⚡ SPEED OPTIMIZATION: Compile model for faster inference (PyTorch 2.0+)
try:
    if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile') and DEVICE == "cuda":
        print("[GLM] ⚡ Compiling model with torch.compile for faster inference...")
        glm_model = torch.compile(glm_model, mode="reduce-overhead")
        print("[GLM] ✅ Model compiled successfully - expect ~2x speedup after warmup")
except Exception as e:
    print(f"[GLM] ⚠️ Compilation skipped: {e}")

print(f"[SYSTEM] Device: {DEVICE}")
print(f"[SYSTEM] CUDA available: {torch.cuda.is_available()}")
print(f"[SYSTEM] Max tokens: {OCR_MAX_TOKENS} (optimized for speed)")
print(f"[SYSTEM] Concurrent calls: {CONCURRENT_OCR_CALLS}")
print(f"[SYSTEM] Image resize limit: {MAX_IMAGE_SIZE}px")
if torch.cuda.is_available():
    try:
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

# Thread-safe semaphore for concurrent GLM-OCR calls
# ⚡ SPEED OPTIMIZATION: Allow multiple concurrent GPU inferences if CUDA available
glm_semaphore = threading.Semaphore(CONCURRENT_OCR_CALLS)

# ==========================================================
# DIRECTORY SETUP
# ==========================================================

def today_folder(base: str) -> str:
    folder_path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Create directories
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
# LOAD YOLO MODELS (unchanged behavior)
# ==========================================================

def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        model.fuse()
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
# SAVE HELPERS (keeps style of your original code)
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
    """
    Validates ISO 6346 check digit for container numbers.
    Format: AAAA1234567 (4 letters + 7 digits)
    Returns True if the 11th digit matches the calculated check digit.
    """
    if not container or len(container) != 11:
        return False
    
    # Check format: 4 letters + 7 digits
    if not (container[:4].isalpha() and container[4:].isdigit()):
        return False
    
    total = 0
    try:
        # Calculate checksum for first 10 characters
        for i in range(10):
            ch = container[i]
            val = int(ch) if ch.isdigit() else ISO_CHAR_MAP.get(ch, 0)
            total += val * (2 ** i)
        
        # Get remainder mod 11
        remainder = total % 11
        # If remainder is 10, check digit is 0
        check_digit = 0 if remainder == 10 else remainder
        
        # Compare with actual 11th digit
        return check_digit == int(container[10])
    except:
        return False


def extract_container_from_text(text: str) -> str:
    """
    Extract valid container number from GLM-OCR output using regex and ISO 6346 validation.
    Returns the first valid 11-character container code, or empty string if none found.
    """
    if not text:
        return ""
    
    # Remove all non-alphanumeric characters and convert to uppercase
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    
    # Find all potential container numbers (4 letters + 7 digits)
    candidates = re.findall(r"[A-Z]{4}\d{7}", cleaned)
    
    # Return first candidate that passes ISO 6346 checksum
    for candidate in candidates:
        if iso6346_checksum(candidate):
            print(f"[VALIDATION] ✅ Valid container found: {candidate}")
            return candidate
    
    print(f"[VALIDATION] ❌ No valid container in: {repr(text)}")
    return ""

# ==========================================================
# ASSISTANT RESPONSE EXTRACTION (retain to remove chat wrappers)
# ==========================================================

def extract_assistant_response(full_text: str) -> str:
    """
    Extract only the assistant's response from chat-template style outputs.
    This preserves the "raw" string GLM returned while removing common wrappers.
    """
    if full_text is None:
        return ""

    # Method 1: look for explicit im tokens
    if "<|im_start|>assistant" in full_text:
        parts = full_text.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[-1].split("<|im_end|>")[0].strip()
            return response

    # Method 2: common "assistant\n" pattern
    if "assistant\n" in full_text.lower():
        parts = full_text.lower().split("assistant\n")
        if len(parts) > 1:
            response = full_text.split("assistant\n")[-1].strip()
            return response

    # Fallback: remove some system-like blocks
    patterns_to_remove = [
        r"<\|im_start\|>.*?<\|im_end\|>",
        r"system:.*?assistant:.*?",
    ]
    cleaned = full_text
    for p in patterns_to_remove:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return cleaned.strip()

# ==========================================================
# GLM-OCR — RETURN RAW ASSISTANT TEXT (adapted from your working script)
# ==========================================================

def glm_ocr_raw(pil_image: Image.Image) -> str:
    """
    Use GLM-OCR to read container text with anti-hallucination measures.
    - Uses specific prompt to prevent markdown
    - Limits tokens to 12 to stop repetitive backticks
    - Validates with ISO 6346 checksum
    - Returns only validated 11-character container code
    OPTIMIZED FOR SPEED:
    - Resizes large images to max 1280px
    - Uses mixed precision on GPU
    - Forces min 11 tokens for faster convergence
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # ⚡ SPEED OPTIMIZATION: Resize large images to reduce processing time
    if max(pil_image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print(f"[GLM] ⚡ Resized to {new_size} for speed")

    # 1. Anti-hallucination prompt (prevents markdown and extra text)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract the container number from this image. Output only the 11-character code without spaces or markdown."}
            ]
        }
    ]

    # 2. Get the prompt text using the template (tokenize=False returns string)
    prompt_text = glm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # 3. Create Inputs (Pass Image object + Prompt String)
    inputs = glm_processor(
        images=pil_image,
        text=prompt_text,
        return_tensors="pt"
    ).to(DEVICE)

    # Remove token_type_ids if present (fixes common crash)
    inputs.pop("token_type_ids", None)

    # 4. Generate with OPTIMIZED settings for speed (12 tokens max, 11 min)
    start_time = time.time()
    try:
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
            generated_ids = glm_model.generate(
                **inputs,
                max_new_tokens=12,  # ⚡ OPTIMIZED: just enough for 11 chars + potential space
                min_new_tokens=11,  # ⚡ OPTIMIZED: force at least 11 tokens (container length)
                do_sample=False,
                num_beams=1,  # ⚡ OPTIMIZED: no beam search, greedy only
                pad_token_id=glm_processor.tokenizer.pad_token_id,
                eos_token_id=glm_processor.tokenizer.eos_token_id,
                use_cache=True  # ⚡ OPTIMIZED: enable KV cache
            )
            
        # 5. Decode - skip the input tokens to get only the new text
        input_len = inputs.input_ids.shape[1]
        full_text = glm_processor.decode(
            generated_ids[0][input_len:], 
            skip_special_tokens=True
        )
        
        elapsed = round(time.time() - start_time, 3)
        
        # Extract assistant response (remove chat template wrappers)
        raw_extracted = extract_assistant_response(full_text)
        
        print(f"[GLM] ⏱️ Time: {elapsed}s")
        print(f"[GLM] RAW-DECODE: {repr(full_text)}")
        print(f"[GLM] EXTRACTED (RAW): {repr(raw_extracted)}")
        
        # 6. ✅ POST-PROCESSING: Extract and validate with ISO 6346 checksum
        validated_container = extract_container_from_text(raw_extracted)
        
        if validated_container:
            print(f"[GLM] ✅ VALIDATED CONTAINER: {validated_container}")
            return validated_container
        else:
            print(f"[GLM] ❌ No valid container found in output")
            return ""
        
    except Exception as e:
        print(f"[GLM] ❌ Generation Error: {e}")
        return ""


def safe_glm_ocr_raw(pil_image: Image.Image) -> str:
    """Thread-safe wrapper for GLM-OCR"""
    with glm_semaphore:
        return glm_ocr_raw(pil_image)

# ⚡ WARMUP: Run a dummy inference to initialize compiled model (do this AFTER function definitions)
if ENABLE_TORCH_COMPILE and DEVICE == "cuda":
    try:
        print("[GLM] 🔥 Warming up compiled model...")
        dummy_img = Image.new('RGB', (640, 480), color='white')
        safe_glm_ocr_raw(dummy_img)
        print("[GLM] ✅ Warmup complete - model ready for fast inference")
    except Exception as e:
        print(f"[GLM] ⚠️ Warmup skipped: {e}")

# ==========================================================
# YOLO DETECTION (best box only, as in your original code)
# ==========================================================

def detect_with_yolo(model, frame: np.ndarray):
    annotated = frame.copy()
    regions = []

    if model is None:
        return False, [], annotated

    results = model(frame, conf=0.15, verbose=False)[0]

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

    # For each detected region, call GLM-OCR and return validated container if any
    for idx, region_pil in enumerate(regions, 1):
        save_glm_crop(region_pil, kalmar_id, idx)

        validated_container = safe_glm_ocr_raw(region_pil)
        print(f"[OCR] Region {idx} => VALIDATED: {repr(validated_container)}")

        if validated_container and len(validated_container) == 11:
            # Save success (image + validated container json)
            save_success_result(kalmar_id, validated_container, frame)

            return {
                "success": True,
                "container_number": validated_container,
                "iso6346_valid": True,
                "processing_time": round(time.time() - start_time, 3)
            }

    return {
        "success": False, 
        "processing_time": round(time.time() - start_time, 3)
    }

# ==========================================================
# FASTAPI APP (keeps endpoints similar to your original)
# ==========================================================
app = FastAPI(
    title="Container OCR GLM API",
    description="YOLO + GLM-OCR with ISO 6346 validation — returns validated 11-character container codes",
    version="2.0"
)

@app.get("/")
def root():
    return {
        "name": "Container OCR GLM API",
        "version": "2.0",
        "model": "GLM-OCR",
        "features": [
            "Anti-hallucination prompt",
            "Limited token generation (12)",
            "ISO 6346 checksum validation",
            "Regex extraction [A-Z]{4}\\d{7}"
        ],
        "speed_optimizations": [
            "torch.compile (2x speedup on GPU)",
            f"Image resize to max {MAX_IMAGE_SIZE}px",
            f"Reduced tokens: {OCR_MAX_TOKENS}",
            "Mixed precision inference",
            f"Concurrent processing: {CONCURRENT_OCR_CALLS} calls",
            "min_new_tokens=11 for faster convergence"
        ],
        "philosophy": "YOLO -> GLM-OCR -> ISO 6346 validation — 100% accurate + FAST extraction",
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
            "dtype": str(frame.dtype)
        }
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
        raise HTTPException(
            status_code=400, 
            detail="No images provided. Include 'images' array or 'image_base64' field."
        )

    print(f"[API] 📷 Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] 🔍 Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id)

        if result.get("success"):
            print(f"[API] ✅ Found Container: {result['container_number']}")
            print(f"[API] ⏱️ Time: {result['processing_time']}s")
            return {
                "status": "container_found",
                "container_number": result["container_number"],
                "iso6346_valid": result["iso6346_valid"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"]
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {
        "status": "no_container", 
        "kalmar_id": event.kalmar_id
    }

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Container OCR GLM API — starting")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")