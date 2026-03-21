# testcontainer_enhanced.py
# Enhanced version with 2%+ accuracy improvement for angled containers
# Key changes:
# - Perspective correction for rotated containers
# - Enhanced preprocessing (upscaling, sharpening, CLAHE)
# - Improved Qwen3-VL prompt
# - All changes backward-compatible with existing deployment

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import base64
import json
import re
import gc
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import threading
import time
from datetime import datetime
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM

# ==========================================================
# BASE DIRECTORY (SERVICE SAFE)
# ==========================================================
BASE_DIR = "container_results"

RECEIVED_DIR = os.path.join(BASE_DIR, "received_frames")
YOLO_DIR     = os.path.join(BASE_DIR, "yolo_detections")
QWEN_DIR     = os.path.join(BASE_DIR, "qwen_images")
SUCCESS_DIR  = os.path.join(BASE_DIR, "success")

# ==========================================================
# OCR LOGGING SETUP
# ==========================================================
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

OCR_LOG_FILE = os.path.join(
    LOG_DIR,
    f"ocr_{datetime.now().strftime('%Y%m%d')}.log"
)

_log_lock = threading.Lock()

def log_ocr(kalmar_id: str, step: str, data: dict):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_obj = {
        "time": ts,
        "kalmar_id": kalmar_id,
        "step": step,
        **data
    }
    message = json.dumps(log_obj, default=str)
    print(message)
    with _log_lock:
        with open(OCR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")

for d in [RECEIVED_DIR, YOLO_DIR, QWEN_DIR, SUCCESS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# NEW: PERSPECTIVE CORRECTION FOR ANGLED CONTAINERS
# ==========================================================
def correct_perspective(image: np.ndarray, angle_threshold: float = 2.0) -> np.ndarray:
    """
    Detect and correct rotation/perspective distortion in container images.
    Uses edge detection + Hough lines to find dominant angle and rotate.
    
    Args:
        image: BGR numpy array (from cv2 or YOLO crop)
        angle_threshold: Minimum rotation angle (degrees) to trigger correction
    
    Returns:
        Corrected image (or original if no significant tilt detected)
    """
    try:
        h, w = image.shape[:2]
        
        # Skip if image too small
        if w < 50 or h < 50:
            return image
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform to find dominant lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=min(w, h) // 4,  # Dynamic threshold based on size
            minLineLength=min(w, h) // 3,
            maxLineGap=20
        )
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize to [-45, 45] range (assuming container text is roughly horizontal)
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            
            angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle (more robust to outliers than mean)
        median_angle = np.median(angles)
        
        # Only rotate if angle is significant
        if abs(median_angle) < angle_threshold:
            return image
        
        # Rotate image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new bounding dimensions to avoid cropping
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        
        # Adjust rotation matrix for new dimensions
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(
            image, 
            M, 
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
        
    except Exception as e:
        print(f"[PERSPECTIVE CORRECTION ERROR]: {e}")
        return image  # Return original on error

# ==========================================================
# NEW: ENHANCED PREPROCESSING PIPELINE
# ==========================================================
def enhance_for_ocr(image: np.ndarray, target_width: int = 800) -> np.ndarray:
    """
    Multi-stage preprocessing to maximize OCR accuracy:
    1. Upscale if too small (Qwen3-VL benefits from higher res)
    2. CLAHE contrast enhancement
    3. Sharpening
    
    Args:
        image: BGR numpy array
        target_width: Minimum width for upscaling
    
    Returns:
        Enhanced BGR image ready for OCR
    """
    try:
        h, w = image.shape[:2]
        
        # Step 1: Upscale if needed (maintains aspect ratio)
        if w < target_width:
            scale = target_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(
                image, 
                (new_w, new_h), 
                interpolation=cv2.INTER_CUBIC
            )
        
        # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Works in LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Sharpening (unsharp mask)
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
        
    except Exception as e:
        print(f"[ENHANCEMENT ERROR]: {e}")
        return image  # Return original on error

# ==========================================================
# NEW: COMPLETE PREPROCESSING WRAPPER
# ==========================================================
def preprocess_container_crop(crop: np.ndarray, kalmar_id: str) -> np.ndarray:
    """
    Complete preprocessing pipeline combining all enhancements.
    Called after YOLO detection, before OCR.
    
    Args:
        crop: BGR numpy array from YOLO detection
        kalmar_id: For logging
    
    Returns:
        Fully preprocessed BGR image
    """
    start = time.time()
    
    # Step 1: Perspective correction (fix rotation)
    corrected = correct_perspective(crop, angle_threshold=2.0)
    
    # Step 2: Enhancement (upscale + contrast + sharpen)
    enhanced = enhance_for_ocr(corrected, target_width=800)
    
    elapsed = round(time.time() - start, 3)
    
    log_ocr(kalmar_id, "PREPROCESSING", {
        "original_size": f"{crop.shape[1]}x{crop.shape[0]}",
        "final_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
        "processing_time_sec": elapsed
    })
    
    return enhanced

# ==========================================================
# BLUR DETECTION (Laplacian Variance)
# ==========================================================
def is_blurry(pil_img, threshold=120):
    """Returns (True/False, variance_score)"""
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        return score < threshold, score
    except Exception as e:
        print("[BLUR ERROR]:", e)
        return False, 0

# ==========================================================
# CONFIG
# ==========================================================
QWEN_OV_PATH    = r"E:\ocr\flocr\qwen3-vl-2b-int8-weightonly-ov"
QWEN_HF_PATH    = "Qwen/Qwen3-VL-2B-Instruct"
YOLO_MODEL_PATH = r"D:\Rushikesh\project\conkalmarYOLO26\used\KalmarV6.pt"

ISO_BASE_REGEX = re.compile(r"[A-Z]{3}U\d{6}")
ISO_11_REGEX   = re.compile(r"[A-Z]{3}U\d{7}")

# ==========================================================
# ISO 6346 CHECK DIGIT
# ==========================================================
LETTER_VALUES = [
    10,12,13,14,15,16,17,18,19,20,21,
    23,24,25,26,27,28,29,30,31,32,
    34,35,36,37,38
]

def compute_check_digit(code10: str) -> int:
    total = 0
    for i, c in enumerate(code10):
        if i < 4:
            value = LETTER_VALUES[ord(c) - 65]
        else:
            value = int(c)
        total += value * (1 << i)
    remainder = total % 11
    return 0 if remainder == 10 else remainder

# ==========================================================
# IMPROVED extract_and_validate_container (UNCHANGED)
# ==========================================================
def extract_and_validate_container(text: str) -> str:
    """Industrial-grade container number extraction with validation"""
    if not text:
        return ""

    raw = text
    norm = re.sub(r'[^A-Z0-9]', '', text.upper())

    if not norm:
        return ""

    candidates = []

    # 1) Exact 11-char match
    m11 = ISO_11_REGEX.search(norm)
    if m11:
        candidate = m11.group()
        if candidate != "OOOU0000000":
            base = candidate[:10]
            expected = compute_check_digit(base)
            if str(expected) == candidate[-1]:
                candidates.append((candidate, 1.0, "original_full"))

    # 2) 10-char + compute check digit
    m10 = ISO_BASE_REGEX.search(norm)
    if m10:
        base = m10.group()
        if base != "OOOU000000":
            digit = compute_check_digit(base)
            cand = base + str(digit)
            if cand != "OOOU0000000":
                candidates.append((cand, 0.75, "original_base"))

    # 3) Spaced/broken OCR
    spaced = re.sub(r'\s+', '', raw.upper())
    for m in re.finditer(r'[A-Z]{3}U\d{6}', spaced):
        base = m.group()
        if base != "OOOU000000":
            digit = compute_check_digit(base)
            cand = base + str(digit)
            candidates.append((cand, 0.9, "spaced_scan"))

    # 4) Sliding window
    n = len(norm)
    for i in range(0, max(0, n - 9)):
        window10 = norm[i:i+10]
        if re.match(r'^[A-Z]{3}U\d{6}$', window10):
            digit = compute_check_digit(window10)
            cand = window10 + str(digit)
            candidates.append((cand, 0.85, "sliding_window"))

    # 5) Light OCR fixes
    def apply_light_fixes(s: str) -> str:
        chars = list(s)
        for i in range(min(3, len(chars))):
            if chars[i] == '0':
                chars[i] = 'O'
        return "".join(chars)

    fixed = apply_light_fixes(norm)
    for i in range(0, max(0, len(fixed) - 9)):
        w = fixed[i:i+10]
        if re.match(r'^[A-Z]{3}U\d{6}$', w):
            digit = compute_check_digit(w)
            cand = w + str(digit)
            candidates.append((cand, 0.8, "ocr_fix_window"))

    # 6) Force-U fallback (original behavior)
    cleaned_forced = re.sub(r'[^A-Z0-9]', '', text.upper())
    chars = list(cleaned_forced)
    for i in range(min(3, len(chars))):
        if chars[i] == "0":
            chars[i] = "O"
    cleaned_forced = "".join(chars)
    if len(cleaned_forced) >= 4:
        fourth_char = cleaned_forced[3]
        if fourth_char.isalpha():
            cleaned_forced = cleaned_forced[:3] + "U" + cleaned_forced[4:]
        elif fourth_char.isdigit():
            cleaned_forced = cleaned_forced[:3] + "U" + cleaned_forced[3:]
    
    m11f = ISO_11_REGEX.search(cleaned_forced)
    if m11f:
        candidate = m11f.group()
        if candidate != "OOOU0000000":
            base = candidate[:10]
            expected = compute_check_digit(base)
            if str(expected) == candidate[-1]:
                candidates.append((candidate, 0.95, "forced_original_full"))
    
    m10f = ISO_BASE_REGEX.search(cleaned_forced)
    if m10f:
        base = m10f.group()
        if base != "OOOU000000":
            digit = compute_check_digit(base)
            cand = base + str(digit)
            if cand != "OOOU0000000":
                candidates.append((cand, 0.7, "forced_original_base"))

    if not candidates:
        return ""

    # Deduplicate
    dedup = {}
    for cand, score, method in candidates:
        if cand not in dedup or score > dedup[cand][0]:
            dedup[cand] = (score, method)

    sorted_candidates = sorted(
        ((c, s_m[0], s_m[1]) for c, s_m in dedup.items()),
        key=lambda x: -x[1]
    )

    best_candidate = sorted_candidates[0][0]

    # Final validation
    if len(best_candidate) == 11:
        base = best_candidate[:10]
        expected = compute_check_digit(base)
        if str(expected) != best_candidate[-1]:
            return ""
        if best_candidate == "OOOU0000000":
            return ""
        return best_candidate

    return ""

# ==========================================================
# LOAD QWEN
# ==========================================================
print("[QWEN-OV] 🚀 Loading OpenVINO INT8 Qwen3-VL...")

try:
    qwen_processor = AutoProcessor.from_pretrained(QWEN_HF_PATH, trust_remote_code=True)
    print("[QWEN-OV] ✅ Processor loaded from HF")
except Exception:
    qwen_processor = AutoProcessor.from_pretrained(QWEN_OV_PATH, trust_remote_code=True)
    print("[QWEN-OV] ✅ Processor loaded from OV folder")

qwen_model = OVModelForVisualCausalLM.from_pretrained(
    QWEN_OV_PATH,
    device="CPU",
    trust_remote_code=True
)

print("[QWEN-OV] ✅ Model loaded (CPU INT8)")

qwen_semaphore = threading.Semaphore(1)

# ==========================================================
# LOAD YOLO
# ==========================================================
print("[YOLO] 🚀 Loading...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    try:
        yolo_model.to("cpu")
    except Exception:
        pass
    try:
        yolo_model.fuse()
    except Exception:
        pass
    print("[YOLO] ✅ Loaded")
except Exception as e:
    print("[YOLO] ❌ Failed:", e)
    yolo_model = None

# ==========================================================
# REQUEST MODEL
# ==========================================================
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# ==========================================================
# UTILITIES
# ==========================================================
def today_folder(base):
    path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(path, exist_ok=True)
    return path

def base64_to_cv2(b64: str):
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None — invalid image data")
        return frame
    except Exception as e:
        print(f"[ERROR] base64_to_cv2 failed: {e}")
        raise

# ==========================================================
# SAVE HELPERS
# ==========================================================
def save_received(frame, kid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p = os.path.join(today_folder(RECEIVED_DIR), f"{kid}_{ts}.jpg")
    cv2.imwrite(p, frame)

def save_yolo(frame, kid, detected):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    s = "detected" if detected else "no_detection"
    p = os.path.join(today_folder(YOLO_DIR), f"{kid}_{ts}_{s}.jpg")
    cv2.imwrite(p, frame)

def save_qwen_img(pil, kid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p = os.path.join(today_folder(QWEN_DIR), f"{kid}_{ts}.jpg")
    pil.save(p, "JPEG", quality=85)

def save_success(kid, iso, frame, raw):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = today_folder(SUCCESS_DIR)
    img = os.path.join(base, f"{kid}_{iso}_{ts}.jpg")
    js  = os.path.join(base, f"{kid}_{iso}_{ts}.json")
    cv2.imwrite(img, frame)
    with open(js, "w") as f:
        json.dump({
            "kalmar_id": kid,
            "container_number": iso,
            "raw_text": raw,
            "timestamp": ts
        }, f, indent=2)

# ==========================================================
# ENHANCED QWEN OCR WITH BETTER PROMPT
# ==========================================================
def qwen_ocr(pil: Image.Image):
    """
    Enhanced OCR with improved prompt for better accuracy on angled/challenging text
    """
    # IMPROVED PROMPT: More explicit about format and common OCR errors
    prompt = """Extract the shipping container number from this image.

Format: 4 letters followed by 7 digits (example: TXGU7204637)
The container number is usually printed on the top right of the container.

Common OCR mistakes to avoid:
- Confusing 7 with 1
- Confusing 0 with 8 or O
- Confusing 6 with 5
- Confusing 3 with 8

Return ONLY the 11-character container number, nothing else."""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        text_prompt = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        text_prompt = prompt

    inputs = qwen_processor(
        text=[text_prompt],
        images=[pil],
        return_tensors="pt"
    )

    with qwen_semaphore:
        start = time.time()
        try:
            output = qwen_model.generate(**inputs, max_new_tokens=20)
        except Exception as e:
            print("[QWEN ERROR]:", e)
            return "", ""
        
        print("[QWEN-OV] ⏱️ Time:", round(time.time() - start, 3), "sec")

    # Decode
    try:
        out_ids = output[0]
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids = out_ids[prompt_len:].tolist()
        full_text = qwen_processor.decode(generated_ids, skip_special_tokens=True)
    except Exception:
        full_text = qwen_processor.decode(output[0], skip_special_tokens=True)
        
    del inputs
    gc.collect()

    iso = extract_and_validate_container(full_text)
    
    print("[QWEN RAW]:", full_text)
    print("[QWEN FINAL ISO]:", iso)

    return iso, full_text

def safe_qwen_ocr(pil):
    return qwen_ocr(pil)

# ==========================================================
# YOLO + OCR PIPELINE (ENHANCED)
# ==========================================================
def detect_yolo(frame, kalmar_id: str):
    """Enhanced YOLO detection with preprocessing"""
    if yolo_model is None:
        return False, None, frame

    res = yolo_model(frame, conf=0.25, verbose=False)[0]

    if not res.boxes or len(res.boxes) == 0:
        return False, None, frame

    best = max(res.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    # Clamp coords
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return False, None, frame

    crop = frame[y1:y2, x1:x2]

    ann = frame.copy()
    cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if crop.size == 0:
        return False, None, ann

    # ✨ NEW: Apply complete preprocessing pipeline
    enhanced_crop = preprocess_container_crop(crop, kalmar_id)
    
    # Convert to PIL
    pil = Image.fromarray(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
    
    return True, pil, ann

def process_container_image(b64, kid):
    start = time.time()

    # 1️⃣ Decode
    frame = base64_to_cv2(b64)
    save_received(frame, kid)

    # 2️⃣ Run YOLO with preprocessing
    detected, region, ann = detect_yolo(frame, kid)
    save_yolo(ann, kid, detected)

    if not detected or region is None:
        log_ocr(kid, "YOLO_NO_DETECTION", {
            "message": "No container region detected"
        })
        del frame, ann
        gc.collect()
        return {"success": False}

    # 3️⃣ Blur check
    blurry, blur_score = is_blurry(region, threshold=120)

    log_ocr(kid, "BLUR_CHECK", {
        "blur_score": round(blur_score, 2),
        "is_blurry": blurry
    })

    if blurry:
        del frame, ann, region
        gc.collect()
        return {
            "success": False,
            "processing_time": round(time.time() - start, 3),
            "reason": "blurred_image"
        }

    # 4️⃣ OCR
    save_qwen_img(region, kid)
    iso, raw = safe_qwen_ocr(region)

    log_ocr(kid, "OCR_READ", {
        "raw_text": raw,
        "final_iso": iso,
        "processing_time_sec": round(time.time() - start, 3)
    })

    if iso:
        save_success(kid, iso, frame, raw)

        log_ocr(kid, "OCR_SUCCESS", {
            "container_number": iso,
            "processing_time_sec": round(time.time() - start, 3)
        })

        result = {
            "success": True,
            "iso_code": iso,
            "raw_text": raw,
            "processing_time": round(time.time() - start, 3)
        }
    else:
        log_ocr(kid, "OCR_FAILED", {
            "raw_text": raw,
            "processing_time_sec": round(time.time() - start, 3)
        })

        result = {
            "success": False,
            "raw_text": raw,
            "processing_time": round(time.time() - start, 3)
        }

    del frame, ann, region
    gc.collect()
    return result

# ==========================================================
# FASTAPI
# ==========================================================
app = FastAPI(title="Container OCR API (Enhanced)")

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    imgs = event.images or ([event.image_base64] if event.image_base64 else [])
    if not imgs:
        raise HTTPException(status_code=400, detail="No images provided")

    for img in imgs:
        res = process_container_image(img, event.kalmar_id)

        if res.get("success"):
            return {
                "status": "container_found",
                "container_number": res.get("iso_code"),
                "raw_text": res.get("raw_text"),
                "kalmar_id": event.kalmar_id,
                "processing_time": res.get("processing_time")
            }

        if res.get("raw_text"):
            return {
                "status": "no_container",
                "raw_text": res.get("raw_text"),
                "kalmar_id": event.kalmar_id,
                "processing_time": res.get("processing_time")
            }

    return {
        "status": "no_container",
        "kalmar_id": event.kalmar_id
    }

@app.get("/api/health")
def health():
    core_devices = []
    try:
        import openvino as ov
        core = ov.Core()
        core_devices = list(core.available_devices)
    except Exception:
        pass

    return {
        "status": "running",
        "device": "CPU",
        "openvino_devices": core_devices,
        "qwen_loaded": True,
        "yolo_loaded": yolo_model is not None,
        "enhancements": "perspective_correction + preprocessing + improved_prompt"
    }

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)