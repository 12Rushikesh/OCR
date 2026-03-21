# int8_api_container_extractor_fixed_full.py
"""
Container OCR SMART API — corrected ranking, partial handling, and robustness improvements.

Key fixes included:
- Proper partial-match detection (handles spaced OCR like "TIIU 436373").
- Alphanumeric fallback for partials.
- Sliding-window hard filter to reject candidates with >=2 substitutions.
- Proper candidates initialization order and consistent status ("full"/"partial").
- API returns container_status in responses.
- Method-priority based ranking with ISO as tie-breaker.

Added: ISO-6346 completion helper to calculate & append check digit when OCR returns
10-character partials (4 letters + 6 digits). Integration points placed where top
candidates are selected so the pipeline will return completed 11-char ISO containers.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

import time
import json
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import threading

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ultralytics import YOLO
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM

# -----------------------------
# CONFIG - adjust paths here
# -----------------------------
QWEN_OV_PATH = r"E:\ocr\flocr\qwen3-vl-2b-int8-weightonly-ov"
QWEN_HF_PATH = "Qwen/Qwen3-VL-2B-Instruct"

YOLO_PRIMARY = r"D:\Rushikesh\project\conkalmarYOLO26\used\KalmarV5.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\conkalmarYOLO26\used\KalmarV5.pt"

QWEN_MAX_CONCURRENCY = 2

HOST = "0.0.0.0"
PORT = 8082

# -----------------------------
# Ensure result folders exist
# -----------------------------
for directory in [
    "container_results/received_frames",
    "container_results/yolo_detections",
    "container_results/qwen_images",
    "container_results/success"
]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Pydantic model
# -----------------------------
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# -----------------------------
# Utility: base64 -> cv2 BGR
# -----------------------------
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

# -----------------------------
# Save helpers
# -----------------------------
def today_folder(base: str) -> str:
    folder_path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_received_frame(frame: np.ndarray, kalmar_id: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(today_folder("container_results/received_frames"), f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVE] Received -> {path}")
    return path

def save_yolo_detection(frame: np.ndarray, kalmar_id: str, detected: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    status = "detected" if detected else "no_detection"
    path = os.path.join(today_folder("container_results/yolo_detections"), f"{kalmar_id}_{timestamp}_{status}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVE] YOLO -> {path}")
    return path

def save_qwen_crop(pil_image: Image.Image, kalmar_id: str, region_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(today_folder("container_results/qwen_images"), f"{kalmar_id}_{timestamp}_r{region_idx}.jpg")
    pil_image.save(path, "JPEG", quality=95)
    print(f"[SAVE] QWEN input -> {path}")
    return path

def save_success_result(kalmar_id: str, raw_text: str, frame: np.ndarray, extracted: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = today_folder("container_results/success")
    image_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    json_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "kalmar_id": kalmar_id,
            "raw_ocr_text": raw_text,
            "extracted": extracted,
            "timestamp": timestamp
        }, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] SUCCESS -> {image_path} + {json_path}")
    return image_path, json_path

# -----------------------------
# Load YOLO models (CPU)
# -----------------------------
def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            model.fuse()
        except Exception:
            pass
        print(f"[{tag}] Loaded: {path}")
        return model
    except Exception as e:
        print(f"[{tag}] Failed to load ({path}): {e}")
        return None

yolo_primary = load_yolo(YOLO_PRIMARY, "YOLO-PRIMARY")
yolo_secondary = load_yolo(YOLO_SECONDARY, "YOLO-SECONDARY")

# -----------------------------
# Load OpenVINO QWEN via Optimum-Intel (CPU)
# -----------------------------
print("[QWEN-OV] Loading OpenVINO Qwen3-VL (INT8 weight-only)...")

processor = None
try:
    processor = AutoProcessor.from_pretrained(QWEN_HF_PATH, trust_remote_code=True)
    print("[QWEN-OV] Processor loaded from HF")
except Exception as e_proc:
    try:
        processor = AutoProcessor.from_pretrained(QWEN_OV_PATH, trust_remote_code=True)
        print("[QWEN-OV] Processor loaded from OV folder")
    except Exception as e2:
        msg = f"Failed to load processor/tokenizer from both HF ({QWEN_HF_PATH}) and OV folder ({QWEN_OV_PATH}).\nHF error: {e_proc}\nOV-folder error: {e2}\n"
        print("[QWEN-OV] " + msg)
        raise RuntimeError(msg)

try:
    qwen_model = OVModelForVisualCausalLM.from_pretrained(QWEN_OV_PATH, device="CPU", trust_remote_code=True)
    print("[QWEN-OV] OpenVINO model loaded")
except Exception as e:
    msg = f"Failed to load OpenVINO model from {QWEN_OV_PATH}: {e}"
    print("[QWEN-OV] " + msg)
    raise RuntimeError(msg)

qwen_semaphore = threading.Semaphore(QWEN_MAX_CONCURRENCY)

# -----------------------------
# Assistant response extraction
# -----------------------------
def extract_assistant_response(full_text: str) -> str:
    if full_text is None:
        return ""
    if "<|im_start|>assistant" in full_text:
        parts = full_text.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[-1].split("<|im_end|>")[0].strip()
            return response
    if "assistant\n" in full_text.lower():
        response = full_text.split("assistant\n")[-1].strip()
        return response
    patterns_to_remove = [r"<\|im_start\|>.*?<\|im_end\|>", r"system:.*?assistant:.*?"]
    cleaned = full_text
    for p in patterns_to_remove:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

# -----------------------------
# QWEN OCR using Optimum + OpenVINO
# -----------------------------
def qwen_ocr_raw(pil_image: Image.Image) -> str:
    prompt = "Read the container number text and return ONLY the exact characters or text you see."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text_prompt = prompt

    inputs = processor(text=[text_prompt], images=[pil_image], return_tensors="pt")

    with qwen_semaphore:
        start_time = time.time()
        output = qwen_model.generate(**inputs, max_new_tokens=50)
        elapsed = round(time.time() - start_time, 3)

    try:
        out_ids = output[0]
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        if hasattr(out_ids, "tolist"):
            if out_ids.ndim == 2:
                generated_ids = out_ids[0][prompt_len:].tolist()
            else:
                generated_ids = out_ids[prompt_len:].tolist()
            if hasattr(processor, "decode"):
                decoded = processor.decode(generated_ids, skip_special_tokens=True)
            elif hasattr(processor, "tokenizer"):
                decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                decoded = str(generated_ids)
        else:
            decoded = str(out_ids)
    except Exception:
        try:
            decoded = processor.decode(output[0], skip_special_tokens=True)
        except Exception:
            decoded = str(output)

    extracted = extract_assistant_response(decoded)
    print(f"[QWEN-OV] Time: {elapsed}s")
    print(f"[QWEN-OV] RAW-DECODE: {repr(decoded)}")
    print(f"[QWEN-OV] EXTRACTED (RAW): {repr(extracted)}")
    return extracted

# -----------------------------
# ISO-6346 utilities
# -----------------------------
LETTER_VALUES = {
    'A':10,'B':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,'K':21,'L':23,'M':24,
    'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,'U':32,'V':34,'W':35,'X':36,'Y':37,'Z':38
}

def iso_6346_check_digit(code: str) -> Optional[int]:
    """
    Accepts 10-character string (4 letters + 6 digits) and returns the check digit (0-9).
    Returns None on invalid input.
    """
    if not code or len(code) != 10:
        return None
    total = 0
    for i, ch in enumerate(code):
        if ch.isalpha():
            val = LETTER_VALUES.get(ch, None)
            if val is None:
                return None
        elif ch.isdigit():
            val = int(ch)
        else:
            return None
        weight = 2 ** i
        total += val * weight
    remainder = total % 11
    check = remainder
    if check == 10:
        check = 0
    return check

def validate_iso_container(container: str) -> bool:
    if not container or len(container) != 11:
        return False
    owner = container[:4]
    serial_block = container[4:10]
    check_digit_char = container[10]
    if not (owner.isalpha() and serial_block.isdigit() and check_digit_char.isdigit()):
        return False
    expected = iso_6346_check_digit((owner + serial_block))
    if expected is None:
        return False
    return expected == int(check_digit_char)

# -----------------------------
# ISO completion helper (new)
# -----------------------------
def complete_iso_container(container: str) -> Tuple[str, bool]:
    """
    If container has 10 chars (4 letters + 6 digits),
    calculate and append ISO check digit.
    Returns (container_11, was_completed)
    """
    if not container:
        return container, False

    container = container.strip().upper()

    # Only handle 10-char partials (4 letters + 6 digits)
    if len(container) == 10 and re.match(r'^[A-Z]{4}\d{6}$', container):
        check_digit = iso_6346_check_digit(container)
        if check_digit is not None:
            return container + str(check_digit), True

    return container, False

# -----------------------------
# OCR substitution maps
# -----------------------------
LETTER_SUBS = {
    '0':'O','1':'I','2':'Z','5':'S','8':'B','4':'A','6':'G','7':'T'
}
DIGIT_SUBS = {v:k for k,v in LETTER_SUBS.items()}

# -----------------------------
# Cleaning & alnum compression
# -----------------------------
def clean_ocr_text(raw: str) -> str:
    if raw is None:
        return ""
    text = raw.upper()
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[:;.,\-–_*/\\]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def alnum_sequence(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# -----------------------------
# Direct regex candidate finder
# -----------------------------
def find_direct_regex_candidates(text: str) -> List[Tuple[str, str]]:
    candidates = []
    if not text:
        return candidates
    for m in re.finditer(r'([A-Z]{4}\d{7})', text):
        candidates.append((m.group(1), "regex_exact"))
    for m in re.finditer(r'([A-Z]{4}\d{7})\b', text):
        if (m.group(1), "regex_word_boundary") not in candidates:
            candidates.append((m.group(1), "regex_word_boundary"))
    for m in re.finditer(r'([A-Z](?:\s*[A-Z]){3}\s*\d(?:\s*\d){6})', text):
        s = re.sub(r'\s+', '', m.group(1))
        candidates.append((s, "regex_spaced"))
    al = alnum_sequence(text)
    for i in range(0, max(0, len(al) - 10)):
        window = al[i:i+11]
        if re.match(r'^[A-Z]{4}\d{7}$', window):
            candidates.append((window, "alnum_window"))
    return candidates

# -----------------------------
# Sliding window reconstruction (returns subs count)
# -----------------------------
def sliding_window_reconstruct(text: str, max_subs: int = 3) -> List[Tuple[str, float, str, int]]:
    results = []
    al = alnum_sequence(text)
    n = len(al)
    for i in range(0, max(0, n - 10)):
        window = al[i:i+11]
        if len(window) < 11:
            continue
        subs = 0
        cand_chars = list(window)
        valid = True
        for idx, ch in enumerate(cand_chars):
            if idx < 4:
                if ch.isalpha():
                    continue
                else:
                    sub = LETTER_SUBS.get(ch)
                    if sub:
                        cand_chars[idx] = sub
                        subs += 1
                    else:
                        valid = False
                        break
            else:
                if ch.isdigit():
                    continue
                else:
                    sub = DIGIT_SUBS.get(ch)
                    if sub:
                        cand_chars[idx] = sub
                        subs += 1
                    else:
                        valid = False
                        break
            if subs > max_subs:
                valid = False
                break
        if not valid:
            continue
        # HARD FILTER: reject windows with 2 or more substitutions (avoid false positives)
        if subs >= 2:
            continue
        candidate = ''.join(cand_chars)
        conf = max(0.0, 1.0 - (subs * 0.18))
        results.append((candidate, conf, "sliding_window", subs))
    # dedupe keep highest conf
    dedup = {}
    for c, conf, method, subs in results:
        if c not in dedup or conf > dedup[c][0]:
            dedup[c] = (conf, method, subs)
    return [(c, dedup[c][0], dedup[c][1], dedup[c][2]) for c in dedup]

# -----------------------------
# Split text reconstruction
# -----------------------------
def reconstruct_from_parts(text: str) -> List[Tuple[str, float, str]]:
    candidates = []
    tokens = re.findall(r'[A-Z]+|\d+', text.upper())
    for i, t1 in enumerate(tokens):
        if re.match(r'^[A-Z]{3,}$', t1):
            for j in range(i+1, min(i+6, len(tokens))):
                t2 = tokens[j]
                if re.match(r'^\d{6,}$', t2):
                    letters = re.sub(r'[^A-Z]', '', t1)[:4]
                    digits = re.sub(r'[^0-9]', '', t2)
                    if len(digits) >= 7:
                        candidate = letters + digits[:7]
                        candidates.append((candidate, 0.45, "reconstruct_direct"))
                    elif len(digits) == 6:
                        next_digit = None
                        if j+1 < len(tokens) and re.match(r'^\d$', tokens[j+1]):
                            next_digit = tokens[j+1]
                        elif j+1 < len(tokens) and re.match(r'^\d{1,2}$', tokens[j+1]):
                            next_digit = tokens[j+1][0]
                        if next_digit:
                            candidate = letters + digits + next_digit[0]
                            candidates.append((candidate, 0.35, "reconstruct_join"))
    seen = {}
    for c, conf, m in candidates:
        if c not in seen or conf > seen[c][0]:
            seen[c] = (conf, m)
    return [(c, seen[c][0], seen[c][1]) for c in seen]

# -----------------------------
# Salvage recovery
# -----------------------------
def salvage_recovery(text: str) -> List[Tuple[str, float, str]]:
    results = []
    al = alnum_sequence(text)
    n = len(al)
    for i in range(0, max(0, n - 9)):
        first10 = al[i:i+10]
        rest = al[i+10:i+13]
        if re.match(r'^[A-Z]{4}\d{6}$', first10) and re.match(r'^\d', rest):
            cand = first10 + rest[0]
            results.append((cand, 0.4, "salvage_10plus"))
    dedup = {}
    for c, conf, m in results:
        if c not in dedup or conf > dedup[c][0]:
            dedup[c] = (conf, m)
    return [(c, dedup[c][0], dedup[c][1]) for c in dedup]

# -----------------------------
# Orchestrator: extract_container_candidates with corrected ranking
# -----------------------------
METHOD_PRIORITY = {
    "regex_exact": 7,
    "regex_word_boundary": 6,
    "regex_spaced": 6,
    "alnum_window": 5,
    "partial_match": 4,
    "partial_alnum": 4,
    "sliding_window": 3,
    "reconstruct_direct": 2,
    "reconstruct_join": 2,
    "salvage_10plus": 1
}

def extract_container_candidates(raw_text: str) -> List[Dict]:
    cleaned = clean_ocr_text(raw_text)
    candidates: Dict[str, Dict] = {}

    # -----------------------------
    # PARTIAL MATCH (handles spaces between owner and digits)
    # pattern captures letters and 6 digits even if separated by spaces
    # -----------------------------
    for m in re.finditer(r'\b([A-Z]{4})\s*(\d{6})\b', cleaned):
        c = m.group(1) + m.group(2)
        # only add if not already present or better than existing
        if c not in candidates or candidates[c]["confidence"] < 0.7:
            candidates[c] = {
                "container": c,
                "valid_iso": False,
                "confidence": 0.7,
                "method": "partial_match",
                "subs": 0,
                "status": "partial"
            }

    # -----------------------------
    # PARTIAL ALNUM fallback: scan alnum string for 4+6 pattern
    # -----------------------------
    al = alnum_sequence(cleaned)
    for i in range(0, max(0, len(al) - 9)):
        window10 = al[i:i+10]
        if re.match(r'^[A-Z]{4}\d{6}$', window10):
            if window10 not in candidates or candidates[window10]["confidence"] < 0.75:
                candidates[window10] = {
                    "container": window10,
                    "valid_iso": False,
                    "confidence": 0.75,
                    "method": "partial_alnum",
                    "subs": 0,
                    "status": "partial"
                }

    # direct regex candidates (full 11-char matches)
    for c, method in find_direct_regex_candidates(cleaned):
        valid = validate_iso_container(c)
        conf = 0.95
        if valid:
            conf = 0.99
        candidates[c] = {"container": c, "valid_iso": valid, "confidence": conf, "method": method, "subs": 0, "status": "full"}

    # sliding window (collect subs) - hard filtered inside function
    for c, conf, method, subs in sliding_window_reconstruct(cleaned, max_subs=3):
        valid = validate_iso_container(c)
        penalized_conf = conf
        if method == "sliding_window":
            # slight penalty per substitution (sliding already filters >=2)
            penalized_conf = max(0.0, penalized_conf - (subs * 0.12))
        if c not in candidates:
            candidates[c] = {"container": c, "valid_iso": valid, "confidence": penalized_conf, "method": method, "subs": subs, "status": "full" if len(c) == 11 else "partial"}
        else:
            if penalized_conf > candidates[c]["confidence"]:
                candidates[c].update({"confidence": penalized_conf, "method": method, "subs": subs, "valid_iso": valid, "status": "full" if len(c) == 11 else "partial"})

    # reconstruct parts
    for c, conf, method in reconstruct_from_parts(cleaned):
        valid = validate_iso_container(c)
        if c not in candidates or conf > candidates[c]["confidence"]:
            candidates[c] = {"container": c, "valid_iso": valid, "confidence": conf, "method": method, "subs": 0, "status": "full" if len(c) == 11 else "partial"}

    # salvage
    for c, conf, method in salvage_recovery(cleaned):
        valid = validate_iso_container(c)
        if c not in candidates or conf > candidates[c]["confidence"]:
            candidates[c] = {"container": c, "valid_iso": valid, "confidence": conf, "method": method, "subs": 0, "status": "full" if len(c) == 11 else "partial"}

    # produce list and sort by corrected policy:
    candidate_list = list(candidates.values())

    # Sorting key: method priority (higher better), then confidence (higher better), then ISO validity (True preferred)
    candidate_list.sort(
        key=lambda x: (
            -METHOD_PRIORITY.get(x.get("method", ""), 0),
            -x.get("confidence", 0.0),
            not x.get("valid_iso", False)
        )
    )

    # final safety clamp: if top candidate is sliding_window with many subs, demote it below alnum_window if available
    if len(candidate_list) >= 2:
        top = candidate_list[0]
        if top["method"] == "sliding_window" and top.get("subs", 0) >= 2:
            second = candidate_list[1]
            if METHOD_PRIORITY.get(second["method"], 0) >= METHOD_PRIORITY.get("alnum_window", 5):
                candidate_list[0], candidate_list[1] = candidate_list[1], candidate_list[0]

    return candidate_list

# -----------------------------
# YOLO detection (single best box)
# -----------------------------
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

    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False, [], annotated

    crop = frame[y1:y2, x1:x2]

    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return bool(regions), regions, annotated

# -----------------------------
# Main processing pipeline
# -----------------------------
def process_container_image(base64_image: str, kalmar_id: str) -> dict:
    start_time = time.time()

    try:
        frame = base64_to_cv2(base64_image)
        save_received_frame(frame, kalmar_id)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return {"success": False, "error": "Invalid image data", "processing_time": round(time.time() - start_time, 3)}

    detected, regions, annotated = detect_with_yolo(yolo_primary, frame)

    if not detected and yolo_secondary is not None:
        print("[PIPELINE] Trying secondary YOLO...")
        detected, regions, annotated = detect_with_yolo(yolo_secondary, frame)

    save_yolo_detection(annotated, kalmar_id, detected)

    if not detected:
        print("[PIPELINE] No YOLO detection")
        return {"success": False, "processing_time": round(time.time() - start_time, 3)}

    best_overall = None
    for idx, region_pil in enumerate(regions, 1):
        save_qwen_crop(region_pil, kalmar_id, idx)
        try:
            raw_text = qwen_ocr_raw(region_pil)
        except Exception as e:
            print(f"[ERROR] QWEN inference failed: {e}")
            raw_text = ""

        print(f"[OCR] Region {idx} => RAW: {repr(raw_text)}")

        if not raw_text or len(raw_text.strip()) == 0:
            continue

        candidates = extract_container_candidates(raw_text)
        print(f"[EXTRACT] Candidates: {candidates}")

        if candidates:
            top = candidates[0]

            # -----------------------------
            # ISO COMPLETION LOGIC (NEW) - attempt to complete 10-char partials to 11
            # -----------------------------
            completed_container, was_completed = complete_iso_container(top.get("container", ""))
            if was_completed:
                top["container"] = completed_container
                top["valid_iso"] = True
                top["status"] = "full"
                top["method"] = (top.get("method", "") or "") + "_iso_completed"
                top["confidence"] = min(0.98, (top.get("confidence", 0.0) or 0.0) + 0.1)

            if best_overall is None:
                best_overall = {"raw_text": raw_text, "candidate": top}
            else:
                prev = best_overall["candidate"]
                cur_pr = METHOD_PRIORITY.get(top["method"], 0)
                prev_pr = METHOD_PRIORITY.get(prev["method"], 0)
                if (cur_pr > prev_pr) or (abs(top["confidence"] - prev["confidence"]) > 0.02 and top["confidence"] > prev["confidence"]):
                    best_overall = {"raw_text": raw_text, "candidate": top}

            # early exit for strong exact match
            if top["method"] in ("regex_exact", "regex_word_boundary", "regex_spaced") and top["confidence"] >= 0.95:
                save_success_result(kalmar_id, raw_text, frame, top)
                return {
                    "success": True,
                    "container_number": top["container"],
                    "status": "full" if len(top["container"]) == 11 else "partial",
                    "confidence": top["confidence"],
                    "valid_iso": top["valid_iso"],
                    "method": top["method"],
                    "raw_text": raw_text,
                    "processing_time": round(time.time() - start_time, 3)
                }

    if best_overall:
        top = best_overall["candidate"]

        # -----------------------------
        # ISO COMPLETION LOGIC (FINAL) - ensure final returned candidate is completed if possible
        # -----------------------------
        completed_container, was_completed = complete_iso_container(top.get("container", ""))
        if was_completed:
            top["container"] = completed_container
            top["valid_iso"] = True
            top["status"] = "full"
            top["method"] = (top.get("method", "") or "") + "_iso_completed"
            top["confidence"] = min(0.98, (top.get("confidence", 0.0) or 0.0) + 0.1)

        save_success_result(kalmar_id, best_overall["raw_text"], frame, top)
        return {
            "success": True,
            "container_number": top["container"],
            "status": "full" if len(top["container"]) == 11 else "partial",
            "confidence": top["confidence"],
            "valid_iso": top["valid_iso"],
            "method": top["method"],
            "raw_text": best_overall["raw_text"],
            "processing_time": round(time.time() - start_time, 3)
        }

    return {"success": False, "processing_time": round(time.time() - start_time, 3)}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Container OCR SMART API (fixed ranking, partial handling)", version="1.0")

@app.get("/")
def root():
    return {
        "name": "Container OCR SMART API",
        "version": "1.0",
        "philosophy": "YOLO -> Qwen -> SMART extractor (fixed ranking)",
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding"
        }
    }

@app.get("/api/health")
def health_check():
    core_devices = []
    try:
        import openvino as ov
        core = ov.Core()
        core_devices = list(core.available_devices)
    except Exception:
        core_devices = []

    return {
        "status": "running",
        "runtime": "openvino+optimum-intel",
        "device": "CPU",
        "openvino_devices": core_devices,
        "models": {
            "qwen_openvino": True,
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
    print("\n" + "="*60)
    print(f"[API] Kalmar ID: {event.kalmar_id}")
    print(f"[API] Time: {event.timestamp}")
    print("="*60)

    images = event.images or ([event.image_base64] if event.image_base64 else [])

    if not images:
        raise HTTPException(status_code=400, detail="No images provided. Include 'images' array or 'image_base64' field.")

    print(f"[API] Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id)

        if result.get("success"):
            print(f"[API] Found: {result.get('container_number')} (conf {result.get('confidence')})")
            print(f"[API] Time: {result['processing_time']}s")
            return {
                "status": "container_found",
                "container_number": result.get("container_number"),
                "container_status": result.get("status"),
                "confidence": result.get("confidence"),
                "valid_iso": result.get("valid_iso"),
                "method": result.get("method"),
                "raw_text": result.get("raw_text"),
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"]
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] No container found")
    return {"status": "no_container", "kalmar_id": event.kalmar_id}

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Container OCR SMART API — starting (fixed ranking + partial handling + ISO completion)")
    print("="*60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")