import os
import cv2
import base64
import uuid
import threading
import numpy as np

from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
from starlette.concurrency import run_in_threadpool

# ================= CPU THREAD LIMIT =================
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

router = APIRouter(prefix="/api", tags=["Damage Detection"])

# ================= CONFIG =================
MODEL_1_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\yolo26m_dent__v4.engine"
MODEL_2_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\rust_v22.engine"
MODEL_3_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\gasket.engine"
# MODEL_4_PATH = r"D:\Rapportsoft\Deployment\SMARTGATEINPY\containerdamagemodels\01-04-2026\broken.pt"
MODEL_5_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\corner_damage_.engine"
MODEL_6_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\dent_v3.2.22.engine"
MODEL_7_PATH = r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\push_in_out.engine"

CONF_THRESHOLD = 0.30

# Local folder to save detected/annotated images
SAVE_DIR = r"D:\Rushikesh\project\coversion\output_images"

DAMAGE_CLASSES_MODEL_1 = {
    0: "DENT",
}

DAMAGE_CLASSES_MODEL_2 = {
    0: "RUST",
}

DAMAGE_CLASSES_MODEL_3 = {
         0: "Gasket Cut",
 }

# DAMAGE_CLASSES_MODEL_4 = {
#     0: "Broken Lock Bar",
# }

DAMAGE_CLASSES_MODEL_5 = {
    0: "Corner Post Dent",
}

DAMAGE_CLASSES_MODEL_6 = {
    0: "DENT",
}

DAMAGE_CLASSES_MODEL_7 = {
    1: "PUSH OUT",
    0: "PUSH IN"
}

# ================= LOAD MODELS =================
print("🚀 Loading YOLO damage models...")

# model_1 = YOLO(MODEL_1_PATH)
# model_1.fuse()

# model_2 = YOLO(MODEL_2_PATH)
# model_2.fuse()

# model_3 = YOLO(MODEL_3_PATH)
# model_3.fuse()

# model_4 = YOLO(MODEL_4_PATH)
# model_4.fuse()

# model_5 = YOLO(MODEL_5_PATH)
# model_5.fuse()

# model_6 = YOLO(MODEL_6_PATH)
# model_6.fuse()

# model_7 = YOLO(MODEL_7_PATH)
# model_7.fuse()

model_1 = YOLO(MODEL_1_PATH, task="detect")
model_2 = YOLO(MODEL_2_PATH, task="detect")
MODEL_3 = YOLO(MODEL_3_PATH, task="detect")
# model_4 = YOLO(MODEL_4_PATH, task="detect")   
MODEL_5 = YOLO(MODEL_5_PATH, task="detect")
model_6 = YOLO(MODEL_6_PATH, task="detect")
model_7 = YOLO(MODEL_7_PATH, task="detect")

print("✅ Both damage models loaded")

# GPU lock (VERY IMPORTANT)
yolo_lock = threading.Lock()

# ================= UTILS =================
def image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def run_yolo(model: YOLO, img: np.ndarray):
    with yolo_lock:
        return model(
            img,
            conf=CONF_THRESHOLD,
            iou=0.7,
            verbose=False
        )[0]


def draw_black_bbox(img, x1, y1, x2, y2, label, confidence):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)

    cv2.rectangle(
        img,
        (x1, y1 - th - 10),
        (x1 + tw + 8, y1),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        text,
        (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3
    )

# ================= CORE PROCESS =================
def extract_detections(results, class_map, model_name):
    detections = []
    labels = set()
    confidences = []

    if results.boxes is None:
        return detections, labels, confidences

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        label = class_map.get(cls_id, f"unknown_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "model": model_name,
            "label": label,
            "confidence": round(conf, 3),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "class_id": cls_id
        })

        labels.add(label)
        confidences.append(conf)

    return detections, labels, confidences


def process_image_sync(image_data: bytes, file_id: str, filename: str):
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # Run both models
    res1 = run_yolo(model_1, img)
    res2 = run_yolo(model_2, img)
    res3 = run_yolo(MODEL_3, img)
    # res4 = run_yolo(model_4, img)
    res5 = run_yolo(MODEL_5, img)
    res6 = run_yolo(model_6, img)
    res7 = run_yolo(model_7, img)

    det1, labels1, conf1 = extract_detections(res1, DAMAGE_CLASSES_MODEL_1, "model_1")
    det2, labels2, conf2 = extract_detections(res2, DAMAGE_CLASSES_MODEL_2, "model_2")
    det3, labels3, conf3 = extract_detections(res3, DAMAGE_CLASSES_MODEL_3, "model_3")
    # det4, labels4, conf4 = extract_detections(res4, DAMAGE_CLASSES_MODEL_4, "model_4")
    det5, labels5, conf5 = extract_detections(res5, DAMAGE_CLASSES_MODEL_5, "model_5")
    det6, labels6, conf6 = extract_detections(res6, DAMAGE_CLASSES_MODEL_6, "model_6")
    det7, labels7, conf7 = extract_detections(res7, DAMAGE_CLASSES_MODEL_7, "model_7")

    # all_detections = det1 + det2 + det3 + det4 + det5 + det6
    # all_labels = set().union(labels1, labels2, labels3, labels4, labels5, labels6)
    # all_confidences = conf1 + conf2 + conf3 + conf4 + conf5 + conf6

    all_detections = det1 + det2 + det6 + det7
    all_labels = set().union(labels1, labels2, labels6, labels7)
    all_confidences = conf1 + conf2 + conf6 + conf7

    is_damaged = bool(all_labels)
    status = "DAMAGED" if is_damaged else "NO DAMAGE"

    annotated = img.copy()
    for d in all_detections:
        b = d["bbox"]
        draw_black_bbox(
            annotated,
            b["x1"],
            b["y1"],
            b["x2"],
            b["y2"],
            d["label"],
            d["confidence"]
        )

    # ================= SAVE ANNOTATED IMAGE LOCALLY =================
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{file_id}.jpg")
    cv2.imwrite(save_path, annotated)

    return {
        "success": True,
        "file_id": file_id,
        "original_filename": filename,
        "status": status,
        "is_damaged": is_damaged,
        "damage_labels": sorted(all_labels),
        "detection_count": len(all_detections),
        "detections": all_detections,
        "average_confidence": round(float(np.mean(all_confidences)), 3)
        if all_confidences else 0,
        "input_image": f"data:image/jpeg;base64,{image_to_base64(img)}",
        "output_image": f"data:image/jpeg;base64,{image_to_base64(annotated)}",
        "processed_at": datetime.now().isoformat()
    }

# ================= ROUTES =================
@router.post("/process-single1")
async def process_single(image: UploadFile = File(...)):
    image_data = await image.read()
    file_id = uuid.uuid4().hex[:8]

    try:
        return await run_in_threadpool(
            process_image_sync,
            image_data,
            file_id,
            image.filename
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/process-images1")
async def process_batch(images: list[UploadFile] = File(...)):
    results = []

    for idx, file in enumerate(images):
        image_data = await file.read()
        file_id = f"{uuid.uuid4().hex[:8]}_{idx+1}"

        result = await run_in_threadpool(
            process_image_sync,
            image_data,
            file_id,
            file.filename
        )
        results.append(result)

    damaged = sum(1 for r in results if r["is_damaged"])

    return {
        "success": True,
        "results": results,
        "summary": {
            "total_images": len(results),
            "damaged_images": damaged,
            "safe_images": len(results) - damaged
        }
    }


@router.get("/health1")
def health():
    return {
        "status": "healthy",
        "model_1_loaded": True,
        "model_2_loaded": True,
        "confidence_threshold": CONF_THRESHOLD,
        "active_damage_labels": list(
            set(DAMAGE_CLASSES_MODEL_1.values()).union(
                DAMAGE_CLASSES_MODEL_2.values()
            )
        )
    }


# ================= MAIN APP =================
from fastapi import FastAPI

app = FastAPI(title="Kalmar Damage Detection API")

app.include_router(router)