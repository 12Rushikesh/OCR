from ultralytics import YOLO

model = YOLO(r"C:\Users\Rapportsoft\Downloads\models\02-04-26\New folder\corner_damage_.pt")

model.export(
    format="engine",   # TensorRT
    device=0,          # GPU
    half=True,         # FP16 (no accuracy loss)
    imgsz=640,
    dynamic=False,
    simplify=True
)

print("✅ TensorRT FP16 model created successfully!")