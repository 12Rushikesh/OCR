import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModel


# ================= CONFIG =================
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
MODEL_DIR = r"E:\ocr\flocr\models\PaddleOCR-VL"  # change if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ================= DOWNLOAD =================
def download_model():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print("⬇️ Downloading PaddleOCR-VL...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,  # REQUIRED on Windows
            resume_download=True
        )
        print("✅ Download complete")
    else:
        print("✅ Model already exists")

# ================= LOAD =================
def load_model():
    print("🔥 Loading PaddleOCR-VL...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)

    model.eval()
    print(f"✅ Model loaded on {DEVICE.upper()}")
    return model, processor

# ================= OCR =================
def run_ocr(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    # PaddleOCR-VL expects image-only or simple instruction
    prompt = "Recognize all text in the image."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    # Move tensors to device
    for k in inputs:
        inputs[k] = inputs[k].to(DEVICE)
        if inputs[k].dtype == torch.float32 and DEVICE == "cuda":
            inputs[k] = inputs[k].to(DTYPE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1
        )

    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()

# ================= MAIN =================
if __name__ == "__main__":
    download_model()
    model, processor = load_model()

    TEST_IMAGE = r"E:\ocr\test.jpg"  # change path

    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Image not found: {TEST_IMAGE}")
        exit(1)

    print("🔍 Running OCR...")
    ocr_result = run_ocr(TEST_IMAGE, model, processor)

    print("\n📄 OCR RESULT:\n")
    print(ocr_result)
