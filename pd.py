import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForCausalLM

# ================= CONFIG =================
MODEL_ID = "zai-org/GLM-OCR"
MODEL_DIR = r"E:\ocr\flocr\models\GLM-OCR"  # change if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE:
# If you face NaN / empty output on GPU, change to torch.float32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ================= DOWNLOAD =================
def download_model():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print("⬇️ Downloading GLM-OCR...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,  # REQUIRED for Windows
            resume_download=True
        )
        print("✅ Download complete")
    else:
        print("✅ Model already exists")

# ================= LOAD =================
def load_model():
    print("🔥 Loading GLM-OCR...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
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
    width, height = image.size

    # Available Florence tasks:
    # "<OCR>" or "<OCR_WITH_REGION>"
    task_prompt = "<OCR>"

    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt"
    )

    # Move tensors safely to device
    for k in inputs:
        inputs[k] = inputs[k].to(DEVICE)
        if inputs[k].dtype == torch.float32 and DEVICE == "cuda":
            inputs[k] = inputs[k].to(DTYPE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3  # Improves OCR accuracy
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # Florence post-processing (VERY IMPORTANT)
    processed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(width, height)
    )

    return processed[task_prompt]

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
