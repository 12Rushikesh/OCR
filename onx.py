import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModel


# ================= CONFIG =================
MODEL_ID = "onnx-community/Qwen2-VL-2B-Instruct"
MODEL_DIR = r"E:\ocr\flocr\models\Qwen2-VL-2B-Instruct"

# ONNX-community models → CPU ONLY
DEVICE = "cpu"
DTYPE = torch.float32

# ================= DOWNLOAD =================
def download_model():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print("⬇️ Downloading Qwen2-VL-2B-Instruct (ONNX)...")
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
    print("🔥 Loading Qwen2-VL-2B-Instruct...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True
    )

    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully on CPU")
    return model, processor

# ================= MAIN =================
if __name__ == "__main__":
    download_model()
    model, processor = load_model()

    print("\n🎉 Qwen2-VL-2B-Instruct is ready to use!")
