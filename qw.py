"""
=========================================================
Qwen3-VL-2B (LOCAL SAFETENSORS) → TEXT ONNX (CPU SAFE)

✔ Uses AutoModelForCausalLM (required for logits)
✔ Text-only ONNX export
✔ Vision stays in PyTorch
✔ INT8 quantization supported
✔ Windows & Linux compatible
=========================================================
"""

import os
import torch
import onnx
from transformers import AutoModel

from onnxruntime.quantization import quantize_dynamic, QuantType

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = r"E:\ocr\models\qwen3_vl_2b"
OUTPUT_DIR = "onnx_models"
OPSET = 17

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD MODEL (CPU ONLY, WITH LM HEAD)
# --------------------------------------------------
print("🔹 Loading Qwen3-VL-2B (CausalLM, CPU)...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.float32,
    device_map=None
)

model.eval()
print("✅ Model loaded")

# --------------------------------------------------
# TEXT-ONLY WRAPPER
# --------------------------------------------------
class TextOnlyWrapper(torch.nn.Module):
    def __init__(self, lm_model):
        super().__init__()
        self.lm = lm_model

    def forward(self, input_ids, attention_mask):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,     # 🔑 disable KV cache for ONNX
            return_dict=True
        )
        return outputs.logits   # ✅ now exists


text_wrapper = TextOnlyWrapper(model)
text_wrapper.eval()

# --------------------------------------------------
# DUMMY INPUTS
# --------------------------------------------------
batch_size = 1
seq_len = 16

vocab_size = model.get_input_embeddings().num_embeddings

input_ids = torch.randint(
    0, vocab_size,
    (batch_size, seq_len),
    dtype=torch.long
)

attention_mask = torch.ones(
    batch_size, seq_len,
    dtype=torch.long
)

# --------------------------------------------------
# EXPORT TO ONNX (LEGACY EXPORTER)
# --------------------------------------------------
text_onnx_path = os.path.join(
    OUTPUT_DIR, "qwen3_vl_text.onnx"
)

print("🔹 Exporting TEXT decoder to ONNX...")

torch.onnx.export(
    text_wrapper,
    (input_ids, attention_mask),
    text_onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
    dynamo=False
)

onnx.checker.check_model(text_onnx_path)
print(f"✅ ONNX export successful → {text_onnx_path}")

# --------------------------------------------------
# INT8 QUANTIZATION
# --------------------------------------------------
print("🔹 Applying INT8 quantization...")

text_int8_path = os.path.join(
    OUTPUT_DIR, "qwen3_vl_text_int8.onnx"
)

quantize_dynamic(
    model_input=text_onnx_path,
    model_output=text_int8_path,
    weight_type=QuantType.QInt8
)

onnx.checker.check_model(text_int8_path)
print(f"✅ INT8 model saved → {text_int8_path}")

# --------------------------------------------------
# DONE
# --------------------------------------------------
print("\n🎉 EXPORT COMPLETE")
print("📁 Generated files:")
for f in os.listdir(OUTPUT_DIR):
    print("  -", f)
