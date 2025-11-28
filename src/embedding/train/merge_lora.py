from transformers import AutoModel
from peft import PeftModel

model = AutoModel.from_pretrained("../../pretrain_weights/embedding/qwen3-0_6b")
model = PeftModel.from_pretrained(model, "../../pretrain_weights/embedding/qwen3-0_6b_finetune/v3-20251010-144733/checkpoint-1400")
model = model.merge_and_unload()
model.save_pretrained("../../pretrain_weights/embedding/qwen3-0_6b_finetune")