from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from learn_vllm.model.loader import load_weight

model_path = "/root/huggingface/Qwen3-0.6B"
text = "你好，请介绍一下自己。"
max_new_tokens = 20
seed = 0
torch.manual_seed(seed)

# 加载模型并编码输入。
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
).to("cuda")
model.eval()


inputs = tokenizer(text, return_tensors="pt")
inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
prompt_length = inputs["input_ids"].shape[1]

# 每轮重算完整输入，选择最高分 token 并追加。
with torch.inference_mode():
    for step in range(max_new_tokens):
        outputs = model(**inputs, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        if step == 0:
            first_step_logits = next_token_logits.cpu()

        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))],
            dim=1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

# 输出生成文本，保存供自写模型对照的基准。
generated_ids = inputs["input_ids"][0, prompt_length:].tolist()
print("参考模型生成文本：", tokenizer.decode(generated_ids, skip_special_tokens=True))

# reference_path = Path(__file__).with_suffix(".pt")
# torch.save(
#     {
#         "model_path": model_path,
#         "text": text,
#         "dtype": str(model.dtype),
#         "seed": seed,
#         "prompt_ids": inputs["input_ids"][:, :prompt_length].cpu(),
#         "first_step_logits": first_step_logits,
#         "generated_ids": generated_ids,
#     },
#     reference_path,
# )
# print("基准已保存：", reference_path)


model1 = load_weight()
model1 = model1.to(dtype=torch.float16)
model1.eval()

inputs = tokenizer(text, return_tensors="pt")
token_ids = inputs["input_ids"].to("cuda")
prompt_length = token_ids.shape[1]
max_context_length = model.config.max_position_embeddings
if prompt_length > max_context_length:
    raise ValueError("输入长度超过模型上下文上限")

with torch.inference_mode():
    for step in range(max_new_tokens):
        if token_ids.shape[1] >= max_context_length:
            break

        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        logits = model1(token_ids, positions)
        next_token_logits = logits[:, -1, :]
        if step == 0:
            # 转成 float32 再计算误差，避免误差统计本身使用半精度。
            actual = next_token_logits.float().cpu()
            expected = first_step_logits.float()
            assert actual.shape == expected.shape, "两边 logits 形状不一致"
            finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
            assert finite.item(), "logits 中存在 NaN 或 Inf"
            difference = (actual - expected).abs()
            print("首轮最后位置 logits 形状：", tuple(actual.shape))
            print("参考 / 自写 logits dtype：", first_step_logits.dtype, next_token_logits.dtype)
            print("最大绝对误差：", difference.max().item())
            print("平均绝对误差：", difference.mean().item())
            print("参考模型首个 token ID：", expected.argmax(dim=-1).item())
            print("自写模型首个 token ID：", actual.argmax(dim=-1).item())

        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

generated_ids = token_ids[0, prompt_length:].tolist()
print("自写模型生成 token IDs：", generated_ids)
print("自写模型生成文本：", tokenizer.decode(generated_ids, skip_special_tokens=True))
