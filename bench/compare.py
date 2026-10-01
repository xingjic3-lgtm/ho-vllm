from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from learn_vllm.model.loader import load_weight
from learn_vllm.config import load_config

model_path = "/root/huggingface/Qwen3-0.6B"
model_config = load_config(model_path)
dtype = getattr(torch, model_config.get("dtype") or model_config["torch_dtype"])
text = "你好，请介绍一下自己。"
max_new_tokens = 100
seed = 0
torch.manual_seed(seed)


tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=dtype,local_files_only=True,).to("cuda")
model.eval()


inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
mask = inputs["attention_mask"].to("cuda")

with torch.inference_mode():
    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        if step == 0:
            baseline_logits = outputs.logits[:,-1,:].float().cpu()
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)

        if (next_token == tokenizer.eos_token_id):
            break

print("生成结果：" + tokenizer.decode(input_ids[0]))




model_ours = load_weight(model_path, dtype=dtype, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, )
inputs = tokenizer(text, return_tensors="pt")
inputs_ids = inputs["input_ids"].to("cuda")

with torch.inference_mode():
    for step in range(max_new_tokens):
        positions = torch.arange(inputs_ids.shape[1], device=inputs_ids.device)
        output = model_ours(inputs_ids, positions)
        if step == 0:
            ours_logits = output[:,-1,:].float().cpu()
        logits = output[:,-1,:]
        next_token = logits.argmax(dim=-1,keepdim=True)
        inputs_ids = torch.cat([inputs_ids, next_token],dim=-1)

        if (next_token.item() == tokenizer.eos_token_id):
            break
print(tokenizer.decode(inputs_ids[0]))



# 判断两者的生成误差
# 直接看生成logits的区别

# =========================
# 1. logits 数值误差
# =========================

difference = (baseline_logits - ours_logits).abs()

print("最大绝对误差：", difference.max().item())
print("平均绝对误差：", difference.mean().item())

relative_l2 = (
    torch.norm(baseline_logits - ours_logits)
    / torch.norm(baseline_logits)
)

print("Relative L2：", relative_l2.item())


# =========================
# 2. Top-K 对比
# =========================

k = 5

baseline_values, baseline_ids = torch.topk(
    baseline_logits,
    k=k,
    dim=-1
)

ours_values, ours_ids = torch.topk(
    ours_logits,
    k=k,
    dim=-1
)

print("\nBaseline Top-5:")
for i in range(k):
    token_id = baseline_ids[0, i].item()
    logit = baseline_values[0, i].item()

    print(
        i + 1,
        "token_id =", token_id,
        "token =", repr(tokenizer.decode([token_id])),
        "logit =", logit
    )


print("\nOurs Top-5:")
for i in range(k):
    token_id = ours_ids[0, i].item()
    logit = ours_values[0, i].item()

    print(
        i + 1,
        "token_id =", token_id,
        "token =", repr(tokenizer.decode([token_id])),
        "logit =", logit
    )


# =========================
# 3. Top-5 token 是否完全一致
# =========================

print(
    "\nTop-5 token完全一致：",
    torch.equal(baseline_ids, ours_ids)
)

print(
    "Top-1一致：",
    baseline_ids[0, 0].item() == ours_ids[0, 0].item()
)
