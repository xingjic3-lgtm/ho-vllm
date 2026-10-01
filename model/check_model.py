"""任务 02 数值验收：CPU/float32，小模型，无需下载权重。"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .model import Qwen3


def main():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=True,
        use_cache=False,
    )
    # 参考模型也采用直接矩阵计算，减少不同 attention 后端的影响。
    config._attn_implementation = "eager"
    ours = Qwen3(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
    ).float().eval()
    reference = Qwen3ForCausalLM(config).float().eval()

    # Norm 的权重不全为 1，才能检查各处独立的缩放参数是否真正生效。
    with torch.no_grad():
        for parameter in ours.parameters():
            if parameter.ndim == 1:
                parameter.uniform_(0.8, 1.2)

    # 参数名称已与参考模型一致，严格复制全部参数。
    reference.load_state_dict(ours.state_dict(), strict=True)
    print("参数复制完成：全部参数严格匹配。")
    print("比较环境：CPU / float32 / eager attention；rtol=1e-5，atol=1e-5。")

    # 分别覆盖单 token、批处理、较长序列和非零起始位置。
    for batch_size, length, offset in ((1, 1, 0), (2, 3, 0), (2, 7, 0), (1, 5, 11)):
        token_ids = torch.randint(0, config.vocab_size, (batch_size, length))
        positions = torch.arange(offset, offset + length)
        with torch.inference_mode():
            actual = ours(token_ids, positions)
            expected = reference(
                input_ids=token_ids,
                position_ids=positions.unsqueeze(0).expand(batch_size, -1),
                use_cache=False,
            ).logits

        difference = (actual - expected).abs()
        print(
            f"B={batch_size}, T={length}, 起始位置={offset}："
            f"最大误差={difference.max().item():.8g}，"
            f"平均误差={difference.mean().item():.8g}"
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    print("通过：全部案例的 logits 与参考模型在指定容差内一致。")


if __name__ == "__main__":
    main()
