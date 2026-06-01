import torch
from models.Qwen35 import Qwen35Config, Qwen35Model, Qwen35SSMLayer, Qwen35AttentionLayer
from models.ops import compute_rope_cos_sin
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"


def _gemma_rmsnorm_weight(hf_weight: torch.Tensor) -> torch.Tensor:
    """Gemma3RMSNorm stores weight offset by 1: forward does (1 + w) * x.
    Our rms_norm does w * x, so we need to add 1."""
    return 1.0 + hf_weight.float()


def load_weights_from_hf(model: Qwen35Model, hf_model):
    config = model.config
    H = config.ssm_group_count   # 16
    D = config.ssm_state_size    # 128

    # Embedding: HF [vocab, hidden]
    model.token_embd_weight = hf_model.model.embed_tokens.weight.data.float().clone()

    # Output norm (Gemma3RMSNorm)
    model.output_norm_weight = _gemma_rmsnorm_weight(hf_model.model.norm.weight.data)

    for i in range(config.block_count):
        hl = hf_model.model.layers[i]
        layer = model.layers[i]

        # Norms (both are Gemma3RMSNorm with +1)
        attn_norm_w = _gemma_rmsnorm_weight(hl.input_layernorm.weight.data)
        ffn_norm_w = _gemma_rmsnorm_weight(hl.post_attention_layernorm.weight.data)

        if isinstance(layer, Qwen35SSMLayer):
            la = hl.linear_attn  # Qwen3NextGatedDeltaNet

            # in_proj_qkvz: [proj_size, hidden] where proj_size = key_dim*2 + value_dim*2
            # HF interleaves per group: Q_h(128), K_h(128), V_h(128), Z_h(128)
            # Our GGUF format: Q_all(2048), K_all(2048), V_all(2048) concatenated, Z separate
            w_qkvz = la.in_proj_qkvz.weight.data.float()  # [8192, 1024]
            w_qkvz = w_qkvz.view(H, 4 * D, -1)            # [16, 512, 1024]

            Q_all = w_qkvz[:, :D, :].reshape(H * D, -1)          # [2048, 1024]
            K_all = w_qkvz[:, D:2*D, :].reshape(H * D, -1)       # [2048, 1024]
            V_all = w_qkvz[:, 2*D:3*D, :].reshape(H * D, -1)     # [2048, 1024]
            Z_all = w_qkvz[:, 3*D:4*D, :].reshape(H * D, -1)     # [2048, 1024]

            layer.qkv_weight = torch.nn.Parameter(torch.cat([Q_all, K_all, V_all], dim=0))
            layer.gate_weight = torch.nn.Parameter(Z_all.contiguous())

            # in_proj_ba: [num_v_heads*2, hidden] → interleaved b, a per group
            w_ba = la.in_proj_ba.weight.data.float()  # [32, 1024]
            w_ba = w_ba.view(H, 2, -1)                # [16, 2, 1024]
            layer.ssm_beta_weight = torch.nn.Parameter(w_ba[:, 0, :].contiguous())   # [16, 1024]
            layer.ssm_alpha_weight = torch.nn.Parameter(w_ba[:, 1, :].contiguous())  # [16, 1024]

            # Conv1d: [conv_dim, 1, kernel] → [conv_dim, kernel]
            layer.conv1d_weight = torch.nn.Parameter(la.conv1d.weight.data.float().squeeze(1))

            # A_log, dt_bias
            layer.ssm_a = torch.nn.Parameter(la.A_log.data.float().clone())
            layer.ssm_dt_bias = torch.nn.Parameter(la.dt_bias.data.float().clone())

            # RMSNormGated weight (init=ones, forward=w*x, no +1)
            layer.ssm_norm_weight = torch.nn.Parameter(la.norm.weight.data.float().clone())

            # Output projection [hidden, value_dim]
            layer.ssm_out_weight = torch.nn.Parameter(la.out_proj.weight.data.float().clone())

            # Norms
            layer.attn_norm_weight = torch.nn.Parameter(attn_norm_w)
            layer.ffn_norm_weight = torch.nn.Parameter(ffn_norm_w)

            # FFN
            layer.ffn_gate_weight = torch.nn.Parameter(hl.mlp.gate_proj.weight.data.float().clone())
            layer.ffn_up_weight = torch.nn.Parameter(hl.mlp.up_proj.weight.data.float().clone())
            layer.ffn_down_weight = torch.nn.Parameter(hl.mlp.down_proj.weight.data.float().clone())

        elif isinstance(layer, Qwen35AttentionLayer):
            sa = hl.self_attn  # Qwen3NextAttention

            # Q + gate packed [num_heads * head_dim * 2, hidden]
            layer.q_weight = torch.nn.Parameter(sa.q_proj.weight.data.float().clone())
            layer.k_weight = torch.nn.Parameter(sa.k_proj.weight.data.float().clone())
            layer.v_weight = torch.nn.Parameter(sa.v_proj.weight.data.float().clone())
            layer.o_weight = torch.nn.Parameter(sa.o_proj.weight.data.float().clone())

            # QK norm (Gemma3RMSNorm with +1)
            layer.q_norm_weight = torch.nn.Parameter(_gemma_rmsnorm_weight(sa.q_norm.weight.data))
            layer.k_norm_weight = torch.nn.Parameter(_gemma_rmsnorm_weight(sa.k_norm.weight.data))

            # Norms
            layer.attn_norm_weight = torch.nn.Parameter(attn_norm_w)
            layer.ffn_norm_weight = torch.nn.Parameter(ffn_norm_w)

            # FFN
            layer.ffn_gate_weight = torch.nn.Parameter(hl.mlp.gate_proj.weight.data.float().clone())
            layer.ffn_up_weight = torch.nn.Parameter(hl.mlp.up_proj.weight.data.float().clone())
            layer.ffn_down_weight = torch.nn.Parameter(hl.mlp.down_proj.weight.data.float().clone())

    # RoPE cache
    model.cos, model.sin = compute_rope_cos_sin(
        config.max_seq_len, config.rope_dimension_count, config.rope_theta
    )


if __name__ == "__main__":
    # 1. Load config
    config = Qwen35Config("weights/Qwen3.5-0.8B-BF16.yaml")
    print(f"Config: hidden={config.embedding_length}, layers={config.block_count}, "
          f"heads={config.head_count}/{config.head_count_kv}, vocab={config.vocab_size}")

    # 2. Load HuggingFace model
    print(f"Loading HF model: {MODEL_NAME} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    hf_model.eval()

    # 3. Build model & load weights from HF
    model = Qwen35Model(config)
    load_weights_from_hf(model, hf_model)

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. Forward inference (chat format)
    prompt = "1+1="
    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    chat_ids = chat_ids["input_ids"] if hasattr(chat_ids, "input_ids") else chat_ids

    print(f"Input:  {prompt!r}")
    print(f"Token ids: {chat_ids}")

    # 6. Greedy decode
    input_ids = list(chat_ids)
    max_new_tokens = 256
    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    # Prefill
    print(f"Prefill ({len(input_ids)} tokens) ...")
    with torch.no_grad():
        logits = model.forward(input_ids)

    print(f"Generating (max {max_new_tokens} tokens) ...")
    for step in range(max_new_tokens):
        with torch.no_grad():
            next_token = logits[0, -1].argmax().item()

        if next_token in EOS_TOKEN_IDS:
            break

        input_ids.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)

        if step < max_new_tokens - 1:
            with torch.no_grad():
                logits = model.forward(input_ids[-1:])

    print("\n")

    # 7. Full output
    print("=========================")
    generated = tokenizer.decode(input_ids)
    print(f"Full output:\n{generated}")

    # 8. Compare with HF model
    with torch.no_grad():
        hf_logits = hf_model(torch.tensor([input_ids])).logits
    diff = (hf_logits[0, -1].float() - logits[0, -1].float()).abs()
    print(f"\nHF logits diff (last token): max={diff.max():.6f}, mean={diff.mean():.6f}")
