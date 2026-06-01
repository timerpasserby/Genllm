import yaml
import struct
import torch
from gguf import GGUFTorchLoader
from transformers import AutoTokenizer
from models.Qwen35 import Qwen35Config, Qwen35Model


def _build_tensor_list(yaml_path: str) -> list:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    weight = raw["weight"]
    config = raw["config"]

    with open(config["model_path"], "rb") as f:
        f.read(4)
        f.read(4)
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        meta_kv_count = struct.unpack('<Q', f.read(8))[0]

        for _ in range(meta_kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            val_type = struct.unpack('<I', f.read(4))[0]
            if val_type == 8:
                strlen = struct.unpack('<Q', f.read(8))[0]
                f.read(strlen)
            elif val_type == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    if arr_type == 8:
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.read(slen)
                    elif arr_type in (0, 1):
                        f.read(1)
                    elif arr_type in (2, 3, 12, 13):
                        f.read(2)
                    elif arr_type in (4, 5, 10):
                        f.read(4)
                    elif arr_type == 6:
                        f.read(4)
                    elif arr_type == 7:
                        f.read(1)
                    elif arr_type == 11:
                        f.read(8)
                    else:
                        f.read(4)
            elif val_type in (0, 1):
                f.read(1)
            elif val_type in (2, 3, 12, 13):
                f.read(2)
            elif val_type in (4, 5, 10):
                f.read(4)
            elif val_type == 6:
                f.read(4)
            elif val_type == 7:
                f.read(1)
            elif val_type == 11:
                f.read(8)
            else:
                f.read(4)

        tensor_list = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack('<I', f.read(4))[0]
            f.read(8 * n_dims)
            f.read(4)
            f.read(8)
            tensor_list.append((name, weight[name][0], weight[name][1]))

    return tensor_list


if __name__ == "__main__":

    config_path = "weights/Qwen3.5-0.8B-BF16.yaml"

    config = Qwen35Config(config_path)
    print(f"Config: embedding={config.embedding_length}, blocks={config.block_count}, "
          f"heads={config.head_count}/{config.head_count_kv}, vocab={config.vocab_size}")

    tensor_list = _build_tensor_list(config_path)
    # print(tensor_list) 

    loader = GGUFTorchLoader(config.model_path, config.base_offset, tensor_list)

    model = Qwen35Model(config)
    model.load_weights(loader)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    prompt = "1+1="
    chat_result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    chat_ids = chat_result["input_ids"] if hasattr(chat_result, "input_ids") else chat_result

    print(f"Token ids: {chat_ids}")

    input_ids = list(chat_ids)
    max_new_tokens = 512
    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    print(EOS_TOKEN_IDS)

    # Prefill: process all prompt tokens at once, populating states/cache
    print(f"Prefill ({len(input_ids)} tokens) ...")
    with torch.no_grad():
        logits = model.forward(input_ids)

    print(f"Generating (max {max_new_tokens} tokens) ...")
    for step in range(max_new_tokens):
        # First token comes from prefill logits; after that, states are already
        # at the right position, feed only the NEW token each decode step
        with torch.no_grad():
            next_token = logits[0, -1].argmax().item()

        if next_token in EOS_TOKEN_IDS:
            break

        input_ids.append(next_token)

        if step < max_new_tokens - 1:
            with torch.no_grad():
                logits = model.forward(input_ids[-1:])

    print("\n")

    print(input_ids)
    print(f"Full output:\n{tokenizer.decode(input_ids)}")
