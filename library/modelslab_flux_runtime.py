import os

import torch


PAIRED_COMPILE = os.environ.get("FLUX_KOHYA_PAIRED_COMPILE") == "1"
FA3_ENABLED = os.environ.get("FLUX_KOHYA_FA3") == "1"
LORA_NORM_DIAGNOSTICS = os.environ.get("FLUX_KOHYA_LORA_NORM_DIAGNOSTICS", "1") == "1"

if FA3_ENABLED:
    from flash_attn_interface import flash_attn_func


def fa3_attention(q, k, v):
    return flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False
    ).transpose(1, 2)


def compile_region(function):
    if not PAIRED_COMPILE:
        return function
    return torch.compile(
        function,
        backend="inductor",
        dynamic=False,
        options={"triton.cudagraphs": False},
    )


@compile_region
def single_pair(first, second, img, vec, pe, txt_attention_mask):
    img = first(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
    return second(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
