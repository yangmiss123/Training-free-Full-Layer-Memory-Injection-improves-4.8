# -*- coding: utf-8 -*-
"""Qwen3-0.6B 记忆单元 pilot 核心机制库。

- compile_memory_rows: 规范答案文本 -> Qwen 前向 -> 候选层池化隐藏态
  -> 该层 input_layernorm/k_proj/k_norm/v_proj -> 锚点预旋转 key + 静态 value
- MemAttention:  在指定层把记忆行 cat 到 key/value 头部（掩码左扩 M 列放行）
- decode_loop:   统一贪心解码（无 cache，每步全量前向），支持 plain/prefix/mem 三种模式
"""
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as mq

MODEL_DIR = r"H:/Qwen3-0.6B/Qwen/Qwen3-0.6B"

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    model = model.to("cuda").eval()
    return model, tok

def _anchor_rotary(model, M):
    """cos/sin at anchor positions 0..M-1 (same rotary as model uses per position)."""
    dev = next(model.parameters()).device
    dummy = torch.zeros(1, 1, model.config.hidden_size, dtype=torch.float32, device=dev)
    pos = torch.arange(M, dtype=torch.long, device=dev).unsqueeze(0)
    cos, sin = model.model.rotary_emb(dummy, pos)
    return cos, sin

def compile_memory_rows(model, tok, answers, inject_layers, M=None):
    """answers: list[str]（每个规范答案文本）。返回 {L0: (k_mem, v_mem)}。
    k_mem/v_mem: (1, kv_heads, M, head_dim) fp32 cuda。
    """
    conf = model.config
    kv_heads = conf.num_key_value_heads
    hd = conf.head_dim
    if M is None:
        M = len(answers)
    cosA, sinA = _anchor_rotary(model, M)  # (1, M, hd)

    # 1) 收集每条答案在所有层输出的池化向量
    pooled_by_layer = {}  # layer_out_idx -> list of (hidden_dim,)
    texts = [tok.encode(a, add_special_tokens=False) for a in answers]
    for tids in texts:
        ids = torch.tensor([tids], dtype=torch.long).to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for L0 in inject_layers:
            h = out.hidden_states[L0]  # (1, len, hidden) = output of layer L0
            pooled = h.mean(dim=1).squeeze(0)  # (hidden,)
            pooled_by_layer.setdefault(L0, []).append(pooled)

    # 2) 逐层投影 + 锚点旋转
    result = {}
    for L0 in inject_layers:
        layer = model.model.layers[L0]
        attn = layer.self_attn
        k_list, v_list = [], []
        for pooled in pooled_by_layer[L0]:
            h_in = layer.input_layernorm(pooled)                      # (hidden,)
            k_raw = attn.k_norm(attn.k_proj(h_in).view(kv_heads, hd))  # RMS 按每头 128 维归一
            v_raw = attn.v_proj(h_in).view(kv_heads, hd)              # (kv_heads, hd)
            k_list.append(k_raw)
            v_list.append(v_raw)
        k_raws = torch.stack(k_list, dim=1)  # (kv_heads, M, hd) 未经旋转
        v_mem = torch.stack(v_list, dim=1)   # (kv_heads, M, hd)
        # 旋转 key（与 attention 内部 apply_rotary_pos_emb 一致，k=None 不被支持故手动）
        k_unsq = k_raws.unsqueeze(0)  # (1, kv_heads, M, hd)
        ca = cosA.unsqueeze(1)
        sa = sinA.unsqueeze(1)
        k_rot = (k_unsq * ca) + (mq.rotate_half(k_unsq) * sa)
        result[L0] = (k_rot.squeeze(0).unsqueeze(0), v_mem.unsqueeze(0))  # (1, kv_heads, M, hd)
    return result

def compile_token_rows(model, tok, texts, inject_layers):
    """方案 B：完整 token 序列的 KV 记忆行（无池化，逐 token 一对 k/v）。

    texts: 记忆文本列表（按顺序拼接为一个连续 bank）。
    返回 {L0: (k_mem, v_mem)}，k/v: (1, kv_heads, M, head_dim) fp32 cuda，
    M = bank 总 token 数。key 按 bank 内绝对位置 0..M-1 锚点预旋转。
    """
    conf = model.config
    kv_heads = conf.num_key_value_heads
    hd = conf.head_dim
    full_text = "\n\n".join(texts)
    tids = tok.encode(full_text, add_special_tokens=False)
    M = len(tids)
    ids = torch.tensor([tids], dtype=torch.long).to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    cosA, sinA = _anchor_rotary(model, M)  # (1, M, hd)
    ca = cosA.unsqueeze(1)
    sa = sinA.unsqueeze(1)

    result = {}
    for L0 in inject_layers:
        layer = model.model.layers[L0]
        attn = layer.self_attn
        h = out.hidden_states[L0][0]                    # (M, hidden) 层 L0 的输入
        h_ln = layer.input_layernorm(h)                 # (M, hidden)
        k_raw = attn.k_norm(attn.k_proj(h_ln).view(M, kv_heads, hd))  # (M, kv_heads, hd)
        v_raw = attn.v_proj(h_ln).view(M, kv_heads, hd)
        k_unsq = k_raw.transpose(0, 1).unsqueeze(0)     # (1, kv_heads, M, hd)
        v_unsq = v_raw.transpose(0, 1).unsqueeze(0)
        k_rot = (k_unsq * ca) + (mq.rotate_half(k_unsq) * sa)
        result[L0] = (k_rot.contiguous().cpu(), v_unsq.contiguous().cpu())
    return result, M

def _extend_mask(attention_mask, q_len, M, device):
    """把 (..., q, kv=seq) 的加性掩码左扩 M 列（记忆列全放行）。"""
    if attention_mask is None:
        mask = torch.zeros(1, 1, q_len, q_len + M, device=device)
        tri = torch.triu(torch.full((q_len, q_len), float("-inf"), device=device), diagonal=1)
        mask[:, :, :, M:] = tri
        return mask
    shape = attention_mask.shape
    if attention_mask.dtype == torch.bool:
        allow = torch.ones((*shape[:-1], M), dtype=torch.bool, device=device)
    else:
        allow = torch.zeros((*shape[:-1], M), dtype=attention_mask.dtype, device=device)
    return torch.cat([allow, attention_mask], dim=-1)

class MemAttention(nn.Module):
    """在注入层重实现 Qwen3Attention.forward，将 M 行记忆 KV 置于序列键头部。"""
    def __init__(self, orig_attn, k_mem, v_mem):
        super().__init__()
        self.orig = orig_attn
        self.k_mem = k_mem  # (1, kv_heads, M, hd)
        self.v_mem = v_mem

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        o = self.orig
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, o.head_dim)
        qs = o.q_norm(o.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        ks = o.k_norm(o.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        vs = o.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        qs, ks = mq.apply_rotary_pos_emb(qs, ks, cos, sin)

        M = self.k_mem.shape[-2]
        k_mem = self.k_mem.to(qs.dtype).to(qs.device)
        v_mem = self.v_mem.to(qs.dtype).to(qs.device)
        ks = torch.cat([k_mem, ks], dim=2)
        vs = torch.cat([v_mem, vs], dim=2)
        q_len = qs.shape[2]
        attention_mask = _extend_mask(attention_mask, q_len, M, qs.device)

        attention_interface = mq.eager_attention_forward
        if o.config._attn_implementation != "eager":
            attention_interface = mq.ALL_ATTENTION_FUNCTIONS[o.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            o, qs, ks, vs, attention_mask,
            dropout=0.0 if not o.training else o.attention_dropout,
            scaling=o.scaling,
            sliding_window=o.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return o.o_proj(attn_output), attn_weights

def attach_memory(model, L0, k_mem, v_mem, device=None):
    """把注入层 L0 的 self_attn 替换为带记忆版本（返回原模块以便还原）。"""
    layer = model.model.layers[L0]
    orig = layer.self_attn
    if device is not None:
        k_mem = k_mem.to(device)
        v_mem = v_mem.to(device)
    layer.self_attn = MemAttention(orig, k_mem, v_mem)
    return orig

def detach_memory(model, L0, orig):
    model.model.layers[L0].self_attn = orig

def decode_loop(model, tok, prompt_ids, mode="plain", mem_rows=None,
                max_new=48, eos_id=None):
    """统一贪心解码（无 KV cache，每步全量前向）。
    mode: plain(裸) | prefix(文本前缀) | mem(头部锚定 KV 注入)
    mem_rows: {L0: (k_mem, v_mem)} 仅 mem 模式用；此时 prompt_ids 为问题本身。
    """
    if eos_id is None:
        eos_id = tok.eos_token_id
    device = next(model.parameters()).device
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if mode == "prefix":
        pass  # prompt_ids 已含前缀（调用方拼接）
    M = 0
    attached = []
    if mode == "mem" and mem_rows:
        first = next(iter(mem_rows.values()))
        M = first[0].shape[-2]
        for L0, (k_mem, v_mem) in mem_rows.items():
            attached.append((L0, attach_memory(model, L0, k_mem, v_mem, device=device)))

    gen = []
    ids = prompt_ids.clone()
    try:
        for _ in range(max_new):
            with torch.no_grad():
                if mode == "mem" and M > 0:
                    pos = torch.arange(M, M + ids.shape[1], device=device).unsqueeze(0)
                    out = model(input_ids=ids, position_ids=pos, use_cache=False)
                else:
                    out = model(input_ids=ids, use_cache=False)
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1).item()
            if nxt == eos_id:
                break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    finally:
        for L0, orig in attached:
            detach_memory(model, L0, orig)
    return tok.decode(gen, skip_special_tokens=True)
