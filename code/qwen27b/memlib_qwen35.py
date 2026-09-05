# -*- coding: utf-8 -*-
"""Qwen3.8-27B (qwen3_5 混合架构, transformers 5.15.1, 海光 DCU) 的 KV 记忆注入核心库。

- 64 层仅 16 个 full_attention 层（idx 3,7,...,63）可注入；linear_attention 层不注入。
- 纯文本直驱 model.model.language_model（Qwen3_5TextModel），logits 用 model.lm_head。
- 编译：记忆文本冻结前向 -> 每 full 层输入隐藏态 -> input_layernorm -> k_proj/k_norm/v_proj
  -> 锚点 0..M-1 旋转（partial rotary，apply_rotary_pos_emb 自带 pass-through）。
- 注入：MemAttention35 复刻 Qwen3_5Attention.forward（q_proj gate 分块、sigmoid gate），记忆 cat 键头、掩码左扩。
- bf16 全程；无 cache 逐轮全量前向。
"""
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn

MODEL_DIR = "/root/Qwen3.8-27B"

def load_model(model_dir=MODEL_DIR):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model = model.to("cuda").eval()
    return model, tok

def full_layer_idx(model):
    tc = model.config.text_config
    return [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]

def _run_lm(model, ids, pos=None):
    """返回 last_hidden(bf16)。lm = Qwen3_5TextModel。"""
    lm = model.model.language_model
    out = lm(input_ids=ids, position_ids=pos, use_cache=False)
    return out.last_hidden_state

def compile_token_rows(model, tok, texts, layers=None):
    """记忆文本(列表)拼为连续 bank -> 各 full 层静态 k/v。返回 ({L: (k_mem, v_mem)}, M)。
    k_mem 已按锚点 0..M-1 partial 旋转。bf16。"""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    tc = model.config.text_config
    kv_heads = tc.num_key_value_heads
    hd = tc.head_dim
    if layers is None:
        layers = full_layer_idx(model)
    lm = model.model.language_model
    layers_list = lm.layers

    full_text = "\n\n".join(texts)
    tids = tok.encode(full_text, add_special_tokens=False)
    M = len(tids)
    ids = torch.tensor([tids], dtype=torch.long, device="cuda")

    # 1) 前向记忆文本，用 forward_pre_hook 抓取每个 full 层的输入隐藏态（确定性，不依赖 hidden_states API）
    captured = {}
    hooks = []
    def _mk(i):
        def _hook(module, args):
            captured[i] = args[0].detach()
        return _hook
    for i in layers:
        hooks.append(layers_list[i].register_forward_pre_hook(_mk(i)))
    try:
        with torch.no_grad():
            lm(input_ids=ids, position_ids=None, use_cache=False)
    finally:
        for hk in hooks:
            hk.remove()
    hid_of = lambda i: captured[i]  # (1, M, hidden)

    # 2) 锚点 cos/sin（0..M-1, 2D）
    with torch.no_grad():
        dummy = torch.zeros(1, 1, tc.hidden_size, dtype=torch.bfloat16, device="cuda")
        pos_a = torch.arange(M, dtype=torch.long, device="cuda").unsqueeze(0)
        cosA, sinA = lm.rotary_emb(dummy, pos_a)  # (1, M, 64)；apply_rotary_pos_emb 内部自行 unsqueeze(1)

    result = {}
    with torch.no_grad():
        for i in layers:
            layer = layers_list[i]
            attn = layer.self_attn
            h = hid_of(i)[0]                       # (M, hidden)
            h_ln = layer.input_layernorm(h)        # (M, hidden)
            k_raw = attn.k_norm(attn.k_proj(h_ln).view(M, kv_heads, hd))   # (M, kv, hd)
            v_raw = attn.v_proj(h_ln).view(M, kv_heads, hd)
            k_u = k_raw.transpose(0, 1).unsqueeze(0)  # (1, kv, M, hd)
            v_u = v_raw.transpose(0, 1).unsqueeze(0)
            k_rot, _ = mq.apply_rotary_pos_emb(k_u, k_u, cosA, sinA)  # partial: 前 64/256 维
            result[i] = (k_rot.contiguous().cpu(), v_u.contiguous().cpu())
    return result, M

def _extend_mask(attention_mask, q_len, M, device):
    if attention_mask is None:
        mask = torch.zeros(1, 1, q_len, q_len + M, device=device)
        tri = torch.triu(torch.full((q_len, q_len), float("-inf"), device=device), diagonal=1)
        mask[:, :, :, M:] = tri
        return mask
    sh = attention_mask.shape
    if attention_mask.dtype == torch.bool:
        allow = torch.ones((*sh[:-1], M), dtype=torch.bool, device=device)
    else:
        allow = torch.zeros((*sh[:-1], M), dtype=attention_mask.dtype, device=device)
    return torch.cat([allow, attention_mask], dim=-1)

class MemAttention35(nn.Module):
    """复刻 Qwen3_5Attention.forward + 记忆 KV 注入（cat 键头，掩码左扩 M）。"""
    def __init__(self, orig, k_mem, v_mem):
        super().__init__()
        self.orig = orig
        self.k_mem = k_mem
        self.v_mem = v_mem

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        o = self.orig
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, o.head_dim)
        qg = o.q_proj(hidden_states).view(*input_shape, -1, o.head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = o.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = o.k_norm(o.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = o.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        M = self.k_mem.shape[-2]
        k_mem = self.k_mem.to(query_states.dtype).to(query_states.device)
        v_mem = self.v_mem.to(query_states.dtype).to(query_states.device)
        key_states = torch.cat([k_mem, key_states], dim=2)
        value_states = torch.cat([v_mem, value_states], dim=2)
        attention_mask = _extend_mask(attention_mask, query_states.shape[2], M, query_states.device)

        attn_output, attn_weights = mq.eager_attention_forward(
            o, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not o.training else o.attention_dropout,
            scaling=o.scaling, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return o.o_proj(attn_output), attn_weights

def attach(model, L, k_mem, v_mem, device="cuda"):
    layer = model.model.language_model.layers[L]
    orig = layer.self_attn
    layer.self_attn = MemAttention35(orig, k_mem.to(device), v_mem.to(device))
    return orig

def detach(model, L, orig):
    model.model.language_model.layers[L].self_attn = orig

def decode(model, tok, prompt_ids, mem_rows=None, max_new=24, eos_id=None):
    """mem_rows: {full层: (k_mem, v_mem)}；所有层共享同一 M。无 cache 逐轮全量前向。"""
    if eos_id is None:
        eos_id = tok.eos_token_id
    lm = model.model.language_model
    device = "cuda"
    M = 0
    attached = []
    if mem_rows:
        first = next(iter(mem_rows.values()))
        M = first[0].shape[-2]
        for L, (k, v) in mem_rows.items():
            attached.append((L, attach(model, L, k, v, device)))
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = []
    try:
        for _ in range(max_new):
            with torch.no_grad():
                if M > 0:
                    pos = torch.arange(M, M + ids.shape[1], device=device).unsqueeze(0)
                    h = _run_lm(model, ids, pos)
                else:
                    h = _run_lm(model, ids, None)
            logits = model.lm_head(h[:, -1:, :])[0, 0]
            nxt = logits.argmax(dim=-1).item()
            if nxt == eos_id:
                break
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    finally:
        for L, orig in attached:
            detach(model, L, orig)
    return tok.decode(gen, skip_special_tokens=True)
