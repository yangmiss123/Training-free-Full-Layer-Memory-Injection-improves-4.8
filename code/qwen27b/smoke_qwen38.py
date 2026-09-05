# -*- coding: utf-8 -*-
"""Qwen3.8-27B 容器冒烟/适配探针：加载 -> 结构检查 -> rotary 形状 -> 纯文本前向/logits
-> 无缓存解码速度 -> compile_token_rows 验证。运行: bash /root/kv/smoke_run.sh
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "/root/kv")
import torch
import memlib_qwen35 as ml

def main():
    t0 = time.time()
    model, tok = ml.load_model()
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    print(f"[vram] {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    lm = model.model.language_model
    print(f"[lm] type={type(lm).__name__} layers={len(lm.layers)}", flush=True)
    tc = model.config.text_config
    print(f"[cfg] kv_heads={tc.num_key_value_heads} heads={tc.num_attention_heads} "
          f"head_dim={tc.head_dim} hidden={tc.hidden_size} vocab={tc.vocab_size}", flush=True)
    full = ml.full_layer_idx(model)
    print(f"[full_layers] {full} n={len(full)}", flush=True)

    # rotary 形状
    with torch.no_grad():
        dummy = torch.zeros(1, 1, tc.hidden_size, dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(6, dtype=torch.long, device="cuda").unsqueeze(0)
        cos, sin = lm.rotary_emb(dummy, pos)
        print(f"[rotary] cos={tuple(cos.shape)} sin={tuple(sin.shape)}", flush=True)

    # 纯文本前向 + logits
    ids = tok.encode("法国的首都是", add_special_tokens=False)
    tids = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = ml._run_lm(model, tids, None)
        logits = model.lm_head(h[:, -1:, :])[0, 0]
    topk = logits.topk(5).indices.tolist()
    print(f"[logits] top5={[tok.decode([t]) for t in topk]}", flush=True)

    # 解码 6 token 测速（plain）
    t1 = time.time()
    out = ml.decode(model, tok, ids, max_new=6)
    dt = time.time() - t1
    print(f"[decode] out={out!r} {6/dt:.1f} tok/s", flush=True)

    # compile_token_rows 验证（1 短文本, full 层）
    t2 = time.time()
    bank, M = ml.compile_token_rows(model, tok, ["测试记忆文本：gamma 取 0.037，股票池 708 只。"],
                                    layers=full)
    print(f"[compile] M={M} time={time.time()-t2:.0f}s", flush=True)
    L0 = full[0]
    k, v = bank[L0]
    print(f"[compile] L{L0} k={tuple(k.shape)} v={tuple(v.shape)} dtype={k.dtype}", flush=True)

    # mem 解码 6 token（注入全部 full 层）
    t3 = time.time()
    out2 = ml.decode(model, tok, ids, mem_rows={L: bank[L] for L in full}, max_new=6)
    dt3 = time.time() - t3
    print(f"[mem_decode] out={out2!r} {6/dt3:.1f} tok/s", flush=True)
    print(f"[vram_final] {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    print("[SMOKE DONE]", flush=True)

if __name__ == "__main__":
    main()
