# -*- coding: utf-8 -*-
"""Qwen3.8-27B 验证：arm0 裸 / arm1 文本前缀 / arm2 16全层注入 / 覆盖度子集(4/8层)。

用法(容器内): export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
  python3 /root/kv/run_qwen38_validate.py --tag v1 --max_new 24 --qa_n 20
判分与 0.6B 实验同协议（锚点子串命中）。输出 /root/kv/results/{tag}.json
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import memlib_qwen35 as ml
from content_quant_svc import get_doc, get_qa

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
SYS = "请用一句话直接回答下面的问题。"

def build_prompt(q, knowledge=None):
    return (f"{SYS}\n\n{knowledge}\n\n问题：{q}\n答案：" if knowledge
            else f"{SYS}\n\n问题：{q}\n答案：")

def grade(text, checks):
    t = text.lower()
    return sum(1 for c in checks if c.lower() in t) / len(checks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--qa_n", type=int, default=20)
    ap.add_argument("--model", default="/root/Qwen3.8-27B")
    args = ap.parse_args()

    doc = get_doc()
    qa = get_qa()[: args.qa_n]
    print(f"[content] qa={len(qa)} doc_words={len(doc.split())}", flush=True)
    t0 = time.time()
    model, tok = ml.load_model(args.model)
    full_idx = ml.full_layer_idx(model)
    print(f"[load] {time.time()-t0:.0f}s full_layers={full_idx}", flush=True)
    print(f"[gpu] vram_GB={torch.cuda.memory_allocated()/1e9:.1f}", flush=True)

    # banks: doc(756 tok 级) 与 answers
    bank_doc, M_doc = ml.compile_token_rows(model, tok, [doc], layers=full_idx)
    bank_ans, M_ans = ml.compile_token_rows(model, tok, [it["a"] for it in qa], layers=full_idx)
    print(f"[compile] M_doc={M_doc} M_ans={M_ans} ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()

    results = {"tag": args.tag, "arms": {}}

    def run_arm(name, mode="plain", mem_for=None, knowledge=None):
        rows = []
        for it in qa:
            pids = tok.encode(build_prompt(it["q"], knowledge), add_special_tokens=False)
            mr = mem_for() if mem_for else None
            t1 = time.time()
            text = ml.decode(model, tok, pids, mem_rows=mr, max_new=args.max_new)
            rows.append({"q": it["q"], "out": text,
                         "score": grade(text, it["checks"]),
                         "sec": round(time.time() - t1, 1)})
        sc = [r["score"] for r in rows]
        results["arms"][name] = {"rows": rows, "sum": {
            "mean": round(sum(sc) / len(sc), 3),
            "full_hit": sum(1 for s in sc if s == 1.0), "n": len(sc)}}
        with open(os.path.join(RES, f"{args.tag}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"[{name}] mean={results['arms'][name]['sum']['mean']} "
              f"full={results['arms'][name]['sum']['full_hit']}/{len(sc)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    run_arm("arm0_plain", "plain")
    run_arm("arm1_textprefix", knowledge=doc)
    run_arm("arm2_doc_all16", mem_for=lambda: {L: bank_doc[L] for L in full_idx})
    run_arm("arm2_ans_all16", mem_for=lambda: {L: bank_ans[L] for L in full_idx})
    # 覆盖度子集：每 2 个取 1 (8层) / 每 4 个取 1 (4层)
    sub8 = full_idx[::2]
    sub4 = full_idx[::4]
    run_arm("arm2_doc_8L", mem_for=lambda: {L: bank_doc[L] for L in sub8})
    run_arm("arm2_doc_4L", mem_for=lambda: {L: bank_doc[L] for L in sub4})

    print("\n=== 汇总 ===")
    for n, arm in results["arms"].items():
        s = arm["sum"]
        print(f"{n:20s} mean={s['mean']:.3f} full={s['full_hit']}/{s['n']}")

if __name__ == "__main__":
    main()
