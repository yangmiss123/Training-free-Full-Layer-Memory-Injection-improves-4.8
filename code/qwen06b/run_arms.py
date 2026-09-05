# -*- coding: utf-8 -*-
"""Pilot：量化SVC主题 —— Arm0(裸) vs Arm1(文本前缀) vs Arm2(KV注入, 逐层)。

用法:
  python run_arms.py --layers 14 24 --max_new 48 [--limit 20] [--tag run1]
判分: 每题检查 checks 子串（小写）在生成文本中的命中比例。
输出: H:/deepseek/pilot/results/{tag}.json  + 控制台汇总表
"""
import argparse, json, os, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import memlib
from content_quant_svc import get_doc, get_qa

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

def grade(text, checks):
    t = text.lower()
    return sum(1 for c in checks if c.lower() in t) / len(checks)

def run_arm(model, tok, qa, mode, mem_rows=None, max_new=48, prefix_ids=None):
    rows = []
    for i, item in enumerate(qa):
        qids = tok.encode(item["q"], add_special_tokens=False)
        if mode == "prefix":
            pids = list(prefix_ids) + qids
        else:
            pids = qids
        t0 = time.time()
        text = memlib.decode_loop(model, tok, pids, mode=mode, mem_rows=mem_rows, max_new=max_new)
        dt = time.time() - t0
        rows.append({"i": i, "v": item["v"], "q": item["q"], "gold": item["a"],
                     "out": text, "score": grade(text, item["checks"]),
                     "checks": item["checks"], "sec": round(dt, 1)})
    return rows

def summarize(name, rows):
    sc = [r["score"] for r in rows]
    acc = sum(1 for s in sc if s == 1.0)
    return {"arm": name, "n": len(rows), "mean": round(sum(sc) / len(sc), 3),
            "full_hit": acc, "per_q": [r["score"] for r in rows]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[14, 24])
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args()

    doc = get_doc()
    qa = get_qa()[: args.limit]
    print(f"[content] doc_tokens={len(doc.split())} qa={len(qa)}", flush=True)

    t0 = time.time()
    model, tok = memlib.load_model()
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    tok.padding_side = "left"

    # 编译记忆行（QA 的规范答案文本 -> 各注入层 k/v）
    answers = [it["a"] for it in qa]
    t1 = time.time()
    mem_rows_all = memlib.compile_memory_rows(model, tok, answers, args.layers)
    print(f"[compile] layers={args.layers} {time.time()-t1:.0f}s", flush=True)

    results = {"tag": args.tag, "doc": doc, "qa": qa, "arms": {}}

    # Arm 0: 裸
    t2 = time.time()
    r0 = run_arm(model, tok, qa, "plain", max_new=args.max_new)
    results["arms"]["arm0_plain"] = {"rows": r0, "sum": summarize("arm0_plain", r0)}
    print(f"[arm0] {time.time()-t2:.0f}s mean={results['arms']['arm0_plain']['sum']['mean']}", flush=True)

    # Arm 1: 文本前缀（内容天花板）
    prefix_ids = tok.encode(doc, add_special_tokens=False)
    t3 = time.time()
    r1 = run_arm(model, tok, qa, "prefix", max_new=args.max_new, prefix_ids=prefix_ids)
    results["arms"]["arm1_textprefix"] = {"rows": r1, "sum": summarize("arm1_textprefix", r1)}
    print(f"[arm1] {time.time()-t3:.0f}s mean={results['arms']['arm1_textprefix']['sum']['mean']}", flush=True)

    # Arm 2: KV 注入（逐层）
    for L0 in args.layers:
        mem_rows = {L0: mem_rows_all[L0]}
        t4 = time.time()
        r2 = run_arm(model, tok, qa, "mem", mem_rows=mem_rows, max_new=args.max_new)
        key = f"arm2_kvinj_L{L0}"
        results["arms"][key] = {"rows": r2, "sum": summarize(key, r2)}
        print(f"[{key}] {time.time()-t4:.0f}s mean={results['arms'][key]['sum']['mean']}", flush=True)

    out_path = os.path.join(RESULT_DIR, f"{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out_path}", flush=True)

    # 汇总表
    print("\n=== 汇总 ===")
    for name, arm in results["arms"].items():
        s = arm["sum"]
        print(f"{name:20s} mean={s['mean']:.3f}  full_hit={s['full_hit']}/{s['n']}")

if __name__ == "__main__":
    main()
