# -*- coding: utf-8 -*-
"""Pilot 迭代2b：token-wise KV 注入 —— 层覆盖度实验 + 全层等价性验证。

配置:
  arm0_plain           裸
  arm1_textprefix      doc 文本前缀（真实基准）
  arm2_doc_allL        bank=doc  KV 注入全部 28 层  （应≈arm1，验证机制正确性）
  arm2_ans_allL        bank=20答案 KV 注入全部 28 层（内容最佳情形）
  arm2_ans_L24         bank=答案 仅层24（单层退化点）
  arm2_ans_L14_L24     bank=答案 层14+24（稀疏层）
"""
import argparse, json, os, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib
from content_quant_svc import get_doc, get_qa

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)
SYS = "请用一句话直接回答下面的问题。"
ALL = list(range(28))

def build_prompt(q, knowledge=None):
    return (f"{SYS}\n\n{knowledge}\n\n问题：{q}\n答案：" if knowledge
            else f"{SYS}\n\n问题：{q}\n答案：")

def grade(text, checks):
    t = text.lower()
    return sum(1 for c in checks if c.lower() in t) / len(checks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--tag", default="iter2b")
    args = ap.parse_args()
    qa = get_qa()
    doc = get_doc()
    model, tok = memlib.load_model()

    t0 = time.time()
    bank_doc, M_doc = memlib.compile_token_rows(model, tok, [doc], ALL)
    bank_ans, M_ans = memlib.compile_token_rows(model, tok, [it["a"] for it in qa], ALL)
    print(f"[compile] M_doc={M_doc} M_ans={M_ans} {time.time()-t0:.0f}s", flush=True)

    results = {"tag": args.tag, "qa": qa, "arms": {}}

    def run_arm(name, mode, mem_rows=None, knowledge=None):
        rows = []
        for i, item in enumerate(qa):
            pids = tok.encode(build_prompt(item["q"], knowledge), add_special_tokens=False)
            t1 = time.time()
            text = memlib.decode_loop(model, tok, pids, mode=mode, mem_rows=mem_rows,
                                      max_new=args.max_new)
            rows.append({"i": i, "v": item["v"], "out": text, "score": grade(text, item["checks"]),
                         "sec": round(time.time() - t1, 1)})
        sc = [r["score"] for r in rows]
        results["arms"][name] = {"rows": rows, "sum": {"mean": round(sum(sc)/len(sc), 3),
                                                       "full_hit": sum(1 for s in sc if s == 1.0),
                                                       "n": len(sc)}}
        with open(os.path.join(RESULT_DIR, f"{args.tag}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"[{name}] mean={results['arms'][name]['sum']['mean']} "
              f"full={results['arms'][name]['sum']['full_hit']}/20 ({time.time()-t0:.0f}s)", flush=True)

    run_arm("arm0_plain", "plain")
    run_arm("arm1_textprefix", "prefix", knowledge=doc)
    run_arm("arm2_doc_allL", "mem", mem_rows={L: bank_doc[L] for L in ALL})
    run_arm("arm2_ans_allL", "mem", mem_rows={L: bank_ans[L] for L in ALL})
    run_arm("arm2_ans_L24", "mem", mem_rows={24: bank_ans[24]})
    run_arm("arm2_ans_L14_L24", "mem", mem_rows={14: bank_ans[14], 24: bank_ans[24]})

    out_path = os.path.join(RESULT_DIR, f"{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out_path}", flush=True)
    print("\n=== 汇总 ===")
    for name, arm in results["arms"].items():
        s = arm["sum"]
        print(f"{name:22s} mean={s['mean']:.3f}  full_hit={s['full_hit']}/20")

if __name__ == "__main__":
    main()
