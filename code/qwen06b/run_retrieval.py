# -*- coding: utf-8 -*-
"""检索实验：3 主题 —— 逐层检索命中率 + top-k(=正确主题) 注入 vs 全注入 vs 错误主题注入。

主题 A: quant_svc (20 QA)  B: events_t0 (9 QA)  C: factor_mgmt (8 QA)  共 37 题
命中率: 原型 = 主题 doc 在层 L 的池化隐藏态；query = 问题在层 L 的隐藏态(3 池化变体)
注入臂(子集 16 题, 全层注入, 协议=指令封装+max_new24):
  arm0_plain / arm_own(本主题 doc KV) / arm_all(3 主题 doc KV) / arm_wrong(其它两主题 KV)
"""
import argparse, json, os, sys, time, torch
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlib
from content_quant_svc import get_doc, get_qa as qaA
from content_extra import TOPIC_B, TOPIC_C

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)
SYS = "请用一句话直接回答下面的问题。"
ALL = list(range(28))
RETR_LAYERS = [8, 14, 20, 24, 27]

def build_prompt(q):
    return f"{SYS}\n\n问题：{q}\n答案："

def grade(text, checks):
    t = text.lower()
    return sum(1 for c in checks if c.lower() in t) / len(checks)

def mean_hidden(model, tok, text, layers):
    """返回 {L: (mean_all, mean_last4, last)} 都在 cuda fp32。"""
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)], dtype=torch.long).to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hh = out.hidden_states
    res = {}
    for L in layers:
        h = hh[L][0]  # (T, hidden)
        res[L] = (h.mean(dim=0), h[-4:].mean(dim=0), h[-1])
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--tag", default="retrieval1")
    ap.add_argument("--subA", type=int, default=6)
    ap.add_argument("--subB", type=int, default=5)
    ap.add_argument("--subC", type=int, default=5)
    args = ap.parse_args()

    topics = [
        {"name": "A_quant_svc", "doc": get_doc(), "qa": qaA()},
        {"name": "B_events_t0", "doc": TOPIC_B["doc"], "qa": TOPIC_B["qa"]},
        {"name": "C_factor_mgmt", "doc": TOPIC_C["doc"], "qa": TOPIC_C["qa"]},
    ]
    # 全部问题带真实主题标签
    all_q = []
    for t in topics:
        for it in t["qa"]:
            all_q.append({"topic": t["name"], "q": it["q"], "a": it["a"], "checks": it["checks"]})

    model, tok = memlib.load_model()
    t0 = time.time()

    # ---- 1) 编译各主题 doc bank（全层）+ 组合 bank ----
    banks = {}
    for t in topics:
        banks[t["name"]] = memlib.compile_token_rows(model, tok, [t["doc"]], ALL)[0]
    names = [t["name"] for t in topics]
    bank_all = memlib.compile_token_rows(model, tok, [t["doc"] for t in topics], ALL)[0]
    M_all = next(iter(bank_all.values()))[0].shape[-2]
    banks_wrong = {}
    for n in names:
        others = [t["doc"] for t in topics if t["name"] != n]
        banks_wrong[n] = memlib.compile_token_rows(model, tok, others, ALL)[0]
    Ms = {n: next(iter(banks[n].values()))[0].shape[-2] for n in names}
    print(f"[compile] M={Ms} all={M_all} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2) 逐层检索命中率 ----
    proto = {t["name"]: mean_hidden(model, tok, t["doc"], RETR_LAYERS) for t in topics}
    qreprs = []
    for item in all_q:
        qreprs.append(mean_hidden(model, tok, item["q"], RETR_LAYERS))
    variants = ["mean_all", "mean_last4", "last"]
    hitrate = {}
    for L in RETR_LAYERS:
        for vi, vname in enumerate(variants):
            correct = 0
            margins = []
            for item, qr in zip(all_q, qreprs):
                qv = qr[L][vi]
                sims = {n: torch.cosine_similarity(qv, proto[n][L][vi], dim=0).item() for n in names}
                pred = max(sims, key=sims.get)
                hit = pred == item["topic"]
                correct += hit
                # margin: 正确主题相似度 - 次高
                ranked = sorted(sims.values(), reverse=True)
                gt_sim = sims[item["topic"]]
                margins.append(gt_sim - ranked[1] if hit else ranked[0] - gt_sim)
            acc = correct / len(all_q)
            hitrate[f"L{L}_{vname}"] = {"acc": round(acc, 3), "n": len(all_q),
                                        "mean_margin": round(sum(margins)/len(margins), 4)}
            print(f"[hitrate] L{L:2d} {vname:10s} acc={acc:.3f} margin={hitrate[f'L{L}_{vname}']['mean_margin']:.4f}", flush=True)

    # ---- 3) 注入臂（子集, 全层注入）----
    sub = []
    sub += [{"topic": "A_quant_svc", "item": it} for it in topics[0]["qa"][: args.subA]]
    sub += [{"topic": "B_events_t0", "item": it} for it in topics[1]["qa"][: args.subB]]
    sub += [{"topic": "C_factor_mgmt", "item": it} for it in topics[2]["qa"][: args.subC]]
    print(f"[decode subset] n={len(sub)}", flush=True)

    results = {"tag": args.tag, "hitrate": hitrate, "arms": {}, "subset": [
        {"topic": s["topic"], "q": s["item"]["q"]} for s in sub]}

    def run_arm(name, mem_for=None):
        """mem_for: callable(s)->mem_rows dict 或 None(plain)。"""
        rows = []
        for s in sub:
            pids = tok.encode(build_prompt(s["item"]["q"]), add_special_tokens=False)
            mem_rows = mem_for(s) if mem_for else None
            mode = "mem" if mem_for else "plain"
            t1 = time.time()
            text = memlib.decode_loop(model, tok, pids, mode=mode, mem_rows=mem_rows,
                                      max_new=args.max_new)
            rows.append({"topic": s["topic"], "q": s["item"]["q"], "out": text,
                         "score": grade(text, s["item"]["checks"]),
                         "sec": round(time.time() - t1, 1)})
        sc = [r["score"] for r in rows]
        results["arms"][name] = {"rows": rows, "sum": {"mean": round(sum(sc)/len(sc), 3),
                                                       "full_hit": sum(1 for x in sc if x == 1.0),
                                                       "n": len(sc)}}
        with open(os.path.join(RESULT_DIR, f"{args.tag}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"[{name}] mean={results['arms'][name]['sum']['mean']} "
              f"full={results['arms'][name]['sum']['full_hit']}/{len(sc)} ({time.time()-t0:.0f}s)", flush=True)

    run_arm("arm0_plain")
    run_arm("arm_own", mem_for=lambda s: {L: banks[s["topic"]][L] for L in ALL})
    run_arm("arm_all", mem_for=lambda s: {L: bank_all[L] for L in ALL})
    run_arm("arm_wrong", mem_for=lambda s: {L: banks_wrong[s["topic"]][L] for L in ALL})

    out_path = os.path.join(RESULT_DIR, f"{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out_path}", flush=True)
    print("\n=== 注入臂汇总 ===")
    for name, arm in results["arms"].items():
        s = arm["sum"]
        print(f"{name:12s} mean={s['mean']:.3f}  full_hit={s['full_hit']}/{s['n']}")

if __name__ == "__main__":
    main()
