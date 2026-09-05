# Training-free Full-Layer Memory Injection improves 4.8×
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22305403.svg)](https://doi.org/10.5281/zenodo.22305403)

**Official implementation of the paper**:
*基于编译期键值缓存的全层记忆注入方法：使用更大参数模型对小模型的知识注入*
(Full-Layer Memory Injection via Compile-Time Key-Value Caches: An External-Knowledge Mechanism for Small-Parameter Language Models)

- 📄 **Paper (Zenodo preprint, Chinese)**: DOI **10.5281/zenodo.22305403** — https://zenodo.org/records/22305403
- 📁 Full text: `paper/paper_zh_preprint.pdf`
- ⚖️ Code license: **MIT** · Paper text license: CC BY 4.0

---

## What is this?

A **training-free** method that injects externally compiled knowledge into a *frozen* small language model, via **compile-time key-value (KV) caches**:

1. An external compiler model (e.g., DeepSeek-V4-Flash) distills knowledge into canonical text (text domain only — it never emits vectors).
2. The **executor model itself** (frozen) vectorizes the text into per-layer static **KV banks** (token-level; rotated at anchor positions 0..M−1).
3. At inference the banks are injected into the executor's attention layers (head-anchored, sequence shifted by M) — **no retraining, no per-query re-encoding**, and memory size is decoupled from the context window.


> ✅ **Reproduction verified**: running `run_arms3.py --tag repro_gh` with this repo's `code/qwen06b/` reproduces paper Table 2 exactly (0.079 / 0.571 / 0.621 / 0.571 / 0.079 / 0.062). Evidence: `results/repro_gh.json`.

## Main results (private-domain QA, same protocol)

| Arm | Qwen3-0.6B (28-layer dense) | Qwen3.8-27B (hybrid: 16/64 full-attention) |
|---|---|---|
| No memory (baseline) | 0.079 | 0.133 |
| Text prefix (ceiling ref.) | 0.571 | 0.583 |
| **Full-layer KV injection** | **0.621** | **0.642 (≈4.8× baseline)** |
| Sparse injection (8/4 layers) | ≈ baseline | 0.279 / 0.183 (linear layers partially compensate) |

Findings verified on both scales: (i) mechanism works — token-level KV injection changes generation and lifts scores; (ii) equivalence — injection ≈ text prefix when keys match (often slightly better); (iii) layer coverage is mandatory over the full-attention set; (iv) transferable across scales/architectures (0.6B dense → 27B hybrid), and linear-attention layers partially compensate sparse injection.

## Repository layout

```
├── paper/paper_zh_preprint.pdf   # 论文全文（中文预印本）
├── code/
│   ├── qwen06b/                  # 0.6B 全注意力基座复现（论文主体）
│   │   ├── memlib.py             # 核心库：token-wise KV 编译/注入/解码
│   │   ├── run_arms3.py          # 迭代2b 六臂主实验（表2 数据源）
│   │   ├── run_arms.py           # Pilot1 池化消融（表3）
│   │   ├── run_retrieval.py      # 3主题检索实验（表4/5）
│   │   └── content_*.py          # 评测内容（判分锚点）
│   └── qwen27b/                  # Qwen3.8-27B 混合架构跨规模验证
│       ├── memlib_qwen35.py      # 16 full_attention 层注入（bf16, DCU）
│       ├── run_qwen38_validate.py# 六臂验证编排
│       └── smoke_qwen38.py       # 冒烟/适配探针
├── results/                      # 全部逐题原始数据 JSON
│   ├── pilot1.json / iter2b.json / retrieval1.json   # 0.6B
│   └── qwen38_v1.json            # 27B
└── docs/REPRODUCE.md             # 环境与逐实验复现指南
```

## Reproduction (0.6B, paper main body)

Requirements: Python 3.9 + torch (CUDA) + transformers 4.57.6 (tokenizers≥0.22), Qwen3-0.6B weights, 8 GB GPU.

```bash
# env: make sure tokenizers>=0.22 is importable (see docs/REPRODUCE.md)
python code/qwen06b/run_arms3.py --tag iter2b      # Table 2 (main result)
python code/qwen06b/run_arms.py  --layers 8 14 24 --max_new 48 --limit 20 --tag pilot1   # Table 3
python code/qwen06b/run_retrieval.py --tag retrieval1                                    # Table 4/5
```

Expected means: `plain=0.079  prefix=0.571  doc_allL=0.621  ans_allL=0.571  L24=0.079  L14_24=0.062`.

27B reproduction (hybrid architecture, bf16, Hygon DCU 64 GiB): see `code/qwen27b/` and `docs/REPRODUCE.md` §Qwen3.8-27B.

## Citation

```bibtex
@misc{yang2026memoryinjection,
  title={Full-Layer Memory Injection via Compile-Time Key-Value Caches: An External-Knowledge Mechanism for Small-Parameter Language Models},
  author={Yang, Donghui},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22305403},
  url={https://zenodo.org/records/22305403}
}
```
