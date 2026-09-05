# 论文复现指南（REPRODUCE）

**论文**：基于编译期键值缓存的全层记忆注入方法：面向小参数语言模型的知识外置机制（计算机学报样式）
**适用范围**：本机上完整复刻论文表 1–表 5 的全部实验数据。
**重要**：本文件仅记录现有文件位置与命令，**不移动、不重命名任何文件**。

---

## 0. 产物依赖关系总览（脚本 → 结果 → 论文位置）

| 论文位置 | 实验 | 运行脚本 | 产物 JSON（唯一数据源） |
|---|---|---|---|
| 表 2（主结果） | 迭代 2b：token-wise 全层 KV 注入 | `H:\deepseek\pilot\run_arms3.py` | `H:\deepseek\pilot\results\iter2b.json` |
| 表 3（池化消融） | Pilot 1：均值池化向量注入 | `H:\deepseek\pilot\run_arms.py` | `H:\deepseek\pilot\results\pilot1.json` |
| 表 4/5（检索） | 3 主题检索命中率 + top-k 注入 | `H:\deepseek\pilot\run_retrieval.py` | `H:\deepseek\pilot\results\retrieval1.json` |
| 表 1（平台） | 基座实测（非实验数据） | `H:\deepseek\qwen_smoke.py` / `H:\deepseek\qwen_fwd_speed.py` | 控制台输出 |
| 正文数值 | — | 从上述 JSON 读取 | — |

> 注意：`results/*.json` 是**输出产物**，复现时会被重新生成；`iter2b.json` 不是脚本输入。

---

## 1. 环境（必须一致，否则不可复刻）

| 项 | 现值（2026-09 实测） |
|---|---|
| Python | `E:\ProgramData\Anaconda3\python.exe`，3.9.13 |
| PyTorch | 2.7.1+cu118，CUDA 可用 |
| GPU | NVIDIA GeForce GTX 1070 Ti，8.59 GB（Pascal sm_61） |
| 精度 | **全程 fp32**（Pascal 无 bf16 硬件，fp16 算力仅 fp32 的 1/32） |
| 局部依赖 | `H:\deepseek\pyext`（**必须**设为 PYTHONPATH，见下） |

### 1.1 为什么需要 `H:\deepseek\pyext`（局部依赖，不动全局环境）

base 环境 transformers 4.57.6 与自带包版本冲突（tokenizers 0.13.3 过旧等）。已把兼容版本装入工作区目录，运行时通过 PYTHONPATH 优先加载。已验证版本：

- tokenizers 0.22.0（目录内另有 0.22.2 的 dist-info 残留，无碍）
- huggingface_hub 0.34.0（目录内另有 1.8.0 的 dist-info 残留，无碍）
- safetensors 0.4.5
- python-docx 1.1.2 + lxml 5.3.0（仅文档生成用，复现实验不需要）
- transformers 4.57.6（base 环境自带，须能随上述包正常 import）

### 1.2 基座模型（本地，离线）

```
H:\Qwen3-0.6B\Qwen\Qwen3-0.6B\    ← 目录（modelscope 下载，含符号链接结构）
  ├─ config.json / model.safetensors(1.43GB) / tokenizer.json / vocab.json / merges.txt ...
```
脚本内硬编码 `MODEL_DIR = r"H:/Qwen3-0.6B/Qwen/Qwen3-0.6B"`（见 `memlib.py`）。改动模型路径需同步改 `memlib.py`。

---

## 2. 逐实验复现命令

PowerShell 下，每个实验前先设：

```powershell
$env:PYTHONPATH = "H:\deepseek\pyext"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
```

### 2.1 表 2 —— 迭代 2b（关键实验）

```powershell
python H:\deepseek\pilot\run_arms3.py --tag iter2b
```
- 参数默认：`--max_new 24`，注入层 = 全部 28 层，20 题（主题 A）
- 输出：`H:\deepseek\pilot\results\iter2b.json`；控制台打印各臂 mean / full_hit
- 耗时：约 6–8 分钟（含模型加载）

### 2.2 表 3 —— Pilot 1（池化向量）

```powershell
python H:\deepseek\pilot\run_arms.py --layers 8 14 24 --max_new 48 --limit 20 --tag pilot1
```
- 输出：`H:\deepseek\pilot\results\pilot1.json`；约 10 分钟

### 2.3 表 4/5 —— 检索实验（需 3 主题内容）

```powershell
python H:\deepseek\pilot\run_retrieval.py --tag retrieval1
```
- 依赖 `content_quant_svc.py`（主题 A）+ `content_extra.py`（主题 B/C）
- 输出：`H:\deepseek\pilot\results\retrieval1.json`；约 4–6 分钟

### 2.4 表 1 —— 平台实测（不属实验数据，供核对）

```powershell
python H:\deepseek\qwen_smoke.py        # 596M / 2.38GB / 6.75 tok/s
python H:\deepseek\qwen_fwd_speed.py    # 预填充 2813/2240/1939 tok/s (L=512/1024/2048)
```

---

## 3. 复现校验（判定一致的标准）

复现后运行比对（或人工核对控制台 `=== 汇总 ===` 表）：

| 臂 | 期望 mean | 期望 full_hit |
|---|---|---|
| arm0_plain | 0.079 | 0/20 |
| arm1_textprefix | 0.571 | 3/20 |
| arm2_doc_allL | **0.621** | 5/20 |
| arm2_ans_allL | 0.571 | 5/20 |
| arm2_ans_L24 | 0.079 | 0/20 |
| arm2_ans_L14_L24 | 0.062 | 0/20 |

检索实验：最佳命中率 L24·mean_last4 = 0.595（三选一随机基线 0.333）；注入臂 own 0.729 / all 0.698 / wrong 0.068 / plain 0.078（16 题）。

生成采用 fp32 贪心 argmax，同一 CUDA/权重下应逐位一致；跨机器若 CUDA 版本不同，存在极小浮点差异可能，但 argmax 序列通常不变，mean 偏差应 <0.01。

---

## 4. 关键脚本内部依赖（运行 run_arms3.py 时实际加载的文件）

| 文件 | 角色 | 位置 |
|---|---|---|
| `run_arms3.py` | 编排：编译 banks → 6 臂解码 → 判分 → 存 JSON | `H:\deepseek\pilot\` |
| `memlib.py` | 机制库：load_model / compile_token_rows / MemAttention / decode_loop（**含模型路径硬编码**） | 同上 |
| `content_quant_svc.py` | 主题 A：doc + 20 问答（判分锚点） | 同上 |
| `content_extra.py` | 主题 B/C（仅检索实验需要） | 同上 |

辅助（非复现必需）：`analyze.py`、`dump_outs.py`（读 JSON 出逐题明细）；`run_arms2.py`、`sanity*.py`、`debug*.py` 均为开发过程残留，已被 run_arms3 取代，可忽略。

---

## 5. 已知注意点（复现踩坑记录）

1. **必须设 PYTHONPATH**：不设则 tokenizers 0.13.3 导致 transformers import 失败。
2. **离线变量**：建议设 TRANSFORMERS_OFFLINE=1，避免 huggingface 联网检查。
3. **显存**：3 主题检索实验需约 6 GB 峰值（多 bank 同时驻留 GPU 曾 OOM，已修复为 bank 存 CPU、attach 时搬 GPU——该修复在 `memlib.py` 内，勿回退为全 GPU 常驻）。
4. **Word 占用 docx 会锁文件**：重跑 `docs\build_cjc_paper.py` 前先关闭已打开的论文文档。
5. **PowerShell 管道限制**：本机若在受限 shell 中运行，`python ... | Select-String` 可能被拒；直接裸跑 python 即可（输出自动截尾）。
6. **模型加载**：冷启动约 108 s，热约 2 s（OS 页缓存）。
7. 首步前向含 CUDA 预热（L=128 实测 1.41 s 属预热），预热后 L=512 前向 0.18 s。

---

## 6. 论文文档的再生成（数据不变，仅排版）

```powershell
python H:\deepseek\pilot\docs\make_fig1.py          # 重绘 图1 → docs\fig1_arch.png
$env:PYTHONPATH = "H:\deepseek\pyext"
python H:\deepseek\pilot\docs\build_cjc_paper.py    # 重新生成论文 docx
python H:\deepseek\pilot\gen_report.py              # 重新生成研究 Web 文档
```

输出：`docs\基于编译期键值缓存的全层记忆注入（计算机学报样式）.docx`、`docs\qwen3_memory_kv_research.html`。

---

*本文件不改变任何现有文件的位置；删除本文件不影响任何实验复现。*
