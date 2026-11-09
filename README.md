# Legal Language Models from Scratch

Small Transformer language models (54–63M parameters) **pretrained from scratch on Indian legal text**, with a custom byte-level BPE tokenizer and four different architectures. The models are then adapted to six downstream legal NLP tasks: named-entity recognition, rhetorical-role labelling, judgment prediction, statute identification, abstractive summarization and prior-case retrieval.

> Capstone project for **CS781 (Legal Language Modelling Challenge), IIT Kanpur**.
> Author: **Suraj Singh Raman** (261110067), M.Tech CSE.
> All seven phases were scored on the course's Codabench leaderboard.

---

## Highlights

- **Everything from scratch.** This covers the tokenizer, attention, positional encodings, normalisation, KV cache, generation loop and every adaptation method: LoRA, LoRMA, rsLoRA, IA³, Houlsby adapters, prefix tuning, QLoRA with a hand-written NF4 quantizer, and knowledge distillation. PyTorch supplies only tensors, autograd, `nn.Linear`/`nn.Embedding` and optimizers. There is no HuggingFace `transformers`, `peft` or `bitsandbytes`.
- **Four base architectures from two paradigms.** A baseline decoder (NoPE), a hybrid local/global attention + GQA decoder, an ALiBi + Multi-head Latent Attention decoder, and a bidirectional encoder-decoder with an auxiliary MLM objective.
- **30 fine-tuning runs across six tasks.** All of them are kept with their full Kaggle outputs: training logs, dev scores and the validated submission.
- **Notable findings:**
  - **Distillation:** a 7-block, 36.6M student **beat its larger teacher** in Phase 4 (0.608 vs 0.542 macro-F1) and in Phase 5 (0.449 vs 0.420), and matched its teacher in Phases 2, 6 and 7.
  - **Decoding:** greedy decoding silently stopped early in summarization. A minimum-length constraint raised ROUGE-1 from 0.20 to 0.59.
  - **Retrieval:** adaptation budgets from 14.6k to 55M trainable parameters all converge to about 300 dense hits@7. With 827 training queries, supervision rather than capacity is the bottleneck.
- **Compute:** single Kaggle T4 GPUs.

---

## Repository layout

```
phase 1/ … phase 7/            one folder per phase
  decoder 01/                  runs on the Dec-01 backbone
  decoder 02/                  runs on Dec-02
  decoder 03/                  runs on Dec-03
  enc dec/                     runs on the encoder-decoder (its encoder in Phases 2–5, 7)
  kd student/                  knowledge-distillation student (7-block Dec-03)
phase 1/tokenizer/             byte-level BPE tokenizer (code + trained vocab/merges)
```

Every run folder contains the **notebook with all its outputs** (as executed on Kaggle) and the **submission zip** uploaded to Codabench, which has `prediction.json` at its root.

Zips are named `submission_p<phase>_<model>_<method>.zip`, e.g. `submission_p4_dec02_lorma.zip`. The model codes are `dec01`, `dec02`, `dec03`, `encdec` and `kd_student`.

- **Extra Phase-2 folders.** `after kd code*/` holds a resubmission (`*_rerun.zip`) made after the KD export cell was added (same model and method), and `enc dec/method_full/` holds the full-fine-tuning submission.
- **Model weights are not included.** That is a course rule, and they are 200+ MB each. Running the Phase-1 notebooks reproduces them; see [Reproducing](#reproducing-the-experiments).
- **Metric conventions.** Unless stated otherwise, scores below are on each phase's **dev split**, untuned, as logged in the notebooks. Codabench test scores come from the leaderboard.

---

## Phase 1: Pretraining (causal language modelling)

**Task:** given the first 512 words of a judgment, generate the next 128 words. Scored by BERTScore F1, with ROUGE-L and BLEU as secondary metrics.

### Tokenizer (`phase 1/tokenizer/`)

| Choice | Value | Reason |
|---|---|---|
| Type | byte-level BPE (256-byte base alphabet) | no unknown tokens; lossless on Hindi/Sanskrit snippets and symbols |
| Vocabulary | 32,000 | small embedding table for 50–300M models; IDs fit in `uint16` |
| Normalisation | none inside the tokenizer | Phase-2 NER is scored on exact character offsets |
| Digits | split into single digits | clean boundaries for section numbers, years and case numbers |
| Offsets | `encode(text, return_offsets=True)` | maps tokens back to characters for span tasks |
| Guard | pre-tokens capped at 1,000 bytes | BPE is O(L²) per pre-token; long runs of dashes or dots in court PDFs otherwise hang encoding |

The tokenizer averages about 1.25–1.42 tokens per word. `capstone_tokenizer_large` is the one used by every model.

**Training data:** 150M tokens from the four official corpora (acts 20M, high courts 65M, Supreme Court 25M, tribunals 40M). Sources are sampled in proportion to size.

### The four base models

| | **Dec-01** | **Dec-02** | **Dec-03** | **Enc-Dec** |
|---|---|---|---|---|
| Paradigm | decoder-only | decoder-only | decoder-only | encoder-decoder |
| Layers × width | 12 × 512 | 12 × 512 | 13 × 512 | 6 enc + 6 dec × 512 |
| Attention | full MHA (8 heads) | **GQA** (8 query / 2 KV heads) + **sliding window 256** on 9 layers; full attention on layers 3, 7, 11 | **MLA** (keys/values through a 128-d latent) | bidirectional self-attention (encoder); causal self + cross-attention (decoder) |
| Positions | **none (NoPE)**, see note | RoPE | **ALiBi** | RoPE |
| Block | pre-norm RMSNorm, SwiGLU | same | same | same |
| Extra objective | – | – | – | masked-LM loss on the encoder |
| Parameters | 54.9M | 54.9M | 53.9M | 61.5M (encoder 35.7M) |
| Steps / time | 5,750 best (log to 5,950 of 6,000), ~7 h | 12,749, 6.9 h | 4,591 (time budget), 8.4 h | 15,489, 6.8 h |
| Tokens seen | ≈195M (fp32 by mistake, 7.5k tok/s) | 418M (fp16, 18k tok/s) | ≈600M (fp16, 20k tok/s) | – (seq2seq + MLM) |
| **Best val loss** | **2.117** (ppl 8.30) | **1.944** (ppl 6.99) | **1.885** (ppl 6.58) | **1.959** seq2seq (ppl 7.09); MLM 1.41 |

> **The val losses are not a clean architecture comparison.** The decoders saw very different numbers of tokens (≈195M, 418M, ≈600M) because of different throughput and time budgets, and the encoder-decoder optimises a different objective. Dec-01 and Dec-02 process about the same number of tokens per step (≈32k), and at step 5,749 Dec-02's val loss is 2.11, about the same as Dec-01's final 2.117. At an equal token budget, then, the hybrid-attention + GQA design is roughly on par with full attention, at a much lower attention cost.

**Design rationale.**
- **Dec-02** reflects the structure of judgments. Most dependencies are local (phrasing, syntax), so most layers attend within a window. A few long-range lookups (party names, earlier citations) are handled by full-attention layers. GQA shrinks the KV cache.
- **Dec-03** swaps both the position mechanism (ALiBi, a linear distance penalty, i.e. an exponential recency prior inside the softmax) and the KV mechanism (MLA, a low-rank factorisation that caches a small latent instead of K and V).
- **Enc-Dec** gives later phases a natively bidirectional encoder, which suits tagging, classification and retrieval.

**Notebooks and submissions**

| Folder | Notebook | Purpose |
|---|---|---|
| `decoder 01/` | `caps-p1-dec-01-train.ipynb` | Training run (full loss curve). Its own submission had some very short outputs (min 1 word). |
| | `caps-p1-dec-01-inference-fixed.ipynb` | Reloads the **same checkpoint** (verifies val loss 2.1168 exactly), then regenerates the submission with fixed decoding (mean 127.1 words) → `submission_p1_dec01_inference_fixed.zip` |
| `decoder 02/` | `caps-p1-dec-02.ipynb` | Training + **three submissions from the same weights** with different decoding (`v1` greedy, `v2` T=0.7 / top-p 0.9, `v3` T=0.5 / top-p 0.9) |
| `decoder 03/` | `caps-p1-dec-03.ipynb` | Training + submission (greedy, repetition penalty 1.1) |
| `enc dec/` | `caps-p1-encdec.ipynb` | **Final encoder-decoder (v2)**: self-tests (KV cache, bidirectionality, MLM masking), training, decoding sweep, submission |
| | `caps-p1-encdec-causal-enc.ipynb` | **First version (v1)**: training run (10,971 steps, val 2.40) |
| | `caps-p1-encdec-causal-enc-inference.ipynb` | v1 inference with fixed generation code → `submission_p1_encdec_v1_causal.zip` |

**Two bugs found and fixed along the way** (documented honestly):
- **Dec-01 is NoPE.** A wiring bug meant RoPE was never applied, so Dec-01 trained with **no positional encoding**. It still learns word order, because the causal mask itself leaks position: token *t* attends to exactly *t* tokens. It is reported as a NoPE decoder. Consequence for later phases: Dec-01 cannot be made bidirectional, since the causal mask is its only source of order.
- **Enc-Dec v1 had a causal encoder.** An inherited default made the v1 encoder causal. v2 fixed this and also added an MLM objective on the encoder and longer windows (768 encoder / 256 decoder tokens, sized from the measured tokens-per-word). The two versions differ in several ways, so **v1 → v2 is a development history, not a controlled ablation.**

**Decoding:** generate at least 128 words. EOS is blocked until then, n-gram repetition is banned, and the prompt is left-truncated so its most recent text is kept. Every output is then cut to exactly 128 words. The submission is validated (IDs, length, no prompt leakage) before zipping. For Enc-Dec v2, a sweep on 113 held-out prompts gave ROUGE-L 0.297 (greedy, submitted) and 0.317 (beam-4).

---

## Phases 2–7: Downstream tasks

**Shared recipe:** Phase-1 backbone, then a task head, an adaptation method, and the metric.

**Shared conventions in every phase:**
- **Adapters start at identity.** The adapted model at step 0 equals the Phase-1 model, and a build-time assert checks this.
- **Unmasking.** A RoPE or ALiBi decoder is turned bidirectional (or only its top 4 blocks, `bidir_top`). The NoPE Dec-01 stays causal or uses the **echo** trick: the input is fed twice and representations are read from the second copy, so every token has seen the whole text.
- **KD student:** Dec-03 with every second block kept (blocks 0, 2, …, 12 → 7 blocks, 36.6M parameters), fully fine-tuned. It learns from a **teacher export file** (soft labels or scores) rather than from teacher weights, so teacher and student can be different architectures. Loss: `α·T²·KL(p_teacher^T ‖ p_student^T) + (1−α)·CE` with α = 0.5, T = 2. The T² factor keeps the soft-target gradient on the same scale as T grows.

### Phase 2: Legal NER
- **Task:** character-span entities with 12 labels (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO). Strict macro-F1.
- **Method:** BIO tagging (25 tags), a linear head per token, class-weighted cross-entropy, and sliding windows (800 tokens, stride 400) whose predictions are merged back to character offsets. Train on fold 1, dev on fold 2.

| Folder | Backbone | Adaptation | Trainable | Dev macro-F1 |
|---|---|---|---|---|
| `enc dec` | Enc-Dec encoder | full fine-tuning | 35.7M | **0.365** |
| `decoder 01` | Dec-01 (echo) | LoRA | – | 0.321 |
| `kd student` | 7-block Dec-03 | **KD** from Dec-01 + LoRA (teacher 0.322) | 36.6M | 0.315 (98 % of teacher) |
| `decoder 03` | Dec-03 (`bidir_top`) | partial FT (top 4 of 13 blocks) | 11.6M | 0.294 |
| `decoder 02` | Dec-02 (bidirectional) | **LoRMA** | 2.0M | 0.270 |

### Phase 3: Rhetorical role labelling
- **Task:** 13 sentence-level roles (Facts, Issue, Arguments, Ratio, …) on two domains (CL, IT). Macro-F1. The dev score below is the mean of the per-domain macro-F1s.
- **Method:** windows of consecutive sentences (up to 800 tokens) are encoded together, and each sentence's tokens are pooled into one vector, which is then classified.

| Folder | Backbone | Adaptation | Trainable | Dev score |
|---|---|---|---|---|
| `decoder 03` | Dec-03 | **rsLoRA**, r = 64 (scale α/√r keeps update size flat as rank grows) | 8.1M | **0.480** |
| `decoder 02` | Dec-02 | **LoRMA** | 2.0M | 0.478 |
| `decoder 01` | Dec-01 (echo) | LoRA | – | 0.467 |
| `enc dec` | Enc-Dec encoder | **IA³** | 21k | 0.402 |

*The knowledge-distillation run for Phase 3 was not completed within the phase window.*

### Phase 4: Court judgment prediction (CJPE)
- **Task:** binary outcome for a whole judgment. Macro-F1.
- **Method:** a hierarchical model. The last 4 chunks × 512 tokens of a case are each mean-pooled, aggregated with additive attention, then classified.

| Folder | Backbone | Adaptation | Trainable | Dev macro-F1 |
|---|---|---|---|---|
| `decoder 01` | Dec-01 | LoRA | – | **0.675** (0.709 with tuned threshold) |
| `enc dec` | Enc-Dec encoder | IA³ | 0.28M | 0.644 |
| `kd student` | 7-block Dec-03 | **KD** from Dec-02 + LoRMA | 36.8M | 0.608 (**retention 1.12: beats its teacher**) |
| `decoder 02` | Dec-02 | **LoRMA** (KD teacher) | 2.2M | 0.542 |
| `decoder 03` | Dec-03 | rsLoRA, r = 64 | 8.4M | 0.364, **failed run** (see below) |

**Failed run (Dec-03 + rsLoRA).**
- The training loss stayed at ≈ ln 2 = 0.693, the loss of always predicting 50/50, for about 5,200 steps.
- It then became NaN, after which every prediction was a single class (macro-F1 0.333).
- The submission uses the epoch-1 checkpoint (0.364, near chance).
- The likely cause is large adapter updates (γ = 2 on every MLA and FFN linear, lr 5e-4) combined with fp16 overflow under unmasked ALiBi on long chunks.
- The same method trained normally on Dec-03 in Phase 3, and a Dec-03-based student (full FT) trained normally on this exact task. So the failure is specific to this recipe, not to the model.

### Phase 5: Legal statute identification
- **Task:** multi-label, 100 statutes, about 4 per case. Macro-F1.
- **Data finding:** explicit statute mentions are masked in the text, but the placeholders mark *where* statutes were cited. An **evidence view** (4 × 1024 tokens) keeps the text around those placeholders.
- **Method:** label-attention head, BCE with `pos_weight`, per-label thresholds tuned on half of dev. Held-out scores are on the other half.

| Folder | Backbone | Adaptation | Trainable | Dev macro-F1 (held-out) |
|---|---|---|---|---|
| `kd student` | 7-block Dec-03 | **KD** from Dec-02 + LoRMA | 36.9M | **0.449 (0.438)**, beats its teacher |
| `decoder 03` | Dec-03 (`bidir_top`) | partial FT, layer-wise LR decay 0.8 | 11.9M | 0.424 (0.414) |
| `decoder 02` | Dec-02 | **LoRMA** (KD teacher) | 2.3M | 0.420 (0.410) |
| `decoder 01` | Dec-01 (causal) | full fine-tuning | 55.3M | 0.409 (0.408) |
| `enc dec` | Enc-Dec encoder | Houlsby adapters | 1.2M | 0.404 (0.385) |

**Why the student can beat the teacher.** Both times the teacher was Dec-02 + LoRMA, a low-rank adapter on a frozen backbone. The fully fine-tuned student uses the teacher's soft labels as a regulariser, which carries graded beliefs and label co-occurrence, and its higher capacity lets it go beyond the teacher.

### Phase 6: Abstractive summarization
- **Task:** summarise a full judgment. BERTScore F1 primary, ROUGE secondary.
- **Data findings:** only 1 % of document-summary pairs fit in 1024 tokens; references are highly extractive (fragment coverage 0.96); summary content follows document order.
- **Method:** each document is cut into 512-token chunks. Each summary sentence is aligned to its source chunk to build chunk-level training targets. At inference, chunks are summarised and the results concatenated in document order.
- **Decoding finding:** greedy decoding stopped too early (≈80 % of chunks produced nothing), because p(EOS) only has to beat every single word token. A minimum-length constraint fixed it: ROUGE-1 went from 0.20 to 0.59.

| Folder | Backbone | Adaptation | Trainable | R-1 | R-2 | R-L | BERTScore |
|---|---|---|---|---|---|---|---|
| `enc dec` | Enc-Dec | Houlsby adapters | 2.0M | **0.596** | **0.336** | 0.309 | **0.856** |
| `kd student` | 7-block Dec-03 | **KD** from Enc-Dec (sequence-level) | 36.6M | 0.595 | 0.334 | **0.311** | 0.854 |
| `decoder 02` | Dec-02 | **LoRMA** | 2.0M | 0.569 | 0.304 | 0.287 | 0.851 |
| `decoder 01` | Dec-01 | prefix tuning (64 virtual tokens) | 1.7M | 0.517 | 0.246 | 0.252 | 0.842 |
| – | TF-IDF centroid (extractive) | – | – | 0.575 | 0.319 | 0.301 | 0.853 |
| – | lead baseline | – | – | 0.522 | 0.251 | 0.244 | 0.832 |

*Validation split of 200 documents.* The KD student keeps the teacher's quality with **42 % fewer parameters**. Dec-03 is used in this phase only as the KD student.

### Phase 7: Prior case retrieval
- **Task:** for each query judgment, return exactly 7 cited candidates. Micro-F1@7.
- Since every query gets 7 predictions, F1@7 = 2H / (7N + G), where H = total hits, N = queries, G = gold pairs. The metric is therefore just "maximise total hits".
- **Method:**
  - A tuned **BM25** baseline: uni+bigrams, k1 = 8, b = 1. The very large k1 makes it behave like length-normalised TF-IDF without saturation.
  - A **dense bi-encoder** trained with multi-positive InfoNCE, BM25 hard negatives and false-negative masking.
  - **Fusion:** z-normalised dense and BM25 scores, mixed with weight α tuned on dev.
  - Queries are removed from their own candidate pools.

| Folder | Run | Trainable | Dev dense H@7 | Dev fused F1@7 |
|---|---|---|---|---|
| `decoder 02` | Dec-02 + **LoRMA** | 2.2M | 299 | **0.450** |
| `decoder 03` | Dec-03 + partial FT (`bidir_top`) | 11.8M | 301 (313 best mean-pool) | 0.448 |
| `kd student` | **KD**, single teacher (Dec-02 + LoRMA) | 36.8M | 299 | 0.448 |
| `decoder 01` | Dec-01 + full fine-tuning | 55.2M | 303 | 0.447 |
| `decoder 01` | Dec-01 + **QLoRA** (own NF4, 4.13 bits/weight) | 2.2M | 302 | 0.447 |
| `kd student` | **KD**, three teachers (Dec-02, Dec-01, Dec-03) | 36.8M | 275 | 0.446 |
| `enc dec` | Enc-Dec encoder + IA³ | 0.28M | 281 | 0.445 |
| – | BM25 alone | – | – | 0.439 |

**Findings:**
- **Capacity isn't the bottleneck.** Budgets from 14.6k to 55M trainable parameters all converge to about 300 dense hits. With 827 training queries, supervision rather than capacity is the limit.
- **QLoRA:** 4-bit NF4 weights cost nothing for retrieval. The result matches full fine-tuning with 3.9 % trainable parameters.
- **Multi-teacher KD did not beat the best single teacher.** The teachers disagree (top-7 overlap 0.47–0.56), but they are of unequal quality, so averaging adds their errors.
- **Pooling:** `[BOS]`/CLS pooling collapses on an encoder never trained to summarise into that token. Mean-centering embeddings helps, which is evidence of anisotropy.

---

## Reproducing the experiments

All notebooks are self-contained and were run on **Kaggle** (T4 GPU, internet on).

1. **Requirements:** Python 3, `torch`, `numpy`, `huggingface_hub`, `tqdm` (plus `ijson` for Phase-1 data streaming). No other ML libraries.
2. **Data:** the course dataset `Exploration-Lab/CS-781-Capstone` on HuggingFace. Add your HF token as a Kaggle secret named `HF`.
3. **Tokenizer:** upload `phase 1/tokenizer/capstone_tokenizer_large/` as a Kaggle dataset. The notebooks expect it at `/kaggle/input/datasets/<user>/capstone-tokenizer-large`.
4. **Phase 1:** run each architecture notebook. Each one writes `runs/<name>/best.pt`.
5. **Phases 2–7:** upload the `best.pt` files as Kaggle datasets (`caps-dec-01-best`, `caps-dec-02-best`, `caps-dec-03-best`, `caps-encdec-01-best`). Then, in each notebook:
   - Edit **Cell 2 only** (the RUN CONFIG: paths, method, hyperparameters).
   - Run with `SMOKE_TEST = True` to check the pipeline.
   - Set `SMOKE_TEST = False` and run with **Save Version**.
   - Outputs go to `runs/<RUN_NAME>/`: `run_log.json`, the `submission*.zip`, and, for teacher runs, the KD export file.
6. **KD runs:** need the teacher run's export file, enabled with `EXPORT_SOFT_LABELS`/`EXPORT_TEACHER = True` in the teacher notebook, then uploaded as a dataset and set in Cell 2 of the student notebook.

**Reproducibility:** fixed seed 1337, a deterministic dev split (Phase 6 asserts a split checksum), and dev-only model selection. Test labels are never used.

---

## Acknowledgements

- Course: CS781, IIT Kanpur. Dataset and evaluation by the course staff (Exploration-Lab).
- Methods are implemented from their papers:
  - LoRA (Hu et al. 2021), LoRMA (Bihany et al. 2025), rsLoRA (Kalajdzievski 2023), IA³ (Liu et al. 2022), adapters (Houlsby et al. 2019), prefix tuning (Li & Liang 2021), QLoRA (Dettmers et al. 2023);
  - ALiBi (Press et al. 2022), MLA (DeepSeek-V2, 2024), GQA (Ainslie et al. 2023);
  - knowledge distillation (Hinton et al. 2015).
