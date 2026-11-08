# Legal Language Models from Scratch

Small Transformer language models (54–63M parameters) **pretrained from scratch on Indian legal text**, with a custom byte-level BPE tokenizer and four different architectures. The models are then adapted to six downstream legal NLP tasks: named-entity recognition, rhetorical-role labelling, judgment prediction, statute identification, abstractive summarization and prior-case retrieval.

> Capstone project for **CS781 (Legal Language Modelling Challenge), IIT Kanpur**.
> Author: **Suraj Singh Raman** (261110067), M.Tech CSE.
> All seven phases were scored on the course's Codabench leaderboard.

---

## Highlights

- **Everything from scratch:** the tokenizer, attention, positional encodings, normalisation, KV cache, generation loop and all adaptation methods (LoRA, LoRMA, rsLoRA, IA³, adapters, prefix tuning, QLoRA with a hand-written NF4 quantizer, knowledge distillation). PyTorch supplies only tensors, autograd, `nn.Linear`/`nn.Embedding` and optimizers. No HuggingFace `transformers`, `peft` or `bitsandbytes`.
- **Four architectures from two paradigms:**
  - a NoPE decoder;
  - a hybrid local/global attention + GQA decoder;
  - an ALiBi + Multi-head Latent Attention decoder;
  - a bidirectional encoder-decoder.
- **Six downstream tasks, 30+ fine-tuning runs.** Every phase compares several adaptation techniques on the same Phase-1 backbones, with LoRMA and knowledge distillation in every phase.
- **Compute:** single Kaggle T4 GPUs.

---

## Repository layout

```
phase 1/                 Pretraining: tokenizer + 4 base language models
  tokenizer/             byte-level BPE tokenizer (code + trained vocab/merges)
  decoder 01/            Dec-01  (baseline decoder, NoPE)
  decoder 02/            Dec-02  (hybrid local/global attention + GQA)
  decoder 03/            Dec-03  (ALiBi + MLA)
  enc_dec_01/            Enc-Dec (bidirectional encoder-decoder; causal_encoder/ = ablation)
phase 2/ … phase 7/      Downstream tasks
  model 1/               runs on Dec-01
  model 2/               runs on Dec-02
  model 3/               runs on Dec-03
  model 4/               knowledge-distillation student (7-block Dec-03)
  model 5/               runs on the Enc-Dec encoder
```

Each run folder holds its notebook (where available), `prediction.json` and the `submission.zip` that was uploaded to Codabench. A file name ending in ` (1).ipynb` is the later revision of the same run; it adds the KD teacher-export cell.

**Model weights are not included** (course rule, and they are 200+ MB each). Running the Phase-1 notebooks reproduces them; see [Reproducing](#reproducing-the-experiments).

---

## Phase 1: Pretraining (causal language modelling)

**Task:** given the first 512 words of a judgment, generate the next 128 words. Scored by BERTScore F1, with ROUGE-L and BLEU as secondary metrics.

### Tokenizer (`phase 1/tokenizer/`)

| Choice | Value | Reason |
|---|---|---|
| Type | byte-level BPE (256-byte base alphabet) | No unknown tokens; lossless on Hindi/Sanskrit snippets and symbols |
| Vocabulary | 32,000 | Small embedding table for 50–300M models; IDs fit in `uint16` |
| Normalisation | none inside the tokenizer | Phase-2 NER is scored on exact character offsets |
| Digits | split into single digits | Clean boundaries for section numbers, years and case numbers |
| Offsets | `encode(text, return_offsets=True)` | Maps tokens back to characters for span tasks |
| Guard | pre-tokens capped at 1,000 bytes | BPE is O(L²) per pre-token; long runs of dashes or dots in court PDFs otherwise hang encoding |

The tokenizer averages about 1.42 tokens per word (1.30 for Supreme Court text, 1.47 for tribunals). `capstone_tokenizer_large` is the one used by every model.

### The four base models

| | **Dec-01** | **Dec-02** | **Dec-03** | **Enc-Dec** |
|---|---|---|---|---|
| Paradigm | decoder-only | decoder-only | decoder-only | encoder-decoder |
| Layers × width | 12 × 512 | 12 × 512 | 13 × 512 | 6 enc + 6 dec × 512 |
| Attention | full MHA (8 heads) | **GQA** (8 query / 2 KV heads) + **sliding window 256** on 9 layers, full attention on layers 3, 7, 11 | **MLA** (keys/values through a 128-d latent) | bidirectional self-attention (encoder); causal self + cross-attention (decoder) |
| Positions | **none (NoPE)**, see note | RoPE | **ALiBi** | RoPE |
| Block | pre-norm RMSNorm, SwiGLU | same | same | same |
| Embeddings | tied input/output | tied | tied | tied across encoder, decoder and head |
| Context | 1024 | 1024 | 1024 | encoder ≈768, decoder 256 |
| Extra objective | – | – | – | masked-LM loss on the encoder |
| Parameters | 54.9M | ≈54.9M | 53.9M | 61.2M (encoder 35.7M) |

**Design rationale.**
- **Dec-02** reflects the structure of judgments. Most dependencies are local (phrasing, syntax), so most layers attend within a window. A few long-range lookups (party names, earlier citations) are handled by full-attention layers. GQA shrinks the KV cache.
- **Dec-03** swaps both the position mechanism (ALiBi, a linear distance penalty and an exponential recency prior inside the softmax) and the KV mechanism (MLA, a low-rank factorisation that caches a small latent instead of K and V).
- **Enc-Dec** gives later phases a natively bidirectional encoder, which suits tagging, classification and retrieval.

**Note on Dec-01.** A wiring bug meant RoPE was never applied, so Dec-01 trained with **no positional encoding**. It still learns word order, because the causal mask itself leaks position: token *t* attends to exactly *t* tokens. The model is reported honestly as a NoPE decoder. A side effect: Dec-01 cannot be made bidirectional in later phases, since the causal mask is its only source of order.

**Ablation.** `enc_dec_01/causal_encoder/` holds an earlier encoder-decoder whose encoder was accidentally causal. It is kept as a controlled comparison against the bidirectional version.

**Training recipe:** AdamW (β = 0.9, 0.95), cosine schedule from 6e-4 to 6e-5 with warmup, weight decay 0.1 (not on norms or embeddings), fp16 autocast with a gradient scaler on T4, init N(0, 0.02) with residual projections scaled by 1/√(2L).

**Decoding:** generate at least 128 words. EOS is blocked until then, n-gram repetition is banned, and the prompt is left-truncated so its most recent text is kept. Every output is then cut to exactly 128 words. The submission is validated (IDs, length, no prompt leakage) before zipping.

---

## Phases 2–7: Downstream tasks

**Shared recipe:** Phase-1 backbone, then a task head, an adaptation method, and the metric.

**Shared conventions in every phase:**
- **Adapters start at identity.** The adapted model at step 0 equals the Phase-1 model, and a build-time assert checks this.
- **Unmasking.** A RoPE or ALiBi decoder is turned bidirectional (or only its top blocks, `bidir_top`). The NoPE Dec-01 stays causal, or uses the **echo** trick: the input is fed twice and representations are read from the second copy, so every token has seen the whole text.
- **KD student:** Dec-03 with every second block kept (7 blocks, 36.6M parameters), fully fine-tuned. It learns from a **teacher export file** (soft labels or scores) rather than from teacher weights, so teacher and student can have different architectures. Loss: `α·T²·KL(p_teacher^T ‖ p_student^T) + (1−α)·CE`. The T² factor keeps the soft-target gradient on the same scale as T grows.

### Phase 2: Legal NER
- **Task:** character-span entities with 12 labels (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO). Strict macro-F1.
- **Method:** BIO tagging (25 tags), one linear layer per token, class-weighted cross-entropy, and sliding windows with overlap whose predictions are merged back to character offsets.

| Folder | Backbone | Adaptation |
|---|---|---|
| model 1 | Dec-01 (echo) | LoRA |
| model 2 | Dec-02 (bidirectional) | **LoRMA** |
| model 3 | Dec-03 (`bidir_top`) | partial fine-tuning (top 4 blocks) |
| model 4 | Dec-03 student | **knowledge distillation** |
| model 5 | Enc-Dec encoder | full fine-tuning (also norm-only via `METHOD`) |

### Phase 3: Rhetorical role labelling
- **Task:** 13 sentence-level roles (Facts, Issue, Arguments, Ratio, …). Macro-F1.
- **Method:** windows of consecutive sentences (up to 800 tokens) are encoded together, and each sentence's tokens are pooled into one vector, which is then classified. One forward pass labels every sentence in the window.

| Folder | Backbone | Adaptation |
|---|---|---|
| model 1 | Dec-01 (echo) | LoRA |
| model 2 | Dec-02 | **LoRMA** |
| model 3 | Dec-03 | **rsLoRA**, r = 64 (scale α/√r instead of α/r keeps the update size flat as rank grows) |
| model 5 | Enc-Dec encoder | **IA³** (14.6k trainable parameters) |

### Phase 4: Court judgment prediction (CJPE)
- **Task:** binary outcome for a whole judgment. Macro-F1.
- **Method:** a hierarchical model. Up to 4 chunks × 512 tokens (default: the last chunks of the case) are each mean-pooled into a vector, aggregated (`mean`, additive attention, or a small Transformer over chunks), then classified. The decision threshold is tuned on dev.

| Folder | Backbone | Adaptation |
|---|---|---|
| model 1 | Dec-01 | LoRA |
| model 2 | Dec-02 | **LoRMA** (KD teacher) |
| model 3 | Dec-03 | rsLoRA |
| model 4 | Dec-03 student | **KD** |
| model 5 | Enc-Dec encoder | IA³ |

### Phase 5: Legal statute identification
- **Task:** multi-label, 100 statutes, about 4 per case. Macro-F1.
- **Data finding:** explicit statute mentions are masked in the text, but the placeholders mark *where* statutes were cited. An **evidence view** keeps the text around those placeholders.
- **Method:** label-attention head, BCE with `pos_weight` and a prior-bias init, and per-label thresholds.

| Folder | Backbone | Adaptation |
|---|---|---|
| model 1 | Dec-01 (causal) | full fine-tuning |
| model 2 | Dec-02 | **LoRMA** (KD teacher) |
| model 3 | Dec-03 | partial fine-tuning with layer-wise LR decay (0.8 per block) |
| model 4 | Dec-03 student | **KD** (per-label soft targets) |
| model 5 | Enc-Dec encoder | Houlsby bottleneck adapters |

### Phase 6: Abstractive summarization
- **Task:** summarise a full judgment. BERTScore F1 primary, ROUGE secondary.
- **Data findings:** only 1 % of document-summary pairs fit in 1024 tokens; references are highly extractive (fragment coverage 0.96); summary content follows document order.
- **Method:** each document is cut into 512-token chunks. Each summary sentence is aligned to its source chunk to build chunk-level training targets. At inference, chunks are summarised and the results concatenated in document order.
- **Decoding finding:** greedy decoding stopped too early (about 80 % of chunks produced nothing), because p(EOS) only has to beat every single word token. A minimum-length constraint fixed it, raising ROUGE-1 from 0.20 to 0.59.

| Folder | Backbone | Adaptation | Trainable | R-1 | R-2 | R-L | BERTScore |
|---|---|---|---|---|---|---|---|
| model 5 | Enc-Dec | Houlsby adapters | 2.00M | **0.596** | **0.336** | 0.309 | **0.855** |
| model 4 | Dec-03 student (36.6M) | **KD** from model 5 | 36.6M | 0.595 | 0.334 | **0.311** | 0.854 |
| model 2 | Dec-02 | **LoRMA** | 1.97M | 0.569 | 0.304 | 0.287 | 0.851 |
| model 1 | Dec-01 | prefix tuning | 1.68M | 0.517 | 0.246 | 0.252 | 0.842 |
| – | TF-IDF centroid (extractive) | – | – | 0.575 | 0.319 | 0.301 | 0.853 |
| – | lead baseline | – | – | 0.522 | 0.251 | 0.244 | 0.832 |

*(Validation split of 200 documents.)* The KD student keeps the teacher's quality with **42 % fewer parameters**.

### Phase 7: Prior case retrieval
- **Task:** for each query judgment, return exactly 7 cited candidates. Micro-F1@7.
- Since every query gets 7 predictions, F1@7 = 2H / (7N + G), where H = total hits, N = queries, G = gold pairs. The metric is therefore just "maximise total hits".
- **Method:**
  - A tuned **BM25** baseline: uni+bigrams, k1 = 8, b = 1. The very large k1 makes it behave like length-normalised TF-IDF without saturation, which suits legal text.
  - A **dense bi-encoder** trained with multi-positive InfoNCE, BM25 hard negatives and false-negative masking.
  - **Fusion:** z-normalised dense and BM25 scores, mixed with weight α tuned on dev.
  - Queries are removed from their own candidate pools.

| Folder | Backbone | Adaptation | Trainable | Dev hits@7 (dense) | Dev fused F1@7 |
|---|---|---|---|---|---|
| model 2 | Dec-02 | **LoRMA** | 2.23M | 299 | **0.450** |
| model 3 | Dec-03 (`bidir_top`) | partial fine-tuning | 11.8M | **313** | 0.448 |
| model 1 | Dec-01 | full fine-tuning | 55.2M | 303 | 0.447 |
| model 5 | Enc-Dec encoder | IA³ | 0.28M | 300 | 0.445 |
| – | BM25 alone | – | – | – | 0.439 |

**Additional runs (notebooks to be added):**
- **Dec-01 + QLoRA** with an NF4 quantizer written from scratch (4.13 bits/weight, matching full fine-tuning at 3.9 % trainable parameters).
- **Single-teacher vs. multi-teacher KD**: the ensemble of unequal teachers did not beat the best single teacher.

**Findings:**
- Adaptation budgets from 14.6k to 55M trainable parameters all converge to about 300 dense hits. With 827 training queries, supervision rather than capacity is the bottleneck.
- `[BOS]`/CLS pooling collapses on an encoder that was never trained to summarise into that token.
- Mean-centering embeddings helps, which is evidence of anisotropy in the Phase-1 representations.

---

## Reproducing the experiments

All notebooks are self-contained and were run on **Kaggle** (T4 GPU, internet on).

1. **Requirements:** Python 3.10+, `torch`, `numpy`, `huggingface_hub`, `tqdm`. No other ML libraries.
2. **Data:** the course dataset `Exploration-Lab/CS-781-Capstone` on HuggingFace. Add your HF token as a Kaggle secret named `HF`; Cell 0 of every notebook logs in with it.
3. **Tokenizer:** upload `phase 1/tokenizer/capstone_tokenizer_large/` as a Kaggle dataset. The notebooks expect it at `/kaggle/input/datasets/<user>/capstone-tokenizer-large`.
4. **Phase 1:** run each architecture notebook. Each one writes `runs/<name>/best.pt`.
5. **Phases 2–7:** upload the `best.pt` files as Kaggle datasets (`caps-dec-01-best`, `caps-dec-02-best`, `caps-dec-03-best`, `caps-encdec-01-best`). Then, in each notebook:
   - Edit **Cell 2 only** (the RUN CONFIG: paths, method, hyperparameters).
   - Run once with `SMOKE_TEST = True` (a few hundred steps, no submission) to check the pipeline.
   - Set `SMOKE_TEST = False` and run with **Save Version**, so the outputs persist.
   - Outputs go to `runs/<RUN_NAME>/`: `run_log.json`, the Codabench `submission*.zip` (with `prediction.json` at its root), and, for teacher runs, the KD export file.
6. **KD runs:** need the teacher run's export file, uploaded as a dataset and set in Cell 2 (`TEACHER_EXPORT` / `TEACHER_PATHS`).

**Reproducibility:** fixed seed 1337, a deterministic dev split (Phase 6 asserts a split checksum), and dev-only model selection. Test labels are never used.

---

## Acknowledgements

- Course: CS781, IIT Kanpur. Dataset and evaluation by the course staff (Exploration-Lab).
- Methods are implemented from their papers:
  - LoRA (Hu et al. 2021), LoRMA (Bihany et al. 2025), rsLoRA (Kalajdzievski 2023), IA³ (Liu et al. 2022), adapters (Houlsby et al. 2019), prefix tuning (Li & Liang 2021), QLoRA (Dettmers et al. 2023);
  - ALiBi (Press et al. 2022), MLA (DeepSeek-V2, 2024), GQA (Ainslie et al. 2023);
  - knowledge distillation (Hinton et al. 2015).
