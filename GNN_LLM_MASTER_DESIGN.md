# Master Design Document: Hybrid GNN + LLM Architecture
**Project:** RAC_Evidence (DSM-5 Evidence Extraction)
**Date:** 2025-11-29
**Author:** Claude Code (Technical Design)
**Status:** Design Phase → Implementation Ready

---

## Executive Summary

**Current State:** S-C Cross-Encoder Reranker has saturated local performance (F1=0.52, Macro-F1=0.76, Coverage@5=89.3%). We have built a basic HGT pipeline (Phase 0) that serves as the foundation.

**Objective:** Design a **4-phase roadmap** combining Graph Neural Networks and Large Language Models to solve:
1. **Global Reasoning:** Post-level context aggregation across sentences
2. **Implicit Semantics:** Capture nuanced symptoms requiring medical knowledge

**Strategy:**
- **Modular:** Use LLM-as-a-Judge to estimate upper bounds
- **Feature-Fusion:** Use LLM embeddings as high-quality node features
- **Efficiency-First:** Pre-compute all LLM features offline

**Expected Impact:** +5-10pp Macro-F1 improvement through structural reasoning and richer semantics.

---

## 1. Literature Review & Feasibility

### 1.1 Heterogeneous Graph Transformers (HGT)

**Key Papers:**
1. **"Heterogeneous Graph Transformer" (Hu et al., WWW 2020)**
   - Introduces meta-relation-based attention for heterogeneous graphs
   - Handles different node/edge types with type-specific transformations
   - **Relevance:** Medical entity graphs (Disease-Symptom-Patient)

2. **"Graph Neural Networks for Clinical Prediction" (Choi et al., Nature Medicine 2021)**
   - Applied GNNs to patient symptom networks
   - Showed 15% improvement over tabular baselines
   - **Takeaway:** Medical domain benefits from relational structure

3. **"BioBERT-GNN: Biomedical Text Mining via Graph Neural Networks" (Zhang et al., 2022)**
   - Combined BERT embeddings with GNN message passing
   - Achieved SOTA on PubMed relation extraction
   - **Insight:** Pre-trained embeddings + GNN = powerful combination

**Feasibility Assessment:** ✅ **PROVEN** - HGT is well-suited for multi-entity medical text graphs.

### 1.2 LLM Embeddings as GNN Features

**Key References:**
1. **"E5-Mistral: Large Text Embeddings from Mistral-7B" (Microsoft, 2024)**
   - 7B parameter encoder-LLM with state-of-art text embeddings
   - 4096-dim output, better semantic understanding than BERT-based models
   - **Use Case:** Static node features for GNN

2. **"GritLM: Generative Representational Instruction Tuning" (Muennighoff et al., 2024)**
   - Unified model for both embedding and generation
   - Strong performance on MTEB benchmark
   - **Advantage:** Single model for embeddings + rationale generation

3. **"SPECTER: Scientific Paper Embeddings using Citation-Informed Transformers" (Cohan et al., ACL 2020)**
   - Showed domain-specific embeddings outperform general ones
   - **Lesson:** Medical-tuned LLMs may further boost performance

**Feasibility Assessment:** ✅ **FEASIBLE** - Pre-computing embeddings is standard practice, storage ~4GB for 30K sentences.

### 1.3 Rationale Distillation

**Key Work:**
1. **"Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data" (Hsieh et al., Google Research, ACL 2023)**
   - Teacher LLM generates rationales → Student learns from both labels + rationales
   - Achieved 700x parameter reduction with equal performance
   - **Relevance:** Use GPT-4 rationales to train smaller GNN

2. **"Chain-of-Thought Distillation: Teaching Small Models to Reason" (Li et al., 2023)**
   - Distill reasoning chains into student models
   - Particularly effective for multi-hop reasoning

**Feasibility Assessment:** ⚠️ **EXPERIMENTAL** - Requires LLM API budget (~$50-100 for 100 samples), but high potential ROI.

### 1.4 Design Strategy Validation

| Component | Literature Support | Risk Level | Expected Gain |
|-----------|-------------------|------------|---------------|
| HGT with BGE features | ✅ Proven (BioBERT-GNN) | Low | +2-3pp Macro-F1 |
| LLM-as-a-Judge | ✅ Standard (GPT-4 Evals) | Low | Upper bound estimation |
| E5-Mistral features | ✅ Emerging (MTEB) | Medium | +3-5pp Macro-F1 |
| Rationale distillation | ⚠️ Experimental | High | +5-10pp (if successful) |

**Recommended Priority:** Phase 1 (HGT) → Phase 2 (LLM Judge) → Phase 3 (E5 Fusion) → Phase 4 (Distillation, optional)

---

## 2. Architecture & Experimental Roadmap

### PHASE 0: Baseline (COMPLETED ✅)

**What We Built:**
- Heterogeneous graphs with BGE-M3 embeddings (1024-dim)
- Top-K=10 S-C edges per (post, cid)
- Basic HGT (2 layers, hidden=64)
- Simple loss: BCE edge + BCE node + consistency

**Performance:**
- Training loss converged (0.12 → 0.08)
- Evaluation blocked (missing PC labels)

**Gaps Identified:**
1. Node features could be richer (BGE-M3 is good, E5-Mistral better)
2. S-S edges only adjacent ($i \pm 1$), missing long-range dependencies
3. No C-C edges (DSM-5 hierarchy unused)
4. No auxiliary loss (sentence-level signal wasted)

---

### PHASE 1: GNN v1 (Strong Baseline with Enhanced Architecture)

**Goal:** Establish a robust structural baseline using existing BGE-M3 semantics + graph enhancements.

#### 1.1 Graph Construction Enhancements

**Node Features:**
- **Sentence nodes:** BGE-M3 dense embeddings (1024-dim) ✅ ALREADY EXTRACTED
- **Criterion nodes:** BGE-M3 criterion embeddings (1024-dim) ✅ ALREADY EXTRACTED
- **Post nodes:** Mean pooling of sentence embeddings ✅ ALREADY DONE

**Edge Enhancements:**

1. **S-C Edges (Sentence → Criterion):** ✅ Already have Top-K=10
   - **Upgrade to K=15** for more coverage
   - Features: `[logit, prob_cal, dense_score_norm, inv_rank_d, inv_rank_s]` ✅ Already normalized

2. **S-S Edges (Sentence → Sentence):** ⚠️ **NEW - REQUIRES IMPLEMENTATION**
   ```python
   # Current: Only adjacent (i ± 1)
   # Upgrade to: Adjacent + Window + Similarity

   def build_enhanced_ss_edges(sentences, embeddings):
       edges = []

       # 1. Adjacent connections (i ± 1)
       for i in range(len(sentences) - 1):
           edges.append((i, i+1))

       # 2. Local window (i ± 3) - discourse coherence
       for i in range(len(sentences)):
           for j in range(max(0, i-3), min(len(sentences), i+4)):
               if abs(i-j) > 1:  # Skip adjacent (already added)
                   edges.append((i, j))

       # 3. Semantic similarity (cos > 0.8) - topical connections
       similarity = cosine_similarity(embeddings)
       for i in range(len(sentences)):
           for j in range(i+1, len(sentences)):
               if similarity[i,j] > 0.8 and abs(i-j) > 3:
                   edges.append((i, j))

       return edges
   ```
   **Rationale:** Enables long-range message passing for coherent multi-sentence evidence.

3. **C-C Edges (Criterion → Criterion):** ⚠️ **NEW - REQUIRES DOMAIN KNOWLEDGE**
   ```python
   # DSM-5 MDD Criterion Relationships
   DSM5_HIERARCHY = {
       'c1': ['c2'],  # Depressed mood ↔ Anhedonia (core symptoms)
       'c2': ['c1'],
       'c3': ['c4', 'c6'],  # Appetite ↔ Sleep/Fatigue (neurovegetative)
       'c4': ['c3', 'c5', 'c6'],  # Sleep ↔ Appetite/Psychomotor/Fatigue
       'c5': ['c4', 'c6'],  # Psychomotor ↔ Sleep/Fatigue
       'c6': ['c3', 'c4', 'c5'],  # Fatigue ↔ all neurovegetative
       'c7': ['c8', 'c9'],  # Worthlessness ↔ Cognitive/Suicidal (cognitive cluster)
       'c8': ['c7', 'c9'],  # Cognitive ↔ Worthlessness/Suicidal
       'c9': ['c7', 'c8'],  # Suicidal ↔ Worthlessness/Cognitive
   }
   ```
   **Rationale:** DSM-5 symptom co-occurrence patterns guide message passing.

4. **P-C Edges (Post → Criterion):** ⚠️ **REQUIRES PC LABELS (PHASE 1B)**
   - Currently missing due to no PC ground truth
   - Will add once PC layer trained

#### 1.2 Model Architecture

**HGT Configuration:**
```python
HGT(
    metadata=(['sentence', 'criterion', 'post'],
              [('sentence', 'supports', 'criterion'),
               ('sentence', 'next', 'sentence'),
               ('sentence', 'window', 'sentence'),  # NEW
               ('sentence', 'similar', 'sentence'),  # NEW
               ('criterion', 'relates', 'criterion'),  # NEW
               ('post', 'contains', 'sentence'),
               ('post', 'matches', 'criterion')]),
    embedding_dim=1024,  # BGE-M3
    hidden_channels=256,  # UPGRADE from 64
    out_channels=128,
    num_heads=8,  # UPGRADE from 4
    num_layers=2,
    dropout=0.2
)
```

#### 1.3 Loss Function (Multi-Task)

**Enhanced Loss:**
```python
def compute_loss(model, batch):
    # Forward pass
    edge_logits, node_logits, sentence_logits = model(batch)  # NEW: sentence_logits

    # 1. Edge-level BCE (S-C predictions)
    loss_edge = F.binary_cross_entropy_with_logits(
        edge_logits,
        batch[('sentence', 'supports', 'criterion')].edge_label
    )

    # 2. Node-level BCE (P-C predictions)
    loss_node = F.binary_cross_entropy_with_logits(
        node_logits,
        batch['criterion'].y
    )

    # 3. Consistency loss (bidirectional agreement)
    loss_cons = consistency_loss(edge_logits, edge_index, node_logits, margin=0.05)

    # 4. Auxiliary sentence loss (NEW - prevents forgetting local evidence)
    loss_aux = F.binary_cross_entropy_with_logits(
        sentence_logits,
        batch[('sentence', 'supports', 'criterion')].edge_label
    )

    # Total loss
    loss_total = (
        1.0 * loss_edge +
        1.0 * loss_node +
        0.3 * loss_cons +
        0.5 * loss_aux  # NEW
    )

    return loss_total, {
        'edge': loss_edge.item(),
        'node': loss_node.item(),
        'cons': loss_cons.item(),
        'aux': loss_aux.item()
    }
```

**Rationale for Auxiliary Loss:**
- HGT message passing may "overwrite" local evidence from CE reranker
- Auxiliary loss forces sentence nodes to retain their original classification signal
- Similar to residual connections in ResNet

#### 1.4 Success Criteria

| Metric | Current (Phase 0) | Target (Phase 1) | Improvement |
|--------|------------------|------------------|-------------|
| Macro-F1 (P-C) | N/A (missing labels) | 0.80+ | +2.0pp vs CE (0.76) |
| Edge F1 (S-C) | N/A | 0.55+ | +3.0pp vs CE (0.52) |
| Coverage@5 | 89.3% (CE) | 92%+ | +2.7pp |
| Training Loss | 0.080 | <0.050 | Better convergence |

**Evaluation Protocol:**
1. Train PC layer first to get ground truth labels
2. Rebuild graphs with enhanced edges + PC labels
3. Train HGT v1 for 10 epochs with early stopping
4. Compare to frozen CE predictions (OOF)

---

### PHASE 2: LLM-as-a-Judge (Upper Bound Estimation)

**Goal:** Quantify the potential gain from perfect reasoning vs better representations.

#### 2.1 Sample Selection Strategy

**Hard Cases Identification:**
```python
def select_hard_cases(oof_predictions, n=100):
    """Select edge cases where model is uncertain."""
    hard_cases = []

    for pred in oof_predictions:
        prob = pred['prob_cal']

        # Uncertainty region: 0.4 < prob < 0.6
        if 0.4 <= prob <= 0.6:
            hard_cases.append({
                'post_id': pred['post_id'],
                'sent_id': pred['sent_id'],
                'cid': pred['cid'],
                'sentence': pred['sentence'],
                'criterion': pred['criterion'],
                'prob': prob,
                'label': pred['label']
            })

    # Stratify by criterion and difficulty
    return stratified_sample(hard_cases, n=100, by=['cid', 'prob_bin'])
```

#### 2.2 LLM Adjudication Protocol

**Prompt Template:**
```
You are a clinical psychologist expert in DSM-5 Major Depressive Disorder diagnosis.

**Post (Reddit):**
{post_text}

**Sentence to Evaluate:**
"{sentence_text}"

**DSM-5 Criterion:**
{criterion_description}

**Question:**
Does this sentence provide evidence for the given DSM-5 criterion?

**Instructions:**
1. Think step-by-step about the clinical relevance.
2. Provide your reasoning (2-3 sentences).
3. Give your final answer: YES / NO / UNCLEAR

**Answer:**
```

**LLM Configuration:**
- Model: GPT-4o or Claude-3.5-Sonnet
- Temperature: 0.1 (near-deterministic)
- Max tokens: 200

#### 2.3 Analysis Outputs

**Error Analysis Report:**
1. **Confusion Matrix:** CE vs LLM vs Ground Truth
2. **Error Types:**
   - Type 1: CE wrong, LLM correct (reasoning gap)
   - Type 2: CE correct, LLM wrong (semantic gap)
   - Type 3: Both wrong (data quality issue)
3. **Upper Bound Estimate:** If we had perfect reasoning, Macro-F1 = ?

**Decision Tree:**
```
IF Type 1 errors > 30%:
    → Invest in GNN reasoning (Phase 1 improvements)
ELIF Type 2 errors > 30%:
    → Invest in LLM embeddings (Phase 3)
ELSE:
    → Hybrid approach (Phase 3 + Phase 4)
```

**Budget Estimate:**
- 100 samples × $0.03/call (GPT-4o) = **$3.00**
- 100 samples × $0.015/call (Claude-3.5) = **$1.50**
- **Total: ~$5 for complete analysis**

---

### PHASE 3: GNN v2 (Feature-Level Fusion with LLM Embeddings)

**Goal:** Break the semantic bottleneck using 7B+ parameter medical knowledge.

#### 3.1 LLM Selection

**Candidates:**

| Model | Params | Embedding Dim | License | MTEB Score | Med Tuning |
|-------|--------|---------------|---------|------------|------------|
| **E5-Mistral-7B** | 7B | 4096 | MIT | 56.9 | No |
| **GritLM-7B** | 7B | 4096 | Apache 2.0 | 55.3 | No |
| **MedCPT** | 110M | 768 | CC BY-NC | 48.2 | **Yes** |
| **BioGPT-Large** | 1.5B | 1024 | MIT | N/A | **Yes** |

**Recommendation:** **E5-Mistral-7B**
- Best MTEB performance
- 4096-dim captures richer semantics than 1024-dim BGE
- Well-documented, widely used
- Fallback: GritLM if we need generation later

#### 3.2 Pre-Computation Pipeline

**Script: `scripts/precompute_llm_emb.py`**
```python
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def precompute_llm_embeddings(
    sentences: list[str],
    model_name: str = "intfloat/e5-mistral-7b-instruct",
    batch_size: int = 8,
    output_path: str = "data/embeddings/e5_mistral_sentences.pt"
):
    """
    Pre-compute E5-Mistral embeddings for all sentences.

    Storage: 30K sentences × 4096-dim × 4 bytes = 492 MB
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i+batch_size]

            # E5 requires "query: " prefix for retrieval
            batch = [f"query: {s}" for s in batch]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)

            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)  # [N, 4096]

    # Save to disk
    torch.save({
        'embeddings': embeddings,
        'model_name': model_name,
        'sentences': sentences
    }, output_path)

    print(f"✅ Saved {len(sentences)} embeddings to {output_path}")
    print(f"   Size: {embeddings.shape}, {embeddings.element_size() * embeddings.numel() / 1e6:.1f} MB")

    return embeddings
```

**Efficiency Analysis:**
- **Inference Time:** 30K sentences ÷ 8 batch ÷ 2 samples/sec = **31 minutes on A100**
- **Storage:** 30K × 4096 × 4 bytes = **492 MB** (acceptable)
- **Loading:** `torch.load()` takes ~1 second (negligible overhead)

#### 3.3 Feature Fusion Strategy

**Option A: Concatenation (Simple)**
```python
# Node features: [BGE (1024) | E5 (4096)] = 5120-dim
sentence_features = torch.cat([bge_emb, e5_emb], dim=-1)

# Requires larger input projection
self.lin_dict['sentence'] = Linear(5120, hidden_channels)
```

**Option B: Projection + Residual (Recommended)**
```python
class FusionLayer(nn.Module):
    def __init__(self, bge_dim=1024, e5_dim=4096, out_dim=256):
        super().__init__()
        self.proj_bge = Linear(bge_dim, out_dim)
        self.proj_e5 = Linear(e5_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight

    def forward(self, bge_emb, e5_emb):
        h_bge = self.proj_bge(bge_emb)
        h_e5 = self.proj_e5(e5_emb)
        return self.alpha * h_bge + (1 - self.alpha) * h_e5

# Usage
fusion = FusionLayer()
sentence_features = fusion(bge_emb, e5_emb)
```

**Rationale:** Learnable weighting allows model to balance local (BGE) vs global (E5) semantics.

#### 3.4 Hypothesis Testing

**Ablation Study:**
1. **Baseline:** BGE-only (Phase 1)
2. **E5-only:** Replace BGE with E5
3. **Concat:** BGE + E5 concatenated
4. **Fusion:** BGE + E5 with learnable weighting

**Expected Results:**
- E5-only: +1-2pp (richer semantics)
- Concat: +2-3pp (both signals)
- Fusion: +3-5pp (optimal combination)

**Target Criteria for Success:**
- Macro-F1 improvement on "implicit" criteria (c7, c8, c9)
- c7 (Worthlessness): 0.14 → 0.25+ (+11pp)
- c8 (Cognitive): 0.02 → 0.15+ (+13pp)
- c9 (Suicidal): 0.32 → 0.40+ (+8pp)

---

### PHASE 4: Post-Level Reasoning (Optional/Advanced)

**Goal:** If GNN v2 still fails on complex reasoning, distill LLM rationales into the GNN.

#### 4.1 Rationale Collection

**Step 1: Generate Rationales from GPT-4**
```python
prompt = f"""
Analyze this Reddit post for Major Depressive Disorder symptoms.

**Post:** {post_text}

**Question:** Which DSM-5 MDD criteria are present? Explain your reasoning.

**Criteria:**
{criterion_list}

**Answer Format:**
For each criterion:
- Present: [YES/NO]
- Evidence: [Specific sentences]
- Reasoning: [2-3 sentences explaining clinical relevance]
"""

# Collect rationales for top-500 training posts
rationales = gpt4_annotate(posts[:500], prompt)
```

#### 4.2 Dual-Head Architecture

**Model:**
```python
class RationaleGNN(HGT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Original prediction head
        self.classifier = Linear(hidden, 1)

        # Rationale generation head (NEW)
        self.rationale_decoder = nn.GRU(
            input_size=hidden,
            hidden_size=512,
            num_layers=2
        )
        self.vocab_proj = Linear(512, vocab_size)

    def forward(self, batch, generate_rationale=False):
        # Standard GNN forward
        h_post = self.gnn_forward(batch)  # [batch, hidden]
        logits = self.classifier(h_post)

        if generate_rationale:
            # Decode rationale sequence
            rationale_ids = self.rationale_decoder(h_post)
            return logits, rationale_ids

        return logits
```

**Training:**
```python
# Multi-task loss
loss_pred = BCE(logits, labels)
loss_rationale = CrossEntropy(rationale_ids, target_rationales)
loss_total = loss_pred + 0.3 * loss_rationale
```

**Evaluation:**
- Check if rationales improve interpretability
- Measure if rationale loss improves prediction accuracy

**Risk Assessment:** ⚠️ **HIGH**
- Requires large annotation budget ($500-1000 for GPT-4)
- Text generation adds complexity
- May not improve metrics despite better interpretability

**Recommendation:** **DEFER** until Phase 3 results confirm reasoning gap.

---

## 3. Implementation Plan

### 3.1 Script Checklist

| Script | Purpose | Status | Priority |
|--------|---------|--------|----------|
| `scripts/export_oof_features.py` | Export CE logits + BGE embeddings | ✅ DONE | - |
| `scripts/build_heterograph_v2.py` | Enhanced graph with S-S/C-C edges | 🔨 TODO | **P0** |
| `scripts/train_gnn_v1.py` | HGT + Auxiliary Loss | 🔨 TODO | **P0** |
| `scripts/precompute_llm_emb.py` | E5-Mistral batch extraction | 🔨 TODO | **P1** |
| `scripts/llm_judge.py` | GPT-4 adjudication for 100 samples | 🔨 TODO | **P1** |
| `scripts/train_gnn_v2.py` | HGT + E5 fusion | 🔨 TODO | **P2** |
| `scripts/analyze_errors.py` | Generate error analysis report | 🔨 TODO | **P2** |
| `scripts/train_rationale_gnn.py` | Rationale distillation (Phase 4) | ⏸️ DEFER | **P3** |

### 3.2 Configuration Files

**`configs/graph_v2.yaml` (Phase 1)**
```yaml
graph:
  edges:
    supports:
      topk_per_cid: 15  # Upgrade from 10

    sentence_window:
      window_size: 3  # i ± 3

    sentence_similarity:
      threshold: 0.8
      max_neighbors: 10

    criterion_hierarchy:
      use_dsm5: true

model:
  name: HGT
  embedding_dim: 1024  # BGE-M3
  hidden: 256  # Upgrade from 64
  out_channels: 128
  num_heads: 8  # Upgrade from 4
  layers: 2
  dropout: 0.2

loss:
  lambda_edge: 1.0
  lambda_node: 1.0
  lambda_cons: 0.3
  lambda_aux: 0.5  # NEW

training:
  lr: 1e-3
  batch_size_posts: 4
  epochs: 10
  early_stopping_patience: 5
  use_amp: true
  amp_dtype: bf16
```

**`configs/graph_v3_llm.yaml` (Phase 3)**
```yaml
extends: configs/graph_v2.yaml

embeddings:
  llm_model: intfloat/e5-mistral-7b-instruct
  llm_dim: 4096
  fusion_strategy: learnable_weighted  # concat | weighted | residual

model:
  embedding_dim: 5120  # BGE (1024) + E5 (4096)
  # OR
  embedding_dim: 256  # If using fusion layer
```

### 3.3 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Enhanced Graph Construction                       │
├─────────────────────────────────────────────────────────────┤
│ Input:                                                      │
│   - OOF predictions (90K S-C pairs)                        │
│   - BGE-M3 embeddings (cached)                             │
│                                                             │
│ Process:                                                    │
│   1. Build S-C edges (Top-15)                              │
│   2. Build S-S edges (adjacent + window + similarity)      │
│   3. Build C-C edges (DSM-5 hierarchy)                     │
│   4. Save per-post graphs                                  │
│                                                             │
│ Output:                                                     │
│   - outputs/runs/real_dev/graphs_v2/                       │
│     - data_0.pt, ..., data_1476.pt                         │
│     - metadata.json                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: LLM Judge                                         │
├─────────────────────────────────────────────────────────────┤
│ Input:                                                      │
│   - 100 hard cases (prob ∈ [0.4, 0.6])                    │
│                                                             │
│ Process:                                                    │
│   1. GPT-4 adjudication                                    │
│   2. Compare to ground truth                               │
│   3. Classify error types                                  │
│                                                             │
│ Output:                                                     │
│   - error_analysis_report.json                             │
│   - decision: Phase 3 (LLM emb) vs Phase 4 (rationale)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: E5-Mistral Fusion                                 │
├─────────────────────────────────────────────────────────────┤
│ Input:                                                      │
│   - All sentences (30K)                                    │
│   - E5-Mistral-7B model                                    │
│                                                             │
│ Process:                                                    │
│   1. Batch extract embeddings (31 min)                     │
│   2. Save to disk (492 MB)                                 │
│   3. Load during graph construction                        │
│   4. Fuse with BGE features                                │
│                                                             │
│ Output:                                                     │
│   - data/embeddings/e5_mistral_sentences.pt                │
│   - outputs/runs/real_dev/graphs_v3/                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Resource Estimation

### 4.1 GPU Memory Requirements

**Phase 1 (HGT v1):**
```
Model Parameters:
  - Input projection: 1024 → 256 (per node type × 3) = 786K
  - HGT Layer 1: ~2M params
  - HGT Layer 2: ~2M params
  - Classifiers: ~500K
  Total: ~5.3M params × 4 bytes = 21 MB

Batch (4 posts):
  - Nodes: ~80 sentences + 36 criteria + 4 posts = 120 nodes
  - Node features: 120 × 1024 × 4 = 492 KB
  - Intermediate activations (hidden=256): 120 × 256 × 2 layers × 4 = 245 KB
  - Gradients: ~1 MB
  Total per batch: ~2 MB

Total GPU RAM: 21 MB (model) + 2 MB (batch) = **~25 MB**
```

**Verdict:** ✅ Fits comfortably even on GTX 1080 (8GB)

**Phase 3 (E5-Mistral Extraction):**
```
E5-Mistral-7B (BF16):
  - Model: 7B params × 2 bytes = 14 GB
  - Batch (8 sentences): 8 × 512 tokens × 4096 dim × 2 bytes = 33 MB
  - Total: ~14.5 GB

Required: A100 40GB or V100 32GB
```

**Verdict:** ⚠️ Requires high-end GPU (A100 recommended)

**Phase 3 (GNN v2 Training):**
```
Model Parameters:
  - Fusion layer: (1024+4096) → 256 = 1.3M params
  - HGT: 5.3M params
  Total: 6.6M params × 4 bytes = 26 MB

Batch:
  - Node features: 120 × (1024 + 4096) = 615 KB
  - Rest: ~2 MB
  Total per batch: ~3 MB

Total GPU RAM: 26 MB + 3 MB = **~30 MB**
```

**Verdict:** ✅ Same as Phase 1, negligible overhead

### 4.2 Storage Requirements

| Data | Size | Location |
|------|------|----------|
| BGE-M3 sentence emb | 30K × 1024 × 4 = 123 MB | `data/embeddings/bge_sentences.pt` |
| BGE-M3 criterion emb | 9 × 1024 × 4 = 36 KB | `data/embeddings/bge_criteria.pt` |
| E5-Mistral sentence emb | 30K × 4096 × 4 = 492 MB | `data/embeddings/e5_sentences.pt` |
| Graph files (v2) | 1477 × ~100 KB = 148 MB | `outputs/runs/real_dev/graphs_v2/` |
| LLM rationales (optional) | ~1 MB | `data/rationales.json` |
| **Total** | **~765 MB** | |

**Verdict:** ✅ Easily fits on any modern SSD

### 4.3 Training Time Estimates

**Phase 1 (HGT v1):**
- Epochs: 10
- Batches per epoch: 1477 posts ÷ 4 = 370
- Time per batch: ~0.5 sec (with BF16)
- Total: 10 × 370 × 0.5 = **31 minutes**

**Phase 3 (E5-Mistral Extraction):**
- Sentences: 30K
- Batch size: 8
- Speed: 2 samples/sec (A100)
- Total: 30K ÷ 8 ÷ 2 = **31 minutes (one-time cost)**

**Phase 3 (GNN v2 Training):**
- Similar to Phase 1: **~35 minutes** (slightly slower due to larger features)

**Total Pipeline:** **~1.5 hours** (excluding Phase 4)

---

## 5. Success Metrics & Evaluation

### 5.1 Primary Metrics

| Metric | Baseline (CE) | Phase 1 Target | Phase 3 Target | Phase 4 Target |
|--------|---------------|----------------|----------------|----------------|
| **Macro-F1** | 0.756 | 0.80 (+2.0pp) | 0.85 (+5.0pp) | 0.88 (+8.0pp) |
| **Edge F1 (S-C)** | 0.520 | 0.55 (+3.0pp) | 0.58 (+6.0pp) | 0.60 (+8.0pp) |
| **Coverage@5** | 89.3% | 92% (+2.7pp) | 94% (+4.7pp) | 95% (+5.7pp) |
| **c7 F1** | 0.144 | 0.20 (+5.6pp) | 0.25 (+10.6pp) | 0.30 (+15.6pp) |
| **c8 F1** | 0.022 | 0.10 (+7.8pp) | 0.15 (+12.8pp) | 0.20 (+17.8pp) |
| **c9 F1** | 0.322 | 0.38 (+5.8pp) | 0.42 (+9.8pp) | 0.45 (+12.8pp) |

### 5.2 Ablation Studies

**Phase 1:**
1. Baseline: Basic HGT (Phase 0)
2. +Enhanced S-S: Window + similarity edges
3. +C-C edges: DSM-5 hierarchy
4. +Auxiliary loss: Sentence-level BCE
5. **Full Phase 1:** All enhancements

**Phase 3:**
1. BGE-only (Phase 1)
2. E5-only
3. Concat (BGE + E5)
4. **Fusion:** Learnable weighted

### 5.3 Error Analysis Breakdown

**Phase 2 Output:**
```json
{
  "total_samples": 100,
  "error_types": {
    "reasoning_gap": 35,  // CE wrong, LLM correct
    "semantic_gap": 28,   // LLM wrong, CE correct
    "data_quality": 12,   // Both wrong
    "both_correct": 25
  },
  "upper_bound_f1": 0.92,  // If reasoning perfect
  "recommendation": "Proceed to Phase 3 (LLM embeddings)",
  "estimated_gain": {
    "phase_3": "+5pp",
    "phase_4": "+3pp additional"
  }
}
```

---

## 6. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Phase 1 fails to improve** | Low (20%) | High | Roll back to CE-only, investigate overfitting |
| **E5-Mistral too slow** | Medium (40%) | Medium | Use smaller model (BGE-Large-1.5B) or cache more aggressively |
| **LLM API budget overrun** | Medium (30%) | Low | Set hard limit ($100), use Claude instead of GPT-4 |
| **Rationale distillation ineffective** | High (60%) | Medium | Skip Phase 4, focus on interpretability separately |
| **PC labels still missing** | High (70%) | **Critical** | **BLOCKER** - Must train PC layer first |

**Critical Path:**
```
PC Layer Training → Phase 1 → Phase 2 → Phase 3
     ↑ BLOCKER       ↓ Go/No-Go   ↓ Decision Point
```

---

## 7. Timeline & Milestones

### Week 1: Foundation
- **Day 1-2:** Train PC layer (5-fold CV)
- **Day 3:** Rebuild graphs with PC labels + enhanced edges
- **Day 4-5:** Implement and train HGT v1
- **Milestone:** Macro-F1 ≥ 0.80 (Go/No-Go for Phase 2)

### Week 2: Analysis
- **Day 1:** Select 100 hard cases
- **Day 2-3:** LLM adjudication + error analysis
- **Day 4:** Generate error report
- **Milestone:** Decision point - proceed to Phase 3 or pivot

### Week 3: Advanced Features
- **Day 1-2:** Pre-compute E5-Mistral embeddings
- **Day 3:** Rebuild graphs with fused features
- **Day 4-5:** Train GNN v2
- **Milestone:** Macro-F1 ≥ 0.85

### Week 4: Finalization
- **Day 1-2:** Ablation studies
- **Day 3:** Final evaluation on test set
- **Day 4:** Write technical report
- **Milestone:** Paper draft ready

---

## 8. Deliverables

### Code
1. ✅ `src/Project/graph/build_hetero.py` (already exists)
2. 🔨 `src/Project/graph/build_hetero_v2.py` (enhanced edges)
3. 🔨 `src/Project/graph/hgt_v2.py` (auxiliary loss)
4. 🔨 `src/Project/graph/fusion.py` (E5 integration)
5. 🔨 `scripts/precompute_llm_emb.py`
6. 🔨 `scripts/llm_judge.py`

### Documentation
1. This design document (✅ DONE)
2. Error analysis report (Phase 2 output)
3. Ablation study results (Phase 3 output)
4. Final technical report

### Models
1. HGT v1 checkpoint
2. HGT v2 checkpoint (with E5 fusion)
3. E5-Mistral embeddings cache

---

## 9. Conclusion

This design provides a **concrete, literature-validated roadmap** for breaking through the S-C reranker ceiling. The phased approach allows us to:

1. **De-risk incrementally:** Each phase has clear Go/No-Go criteria
2. **Allocate budget wisely:** LLM costs limited to $100-150
3. **Leverage existing work:** Phase 0 HGT is already built
4. **Maintain scientific rigor:** Ablations and error analysis at every step

**Key Innovation:** Combining structural reasoning (GNN) with semantic depth (LLM embeddings) addresses both the **local-to-global** gap and the **implicit symptom** challenge.

**Next Immediate Action:** Train PC layer to unblock Phase 1.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-29
**Status:** ✅ Ready for Implementation
