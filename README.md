SCOUT detects vulnerabilities in smart contracts from **opcode-level runtime traces**. It models known vulnerability classes with a CNN–BiLSTM–Attention–GCN network fused with CBOW opcode embeddings, parameter-type embeddings, and statistical features, then detects *unknown* vulnerabilities via an anomaly/OOD scoring scheme (anomaly head combined with a Mahalanobis minimum-distance criterion).

Six vulnerability categories are covered:

| Category | Directory |
|---|---|
| Reentrancy | `reentrancy` |
| Timestamp dependency | `timestamp_dependency` |
| Balance check (strict equality) | `balance_check` |
| Block-number dependency | `block_number_dependency` |
| Integer overflow / underflow | `integer_overflow` |
| Unchecked external call | `unchecked_external_call` |

## Repository layout

```
.
├── models/          # Main DL framework (V1–V5, cross-validation, evaluation, plotting)
├── *.py             # Data preprocessing / feature extraction / simulation scripts
├── requirements.txt
└── .gitignore
```

## Requirements

- **Python 3.11** (tested). PyTorch 2.5.1 with CUDA 12.1 was used; any `torch>=2.1` should work.
- **Node.js 18+** and Hardhat — only needed for the tracer (`tracer/`).

```bash
pip install -r requirements.txt
# torch with CUDA (pick the index for your hardware):
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

```
### Data sources

The raw smart-contract source code (Solidity) used to produce these traces is drawn from three public benchmarks:

- [IR-Fuzz](https://github.com/Messi-Q/IR-Fuzz)
- [SolidiFI](https://github.com/DependableSystemsLab/SolidiFI)
- [SmartBugs](https://smartbugs.github.io/)

## Pipeline

1. **Extract parameter features** for normal traces:
   ```bash
   python generate_normal_param_features.py
   ```

2. **Clean the dataset** (dedup / drop malformed traces):
   ```bash
   python clean_traces.py
   ```

3. **Full pipeline** (vocab → precompute stats → CBOW embeddings → train → detect → evaluate → plots):
   ```bash
   cd models
   python run_experiments.py              # full run
   python run_experiments.py --quick      # smoke test on 1000 samples
   ```
4. **6-fold leave-one-category-out cross-validation** (resume-safe):
   ```bash
   cd models
   bash run_cv.sh
   ```

5. **Final detection (V4)** — anomaly OR Mahalanobis min-distance:
   ```bash
   cd models
   python detect_v4.py
   ```

## Key files in `models/`

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters (single source of truth) |
| `run_experiments.py` | End-to-end experiment runner |ation |
| `train.py` / `train_v2.py` / `train_v5.py` | Training variants (V1/V2/V5) |
| `train_anomaly.py` / `anomaly_head.py` | Anomaly head training |
| `detect_v4.py` | Final V4 detection (anomaly OR min-distance) |
| `detect_v4_ablation.py` | V4 ablations |
| `model.py` / `embeddings.py` / `vocabulary.py` / `precompute.py` / `dataset.py` | Model, CBOW embeddings, vocab, feature precompute, data loading |
| `plot.py` | Figure generation |
| `opc_vocab.pkl` / `param_vocab.pkl` / `atom_registry.pkl` | Pre-built vocabularies (committed for inference reproducibility) |

Model checkpoints and experiment outputs are generated under `models/output/` and `models/checkpoints/` at runtime.
