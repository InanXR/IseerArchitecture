# Iseer Architecture

**A Novel Mamba × MoE Hybrid Language Model**

Built from scratch by [Inan](https://github.com/InanXR) • Iseer & Co.

---

## 🧬 Architecture

Iseer combines two cutting-edge innovations:
- **Mamba (State Space Models)** — O(n) linear sequence modeling
- **Mixture of Experts (MoE)** — Sparse activation for efficiency

```
┌─────────────────────────────────────────┐
│           ISEER BLOCK                   │
├─────────────────────────────────────────┤
│  Input → RMSNorm → Mamba SSM            │
│            ↓                            │
│  + residual                             │
│            ↓                            │
│  RMSNorm → MoE (top-k routing)          │
│            ↓                            │
│  + residual → Output                    │
└─────────────────────────────────────────┘
```

## 🚀 Features

- **Linear Complexity**: O(n) instead of O(n²) attention
- **Sparse Compute**: Only top-k experts active per token
- **Triton Kernels**: Custom GPU kernels for selective scan
- **Bilingual**: Trained on Bengali + English

## 📦 Installation

```bash
pip install -e .
```

## 🔧 Usage

```python
from iseer import Iseer, IseerConfig

config = IseerConfig(
    vocab_size=32000,
    d_model=512,
    n_layers=12,
    n_experts=8,
    top_k=2,
)

model = Iseer(config)
```

## 📊 Model Variants

| Model | Params | Active | Context |
|-------|--------|--------|---------|
| ISEER-SM | 30M | 20M | 2048 |
| ISEER-MD | 120M | 45M | 4096 |

## 📄 License

MIT

## 🔗 Links

- [Iseer & Co.](https://iseer.co)
- [Paper (coming soon)]()
