# LDP Attacker Detection

A deep learning-based attacker detection system for Local Differential Privacy (LDP) protocols.

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Multi-layer perceptron with BatchNorm and Dropout |
| `gan` | GAN-style discriminator with LayerNorm |
| `attention` | Transformer-style with per-feature embeddings and multi-head attention |

## Installation

```bash
conda env create -f environment.yml
conda activate attacker-detector
```

## Dataset Generation

Generate LDP attack detection training data:

```bash
# Generate with defaults
python generate_dataset.py --output dataset.csv

# Custom configuration
python generate_dataset.py --output custom.csv \
    --protocols OUE OLH_Server OLH_User HST_User HST_Server \
    --epsilons 0.5 1.0 2.0 \
    --datasets zipf emoji fire \
    --ratios 0.10 0.15 0.20 \
    --experiments 5
```

### Dataset Generation CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Output CSV file path | *required* |
| `--protocols` | LDP protocols: `OUE`, `OLH` | `['OUE', 'OLH']` |
| `--epsilons` | Privacy parameters | `[0.5, 0.7, 1.0, 1.5]` |
| `--datasets` | Dataset types: `zipf`, `emoji`, `fire` | all three |
| `--ratios` | Attacker ratios | `[0.10, 0.15, 0.20]` |
| `--target-sizes` | Target set sizes | `[2, 4, 6, 8]` |
| `--splits` | Split values | `[2, 4, 6, 8]` |
| `--experiments` | Experiments per config | 5 |
| `--full-scale` | Use full-scale dataset sizes | False |
| `--n` | Override number of users | None |
| `--processors` | Parallel processes | 4 |

## Model Training

## Usage

```bash
# Train MLP model
python main.py --data-path dataset.csv --model mlp --epochs 5

# Train GAN model
python main.py --data-path dataset.csv --model gan --epochs 10

# Train Attention model
python main.py --data-path dataset.csv --model attention --epochs 15

# Save outputs to directory
python main.py --data-path dataset.csv --model mlp --output-dir ./results

# Cross-dataset: train on zipf, test on emoji
python main.py --data-path dataset.csv --model mlp \
    --training-method cross --train-dataset zipf --test-dataset emoji

# Three-way: train on zipf, test on emoji, evaluate on fire
python main.py --data-path dataset.csv --model mlp \
    --training-method three-way \
    --train-dataset zipf --test-dataset emoji --eval-dataset fire
```
### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path`, `-d` | Path to CSV dataset | *required* |
| `--model`, `-m` | Model type: `mlp`, `gan`, `attention` | *required* |
| `--epochs`, `-e` | Training epochs | 5 |
| `--batch-size`, `-b` | Batch size | 256 |
| `--lr` | Learning rate | 0.001 |
| `--dropout` | Dropout rate | 0.2 |
| `--test-size` | Test split ratio (only for `none` mode) | 0.2 |
| `--seed` | Random seed | 42 |
| `--output-dir`, `-o` | Save model/plots here | None |
| `--no-plot` | Skip sensitivity plots | False |

### Generalizability Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--training-method` | `none` (conventional random split), `cross` (train on one dataset type, test on another), `three-way` (train/test/eval on separate types) | `none` |
| `--train-dataset` | Dataset type for training: `zipf`, `emoji`, `fire` (required for `cross` / `three-way`) | None |
| `--test-dataset` | Dataset type for testing: `zipf`, `emoji`, `fire` (required for `cross` / `three-way`) | None |
| `--eval-dataset` | Dataset type for evaluation: `zipf`, `emoji`, `fire` (required for `three-way`) | None |

**Training methods:**

- **`none`** — Conventional training. All dataset types are mixed together and split randomly into train/test sets using `--test-size`.
- **`cross`** — Cross-dataset generalizability. The model is trained entirely on one dataset type and tested on another (e.g., train on `zipf`, test on `emoji`).
- **`three-way`** — Full generalizability evaluation. Train on one type, test on a second, and run an additional evaluation + sensitivity analysis on a third (e.g., train `zipf`, test `emoji`, evaluate `fire`).

## License

MIT
