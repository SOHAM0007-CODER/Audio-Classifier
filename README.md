# Multi-Task Audio Spectrogram Transformer (AST) Classifier

A multi-task audio classification pipeline leveraging the Audio Spectrogram Transformer (AST) for joint classification of music genres (GTZAN) and environmental sounds (ESC-50), optimized for consumer hardware.

---

## 🎯 Architecture & Datasets

- **Music Domain (`domain = 0`)**: GTZAN dataset with 10 musical genres (`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`).
- **Environmental Domain (`domain = 1`)**: ESC-50 dataset with 50 environmental sound categories.
- **Audio Preprocessing**:
  - Resampled to 16,000 Hz mono.
  - Sliced into non-overlapping 5.0s clips (80,000 samples each).
  - **Leakage Prevention**: GTZAN tracks are split at the *song level* (80/10/10) *before* slicing into six 5.0s chunks, ensuring zero cross-split track leakage.
  - **Corrupt File Resilience**: Known problematic files (like `jazz.00054.wav`) are caught and handled via `try/except`.
  - **ESC-50 Splits**: 80/10/10 stratified split respecting official 5-fold partitions.
- **Feature Extractor**: HuggingFace `AutoFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')`, generating log-mel spectrogram tensors of shape `(1024, 128)`.

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Manifests & Process Datasets

Run the manifest generation script:
```bash
python scripts/prepare_manifests.py
```

This will:
1. Scan GTZAN tracks in `Data/genres_original/`.
2. Automatically download and unpack ESC-50 if not already present.
3. Resample all audio to 16 kHz mono and slice into 5.0s chunks saved in `data/processed/`.
4. Output CSV manifests in `data/manifests/`:
   - `train.csv`
   - `val.csv`
   - `test.csv`
   Columns: `filepath`, `domain`, `label`
5. Export class mappings:
   - `data/manifests/music_classes.json` (0-9)
   - `data/manifests/env_classes.json` (0-49)

### 3. Load Dataset in PyTorch

```python
from src.dataset import AudioDataset, create_dataloader

# Load train dataset
train_dataset = AudioDataset(manifest_path="data/manifests/train.csv")
print(f"Total samples: {len(train_dataset)}")

# Fetch a sample
sample = train_dataset[0]
print("input_values shape:", sample["input_values"].shape)  # torch.Size([1024, 128])
print("domain:", sample["domain"])                          # 0 or 1
print("label:", sample["label"])                            # class index

# Create DataLoader optimized for consumer GPU
train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
for batch in train_loader:
    print(batch["input_values"].shape)  # torch.Size([8, 1024, 128])
    print(batch["domain"].shape)        # torch.Size([8])
    print(batch["label"].shape)         # torch.Size([8])
    break
```

---

## 💻 Consumer Hardware Optimizations (GTX 1650 4GB)

1. **Offline Audio Slicing & Resampling**: All heavy resampling (from 22.05 kHz or 44.1 kHz to 16 kHz) is pre-computed during manifest preparation. Audio clips are saved as exact 80,000-sample WAV files, eliminating runtime CPU bottlenecks during training.
2. **Squeezed Spectrogram Output**: `AudioDataset` outputs 2D tensors `(1024, 128)`, allowing standard PyTorch `DataLoader` collation into `(batch_size, 1024, 128)`.
3. **Memory Management**: Gradient checkpointing is on by default, so a batch of 8 fits in ~2.4 GiB. float16 mixed precision is **off** by default: GTX 16xx cards have no tensor cores and ran AST training ~5x slower in float16 than in float32 (see Training below).

---

## 🏋️ Training

```bash
python -m src.train                                          # 20 epochs, batch 8, AdamW, 2-epoch warmup + cosine decay
python -m src.train --batch_size 4 --grad_accum_steps 2      # same effective batch with less VRAM
python -m src.train --config my_config.yaml --lr_heads 1e-3  # YAML values; CLI flags override them
python -m src.train --resume auto                            # continue from checkpoints/latest_checkpoint.pt
python -m src.train --help                                   # all options
```

Key options (defaults): `--lr_backbone` (2e-5), `--lr_heads` (5e-4), `--weight_decay` (1e-4), `--epochs` (20), `--warmup_epochs` (2), `--grad_accum_steps` (1), `--music_loss_weight` / `--env_loss_weight` (1.0), `--fp16` / `--no-fp16` (off), `--gradient_checkpointing` / `--no-gradient_checkpointing` (on), `--num_workers` (0 on Windows), `--output_dir` (`checkpoints/`).

After every epoch the model is evaluated on `val.csv`: accuracy and macro-F1 for each task, plus `combined_metric = 0.5 * music_acc + 0.5 * env_acc`. Files written to `output_dir`:

| File | Contents |
| --- | --- |
| `best_model.pt` | Model weights and validation metrics from the epoch with the highest `combined_metric` |
| `latest_checkpoint.pt` | Model, optimizer, scheduler and scaler state plus history, for `--resume` |
| `metrics.jsonl` | Losses, learning rates and grad norm per log step; all metrics per epoch |
| `train.log`, `config.yaml` | Console log and the resolved configuration |
| `test_metrics.json` | Test-set metrics of `best_model.pt`, written after training |

Measured training-step cost on a GTX 1650 4GB (measured while another process shared the GPU, so times are upper bounds):

| Batch | Gradient checkpointing | Precision | Peak reserved memory | Seconds / step |
| --- | --- | --- | --- | --- |
| 4 | on | float32 | 1.87 GiB | 5.7 |
| 8 | on | float32 | 2.40 GiB | 11.7 |
| 4 | off | float32 | 4.16 GiB (spills into system RAM) | 45.6 |
| 4 | on | float16 | 1.84 GiB | 26.8 |

---

## ☁️ Cloud GPU Training

Full training is meant for a cloud GPU with 16 GB+ VRAM (T4, A10G, RTX 3090, A100); the GTX 1650 above is only practical for tests. `configs/cloud_train.yaml` uses batch 16, float16 mixed precision, no gradient checkpointing, 4 workers and 20 epochs, and writes to `checkpoints/cloud_run/`.

### 1. Get the code onto the machine

- **Git:** push the repository and clone it (datasets, manifests and checkpoints are git-ignored), or
- **Zip:** `python scripts/package_for_cloud.py` writes `dist/multitask_ast_code.zip` with only the code, configs, scripts, notebook and tests. Upload it to Google Drive or the instance.

Datasets are not uploaded. GTZAN is downloaded on the machine from Kaggle (`andradaolteanu/gtzan-dataset-music-genre-classification`, via `kagglehub`), ESC-50 from GitHub by `scripts/prepare_manifests.py`, and the manifests are regenerated. On Linux, `Data/` (raw GTZAN / ESC-50) and `data/` (processed clips and manifests) are separate directories.

### 2a. Google Colab

Open `notebooks/train_colab.ipynb`, select a GPU runtime, set `REPO_URL` or `PROJECT_ZIP` in cell 0 and run the cells in order:

| Cell | Step |
| --- | --- |
| 0 | Mount Google Drive, get the code, keep `checkpoints/` on Drive |
| 1 | `nvidia-smi`, `pip install -r requirements.txt`, pre-flight unit tests |
| 2 | Download GTZAN, `python scripts/prepare_manifests.py` (ESC-50 auto-download), manifest and leakage checks |
| 3 | `python -m src.train --config configs/cloud_train.yaml --resume auto` |
| 4 | `python -m src.evaluate` on the test split and a `python -m src.predict` demo |

If the session disconnects, re-run cells 0-3: training resumes from `checkpoints/cloud_run/latest_checkpoint.pt` on Drive.

### 2b. RunPod / Lambda / Vast.ai (any Linux GPU machine)

```bash
bash scripts/train_cloud.sh                                        # all four steps; re-run to resume
GTZAN_DIR=/workspace/genres_original bash scripts/train_cloud.sh   # use an existing GTZAN copy
bash scripts/train_cloud.sh --batch_size 8 --grad_accum_steps 2    # extra flags are passed to src.train
```

Or step by step:

```bash
pip install -r requirements.txt
python scripts/prepare_manifests.py --gtzan-dir /path/to/genres_original
python -m src.train --config configs/cloud_train.yaml --resume auto
python -m src.evaluate --checkpoint checkpoints/cloud_run/best_model.pt --fp16 --output checkpoints/cloud_run/test_metrics.json
python -m src.predict --audio path/to/clip.wav --checkpoint checkpoints/cloud_run/best_model.pt
```

Notes:
- Flags override the YAML file, e.g. `--num_workers 2` on a 2-vCPU machine (the notebook and script cap workers at the CPU count) or `--batch_size 8 --grad_accum_steps 2` if CUDA runs out of memory.
- `requirements.txt` pins `transformers` 5.x. Checkpoint weight names depend on its AST module layout, so use the same major version in the cloud and locally.
- To use the trained model locally, copy `checkpoints/cloud_run/best_model.pt` to `checkpoints/best_model.pt` and run `python -m src.predict --audio clip.wav`.

---

## 🔮 Inference

```bash
python -m src.predict --audio path/to/clip.wav                     # both heads, top-3
python -m src.predict --audio song.flac --domain music --top_k 5  # GTZAN genres only
python -m src.predict --audio dog.wav --domain env --json         # ESC-50 sounds only, JSON output
python -m src.predict --audio clip.wav --checkpoint none          # pretrained backbone, untrained heads (pipeline check)
```

Audio is converted to 16 kHz mono and split into 5.0 s windows (the last window is zero-padded and kept if it is at least 2.5 s long or the only window). Softmax probabilities are averaged over up to `--max_chunks` windows (default 6, the first 30 s). `--checkpoint` defaults to `checkpoints/best_model.pt`; if that file does not exist, the script warns and falls back to untrained heads.

```python
from src.predict import AudioClassifier, predict

result = predict("clip.wav", checkpoint="checkpoints/best_model.pt", domain="music", top_k=5)
print(result["predictions"]["music"][0])  # {'rank': 1, 'label': 'rock', 'index': 9, 'confidence': 0.82}

classifier = AudioClassifier.from_checkpoint("checkpoints/best_model.pt")  # load once for many files
for path in ["a.wav", "b.wav"]:
    print(classifier.predict(path)["predictions"])
```

---

## 🧪 Testing

Run test suite:
```bash
pytest tests/ -v
```
