# PersonalComment & DiffuPercom

Official dataset and experimental code for:

> Jiamiao Liu, Pengsen Cheng, Jinqiao Dai, and Jiayong Liu. **DiffuPercom: A
> novel simplex-based diffusion framework for personalized comment
> generation.** *Applied Soft Computing*, 201:115493, 2026.
> [https://doi.org/10.1016/j.asoc.2026.115493](https://doi.org/10.1016/j.asoc.2026.115493)

PersonalComment contains Chinese news articles, comments, and user attributes.
DiffuPercom generates personalized comments in logit-simplex space using
separate article and persona encoders, a denoising decoder, and Personalized
Fusion Attention (PFA). This repository now keeps the dataset entry point,
data examples, model code, training configuration, and evaluation protocol in
one place.

## Repository contents

- `data_demo/data_demo.json`: 1,000 example articles and 8,833 comments in the
  original nested dataset format.
- `data/example/`: two synthetic records in the model-ready split format.
- `scripts/prepare_personalcomment.py`: converts the nested format into the
  `articles.data` and `comments.data` files consumed by the model.
- `sdlm/`: diffusion model, personalized encoders, scheduler, trainer, and
  metrics.
- `classifier/`: sentiment and demographic classifier experiments.
- `configs/`: portable training, evaluation, and classifier configurations.
- `docs/REPRODUCIBILITY.md`: paper settings, dataset statistics, metrics, and
  reported results.
- `docs/BASELINES.md`: comparison methods and upstream repositories.
- `CITATION.cff` and `CITATION.bib`: machine-readable citation metadata.

The release is source-only. It does not redistribute model checkpoints,
experiment logs, generated outputs, or third-party baseline repositories.

## Dataset

The full PersonalComment dataset can be obtained from the
[Google Drive folder](https://drive.google.com/drive/folders/1wmjgPAPbkmoc7FB4GDi9_oAwxMPDQMg4?usp=drive_link).
It was collected from Sina News between November 2021 and August 2023 and
contains the following reported splits:

| Item | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Articles | 67,597 | 65,597 | 1,000 | 1,000 |
| Comments | 580,748 | 564,191 | 8,833 | 7,724 |
| Personalized attributes | 580,748 | 564,191 | 8,833 | 7,724 |

The records include article text, comments, age, gender, location, profile
description, and automatically assigned sentiment polarity. See
[`data/README.md`](data/README.md) for the exact schemas and privacy notes.

### Prepare model-ready splits

The training code reads one directory per split, with `articles.data` and
`comments.data` inside. If a downloaded split uses the same nested format as
`data_demo/data_demo.json`, convert it with:

```bash
python scripts/prepare_personalcomment.py \
  data_demo/data_demo.json \
  data/valid
```

Repeat the command for the train and test source files, changing the output
directory accordingly. The resulting layout is:

```text
data/
  train/articles.data
  train/comments.data
  valid/articles.data
  valid/comments.data
  test/articles.data
  test/comments.data
```

Raw and converted dataset splits are ignored by Git. Review the dataset's
license, access terms, and privacy requirements before redistribution.

## Environment

The experimental snapshot was developed with Python 3.8, PyTorch 1.12.1 with
CUDA 11.3, and Transformers 4.27.1. A GPU is strongly recommended.

```bash
conda create -n diffupercom python=3.8 -y
conda activate diffupercom

# Select a PyTorch build matching the CUDA version on your host if needed.
pip install torch==1.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
```

Set GPU visibility outside the program:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Weights & Biases defaults to offline mode. Set `WANDB_MODE=online` before a run
if online logging is desired.

## Train DiffuPercom

`configs/diffupercom_train.json` contains the paper configuration: two
12-layer encoders and one 12-layer decoder, hidden size 768, 12 attention
heads, effective batch size 16 on two GPUs, 2,000,000 optimization steps,
5,000 training diffusion steps, 2,500 sampling steps, and simplex value
`K=10`.

For the reported two-GPU setup:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m sdlm.run_summarization configs/diffupercom_train.json
```

For one process, `bash scripts/train.sh` invokes the same entry point. The
default model is `hfl/chinese-roberta-wwm-ext`, and checkpoints are written
below `outputs/`.

## Evaluate a checkpoint

Set `model_name_or_path` in `configs/diffupercom_eval.json`, then run:

```bash
bash scripts/evaluate.sh
```

The checkpoint directory must include the model, tokenizer, and `config.json`.
If `trainer_state.json` is present, the runner restores it; otherwise
evaluation continues without trainer-state restoration.

### Classifier guidance

Train the sentiment classifier with:

```bash
python classifier/sentiment/classifier_train.py \
  --mode train \
  --config configs/classifier/sentiment.json
```

Then enable `is_classifier_guidance` and set `ctr_model_name`,
`ctr_opt_label_idx`, and `decode_ctr_lr` in the DiffuPercom configuration.

### Text metrics

`evaluation.py` expects a JSON list whose entries contain `article`,
`ground_truth_sentence`, and `generated_sentence`:

```bash
python evaluation.py path/to/generation_results.json \
  --skip-attribute-classifiers
```

Metric models are downloaded on first use and can require substantial GPU
memory. Attribute accuracy additionally requires demographic classifier
checkpoints, which are not redistributed. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact protocol.

## License and provenance

This repository is distributed under the Apache License 2.0; see `LICENSE`.
Parts of the implementation derive from the Simplex Diffusion Language Model
codebase, and the customized RoBERTa module derives from Hugging Face
Transformers 4.27.1. Retained attributions are documented in `NOTICE`.

## Citation

```bibtex
@article{liu2026diffupercom,
  title   = {DiffuPercom: A novel simplex-based diffusion framework for personalized comment generation},
  author  = {Liu, Jiamiao and Cheng, Pengsen and Dai, Jinqiao and Liu, Jiayong},
  journal = {Applied Soft Computing},
  volume  = {201},
  pages   = {115493},
  year    = {2026},
  doi     = {10.1016/j.asoc.2026.115493}
}
```
