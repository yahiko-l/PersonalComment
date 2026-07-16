# Paper reproducibility guide

This guide maps the published experimental protocol to the released code and
configuration. The authoritative paper is:

Jiamiao Liu, Pengsen Cheng, Jinqiao Dai, and Jiayong Liu. “DiffuPercom:
A novel simplex-based diffusion framework for personalized comment
generation.” *Applied Soft Computing* 201 (2026), article 115493.
[doi:10.1016/j.asoc.2026.115493](https://doi.org/10.1016/j.asoc.2026.115493).

## Paper configuration

| Paper setting | Released configuration |
|---|---|
| Two NVIDIA GeForce RTX 4090 GPUs | `torchrun --nproc_per_node=2` |
| Two encoders and one decoder | `RobertaForDiffusionSeq2seq` |
| 12 layers per encoder/decoder | Chinese RoBERTa base configuration |
| Hidden size 768; 12 attention heads | Chinese RoBERTa base configuration |
| Character vocabulary size 21,128 | `hfl/chinese-roberta-wwm-ext` tokenizer |
| Article/source maximum 512 | `max_article_len=510`, retaining the 512-position budget for special-token handling |
| Comment/target maximum 64 | `max_comment_len=64` |
| Persona maximum 64 | `max_persona_len=64` |
| Simplex value `K=10` | `simplex_value=10.0` |
| Training diffusion steps 5,000 | `num_diffusion_steps=5000` |
| Sampling steps 2,500 | `num_inference_diffusion_steps=2500` |
| Total batch size 16 | per-device batch 8 × 2 processes |
| Training steps 2,000,000 | `max_steps=2000000` |
| AdamW, learning rate `1e-5` | `learning_rate=1e-5` |
| Cosine learning-rate schedule | `lr_scheduler_type=cosine` |
| Learning-rate warmup | `warmup_steps=2000` in the released configuration |
| Cosine diffusion-noise schedule | `beta_schedule=squaredcos_improved_ddpm` |
| Self-conditioning | `self_condition=logits_addition` |
| Greedy logit-simplex projection | `sampling_type=argmax` |

The paper reports that some decoder parameters are initialized uniformly in
`[-0.02, 0.02]`, while the article and persona encoders are initialized from
Chinese RoBERTa. Parameters such as logging frequency, checkpoint frequency,
and the number of retained checkpoints are operational choices and do not
change the paper's model definition.

## Dataset

The paper reports the following PersonalComment split:

| Item | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Articles | 67,597 | 65,597 | 1,000 | 1,000 |
| Comments | 580,748 | 564,191 | 8,833 | 7,724 |
| Personalized attributes | 580,748 | 564,191 | 8,833 | 7,724 |

The dataset repository named in the paper is
[yahiko-l/PersonalComment](https://github.com/yahiko-l/PersonalComment). The
dataset is not embedded in this source archive.

## Training and evaluation

Install the environment as described in the root README. Review data paths in
the JSON configuration, then launch the two-GPU paper setup:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m sdlm.run_summarization configs/diffupercom_train.json
```

For evaluation, update `model_name_or_path` in
`configs/diffupercom_eval.json` and run:

```bash
python -m sdlm.run_summarization configs/diffupercom_eval.json
```

The paper uses greedy logit projection. The released inference utility accepts
`argmax` (or its alias `greedy`) explicitly; `top_p` remains available for
additional non-paper experiments.

## Paper metrics

The automatic evaluation excludes BLEU and ROUGE because personalized comment
generation is open-ended. It reports:

- fluency: perplexity (PPL; lower is conventionally preferred, with the caveat
  discussed in the paper);
- article-comment relevance: SimCSE cosine similarity;
- diversity: Self-BLEU (lower), Dist-1 and Dist-2 (higher);
- personalized controllability: gender, age, and location classifier accuracy.

Gender uses two labels. Age uses four decade groups (post-70s, post-80s,
post-90s, post-00s). Location uses seven regions. The paper reports classifier
accuracies of 76.50%, 67.04%, and 65.16%, respectively.

The standalone `evaluation.py` additionally computes BERTScore and basic length
statistics. Demographic accuracy requires the separately trained classifier
checkpoints.

## Main reported automatic results

| Model | PPL ↓ | SimCSE ↑ | Self-BLEU ↓ | Dist-1 ↑ | Dist-2 ↑ | Gender ↑ | Age ↑ | Location ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference comments | 47.99 | 76.28 | 25.61 | 8.59 | 20.32 | 74.27 | 33.17 | 53.42 |
| PCGN | 9.53 | 76.06 | 96.84 | 1.23 | 1.63 | 70.21 | 31.79 | 52.38 |
| Seq2seqEmb | 21.60 | 76.40 | 71.95 | 2.65 | 5.98 | 71.90 | 32.87 | 52.05 |
| PersonalityTrait | 10.15 | 75.94 | 97.37 | 1.22 | 1.52 | 70.87 | 31.42 | 52.61 |
| PersonaWAE | 31.45 | 71.96 | 98.96 | 1.07 | 0.90 | 72.30 | 32.82 | 52.66 |
| Convai2 | 10.75 | 75.65 | 92.65 | 1.26 | 1.61 | 71.71 | 33.05 | 51.59 |
| AttentionRouting | 19.78 | 77.24 | 51.32 | 3.25 | 7.99 | 73.12 | 33.10 | 50.02 |
| AttentionRoutingPlus | 19.31 | 77.08 | 54.35 | 3.05 | 6.73 | 73.32 | 32.06 | 50.43 |
| DiffuSeq | 61.77 | 75.91 | 48.82 | 2.51 | 7.81 | 71.68 | 31.58 | 47.64 |
| SeqDiffuSeq | 15.68 | 72.56 | 56.14 | 2.18 | 6.06 | 73.40 | 28.13 | 52.38 |
| **DiffuPercom (`K=10`)** | **28.45** | **76.59** | **40.80** | **3.43** | **9.42** | **75.02** | **33.89** | **52.89** |

These are reported paper values, not results regenerated during packaging.

## Human evaluation

Three linguistics professionals rated fluency, relevance, and personality
control on a five-point scale. The paper sampled 50 test articles and generated
three persona-conditioned comments per article, for 150 evaluated samples.
Cohen's kappa exceeded 0.6. DiffuPercom (`K=10`) obtained 3.39 fluency, 2.87
relevance, and 3.82 personality control.

## Ablations

To reproduce the paper's ablation axes, change only the indicated JSON field:

| Study | Values | Configuration field |
|---|---|---|
| Self-conditioning | off / on | `self_condition=null` / `logits_addition` |
| Sampling steps | 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000 | `num_inference_diffusion_steps` |
| Noise schedule | linear, scaled linear, cosine, sigmoid | `beta_schedule`: `linear`, `scaled_linear`, `squaredcos_improved_ddpm`, `sigmoid` |
| Simplex value `K` | 1, 3, 5, 7, 10, 20 | `simplex_value` |

The paper selects 2,500 sampling steps and `K=10` for the main evaluation.
Sampling steps above 2,500 provide limited reported improvement relative to
their additional inference cost.

Generate all of these JSON variants with:

```bash
python scripts/make_ablation_configs.py
```

Sampling-step variants inherit `model_name_or_path` from the evaluation base;
set that checkpoint path before running them. `K`, noise-schedule, and
self-conditioning variants are training configurations.

## Reproducibility boundary

The release validates source syntax, configuration parsing, package imports,
custom RoBERTa construction, and Python wheel construction. Full numerical
reproduction still requires PersonalComment, pretrained weights, the trained
attribute classifiers, two suitable GPUs, and the compute budget for two
million optimization steps.
