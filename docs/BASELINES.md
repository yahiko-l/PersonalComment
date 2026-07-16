# Baseline implementations

Third-party baseline source is deliberately not vendored into this release.
The paper compares DiffuPercom with the following systems. Use the original
repositories and review each project's license before reproducing comparisons.

| Baseline | Paradigm | Upstream named in the paper |
|---|---|---|
| PCGN | autoregressive personalized comment generation | [Walleclipse/AGPC](https://github.com/Walleclipse/AGPC) |
| Seq2seqEmb | autoregressive persona-based conversation | [sourav-ranjan/Neural-Persona-based-Conversation-Model-Python-Version](https://github.com/sourav-ranjan/Neural-Persona-based-Conversation-Model-Python-Version) |
| PersonalityTrait | autoregressive profile-aware generation | [ghosthamlet/persona](https://github.com/ghosthamlet/persona) |
| PersonaWAE | autoregressive latent-variable response generation | Refer to the paper's citation [18] |
| Convai2 | autoregressive PersonaChat system | [facebookresearch/ParlAI, convai2archive](https://github.com/facebookresearch/ParlAI/tree/convai2archive/projects/convai2) |
| AttentionRouting | autoregressive persona-sparse generation | [ghosthamlet/persona](https://github.com/ghosthamlet/persona) |
| AttentionRoutingPlus | improved AttentionRouting | [ghosthamlet/persona](https://github.com/ghosthamlet/persona) |
| SeqDiffuSeq | iterative non-autoregressive diffusion | [Yuanhy1997/SeqDiffuSeq](https://github.com/Yuanhy1997/SeqDiffuSeq) |
| DiffuSeq | iterative non-autoregressive diffusion | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) |

For a fair comparison, evaluate generated files with the same PersonalComment
test split, preprocessing, sampling count, and metric checkpoints. The paper
uses greedy decoding for autoregressive baselines and iterative denoising for
the diffusion systems.
