# AV-ECHO

**Representation Steering for Cross-Modal Hallucination Mitigation in Audio-Visual Large Language Models**

Jaeho Han and Junyeong Kim — Chung-Ang University

---

## Overview

Audio-visual large language models (AV-LLMs) often hallucinate by grounding
predictions in irrelevant cross-modal cues. Existing training-free remedies
correct output logits using contrastive or modality-ablated inputs, but this
late-stage intervention discards rich evidence already present in intermediate
representations.

**AV-ECHO** (Audio-Visual Evidence-Contrastive Head Output Steering) instead
steers the activations of grounding-sensitive attention heads during decoding,
using contrastive directions derived from intact and disrupted audio-visual
inputs. It updates no model parameters and adds no extra forward pass at
inference time.

## How it works

AV-ECHO runs in two stages:

1. **Offline profiling.** For each sample, build a *faithful* condition (original
   input) and a *disrupted* condition (the question-relevant modality replaced by
   an unrelated sample). Pass both through the frozen AV-LLM and collect
   per-head activations at the answer position. The mean difference between
   confident-correct (faithful) and confident-wrong (disrupted) activations is
   stored as a **restoration vector**. Heads are selected in two steps: keep
   heads whose positive/negative activations are linearly separable, then prefer
   heads with a large angle between the audio and visual restoration vectors
   (top-12 per backbone).

2. **Online steering.** At inference, add the precomputed restoration vector to
   the selected head outputs before computing logits. Evidence is corrected in
   representation space, before it is compressed into token probabilities.

## Results

Evaluated on two backbones (VideoLLaMA2-AV, Qwen2.5-Omni-7B) against two
hallucination benchmarks (CMM, AVHBench).

| Backbone         | Method  | CMM Overall | AVHBench Overall | Cost   |
|------------------|---------|:-----------:|:----------------:|:------:|
| VideoLLaMA2-AV   | Base    | 75.9        | 77.4             | 1.00×  |
| VideoLLaMA2-AV   | AVCD    | 77.9        | 79.3             | 2.48×  |
| VideoLLaMA2-AV   | MAD     | 83.3        | 79.4             | 2.33×  |
| VideoLLaMA2-AV   | **AV-ECHO** | **84.8** | **79.9**        | **1.05×** |
| Qwen2.5-Omni-7B  | Base    | 68.4        | 76.9             | 1.00×  |
| Qwen2.5-Omni-7B  | AVCD    | 69.6        | 77.8             | 2.38×  |
| Qwen2.5-Omni-7B  | MAD     | 80.6        | 81.6             | 3.56×  |
| Qwen2.5-Omni-7B  | **AV-ECHO** | **82.3** | **83.9**        | **1.07×** |

## Repository layout

```
av_echo/
  profiling.py    # offline: collect head activations, build restoration vectors
  steering.py     # online: add restoration vectors to selected head outputs
scripts/
  run.sh          # end-to-end entry point
```

> Note: this repository provides a reference outline of the method. Backbone
> weights, benchmark data, and full evaluation harness are not included.

## Citation

```bibtex
@inproceedings{han2026avecho,
  title     = {AV-ECHO: Representation Steering for Cross-Modal Hallucination
               Mitigation in Audio-Visual Large Language Models},
  author    = {Han, Jaeho and Kim, Junyeong},
  year      = {2026}
}
```

## Acknowledgments

This work was partly supported by the IITP grant funded by the Korea government
(MSIT) [RS-2021-II211341, AI Graduate School Program (Chung-Ang University)] and
the NRF grant [RS-2026-25498346].

## License

MIT
