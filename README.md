# AV-ECHO

**Representation Steering for Cross-Modal Hallucination Mitigation in Audio-Visual Large Language Models**

This repository provides a minimal reference implementation of AV-ECHO, a training-free representation steering method for mitigating cross-modal hallucination in audio-visual large language models.

## Overview

AV-ECHO steers selected attention-head outputs using evidence-contrastive directions computed from intact and disrupted audio-visual inputs. Instead of correcting final logits, it intervenes in the internal representation space before token probabilities are produced.

## Repository Structure

```text
av_echo/
  profiling.py    # build restoration vectors from contrastive AV inputs
  steering.py     # apply restoration vectors during inference
scripts/
  run.sh          # example entry point
requirements.txt
LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
bash scripts/run.sh qwen2.5-omni-7b
```

or

```bash
bash scripts/run.sh videollama2-av
```

## License

This project is released under the MIT License.

## Citation

```bibtex
@inproceedings{han2026avecho,
  title  = {AV-ECHO: Representation Steering for Cross-Modal Hallucination Mitigation in Audio-Visual Large Language Models},
  author = {Han, Jaeho and Kim, Junyeong},
  year   = {2026}
}
```
