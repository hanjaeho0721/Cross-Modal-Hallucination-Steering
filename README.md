# Cross-Modal-Hallucination-Steering

A minimal implementation of representation steering for reducing cross-modal hallucination in audio-visual large language models.

## Overview

This repository provides a lightweight training-free pipeline that profiles audio-visual evidence contrastively and steers selected attention-head outputs during inference. The goal is to reduce hallucination caused by irrelevant audio or visual cues while keeping inference overhead low.

## Structure

```text
av_echo/
  profiling.py    # builds steering directions from intact/disrupted AV inputs
  steering.py     # applies head-output steering during inference
scripts/
  run.sh          # example script
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
