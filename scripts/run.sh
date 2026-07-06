#!/usr/bin/env bash
# End-to-end entry point: offline profiling then steered evaluation.
set -euo pipefail

BACKBONE=${1:-qwen2.5-omni-7b}   # or videollama2-av
PROFILE=profiles/${BACKBONE}.pt

# 1. Offline: build restoration vectors on AVQA samples.
python -m av_echo.profiling --backbone "$BACKBONE" --out "$PROFILE"

# 2. Online: evaluate with steering on CMM / AVHBench.
python -m av_echo.steering --backbone "$BACKBONE" --profile "$PROFILE" --benchmark cmm
python -m av_echo.steering --backbone "$BACKBONE" --profile "$PROFILE" --benchmark avhbench
