"""Offline profiling: build per-head restoration vectors.

For each sample we form two conditions:
  - faithful:  original audio-visual input
  - disrupted: the question-relevant modality replaced by an unrelated sample

We collect per-head activations at the answer position under both conditions and
store the mean difference (confident-correct minus confident-wrong) as the
restoration vector for that head.
"""

from dataclasses import dataclass

import torch


@dataclass
class ProfilingConfig:
    top_k_heads: int = 12          # heads kept per backbone
    separability_thresh: float = 0.0  # min linear-separability margin to keep a head
    answer_position: int = -1      # token position used for scoring


def collect_head_activations(model, sample, condition):
    """Run the frozen model on `sample` under `condition` and return
    per-head activations at the answer position.

    Returns a tensor of shape (num_layers, num_heads, head_dim).
    """
    raise NotImplementedError("Hook the backbone's attention heads here.")


def build_restoration_vectors(faithful_acts, disrupted_acts):
    """Mean difference between faithful (correct) and disrupted (wrong)
    activations, per head."""
    return faithful_acts.mean(0) - disrupted_acts.mean(0)


def select_heads(faithful_acts, disrupted_acts, cfg: ProfilingConfig):
    """Two-step head selection:
      1. keep heads whose positive/negative activations are linearly separable
      2. among those, prefer heads with a large angle between the audio and
         visual restoration vectors; keep the top-k.
    """
    raise NotImplementedError("Implement separability filter + angle ranking.")


def run_profiling(model, dataset, cfg: ProfilingConfig = ProfilingConfig()):
    faithful, disrupted = [], []
    for sample in dataset:
        faithful.append(collect_head_activations(model, sample, "faithful"))
        disrupted.append(collect_head_activations(model, sample, "disrupted"))
    faithful = torch.stack(faithful)
    disrupted = torch.stack(disrupted)

    heads = select_heads(faithful, disrupted, cfg)
    vectors = build_restoration_vectors(faithful, disrupted)
    return {"heads": heads, "restoration_vectors": vectors}
