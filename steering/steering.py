"""Online steering: add precomputed restoration vectors to selected heads.

At inference the selected head outputs are shifted by their restoration vectors
before logits are computed. No parameters are updated and no extra forward pass
is needed.
"""

import torch


class AVEchoSteerer:
    def __init__(self, selected_heads, restoration_vectors, alpha: float = 1.0):
        self.selected_heads = selected_heads          # list of (layer, head)
        self.restoration_vectors = restoration_vectors  # (num_layers, num_heads, head_dim)
        self.alpha = alpha
        self._handles = []

    def _make_hook(self, layer):
        heads = [h for (l, h) in self.selected_heads if l == layer]
        vecs = self.restoration_vectors[layer]

        def hook(module, inputs, output):
            # output: (batch, seq, num_heads, head_dim) at the attention-head level
            for h in heads:
                output[..., h, :] += self.alpha * vecs[h].to(output.dtype)
            return output

        return hook

    def attach(self, model):
        """Register forward hooks on the attention modules of `model`."""
        raise NotImplementedError("Register per-layer hooks on the backbone.")

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def load_profile(path):
    obj = torch.load(path)
    return AVEchoSteerer(obj["heads"], obj["restoration_vectors"])
