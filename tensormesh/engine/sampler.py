from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> int:
    """Pick the next token id from a [vocab] logits vector for the last position."""
    if params.temperature == 0.0:
        return int(torch.argmax(logits))

    logits = logits.float() / params.temperature
    if 0 < params.top_k < logits.numel():
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        # Drop a token once the mass *before* it already reaches top_p, so the
        # token that crosses the threshold stays in the nucleus.
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs[mass_before >= params.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    # multinomial renormalises, so masked-out mass needn't be redistributed by hand.
    return int(torch.multinomial(probs, 1, generator=generator))
