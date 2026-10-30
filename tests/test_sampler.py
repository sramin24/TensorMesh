import pytest
import torch

from tensormesh.engine.sampler import SamplingParams, sample


def test_zero_temperature_is_greedy():
    logits = torch.tensor([0.1, 2.0, 1.9, -5.0])

    assert sample(logits, SamplingParams(temperature=0.0)) == 1


def draw(logits, params, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.tensor([sample(logits, params, g) for _ in range(n)])


def test_temperature_one_samples_from_the_softmax_distribution():
    logits = torch.log(torch.tensor([0.25, 0.75]))

    tokens = draw(logits, SamplingParams(temperature=1.0))

    assert (tokens == 1).float().mean().item() == pytest.approx(0.75, abs=0.03)


def test_low_temperature_sharpens_the_distribution():
    # p ∝ [0.25, 0.75]^(1/0.5) = [0.0625, 0.5625] -> P(token 1) = 0.9
    logits = torch.log(torch.tensor([0.25, 0.75]))

    tokens = draw(logits, SamplingParams(temperature=0.5))

    assert (tokens == 1).float().mean().item() == pytest.approx(0.9, abs=0.03)


def test_top_k_only_draws_from_the_k_most_likely_tokens():
    logits = torch.tensor([1.0, 3.0, 0.5, 2.9, 2.0])

    tokens = draw(logits, SamplingParams(temperature=1.0, top_k=2), n=1000)

    assert set(tokens.tolist()) == {1, 3}


def test_top_p_keeps_the_smallest_nucleus_reaching_p_and_renormalises():
    # Cumulative mass: 0.5, 0.8, 0.95, 1.0 -> top_p=0.7 needs tokens {0, 1}.
    # Renormalised: P(0) = 0.5 / 0.8 = 0.625.
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))

    tokens = draw(logits, SamplingParams(temperature=1.0, top_p=0.7))

    assert set(tokens.tolist()) == {0, 1}
    assert (tokens == 0).float().mean().item() == pytest.approx(0.625, abs=0.03)


@pytest.mark.parametrize("kwargs", [{"temperature": -0.1}, {"top_k": -1}, {"top_p": 0.0}, {"top_p": 1.5}])
def test_invalid_params_are_rejected_up_front(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
