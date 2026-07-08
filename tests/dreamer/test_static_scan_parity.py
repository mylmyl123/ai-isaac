"""Parity tests for the rewritten `static_scan`.

The vendor `static_scan` was O(T^2) due to repeated `torch.cat` in a Python
loop. Replaced by a preallocated list + single `torch.stack`. This test
guards that the new version produces bit-identical forward outputs AND
bit-identical gradients as the legacy version, across the shapes the RSSM
uses (dict-of-dict for observe, dict for img_step).
"""
from __future__ import annotations

import pytest
import torch

from isaac_rl.dreamer.vendor.tools import static_scan, _static_scan_legacy


def _tiny_step_fn_dict(prev_state, action, embed):
    """Toy 'obs_step': returns a dict state that mixes action + embed + prev."""
    return {
        "deter": torch.tanh(prev_state["deter"] + action + embed),
        "stoch": torch.sigmoid(prev_state["stoch"] * 0.9 + embed),
    }


def _tiny_step_fn_tuple(prev_state, action, embed):
    """Toy 'observe': returns (post, prior) each a dict — mirrors RSSM."""
    prev_post, prev_prior = prev_state
    post = {
        "deter": torch.tanh(prev_post["deter"] + action + embed),
        "logit": prev_post["logit"] * 0.5 + embed[..., None],
    }
    prior = {
        "deter": torch.tanh(prev_prior["deter"] + action),
        "logit": prev_prior["logit"] * 0.3 + action[..., None],
    }
    return post, prior


@pytest.mark.parametrize("T,B,D", [(4, 3, 5), (16, 2, 8), (32, 4, 6)])
def test_dict_mode_parity_forward(T, B, D):
    torch.manual_seed(0)
    action = torch.randn(T, B, D)
    embed = torch.randn(T, B, D)
    state = {"deter": torch.zeros(B, D), "stoch": torch.zeros(B, D)}

    out_new = static_scan(_tiny_step_fn_dict, (action, embed), state)
    out_old = _static_scan_legacy(_tiny_step_fn_dict, (action, embed), state)

    # Both return [dict] for dict-input.
    assert isinstance(out_new, list) and len(out_new) == 1
    assert isinstance(out_old, list) and len(out_old) == 1
    for k in out_new[0]:
        assert out_new[0][k].shape == out_old[0][k].shape == (T, B, D)
        assert torch.allclose(out_new[0][k], out_old[0][k], atol=0, rtol=0), (
            f"forward mismatch on key {k}"
        )


@pytest.mark.parametrize("T", [4, 16, 32])
def test_tuple_mode_parity_forward(T):
    B, D = 3, 5
    torch.manual_seed(1)
    action = torch.randn(T, B, D)
    embed = torch.randn(T, B, D)
    post0 = {"deter": torch.zeros(B, D), "logit": torch.zeros(B, D, 1)}
    prior0 = {"deter": torch.zeros(B, D), "logit": torch.zeros(B, D, 1)}
    state = (post0, prior0)

    out_new = static_scan(_tiny_step_fn_tuple, (action, embed), state)
    out_old = _static_scan_legacy(_tiny_step_fn_tuple, (action, embed), state)

    # Both return list-of-dict for tuple-input.
    assert isinstance(out_new, list) and len(out_new) == 2
    assert isinstance(out_old, list) and len(out_old) == 2
    for j in range(2):
        for k in out_new[j]:
            assert torch.allclose(out_new[j][k], out_old[j][k], atol=0, rtol=0), (
                f"forward mismatch on tuple[{j}][{k}]"
            )


def test_dict_mode_grad_parity():
    """Gradients through the scan must be identical (autograd graph equiv)."""
    T, B, D = 8, 2, 4
    torch.manual_seed(2)
    action_a = torch.randn(T, B, D, requires_grad=True)
    embed_a = torch.randn(T, B, D, requires_grad=True)
    action_b = action_a.detach().clone().requires_grad_(True)
    embed_b = embed_a.detach().clone().requires_grad_(True)

    state_a = {"deter": torch.zeros(B, D), "stoch": torch.zeros(B, D)}
    state_b = {"deter": torch.zeros(B, D), "stoch": torch.zeros(B, D)}

    out_new = static_scan(_tiny_step_fn_dict, (action_a, embed_a), state_a)[0]
    out_old = _static_scan_legacy(_tiny_step_fn_dict, (action_b, embed_b), state_b)[0]

    loss_new = sum(v.sum() for v in out_new.values())
    loss_old = sum(v.sum() for v in out_old.values())
    loss_new.backward()
    loss_old.backward()

    assert torch.allclose(action_a.grad, action_b.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(embed_a.grad, embed_b.grad, atol=1e-6, rtol=1e-6)


def test_empty_input_raises():
    action = torch.zeros(0, 2, 3)
    embed = torch.zeros(0, 2, 3)
    state = {"deter": torch.zeros(2, 3), "stoch": torch.zeros(2, 3)}
    with pytest.raises(ValueError, match="T=0"):
        static_scan(_tiny_step_fn_dict, (action, embed), state)


def test_none_placeholder_start():
    """RSSM's observe(state=None) passes (None, None) as start.

    The scan must not use the None placeholders to shape its buffers —
    buffer allocation MUST be lazy (based on fn's first return), not eager.
    Regression guard for the 2026-07-07 bug where buffers were built from
    `start` shape and mismatched fn's dict return.
    """
    T, B, D = 5, 2, 4

    def obs_step_like(prev, act, emb):
        # First-call path: prev is (None, None); ignore it and emit fresh dicts.
        # Subsequent calls: prev is (post, prior) — use them.
        if prev[0] is None:
            post = {"deter": act + emb, "logit": act.unsqueeze(-1) * 0.0}
            prior = {"deter": act.clone(), "logit": emb.unsqueeze(-1) * 0.0}
        else:
            p_post, p_prior = prev
            post = {"deter": p_post["deter"] + act + emb, "logit": p_post["logit"] * 0.9}
            prior = {"deter": p_prior["deter"] + act, "logit": p_prior["logit"] * 0.9}
        return post, prior

    action = torch.randn(T, B, D)
    embed = torch.randn(T, B, D)

    out_new = static_scan(obs_step_like, (action, embed), (None, None))
    out_old = _static_scan_legacy(obs_step_like, (action, embed), (None, None))

    assert isinstance(out_new, list) and len(out_new) == 2
    for j in range(2):
        for k in out_new[j]:
            assert torch.allclose(out_new[j][k], out_old[j][k], atol=0, rtol=0)
