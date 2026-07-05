"""RSSM observe/imagine shape correctness. Uses vendor RSSM directly."""
import torch

from isaac_rl.dreamer.vendor import networks


def _mk_rssm(num_actions: int = 14, embed: int = 128, stoch: int = 8, discrete: int = 8, deter: int = 64):
    return networks.RSSM(
        stoch=stoch, deter=deter, hidden=64, rec_depth=1, discrete=discrete,
        act="SiLU", norm=True, mean_act="none", std_act="softplus",
        min_std=0.1, unimix_ratio=0.01, initial="zeros",
        num_actions=num_actions, embed=embed, device="cpu",
    )


def test_rssm_initial_state_shapes():
    rssm = _mk_rssm()
    st = rssm.initial(batch_size=3)
    assert st["deter"].shape == (3, 64)
    assert st["stoch"].shape == (3, 8, 8)
    assert st["logit"].shape == (3, 8, 8)


def test_rssm_observe_shape():
    rssm = _mk_rssm()
    B, T = 2, 5
    embed = torch.randn(B, T, 128)
    action = torch.zeros(B, T, 14)
    action[..., 0] = 1.0
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    post, prior = rssm.observe(embed, action, is_first)
    assert post["deter"].shape == (B, T, 64)
    assert post["stoch"].shape == (B, T, 8, 8)
    assert prior["deter"].shape == (B, T, 64)


def test_rssm_imagine_shape():
    rssm = _mk_rssm()
    state = rssm.initial(batch_size=4)
    action = torch.zeros(3, 4, 14)   # [H=3, B=4, num_actions]
    action[..., 0] = 1.0
    prior = rssm.imagine_with_action(action.permute(1, 0, 2), state)   # imagine_with_action wants [B, H, ...]
    assert prior["deter"].shape == (4, 3, 64)
    assert prior["stoch"].shape == (4, 3, 8, 8)


def test_rssm_kl_loss_shape_and_finite():
    rssm = _mk_rssm()
    B, T = 2, 4
    embed = torch.randn(B, T, 128)
    action = torch.zeros(B, T, 14)
    action[..., 0] = 1.0
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    post, prior = rssm.observe(embed, action, is_first)
    kl_loss, kl_value, dyn_loss, rep_loss = rssm.kl_loss(post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1)
    assert kl_loss.shape == (B, T)
    assert torch.isfinite(kl_loss).all()
    assert torch.isfinite(kl_value).all()


def test_rssm_gradient_flows():
    rssm = _mk_rssm()
    B, T = 2, 4
    embed = torch.randn(B, T, 128, requires_grad=True)
    action = torch.zeros(B, T, 14)
    action[..., 0] = 1.0
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    post, prior = rssm.observe(embed, action, is_first)
    kl_loss, _, _, _ = rssm.kl_loss(post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1)
    kl_loss.mean().backward()
    assert embed.grad is not None
    got_rssm_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in rssm.parameters())
    assert got_rssm_grad
