# Vendored: NM512/dreamerv3-torch

Source: https://github.com/NM512/dreamerv3-torch
Commit: 6ef8646d807cd10ce0c88e10a7e943211e7fc44c
License: MIT (see LICENSE in this folder)

## Files

- `LICENSE` — original MIT license, preserved verbatim.
- `tools.py` — DiscDist (255-bin twohot), OneHotDist, symlog/symexp, KL utilities, Optimizer,
  `static_scan`, `lambda_return`, `weight_init`, etc. Used verbatim except for the top-level
  `import tools` and `import networks` statements — these are rewritten to relative imports
  by our patch.
- `networks.py` — RSSM, MLP, MultiEncoder, MultiDecoder, GRUCell, ConvEncoder/Decoder.
  We use RSSM and MLP directly. `MultiEncoder`/`MultiDecoder` are NOT used — replaced by our
  `IsaacObsEncoder` / `IsaacObsDecoder` which speak our exact obs schema.
- `models.py` — `WorldModel` and `ImagBehavior`. Both subclassed in `dreamer/isaac_models.py`
  to swap in our encoder/decoder and MultiDiscrete actor.

## What we patched vs used as-is

**Used as-is:**
- `tools.DiscDist` — 255-bin twohot over symlog([-20, 20])
- `tools.OneHotDist` — categorical with unimix
- `tools.symlog` / `tools.symexp`
- `tools.static_scan` — used inside RSSM.observe / imagine
- `tools.lambda_return` — used by ImagBehavior for λ-returns
- `tools.Optimizer` — AMP-aware optimizer wrapper
- `tools.weight_init` / `tools.uniform_weight_init`
- `networks.RSSM` (with `num_actions=14` for our one-hot MultiDiscrete)
- `networks.MLP` (with `dist="symlog_disc"` for reward, `dist="binary"` for cont)
- `networks.GRUCell`

**Replaced (do not use):**
- `networks.MultiEncoder` → replaced by `dreamer/encoder.py::IsaacObsEncoder`
- `networks.MultiDecoder` → replaced by `dreamer/decoder.py::IsaacObsDecoder`
- `networks.ConvEncoder` / `ConvDecoder` — Isaac uses no pixel obs
- `models.WorldModel` — subclassed in `dreamer/isaac_models.py::IsaacWorldModel`
- `models.ImagBehavior` — subclassed in `dreamer/isaac_models.py::IsaacImagBehavior`
  to use our MultiDiscreteActionHead

## Import rewrite

The vendor files use bare `import tools` / `import networks`. We convert these to
relative imports (`from . import tools` / `from . import networks`) at vendor-time
so the package works when installed inside `isaac_rl.dreamer.vendor`.
