# Progress

## Status
Research subagent (recent papers) — Complete

## Tasks
- [x] Category 1: Roguelike/procedural RL (Craftax, MiniHack, Procgen)
- [x] Category 2: Sample-efficient RL (DreamerV3, EfficientZeroV2, DIAMOND, PPO+Dreamer tricks)
- [x] Category 3: Sparse-reward exploration (E3B, DRND, DuRND, ELLM, ExploRLLM, GCRL)
- [x] Category 4: Auxiliary tasks (SPR unifying, self-prediction, successor features)
- [x] Category 5: Reward shaping (PBIM, PBRS sample efficiency, stationary IM)
- [x] Category 6: PPO improvements (PPG Reloaded, Transformer/LSTM PPO, POMDP theory)
- [x] Category 7: Video game agents (D2AH-PPO ViZDoom, Open-P2P, diffusion policies, Isaac hobbyist repos)
- [x] Category 8: BC + RL (VPT, C-GAIL, DPAIL)
- [x] Top-5 recommendations

## Files Changed
- swarm-outputs/02-recent-papers.md (new, ~25 papers surveyed, top-5 ranked)

## Notes
- No peer-reviewed RL work on Binding of Isaac exists — hobbyist repos only.
- Craftax (Matthews 2024) is the closest published analog; use their PPO+ICM+E3B baseline as reference.
- Top recommendations optimized for engineering-hours-to-impact: DreamerV3 tricks → BC pretrain → LSTM+AchievementDistill → PPO+ICM+E3B → PBIM shaping.
