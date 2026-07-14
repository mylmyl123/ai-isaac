# Recent RL Papers Relevant to Isaac RL (2024–2026)

Context: PPO+MLP agent training on The Binding of Isaac. The trivial 1-fly sealed-room task collapses into a corner-camping policy (kill=+1, death=-1, step=-0.001). We are looking for algorithmic, architectural, and reward-shaping fixes with evidence in 2024-2026 literature.

---

## Category 1: Roguelike / procedural RL

### Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning — Matthews et al., ICML 2024
- **Technique**: JAX rewrite of Crafter (Craftax-Classic) + a much harder "Craftax" env with dungeons, ranged combat, hunger. Baselines: PPO, PPO-RNN, PPO+ICM, PPO+RND, PPO+E3B. Also tests Dreamer.
- **Result**: PPO alone hits ~90% of optimal on Craftax-Classic in 1B steps / <1h single-GPU. On full Craftax, plain PPO gets ~2.3% score; adding **E3B (elliptical episodic bonus) + ICM** roughly doubles it. PPO-RNN materially beats MLP PPO. Dreamer underperforms in walltime.
- **Apply to Isaac**: This is the closest published analog to our setting — procedural top-down grid, sparse achievement-like rewards, PPO baselines. Their tuned intrinsic-reward recipe (**PPO + ICM + E3B**) is a plug-and-play upgrade over vanilla PPO+MLP. Also strong evidence to switch MLP → recurrent trunk. [arXiv 2402.16801](https://arxiv.org/abs/2402.16801) · [baselines repo](https://github.com/MichaelTMatthews/Craftax_Baselines)

### Discovering Hierarchical Achievements in RL via Contrastive Learning ("Achievement Distillation") — Moon et al., NeurIPS 2023 (still SOTA on Crafter as of 2025)
- **Technique**: PPO backbone + an auxiliary contrastive loss that pulls together representations of trajectories that reach the same achievement and pushes apart those that reach different ones. No world model, no explicit hierarchy.
- **Result**: Beats DreamerV3 on Crafter with 1M steps; PPO-alone was already found to outperform DreamerV3 at 1M steps in their study.
- **Apply to Isaac**: In Isaac, "achievements" map naturally to events (first hit, first kill, door opened, item picked up). An achievement-distillation auxiliary loss over these sub-events would provide dense representation-level signal *without* changing the reward, sidestepping the corner-camping incentive. [arXiv 2307.03486](https://arxiv.org/abs/2307.03486) · [code](https://github.com/snu-mllab/Achievement-Distillation)

### Scalable Option Learning in High-Throughput Environments — Meta AI, arXiv 2509.00338 (Sept 2025)
- **Technique**: Hierarchical option-based agent on MiniHack / NetHack Learning Environment, "Chaotic Dwarven GPT5"-style CNN+LSTM architecture, IMPALA distributed training.
- **Result**: New SOTA on NLE / MiniHack navigation and skill tasks; showed options + LSTM beat flat policies significantly at low sample counts.
- **Apply to Isaac**: A room in Isaac is a natural "option horizon" (enter → clear → exit). Even a two-level policy (meta-action: engage / kite / reposition; low-level: WASD+fire) could break corner-camping by making "engage" an explicit choice. [arXiv 2509.00338](https://arxiv.org/html/2509.00338v1)

### Improving Generalization on ProcGen with Simple Architectural Changes and Scale — Wang et al., arXiv 2410.10905 (2024)
- **Technique**: Vanilla PPO on Procgen, but with (a) frame stacking, (b) 2D convs replaced with **3D convs**, (c) wider conv channels.
- **Result**: Substantial gains on Procgen test-set generalization with no algorithmic changes.
- **Apply to Isaac**: If we move to pixel or grid-tensor observations, this is the cheapest architectural win. Frame stacking especially matters because a single frame doesn't disambiguate velocity of the fly and its bullets. [arXiv 2410.10905](https://arxiv.org/html/2410.10905v1)

### Implicit Curriculum in Procgen Made Explicit — NeurIPS 2024
- **Technique**: Analyzes how multi-level Procgen training naturally forms a curriculum; proposes explicit level-difficulty sampling.
- **Result**: Better sample efficiency by scheduling easier levels first.
- **Apply to Isaac**: We *have* an explicit curriculum knob — number/type of enemies, room size. Their scheduling heuristic (prioritize levels with high learning-progress) transfers directly. [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/24662461d2194d1bc70a47b6b6771026-Paper-Conference.pdf)

---

## Category 2: Sample-efficient RL

### DreamerV3: Mastering Diverse Domains through World Models — Hafner et al., Nature 2025 / arXiv 2301.04104
- **Technique**: Recurrent state-space world model + symlog reward transforms + KL balancing. Single hyperparameter set across 150+ tasks.
- **Result**: SOTA on many benchmarks (Atari100k, Crafter, DMLab, Minecraft-diamond from scratch), single hyperparameter set. But: on Crafter at 1M steps, **PPO can match or beat DreamerV3** (per Achievement-Distillation paper).
- **Apply to Isaac**: Attractive because Isaac has stochastic enemies and Dreamer handles that natively. But wall-clock is worse and Dreamer over-fits when episodes are trivial (early corner-camping wouldn't gather useful world-model transitions). Consider only after fixing exploration. [arXiv 2301.04104](https://arxiv.org/pdf/2301.04104)

### EfficientZero V2 — Wang et al., ICML 2024
- **Technique**: MuZero-family (learned dynamics + MCTS) with search-based value targets, mixed-precision reanalysis, sampled actions for continuous control.
- **Result**: Consistently beats all baselines on Atari100k, DMControl-100k, MiniGrid-sparse with limited data. Notably strong on sparse-reward tasks.
- **Apply to Isaac**: Isaac is discrete-action (9 dirs × fire × 4 dirs), which fits MuZero's tree search well. But engineering cost is high; probably not first move. [arXiv 2403.00564](https://arxiv.org/abs/2403.00564)

### Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) — Alonso et al., NeurIPS 2024
- **Technique**: Uses a diffusion model as the world model (instead of DreamerV3's discrete VAE latent). Preserves fine visual detail.
- **Result**: Beats DreamerV3 on 46/57 Atari100k games; shows visual details captured by diffusion matter for agents that need to spot small enemies/bullets.
- **Apply to Isaac**: The fly is a tiny sprite — discrete latents in DreamerV3 could easily lose it. If we go model-based, DIAMOND is the better bet. [arXiv 2405.12399](https://doi.org/10.48550/arxiv.2405.12399)

### Reward Scale Robustness for PPO via DreamerV3 Tricks — Sullivan et al., 2024 (RLC/OpenReview)
- **Technique**: Ports DreamerV3's symlog reward transform, twohot value head, and normalization into vanilla PPO.
- **Result**: PPO becomes far more robust to arbitrary reward scales; gains on Craftax, Atari, DMC. Almost free.
- **Apply to Isaac**: Directly relevant. Our (+1, -1, -0.001) reward mix already has a scale imbalance; symlog value head + reward normalization would let us add larger shaping terms without value collapse. [OpenReview EY4OHikuBm](https://openreview.net/forum?id=EY4OHikuBm)

---

## Category 3: Sparse-reward exploration

### Exploration via Elliptical Episodic Bonuses (E3B) — Henaff et al., NeurIPS 2022 (still SOTA / used as Craftax baseline in 2024)
- **Technique**: Episodic exploration bonus proportional to Mahalanobis distance in a learned feature space; resets each episode.
- **Result**: On MiniHack, beats RND/ICM/NGU when the state space is huge and stochastic. Critically, **removes the "noisy TV" pathology** (agent reward-hacking a random pixel).
- **Apply to Isaac**: The Isaac fly moves randomly (noisy TV risk for prediction-error methods like ICM). E3B's episodic Mahalanobis bonus specifically fixes this. This is the *exact* method Craftax adopted. [arXiv 2210.05805](https://arxiv.org/pdf/2210.05805)

### Distributional RND (DRND) — Yang et al., ICML 2024
- **Technique**: RND with distributional prediction targets rather than a single fixed net.
- **Result**: Fixes "bonus inconsistency" in vanilla RND — beats RND on Montezuma's Revenge, Adventure, hard-exploration Atari.
- **Apply to Isaac**: Drop-in replacement for RND if we want a global (not episodic) novelty signal. Cheap to implement. [arXiv 2401.09750](https://arxiv.org/pdf/2401.09750)

### DuRND: Dual Random Networks for Balancing Exploration and Exploitation — Ma et al., ICML 2025
- **Technique**: Two RND heads — one for exploration bonus, one for a value/significance signal — combined so exploration doesn't distract from task rewards.
- **Result**: Beats RND, NGU, E3B on sparse MiniGrid/MiniHack tasks.
- **Apply to Isaac**: Directly addresses our worry: a naive exploration bonus could reward wandering through rooms instead of killing enemies. DuRND is designed for exactly this trade-off. [PMLR v267](https://proceedings.mlr.press/v267/ma25j.html)

### Guiding Pretraining in RL with LLMs (ELLM) — Du et al., ICML 2023 (still the canonical LLM-guided exploration reference)
- **Technique**: LLM proposes goals in language ("chop a tree", "kill a zombie"); reward the agent for accomplishing them; used as a pretraining phase.
- **Result**: Big gains on Crafter over vanilla RND — the LLM biases exploration toward semantically meaningful states.
- **Apply to Isaac**: Isaac has strong text-describable state ("shoot the fly", "avoid the tear", "pick up bomb"). An LLM prompt bank could generate goal candidates we score against game state to shape exploration. [PMLR 202/du23f](https://proceedings.mlr.press/v202/du23f.html)

### ExploRLLM — Ma et al., arXiv 2403.09583 (2024)
- **Technique**: Uses foundation models to steer PPO exploration in high-dim manipulation; LLM outputs coarse action proposals, RL fine-tunes.
- **Result**: Order-of-magnitude sample efficiency gains on cluttered pick-and-place.
- **Apply to Isaac**: Similar idea — LLM proposes "kite the fly", RL learns joystick primitives. Useful once we have a text-describable observation. [arXiv 2403.09583](https://arxiv.org/html/2403.09583v2)

### Accelerating Goal-Conditioned RL Algorithms and Research — Bortkiewicz et al., arXiv 2408.11052 (2024)
- **Technique**: Fast JAX GCRL benchmark + strong contrastive-RL baselines (goal = future state).
- **Result**: 22× wall-clock speedup vs prior GCRL infrastructure; contrastive GCRL matches or beats RND-style methods on sparse tasks.
- **Apply to Isaac**: We could reformulate "clear the room" as goal-conditioned ("reach the state where enemy is dead"), which converts sparse reward into a per-step distance-in-latent signal. [arXiv 2408.11052](https://arxiv.org/pdf/2408.11052)

---

## Category 4: Auxiliary tasks for RL

### Bridging State and History Representations: Understanding Self-Predictive RL — Ni et al., ICLR 2024
- **Technique**: Unifies SPR, DeepMDP, BYOL-RL under one framework; proposes a minimal end-to-end algorithm with a single auxiliary loss.
- **Result**: Matches or beats specialized methods on MiniGrid, DMC, Atari; stable in POMDPs.
- **Apply to Isaac**: A single self-predictive auxiliary loss over the shared trunk would give us representation shaping for free, useful when the reward is too sparse to shape the encoder. [arXiv 2401.08898](https://arxiv.org/html/2401.08898v3)

### When does Self-Prediction help? — Voelcker et al., RLC 2024
- **Technique**: Theoretical + empirical study of when reconstruction vs. latent self-prediction auxiliary tasks help.
- **Result**: Latent self-prediction helps in observation-noise settings; reconstruction hurts when observations contain distractors.
- **Apply to Isaac**: Isaac has many distractors (particles, tear trails). Suggests SPR-style latent prediction, *not* pixel reconstruction. [RLJ 2024](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_197.pdf)

### Learning Successor Features the Simple Way — Farebrother et al., NeurIPS 2024
- **Technique**: A direct pixel-to-SF loss without pretraining; avoids representation collapse.
- **Result**: Zero-shot transfer across DMC tasks; representations don't collapse under non-stationarity.
- **Apply to Isaac**: SFs are attractive because they let us swap reward weights (kill vs. survival vs. movement) at inference without retraining. Useful once we scale to multiple rooms/objectives. [arXiv 2410.22133](https://arxiv.org/html/2410.22133v1)

---

## Category 5: Reward shaping without hacks

### Potential-Based Intrinsic Motivation (PBIM) — Forbes et al., AAAI 2024 (arXiv 2402.07411 / 2410.12197 extended)
- **Technique**: Converts *any* intrinsic-motivation bonus (RND, ICM, curiosity) into a potential-based form F(s,s') = γΦ(s') − Φ(s), preserving Ng et al. 1999's optimal-policy invariance.
- **Result**: PBIM stops IM methods from changing the optimal policy while retaining most exploration gains. Empirical wins on MiniGrid/Atari.
- **Apply to Isaac**: **Very high value.** If we shape ("get closer to the fly") we risk creating a new optimum (chase forever). PBIM turns any shaping into a *provably safe* one. This is probably the single most cited-worthy fix for our reward hacking concern. [arXiv 2410.12197](https://arxiv.org/pdf/2410.12197)

### On the Sample Efficiency of Abstractions and PBRS — Icarte et al. 2024
- **Technique**: Formal analysis: how much sample-efficiency PBRS gives you as a function of how well the potential approximates V*.
- **Result**: Even coarse abstractions (grid cells, object counts) yield large sample gains; the closer Φ is to V*, the faster.
- **Apply to Isaac**: Suggests handcrafted Φ(s) = -dist_to_enemy or Φ = -enemies_alive gives measurable speedup without hacking rewards. [arXiv 2404.07826](https://arxiv.org/pdf/2404.07826)

### Improving Intrinsic Exploration by Creating Stationary Objectives — Kim et al., ICLR 2024
- **Technique**: Turns non-stationary count-based/pseudo-count bonuses into a stationary reward by augmenting state with visitation info.
- **Result**: Stabilizes intrinsic-reward training on MiniHack, Procgen.
- **Apply to Isaac**: Directly addresses "the exploration bonus keeps shifting under the agent", which is a known cause of PPO thrashing on sparse tasks. [arXiv 2310.18144](https://arxiv.org/html/2310.18144v4)

---

## Category 6: PPO improvements

### PPG Reloaded — Wang et al., ICML 2023
- **Technique**: Empirical study of Phasic Policy Gradient; identifies which design choices matter (separate value network, auxiliary phase length).
- **Result**: PPG's advantage over PPO on Procgen mostly comes from the *decoupled value network with an auxiliary phase*, not the exact algorithm.
- **Apply to Isaac**: Cheap PPO upgrade: separate value & policy nets, add an aux phase that distills value into policy features. Improves generalization across procedural rooms. [PMLR v202/wang23aw](https://proceedings.mlr.press/v202/wang23aw.html)

### Episodic Transformer Memory PPO (TrXL/GTrXL-PPO) — Meter et al., open-source 2023-2024
- **Technique**: PPO with TransformerXL / GTrXL memory backbone; clean baseline on MiniGrid Memory, ProofOfConcept memory tasks.
- **Result**: Matches/beats LSTM-PPO on partially observable memory tasks; more stable than vanilla Transformer.
- **Apply to Isaac**: A single frame doesn't reveal fly velocity or bullet trajectories → Isaac *is* POMDP. Either LSTM-PPO or GTrXL-PPO is a required change. [repo](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)

### Rethinking Transformers in Solving POMDPs — Lu et al., ICML 2024
- **Technique**: Theoretical result — vanilla Transformers can't represent all regular languages, so plain PPO+Transformer is limited on POMDPs. Proposes point-wise recurrent Transformer.
- **Result**: Beats LSTM/Transformer PPO on POMDP benchmarks.
- **Apply to Isaac**: Warns us not to blindly slap a Transformer on and expect it to solve memory — LSTM is often better if we don't have compute for long context. [PMLR 235/lu24h](https://proceedings.mlr.press/v235/lu24h.html)

---

## Category 7: Video game agents

### D2AH-PPO: ViZDoom with Object-Aware Hierarchical RL — Wei et al., ISAS 2024
- **Technique**: Depth-detection + object-aware hierarchical PPO. High-level policy picks "engage / explore"; low-level executes.
- **Result**: SOTA on ViZDoom scenarios with sparse reward and partial observability.
- **Apply to Isaac**: Very close analog — top-down 2D can substitute for FPS. An object-aware feature (enemy positions as vector inputs) plus a two-level "engage vs move" policy is a clean architectural pattern for the corner-camping symptom. [DOI](https://doi.org/10.1109/isas61044.2024.10552510)

### Scaling Behavior Cloning Improves Causal Reasoning (Open Pixel2Play) — Elefant AI, arXiv 2601.04575 (early-2026)
- **Technique**: Foundation model trained by BC on 8,300 h human gameplay across many titles; keyboard+mouse output; released under open license.
- **Result**: Real-time video-game agent, single-GPU inference; strong zero-shot generalization.
- **Apply to Isaac**: If we can record even a few hours of Isaac human demos, BC pretraining → RL fine-tune (per this recipe) will dwarf pure-scratch PPO. [arXiv 2601.04575](https://arxiv.org/pdf/2601.04575)

### Real-Time Diffusion Policies for Games (CPQE) — Zhang et al., arXiv 2503.16978 (2025)
- **Technique**: Consistency policy (one-step diffusion) combined with Q-ensembles; multi-modal action distributions at real-time speed.
- **Result**: Beats Gaussian policies on games needing multi-modal behavior (e.g., "either kite left or kite right").
- **Apply to Isaac**: Corner-camping is a mode-collapse into one Gaussian action mode. Multi-modal policies could break this without any reward change. [arXiv 2503.16978](https://arxiv.org/html/2503.16978v1)

---

## Category 8: BC + RL hybrid

### Video PreTraining (VPT) — Baker et al., NeurIPS 2022 (still the reference for BC→RL pipelines in 2024-26)
- **Technique**: Train an inverse dynamics model on labeled data, use it to *pseudo-label* huge amounts of unlabeled human video, BC-pretrain, then RL fine-tune.
- **Result**: First agent to craft a diamond pickaxe in Minecraft from scratch — impossible for pure RL.
- **Apply to Isaac**: Isaac has abundant YouTube playthroughs. Even without inverse-dynamics, direct BC on our own recorded gameplay gives a warm start that starts *far* from the corner-camping local optimum. [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf)

### C-GAIL: Stabilizing GAIL with Control Theory — Luo et al., NeurIPS 2024
- **Technique**: Applies control-theoretic damping to the GAIL discriminator update; removes the training oscillations that plague vanilla GAIL.
- **Result**: Cleaner convergence, better final scores across MuJoCo and Atari.
- **Apply to Isaac**: If we can produce a few Isaac demos, GAIL learns a reward from them, sidestepping our hand-crafted (+1/-1/-0.001) entirely. C-GAIL makes GAIL practical. [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/34293d684b1012ed45c3274b4a7edc00-Paper-Conference.pdf)

### DPAIL: Diffusion-Policy Adversarial Imitation Learning — anonymous, OpenReview 2024
- **Technique**: Diffusion policy inside a GAIL-style adversarial IL loop; captures multi-modal demonstrations.
- **Result**: Beats GAIL/BC on Robomimic tasks where experts use multiple strategies.
- **Apply to Isaac**: Human Isaac play is multi-modal (kite / rush / bomb). DPAIL preserves that; a Gaussian BC would average them into something bad. [OpenReview](https://openreview.net/pdf?id=F2ILoE1eDj)

---

## Top 5 Recommendations for the Isaac problem

**Rationale for ranking:** the immediate failure mode is a degenerate local optimum (corner-camping) caused by (a) sparse reward, (b) mode-collapsed Gaussian policy, (c) memoryless MLP, (d) a step penalty that incentivizes hiding. Best fixes attack these directly with minimal engineering.

### 1. **PPO + ICM + E3B (Craftax recipe)** — Matthews et al., ICML 2024
This is *the* proven baseline for our setting (top-down procedural sparse-reward). E3B specifically resists the noisy-TV pathology that a random-moving fly would trigger in ICM/RND alone. Implementation is a single reference in `Craftax_Baselines`. Expected effect: strictly forces the agent to reach *new* states in each episode, which corner-camping does not.

### 2. **Potential-Based Intrinsic Motivation (PBIM)** — Forbes et al., AAAI 2024
Before we add any shaping (distance-to-enemy, damage-dealt), wrap it in PBIM. This is a ~10-line change that provably keeps the optimal policy invariant. Directly addresses the concern "shaping is a hack" by making it *not* a hack. Combine with a simple Φ(s) = −enemies_alive_count for an instant sparse-to-dense conversion.

### 3. **Recurrent PPO (LSTM) + Achievement-Distillation aux loss** — Moon et al., NeurIPS 2023 + standard Recurrent PPO
Two cheap, orthogonal wins. LSTM makes the observation Markovian enough to disambiguate enemy velocity and tear trajectories (Isaac is genuinely POMDP). Achievement distillation adds a contrastive representation loss over event tags (hit_enemy, took_damage, killed_enemy, opened_door) — dense representation-level signal *without* touching the reward function. This is what beats DreamerV3 on Crafter at 1M steps.

### 4. **BC pretraining → PPO fine-tune** (per VPT / Open-P2P recipe)
Record 1–5 hours of your own play, BC a policy, then unfreeze with PPO. The BC-initialized policy starts *outside* the corner-camping basin. Cost: a few hours of recording; expected reward: dramatic. Cheaper than any RL algorithmic change and empirically decisive in Minecraft (VPT), Hill Climb, Bubble Trouble, and now Open-P2P (2026).

### 5. **DreamerV3 tricks in PPO (symlog / twohot value)** — Sullivan et al., 2024
Almost free change to our existing PPO. Fixes reward-scale imbalance between +1 kill, −1 death, and any shaping we add. Prerequisite before layering intrinsic rewards / PBIM shaping — otherwise the value function is unstable.

**Ordering strategy:** do #5 first (5 lines of code, prevents downstream instability). Then #4 (highest sample-efficiency payoff per engineer-hour). Then #3 (architectural). Then #1 (intrinsic reward). Then #2 (only once you're ready to shape).

---

## Sources — kept vs. dropped

- Kept: Craftax paper — closest procedural-RL analog with published PPO+intrinsic-reward numbers.
- Kept: Achievement Distillation — beats DreamerV3 on procedural, clean auxiliary-loss recipe.
- Kept: E3B — specifically solves noisy-TV problem that Isaac fly would cause.
- Kept: PBIM (2024) — the theoretically clean answer to "how do I shape without hacks".
- Kept: VPT / Open-P2P — canonical BC→RL pipeline for games.
- Kept: DreamerV3 tricks-in-PPO — trivial to implement, big robustness win.
- Kept: D2AH-PPO ViZDoom — direct analog of hierarchical FPS agent, similar sparse-reward top-down/first-person shooter setting.
- Kept: DIAMOND / EfficientZeroV2 — if we ever pivot to model-based.
- Kept: Recurrent Transformer/LSTM PPO refs — Isaac is POMDP.
- Dropped: "Real-Time Diffusion Policies for Games" as a top-5 — too new, engineering-heavy, and mode-collapse can be fixed with entropy tuning first.
- Dropped: Preference-based RL (STAR, RLHF-style) — overkill; we have programmatic reward access.
- Dropped: Successor Features (Farebrother 2024) as top-5 — high-value long-term but not the immediate fix.
- Dropped: NetHack option-learning (Meta 2025) — Isaac room structure is much shallower than NetHack; overkill until we scale to full runs.
- Dropped: Random SerpentAI Isaac repos — hobbyist code, no published results, no reproducible signal.
- Dropped: LLM-guided exploration (ELLM, ExploRLLM) from top-5 — promising but requires text state or a vision-language pipeline we don't have yet.

## Gaps

- **No peer-reviewed RL work directly on Binding of Isaac.** Existing Isaac-AI repos (SerpentAI plugins, `2-X/binding-of-isaac-ai`, `yydinc.com` SAC agent) are hobbyist, no published results, no ablations. We are effectively the first serious writeup.
- **No 2024-26 paper on corner-camping specifically** in top-down shooters — the "Delightful Gradients" (2026) paper on softmax-PG corner escape is bandit-only and not directly applicable to deep PPO.
- **No published numbers on BC→PPO transfer for roguelike top-down shooters.** VPT (Minecraft), Open-P2P (mixed 2D/3D games), and small game-specific repos suggest it should work but exact sample-efficiency ratios are unknown.
- **Suggested next steps**: (1) Run the exact Craftax `train_icm --use_e3b` config on a stub Isaac env to see if the pathology reproduces on their harness. (2) Ablate PPO+LSTM vs PPO+MLP on our fly task in isolation. (3) Record 2 h demos, BC-pretrain, and compare initial reward to random-init PPO — this experiment answers the highest-leverage open question.
