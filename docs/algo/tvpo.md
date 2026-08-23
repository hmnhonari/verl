# Total-Variation Policy Optimization (TVPO)

Last updated: 08/23/2026.

TVPO is a variant of [DPPO-Binary-TV](dppo.md) that moves the total-variation (TV) trust region
from the individual token to the batch — or, by default, to the prompt group.

## ✨Getting started

```bash
# TVPO with a per-prompt trust region (default)
bash examples/tvpo_trainer/run_qwen3_30b_a3b_megatron.sh

# TVPO with a single batch-level trust region
TV_SCOPE=batch bash examples/tvpo_trainer/run_qwen3_30b_a3b_megatron.sh
```

The dataset and model preparation steps are the same as for [DPPO](dppo.md).

## 📖Introduction

PPO (and GRPO) masks each token whose probability *ratio* leaves `[1 - eps, 1 + eps]`.
DPPO-Binary-TV replaces that heuristic with a principled per-token divergence mask: a token is
masked when its own probability moved further than the budget in the direction its advantage
pushes. Both are per-token decisions, so a batch that has barely moved still pays for a mask.

TVPO asks a different question: **how far has the policy moved in total?**

1. The realised TV distance is estimated from the importance ratio,

   ```
   TV(pi, pi_old) = 0.5 * E_{a ~ pi_old} [ |pi(a)/pi_old(a) - 1| ]
   ```

   and compared against `clip_ratio / 2`. `clip_ratio` is read as the budget on the L1 distance
   `sum_a |pi(a|s) - pi_old(a|s)|`, which is twice the TV distance.
2. While the estimate is **inside** the trust region, every token is updated — TVPO is then the
   plain policy gradient with truncated importance sampling, and pays no masking cost.
3. Once the estimate **leaves** the trust region, only the tokens whose update pulls the policy
   back towards `pi_old` survive. The gradient of the surrogate w.r.t. `log pi` has the sign of the
   advantage, so an update grows the divergence exactly when the advantage agrees with the
   probability change that already happened. Requiring

   ```
   advantage * sign(pi - pi_old) <= 0
   ```

   keeps only the tokens that shrink the divergence. A token that has not moved at all
   (`sign = 0`) is always kept.

The estimate is computed over the local micro-batch, so — like the per-token masks of the other
trust-region losses — it is not synchronised across data-parallel ranks.

## Scope of the divergence estimate

`actor_rollout_ref.actor.policy_loss.tvpo_tv_scope` selects what the estimate is taken over:

| Value      | Behaviour                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `prompt`   | (default) The divergence is estimated per prompt group as well as over the batch. A group still inside the trust region keeps **all** of its tokens, even when the batch as a whole has left it. Falls back to `batch` when no prompt grouping key (`uid`) is available. |
| `batch`    | A single estimate over the micro-batch governs every token.                                                                                                     |

Under `prompt`, a group's estimate is the average over its sequences that are present in the same
micro-batch; sequences without a scored token are excluded, so padding sequences never dilute a
group.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: tvpo
      tvpo_tv_scope: prompt   # or: batch
    # the L1 budget; the TV threshold is half of it.
    # clip_ratio_low / clip_ratio_high are not used by this loss.
    clip_ratio: 0.015
    # TIS truncation of the importance ratio; set it high to disable truncation bias
    clip_ratio_c: 10000.0
```

Note that `clip_ratio` means something different here than it does for PPO: it is a divergence
budget, not a ratio window, and the useful values are correspondingly smaller.

## Metrics

| Metric                     | Meaning                                                                     |
|----------------------------|-----------------------------------------------------------------------------|
| `actor/ppo_tv`             | The TV divergence estimate that governs the mask.                            |
| `actor/pg_clipfrac`        | Fraction of tokens masked out of the update.                                 |
| `actor/ppo_kl`             | Mean `-(log pi - log pi_old)` over the response tokens.                      |
| `actor/pg_clipfrac_lower`  | Fraction of unmasked tokens whose importance ratio was truncated at `clip_ratio_c`. |

`actor/ppo_tv` staying below `clip_ratio / 2` with `actor/pg_clipfrac` at zero means the trust
region is never binding — lower `clip_ratio` if you want it to bite.

## Citation

TVPO builds on the trust region introduced by DPPO:

```bibtex
@article{qi2026dppo,
  title={Rethinking the Trust Region in LLM Reinforcement Learning},
  author={Qi, Penghui and Zhou, Xiangxin and Liu, Zichen and Pang, Tianyu and Du, Chao and Lin, Min and Lee, Wee Sun},
  journal={arXiv preprint arXiv:2602.04879},
  year={2026}
}
```
