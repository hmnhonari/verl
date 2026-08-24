# TVPO upstream port — verification instructions

> **For Claude (or a human) on another cluster.** This file lives on the *old* research repo
> (`hmnhonari/verl`, branch `tvpo-verification`, forked off `tvpo`). The code to verify lives on a
> **different branch**: `tvpo-pr`. Do not confuse the two.

## 0. TL;DR of the task

TVPO was originally implemented in this repo (old research fork, based on verl ~#5860, Apr–May
2026). It has been re-implemented on top of **current upstream verl** (~#7491) and is about to be
submitted as a pull request to `verl-project/verl`.

Your job is to **independently verify the port is behaviourally faithful and bug-free** before
that PR goes out. Concretely: confirm the claimed equivalences hold, confirm the claimed bug fixes
are real fixes and not behaviour changes in disguise, and find anything that was missed.

Assume nothing in this file is correct. It is a description of what was *intended*, written by the
agent that did the port. Verify, don't trust.

## 1. Getting the two versions side by side

The old implementation is already in this checkout:

```bash
# old TVPO, as actually run in the experiments
sed -n '/@register_policy_loss("tvpo")/,/^@register_policy_loss("dppo_kl")/p' \
    verl/trainer/ppo/core_algos.py
```

Note there are **two** old versions in this repo's history, and they differ:

| Where | What |
|---|---|
| `main` @ `b0f7cbae` | first version: batch-level TV (token-weighted `masked_mean` over the whole micro-batch), sign-agreement mask, `torch.where` with detached losses |
| `tvpo` @ `3e4658ea` (this branch's parent) | final version: prompt-level TV grouped by `uid`, `valid = prompt_valid \| token_valid`, mask multiplication, plus a lot of commented-out debris |

The port is based on the **`tvpo` branch version**, gated by a config flag so the batch-level
behaviour is still reachable.

Fetch the new code:

```bash
git fetch origin tvpo-pr:tvpo-pr
git worktree add ../verl-tvpo-pr tvpo-pr    # keeps this checkout untouched
cd ../verl-tvpo-pr
sed -n '/@register_policy_loss("tvpo")/,/^@register_policy_loss("dppo_kl")/p' \
    verl/trainer/ppo/core_algos.py
```

The whole port is one commit. `git show --stat tvpo-pr` should list 15 files; the only non-doc,
non-test, non-config code changes are `verl/trainer/ppo/core_algos.py` (new function only) and
`verl/workers/utils/losses.py`.

## 2. What changed and why — the claims to check

### 2.1 Claimed bug fixes (verify these are fixes, not regressions)

1. **`torch.zeros_like(ppo_tv)` shape bug.** In `main`'s version, the "inside the trust region"
   branch set `valid_mask = torch.ones_like(ppo_tv)` — `ppo_tv` is a **scalar**, so the mask was a
   0-dim tensor, not `(B, T)`. It broadcast silently instead of erroring. The `tvpo` branch had
   already fixed this to `torch.ones_like(advantages, dtype=torch.bool)`. The port builds the mask
   branchlessly instead:
   ```python
   valid_mask = (ppo_tv <= clip_divergence) | token_valid
   if prompt_tv_per_sample is not None:
       valid_mask = valid_mask | (prompt_tv_per_sample.unsqueeze(-1) <= clip_divergence)
   ```
   **Verify:** this is logically identical to the old `if ppo_tv <= clip_divergence: all-ones /
   else: prompt_valid | token_valid`. Convince yourself of the boolean algebra, then check it
   empirically (§3.2).

2. **Sign taken in log space.** Old: `ref_grad = torch.sign(prob - old_prob)` with
   `prob = exp(log_prob)`. New: `torch.sign(negative_approx_kl)` where
   `negative_approx_kl = clamp(log_prob - old_log_prob, -20, 20)`.
   **Verify:** `sign(exp(a) - exp(b)) == sign(a - b)` for all finite `a, b`, and clamping to
   `±20` preserves the sign. Then verify the new form is strictly better: construct log-probs
   small enough that `exp()` underflows to `0.0` in fp32 (both terms) — the old form returns
   `sign(0 - 0) == 0` (token wrongly treated as "unmoved", hence always kept), the new form
   returns the correct sign. **This is a real behaviour change on rare tokens.** Decide whether it
   is the right one.

3. **Empty sequences diluting a prompt group.** Old:
   ```python
   prompt_count = torch.bincount(prompt_index).to(dtype)
   prompt_tv_sum = zeros.index_add_(0, prompt_index, per_sample_tv)
   prompt_tv_unique = prompt_tv_sum / prompt_count.clamp_min(1.0)
   ```
   A sequence with an all-zero `response_mask` yields `per_sample_tv == 0` (because
   `masked_mean` divides by `mask.sum() + 1e-8`), but still counts `1` in `prompt_count` — so it
   drags its group's TV toward zero and can wrongly keep the group inside the trust region. The
   port excludes zero-token rows from both the sum and the count.
   **Verify:** does verl actually put zero-`response_mask` rows in a micro-batch? See
   `verl/trainer/ppo/padding_utils.py` (`template_sample["uid"] = pad_uid`) and the
   `min_num_micro_batch` padding path in `verl/utils/seqlen_balancing.py`. If it can happen, the
   fix matters; if it provably cannot, the fix is harmless but the justification is wrong.

4. **TV estimate detached and computed in fp32.** Old built an autograd graph for
   `per_sample_tv` that was never used for gradient (the mask is boolean), and reduced in the
   log-prob dtype.
   **Verify:** no gradient path is lost. Specifically confirm that in the old code no gradient
   ever flowed from `ppo_tv` / `prompt_tv` into the loss.

### 2.2 Deliberate behaviour differences (NOT bug fixes — flag if you disagree)

1. **`tvpo_tv_scope="batch"` is not the old `index=None` fallback.**
   - Old (`tvpo` branch, `index is None`): `ppo_tv = per_sample_tv.mean()` — an **unweighted mean
     of per-sequence means**, so a 100-token sequence and a 5-token sequence count equally.
   - New (`tvpo_tv_scope="batch"`): `ppo_tv = masked_mean(tv_per_token, response_mask)` — a
     **token-weighted mean** over the micro-batch. This matches `main` @ `b0f7cbae`.

   These differ whenever sequence lengths differ, which is always. The port chose the
   token-weighted form because it is the standard TV estimator and matches the first
   implementation. **This is the single most important thing to confirm is what was wanted.**

2. **Metrics dropped.** The old version logged `actor/ppo_tv_prompt/{min,max,std,hist}`. The
   `hist` entry is a Python list, which upstream's `Metric.from_dict` cannot aggregate (the old
   branch patched `losses.py` to split scalar and list metrics). The port keeps only
   `actor/ppo_tv` and does not touch the metric pipeline. If the per-prompt spread is needed for
   research, it has to go back in — but not in the upstream PR.

3. **`index` is passed only to losses that declare it.** The old branch added
   `index: ... = None` to **all** registered policy losses and always passed it. The port uses a
   cached `inspect.signature` check in `verl/workers/utils/losses.py` so the other twelve losses
   are untouched and externally registered losses do not break with `TypeError`.
   **Verify:** `_accepts_prompt_index` is cached on the function object; confirm the cache cannot
   go stale if a loss is re-registered at runtime (it can't in practice — the registry is
   populated at import — but check).

### 2.3 New config surface

`actor_rollout_ref.actor.policy_loss.tvpo_tv_scope`, default `"prompt"`, values `prompt | batch`,
validated in `PolicyLossConfig.__post_init__`. Mirrored into
`verl/trainer/config/actor/actor.yaml` and the four `_generated_*.yaml` files.

## 3. Verification steps

### 3.1 Run the test suite and the lint gates

From the `tvpo-pr` worktree:

```bash
pytest tests/trainer/ppo/test_tvpo_on_cpu.py -v            # 16 tests, all should pass
pytest tests/trainer/ppo/ -q --ignore=tests/trainer/ppo/v1 # 159 tests

ruff check verl/ tests/
ruff format --check verl/ tests/
mypy verl/trainer/ppo/core_algos.py                        # in pyproject's strict override list
scripts/generate_trainer_config.sh                         # must report "All good" (no diff)
python3 tests/special_sanity/check_docstrings.py
python3 tests/special_sanity/check_docs_time_info.py
python3 tests/special_sanity/check_example_naming.py --root examples
```

Known-unrelated failures in a bare CPU env: `tests/utils/test_config_on_cpu.py::TestPrintCfgCommand::test_command_with_override`
and `tests/workers/config/test_model_config_on_cpu.py::TestHFModelConfigCPU::test_target_modules_raises_on_invalid_type`
— both need a local HF model path. Confirm they fail identically on unmodified upstream `main`
before dismissing them.

**Do not just check the tests pass — read them and decide whether they test the right thing.**
In particular `test_prompt_grouping_key_reaches_the_loss_through_ppo_loss` is the one that proves
`uid` survives micro-batching; if it were wrong, `tvpo_tv_scope=prompt` would silently degrade to
`batch` in real training and every other test would still pass.

### 3.2 Differential test: old implementation vs new

This is the highest-value check. Port the **old** function verbatim into a standalone script and
compare against the new one on random inputs. Save as `/tmp/tvpo_diff.py` in the `tvpo-pr`
worktree:

```python
import numpy as np, torch
import verl.utils.torch_functional as verl_F
from verl.utils import as_torch_index
from verl.trainer.ppo.core_algos import compute_policy_loss_tvpo
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def old_tvpo(old_log_prob, log_prob, advantages, response_mask, clip_ratio, clip_ratio_c, index):
    """Verbatim transcription of hmnhonari/verl@3e4658ea compute_policy_loss_tvpo."""
    clip_divergence = clip_ratio / 2
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    per_sample_tv = verl_F.masked_mean(torch.abs(ratio - 1.0), response_mask, axis=-1) / 2
    if index is None:
        prompt_tv_per_sample = prompt_tv_unique = per_sample_tv
    else:
        prompt_index = as_torch_index(index, device=per_sample_tv.device)
        prompt_count = torch.bincount(prompt_index).to(per_sample_tv.dtype)
        prompt_tv_sum = torch.zeros_like(prompt_count).index_add_(0, prompt_index, per_sample_tv)
        prompt_tv_unique = prompt_tv_sum / prompt_count.clamp_min(1.0)
        prompt_tv_per_sample = prompt_tv_unique[prompt_index]
    ppo_tv = prompt_tv_unique.mean()
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c).detach()
    prob, old_prob = torch.exp(log_prob), torch.exp(old_log_prob)
    if ppo_tv <= clip_divergence:
        valid_mask = torch.ones_like(advantages, dtype=torch.bool)
    else:
        token_valid = (advantages * torch.sign(prob - old_prob)) <= 0
        prompt_valid = prompt_tv_per_sample.unsqueeze(-1) <= clip_divergence
        valid_mask = (prompt_valid | token_valid) if index is not None else token_valid
    valid_mask = valid_mask.detach().float()
    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask
    loss = verl_F.masked_mean(pg_losses, response_mask)
    return loss, ppo_tv.item(), valid_mask


def cfg(clip_ratio, scope, clip_ratio_c=20.0):
    return ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
        clip_ratio=clip_ratio, clip_ratio_low=clip_ratio, clip_ratio_high=clip_ratio,
        clip_ratio_c=clip_ratio_c, loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode="tvpo", tvpo_tv_scope=scope),
    )


torch.manual_seed(0)
worst = 0.0
for trial in range(500):
    B, T, G = 8, 16, 4
    # EQUAL LENGTHS + FULL MASK: this is where old and new are expected to agree.
    response_mask = torch.ones(B, T)
    old_lp = -torch.rand(B, T) * 3 - 0.1
    log_prob = (old_lp + torch.randn(B, T) * 0.3).requires_grad_(True)
    adv = torch.randn(B, T)
    index = np.array([f"g{i % G}" for i in range(B)], dtype=object)
    clip_ratio = float(np.random.choice([0.01, 0.05, 0.2, 1.0, 5.0]))

    new_loss, new_m = compute_policy_loss_tvpo(
        old_log_prob=old_lp, log_prob=log_prob, advantages=adv,
        response_mask=response_mask, loss_agg_mode="token-mean",
        config=cfg(clip_ratio, "prompt"), index=index,
    )
    old_loss, old_tv, _ = old_tvpo(old_lp, log_prob, adv, response_mask, clip_ratio, 20.0, index)
    d_loss = (new_loss - old_loss).abs().item()
    d_tv = abs(new_m["actor/ppo_tv"] - old_tv)
    worst = max(worst, d_loss, d_tv)
    assert d_loss < 1e-5 and d_tv < 1e-5, (trial, clip_ratio, d_loss, d_tv)
print("prompt scope: old == new on equal-length full-mask batches, worst diff", worst)
```

Expected result: **prompt scope agrees with the old implementation exactly** on equal-length,
fully-masked batches. When the port was written this printed `worst diff 0.0` over 500 trials —
bit-for-bit, not merely within tolerance. Anything non-zero means something drifted.

Then extend the script yourself. These four extensions were run at port time; the measured values
are given so you can check you are reproducing the same situation, **not** so you can skip running
them:

| Extension | Expected | Measured at port time |
|---|---|---|
| Ragged (non-empty) masks, prompt scope | agree | `worst diff 0.000e+00` |
| One all-zero `response_mask` row in a group | **disagree**, new ignores the empty row | old `ppo_tv=0.8750`, new `ppo_tv=1.0000`, true TV `1.0` |
| `batch` scope vs old `index=None`, ragged | **disagree** (§2.2.1) | old (seq-weighted) `0.23506`, new (token-weighted) `0.22802` |
| Both probabilities underflow to `0.0` | **disagree**, new is right | old `clipfrac=0.00` (wrongly keeps every token), new `clipfrac=0.50` |

Details of each:

- **Ragged masks.** Give sequences different lengths. Prompt scope should still agree (the
  zero-token exclusion only bites on *empty* rows, not short ones). If it disagrees, the port has
  changed something it should not have — investigate.
- **Empty rows.** Set one row's `response_mask` to all zeros. Old and new should now **disagree**,
  and the new value should be the one that ignores the empty row. Confirm the direction.
- **Batch scope.** Compare `tvpo_tv_scope="batch"` against `old_tvpo(..., index=None)`. They
  should disagree on ragged batches (§2.2.1) and agree on equal-length ones. Confirm this is the
  intended difference and not an accident.
- **Underflow.** `old_lp = torch.full((B, T), -200.0)`. Old and new should disagree; new should be
  right.
- **Gradient masking.** Backward through both and compare `log_prob.grad`, not just the loss —
  a mask can be right in value and wrong in gradient.

### 3.3 Read the maths, not just the code

Independently re-derive:

1. `TV(p, q) = 0.5 * sum_a |p(a) - q(a)| = 0.5 * E_{a~q}[|p(a)/q(a) - 1|]`. Confirm the code's
   estimator (`mean over sampled tokens of |ratio - 1| / 2`) is the single-sample Monte Carlo
   estimate of this, and note it is a *biased* estimate of the per-state TV when averaged across
   states — decide whether that matters for a threshold test.
2. `clip_divergence = config.clip_ratio / 2`. The docstring claims `clip_ratio` is the L1 budget
   `sum_a |pi - pi_old|` and TV is half of it. Confirm the factor of 2 is consistent between the
   threshold and the estimator, and that it matches the old code (it does — both halve).
3. The mask rule. The surrogate is `-A * stopgrad(rho) * log pi`, so
   `d/d(log pi) = -A * rho`, i.e. the update moves `log pi` in the direction of `+A`. The
   divergence grows when that direction agrees with the move already made,
   `sign(pi - pi_old)`. Confirm `advantage * sign(pi - pi_old) <= 0` is therefore the
   "contracting" set, and confirm the `sign == 0` case (unmoved token) landing in the *kept* set
   is intended.

### 3.4 Known non-idealities to sanity check (present in the old code too)

- **`clip_ratio_c` fallback is dead code.** `config.get("clip_ratio_c", 20.0)` never returns
  `20.0`, because `ActorConfig.clip_ratio_c` has a default of `3.0`. Inherited verbatim from
  upstream `dppo_tv` and kept for symmetry. The example script sets `10000.0` explicitly. Confirm
  you agree with keeping the quirk rather than fixing it in this PR.
- **Micro-batch locality.** The TV estimate is computed per micro-batch and is *not* all-reduced
  across data-parallel ranks, so different ranks (and, under gradient accumulation, different
  micro-batches on the same rank) can make different in/out-of-region decisions. The old code has
  the same property. Confirm this is acceptable for the algorithm; if not, it needs a
  `dist.all_reduce` and the PR needs rework.
- **Prompt groups are per-micro-batch.** A prompt's rollouts can be split across micro-batches, so
  a group's TV is the average over only the rollouts present in that micro-batch. Same in the old
  code. Check whether `balance_batch=True` / dynamic-bsz reordering makes this worse.

### 3.5 If GPUs are available

Run a short end-to-end training smoke test (a small model, a few steps) with:

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=tvpo \
actor_rollout_ref.actor.policy_loss.tvpo_tv_scope=prompt \
actor_rollout_ref.actor.clip_ratio=0.015 \
actor_rollout_ref.actor.clip_ratio_c=10000.0
```

Check that `actor/ppo_tv` is logged, is finite, and is on the same order as what the old
implementation produced on comparable runs. Then repeat with `tvpo_tv_scope=batch` and confirm
`actor/ppo_tv` shifts in the direction §2.2.1 predicts (token-weighted vs sequence-weighted).

Also run one step under Megatron with pipeline parallelism if possible — that is the path where
`uid` has the most machinery between the trainer and the loss.

## 4. What was NOT verified by the port

- **No training-scale validation.** The port was verified by unit tests and static analysis only.
  No training curve, no eval numbers. verl maintainers commonly ask for these on algorithm PRs.
- **No multi-GPU / multi-node run.** The `uid` plumbing was verified end-to-end through
  `ppo_loss` on a hand-built micro-batch TensorDict, and by reading
  `index_select_tensor_dict` / `chunk_tensordict` (both handle `NonTensorStack` explicitly), but
  never on real distributed hardware.
- **Megatron path untested at runtime.** Same reasoning as above, no execution.

## 5. Reporting back

Produce: (a) pass/fail for every command in §3.1, (b) the differential-test table from §3.2 with
actual numbers, (c) a verdict on each claim in §2.1 and §2.2, (d) anything found that is not in
this file. Flag §2.2.1 explicitly — it is a silent semantic change to `batch` scope and needs a
human decision.
