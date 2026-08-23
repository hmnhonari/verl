# Total-Variation Policy Optimization (TVPO)

TVPO is a variant of [DPPO-Binary-TV](../dppo_trainer/README.md) that moves the total-variation
trust region from the individual token to the batch — or, by default, to the prompt group.

## ✨Getting started

1. Prepare the datasets by running [prepare_dapo_data.sh](https://github.com/verl-project/verl-recipe/blob/3490a22a0a3adeb7e4787fe70b1060b642efbae4/dapo/prepare_dapo_data.sh):

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Prepare the model:

```bash
hf download Qwen/Qwen3-30B-A3B-Base --local-dir ${HOME}/verl/models/Qwen3-30B-A3B-Base
```

3. Run the script:

```bash
# TVPO with a per-prompt trust region (default)
bash examples/tvpo_trainer/run_qwen3_30b_a3b_megatron.sh

# TVPO with a single batch-level trust region
TV_SCOPE=batch bash examples/tvpo_trainer/run_qwen3_30b_a3b_megatron.sh

# a wider trust region
CLIP_RATIO=0.05 bash examples/tvpo_trainer/run_qwen3_30b_a3b_megatron.sh
```

## 📖How it works

PPO and DPPO both mask **each token** whose own move is too large. TVPO instead asks how far the
policy has moved *in total*:

1. It estimates the realised total-variation distance from the importance ratio,
   `TV(pi, pi_old) = 0.5 * E_{a ~ pi_old}[|pi(a)/pi_old(a) - 1|]`, and compares it against
   `clip_ratio / 2` (`clip_ratio` is the budget on the L1 distance `sum_a |pi(a|s) - pi_old(a|s)|`,
   which is twice the TV distance).
2. While the estimate is **inside** the trust region no token is masked, so the update does not pay
   the optimisation cost of a per-token mask when the policy has barely moved.
3. Once the estimate leaves the trust region, only the tokens whose update pulls the policy **back
   towards** `pi_old` survive. The gradient of the surrogate w.r.t. `log pi` carries the sign of the
   advantage, so an update grows the divergence exactly when the advantage agrees with the
   probability change that already happened; requiring `advantage * sign(pi - pi_old) <= 0` keeps
   only the tokens that shrink it.

With `policy_loss.tvpo_tv_scope=prompt` (the default) the divergence is estimated per prompt group
as well, and a group still inside the trust region keeps all of its tokens even when the batch as a
whole has left it. With `batch`, a single estimate over the micro-batch governs every token.

See [the TVPO documentation](../../docs/algo/tvpo.md) for the configuration reference.

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
