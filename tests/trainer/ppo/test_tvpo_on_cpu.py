# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU coverage for the TVPO policy loss."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_tvpo, get_policy_loss_fn
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _actor_config(*, clip_ratio: float, tv_scope: str = "prompt", clip_ratio_c: float = 20.0) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode="tvpo", tvpo_tv_scope=tv_scope),
    )


def _tvpo(old_log_prob, log_prob, advantages, response_mask, config, index=None):
    return compute_policy_loss_tvpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
        index=index,
    )


def test_tvpo_is_registered():
    assert get_policy_loss_fn("tvpo") is compute_policy_loss_tvpo


def test_inside_trust_region_updates_every_token():
    """Below the TV threshold TVPO is the plain (TIS-truncated) policy gradient: nothing is masked."""
    old_log_prob = torch.zeros(2, 3)
    # ratio = 1.1 everywhere -> TV = 0.05, threshold = clip_ratio / 2 = 0.1
    log_prob = torch.full((2, 3), float(np.log(1.1)), requires_grad=True)
    advantages = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]])
    response_mask = torch.ones(2, 3)
    config = _actor_config(clip_ratio=0.2, tv_scope="batch")

    loss, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    assert metrics["actor/ppo_tv"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)
    expected = (-advantages * 1.1 * log_prob).mean()
    torch.testing.assert_close(loss, expected)


def test_outside_trust_region_keeps_only_contracting_tokens():
    """Above the threshold only tokens whose update moves pi back towards pi_old survive."""
    old_log_prob = torch.zeros(1, 4)
    # ratios 2, 2, 0.5, 0.5 -> TV = (1 + 1 + 0.5 + 0.5) / 4 / 2 = 0.375 > 0.1
    log_prob = torch.tensor([[np.log(2.0), np.log(2.0), np.log(0.5), np.log(0.5)]], requires_grad=True)
    # sign(pi - pi_old) = +1, +1, -1, -1
    advantages = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    response_mask = torch.ones(1, 4)
    config = _actor_config(clip_ratio=0.2, tv_scope="batch")

    loss, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    assert metrics["actor/ppo_tv"] == pytest.approx(0.375, abs=1e-6)
    # tokens 0 and 3 push the policy further away and are masked out
    expected_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)

    ratio = torch.exp(log_prob.detach())
    expected = (-advantages * ratio * log_prob * expected_mask).mean()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    # the masked tokens receive no gradient
    assert log_prob.grad[0, 0].item() == 0.0
    assert log_prob.grad[0, 3].item() == 0.0
    assert log_prob.grad[0, 1].item() != 0.0
    assert log_prob.grad[0, 2].item() != 0.0


def test_unmoved_tokens_are_never_masked():
    """sign(pi - pi_old) == 0 keeps a token trainable, whatever the sign of its advantage."""
    old_log_prob = torch.zeros(1, 3)
    log_prob = torch.tensor([[np.log(4.0), 0.0, 0.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, -1.0]])
    response_mask = torch.ones(1, 3)
    config = _actor_config(clip_ratio=0.2, tv_scope="batch")

    _, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    # only the first token moved, and it moved the wrong way
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0 / 3.0)


def test_prompt_scope_spares_groups_inside_the_trust_region():
    """A group still inside the trust region keeps every token, even when the batch has left it."""
    old_log_prob = torch.zeros(4, 2)
    # group "a" barely moves (per-token TV 0.05), group "b" triples every probability (TV 1.0)
    log_prob = torch.tensor(
        [
            [np.log(1.1), np.log(1.1)],
            [np.log(1.1), np.log(1.1)],
            [np.log(3.0), np.log(3.0)],
            [np.log(3.0), np.log(3.0)],
        ],
        requires_grad=True,
    )
    # every token moved up with a positive advantage, so the sign rule alone masks all of them
    advantages = torch.ones(4, 2)
    response_mask = torch.ones(4, 2)
    index = np.array(["a", "a", "b", "b"], dtype=object)

    # batch estimate is (0.05 + 1.0) / 2 = 0.525, above the 0.25 threshold
    prompt_cfg = _actor_config(clip_ratio=0.5, tv_scope="prompt")
    _, prompt_metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, prompt_cfg, index=index)
    assert prompt_metrics["actor/ppo_tv"] == pytest.approx(0.525, abs=1e-6)
    # group "a" is inside its own trust region and survives, group "b" is fully masked
    assert prompt_metrics["actor/pg_clipfrac"] == pytest.approx(0.5)

    # the same batch under a single batch-level estimate masks everything
    batch_cfg = _actor_config(clip_ratio=0.5, tv_scope="batch")
    _, batch_metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, batch_cfg, index=index)
    assert batch_metrics["actor/pg_clipfrac"] == pytest.approx(1.0)


def test_prompt_scope_keeps_everything_while_the_batch_is_inside_the_region():
    old_log_prob = torch.zeros(4, 2)
    log_prob = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [np.log(3.0), np.log(3.0)],
            [np.log(3.0), np.log(3.0)],
        ],
        requires_grad=True,
    )
    advantages = torch.ones(4, 2)
    response_mask = torch.ones(4, 2)
    index = np.array(["a", "a", "b", "b"], dtype=object)
    # prompt TVs are 0.0 and 1.0, so the batch estimate is 0.5, below the 0.6 threshold
    config = _actor_config(clip_ratio=1.2, tv_scope="prompt")

    _, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config, index=index)

    assert metrics["actor/ppo_tv"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)


def test_batch_scope_ignores_the_grouping_key():
    old_log_prob = torch.zeros(4, 2)
    log_prob = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [np.log(3.0), np.log(3.0)],
            [np.log(3.0), np.log(3.0)],
        ],
        requires_grad=True,
    )
    advantages = torch.ones(4, 2)
    response_mask = torch.ones(4, 2)
    index = np.array(["a", "a", "b", "b"], dtype=object)
    config = _actor_config(clip_ratio=0.5, tv_scope="batch")

    _, with_index = _tvpo(old_log_prob, log_prob, advantages, response_mask, config, index=index)
    _, without_index = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    assert with_index == without_index
    # every token that moved up with a positive advantage is masked, the unmoved ones survive
    assert with_index["actor/pg_clipfrac"] == pytest.approx(0.5)


def test_prompt_scope_falls_back_to_batch_without_a_grouping_key():
    old_log_prob = torch.zeros(2, 2)
    log_prob = torch.tensor([[0.0, 0.0], [np.log(3.0), np.log(3.0)]], requires_grad=True)
    advantages = torch.ones(2, 2)
    response_mask = torch.ones(2, 2)

    prompt_cfg = _actor_config(clip_ratio=0.5, tv_scope="prompt")
    batch_cfg = _actor_config(clip_ratio=0.5, tv_scope="batch")

    _, prompt_metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, prompt_cfg)
    _, batch_metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, batch_cfg)

    assert prompt_metrics == batch_metrics


def test_empty_sequences_do_not_dilute_their_group():
    """A padding sequence with no scored token must not drag its group's TV estimate down."""
    old_log_prob = torch.zeros(3, 2)
    log_prob = torch.tensor(
        [
            [np.log(3.0), np.log(3.0)],
            [np.log(3.0), np.log(3.0)],
            [np.log(3.0), np.log(3.0)],
        ],
        requires_grad=True,
    )
    advantages = torch.ones(3, 2)
    index = np.array(["a", "a", "a"], dtype=object)
    config = _actor_config(clip_ratio=1.0, tv_scope="prompt")

    full_mask = torch.ones(3, 2)
    _, dense = _tvpo(old_log_prob, log_prob, advantages, full_mask, config, index=index)

    padded_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    _, padded = _tvpo(old_log_prob, log_prob, advantages, padded_mask, config, index=index)

    assert padded["actor/ppo_tv"] == pytest.approx(dense["actor/ppo_tv"], abs=1e-6)
    assert padded["actor/ppo_tv"] == pytest.approx(1.0, abs=1e-6)
    assert not np.isnan(padded["actor/ppo_tv"])


def test_tv_estimate_survives_probability_underflow():
    """The mask is decided in log space, so vanishing probabilities keep the right sign."""
    old_log_prob = torch.full((1, 2), -200.0)
    # both probabilities underflow to 0.0 in float32, but the ratio is still well defined
    log_prob = torch.tensor([[-199.0, -201.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0]])
    response_mask = torch.ones(1, 2)
    config = _actor_config(clip_ratio=0.01, tv_scope="batch")

    assert torch.exp(old_log_prob).max().item() == 0.0

    _, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    # token 0 moved up with a positive advantage (masked), token 1 moved down (kept)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)


def test_half_precision_log_probs_are_handled():
    """bf16 log-probs must still produce a finite fp32 divergence estimate and a sane mask."""
    old_log_prob = torch.zeros(8, 4, dtype=torch.bfloat16)
    log_prob = torch.full((8, 4), float(np.log(3.0)), dtype=torch.bfloat16, requires_grad=True)
    advantages = torch.ones(8, 4, dtype=torch.bfloat16)
    response_mask = torch.ones(8, 4)
    index = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    config = _actor_config(clip_ratio=0.5, tv_scope="prompt")

    loss, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config, index=index)

    assert np.isfinite(metrics["actor/ppo_tv"])
    assert metrics["actor/ppo_tv"] == pytest.approx(1.0, abs=1e-2)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)
    assert torch.isfinite(loss).all()


def test_response_mask_excludes_prompt_padding_from_the_loss():
    old_log_prob = torch.zeros(1, 3)
    log_prob = torch.tensor([[np.log(1.1), np.log(1.1), 5.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1000.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    config = _actor_config(clip_ratio=0.2, tv_scope="batch")

    loss, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    assert metrics["actor/ppo_tv"] == pytest.approx(0.05, abs=1e-6)
    ratio = torch.exp(log_prob.detach())
    expected = (-advantages * ratio * log_prob * response_mask)[:, :2].sum() / 2
    torch.testing.assert_close(loss, expected)


def test_truncated_importance_sampling_caps_the_ratio():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.tensor([[np.log(10.0), 0.0]], requires_grad=True)
    advantages = torch.tensor([[-1.0, -1.0]])
    response_mask = torch.ones(1, 2)
    config = _actor_config(clip_ratio=100.0, tv_scope="batch", clip_ratio_c=3.0)

    loss, metrics = _tvpo(old_log_prob, log_prob, advantages, response_mask, config)

    # inside the trust region, so nothing is masked, but the ratio of token 0 is capped at 3.0
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.0)
    assert metrics["actor/pg_clipfrac_lower"] == pytest.approx(0.5)
    expected = (-advantages * torch.tensor([[3.0, 1.0]]) * log_prob).mean()
    torch.testing.assert_close(loss, expected)


def test_rollout_correction_weights_are_applied():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.tensor([[np.log(1.1), np.log(1.1)]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0]])
    response_mask = torch.ones(1, 2)
    weights = torch.tensor([[0.5, 2.0]])
    config = _actor_config(clip_ratio=0.2, tv_scope="batch")

    loss, _ = compute_policy_loss_tvpo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
        rollout_is_weights=weights,
    )

    expected = (-advantages * 1.1 * log_prob * weights).mean()
    torch.testing.assert_close(loss, expected)


def test_invalid_tv_scope_is_rejected():
    with pytest.raises(ValueError, match="tvpo_tv_scope"):
        PolicyLossConfig(loss_mode="tvpo", tvpo_tv_scope="sequence")


def _micro_batch(ratios_per_sample, uids, prompt_len=2):
    """Build the no-padding micro-batch TensorDict that the model engine hands to ``ppo_loss``."""
    from verl.utils import tensordict_utils as tu

    def nest(vals):
        return torch.nested.as_nested_tensor([torch.as_tensor(v) for v in vals], layout=torch.jagged)

    resp_lens = [len(r) for r in ratios_per_sample]
    data = tu.get_tensordict(
        tensor_dict={
            "prompts": nest([torch.ones(prompt_len, dtype=torch.long) for _ in resp_lens]),
            "responses": nest([torch.ones(r, dtype=torch.long) for r in resp_lens]),
            "response_mask": nest([torch.ones(r) for r in resp_lens]),
            "old_log_probs": nest([torch.zeros(r) for r in resp_lens]),
            "advantages": nest([torch.ones(r) for r in resp_lens]),
            "uid": list(uids),
        },
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)

    # the model output covers prompt + response; ``no_padding_2_padding`` slices the response out
    # with the usual one-token left shift (a position predicts the *next* token), so place the
    # response log-probs one step early and let the last position fall off the end.
    log_probs = nest(
        [
            torch.cat(
                [
                    torch.zeros(prompt_len - 1),
                    torch.log(torch.tensor(ratios, dtype=torch.float32)),
                    torch.zeros(1),
                ]
            )
            for ratios in ratios_per_sample
        ]
    )
    return {"log_probs": log_probs.detach().requires_grad_(True)}, data


def test_prompt_grouping_key_reaches_the_loss_through_ppo_loss():
    """The ``uid`` of the micro-batch must reach TVPO, otherwise 'prompt' silently means 'batch'."""
    from verl.workers.utils.losses import ppo_loss

    # group "a" stays inside the trust region, group "b" leaves it
    ratios = [[1.1, 1.1, 1.1], [1.1, 1.1], [3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    uids = ["a", "a", "b", "b"]

    prompt_cfg = _actor_config(clip_ratio=0.5, tv_scope="prompt")
    model_output, data = _micro_batch(ratios, uids)
    _, prompt_metrics = ppo_loss(prompt_cfg, model_output, data)

    batch_cfg = _actor_config(clip_ratio=0.5, tv_scope="batch")
    model_output, data = _micro_batch(ratios, uids)
    _, batch_metrics = ppo_loss(batch_cfg, model_output, data)

    def clipfrac(metrics):
        value = metrics["actor/pg_clipfrac"]
        return value.aggregate() if hasattr(value, "aggregate") else value

    # every token moved up with a positive advantage, so batch scope masks all of them; under
    # prompt scope the five tokens of group "a" are still inside their own trust region
    assert clipfrac(batch_metrics) == pytest.approx(1.0)
    assert clipfrac(prompt_metrics) == pytest.approx(7.0 / 12.0)
