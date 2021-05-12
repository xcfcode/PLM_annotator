# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod,
    lengths_to_mask,
)
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import MultiheadAttention
from fairseq.utils import convert_padding_direction

from . import register_monotonic_attention


@with_incremental_state
class MonotonicAttention(nn.Module):
    """
    Abstract class of monotonic attentions
    """

    def __init__(self, args):
        self.eps = args.attention_eps
        self.mass_preservation = args.mass_preservation

        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var

        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if args.energy_bias is True
            else 0
        )

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--no-mass-preservation', action="store_false", dest="mass_preservation",
                            help='Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action="store_true", dest="mass_preservation",
                            help='Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)

        parser.add_argument('--noise-var', type=float, default=1.0,
                            help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0,
                            help='Mean of discretness noise')
        parser.add_argument('--energy-bias', action="store_true", default=False,
                            help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
                            help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-6,
                            help='Epsilon when calculating expected attention')
        # fmt: on

    def p_choose(self, *args):
        raise NotImplementedError

    def input_projections(self, *args):
        raise NotImplementedError

    def attn_energy(self, q_proj, k_proj, key_padding_mask=None):
        """
        Calculating monotonic energies

        ============================================================
        Expected input size
        q_proj: bsz * num_heads, tgt_len, self.head_dim
        k_proj: bsz * num_heads, src_len, self.head_dim
        key_padding_mask: bsz, src_len
        attn_mask: tgt_len, src_len
        """
        bsz, tgt_len, embed_dim = q_proj.size()
        bsz = bsz // self.num_heads
        src_len = k_proj.size(1)

        attn_energy = torch.bmm(q_proj, k_proj.transpose(1, 2)) + self.energy_bias

        attn_energy = attn_energy.view(bsz, self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            attn_energy = attn_energy.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).bool(),
                float("-inf"),
            )

        return attn_energy

    def expected_alignment_train(self, p_choose, key_padding_mask):
        """
        Calculating expected alignment for MMA
        Mask is not need because p_choose will be 0 if masked

        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij

        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        """

        # p_choose: bsz * num_heads, tgt_len, src_len
        bsz_num_heads, tgt_len, src_len = p_choose.size()

        # cumprod_1mp : bsz * num_heads, tgt_len, src_len
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=self.eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, self.eps, 1.0)

        init_attention = p_choose.new_zeros([bsz_num_heads, 1, src_len])
        init_attention[:, :, 0] = 1.0

        previous_attn = [init_attention]

        for i in range(tgt_len):
            # p_choose: bsz * num_heads, tgt_len, src_len
            # cumprod_1mp_clamp : bsz * num_heads, tgt_len, src_len
            # previous_attn[i]: bsz * num_heads, 1, src_len
            # alpha_i: bsz * num_heads, src_len
            alpha_i = (
                p_choose[:, i]
                * cumprod_1mp[:, i]
                * torch.cumsum(previous_attn[i][:, 0] / cumprod_1mp_clamp[:, i], dim=1)
            ).clamp(0, 1.0)
            previous_attn.append(alpha_i.unsqueeze(1))

        # alpha: bsz * num_heads, tgt_len, src_len
        alpha = torch.cat(previous_attn[1:], dim=1)

        if self.mass_preservation:
            # Last token has the residual probabilities
            alpha[:, :, -1] = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0.0, 1.0)

        assert not torch.isnan(alpha).any(), "NaN detected in alpha."

        return alpha

    def expected_alignment_infer(self, p_choose, key_padding_mask, incremental_state):
        """
        Calculating mo alignment for MMA during inference time

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        key_padding_mask: bsz * src_len
        incremental_state: dict
        """
        # p_choose: bsz * self.num_heads, src_len
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        # One token at a time
        assert tgt_len == 1
        p_choose = p_choose[:, 0, :]

        monotonic_cache = self._get_monotonic_buffer(incremental_state)

        # prev_monotonic_step: bsz, num_heads
        bsz = bsz_num_heads // self.num_heads
        prev_monotonic_step = monotonic_cache.get(
            "step", p_choose.new_zeros([bsz, self.num_heads]).long()
        )
        bsz, num_heads = prev_monotonic_step.size()
        assert num_heads == self.num_heads
        assert bsz * num_heads == bsz_num_heads

        # p_choose: bsz, num_heads, src_len
        p_choose = p_choose.view(bsz, num_heads, src_len)

        if key_padding_mask is not None:
            src_lengths = src_len - key_padding_mask.sum(dim=1, keepdim=True).long()
        else:
            src_lengths = prev_monotonic_step.new_ones(bsz, 1) * src_len

        # src_lengths: bsz, num_heads
        src_lengths = src_lengths.expand_as(prev_monotonic_step)
        # new_monotonic_step: bsz, num_heads
        new_monotonic_step = prev_monotonic_step

        step_offset = 0
        if key_padding_mask is not None:
            if key_padding_mask[:, 0].any():
                # left_pad_source = True:
                step_offset = key_padding_mask.sum(dim=-1, keepdim=True)

        max_steps = src_lengths - 1 if self.mass_preservation else src_lengths

        # finish_read: bsz, num_heads
        finish_read = new_monotonic_step.eq(max_steps)

        while finish_read.sum().item() < bsz * self.num_heads:
            # p_choose: bsz * self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: bsz , self.num_heads
            p_choose_i = (
                p_choose.gather(
                    2,
                    (step_offset + new_monotonic_step)
                    .unsqueeze(2)
                    .clamp(0, src_len - 1),
                )
            ).squeeze(2)

            action = (
                (p_choose_i < 0.5)
                .type_as(prev_monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # 1 x bsz
            # sample actions on unfinished seq
            # 1 means stay, finish reading
            # 0 means leave, continue reading
            # dist = torch.distributions.bernoulli.Bernoulli(p_choose)
            # action = dist.sample().type_as(finish_read) * (1 - finish_read)

            new_monotonic_step += action

            finish_read = new_monotonic_step.eq(max_steps) | (action == 0)
            # finish_read = (~ (finish_read.sum(dim=1, keepdim=True) < self.num_heads / 2)) | finish_read

        monotonic_cache["step"] = new_monotonic_step

        # alpha: bsz * num_heads, 1, src_len
        # new_monotonic_step: bsz, num_heads
        alpha = p_choose.new_zeros([bsz * self.num_heads, src_len]).scatter(
            1,
            (step_offset + new_monotonic_step)
            .view(bsz * self.num_heads, 1)
            .clamp(0, src_len - 1),
            1,
        )

        if not self.mass_preservation:
            alpha = alpha.masked_fill(
                (new_monotonic_step == max_steps).view(bsz * self.num_heads, 1), 0
            )

        alpha = alpha.unsqueeze(1)

        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        return alpha

    def v_proj_output(self, value):
        raise NotImplementedError

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        *args,
        **kwargs,
    ):

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        # stepwise prob
        # p_choose: bsz * self.num_heads, tgt_len, src_len
        p_choose = self.p_choose(query, key, key_padding_mask)

        # expected alignment alpha
        # bsz * self.num_heads, tgt_len, src_len
        if incremental_state is not None:
            alpha = self.expected_alignment_infer(
                p_choose, key_padding_mask, incremental_state
            )
        else:
            alpha = self.expected_alignment_train(p_choose, key_padding_mask)

        # expected attention beta
        # bsz * self.num_heads, tgt_len, src_len
        beta = self.expected_attention(
            alpha, query, key, value, key_padding_mask, incremental_state
        )

        attn_weights = beta

        v_proj = self.v_proj_output(value)
        attn = torch.bmm(attn_weights.type_as(v_proj), v_proj)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {"alpha": alpha, "beta": beta, "p_choose": p_choose}

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(incremental_state, new_order)
        input_buffer = self._get_monotonic_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_monotonic_buffer(incremental_state, input_buffer)

    def _get_monotonic_buffer(self, incremental_state):
        return (
            utils.get_incremental_state(
                self,
                incremental_state,
                "monotonic",
            )
            or {}
        )

    def _set_monotonic_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            "monotonic",
            buffer,
        )

    def get_pointer(self, incremental_state):
        return (
            utils.get_incremental_state(
                self,
                incremental_state,
                "monotonic",
            )
            or {}
        )

    def get_fastest_pointer(self, incremental_state):
        return self.get_pointer(incremental_state)["step"].max(0)[0]

    def set_pointer(self, incremental_state, p_choose):
        curr_pointer = self.get_pointer(incremental_state)
        if len(curr_pointer) == 0:
            buffer = torch.zeros_like(p_choose)
        else:
            buffer = self.get_pointer(incremental_state)["step"]

        buffer += (p_choose < 0.5).type_as(buffer)

        utils.set_incremental_state(
            self,
            incremental_state,
            "monotonic",
            {"step": buffer},
        )


@register_monotonic_attention("hard_aligned")
class MonotonicMultiheadAttentionHard(MonotonicAttention, MultiheadAttention):
    def __init__(self, args):
        MultiheadAttention.__init__(
            self,
            embed_dim=args.decoder_embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        MonotonicAttention.__init__(self, args)

        self.k_in_proj = {"monotonic": self.k_proj}
        self.q_in_proj = {"monotonic": self.q_proj}
        self.v_in_proj = {"output": self.v_proj}

    def input_projections(self, query, key, value, name):
        """
        Prepare inputs for multihead attention

        ============================================================
        Expected input size
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        name: monotonic or soft
        """

        if query is not None:
            bsz = query.size(1)
            q = self.q_in_proj[name](query)
            q *= self.scaling
            q = (
                q.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            q = None

        if key is not None:
            bsz = key.size(1)
            k = self.k_in_proj[name](key)
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            k = None

        if value is not None:
            bsz = value.size(1)
            v = self.v_in_proj[name](value)
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        else:
            v = None

        return q, k, v

    def p_choose(self, query, key, key_padding_mask=None):
        """
        Calculating step wise prob for reading and writing
        1 to read, 0 to write

        ============================================================
        Expected input size
        query: bsz, tgt_len, embed_dim
        key: bsz, src_len, embed_dim
        value: bsz, src_len, embed_dim
        key_padding_mask: bsz, src_len
        attn_mask: bsz, src_len
        query: bsz, tgt_len, embed_dim
        """

        # prepare inputs
        q_proj, k_proj, _ = self.input_projections(query, key, None, "monotonic")

        # attention energy
        attn_energy = self.attn_energy(q_proj, k_proj, key_padding_mask)

        noise = 0

        if self.training:
            # add noise here to encourage discretness
            noise = (
                torch.normal(self.noise_mean, self.noise_var, attn_energy.size())
                .type_as(attn_energy)
                .to(attn_energy.device)
            )

        p_choose = torch.sigmoid(attn_energy + noise)
        _, _, tgt_len, src_len = p_choose.size()

        # p_choose: bsz * self.num_heads, tgt_len, src_len
        return p_choose.view(-1, tgt_len, src_len)

    def expected_attention(self, alpha, *args):
        """
        For MMA-H, beta = alpha
        """
        return alpha

    def v_proj_output(self, value):
        _, _, v_proj = self.input_projections(None, None, value, "output")
        return v_proj


@register_monotonic_attention("infinite_lookback")
class MonotonicMultiheadAttentionInfiniteLookback(MonotonicMultiheadAttentionHard):
    def __init__(self, args):
        super().__init__(args)
        self.init_soft_attention()

    def init_soft_attention(self):
        self.k_proj_soft = nn.Linear(self.kdim, self.embed_dim, bias=True)
        self.q_proj_soft = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_in_proj["soft"] = self.k_proj_soft
        self.q_in_proj["soft"] = self.q_proj_soft

        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(
                self.q_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
        else:
            nn.init.xavier_uniform_(self.k_in_proj["soft"].weight)
            nn.init.xavier_uniform_(self.q_in_proj["soft"].weight)

    def expected_attention(
        self, alpha, query, key, value, key_padding_mask, incremental_state
    ):
        # monotonic attention, we will calculate milk here
        bsz_x_num_heads, tgt_len, src_len = alpha.size()
        bsz = int(bsz_x_num_heads / self.num_heads)

        q, k, _ = self.input_projections(query, key, None, "soft")
        soft_energy = self.attn_energy(q, k, key_padding_mask)

        assert list(soft_energy.size()) == [bsz, self.num_heads, tgt_len, src_len]

        soft_energy = soft_energy.view(bsz * self.num_heads, tgt_len, src_len)

        if incremental_state is not None:
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            monotonic_step = monotonic_cache["step"] + 1
            step_offset = 0
            if key_padding_mask is not None:
                if key_padding_mask[:, 0].any():
                    # left_pad_source = True:
                    step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
            monotonic_step += step_offset
            mask = lengths_to_mask(
                monotonic_step.view(-1), soft_energy.size(2), 1
            ).unsqueeze(1)

            soft_energy = soft_energy.masked_fill(~mask.bool(), float("-inf"))
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_sum = exp_soft_energy.sum(dim=2)
            beta = exp_soft_energy / exp_soft_energy_sum.unsqueeze(2)

        else:
            # bsz * num_heads, tgt_len, src_len
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_cumsum = torch.cumsum(exp_soft_energy, dim=2)

            if key_padding_mask is not None:
                if key_padding_mask.any():
                    exp_soft_energy_cumsum = (
                        exp_soft_energy_cumsum.view(
                            -1, self.num_heads, tgt_len, src_len
                        )
                        .masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(1), self.eps
                        )
                        .view(-1, tgt_len, src_len)
                    )

            inner_items = alpha / exp_soft_energy_cumsum

            beta = exp_soft_energy * torch.cumsum(
                inner_items.flip(dims=[2]), dim=2
            ).flip(dims=[2])

            beta = self.dropout_module(beta)

        assert not torch.isnan(beta).any(), "NaN detected in beta."

        return beta


@register_monotonic_attention("waitk")
class MonotonicMultiheadAttentionWaitk(MonotonicMultiheadAttentionInfiniteLookback):
    def __init__(self, args):
        super().__init__(args)
        self.q_in_proj["soft"] = self.q_in_proj["monotonic"]
        self.k_in_proj["soft"] = self.k_in_proj["monotonic"]
        self.waitk_lagging = args.waitk_lagging
        assert (
            self.waitk_lagging > 0
        ), f"Lagging has to been larger than 0, get {self.waitk_lagging}."

    @staticmethod
    def add_args(parser):
        super(
            MonotonicMultiheadAttentionWaitk,
            MonotonicMultiheadAttentionWaitk,
        ).add_args(parser)

        parser.add_argument(
            "--waitk-lagging", type=int, required=True, help="Wait k lagging"
        )

    def p_choose(
        self, query, key, key_padding_mask=None, attn_mask=None, incremental_state=None
    ):
        """
        query: bsz, tgt_len
        key: bsz, src_len
        key_padding_mask: bsz, src_len
        """
        src_len, bsz, _ = key.size()
        tgt_len, bsz, _ = query.size()
        p_choose = query.new_ones(bsz, tgt_len, src_len)
        p_choose = torch.tril(p_choose, diagonal=self.waitk_lagging - 1)
        p_choose = torch.triu(p_choose, diagonal=self.waitk_lagging - 1)

        if key_padding_mask is not None and key_padding_mask[:, 0].eq(1).any():
            # Left pad source
            # add -1 to the end
            p_choose = p_choose.masked_fill(
                key_padding_mask.float().flip(1).unsqueeze(1).bool(), -1
            )
            p_choose = convert_padding_direction(
                p_choose.view(-1, src_len).long(), padding_idx=-1, right_to_left=True
            )
            p_choose = p_choose.view(bsz, tgt_len, src_len).type_as(query)
            # remove -1
            p_choose[p_choose.eq(-1)] = 0

        # Extend to each head
        p_choose = (
            p_choose.contiguous()
            .unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .contiguous()
            .view(-1, tgt_len, src_len)
        )

        return p_choose
