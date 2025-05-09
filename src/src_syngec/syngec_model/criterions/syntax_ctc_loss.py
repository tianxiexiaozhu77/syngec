# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from math import log

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
import numpy as np

'''
1.LabelSmoothedCTCCriterion(FairseqCriterion)改为SyntaxLabelSmoothedCTCCriterion
2.@register_criterion("ctc_loss")改为@register_criterion("syntax_ctc_loss")
3.
###############################--改↓--################################
        # outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat)
        outputs = model(src_tokens=src_tokens, src_lengths=src_lengths, 
        prev_output_tokens=prev_output_tokens, tgt_tokens=tgt_tokens,
        src_outcoming_arc_mask=sample["net_input"]['src_outcoming_arc_mask'], 
        src_incoming_arc_mask=sample["net_input"]['src_incoming_arc_mask'],
        src_dpd_matrix=sample["net_input"]['src_dpd_matrix'], 
        src_probs_matrix=sample["net_input"]['src_probs_matrix'],
        glat=glat,
        )
################################--改↑--###############################
4.在/opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model/criterions/__init__.py下添加
syntax_ctc_loss as _,
'''


@register_criterion("syntax_ctc_loss")
class SyntaxLabelSmoothedCTCCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.blank_id = task.tgt_dict.unk()
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_id, reduction='none', zero_infinity=True)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument('--mse-lambda', default=10, type=float, metavar='D')

    def _compute_loss(
            self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_ctc_loss(
            self, outputs, targets, output_mask, target_mask, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        output_lens = output_mask.sum(-1)
        target_lens = target_mask.sum(-1)
        lprobs = F.log_softmax(outputs, dim=-1, dtype=torch.float32)
        losses = self.ctc_loss(lprobs.transpose(0, 1), targets, output_lens, target_lens)
        ctc_loss = losses.sum()/(output_lens.sum().float())
        lprobs = lprobs[output_mask]
        if label_smoothing > 0:
            loss = (
                    ctc_loss * (1 - label_smoothing) - mean_ds(lprobs) * label_smoothing
            )
        else:
            loss = ctc_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "ctc_loss": ctc_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]
        if 'glat' in sample:
            glat = sample['glat']
        else:
            glat = None
        prev_output_tokens = model.initialize_ctc_input(src_tokens)
###############################--改↓--################################
        # outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat)
        outputs = model(src_tokens=src_tokens, src_lengths=src_lengths, 
        prev_output_tokens=prev_output_tokens, tgt_tokens=tgt_tokens,
        src_outcoming_arc_mask=sample["net_input"]['src_outcoming_arc_mask'], 
        src_incoming_arc_mask=sample["net_input"]['src_incoming_arc_mask'],
        src_dpd_matrix=sample["net_input"]['src_dpd_matrix'], 
        src_probs_matrix=sample["net_input"]['src_probs_matrix'],
        glat=glat,
        )
################################--改↑--###############################
        losses, ctc_loss = [], []

        for obj in outputs:
            if obj.startswith('glat'):
                continue
            if obj == "word_ins":
                _losses = self._compute_ctc_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    prev_output_tokens.ne(self.task.tgt_dict.pad()),
                    tgt_tokens.ne(self.task.tgt_dict.pad()),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("ctc_loss", False):
                ctc_loss += [_losses.get("ctc_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        ctc_loss = sum(l for l in ctc_loss) if len(ctc_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ctc_loss": ctc_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if "glat_accu" in outputs:
            logging_output["glat_accu"] = outputs['glat_accu']
        if "glat_context_p" in outputs:
            logging_output['glat_context_p'] = outputs['glat_context_p']

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        log_metric("glat_accu", logging_outputs)
        log_metric("glat_context_p", logging_outputs)

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def log_metric(key, logging_outputs):
    if len(logging_outputs) > 0 and key in logging_outputs[0]:
        metrics.log_scalar(
            key, utils.item(np.mean([log.get(key, 0) for log in logging_outputs])), priority=10, round=3
        )