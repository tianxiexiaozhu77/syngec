# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATSharedDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Union
import logging
import random, math
# from .nat_sd_ss import NATransformerDecoder
from .dslp_syntax_shared import DSLPNATransformerDecoder
from .syntax_glat_nonautoregressive import SyntaxGlatNATransformerEncoder
from fairseq.torch_imputer import best_alignment, imputer_loss
from lunanlp import torch_seed


import math
import torch.nn as nn
from fairseq.models.fairseq_encoder import GlatEncoderOut, EncoderOut
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    GATSyntaxGuidedTransformerEncoderLayer,  # 已添加
    GCNSyntaxGuidedTransformerEncoderLayer,  # 已添加
    ZYGCNSyntaxGuidedTransformerEncoderLayer,  # 已添加
    DSATransformerEncoderLayer,  # 已添加
    GradMultiply,
)
from .syntax_enhanced_fairseq_nat_model import SyntaxFairseqNATModel, ensemble_decoder
logger = logging.getLogger(__name__)

'''
所有的修改
1.
    # def forward(
    #         self, src_tokens, src_lengths, prev_output_tokens, 
    #         tgt_tokens, reduce=True, train_ratio=None, **kwargs
    # ):改为
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
        src_outcoming_arc_mask, 
        src_incoming_arc_mask,
        src_dpd_matrix, 
        src_probs_matrix,
        train_ratio=None, **kwargs):
2. @register_model("nat_ctc_sd_ss")改为@register_model("nat_ctc_sd_ss_syntax")
3."at_ctc_sd_ss"改为"at_ctc_sd_ss_syntax"
@register_model_architecture(
    "nat_ctc_sd_ss_syntax", "nat_ctc_sd_ss_syntax"
)
4.from ... import ... row24-42
5.
    @staticmethod
    def add_args(parser):
        SyntaxFairseqNATModel.add_args(parser)
6.添加build_encoder方法
    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        encoder = SyntaxNATransformerEncoder(args, tgt_dict, embed_tokens, syntax_label_dict, embed_rels)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder
7.
8. 
9.
        # encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # 改为
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths,
                            src_outcoming_arc_mask=src_outcoming_arc_mask,
                            src_incoming_arc_mask=src_incoming_arc_mask,
                            src_dpd_matrix=src_dpd_matrix,
                            src_probs_matrix=src_probs_matrix,
                            **kwargs)  # 其他的参数通过**kwargs这里添加进去的
10. 在__init__下添加
from .nat_ctc_sd_ss_syntax import *
'''

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


def _gumbel_softmax(logits, temperature=1.0, withnoise=True, hard=True, eps=1e-10):
    def sample_gumbel(shape, eps=1e-10):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    if withnoise:
        gumbels = sample_gumbel(logits.size())
        y_soft = ((logits + gumbels) * 1.0 / temperature).softmax(2)
    else:
        y_soft = (logits * 1.0 / temperature).softmax(2)

    index = y_soft.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(2, index, 1.0)
    ret = (y_hard - y_soft).detach() + y_soft
    if hard:
        return ret
    else:
        return y_soft


@register_model("nat_ctc_sd_ss_syntax")
class NATransformerModel(SyntaxFairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        from ctcdecode import CTCBeamDecoder
        self.ctc_decoder = CTCBeamDecoder(
            decoder.dictionary.symbols,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=args.ctc_beam_size,
            num_processes=20,
            blank_id=decoder.dictionary.blank_index,
            log_probs_input=False
        )
        self.inference_decoder_layer = getattr(args, 'inference_decoder_layer', -1)
        self.plain_ctc = getattr(args, 'plain_ctc', False)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        SyntaxFairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--src-upsample-scale",
            type=int,
            default=1
        )
        parser.add_argument(
            '--use-ctc-decoder',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--ctc-beam-size',
            default=1,
            type=int
        )
        parser.add_argument(
            '--ctc-beam-size-train',
            default=1,
            type=int
        )
        parser.add_argument(
            '--num-cross-layer-sample',
            default=0,
            type=int
        )
        # parser.add_argument(
        #     '--hard-argmax',
        #     action='store_true',
        #     default=False
        # )
        # parser.add_argument(
        #     '--yhat-temp',
        #     type=float,
        #     default=0.1
        # )
        parser.add_argument(
            '--inference-decoder-layer',
            type=int,
            default=-1
        )
        parser.add_argument(
            '--share-ffn',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--share-attn',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--sample-option',
            type=str,
            default='hard'
        )
        parser.add_argument(
            '--softmax-temp',
            type=float,
            default=1
        )
        parser.add_argument(
            '--temp-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--num-topk',
            default=1,
            type=int
        )
        parser.add_argument(
            '--copy-src-token',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--force-detach',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy-temp',
            default=5,
            type=float
        )
        parser.add_argument(
            '--concat-yhat',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--concat-dropout',
            type=float,
            default=0
        )
        parser.add_argument(
            '--layer-drop-ratio',
            type=float,
            default=0.0
        )
        parser.add_argument(
            '--all-layer-drop',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--yhat-posemb',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal-end-ratio',
            type=float,
            default=0
        )
        parser.add_argument(
            '--force-ls',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--repeat-layer',
            type=int,
            default=0
        )
        parser.add_argument(
            '--masked-loss',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--ss-ratio',
            type=float,
            default=0.3
        )
        parser.add_argument(
            '--fixed-ss-ratio',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--no-empty',
            action='store_true',
            default=False
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        encoder = SyntaxGlatNATransformerEncoder(args, tgt_dict, embed_tokens, syntax_label_dict, embed_rels)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DSLPNATransformerDecoder(args, tgt_dict, embed_tokens)
        decoder.repeat_layer = getattr(args, 'repeat_layer', 0)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True,
                                      force_emit=None
                                      ) -> torch.FloatTensor:
        # # lengths : (batch_size, )
        # if self.args.force_ls:  # NOTE temp fix, to really try ls without mess up previous exps
        #     label_smoothing = self.args.label_smoothing
        # else:
        #     label_smoothing = label_smoothing
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        

        if self.args.masked_loss and force_emit is not None:
            loss = imputer_loss(
                    log_probs_T.float(),
                    targets,
                    force_emit,
                    logit_lengths,
                    target_lengths,
                    blank=blank_index,
                    reduction="mean",
                    zero_infinity=True,
                )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",
                zero_infinity=True,
            )

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss

    # def forward(
    #         self, src_tokens, src_lengths, prev_output_tokens, 
    #         tgt_tokens, reduce=True, train_ratio=None, **kwargs
    # ):
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
        src_outcoming_arc_mask, 
        src_incoming_arc_mask,
        src_dpd_matrix, 
        src_probs_matrix,
        train_ratio=None, reduce=True, **kwargs):

        if train_ratio is not None:
            self.encoder.train_ratio = train_ratio
            self.decoder.train_ratio = train_ratio

        prev_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        target_mask = tgt_tokens.ne(self.pad)
        # encoding
        # encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths,
                            src_outcoming_arc_mask=src_outcoming_arc_mask,
                            src_incoming_arc_mask=src_incoming_arc_mask,
                            src_dpd_matrix=src_dpd_matrix,
                            src_probs_matrix=src_probs_matrix,
                            **kwargs)  # 其他的参数通过**kwargs这里添加进去的

        
        rand_seed = random.randint(0, 19260817)
        if train_ratio is not None:
            # NOTE: Find best CTC alignments as pseudo GT
            with torch.no_grad():
                # decoding
                with torch_seed(rand_seed):
                    output_logits_list = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                        train_ratio=train_ratio
                    )
                target_mask = tgt_tokens.ne(self.pad)
                target_length = target_mask.sum(dim=-1) 

                output_masks = prev_output_tokens.ne(self.pad)
                output_length = output_masks.sum(dim=-1)

                logits_T = output_logits_list[-1].transpose(0, 1).float()

                best_aligns = best_alignment(logits_T, tgt_tokens, output_length, target_length, self.pad, zero_infinity=True)
                best_aligns_pad = torch.tensor([a + [0] * (logits_T.size(0) - len(a)) for a in best_aligns], device=logits_T.device, dtype=tgt_tokens.dtype)
                oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1]-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                if not self.args.no_empty:
                    oracle = oracle.masked_fill(best_aligns_pad % 2 == 0, self.tgt_dict.mask_index)
            

            if self.args.fixed_ss_ratio:
                ss_ratio = self.args.ss_ratio
            else:
                ss_ratio = self.args.ss_ratio * (1 - train_ratio)
                
            ss_mask = (torch.rand(oracle.size(), device=oracle.device) < ss_ratio).bool()
            force_emit = best_aligns_pad.masked_fill(~ss_mask, -1)
        else:
            ss_mask = None
            force_emit = None
            oracle = None


        with torch_seed(rand_seed):
            output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=oracle,
            ss_mask=ss_mask
        )

        if self.args.num_cross_layer_sample != 0:
            output_logits_list = torch.stack(output_logits_list, dim=0)

            N_SAMPLE = self.args.num_cross_layer_sample

            num_decoder_layer = output_logits_list.size(0)
            num_tokens = prev_output_tokens.size(1)
            num_vocab = output_logits_list.size(-1)
            batch_size = prev_output_tokens.size(0)

            all_sample_ctc_loss = 0

            for _ in range(N_SAMPLE):
                cross_layer_sampled_ids_ts = torch.randint(num_decoder_layer, (batch_size * num_tokens,), device=output_logits_list.device)
                output_logits_list = output_logits_list.view(num_decoder_layer, -1, num_vocab)
                gather_idx = cross_layer_sampled_ids_ts.unsqueeze(1).expand(-1, num_vocab).unsqueeze(0)
                gather_logits = output_logits_list.gather(0, gather_idx).view(batch_size, num_tokens, num_vocab)

                # if self.args.use_ctc:
                ctc_loss = self.sequence_ctc_loss_with_logits(
                    logits=gather_logits,
                    logit_mask=prev_output_tokens_mask,
                    targets=tgt_tokens,
                    target_mask=target_mask,
                    blank_index=self.tgt_dict.blank_index,
                    label_smoothing=self.args.label_smoothing,
                    reduce=reduce,
                    force_emit=force_emit
                )
                all_sample_ctc_loss += ctc_loss

            ret_val = {
                "ctc_loss": {"loss": all_sample_ctc_loss / len(output_logits_list)},
            }

        else:
            # if self.args.use_ctc:
            all_layer_ctc_loss = 0
            normalized_factor = 0
            for layer_idx, word_ins_out in enumerate(output_logits_list):
                ctc_loss = self.sequence_ctc_loss_with_logits(
                    logits=word_ins_out,
                    logit_mask=prev_output_tokens_mask,
                    targets=tgt_tokens,
                    target_mask=target_mask,
                    blank_index=self.tgt_dict.blank_index,
                    label_smoothing=self.args.label_smoothing, #NOTE: enable and double check with it later
                    reduce=reduce,
                    force_emit=force_emit
                )
                factor = 1  # math.sqrt(layer_idx + 1)
                all_layer_ctc_loss += ctc_loss * factor
                normalized_factor += factor
            ret_val = {
                "ctc_loss": {"loss": all_layer_ctc_loss / normalized_factor},
            }


        softcopy_learnable = getattr(self.decoder, "softcopy_learnable", False)
        if softcopy_learnable:
            ret_val.update(
                {
                    "stat:softcopy_temp": self.decoder.para_softcopy_temp.item()
                }
            )

        return ret_val

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # set CTC decoder beam size
        self.ctc_decoder._beam_width = self.args.ctc_beam_size
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history

        # execute the decoder
        output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )

        inference_decoder_layer = self.inference_decoder_layer
        output_logits = output_logits_list[inference_decoder_layer]

        if self.plain_ctc:
            output_scores = decoder_out.output_scores
            _scores, _tokens = output_logits.max(-1)
            output_masks = output_tokens.ne(self.pad)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            )
        else:
            # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
            # _scores == beam_results[:,0,:]
            output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                     output_length)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
            # output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=top_beam_tokens.to(output_logits.device),
                output_scores=torch.full(top_beam_tokens.size(), 1.0),
                attn=None,
                history=history,
            )

    def get_search_results(self, decoder_out, encoder_out, beam_size=None, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history
        # Set ctc beam size
        if beam_size is not None:
            self.ctc_decoder._beam_width = beam_size
        else:
            beam_size = self.ctc_decoder._beam_width

        # execute the decoder
        output_logits = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
        # _scores == beam_results[:,0,:]
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                 output_length)

        beam_results = beam_results[:, :, :out_lens.max()]
        beam_size = beam_scores.size(1)
        for beam_idx in range(beam_size):
            top_beam_tokens = beam_results[:, beam_idx, :]
            top_beam_len = out_lens[:, beam_idx]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
        return beam_results, beam_scores

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.args.copy_src_token:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if self.args.src_upsample_scale > 2:
                length_tgt = length_tgt * self.args.src_upsample_scale
            else:
                length_tgt = length_tgt * self.args.src_upsample_scale  # + 10
            max_length = length_tgt.clamp_(min=2).max()
            idx_length = utils.new_arange(src_tokens, max_length)

            initial_output_tokens = src_tokens.new_zeros(
                src_tokens.size(0), max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
            return initial_output_tokens
        else:
            if self.args.src_upsample_scale <= 1:
                return src_tokens

            def _us(x, s):
                B = x.size(0)
                _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
                return _x

            return _us(src_tokens, self.args.src_upsample_scale)

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        initial_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

#################################以下为编码器row731-1675####################################
class SyntaxNATransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, src_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        self.args = args
        super().__init__(src_dict)
        self.sentence_encoder = self.build_sentence_encoder(args, src_dict, embed_tokens)
        if syntax_label_dict is not None:
            syntax_type = ["dep"] * len(syntax_label_dict) if len(self.args.syntax_type) < len(
                syntax_label_dict) else self.args.syntax_type
        if self.args.use_syntax:
            if len(syntax_label_dict) > 1:
                self.syntax_encoder = nn.ModuleList([])  # 加入异构句法的支持，每种句法分别用一个GCN编码，最后将所有表示拼接
                for i, (d, e) in enumerate(zip(syntax_label_dict, embed_rels)):
                    self.syntax_encoder.append(
                        self.build_syntax_guided_encoder(args, d, e, syntax_type[i]))  # 创建syntax encoder模块
            else:
                self.syntax_encoder = self.build_syntax_guided_encoder(args, syntax_label_dict[0], embed_rels,
                                                                       syntax_type[0])  ## 看看这个走的哪里
            if getattr(args, 'gated_sum', False):  # 看看有没有进来，是否需要了解apply_quant_noise_
                self.quant_noise = getattr(args, "quant_noise_pq", 0)
                self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
                self.fc = self.build_fc(2 * args.encoder_embed_dim, 1, self.quant_noise,
                                        self.quant_noise_block_size)  # 对应论文公式(7)的FFN层
                self.fc1 = self.build_fc(2 * args.encoder_embed_dim, args.encoder_embed_dim, self.quant_noise,
                                         self.quant_noise_block_size)

    def build_sentence_encoder(self, args, dictionary, embed):
        return SyntaxSentenceNATransformerEncoder(args, dictionary, embed)

    def build_syntax_guided_encoder(self, args, dictionary, embed, syntax_type="dep"):
        return SyntaxSyntaxNATransformerEncoder(args, dictionary, embed, syntax_type)

    def build_fc(self, input_dim, output_dim, q_noise=None, qn_block_size=None):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    # @staticmethod
    # @torchsnooper.snoop()
    def dual_aggregation(self, o1, o2, beta=0.5):
        if self.args.cross_syntax_fuse:  # F
            sentence_encoder_out = o1.encoder_out
            final_encoder_out = []
            for syntax_encoder_out in o2:
                final_encoder_out.append(beta * sentence_encoder_out + (1.0 - beta) * syntax_encoder_out.encoder_out)
            # return GlatEncoderOut(
            #     encoder_out=final_encoder_out,  # T x B x C
            #     encoder_padding_mask=o1.encoder_padding_mask,  # B x T
            #     encoder_embedding=o1.encoder_embedding,  # B x T x C
            #     encoder_pos=o1.encoder_pos,
            #     length_out=o1.length_out,
            #     encoder_states=o1.encoder_states,  # List[T x B x C]
            #     src_tokens=None,
            #     src_lengths=None,
            # )
            return {
                "encoder_out": [final_encoder_out],  # T x B x C
                "encoder_padding_mask": [o1.encoder_padding_mask],  # B x T
                "encoder_embedding": [o1.encoder_embedding],  # B x T x C
                "encoder_pos": [o1.encoder_pos],
                "length_out": [o1.length_out],
                "encoder_states": o1.encoder_states,  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
            }
        else:  # 走这里
            sentence_encoder_out = o1.encoder_out
            if len(o2) == 1:
                syntax_encoder_out = o2[0].encoder_out
            elif len(o2) == 2:  # F
                gate_value = torch.sigmoid(self.fc(torch.cat([o2[0].encoder_out, o2[1].encoder_out], -1))) if getattr(
                    self.args, 'gated_sum', False) else 0.5
                syntax_encoder_out = gate_value * o2[0].encoder_out + (1 - gate_value) * o2[0].encoder_out
            else:
                syntax_encoder_out = o2[0].encoder_out
                for i in range(1, len(o2)):
                    syntax_encoder_out = syntax_encoder_out + o2[i].encoder_out
            if self.training and self.args.scale_syntax_encoder_lr is not None:  # F
                syntax_encoder_out = GradMultiply.apply(syntax_encoder_out, self.args.scale_syntax_encoder_lr)
            final_encoder_out = beta * sentence_encoder_out + (1.0 - beta) * syntax_encoder_out
            # x = torch.cat([sentence_encoder_out,syntax_encoder_out], dim=-1)
            # final_encoder_out = self.fc1(x)
            #############
            # return GlatEncoderOut(
            #     encoder_out=final_encoder_out,  # T x B x C
            #     encoder_padding_mask=o1.encoder_padding_mask,  # B x T
            #     encoder_embedding=o1.encoder_embedding,  # B x T x C
            #     encoder_pos=o1.encoder_pos,  # B x T x C
            #     length_out=o1.length_out,  # B x C
            #     encoder_states=o1.encoder_states,  # List[T x B x C]
            #     src_tokens=None,
            #     src_lengths=None,
            # )
            return {
                "encoder_out": [final_encoder_out],  # T x B x C
                "encoder_padding_mask": [o1.encoder_padding_mask],  # B x T
                "encoder_embedding": [o1.encoder_embedding],  # B x T x C
                "encoder_pos": [o1.encoder_pos],  # B x T x C
                "length_out": [o1.length_out],  # B x C
                "encoder_states": o1.encoder_states,  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
            }

    # @torchsnooper.snoop()
    def forward(
            self,
            src_tokens,
            src_lengths,
            source_tokens_nt=None,
            source_tokens_nt_lengths=None,
            src_outcoming_arc_mask=None,  # 描述源端句法树，出弧（指向孩子节点）掩码矩阵
            src_incoming_arc_mask=None,  # 描述源端句法树，入弧（指向孩子节点）掩码矩阵
            src_dpd_matrix=None,  # 描述源端句法树，依存距离矩阵
            src_probs_matrix=None,  # 描述源端句法树，依存弧概率矩阵
            return_all_hiddens: bool = True,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        if (not self.args.use_syntax) or self.args.only_gnn:  # F or T -> T
            sentence_encoder_out = self.sentence_encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, src_dpd_matrix=None
            )  # B * L * D
        else:
            sentence_encoder_out = self.sentence_encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
                src_dpd_matrix=src_dpd_matrix[0]
            )  # B * L * D

        if self.args.only_dsa:  # F
            return sentence_encoder_out

        if hasattr(self, "syntax_encoder") and isinstance(self.syntax_encoder,
                                                          torch.nn.modules.container.ModuleList):  # T and F
            assert len(self.syntax_encoder) == len(src_outcoming_arc_mask)
        # 这里需要再设计一个syntax-encoder，将encoder_out传入该encoder，得到syntax-encoder结果，然后进行一个dual aggregation，得到encoder的最终结果，传入decoder
        if self.args.use_syntax:  # T
            syntax_encoder_out_all = []
            if self.training and self.args.scale_syntax_encoder_lr is not None:  # F
                h = GradMultiply.apply(sentence_encoder_out.encoder_out, 1 / self.args.scale_syntax_encoder_lr)
            else:  # T
                h = sentence_encoder_out.encoder_out
            h_t = h.transpose(0, 1)
            if isinstance(self.syntax_encoder, torch.nn.modules.container.ModuleList):  # F
                for i in range(len(self.syntax_encoder)):
                    if self.args.syntax_type[i] == "dep":
                        assert src_incoming_arc_mask[i].shape[1] == src_tokens.shape[1]  # 确保tokens和掩码矩阵是同一个batch
                        syntax_encoder_out_all.append(self.syntax_encoder[i](
                            src_tokens, h=h, src_outcoming_arc_mask=src_outcoming_arc_mask[i],
                            src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths,
                            src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens
                        ))
                    else:  # TODO:暂时不支持成分句法的异构编码，即成分句法树最多只能引入一个，但是可以和依存句法树同时使用
                        if source_tokens_nt is not None:  # 把non-terminal节点加到句尾的scheme
                            h_new, _ = self.sentence_encoder.forward_embedding(
                                source_tokens_nt)  # 原始token编码从sentence encoder来，non-terminal节点重新过一遍embedding层；
                            B, T, D = h_new.shape
                            # 加入non-terminal节点的编码
                            for j in range(B):  # 遍历每个sentence，除了Padding（不重要）和
                                L1 = src_lengths[j].item()
                                L2 = source_tokens_nt_lengths[j].item()
                                h_new[j, -L2:-(L2 - L1 + 1), :] = h_t[j, -L1:-1,
                                                                  :]  # 正常token的表示需要用sentence encoder的编码结果
                            h_new[:, -1, :] = h_t[:, -1, :]  # EOS表示也用sentence encoder的编码结果
                            h_new = h_new.transpose(0, 1)
                            # 丢弃non-terminal节点
                            res = self.syntax_encoder[i](
                                src_tokens, h=h_new, src_outcoming_arc_mask=src_outcoming_arc_mask[i],
                                src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths,
                                src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens,
                            )
                            h_syn = res.encoder_out  # T_NT x B x D
                            h_refine = h.clone().detach()  # T x B x D，记住要detach，否则会连带计算图一起复制
                            for j in range(h_refine.shape[1]):
                                L1 = src_lengths[j].item()
                                L2 = source_tokens_nt_lengths[j].item()
                                h_refine[-L1:-1, j, :] = h_syn[-L2:-(L2 - L1 + 1), j, :]  # 首先还原terminal节点
                            h_refine[-1, :, :] = h_syn[-1, :, :]  # 再还原EOS表示
                            res = EncoderOut(
                                encoder_out=h_refine,  # T x B x C
                                encoder_padding_mask=res.encoder_padding_mask,  # B x T
                                encoder_embedding=None,  # B x T x C
                                encoder_states=res.encoder_states,  # List[T x B x C]
                                src_tokens=None,
                                src_lengths=None,
                            )
                            syntax_encoder_out_all.append(res)
            else:
                if self.args.syntax_type[0] == "dep":  # T
                    assert src_incoming_arc_mask[0].shape[1] == src_tokens.shape[1]  # 确保tokens和掩码矩阵是同一个batch
                    syntax_encoder_out_all.append(self.syntax_encoder(
                        src_tokens, h=h, src_outcoming_arc_mask=src_outcoming_arc_mask[0],
                        # h是sentence_encoder_out.encoder_out
                        src_incoming_arc_mask=src_incoming_arc_mask[0], src_lengths=src_lengths,
                        src_probs_matrix=src_probs_matrix[0], return_all_hiddens=return_all_hiddens
                    ))
                else:  # constituent tree
                    if source_tokens_nt is not None:  # 把non-terminal节点加到句尾的scheme
                        i = 0
                        h_new, _ = self.sentence_encoder.forward_embedding(
                            source_tokens_nt)  # 原始token编码从sentence encoder来，non-terminal节点重新过一遍embedding层；
                        B, T, D = h_new.shape
                        # 加入non-terminal节点的编码
                        for j in range(B):  # 遍历每个sentence，除了Padding（不重要）和
                            L1 = src_lengths[j].item()
                            L2 = source_tokens_nt_lengths[j].item()
                            h_new[j, -L2:-(L2 - L1 + 1), :] = h_t[j, -L1:-1, :]  # 正常token的表示需要用sentence encoder的编码结果
                        h_new[:, -1, :] = h_t[:, -1, :]  # EOS表示也用sentence encoder的编码结果
                        h_new = h_new.transpose(0, 1)
                        # 丢弃non-terminal节点
                        res = self.syntax_encoder(
                            src_tokens, h=h_new, src_outcoming_arc_mask=src_outcoming_arc_mask[i],
                            src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths,
                            src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens,
                        )
                        h_syn = res.encoder_out  # T_NT x B x D
                        h_refine = h.clone().detach()  # T x B x D，记住要detach，否则会连带计算图一起复制
                        for j in range(h_refine.shape[1]):
                            L1 = src_lengths[j].item()
                            L2 = source_tokens_nt_lengths[j].item()
                            h_refine[-L1:-1, j, :] = h_syn[-L2:-(L2 - L1 + 1), j, :]  # 首先还原terminal节点
                        h_refine[-1, :, :] = h_syn[-1, :, :]  # 再还原EOS表示
                        res = EncoderOut(
                            encoder_out=h_refine,  # T x B x C
                            encoder_padding_mask=res.encoder_padding_mask,  # B x T
                            encoder_embedding=None,  # B x T x C
                            encoder_states=res.encoder_states,  # List[T x B x C]
                            src_tokens=None,
                            src_lengths=None,
                        )
                        syntax_encoder_out_all.append(res)
            encoder_out = self.dual_aggregation(sentence_encoder_out, syntax_encoder_out_all,
                                                self.args.dual_aggregation_beta)  # 聚集操作，保留原始sentence encoder的信息
            return encoder_out
        else:
            return sentence_encoder_out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        # encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        # encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        # if encoder_out.encoder_out is None:
        #     new_encoder_out = (encoder_out.encoder_out)
        # elif isinstance(encoder_out.encoder_out, list):
        #     new_encoder_out = ([eo.index_select(1, new_order) for eo in encoder_out.encoder_out])
        # else:
        #     new_encoder_out = (encoder_out.encoder_out.index_select(1, new_order))

        # new_encoder_padding_mask = (
        #     encoder_padding_mask
        #     if encoder_padding_mask is None
        #     else encoder_padding_mask.index_select(0, new_order)
        # )
        # new_encoder_embedding = (
        #     encoder_embedding
        #     if encoder_embedding is None
        #     else encoder_embedding.index_select(0, new_order)
        # )

        # encoder_pos = encoder_out.encoder_pos
        # if encoder_pos is not None:
        #     new_encoder_pos = encoder_pos.index_select(0, new_order)

        # length_out = encoder_out.length_out
        # if length_out is not None:
        #     new_length_out = length_out.index_select(0, new_order)

        # src_tokens = encoder_out.src_tokens
        # if src_tokens is not None:
        #     src_tokens = src_tokens.index_select(0, new_order)

        # src_lengths = encoder_out.src_lengths
        # if src_lengths is not None:
        #     src_lengths = src_lengths.index_select(0, new_order)

        # encoder_states = encoder_out.encoder_states
        # if encoder_states is not None:
        #     for idx, state in enumerate(encoder_states):
        #         encoder_states[idx] = state.index_select(1, new_order)

        # return GlatEncoderOut(
        #     encoder_out=new_encoder_out,  # T x B x C
        #     encoder_padding_mask=new_encoder_padding_mask,  # B x T
        #     encoder_embedding=new_encoder_embedding,  # B x T x C
        #     encoder_pos=new_encoder_pos,
        #     length_out=new_length_out,
        #     encoder_states=encoder_states,  # List[T x B x C]
        #     src_tokens=src_tokens,  # B x T
        #     src_lengths=src_lengths,  # B x 1
        # )
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["length_out"]) == 0:
            new_length_out = []
        else:
            new_length_out = [encoder_out["length_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_pos"]) == 0:
            new_encoder_pos = []
        else:
            new_encoder_pos = [
                encoder_out["encoder_pos"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C  有
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T  有
            "encoder_embedding": new_encoder_embedding,  # B x T x C  有
            "encoder_pos": new_encoder_pos,  # 有
            "length_out": new_length_out,  # 有
            "encoder_states": encoder_states,  # List[T x B x C] # 有
            "src_tokens": src_tokens,  # B x T  有
            "src_lengths": src_lengths,  # B x 1 # 有
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.sentence_encoder.embed_positions is None:
            return self.sentence_encoder.max_source_positions
        return min(self.sentence_encoder.max_source_positions, self.sentence_encoder.embed_positions.max_positions)

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     return self.sentence_encoder.upgrade_state_dict_named(state_dict, name)


class SyntaxSentenceNATransformerEncoder(FairseqEncoder):  # sentence encoder
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding

    编码句子token信息，和传统的Transformer Encoder一致
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        self.src_dropout = args.source_word_dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)  # if full mask then embed_scale must be 1

        # print(self.padding_idx)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return DSATransformerEncoderLayer(args)

    """修改记录
    Yue Zhang
    2021.12.27
    源词丢弃策略，提高模型泛化性
    """

    def SRC_dropout(self, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens * (1 / keep_prob)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        if self.training:
            token_embedding = self.SRC_dropout(token_embedding, self.src_dropout)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        ###########################--↓--############################
        x = self.dropout_module(x)
        ###########################--↑--############################
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    # @torchsnooper.snoop()
    def forward(
            self,
            src_tokens,
            src_lengths,
            src_dpd_matrix=None,  # 描述源端句法树，依存距离矩阵
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        #####################################以下为添加的######################################
        encoder_pos = self.embed_positions(src_tokens)
        #####################################以上为添加的######################################
        """修改记录
        Yue Zhang
        2022.1.29

        为Sentence Encoder添加DSA机制（CIKM21）  （Dual Self-Attention）
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        ###########################--↓--############################
        # if encoder_padding_mask is not None:
        #     x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        # x = self.dropout_module(x)
        ###########################--↑--############################
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # print(x.shape)
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for idx, layer in enumerate(self.layers):  # 6层
            x = layer(x, encoder_padding_mask, layer_num=idx, src_dpd_matrix=src_dpd_matrix)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:  # F
            x = self.layer_norm(x)
        #####################################以下为添加的######################################
        len_x = x[0, :, :]
        #####################################以上为添加的######################################
        return GlatEncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_pos=encoder_pos,
            length_out=len_x,
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )

        encoder_pos = encoder_out.encoder_pos
        if encoder_pos is not None:
            new_encoder_pos = encoder_pos.index_select(0, new_order)

        length_out = encoder_out.length_out
        if length_out is not None:
            new_length_out = length_out.index_select(0, new_order)

        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return GlatEncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            encoder_pos=new_encoder_pos,
            length_out=new_length_out,
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


"""修改记录
Yue Zhang
2021.12.29
句法诱导的Encoder，复现自论文《A Syntax-Guided Grammatical Error Correction Model with Dependency Tree Correction》
"""


class SyntaxSyntaxNATransformerEncoder(FairseqEncoder):  # Syntax tree encoder
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding

    额外编码句法信息
    """

    def __init__(self, args, dictionary, embed_tokens, syntax_type="dep"):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.syntax_type = syntax_type
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        self.encoder_type = args.syntax_encoder
        self.src_dropout = args.source_word_dropout if getattr(args, "source_word_dropout_probs", False) else 0

        embed_dim = embed_tokens.embedding_dim  # 这里主要是对Relation做Embedding
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(
            embed_dim)  # if full mask then embed_scale must be 1

        self.embed_positions = None  # 关系嵌入不考虑Position

        self.layernorm_embedding = None  # 关系嵌入暂时不考虑对嵌入结果做LayerNorm

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.gat_encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        if args.syntax_encoder == "GAT":
            return GATSyntaxGuidedTransformerEncoderLayer(args)
        elif args.syntax_encoder == "GCN":
            return GCNSyntaxGuidedTransformerEncoderLayer(args, self.syntax_type)
        else:
            raise NotImplementedError

    """修改记录
    Yue Zhang
    2021.12.27
    源词丢弃策略，提高模型泛化性
    """

    def SRC_dropout(self, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens

    def forward_label_embedding(  # 这里要仔细看维度
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        """
        Yue Zhang
        2021.12.29
        获取句法label的嵌入表示
        """
        # arc_mask: bsz * seq_len * seq_len
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:  # 看看有没有这个
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        # x: bsz * seq_len * seq_len * embed_dim
        return x, embed

    # @torchsnooper.snoop()
    def forward(
            self,
            src_tokens,
            src_lengths,
            h,  # Sentence Encoder 的输出, T x B x C
            src_outcoming_arc_mask=None,  # 描述源端句法树，出弧（指向孩子节点）掩码矩阵
            src_incoming_arc_mask=None,  # 描述源端句法树，入弧（指向孩子节点）掩码矩阵
            src_probs_matrix=None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # B X T X T X C
        src_outcoming_arc_x, src_outcoming_arc_embed = self.forward_label_embedding(
            src_outcoming_arc_mask)  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        src_incoming_arc_x, src_incoming_arc_embed = self.forward_label_embedding(
            src_incoming_arc_mask)  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        del src_outcoming_arc_embed
        del src_incoming_arc_embed

        # # T x B x C -> B x T x C (Todo: 继续将T放在第一维度)
        # compute padding mask
        src_arc_padding_mask = (src_outcoming_arc_mask + src_incoming_arc_mask).eq(
            self.padding_idx)  # B x T x T，由于一个节点不可能同时是另一个节点的父亲/孩子，因此可以合并两个mask矩阵，pad的含义是：既非父亲也非孩子的那些节点
        encoder_states = [] if return_all_hiddens else None
        if self.training:
            src_probs_matrix = self.SRC_dropout(src_probs_matrix, self.src_dropout)
        # encoder layers
        for layer in self.layers:  # 3层
            if self.encoder_type == "GAT":
                if self.syntax_type == "con":  # GAT暂时不支持编码成分句法
                    raise NotImplementedError
                encoder_padding_mask = src_tokens.eq(1)  # B x T
                h = layer(h, src_outcoming_arc_x, src_incoming_arc_x, encoder_padding_mask,
                          attn_mask=src_arc_padding_mask)
            else:  # "GCN"
                encoder_padding_mask = src_tokens.ne(1)  # B x T
                h = layer(h, src_outcoming_arc_x, src_incoming_arc_x, encoder_padding_mask,
                          src_probs_matrix=src_probs_matrix)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(h)

        if self.layer_norm is not None:
            h = self.layer_norm(h)  # Layer norm是否对Batch First生效？
        return EncoderOut(
            encoder_out=h,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


#################################以上为编码器####################################








@register_model_architecture(
    "nat_ctc_sd_ss_syntax", "nat_ctc_sd_ss_syntax"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)  # NOTE
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

