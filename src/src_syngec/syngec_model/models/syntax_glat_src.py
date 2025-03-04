# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch
# from fairseq.models.nat.nonautoregressive_transformer import NATransformerEncoder, NATransformerDecoder, NATransformerModel
from .syntax_glat_nonautoregressive import SyntaxGlatNATransformerModel, SyntaxGlatNATransformerEncoder, SyntaxGlatNATransformerDecoder
import logging
import random
from contextlib import contextmanager
from .syntax_glat_decoder_input import SyntaxDecNATransformerDecoder

logger = logging.getLogger(__name__)

'''
所有修改
1. 
    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
            src_outcoming_arc_mask, 
            src_incoming_arc_mask,
            src_dpd_matrix, 
            src_probs_matrix,
            glat=None, **kwargs
2. 
@register_model("syntax_glat")
class SyntaxGlat(FairseqNATModel):
3.
@register_model_architecture(
    "syntax_glat", "syntax_glat")
4.
from .syntax_glat_nonautoregressive import SyntaxGlatNATransformerModel, SyntaxGlatNATransformerEncoder, SyntaxGlatNATransformerDecoder
5.
    forward_decoder = SyntaxGlatNATransformerModel.forward_decoder
    initialize_output_tokens = SyntaxGlatNATransformerModel.initialize_output_tokens
    regenerate_length_beam = SyntaxGlatNATransformerModel.regenerate_length_beam
6.
    def build_encoder(cls, args, tgt_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        encoder = SyntaxGlatNATransformerEncoder(args, tgt_dict, embed_tokens, syntax_label_dict, embed_rels)
7.
decoder = SyntaxGlatNATransformerDecoder(args, tgt_dict, embed_tokens)
8.
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, 
                    src_outcoming_arc_mask=src_outcoming_arc_mask, 
                    src_incoming_arc_mask=src_incoming_arc_mask,
                    src_dpd_matrix=src_dpd_matrix, 
                    src_probs_matrix=src_probs_matrix,
                    **kwargs)  # 其他的参数通过**kwargs这里添加进去的
9. 
    @staticmethod
    def add_args(parser):
        SyntaxGlatNATransformerModel.add_args(parser)
        和参数注释
'''


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


@register_model("syntax_glat_src")
class SyntaxSrcGlat(SyntaxGlatNATransformerModel):
    # forward_decoder = SyntaxGlatNATransformerModel.forward_decoder
    initialize_output_tokens = SyntaxGlatNATransformerModel.initialize_output_tokens
    regenerate_length_beam = SyntaxGlatNATransformerModel.regenerate_length_beam

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        SyntaxGlatNATransformerModel.add_args(parser)

        # length prediction
        # parser.add_argument(
        #     "--src-embedding-copy",
        #     action="store_true",
        #     help="copy encoder word embeddings as the initial input of the decoder",
        # )
        # parser.add_argument(
        #     "--pred-length-offset",
        #     action="store_true",
        #     help="predicting the length difference between the target and source sentences",
        # )
        # parser.add_argument(
        #     "--sg-length-pred",
        #     action="store_true",
        #     help="stop the gradients back-propagated from the length predictor",
        # )
        # parser.add_argument(
        #     "--length-loss-factor",
        #     type=float,
        #     help="weights on the length prediction loss",
        # )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        encoder = SyntaxGlatNATransformerEncoder(args, tgt_dict, embed_tokens, syntax_label_dict, embed_rels)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    # @classmethod
    # def build_decoder(cls, args, tgt_dict, embed_tokens):
    #     decoder = SyntaxGlatNATransformerDecoder(args, tgt_dict, embed_tokens)
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        decoder = SyntaxDecNATransformerDecoder(args, tgt_dict, embed_tokens, syntax_label_dict, embed_rels)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, 
            src_outcoming_arc_mask, 
            src_incoming_arc_mask,
            src_dpd_matrix, 
            src_probs_matrix,
            glat=None, **kwargs
    ):  # 其他的参数通过**kwargs这里添加进去的
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, 
                    src_outcoming_arc_mask=src_outcoming_arc_mask, 
                    src_incoming_arc_mask=src_incoming_arc_mask,
                    src_dpd_matrix=src_dpd_matrix, 
                    src_probs_matrix=src_probs_matrix,
                    **kwargs)  # 其他的参数通过**kwargs这里添加进去的

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )
        nonpad_positions = tgt_tokens.ne(self.pad)
        seq_lens = (nonpad_positions).sum(1)
        rand_seed = random.randint(0, 19260817)
        # rand_seed = 9143233
        # glancing sampling
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                        src_outcoming_arc_mask=src_outcoming_arc_mask,
                        src_incoming_arc_mask=src_incoming_arc_mask,
                        src_dpd_matrix=src_dpd_matrix,
                        src_probs_matrix=src_probs_matrix,    
                    )
                pred_tokens = word_ins_out.argmax(-1)
                same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
                input_mask = torch.ones_like(nonpad_positions)
                bsz, seq_len = tgt_tokens.size()
                for li in range(bsz):
                    target_num = (((seq_lens[li] - same_num[li].sum()).float()) * glat['context_p']).long()
                    if target_num > 0:
                        input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
                input_mask = input_mask.eq(1)
                input_mask = input_mask.masked_fill(~nonpad_positions,False)
                glat_prev_output_tokens = prev_output_tokens.masked_fill(~input_mask, 0) + tgt_tokens.masked_fill(input_mask, 0)
                glat_tgt_tokens = tgt_tokens.masked_fill(~input_mask, self.pad)

                prev_output_tokens, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens

                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                    "glat_context_p": glat['context_p'],
                }

        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                src_outcoming_arc_mask=src_outcoming_arc_mask,
                src_incoming_arc_mask=src_incoming_arc_mask,
                src_dpd_matrix=src_dpd_matrix,
                src_probs_matrix=src_probs_matrix,
            )

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor*(tgt_tokens.ne(self.pad).sum().item())/(seq_lens.sum().item()),
            }
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def forward_decoder(self, decoder_out, encoder_out, 
                        src_outcoming_arc_mask=None,
                        src_incoming_arc_mask=None,
                        src_dpd_matrix=None,
                        src_probs_matrix=None,
                        decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            src_outcoming_arc_mask=src_outcoming_arc_mask,
            src_incoming_arc_mask=src_incoming_arc_mask,
            src_dpd_matrix=src_dpd_matrix,
            src_probs_matrix=src_probs_matrix,
        ).max(-1)

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
    
    # @classmethod
    # def build_model(cls, args, task):
    #     """Build a new model instance."""

    #     # make sure all arguments are present in older models
    #     syntax_enhanced_transformer(args)  # 只是用来初始化参数的

    #     if args.encoder_layers_to_keep:  # 可以只使用一些特定的layer # N
    #         args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
    #     if args.decoder_layers_to_keep:  # N
    #         args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

    #     if getattr(args, "max_source_positions", None) is None:  # 控制输入输出序列的最大长度  # 64
    #         args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
    #     if getattr(args, "max_target_positions", None) is None:  # 64
    #         args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

    #     src_dict, tgt_dict = task.source_dictionary, task.target_dictionary  # 源字典，目标字典
        
    #     syntax_label_dict = getattr(task, "syntax_label_dict", None)  # 句法标签词典

    #     if args.share_all_embeddings:  # True Transformer的Encoder和Decoder可以共享词嵌入矩阵，减少参数量，效果也不差（前提：共享词表）
    #         if src_dict != tgt_dict:
    #             raise ValueError("--share-all-embeddings requires a joined dictionary")
    #         if args.encoder_embed_dim != args.decoder_embed_dim:  # 512
    #             raise ValueError(
    #                 "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
    #             )
    #         if args.decoder_embed_path and (
    #             args.decoder_embed_path != args.encoder_embed_path
    #         ):
    #             raise ValueError(
    #                 "--share-all-embeddings not compatible with --decoder-embed-path"
    #             )
    #         encoder_embed_tokens = cls.build_embedding(  # 编码器和解码器共享词典的嵌入矩阵
    #             args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
    #         )
    #         decoder_embed_tokens = encoder_embed_tokens
    #         args.share_decoder_input_output_embed = True
    #     else:
    #         encoder_embed_tokens = cls.build_embedding(  # 编码器和解码器不共享词典的嵌入矩阵
    #             args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
    #         )
    #         decoder_embed_tokens = cls.build_embedding(
    #             args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
    #         )
    #     syntax_label_embed_tokens = None
    #     if args.use_syntax:
    #         if len(syntax_label_dict) > 1:
    #             syntax_label_embed_tokens = []
    #             for d in syntax_label_dict:
    #                 syntax_label_embed_tokens.append(cls.build_embedding(  # 句法的邻接边类型的词典的嵌入
    #                     args, d, args.encoder_embed_dim
    #                 ))
    #         else:
    #             syntax_label_embed_tokens = cls.build_embedding(  # 句法的邻接边类型的词典的嵌入
    #                     args, syntax_label_dict[0], args.encoder_embed_dim
    #                 )

    #     encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, syntax_label_dict=syntax_label_dict, embed_rels=syntax_label_embed_tokens)  # , syntax_label_dict=syntax_label_dict, embed_rels=syntax_label_embed_tokens
    #     decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, syntax_label_dict=syntax_label_dict, embed_rels=syntax_label_embed_tokens) #, syntax_label_dict=syntax_label_dict, embed_rels=syntax_label_embed_tokens)
    #     if getattr(args, "freeze_bart_parameters", False):  # 如果冻结bart的参数
    #         for name, param in encoder.named_parameters():
    #             if "sentence_encoder" in name:
    #                 param.requires_grad = False
    #         for name, param in decoder.named_parameters():
    #             param.requires_grad = False
    #     return cls(args, encoder, decoder)


@register_model_architecture(
    "syntax_glat_src", "syntax_glat_src_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
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


@register_model_architecture(
    "syntax_glat_src", "syntax_glat_src"
)
def glat_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_embed_dim*4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", args.encoder_embed_dim//64)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim*4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.decoder_embed_dim//64)
    base_architecture(args)

@register_model_architecture(
    "syntax_glat_src", "syntax_glat_src_base"
)
def base_architecture2(args):
    base_architecture(args)

@register_model_architecture(
    "syntax_glat_src", "syntax_glat_src_wmt_en_de"
)
def syntax_glat_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture(
    "syntax_glat_src", "syntax_glat_src_iwslt")
def syntax_glat_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_architecture(args)
