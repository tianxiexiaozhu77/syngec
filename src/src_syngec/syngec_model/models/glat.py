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
from fairseq.models.nat.glat_nonautoregressive_transformer import NATransformerDecoder, NATransformerModel, NATransformerEncoder
# from fairseq.models.nat.glat_nonautoregressive import NATransformerDecoder, NATransformerModel, NATransformerEncoder
# from fairseq.models.nat.glat_nonauto import NATransformerDecoder, NATransformerModel, NATransformerEncoder

# from fairseq.models.nat.fairseq_nat_model import FairseqNATEncoder
import logging
import random
from contextlib import contextmanager

logger = logging.getLogger(__name__)

'''跟作者的源码相比
1. from fairseq.models.nat.nonautoregressive_transformer import NATransformerEncoder, NATransformerDecoder, NATransformerModel
改为
from fairseq.models.nat.glat_nonautoregressive_transformer import NATransformerDecoder, NATransformerModel, NATransformerEncoder
2. 加 from fairseq.models.nat.fairseq_nat_model import FairseqNATEncoder
3. 
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        改为
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = FairseqNATEncoder(args, tgt_dict, embed_tokens)
4. 以上1-3都是用了原来的。只是把glat中的nonantiregressive_transformer.py放到本项目中的fairseq models nat下了，
并重新命名为glat_nonantiregressive_transformer.py。另外修改了nat下的__init__.py文件。
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


@register_model("glat")
class Glat(FairseqNATModel):
    forward_decoder = NATransformerModel.forward_decoder  # 类属性，它们在类定义时就被赋值。生成的时候用
    initialize_output_tokens = NATransformerModel.initialize_output_tokens
    regenerate_length_beam = NATransformerModel.regenerate_length_beam

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

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

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        # encoder = FairseqNATEncoder(args, tgt_dict, embed_tokens)  # NATransformerEncoder
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)  # NATransformerEncoder
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)  # 这里面有返回的一些额外添加了一个列表

        # length prediction
        length_out = self.decoder.forward_length(  # 注意列表元素的使用方式
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(  # 注意列表元素的使用方式
            length_out, encoder_out, tgt_tokens
        )
        nonpad_positions = tgt_tokens.ne(self.pad)  # tgt张量维度一致，不是pad的位置都为True，是pad的位置为False
        seq_lens = (nonpad_positions).sum(1)  # 一个batch中每个样本的tgt的长度
        rand_seed = random.randint(0, 19260817)
        # glancing sampling
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(  # torch.Size([104, 74, 10152])    # 注意列表元素的使用方式
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                    )
                pred_tokens = word_ins_out.argmax(-1)  # torch.Size([104, 74])
                same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)  # tgt_tokens torch.Size([104, 74])  tgt中非pad中预测的词等于目标词的个数
                input_mask = torch.ones_like(nonpad_positions)
                bsz, seq_len = tgt_tokens.size()
                for li in range(bsz):  # 对batchsize中的每个样本
                    ###############################################
                    # context_p = 0.5
                    # glat_acc = (same_num.sum() / seq_lens.sum()).item()
                    # if glat_acc > 0.8:
                    #     context_p = context_p * 0.5
                    # elif glat_acc > 0.6:
                    #     context_p = context_p * 0.7
                    # elif glat_acc > 0.4:
                    #     context_p = context_p * 0.9
                    # glat['context_p'] = context_p
                    # target_num = (((seq_lens[li] - same_num[li].sum()).float()) * glat['context_p']).long()  # 一个样本中tgt的所有的词与tgt预测正确的词的差 ×0.5 为 在第二次decoder中输入的tgt中ground truth的token的数量
                    ###############################################
                    target_num = (((seq_lens[li] - same_num[li].sum()).float()) * glat['context_p']).long()  # 一个样本中tgt的所有的词与tgt预测正确的词的差 ×0.5 为 在第二次decoder中输入的tgt中ground truth的token的数量
                    if target_num > 0:
                        input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
                input_mask = input_mask.eq(1)
                input_mask = input_mask.masked_fill(~nonpad_positions,False)  # pad的地方用false填充
                glat_prev_output_tokens = prev_output_tokens.masked_fill(~input_mask, 0) + tgt_tokens.masked_fill(input_mask, 0)
                glat_tgt_tokens = tgt_tokens.masked_fill(~input_mask, self.pad)

                prev_output_tokens, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens

                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),  # 预测正确的词的个数/句子测长度
                    "glat_context_p": glat['context_p'],
                }

        with torch_seed(rand_seed):
            word_ins_out = self.decoder(  # 注意列表元素的使用方式
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
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


@register_model_architecture(
    "glat", "glat_6e6d512"
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
    "glat", "glat"
)
def glat_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_architecture(args)
# @register_model_architecture(
#     "syntax_nonautoregressive_transformer", "syntax_nonautoregressive_transformer_iwslt")
# def syntax_nonautoregressive_transformer_iwslt(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     base_architecture(args)

@register_model_architecture(
    "glat", "glat_wmt"
)
def base_architecture2(args):
    base_architecture(args)
