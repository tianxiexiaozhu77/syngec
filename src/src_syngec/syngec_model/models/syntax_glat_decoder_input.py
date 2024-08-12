# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel, FairseqNATDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.fairseq_dropout import FairseqDropout
import torch
from fairseq.models.nat.glat_nonautoregressive_transformer import NATransformerEncoder, NATransformerDecoder, NATransformerModel
from .syntax_glat_nonautoregressive import SyntaxGlatNATransformerModel, SyntaxGlatNATransformerEncoder, SyntaxGlatNATransformerDecoder
import logging
import random
from contextlib import contextmanager
from .syntax_enhanced_fairseq_nat_model import ensemble_decoder
from fairseq.models.transformer import Embedding

logger = logging.getLogger(__name__)

'''
所有修改

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
class SyntaxDecNATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, syntax_label_dict=None, embed_rels=None, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        if self.length_loss_factor > 0:
            self.embed_length = Embedding(getattr(args, "max_target_positions", 256), self.encoder_embed_dim, None)
            torch.nn.init.normal_(self.embed_length.weight, mean=0, std=0.02)
        if self.src_embedding_copy:
            self.copy_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        syn_embed_dim = embed_rels.embedding_dim  # 这里主要是对Relation做Embedding
        self.syn_padding_idx = embed_rels.padding_idx
        self.max_source_positions = args.max_source_positions

        self.syn_embed_tokens = embed_rels
        self.arc_attn_linear_src = nn.Linear(syn_embed_dim, syn_embed_dim)
        self.arc_attn_linear_pos = nn.Linear(syn_embed_dim, syn_embed_dim)
        self.arc_attn_v = nn.Linear(syn_embed_dim, 1)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )  # 给incoming arc的激活函数，对应论文公式(1)
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.W_in = self.build_fc(2 * self.encoder_embed_dim, self.encoder_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.W_out = self.build_fc(2 * self.embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)

    def build_fc(self, input_dim, output_dim, q_noise=None, qn_block_size=None):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
        

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
        if token_embedding is None:  # T
            token_embedding = self.syn_embed_tokens(src_tokens)  # self.embed_tokens: Embedding(51, 512, padding_idx=0)
        x = embed = self.embed_scale * token_embedding 
        if self.embed_positions is not None:  # F
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:  # F
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # if self.quant_noise is not None:  # F
        #     x = self.quant_noise(x)
        # x: bsz * seq_len * seq_len * embed_dim
        return x, embed

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0,
                src_outcoming_arc_mask=None,
                src_incoming_arc_mask=None,
                src_dpd_matrix=None,
                src_probs_matrix=None, 
                **unused):
        src_outcoming_arc_x, src_outcoming_arc_embed = self.forward_label_embedding(src_outcoming_arc_mask[0])  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        src_incoming_arc_x, src_incoming_arc_embed = self.forward_label_embedding(src_incoming_arc_mask[0])  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        src_arc_padding_mask = (src_outcoming_arc_mask[0] + src_incoming_arc_mask[0]).eq(self.syn_padding_idx)

        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            src_outcoming_arc_x=src_outcoming_arc_x,
            src_incoming_arc_x=src_incoming_arc_x,
            src_arc_padding_mask=src_arc_padding_mask,

        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:  # 停止从长度预测器反向传播的梯度
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            src_outcoming_arc_x=None,
            src_incoming_arc_x=None,
            src_arc_padding_mask=None,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        positions = (  # (B T C:40 45 512)
            self.embed_positions(prev_output_tokens)  # prev_output_tokens：（B T:40 45）
            if self.embed_positions is not None
            else None
        )
        # # --原来的↓--
        # # embedding
        # if embedding_copy:  # T
        #     #########################--↓--#########################
        #     src_embd = encoder_out.encoder_embedding  # :(B T C :40 50 512
        #     #########################--↑--#########################
        #     src_mask = encoder_out.encoder_padding_mask  # (B T:40 50)
        #     bsz, seq_len = prev_output_tokens.size()  # (B T:40 45)
        #     #########################--↓--#########################
        #     attn_score = torch.bmm(self.copy_attn(positions),# positions(B T C)->self.copy_attn(positions) (40 45 512->40 45 512)
        #                            (src_embd + encoder_out.encoder_pos).transpose(1,2))  # src_embd + encoder_out.encoder_pos:(B T C:40 50 512)  (src_embd + encoder_out.encoder_pos).transpose(1, 2)(B C T:40 512 50)
        #     #########################--↑--#########################
        #     if src_mask is not None:  # (B T:40 50)
        #         attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1),
        #                                             float('-inf'))  # # attn_score (B T_target T_source:40, 45, 50)
        #     attn_weight = F.softmax(attn_score, dim=-1)  # attn_weight (B T_target T_source:40, 45, 50)
        #     #########################--↓--#########################
        #     x = torch.bmm(attn_weight, src_embd)  #
        #     #########################--↑--#########################
        #     mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)  #
        #     output_mask = prev_output_tokens.eq(self.unk)  # self.unk：3
        #     cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2))  #
        #     x = cat_x.index_select(dim=0, index=torch.arange(bsz * seq_len).cuda() * 2 +
        #                                         output_mask.view(-1).long()).reshape(bsz, seq_len, x.size(2))  #
        #     #--原来的↑--


        # if embedding_copy:
        #     src_embd = encoder_out.encoder_embedding  # :(B T C :40 50 512)
        #     src_pos = encoder_out.encoder_pos  # Position embeddings
        #     src_mask = encoder_out.encoder_padding_mask  # (B T:40 50)
        #     bsz, seq_len = prev_output_tokens.size()  # (B T:40 45)
            
        #     # Original attention
        #     positions_proj = self.copy_attn(positions)  # Applying attention
        #     attn_score = torch.bmm(positions_proj,  # positions(B T C)->self.copy_attn(positions) (40 45 512->40 45 512)
        #                            (src_embd + src_pos).transpose(1, 2))
            
        #     if src_mask is not None:  # (B T:40 50)
        #         attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1), float('-inf'))  # attn_score (B T_target T_source:40, 45, 50)
        #     attn_weight = F.softmax(attn_score, dim=-1)  # attn_weight (B T_target T_source:40, 45, 50)
        #     x = torch.bmm(attn_weight, src_embd)

        #     # Integrate syntactic information
        #     # src_incoming_arc_x = encoder_out.src_incoming_arc_x  # (B T_src T_src C)
        #     # src_arc_padding_mask = encoder_out.src_arc_padding_mask  # (B T_src T_src)
            
        #     # Expand positions_proj to match src_incoming_arc_x
        #     positions_proj_1 = self.arc_attn_linear_src(positions)
        #     # positions_proj_expanded = positions_proj_1.unsqueeze(1).expand(-1, src_incoming_arc_x.size(1), -1, -1)  # (B T_tgt 1 C) -> (B T_tgt T_src C)
            
        #     # Compute additive attention scores for syntactic information
        #     # arc_attn_src = self.arc_attn_linear_src(src_incoming_arc_x)  # (B T_src T_src C)
        #     # arc_attn_pos = self.arc_attn_linear_pos(positions_proj_expanded)  # (B T_tgt T_src C)
        #     # arc_attn_score = torch.matmul(positions_proj_expanded,  # positions(B T C)->self.copy_attn(positions) (40 45 512->40 45 512)
        #     #                        src_incoming_arc_x.transpose(-1, -2)).sum(1)  # (B T_tgt T_src C)
        #     # arc_attn_score = self.arc_attn_v(arc_attn_score).squeeze(-1)  # (B T_tgt T_src)
        #     arc_attn_score = torch.bmm(positions_proj_1,  # positions(B T C)->self.copy_attn(positions) (40 45 512->40 45 512)
        #                            src_incoming_arc_x[:,:,0].transpose(1, 2))

        #     if src_arc_padding_mask is not None:  # (B T_src T_src)
        #         src_arc_padding_mask_expanded = src_arc_padding_mask[:,:,0].unsqueeze(1).expand(-1, arc_attn_score.size(1), -1)
        #         # src_arc_padding_mask_expanded = src_arc_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)  # (B 1 T_src T_src) -> (B T_tgt T_src)
        #         arc_attn_score = arc_attn_score.masked_fill(src_arc_padding_mask_expanded, float('-inf'))  # (B T_tgt T_src)
        #     arc_attn_weight = F.softmax(arc_attn_score, dim=-1)  # (B T_tgt T_src)
            
        #     arc_x = torch.bmm(arc_attn_weight, src_incoming_arc_x[:,:,0])  # (B T_tgt C)

        #     # Combine the original attention result and syntactic attention result
        #     # combined_x = x + arc_x  # (B T C)
        #     combined_x = arc_x

        #     mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)  #
        #     output_mask = prev_output_tokens.eq(self.unk)  # self.unk：3
        #     cat_x = torch.cat([mask_target_x.unsqueeze(2), combined_x.unsqueeze(2)], dim=2).view(-1, combined_x.size(2))  #
        #     x = cat_x.index_select(dim=0, index=torch.arange(bsz * seq_len).cuda() * 2 +
        #                            output_mask.view(-1).long()).reshape(bsz, seq_len, combined_x.size(2))  #
        if embedding_copy:
            src_embd = encoder_out.encoder_embedding  # :(B T C :40 50 512) 源句子长度50
            src_pos = encoder_out.encoder_pos  # Position embeddings
            src_mask = encoder_out.encoder_padding_mask  # (B T:40 50)
            bsz, seq_len = prev_output_tokens.size()  # (B T:40 45) 目标句子长度45
            source_seq_len = src_embd.shape[1]
            h = src_embd.unsqueeze(1)
            h = h.repeat(1, source_seq_len, 1, 1)
            #################有src_incoming_arc_x--⬇---####################
            U_in = torch.cat((h, src_incoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            U_in = self.W_in(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            U_in = self.activation_fn(U_in)   # torch.Size([48, 35, 35, 512])
            U_in = self.activation_dropout_module(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            h = U_in.sum(dim=2)  # .transpose(0, 1)
            #################有src_incoming_arc_x--⬆---####################

            # #################有src_outcoming_arc--⬇---####################
            # # 对应论文式(1)
            # U_out = torch.cat((h, src_outcoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_out = self.W_out(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # U_out = self.activation_fn(U_out)
            # U_out = self.activation_dropout_module(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # h = U_out.sum(dim=2) # .transpose(0, 1)  # T x B x D  # torch.Size([35, 48, 512])
            # #################src_outcoming_arc--⬆---####################

            # #################只有src_incoming_arc_x和src_probs_matrix--⬇---####################
            # U_in = torch.cat((h, src_incoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_in = self.W_in(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_in = self.W_in(h)  # B x T x T x D
            # U_in = self.activation_fn(U_in)   # torch.Size([48, 35, 35, 512])
            # U_in = self.activation_dropout_module(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # src_probs_matrix = src_probs_matrix.unsqueeze(-1)  # torch.Size([48, 35, 35])->torch.Size([48, 35, 35, 1])

            # U_in = torch.mul(src_probs_matrix, U_in)  # torch.Size([48, 35, 35, 512])
            # # U_out = torch.mul(src_probs_matrix.transpose(1,2), U_out)  # torch.Size([48, 35, 35, 1])->torch.Size([48, 35, 35, 1])  src_probs_matrix->src_probs_matrix.transpose(1,2)
            # # h = (U_in + U_out).sum(dim=2).transpose(0, 1)  # T x B x D
            # h = U_in.sum(dim=2) # .transpose(0, 1)  # T x B x D  torch.Size([35, 48, 512])
            # #################只有src_incoming_arc_x和src_probs_matrix--⬆---####################
            
            
            # #################只有src_outcoming_arc_x和src_probs_matrix--⬇---####################
            # # 对应论文式(1)
            # U_out = torch.cat((h, src_outcoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_out = self.W_out(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_out = self.W_out(h)  # B x T x T x D
            # U_out = self.activation_fn(U_out)
            # U_out = self.activation_dropout_module(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])

            # src_probs_matrix = src_probs_matrix.unsqueeze(-1)  # torch.Size([48, 35, 35])->torch.Size([48, 35, 35, 1])
            # U_out = torch.mul(src_probs_matrix.transpose(1,2), U_out)  # torch.Size([48, 35, 35, 1])->torch.Size([48, 35, 35, 1])  src_probs_matrix->src_probs_matrix.transpose(1,2)
            # h = U_out.sum(dim=2) # .transpose(0, 1)  # T x B x D  torch.Size([35, 48, 512])
            # #################只有src_outcoming_arc_x和src_probs_matrix--⬆---####################
            
            
            # #################只有src_probs_matrix--⬇---####################
            # src_probs_matrix = src_probs_matrix.unsqueeze(-1)
            # h = torch.mul(src_probs_matrix, h)
            # h = h.sum(dim=2) # .transpose(0, 1)  # torch.Size([35, 48, 512])
            # #################只有src_probs_matrix--⬆---####################

            # #################有src_incoming_arc_x和src_outcoming_arc--⬇---####################
            # # 对应论文式(1)
            # U_out = torch.cat((h, src_outcoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_out = self.W_out(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_out = self.W_out(h)  # B x T x T x D
            # U_out = self.activation_fn(U_out)
            # U_out = self.activation_dropout_module(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])

            # # 对应论文式(2)
            # U_in = torch.cat((h, src_incoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_in = self.W_in(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_in = self.W_in(h)  # B x T x T x D
            # U_in = self.activation_fn(U_in)   # torch.Size([48, 35, 35, 512])
            # U_in = self.activation_dropout_module(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # h = (U_in + U_out).sum(dim=2) # .transpose(0, 1)  # T x B x D  # torch.Size([35, 48, 512])
            # #################有src_incoming_arc_x和src_outcoming_arc--⬆---####################

            #################有src_incoming_arc_x和src_outcoming_arc和src_probs_matrix--⬇---####################
            # # 对应论文式(1)
            # U_out = torch.cat((h, src_outcoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_out = self.W_out(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_out = self.W_out(h)  # B x T x T x D
            # U_out = self.activation_fn(U_out)
            # U_out = self.activation_dropout_module(U_out)  # B x T x T x D   torch.Size([48, 35, 35, 512])

            # # 对应论文式(2)
            # U_in = torch.cat((h, src_incoming_arc_x), -1)  # B x T x T x 2D  torch.Size([48, 35, 35, 1024])
            # U_in = self.W_in(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # # U_in = self.W_in(h)  # B x T x T x D
            # U_in = self.activation_fn(U_in)   # torch.Size([48, 35, 35, 512])
            # U_in = self.activation_dropout_module(U_in)  # B x T x T x D   torch.Size([48, 35, 35, 512])
            # src_probs_matrix = src_probs_matrix.unsqueeze(-1)  # torch.Size([48, 35, 35])->torch.Size([48, 35, 35, 1])

            # U_in = torch.mul(src_probs_matrix, U_in)  # torch.Size([48, 35, 35, 512])
            # U_out = torch.mul(src_probs_matrix.transpose(1,2), U_out)  # torch.Size([48, 35, 35, 1])->torch.Size([48, 35, 35, 1]):src_probs_matrix->src_probs_matrix.transpose(1,2)
            # h = (U_in + U_out).sum(dim=2) # .transpose(0, 1)  # T x B x D  # torch.Size([35, 48, 512])
            #################有src_incoming_arc_x和src_outcoming_arc和src_probs_matrix--⬆---####################


            positions_proj = self.copy_attn(positions)  # Applying attention
            attn_score = torch.bmm(positions_proj,  # positions(B T C)->self.copy_attn(positions) (40 45 512->40 45 512)
                                   (h + src_pos).transpose(1, 2))
            if src_mask is not None:  # (B T:40 50)
                attn_score = attn_score.masked_fill(src_mask.unsqueeze(1).expand(-1, seq_len, -1), float('-inf'))  # attn_score (B T_target T_source:40, 45, 50)
            attn_weight = F.softmax(attn_score, dim=-1)  # attn_weight (B T_target T_source:40, 45, 50)
            x = torch.bmm(attn_weight, h)
            mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)  #
            output_mask = prev_output_tokens.eq(self.unk)  # self.unk：3
            cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2))  #
            x = cat_x.index_select(dim=0, index=torch.arange(bsz * seq_len).cuda() * 2 +
                                   output_mask.view(-1).long()).reshape(bsz, seq_len, x.size(2))  #

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)  #

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        positions = positions.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            if positions is not None and (i == 0 or embedding_copy):
                x += positions
                x = self.dropout_module(x)

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,  # 下三角矩阵
                self_attn_padding_mask=decoder_padding_mask,  # pad矩阵
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}  # 这里返回是一样de

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (  # positions(B T_target C:[40, 45, 512])
            self.embed_positions(prev_output_tokens)  # prev_output_tokens: (B T_target:40 45)
            if self.embed_positions is not None
            else None
        )
        # embed tokens and positions
        if states is None:  # T
            x = self.embed_tokens(prev_output_tokens)  # self.embed_tokens:(10104 512)
            if self.project_in_dim is not None:  # F
                x = self.project_in_dim(x)
        else:
            x = states  # (B T_target C:40 45 512)

        # if positions is not None:
        #     x += positions
        # x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)  # (B T_target:40 45)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=1023)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt

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


@register_model("syntax_glat_dec")
class SyntaxGlat(SyntaxGlatNATransformerModel):
    forward_decoder = SyntaxGlatNATransformerModel.forward_decoder
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
    def build_encoder(cls, args, tgt_dict, embed_tokens):  # , syntax_label_dict=None, embed_rels=None
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)  # , syntax_label_dict, embed_rels
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

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
                    # src_outcoming_arc_mask=src_outcoming_arc_mask, 
                    # src_incoming_arc_mask=src_incoming_arc_mask,
                    # src_dpd_matrix=src_dpd_matrix, 
                    # src_probs_matrix=src_probs_matrix,
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


@register_model_architecture(
    "syntax_glat_dec", "syntax_glat_dec_6e6d512"
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
    "syntax_glat_dec", "syntax_glat_dec"
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
    "syntax_glat_dec", "syntax_glat_dec_base"
)
def base_architecture2(args):
    base_architecture(args)

@register_model_architecture(
    "syntax_glat_dec", "syntax_glat_dec_wmt_en_de"
)
def syntax_glat_dec_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture(
    "syntax_glat_dec", "syntax_glat_dec_iwslt")
def syntax_glat_dec_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_architecture(args)
