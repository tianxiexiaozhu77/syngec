fairseq plugins loaded...
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 3): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 1): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 7): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 6): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 4): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 5): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 2): tcp://localhost:18630
[W socket.cpp:558] [c10d] The client socket has failed to connect to [localhost]:18630 (errno: 99 - Cannot assign requested address).
2023-06-13 17:29:37 | INFO | fairseq.distributed_utils | distributed init (rank 0): tcp://localhost:18630
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 3
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 1
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 7
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 6
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 4
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 5
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 2
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Added key: store_based_barrier_key:1 to store for rank: 0
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 6
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 0
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 3
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 5
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 1
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 7
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 4
2023-06-13 17:29:38 | INFO | torch.distributed.distributed_c10d | Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
2023-06-13 17:29:38 | INFO | fairseq.distributed_utils | initialized host syntax as rank 2
fairseq plugins loaded...
2023-06-13 17:29:41 | INFO | fairseq_cli.train | Namespace(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9, 0.999)', adam_eps=1e-06, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, apply_bert_init=True, arch='syntax_glat', attention_dropout=0.0, bart_model_file_from_transformers=None, batch_size=None, batch_size_valid=None, best_checkpoint_metric='bleu', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=5.0, conll_suffix=['conll'], cpu=False, criterion='syntax_glat_loss', cross_self_attention=False, curriculum=0, data='/opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method='tcp://localhost:18630', distributed_no_spawn=False, distributed_num_procs=8, distributed_port=-1, distributed_rank=0, distributed_world_size=8, distributed_wrapper='DDP', dpd_suffix=['dpd'], dropout=0.3, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, eval_bleu=True, eval_bleu_args='{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}', eval_bleu_detok='space', eval_bleu_detok_args=None, eval_bleu_print_samples=False, eval_bleu_remove_bpe='@@ ', eval_tokenized_bleu=True, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=7, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, label_smoothing=0.1, left_pad_source='True', left_pad_target='False', length_loss_factor=0.05, load_alignments=False, localsgd_frequency=3, log_format='simple', log_interval=100, lr=[0.0005], lr_scheduler='inverse_sqrt', max_epoch=0, max_source_positions=1024, max_target_positions=1024, max_tokens=4096, max_tokens_valid=4096, max_update=300000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=1e-09, minus_p=0.2, model_parallel_size=1, mse_lambda=10, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=True, no_seed_provided=False, no_token_positional_embeddings=False, noise='full_mask', nprocs_per_node=8, num_batch_buckets=0, num_shards=1, num_workers=1, only_gnn=True, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pred_length_offset=True, probs_suffix=['probs'], profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='/opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59', save_interval=1, save_interval_updates=0, scoring='bleu', seed=0, sentence_avg=False, sg_length_pred=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang='en', source_lang_with_nt='src_nt', source_word_dropout=0.2, src_embedding_copy=True, start_p=0.5, stop_time_hours=0, swm_suffix='swm', syntax_encoder='GCN', syntax_model_file=None, syntax_type=['dep'], target_lang='ro', task='syntax-glat-task', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_up=300000, tpu=False, train_subset='train', truncate_source=False, update_freq=[2], upsample_primary=1, use_bmuf=False, use_dpd=False, use_old_adam=False, use_syntax=True, user_dir='/opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_init_lr=1e-07, warmup_updates=4000, weight_decay=0.01, zero_sharding='none')
2023-06-13 17:29:41 | INFO | syngec_model.tasks.syntax_glat_nat_task | [en] dictionary: 34983 types
2023-06-13 17:29:41 | INFO | syngec_model.tasks.syntax_glat_nat_task | [ro] dictionary: 34983 types
2023-06-13 17:29:41 | INFO | syngec_model.tasks.syntax_glat_nat_task | [syntax label0] dictionary: 51 types
2023-06-13 17:29:41 | INFO | fairseq.data.data_utils | loaded 1999 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/valid.en-ro.en
2023-06-13 17:29:41 | INFO | fairseq.data.data_utils | loaded 1999 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/valid.en-ro.ro
2023-06-13 17:29:41 | INFO | syngec_model.tasks.syntax_glat_nat_task | /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin valid en-ro 1999 examples
2023-06-13 17:29:41 | INFO | fairseq.data.data_utils | loaded 1999 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/valid.conll.en-ro.en
2023-06-13 17:29:41 | INFO | fairseq.data.data_utils | loaded 1999 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/valid.dpd.en-ro.en
2023-06-13 17:29:41 | INFO | fairseq.data.data_utils | loaded 1999 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/valid.probs.en-ro.en
2023-06-13 17:29:41 | INFO | fairseq.data.language_pair_dataset | success! syntax types: 1, source conll lines: 1999
2023-06-13 17:29:42 | INFO | fairseq_cli.train | SyntaxGlat(
  (encoder): SyntaxGlatNATransformerEncoder(
    (sentence_encoder): SyntaxGlatSentenceNATransformerEncoder(
      (dropout_module): FairseqDropout()
      (embed_tokens): Embedding(34983, 512, padding_idx=1)
      (embed_positions): LearnedPositionalEmbedding(1026, 512, padding_idx=1)
      (layers): ModuleList(
        (0): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (1): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (2): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (3): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (4): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (5): DSATransformerEncoderLayer(
          (self_attn): DSAMultiheadAttention(
            (dropout_module): FairseqDropout()
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (syntax_encoder): SyntaxGlatSyntaxNATransformerEncoder(
      (dropout_module): FairseqDropout()
      (embed_tokens): Embedding(51, 512, padding_idx=0)
      (layers): ModuleList(
        (0): GCNSyntaxGuidedTransformerEncoderLayer(
          (gcn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (W_out): Linear(in_features=1024, out_features=512, bias=True)
          (W_in): Linear(in_features=1024, out_features=512, bias=True)
          (fc): Linear(in_features=512, out_features=512, bias=True)
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (1): GCNSyntaxGuidedTransformerEncoderLayer(
          (gcn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (W_out): Linear(in_features=1024, out_features=512, bias=True)
          (W_in): Linear(in_features=1024, out_features=512, bias=True)
          (fc): Linear(in_features=512, out_features=512, bias=True)
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (2): GCNSyntaxGuidedTransformerEncoderLayer(
          (gcn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout_module): FairseqDropout()
          (activation_dropout_module): FairseqDropout()
          (W_out): Linear(in_features=1024, out_features=512, bias=True)
          (W_in): Linear(in_features=1024, out_features=512, bias=True)
          (fc): Linear(in_features=512, out_features=512, bias=True)
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (decoder): SyntaxGlatNATransformerDecoder(
    (dropout_module): FairseqDropout()
    (embed_tokens): Embedding(34983, 512, padding_idx=1)
    (embed_positions): LearnedPositionalEmbedding(1026, 512, padding_idx=1)
    (layers): ModuleList(
      (0): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (1): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (2): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (3): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (4): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (5): TransformerDecoderLayer(
        (dropout_module): FairseqDropout()
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (activation_dropout_module): FairseqDropout()
        (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttention(
          (dropout_module): FairseqDropout()
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
        )
        (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
    (output_projection): Linear(in_features=512, out_features=34983, bias=False)
    (embed_length): Embedding(1024, 512)
    (copy_attn): Linear(in_features=512, out_features=512, bias=False)
  )
)
2023-06-13 17:29:42 | INFO | fairseq_cli.train | task: syntax-glat-task (SyntaxGlatEnhancedNATTask)
2023-06-13 17:29:42 | INFO | fairseq_cli.train | model: syntax_glat (SyntaxGlat)
2023-06-13 17:29:42 | INFO | fairseq_cli.train | criterion: syntax_glat_loss (SyntaxGlatLabelSmoothedDualImitationCriterion)
2023-06-13 17:29:42 | INFO | fairseq_cli.train | num. model params: 74155008 (num. trained: 74155008)
2023-06-13 17:29:47 | INFO | fairseq.trainer | detected shared parameter: encoder.sentence_encoder.embed_tokens.weight <- decoder.embed_tokens.weight
2023-06-13 17:29:47 | INFO | fairseq.trainer | detected shared parameter: encoder.sentence_encoder.embed_tokens.weight <- decoder.output_projection.weight
2023-06-13 17:29:47 | INFO | fairseq.trainer | detected shared parameter: decoder.output_projection.bias <- decoder.copy_attn.bias
2023-06-13 17:29:47 | INFO | fairseq.utils | ***********************CUDA enviroments for all 8 workers***********************
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   0: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   1: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   2: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   3: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   4: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   5: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   6: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | rank   7: capabilities =  8.6  ; total memory = 23.700 GB ; name = NVIDIA GeForce RTX 3090                 
2023-06-13 17:29:47 | INFO | fairseq.utils | ***********************CUDA enviroments for all 8 workers***********************
2023-06-13 17:29:47 | INFO | fairseq_cli.train | training on 8 devices (GPUs/TPUs)
2023-06-13 17:29:47 | INFO | fairseq_cli.train | max tokens per GPU = 4096 and max sentences per GPU = None
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
2023-06-13 17:29:55 | INFO | fairseq.trainer | loaded checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint_last.pt (epoch 143 @ 40015 updates)
2023-06-13 17:29:55 | INFO | fairseq.trainer | loading train data for epoch 143
2023-06-13 17:29:55 | INFO | fairseq.data.data_utils | loaded 608319 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/train.en-ro.en
2023-06-13 17:29:55 | INFO | fairseq.data.data_utils | loaded 608319 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/train.en-ro.ro
2023-06-13 17:29:55 | INFO | syngec_model.tasks.syntax_glat_nat_task | /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin train en-ro 608319 examples
2023-06-13 17:30:12 | INFO | fairseq.data.data_utils | loaded 608319 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/train.conll.en-ro.en
2023-06-13 17:30:16 | INFO | fairseq.data.data_utils | loaded 608319 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/train.dpd.en-ro.en
2023-06-13 17:30:21 | INFO | fairseq.data.data_utils | loaded 608319 examples from: /opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-ro/deal/all/bin/train.probs.en-ro.en
2023-06-13 17:30:21 | INFO | fairseq.data.language_pair_dataset | success! syntax types: 1, source conll lines: 608319
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
fairseq plugins loaded...
SyntaxGlatLabelSmoothedDualImitationCriterion SyntaxGlatLabelSmoothedDualImitationCriterion
2023-06-13 17:30:21 | INFO | fairseq.trainer | begin training epoch 143
2023-06-13 17:31:08 | INFO | train_inner | epoch 143:     85 / 282 loss=3.147, nll_loss=1.236, glat_accu=0.518, glat_context_p=0.473, word_ins=3.028, length=3.071, ppl=8.86, wps=72384.3, ups=1.21, wpb=59997.9, bsz=2186.3, num_updates=40100, lr=0.000157917, gnorm=0.489, clip=0, loss_scale=32768, train_wall=48, wall=0
2023-06-13 17:31:54 | INFO | train_inner | epoch 143:    185 / 282 loss=3.153, nll_loss=1.243, glat_accu=0.508, glat_context_p=0.473, word_ins=3.034, length=3.068, ppl=8.89, wps=132396, ups=2.18, wpb=60718.4, bsz=2141.4, num_updates=40200, lr=0.00015772, gnorm=0.483, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:32:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:32:41 | INFO | valid | epoch 143 | valid on 'valid' subset | loss 12.677 | nll_loss 11.546 | word_ins 12.459 | length 4.375 | ppl 6550.94 | bleu 29.52 | wps 88225 | wpb 21176.3 | bsz 666.3 | num_updates 40297 | best_bleu 30.53
2023-06-13 17:32:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:32:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint143.pt (epoch 143 @ 40297 updates, score 29.52) (writing took 12.236636321991682 seconds)
2023-06-13 17:32:53 | INFO | fairseq_cli.train | end of epoch 143 (average epoch stats below)
2023-06-13 17:32:53 | INFO | train | epoch 143 | loss 3.14 | nll_loss 1.23 | glat_accu 0.5 | glat_context_p 0.473 | word_ins 3.023 | length 3.065 | ppl 8.82 | wps 106814 | ups 1.77 | wpb 60411 | bsz 2157.9 | num_updates 40297 | lr 0.00015753 | gnorm 0.485 | clip 0 | loss_scale 32768 | train_wall 260 | wall 0
2023-06-13 17:32:54 | INFO | fairseq.trainer | begin training epoch 144
2023-06-13 17:33:01 | INFO | train_inner | epoch 144:      3 / 282 loss=3.14, nll_loss=1.23, glat_accu=0.495, glat_context_p=0.473, word_ins=3.023, length=3.076, ppl=8.82, wps=89307.1, ups=1.49, wpb=60051.3, bsz=2117.5, num_updates=40300, lr=0.000157524, gnorm=0.491, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 17:33:48 | INFO | train_inner | epoch 144:    103 / 282 loss=3.126, nll_loss=1.214, glat_accu=0.485, glat_context_p=0.473, word_ins=3.008, length=3.085, ppl=8.73, wps=128796, ups=2.13, wpb=60501.2, bsz=2133.1, num_updates=40400, lr=0.000157329, gnorm=0.473, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 17:34:34 | INFO | train_inner | epoch 144:    203 / 282 loss=3.122, nll_loss=1.212, glat_accu=0.483, glat_context_p=0.473, word_ins=3.007, length=3.05, ppl=8.71, wps=131361, ups=2.17, wpb=60610.6, bsz=2176.2, num_updates=40500, lr=0.000157135, gnorm=0.468, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:35:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:35:14 | INFO | valid | epoch 144 | valid on 'valid' subset | loss 12.66 | nll_loss 11.518 | word_ins 12.431 | length 4.596 | ppl 6472.42 | bleu 29.03 | wps 86118.5 | wpb 21176.3 | bsz 666.3 | num_updates 40579 | best_bleu 30.53
2023-06-13 17:35:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:35:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint144.pt (epoch 144 @ 40579 updates, score 29.03) (writing took 12.241667237132788 seconds)
2023-06-13 17:35:26 | INFO | fairseq_cli.train | end of epoch 144 (average epoch stats below)
2023-06-13 17:35:26 | INFO | train | epoch 144 | loss 3.122 | nll_loss 1.211 | glat_accu 0.485 | glat_context_p 0.473 | word_ins 3.006 | length 3.063 | ppl 8.71 | wps 111849 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 40579 | lr 0.000156982 | gnorm 0.472 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-13 17:35:26 | INFO | fairseq.trainer | begin training epoch 145
2023-06-13 17:35:41 | INFO | train_inner | epoch 145:     21 / 282 loss=3.12, nll_loss=1.209, glat_accu=0.488, glat_context_p=0.473, word_ins=3.004, length=3.055, ppl=8.69, wps=89594.4, ups=1.49, wpb=60261, bsz=2165, num_updates=40600, lr=0.000156941, gnorm=0.48, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:36:27 | INFO | train_inner | epoch 145:    121 / 282 loss=3.139, nll_loss=1.228, glat_accu=0.515, glat_context_p=0.473, word_ins=3.02, length=3.055, ppl=8.81, wps=133152, ups=2.2, wpb=60468, bsz=2177.2, num_updates=40700, lr=0.000156748, gnorm=0.481, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 17:37:13 | INFO | train_inner | epoch 145:    221 / 282 loss=3.14, nll_loss=1.231, glat_accu=0.5, glat_context_p=0.473, word_ins=3.023, length=3.058, ppl=8.82, wps=132713, ups=2.19, wpb=60722.3, bsz=2156.7, num_updates=40800, lr=0.000156556, gnorm=0.475, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:37:40 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 17:37:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:37:44 | INFO | valid | epoch 145 | valid on 'valid' subset | loss 12.144 | nll_loss 10.956 | word_ins 11.925 | length 4.376 | ppl 4525.39 | bleu 30.17 | wps 82945.9 | wpb 21176.3 | bsz 666.3 | num_updates 40860 | best_bleu 30.53
2023-06-13 17:37:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:37:54 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint145.pt (epoch 145 @ 40860 updates, score 30.17) (writing took 10.804873067885637 seconds)
2023-06-13 17:37:54 | INFO | fairseq_cli.train | end of epoch 145 (average epoch stats below)
2023-06-13 17:37:54 | INFO | train | epoch 145 | loss 3.141 | nll_loss 1.231 | glat_accu 0.508 | glat_context_p 0.473 | word_ins 3.023 | length 3.061 | ppl 8.82 | wps 114280 | ups 1.89 | wpb 60415.2 | bsz 2156.4 | num_updates 40860 | lr 0.000156441 | gnorm 0.484 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 17:37:54 | INFO | fairseq.trainer | begin training epoch 146
2023-06-13 17:38:19 | INFO | train_inner | epoch 146:     40 / 282 loss=3.148, nll_loss=1.238, glat_accu=0.517, glat_context_p=0.473, word_ins=3.03, length=3.039, ppl=8.86, wps=90307.1, ups=1.5, wpb=60115.6, bsz=2162.1, num_updates=40900, lr=0.000156365, gnorm=0.499, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:39:05 | INFO | train_inner | epoch 146:    140 / 282 loss=3.145, nll_loss=1.235, glat_accu=0.518, glat_context_p=0.473, word_ins=3.026, length=3.052, ppl=8.84, wps=132284, ups=2.19, wpb=60462.9, bsz=2175.8, num_updates=41000, lr=0.000156174, gnorm=0.488, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:39:51 | INFO | train_inner | epoch 146:    240 / 282 loss=3.145, nll_loss=1.234, glat_accu=0.504, glat_context_p=0.473, word_ins=3.026, length=3.085, ppl=8.85, wps=131452, ups=2.17, wpb=60620.2, bsz=2100.9, num_updates=41100, lr=0.000155984, gnorm=0.49, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:40:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:40:14 | INFO | valid | epoch 146 | valid on 'valid' subset | loss 12.566 | nll_loss 11.417 | word_ins 12.34 | length 4.501 | ppl 6064.81 | bleu 29.37 | wps 89219.1 | wpb 21176.3 | bsz 666.3 | num_updates 41142 | best_bleu 30.53
2023-06-13 17:40:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:40:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint146.pt (epoch 146 @ 41142 updates, score 29.37) (writing took 12.29176290333271 seconds)
2023-06-13 17:40:26 | INFO | fairseq_cli.train | end of epoch 146 (average epoch stats below)
2023-06-13 17:40:26 | INFO | train | epoch 146 | loss 3.144 | nll_loss 1.234 | glat_accu 0.511 | glat_context_p 0.473 | word_ins 3.026 | length 3.057 | ppl 8.84 | wps 112472 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 41142 | lr 0.000155904 | gnorm 0.491 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 17:40:26 | INFO | fairseq.trainer | begin training epoch 147
2023-06-13 17:40:59 | INFO | train_inner | epoch 147:     58 / 282 loss=3.126, nll_loss=1.215, glat_accu=0.505, glat_context_p=0.473, word_ins=3.009, length=3.04, ppl=8.73, wps=88947.9, ups=1.48, wpb=60071.9, bsz=2224.3, num_updates=41200, lr=0.000155794, gnorm=0.482, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:41:45 | INFO | train_inner | epoch 147:    158 / 282 loss=3.134, nll_loss=1.224, glat_accu=0.502, glat_context_p=0.473, word_ins=3.017, length=3.059, ppl=8.78, wps=130486, ups=2.15, wpb=60564.7, bsz=2164.2, num_updates=41300, lr=0.000155606, gnorm=0.487, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:42:31 | INFO | train_inner | epoch 147:    258 / 282 loss=3.122, nll_loss=1.212, glat_accu=0.483, glat_context_p=0.472, word_ins=3.006, length=3.062, ppl=8.71, wps=132569, ups=2.19, wpb=60571.8, bsz=2127.5, num_updates=41400, lr=0.000155417, gnorm=0.473, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:42:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:42:45 | INFO | valid | epoch 147 | valid on 'valid' subset | loss 12.605 | nll_loss 11.461 | word_ins 12.38 | length 4.531 | ppl 6230.81 | bleu 29.38 | wps 89346.5 | wpb 21176.3 | bsz 666.3 | num_updates 41424 | best_bleu 30.53
2023-06-13 17:42:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:43:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint147.pt (epoch 147 @ 41424 updates, score 29.38) (writing took 15.482302289456129 seconds)
2023-06-13 17:43:00 | INFO | fairseq_cli.train | end of epoch 147 (average epoch stats below)
2023-06-13 17:43:00 | INFO | train | epoch 147 | loss 3.127 | nll_loss 1.216 | glat_accu 0.496 | glat_context_p 0.472 | word_ins 3.01 | length 3.054 | ppl 8.73 | wps 110247 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 41424 | lr 0.000155372 | gnorm 0.481 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 17:43:00 | INFO | fairseq.trainer | begin training epoch 148
2023-06-13 17:43:41 | INFO | train_inner | epoch 148:     76 / 282 loss=3.142, nll_loss=1.231, glat_accu=0.514, glat_context_p=0.472, word_ins=3.023, length=3.064, ppl=8.83, wps=85304.9, ups=1.42, wpb=60114.8, bsz=2142.2, num_updates=41500, lr=0.00015523, gnorm=0.502, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 17:44:28 | INFO | train_inner | epoch 148:    176 / 282 loss=3.138, nll_loss=1.228, glat_accu=0.51, glat_context_p=0.472, word_ins=3.02, length=3.052, ppl=8.8, wps=130349, ups=2.15, wpb=60519.9, bsz=2194.1, num_updates=41600, lr=0.000155043, gnorm=0.494, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:45:13 | INFO | train_inner | epoch 148:    276 / 282 loss=3.136, nll_loss=1.226, glat_accu=0.501, glat_context_p=0.472, word_ins=3.018, length=3.061, ppl=8.79, wps=132761, ups=2.19, wpb=60709, bsz=2126.7, num_updates=41700, lr=0.000154857, gnorm=0.488, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:45:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:45:19 | INFO | valid | epoch 148 | valid on 'valid' subset | loss 12.295 | nll_loss 11.114 | word_ins 12.071 | length 4.492 | ppl 5023.63 | bleu 29.92 | wps 89173.9 | wpb 21176.3 | bsz 666.3 | num_updates 41706 | best_bleu 30.53
2023-06-13 17:45:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:45:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint148.pt (epoch 148 @ 41706 updates, score 29.92) (writing took 12.920390415936708 seconds)
2023-06-13 17:45:32 | INFO | fairseq_cli.train | end of epoch 148 (average epoch stats below)
2023-06-13 17:45:32 | INFO | train | epoch 148 | loss 3.139 | nll_loss 1.228 | glat_accu 0.51 | glat_context_p 0.472 | word_ins 3.021 | length 3.06 | ppl 8.81 | wps 112359 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 41706 | lr 0.000154846 | gnorm 0.495 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 17:45:32 | INFO | fairseq.trainer | begin training epoch 149
2023-06-13 17:46:22 | INFO | train_inner | epoch 149:     94 / 282 loss=3.133, nll_loss=1.221, glat_accu=0.512, glat_context_p=0.472, word_ins=3.014, length=3.054, ppl=8.77, wps=87120.4, ups=1.45, wpb=60035.7, bsz=2120.2, num_updates=41800, lr=0.000154672, gnorm=0.494, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:47:08 | INFO | train_inner | epoch 149:    194 / 282 loss=3.13, nll_loss=1.22, glat_accu=0.501, glat_context_p=0.472, word_ins=3.013, length=3.044, ppl=8.75, wps=131226, ups=2.16, wpb=60696.3, bsz=2179.9, num_updates=41900, lr=0.000154487, gnorm=0.486, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:47:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:47:52 | INFO | valid | epoch 149 | valid on 'valid' subset | loss 12.374 | nll_loss 11.205 | word_ins 12.151 | length 4.447 | ppl 5306.92 | bleu 30.34 | wps 89200.1 | wpb 21176.3 | bsz 666.3 | num_updates 41988 | best_bleu 30.53
2023-06-13 17:47:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:48:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint149.pt (epoch 149 @ 41988 updates, score 30.34) (writing took 12.29564269632101 seconds)
2023-06-13 17:48:04 | INFO | fairseq_cli.train | end of epoch 149 (average epoch stats below)
2023-06-13 17:48:04 | INFO | train | epoch 149 | loss 3.136 | nll_loss 1.226 | glat_accu 0.509 | glat_context_p 0.472 | word_ins 3.018 | length 3.056 | ppl 8.79 | wps 111829 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 41988 | lr 0.000154325 | gnorm 0.492 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 17:48:04 | INFO | fairseq.trainer | begin training epoch 150
2023-06-13 17:48:16 | INFO | train_inner | epoch 150:     12 / 282 loss=3.146, nll_loss=1.236, glat_accu=0.517, glat_context_p=0.472, word_ins=3.027, length=3.069, ppl=8.85, wps=88808.2, ups=1.48, wpb=60063.8, bsz=2155.5, num_updates=42000, lr=0.000154303, gnorm=0.503, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:49:03 | INFO | train_inner | epoch 150:    112 / 282 loss=3.139, nll_loss=1.229, glat_accu=0.513, glat_context_p=0.472, word_ins=3.021, length=3.057, ppl=8.81, wps=129330, ups=2.14, wpb=60560.3, bsz=2131.5, num_updates=42100, lr=0.00015412, gnorm=0.487, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 17:49:49 | INFO | train_inner | epoch 150:    212 / 282 loss=3.149, nll_loss=1.238, glat_accu=0.523, glat_context_p=0.472, word_ins=3.029, length=3.07, ppl=8.87, wps=129744, ups=2.15, wpb=60413.8, bsz=2153.3, num_updates=42200, lr=0.000153937, gnorm=0.5, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:50:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:50:24 | INFO | valid | epoch 150 | valid on 'valid' subset | loss 12.401 | nll_loss 11.212 | word_ins 12.156 | length 4.893 | ppl 5408.83 | bleu 29.55 | wps 87824 | wpb 21176.3 | bsz 666.3 | num_updates 42270 | best_bleu 30.53
2023-06-13 17:50:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:50:39 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint150.pt (epoch 150 @ 42270 updates, score 29.55) (writing took 14.565176963806152 seconds)
2023-06-13 17:50:39 | INFO | fairseq_cli.train | end of epoch 150 (average epoch stats below)
2023-06-13 17:50:39 | INFO | train | epoch 150 | loss 3.144 | nll_loss 1.234 | glat_accu 0.521 | glat_context_p 0.472 | word_ins 3.025 | length 3.054 | ppl 8.84 | wps 110232 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 42270 | lr 0.00015381 | gnorm 0.497 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-13 17:50:39 | INFO | fairseq.trainer | begin training epoch 151
2023-06-13 17:50:59 | INFO | train_inner | epoch 151:     30 / 282 loss=3.145, nll_loss=1.235, glat_accu=0.529, glat_context_p=0.472, word_ins=3.026, length=3.051, ppl=8.85, wps=87019, ups=1.45, wpb=60109.7, bsz=2199, num_updates=42300, lr=0.000153755, gnorm=0.507, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 17:51:44 | INFO | train_inner | epoch 151:    130 / 282 loss=3.127, nll_loss=1.217, glat_accu=0.506, glat_context_p=0.472, word_ins=3.01, length=3.05, ppl=8.74, wps=132378, ups=2.19, wpb=60575.8, bsz=2177, num_updates=42400, lr=0.000153574, gnorm=0.486, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:52:30 | INFO | train_inner | epoch 151:    230 / 282 loss=3.137, nll_loss=1.226, glat_accu=0.511, glat_context_p=0.472, word_ins=3.019, length=3.051, ppl=8.8, wps=131792, ups=2.17, wpb=60627.7, bsz=2154.2, num_updates=42500, lr=0.000153393, gnorm=0.492, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 17:52:54 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 17:52:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:52:58 | INFO | valid | epoch 151 | valid on 'valid' subset | loss 12.325 | nll_loss 11.15 | word_ins 12.101 | length 4.487 | ppl 5129.68 | bleu 30.32 | wps 87087.9 | wpb 21176.3 | bsz 666.3 | num_updates 42551 | best_bleu 30.53
2023-06-13 17:52:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:53:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint151.pt (epoch 151 @ 42551 updates, score 30.32) (writing took 13.586838316172361 seconds)
2023-06-13 17:53:11 | INFO | fairseq_cli.train | end of epoch 151 (average epoch stats below)
2023-06-13 17:53:11 | INFO | train | epoch 151 | loss 3.135 | nll_loss 1.225 | glat_accu 0.513 | glat_context_p 0.472 | word_ins 3.017 | length 3.053 | ppl 8.79 | wps 111809 | ups 1.85 | wpb 60591.7 | bsz 2164.1 | num_updates 42551 | lr 0.000153301 | gnorm 0.493 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 17:53:11 | INFO | fairseq.trainer | begin training epoch 152
2023-06-13 17:53:40 | INFO | train_inner | epoch 152:     49 / 282 loss=3.129, nll_loss=1.219, glat_accu=0.506, glat_context_p=0.472, word_ins=3.012, length=3.038, ppl=8.75, wps=87155.6, ups=1.44, wpb=60729, bsz=2137.9, num_updates=42600, lr=0.000153213, gnorm=0.483, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:54:26 | INFO | train_inner | epoch 152:    149 / 282 loss=3.127, nll_loss=1.215, glat_accu=0.511, glat_context_p=0.472, word_ins=3.009, length=3.052, ppl=8.73, wps=132726, ups=2.19, wpb=60506, bsz=2171.8, num_updates=42700, lr=0.000153033, gnorm=0.483, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 17:55:12 | INFO | train_inner | epoch 152:    249 / 282 loss=3.133, nll_loss=1.222, glat_accu=0.508, glat_context_p=0.472, word_ins=3.015, length=3.048, ppl=8.77, wps=131951, ups=2.17, wpb=60701.6, bsz=2171.8, num_updates=42800, lr=0.000152854, gnorm=0.489, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:55:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:55:30 | INFO | valid | epoch 152 | valid on 'valid' subset | loss 12.179 | nll_loss 10.985 | word_ins 11.951 | length 4.559 | ppl 4636.64 | bleu 30.02 | wps 87236.9 | wpb 21176.3 | bsz 666.3 | num_updates 42833 | best_bleu 30.53
2023-06-13 17:55:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:55:41 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint152.pt (epoch 152 @ 42833 updates, score 30.02) (writing took 11.017280511558056 seconds)
2023-06-13 17:55:41 | INFO | fairseq_cli.train | end of epoch 152 (average epoch stats below)
2023-06-13 17:55:41 | INFO | train | epoch 152 | loss 3.129 | nll_loss 1.218 | glat_accu 0.507 | glat_context_p 0.472 | word_ins 3.011 | length 3.048 | ppl 8.75 | wps 113525 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 42833 | lr 0.000152796 | gnorm 0.486 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 17:55:41 | INFO | fairseq.trainer | begin training epoch 153
2023-06-13 17:56:18 | INFO | train_inner | epoch 153:     67 / 282 loss=3.142, nll_loss=1.232, glat_accu=0.514, glat_context_p=0.471, word_ins=3.023, length=3.058, ppl=8.83, wps=90111.7, ups=1.5, wpb=60147.4, bsz=2120, num_updates=42900, lr=0.000152676, gnorm=0.492, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:57:04 | INFO | train_inner | epoch 153:    167 / 282 loss=3.112, nll_loss=1.2, glat_accu=0.491, glat_context_p=0.471, word_ins=2.995, length=3.063, ppl=8.65, wps=131551, ups=2.17, wpb=60556.9, bsz=2160.9, num_updates=43000, lr=0.000152499, gnorm=0.473, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:57:51 | INFO | train_inner | epoch 153:    267 / 282 loss=3.106, nll_loss=1.195, glat_accu=0.484, glat_context_p=0.471, word_ins=2.991, length=3.049, ppl=8.61, wps=130466, ups=2.16, wpb=60511.9, bsz=2192, num_updates=43100, lr=0.000152322, gnorm=0.475, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:57:57 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 17:58:01 | INFO | valid | epoch 153 | valid on 'valid' subset | loss 12.412 | nll_loss 11.241 | word_ins 12.184 | length 4.547 | ppl 5449.4 | bleu 29.87 | wps 89179 | wpb 21176.3 | bsz 666.3 | num_updates 43115 | best_bleu 30.53
2023-06-13 17:58:01 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 17:58:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint153.pt (epoch 153 @ 43115 updates, score 29.87) (writing took 11.18710159510374 seconds)
2023-06-13 17:58:12 | INFO | fairseq_cli.train | end of epoch 153 (average epoch stats below)
2023-06-13 17:58:12 | INFO | train | epoch 153 | loss 3.117 | nll_loss 1.206 | glat_accu 0.494 | glat_context_p 0.471 | word_ins 3.001 | length 3.057 | ppl 8.68 | wps 113061 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 43115 | lr 0.000152295 | gnorm 0.481 | clip 0 | loss_scale 16384 | train_wall 130 | wall 0
2023-06-13 17:58:12 | INFO | fairseq.trainer | begin training epoch 154
2023-06-13 17:58:57 | INFO | train_inner | epoch 154:     85 / 282 loss=3.133, nll_loss=1.222, glat_accu=0.523, glat_context_p=0.471, word_ins=3.015, length=3.036, ppl=8.77, wps=90186.7, ups=1.5, wpb=59950.7, bsz=2162.2, num_updates=43200, lr=0.000152145, gnorm=0.513, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 17:59:44 | INFO | train_inner | epoch 154:    185 / 282 loss=3.129, nll_loss=1.218, glat_accu=0.506, glat_context_p=0.471, word_ins=3.011, length=3.058, ppl=8.75, wps=129316, ups=2.13, wpb=60613.4, bsz=2167, num_updates=43300, lr=0.000151969, gnorm=0.479, clip=0, loss_scale=16384, train_wall=47, wall=0
2023-06-13 18:00:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:00:32 | INFO | valid | epoch 154 | valid on 'valid' subset | loss 12.283 | nll_loss 11.106 | word_ins 12.061 | length 4.446 | ppl 4983.69 | bleu 30.46 | wps 87913.1 | wpb 21176.3 | bsz 666.3 | num_updates 43397 | best_bleu 30.53
2023-06-13 18:00:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:00:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint154.pt (epoch 154 @ 43397 updates, score 30.46) (writing took 12.298829331994057 seconds)
2023-06-13 18:00:45 | INFO | fairseq_cli.train | end of epoch 154 (average epoch stats below)
2023-06-13 18:00:45 | INFO | train | epoch 154 | loss 3.13 | nll_loss 1.219 | glat_accu 0.513 | glat_context_p 0.471 | word_ins 3.012 | length 3.053 | ppl 8.76 | wps 111648 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 43397 | lr 0.000151799 | gnorm 0.492 | clip 0 | loss_scale 16384 | train_wall 130 | wall 0
2023-06-13 18:00:45 | INFO | fairseq.trainer | begin training epoch 155
2023-06-13 18:00:52 | INFO | train_inner | epoch 155:      3 / 282 loss=3.128, nll_loss=1.217, glat_accu=0.507, glat_context_p=0.471, word_ins=3.01, length=3.068, ppl=8.74, wps=88851.8, ups=1.47, wpb=60247.3, bsz=2124.1, num_updates=43400, lr=0.000151794, gnorm=0.493, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:01:38 | INFO | train_inner | epoch 155:    103 / 282 loss=3.133, nll_loss=1.221, glat_accu=0.523, glat_context_p=0.471, word_ins=3.014, length=3.048, ppl=8.77, wps=131517, ups=2.17, wpb=60564.6, bsz=2163.3, num_updates=43500, lr=0.00015162, gnorm=0.497, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:02:24 | INFO | train_inner | epoch 155:    203 / 282 loss=3.143, nll_loss=1.233, glat_accu=0.521, glat_context_p=0.471, word_ins=3.024, length=3.067, ppl=8.84, wps=131096, ups=2.16, wpb=60570.7, bsz=2158.6, num_updates=43600, lr=0.000151446, gnorm=0.504, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:03:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:03:04 | INFO | valid | epoch 155 | valid on 'valid' subset | loss 12.576 | nll_loss 11.435 | word_ins 12.35 | length 4.516 | ppl 6106.15 | bleu 29.02 | wps 89090.2 | wpb 21176.3 | bsz 666.3 | num_updates 43679 | best_bleu 30.53
2023-06-13 18:03:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:03:14 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint155.pt (epoch 155 @ 43679 updates, score 29.02) (writing took 10.52864532545209 seconds)
2023-06-13 18:03:14 | INFO | fairseq_cli.train | end of epoch 155 (average epoch stats below)
2023-06-13 18:03:14 | INFO | train | epoch 155 | loss 3.131 | nll_loss 1.22 | glat_accu 0.514 | glat_context_p 0.471 | word_ins 3.013 | length 3.053 | ppl 8.76 | wps 113682 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 43679 | lr 0.000151309 | gnorm 0.493 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-13 18:03:14 | INFO | fairseq.trainer | begin training epoch 156
2023-06-13 18:03:30 | INFO | train_inner | epoch 156:     21 / 282 loss=3.113, nll_loss=1.202, glat_accu=0.492, glat_context_p=0.471, word_ins=2.997, length=3.045, ppl=8.65, wps=91515.7, ups=1.52, wpb=60178.1, bsz=2138.4, num_updates=43700, lr=0.000151272, gnorm=0.474, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:04:16 | INFO | train_inner | epoch 156:    121 / 282 loss=3.101, nll_loss=1.19, glat_accu=0.477, glat_context_p=0.471, word_ins=2.986, length=3.063, ppl=8.58, wps=130943, ups=2.16, wpb=60609, bsz=2134.1, num_updates=43800, lr=0.000151099, gnorm=0.473, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:05:02 | INFO | train_inner | epoch 156:    221 / 282 loss=3.101, nll_loss=1.189, glat_accu=0.492, glat_context_p=0.471, word_ins=2.985, length=3.036, ppl=8.58, wps=132711, ups=2.19, wpb=60635.4, bsz=2205, num_updates=43900, lr=0.000150927, gnorm=0.476, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:05:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:05:33 | INFO | valid | epoch 156 | valid on 'valid' subset | loss 12.602 | nll_loss 11.456 | word_ins 12.372 | length 4.587 | ppl 6214.96 | bleu 29.58 | wps 88132.5 | wpb 21176.3 | bsz 666.3 | num_updates 43961 | best_bleu 30.53
2023-06-13 18:05:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:05:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint156.pt (epoch 156 @ 43961 updates, score 29.58) (writing took 11.364179488271475 seconds)
2023-06-13 18:05:44 | INFO | fairseq_cli.train | end of epoch 156 (average epoch stats below)
2023-06-13 18:05:44 | INFO | train | epoch 156 | loss 3.106 | nll_loss 1.195 | glat_accu 0.489 | glat_context_p 0.471 | word_ins 2.99 | length 3.049 | ppl 8.61 | wps 113662 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 43961 | lr 0.000150823 | gnorm 0.476 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:05:44 | INFO | fairseq.trainer | begin training epoch 157
2023-06-13 18:06:08 | INFO | train_inner | epoch 157:     39 / 282 loss=3.116, nll_loss=1.204, glat_accu=0.504, glat_context_p=0.471, word_ins=2.999, length=3.041, ppl=8.67, wps=90074.6, ups=1.5, wpb=59888.9, bsz=2167.6, num_updates=44000, lr=0.000150756, gnorm=0.478, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:06:54 | INFO | train_inner | epoch 157:    139 / 282 loss=3.094, nll_loss=1.181, glat_accu=0.475, glat_context_p=0.471, word_ins=2.978, length=3.052, ppl=8.54, wps=132632, ups=2.19, wpb=60613.8, bsz=2146.2, num_updates=44100, lr=0.000150585, gnorm=0.466, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:07:40 | INFO | train_inner | epoch 157:    239 / 282 loss=3.116, nll_loss=1.206, glat_accu=0.499, glat_context_p=0.471, word_ins=2.999, length=3.049, ppl=8.67, wps=131796, ups=2.17, wpb=60605.8, bsz=2172.6, num_updates=44200, lr=0.000150414, gnorm=0.489, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:08:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:08:03 | INFO | valid | epoch 157 | valid on 'valid' subset | loss 12.494 | nll_loss 11.327 | word_ins 12.256 | length 4.761 | ppl 5767.46 | bleu 29.67 | wps 88567.4 | wpb 21176.3 | bsz 666.3 | num_updates 44243 | best_bleu 30.53
2023-06-13 18:08:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:08:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint157.pt (epoch 157 @ 44243 updates, score 29.67) (writing took 9.51749249175191 seconds)
2023-06-13 18:08:12 | INFO | fairseq_cli.train | end of epoch 157 (average epoch stats below)
2023-06-13 18:08:12 | INFO | train | epoch 157 | loss 3.108 | nll_loss 1.196 | glat_accu 0.492 | glat_context_p 0.471 | word_ins 2.992 | length 3.047 | ppl 8.62 | wps 115073 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 44243 | lr 0.000150341 | gnorm 0.479 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:08:12 | INFO | fairseq.trainer | begin training epoch 158
2023-06-13 18:08:44 | INFO | train_inner | epoch 158:     57 / 282 loss=3.11, nll_loss=1.197, glat_accu=0.494, glat_context_p=0.471, word_ins=2.992, length=3.065, ppl=8.63, wps=94222.9, ups=1.57, wpb=60045.5, bsz=2108.8, num_updates=44300, lr=0.000150244, gnorm=0.49, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:09:30 | INFO | train_inner | epoch 158:    157 / 282 loss=3.109, nll_loss=1.197, glat_accu=0.5, glat_context_p=0.47, word_ins=2.992, length=3.038, ppl=8.63, wps=131578, ups=2.17, wpb=60650.1, bsz=2139.6, num_updates=44400, lr=0.000150075, gnorm=0.488, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:10:16 | INFO | train_inner | epoch 158:    257 / 282 loss=3.11, nll_loss=1.198, glat_accu=0.494, glat_context_p=0.47, word_ins=2.993, length=3.047, ppl=8.63, wps=132784, ups=2.19, wpb=60637.7, bsz=2180.7, num_updates=44500, lr=0.000149906, gnorm=0.487, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:10:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:10:30 | INFO | valid | epoch 158 | valid on 'valid' subset | loss 12.562 | nll_loss 11.42 | word_ins 12.335 | length 4.519 | ppl 6045.27 | bleu 29.22 | wps 89343 | wpb 21176.3 | bsz 666.3 | num_updates 44525 | best_bleu 30.53
2023-06-13 18:10:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:10:41 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint158.pt (epoch 158 @ 44525 updates, score 29.22) (writing took 11.38694366440177 seconds)
2023-06-13 18:10:41 | INFO | fairseq_cli.train | end of epoch 158 (average epoch stats below)
2023-06-13 18:10:41 | INFO | train | epoch 158 | loss 3.108 | nll_loss 1.196 | glat_accu 0.495 | glat_context_p 0.47 | word_ins 2.991 | length 3.049 | ppl 8.62 | wps 114231 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 44525 | lr 0.000149864 | gnorm 0.488 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:10:42 | INFO | fairseq.trainer | begin training epoch 159
2023-06-13 18:11:22 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 18:11:23 | INFO | train_inner | epoch 159:     76 / 282 loss=3.122, nll_loss=1.211, glat_accu=0.516, glat_context_p=0.47, word_ins=3.004, length=3.032, ppl=8.7, wps=89588.4, ups=1.49, wpb=60011.2, bsz=2198.3, num_updates=44600, lr=0.000149738, gnorm=0.491, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:12:08 | INFO | train_inner | epoch 159:    176 / 282 loss=3.134, nll_loss=1.222, glat_accu=0.524, glat_context_p=0.47, word_ins=3.014, length=3.054, ppl=8.78, wps=132801, ups=2.19, wpb=60639.9, bsz=2138, num_updates=44700, lr=0.000149571, gnorm=0.507, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:12:54 | INFO | train_inner | epoch 159:    276 / 282 loss=3.135, nll_loss=1.224, glat_accu=0.518, glat_context_p=0.47, word_ins=3.016, length=3.067, ppl=8.79, wps=131317, ups=2.17, wpb=60599.5, bsz=2151, num_updates=44800, lr=0.000149404, gnorm=0.497, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:12:57 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:13:00 | INFO | valid | epoch 159 | valid on 'valid' subset | loss 12.521 | nll_loss 11.352 | word_ins 12.285 | length 4.702 | ppl 5877.22 | bleu 30.08 | wps 89574 | wpb 21176.3 | bsz 666.3 | num_updates 44806 | best_bleu 30.53
2023-06-13 18:13:00 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:13:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint159.pt (epoch 159 @ 44806 updates, score 30.08) (writing took 8.00960198417306 seconds)
2023-06-13 18:13:08 | INFO | fairseq_cli.train | end of epoch 159 (average epoch stats below)
2023-06-13 18:13:08 | INFO | train | epoch 159 | loss 3.132 | nll_loss 1.221 | glat_accu 0.523 | glat_context_p 0.47 | word_ins 3.013 | length 3.049 | ppl 8.77 | wps 115707 | ups 1.92 | wpb 60412.4 | bsz 2158.4 | num_updates 44806 | lr 0.000149394 | gnorm 0.5 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:13:08 | INFO | fairseq.trainer | begin training epoch 160
2023-06-13 18:13:58 | INFO | train_inner | epoch 160:     94 / 282 loss=3.121, nll_loss=1.21, glat_accu=0.526, glat_context_p=0.47, word_ins=3.003, length=3.023, ppl=8.7, wps=94980.6, ups=1.58, wpb=60119.8, bsz=2192.2, num_updates=44900, lr=0.000149237, gnorm=0.501, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:14:44 | INFO | train_inner | epoch 160:    194 / 282 loss=3.13, nll_loss=1.218, glat_accu=0.524, glat_context_p=0.47, word_ins=3.011, length=3.049, ppl=8.75, wps=130550, ups=2.16, wpb=60523.5, bsz=2148.4, num_updates=45000, lr=0.000149071, gnorm=0.488, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:15:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:15:27 | INFO | valid | epoch 160 | valid on 'valid' subset | loss 12.413 | nll_loss 11.242 | word_ins 12.177 | length 4.718 | ppl 5452.18 | bleu 29.85 | wps 88589.1 | wpb 21176.3 | bsz 666.3 | num_updates 45088 | best_bleu 30.53
2023-06-13 18:15:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:15:38 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint160.pt (epoch 160 @ 45088 updates, score 29.85) (writing took 10.83853891119361 seconds)
2023-06-13 18:15:38 | INFO | fairseq_cli.train | end of epoch 160 (average epoch stats below)
2023-06-13 18:15:38 | INFO | train | epoch 160 | loss 3.125 | nll_loss 1.213 | glat_accu 0.518 | glat_context_p 0.47 | word_ins 3.007 | length 3.046 | ppl 8.72 | wps 113545 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 45088 | lr 0.000148926 | gnorm 0.495 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:15:38 | INFO | fairseq.trainer | begin training epoch 161
2023-06-13 18:15:50 | INFO | train_inner | epoch 161:     12 / 282 loss=3.122, nll_loss=1.21, glat_accu=0.509, glat_context_p=0.47, word_ins=3.004, length=3.049, ppl=8.7, wps=91490.1, ups=1.52, wpb=60144.7, bsz=2145.5, num_updates=45100, lr=0.000148906, gnorm=0.502, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:16:36 | INFO | train_inner | epoch 161:    112 / 282 loss=3.112, nll_loss=1.2, glat_accu=0.505, glat_context_p=0.47, word_ins=2.994, length=3.054, ppl=8.65, wps=132281, ups=2.18, wpb=60585.1, bsz=2139.8, num_updates=45200, lr=0.000148741, gnorm=0.487, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:17:22 | INFO | train_inner | epoch 161:    212 / 282 loss=3.115, nll_loss=1.203, glat_accu=0.515, glat_context_p=0.47, word_ins=2.997, length=3.042, ppl=8.67, wps=130075, ups=2.14, wpb=60656.7, bsz=2187.1, num_updates=45300, lr=0.000148577, gnorm=0.5, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:17:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:17:58 | INFO | valid | epoch 161 | valid on 'valid' subset | loss 12.48 | nll_loss 11.322 | word_ins 12.252 | length 4.553 | ppl 5712.86 | bleu 29.57 | wps 89142.1 | wpb 21176.3 | bsz 666.3 | num_updates 45370 | best_bleu 30.53
2023-06-13 18:17:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:18:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint161.pt (epoch 161 @ 45370 updates, score 29.57) (writing took 11.046295884996653 seconds)
2023-06-13 18:18:09 | INFO | fairseq_cli.train | end of epoch 161 (average epoch stats below)
2023-06-13 18:18:09 | INFO | train | epoch 161 | loss 3.114 | nll_loss 1.202 | glat_accu 0.509 | glat_context_p 0.47 | word_ins 2.996 | length 3.051 | ppl 8.66 | wps 113310 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 45370 | lr 0.000148462 | gnorm 0.494 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:18:09 | INFO | fairseq.trainer | begin training epoch 162
2023-06-13 18:18:28 | INFO | train_inner | epoch 162:     30 / 282 loss=3.113, nll_loss=1.201, glat_accu=0.504, glat_context_p=0.47, word_ins=2.995, length=3.053, ppl=8.65, wps=90836.3, ups=1.51, wpb=60114.2, bsz=2130.6, num_updates=45400, lr=0.000148413, gnorm=0.492, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:19:14 | INFO | train_inner | epoch 162:    130 / 282 loss=3.114, nll_loss=1.201, glat_accu=0.506, glat_context_p=0.47, word_ins=2.996, length=3.056, ppl=8.66, wps=133706, ups=2.21, wpb=60617.6, bsz=2167.3, num_updates=45500, lr=0.00014825, gnorm=0.497, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:19:33 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 18:20:00 | INFO | train_inner | epoch 162:    231 / 282 loss=3.127, nll_loss=1.216, glat_accu=0.524, glat_context_p=0.47, word_ins=3.009, length=3.04, ppl=8.74, wps=130646, ups=2.16, wpb=60568.4, bsz=2178.7, num_updates=45600, lr=0.000148087, gnorm=0.494, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:20:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:20:27 | INFO | valid | epoch 162 | valid on 'valid' subset | loss 12.337 | nll_loss 11.169 | word_ins 12.109 | length 4.558 | ppl 5172.55 | bleu 30.03 | wps 88465.6 | wpb 21176.3 | bsz 666.3 | num_updates 45651 | best_bleu 30.53
2023-06-13 18:20:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:20:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint162.pt (epoch 162 @ 45651 updates, score 30.03) (writing took 12.836730297654867 seconds)
2023-06-13 18:20:40 | INFO | fairseq_cli.train | end of epoch 162 (average epoch stats below)
2023-06-13 18:20:40 | INFO | train | epoch 162 | loss 3.121 | nll_loss 1.21 | glat_accu 0.514 | glat_context_p 0.47 | word_ins 3.003 | length 3.048 | ppl 8.7 | wps 112327 | ups 1.86 | wpb 60411 | bsz 2157.9 | num_updates 45651 | lr 0.000148004 | gnorm 0.498 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 18:20:40 | INFO | fairseq.trainer | begin training epoch 163
2023-06-13 18:21:09 | INFO | train_inner | epoch 163:     49 / 282 loss=3.126, nll_loss=1.214, glat_accu=0.518, glat_context_p=0.47, word_ins=3.007, length=3.06, ppl=8.73, wps=87671.4, ups=1.46, wpb=59990.6, bsz=2119.7, num_updates=45700, lr=0.000147925, gnorm=0.513, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:21:54 | INFO | train_inner | epoch 163:    149 / 282 loss=3.123, nll_loss=1.212, glat_accu=0.521, glat_context_p=0.47, word_ins=3.005, length=3.039, ppl=8.71, wps=132397, ups=2.18, wpb=60598, bsz=2172.7, num_updates=45800, lr=0.000147764, gnorm=0.489, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:22:40 | INFO | train_inner | epoch 163:    249 / 282 loss=3.105, nll_loss=1.193, glat_accu=0.495, glat_context_p=0.469, word_ins=2.988, length=3.036, ppl=8.6, wps=131547, ups=2.17, wpb=60566, bsz=2174.6, num_updates=45900, lr=0.000147602, gnorm=0.479, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:22:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:22:58 | INFO | valid | epoch 163 | valid on 'valid' subset | loss 12.507 | nll_loss 11.346 | word_ins 12.276 | length 4.647 | ppl 5822.57 | bleu 29.55 | wps 88868.5 | wpb 21176.3 | bsz 666.3 | num_updates 45933 | best_bleu 30.53
2023-06-13 18:22:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:23:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint163.pt (epoch 163 @ 45933 updates, score 29.55) (writing took 11.10513224452734 seconds)
2023-06-13 18:23:09 | INFO | fairseq_cli.train | end of epoch 163 (average epoch stats below)
2023-06-13 18:23:09 | INFO | train | epoch 163 | loss 3.116 | nll_loss 1.204 | glat_accu 0.509 | glat_context_p 0.469 | word_ins 2.998 | length 3.042 | ppl 8.67 | wps 113782 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 45933 | lr 0.000147549 | gnorm 0.491 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 18:23:10 | INFO | fairseq.trainer | begin training epoch 164
2023-06-13 18:23:46 | INFO | train_inner | epoch 164:     67 / 282 loss=3.115, nll_loss=1.204, glat_accu=0.506, glat_context_p=0.469, word_ins=2.998, length=3.044, ppl=8.67, wps=90920.8, ups=1.51, wpb=60078, bsz=2154.8, num_updates=46000, lr=0.000147442, gnorm=0.495, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 18:24:33 | INFO | train_inner | epoch 164:    167 / 282 loss=3.091, nll_loss=1.179, glat_accu=0.485, glat_context_p=0.469, word_ins=2.976, length=3.044, ppl=8.52, wps=131135, ups=2.16, wpb=60611.9, bsz=2155.4, num_updates=46100, lr=0.000147282, gnorm=0.479, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:25:18 | INFO | train_inner | epoch 164:    267 / 282 loss=3.107, nll_loss=1.196, glat_accu=0.501, glat_context_p=0.469, word_ins=2.991, length=3.029, ppl=8.62, wps=133766, ups=2.21, wpb=60603.8, bsz=2168.7, num_updates=46200, lr=0.000147122, gnorm=0.488, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 18:25:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:25:28 | INFO | valid | epoch 164 | valid on 'valid' subset | loss 12.508 | nll_loss 11.353 | word_ins 12.283 | length 4.505 | ppl 5824.55 | bleu 29.8 | wps 88203.7 | wpb 21176.3 | bsz 666.3 | num_updates 46215 | best_bleu 30.53
2023-06-13 18:25:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:25:38 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint164.pt (epoch 164 @ 46215 updates, score 29.8) (writing took 10.656541369855404 seconds)
2023-06-13 18:25:38 | INFO | fairseq_cli.train | end of epoch 164 (average epoch stats below)
2023-06-13 18:25:38 | INFO | train | epoch 164 | loss 3.103 | nll_loss 1.191 | glat_accu 0.497 | glat_context_p 0.469 | word_ins 2.987 | length 3.039 | ppl 8.59 | wps 114339 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 46215 | lr 0.000147099 | gnorm 0.485 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 18:25:39 | INFO | fairseq.trainer | begin training epoch 165
2023-06-13 18:26:23 | INFO | train_inner | epoch 165:     85 / 282 loss=3.109, nll_loss=1.196, glat_accu=0.5, glat_context_p=0.469, word_ins=2.991, length=3.066, ppl=8.63, wps=91937.6, ups=1.54, wpb=59831.6, bsz=2083.2, num_updates=46300, lr=0.000146964, gnorm=0.488, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 18:27:09 | INFO | train_inner | epoch 165:    185 / 282 loss=3.107, nll_loss=1.196, glat_accu=0.503, glat_context_p=0.469, word_ins=2.99, length=3.035, ppl=8.62, wps=131449, ups=2.17, wpb=60681.2, bsz=2156.4, num_updates=46400, lr=0.000146805, gnorm=0.485, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:27:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:27:57 | INFO | valid | epoch 165 | valid on 'valid' subset | loss 12.472 | nll_loss 11.316 | word_ins 12.245 | length 4.52 | ppl 5680.93 | bleu 30.28 | wps 85687.3 | wpb 21176.3 | bsz 666.3 | num_updates 46497 | best_bleu 30.53
2023-06-13 18:27:57 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:28:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint165.pt (epoch 165 @ 46497 updates, score 30.28) (writing took 10.800905786454678 seconds)
2023-06-13 18:28:08 | INFO | fairseq_cli.train | end of epoch 165 (average epoch stats below)
2023-06-13 18:28:08 | INFO | train | epoch 165 | loss 3.109 | nll_loss 1.198 | glat_accu 0.506 | glat_context_p 0.469 | word_ins 2.992 | length 3.041 | ppl 8.63 | wps 113992 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 46497 | lr 0.000146652 | gnorm 0.486 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 18:28:08 | INFO | fairseq.trainer | begin training epoch 166
2023-06-13 18:28:15 | INFO | train_inner | epoch 166:      3 / 282 loss=3.112, nll_loss=1.201, glat_accu=0.515, glat_context_p=0.469, word_ins=2.995, length=3.027, ppl=8.65, wps=91368.7, ups=1.52, wpb=60266.2, bsz=2195.4, num_updates=46500, lr=0.000146647, gnorm=0.486, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 18:29:01 | INFO | train_inner | epoch 166:    103 / 282 loss=3.117, nll_loss=1.205, glat_accu=0.52, glat_context_p=0.469, word_ins=2.998, length=3.051, ppl=8.68, wps=132581, ups=2.19, wpb=60659.5, bsz=2150, num_updates=46600, lr=0.00014649, gnorm=0.499, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:29:48 | INFO | train_inner | epoch 166:    203 / 282 loss=3.125, nll_loss=1.213, glat_accu=0.528, glat_context_p=0.469, word_ins=3.006, length=3.044, ppl=8.72, wps=129590, ups=2.14, wpb=60427.8, bsz=2190, num_updates=46700, lr=0.000146333, gnorm=0.494, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:30:23 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:30:27 | INFO | valid | epoch 166 | valid on 'valid' subset | loss 12.443 | nll_loss 11.284 | word_ins 12.219 | length 4.48 | ppl 5569.16 | bleu 30.25 | wps 87867.4 | wpb 21176.3 | bsz 666.3 | num_updates 46779 | best_bleu 30.53
2023-06-13 18:30:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:30:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint166.pt (epoch 166 @ 46779 updates, score 30.25) (writing took 9.333011746406555 seconds)
2023-06-13 18:30:37 | INFO | fairseq_cli.train | end of epoch 166 (average epoch stats below)
2023-06-13 18:30:37 | INFO | train | epoch 166 | loss 3.119 | nll_loss 1.207 | glat_accu 0.522 | glat_context_p 0.469 | word_ins 3.001 | length 3.045 | ppl 8.69 | wps 114403 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 46779 | lr 0.000146209 | gnorm 0.496 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:30:37 | INFO | fairseq.trainer | begin training epoch 167
2023-06-13 18:30:52 | INFO | train_inner | epoch 167:     21 / 282 loss=3.116, nll_loss=1.204, glat_accu=0.517, glat_context_p=0.469, word_ins=2.998, length=3.037, ppl=8.67, wps=93727.9, ups=1.56, wpb=60205.8, bsz=2139, num_updates=46800, lr=0.000146176, gnorm=0.493, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:31:38 | INFO | train_inner | epoch 167:    121 / 282 loss=3.124, nll_loss=1.213, glat_accu=0.52, glat_context_p=0.469, word_ins=3.006, length=3.045, ppl=8.72, wps=131886, ups=2.18, wpb=60609.5, bsz=2150.1, num_updates=46900, lr=0.00014602, gnorm=0.503, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:32:24 | INFO | train_inner | epoch 167:    221 / 282 loss=3.121, nll_loss=1.21, glat_accu=0.526, glat_context_p=0.469, word_ins=3.003, length=3.034, ppl=8.7, wps=131869, ups=2.18, wpb=60555, bsz=2172.2, num_updates=47000, lr=0.000145865, gnorm=0.5, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:32:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:32:54 | INFO | valid | epoch 167 | valid on 'valid' subset | loss 12.467 | nll_loss 11.313 | word_ins 12.242 | length 4.508 | ppl 5662.63 | bleu 29.66 | wps 89314.5 | wpb 21176.3 | bsz 666.3 | num_updates 47061 | best_bleu 30.53
2023-06-13 18:32:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:33:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint167.pt (epoch 167 @ 47061 updates, score 29.66) (writing took 11.60847981646657 seconds)
2023-06-13 18:33:06 | INFO | fairseq_cli.train | end of epoch 167 (average epoch stats below)
2023-06-13 18:33:06 | INFO | train | epoch 167 | loss 3.12 | nll_loss 1.208 | glat_accu 0.521 | glat_context_p 0.469 | word_ins 3.002 | length 3.04 | ppl 8.69 | wps 114299 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 47061 | lr 0.00014577 | gnorm 0.499 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 18:33:06 | INFO | fairseq.trainer | begin training epoch 168
2023-06-13 18:33:30 | INFO | train_inner | epoch 168:     39 / 282 loss=3.108, nll_loss=1.195, glat_accu=0.508, glat_context_p=0.469, word_ins=2.99, length=3.045, ppl=8.62, wps=90898.4, ups=1.51, wpb=60075.7, bsz=2141.1, num_updates=47100, lr=0.00014571, gnorm=0.487, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:34:16 | INFO | train_inner | epoch 168:    139 / 282 loss=3.111, nll_loss=1.198, glat_accu=0.511, glat_context_p=0.469, word_ins=2.993, length=3.048, ppl=8.64, wps=130732, ups=2.16, wpb=60514.5, bsz=2137.2, num_updates=47200, lr=0.000145556, gnorm=0.494, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:35:02 | INFO | train_inner | epoch 168:    239 / 282 loss=3.109, nll_loss=1.197, glat_accu=0.519, glat_context_p=0.469, word_ins=2.991, length=3.02, ppl=8.63, wps=133362, ups=2.2, wpb=60747.8, bsz=2208.6, num_updates=47300, lr=0.000145402, gnorm=0.495, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:35:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:35:24 | INFO | valid | epoch 168 | valid on 'valid' subset | loss 12.425 | nll_loss 11.264 | word_ins 12.201 | length 4.471 | ppl 5497.74 | bleu 30.48 | wps 89012.9 | wpb 21176.3 | bsz 666.3 | num_updates 47343 | best_bleu 30.53
2023-06-13 18:35:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:35:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint168.pt (epoch 168 @ 47343 updates, score 30.48) (writing took 11.290269881486893 seconds)
2023-06-13 18:35:36 | INFO | fairseq_cli.train | end of epoch 168 (average epoch stats below)
2023-06-13 18:35:36 | INFO | train | epoch 168 | loss 3.11 | nll_loss 1.198 | glat_accu 0.514 | glat_context_p 0.469 | word_ins 2.992 | length 3.037 | ppl 8.64 | wps 113817 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 47343 | lr 0.000145336 | gnorm 0.498 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 18:35:36 | INFO | fairseq.trainer | begin training epoch 169
2023-06-13 18:36:08 | INFO | train_inner | epoch 169:     57 / 282 loss=3.126, nll_loss=1.214, glat_accu=0.524, glat_context_p=0.468, word_ins=3.007, length=3.051, ppl=8.73, wps=90783.2, ups=1.51, wpb=60003.7, bsz=2120.1, num_updates=47400, lr=0.000145248, gnorm=0.515, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:36:53 | INFO | train_inner | epoch 169:    157 / 282 loss=3.123, nll_loss=1.211, glat_accu=0.532, glat_context_p=0.468, word_ins=3.004, length=3.032, ppl=8.71, wps=132600, ups=2.19, wpb=60566.8, bsz=2172.7, num_updates=47500, lr=0.000145095, gnorm=0.497, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:37:35 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 18:37:40 | INFO | train_inner | epoch 169:    258 / 282 loss=3.123, nll_loss=1.211, glat_accu=0.527, glat_context_p=0.468, word_ins=3.004, length=3.035, ppl=8.71, wps=130019, ups=2.14, wpb=60629.3, bsz=2152.4, num_updates=47600, lr=0.000144943, gnorm=0.503, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:37:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:37:54 | INFO | valid | epoch 169 | valid on 'valid' subset | loss 12.581 | nll_loss 11.439 | word_ins 12.36 | length 4.441 | ppl 6127.97 | bleu 30.18 | wps 89157.3 | wpb 21176.3 | bsz 666.3 | num_updates 47624 | best_bleu 30.53
2023-06-13 18:37:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:38:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint169.pt (epoch 169 @ 47624 updates, score 30.18) (writing took 10.497435741126537 seconds)
2023-06-13 18:38:04 | INFO | fairseq_cli.train | end of epoch 169 (average epoch stats below)
2023-06-13 18:38:04 | INFO | train | epoch 169 | loss 3.124 | nll_loss 1.212 | glat_accu 0.528 | glat_context_p 0.468 | word_ins 3.005 | length 3.037 | ppl 8.72 | wps 114016 | ups 1.89 | wpb 60417.3 | bsz 2158 | num_updates 47624 | lr 0.000144906 | gnorm 0.503 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:38:05 | INFO | fairseq.trainer | begin training epoch 170
2023-06-13 18:38:46 | INFO | train_inner | epoch 170:     76 / 282 loss=3.111, nll_loss=1.198, glat_accu=0.507, glat_context_p=0.468, word_ins=2.993, length=3.055, ppl=8.64, wps=91139.5, ups=1.52, wpb=60071.2, bsz=2113.2, num_updates=47700, lr=0.000144791, gnorm=0.502, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:39:32 | INFO | train_inner | epoch 170:    176 / 282 loss=3.123, nll_loss=1.212, glat_accu=0.532, glat_context_p=0.468, word_ins=3.004, length=3.021, ppl=8.71, wps=131616, ups=2.17, wpb=60541.8, bsz=2178.5, num_updates=47800, lr=0.000144639, gnorm=0.496, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:40:18 | INFO | train_inner | epoch 170:    276 / 282 loss=3.121, nll_loss=1.209, glat_accu=0.532, glat_context_p=0.468, word_ins=3.002, length=3.038, ppl=8.7, wps=130717, ups=2.15, wpb=60676.2, bsz=2195.1, num_updates=47900, lr=0.000144488, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:40:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:40:24 | INFO | valid | epoch 170 | valid on 'valid' subset | loss 12.473 | nll_loss 11.317 | word_ins 12.254 | length 4.408 | ppl 5685.76 | bleu 30.38 | wps 88672.7 | wpb 21176.3 | bsz 666.3 | num_updates 47906 | best_bleu 30.53
2023-06-13 18:40:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:40:35 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint170.pt (epoch 170 @ 47906 updates, score 30.38) (writing took 10.845062673091888 seconds)
2023-06-13 18:40:35 | INFO | fairseq_cli.train | end of epoch 170 (average epoch stats below)
2023-06-13 18:40:35 | INFO | train | epoch 170 | loss 3.119 | nll_loss 1.207 | glat_accu 0.524 | glat_context_p 0.468 | word_ins 3 | length 3.041 | ppl 8.69 | wps 113232 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 47906 | lr 0.000144479 | gnorm 0.5 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:40:35 | INFO | fairseq.trainer | begin training epoch 171
2023-06-13 18:41:25 | INFO | train_inner | epoch 171:     94 / 282 loss=3.114, nll_loss=1.201, glat_accu=0.526, glat_context_p=0.468, word_ins=2.995, length=3.041, ppl=8.66, wps=90698.5, ups=1.51, wpb=60072.5, bsz=2159.4, num_updates=48000, lr=0.000144338, gnorm=0.506, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:42:11 | INFO | train_inner | epoch 171:    194 / 282 loss=3.126, nll_loss=1.214, glat_accu=0.531, glat_context_p=0.468, word_ins=3.006, length=3.041, ppl=8.73, wps=130070, ups=2.15, wpb=60562.2, bsz=2140.5, num_updates=48100, lr=0.000144187, gnorm=0.522, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:42:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:42:54 | INFO | valid | epoch 171 | valid on 'valid' subset | loss 12.433 | nll_loss 11.277 | word_ins 12.212 | length 4.422 | ppl 5530.52 | bleu 30.12 | wps 89933.7 | wpb 21176.3 | bsz 666.3 | num_updates 48188 | best_bleu 30.53
2023-06-13 18:42:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:43:03 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint171.pt (epoch 171 @ 48188 updates, score 30.12) (writing took 9.319382902234793 seconds)
2023-06-13 18:43:03 | INFO | fairseq_cli.train | end of epoch 171 (average epoch stats below)
2023-06-13 18:43:03 | INFO | train | epoch 171 | loss 3.123 | nll_loss 1.211 | glat_accu 0.532 | glat_context_p 0.468 | word_ins 3.003 | length 3.038 | ppl 8.71 | wps 114743 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 48188 | lr 0.000144056 | gnorm 0.516 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:43:03 | INFO | fairseq.trainer | begin training epoch 172
2023-06-13 18:43:15 | INFO | train_inner | epoch 172:     12 / 282 loss=3.128, nll_loss=1.217, glat_accu=0.538, glat_context_p=0.468, word_ins=3.009, length=3.029, ppl=8.74, wps=94084.9, ups=1.57, wpb=60108.8, bsz=2148.1, num_updates=48200, lr=0.000144038, gnorm=0.525, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:44:00 | INFO | train_inner | epoch 172:    112 / 282 loss=3.124, nll_loss=1.212, glat_accu=0.541, glat_context_p=0.468, word_ins=3.005, length=3.018, ppl=8.72, wps=133676, ups=2.21, wpb=60497.3, bsz=2199.7, num_updates=48300, lr=0.000143889, gnorm=0.505, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:44:46 | INFO | train_inner | epoch 172:    212 / 282 loss=3.122, nll_loss=1.21, glat_accu=0.531, glat_context_p=0.468, word_ins=3.003, length=3.047, ppl=8.71, wps=131877, ups=2.17, wpb=60665.5, bsz=2157.3, num_updates=48400, lr=0.00014374, gnorm=0.497, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:45:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:45:22 | INFO | valid | epoch 172 | valid on 'valid' subset | loss 12.203 | nll_loss 11.011 | word_ins 11.974 | length 4.581 | ppl 4715.26 | bleu 30.63 | wps 88265.9 | wpb 21176.3 | bsz 666.3 | num_updates 48470 | best_bleu 30.63
2023-06-13 18:45:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:45:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint172.pt (epoch 172 @ 48470 updates, score 30.63) (writing took 12.298755213618279 seconds)
2023-06-13 18:45:34 | INFO | fairseq_cli.train | end of epoch 172 (average epoch stats below)
2023-06-13 18:45:34 | INFO | train | epoch 172 | loss 3.124 | nll_loss 1.212 | glat_accu 0.533 | glat_context_p 0.468 | word_ins 3.005 | length 3.037 | ppl 8.72 | wps 112995 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 48470 | lr 0.000143636 | gnorm 0.504 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:45:34 | INFO | fairseq.trainer | begin training epoch 173
2023-06-13 18:45:55 | INFO | train_inner | epoch 173:     30 / 282 loss=3.126, nll_loss=1.214, glat_accu=0.526, glat_context_p=0.468, word_ins=3.006, length=3.059, ppl=8.73, wps=87427.1, ups=1.46, wpb=60062, bsz=2117.8, num_updates=48500, lr=0.000143592, gnorm=0.511, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:46:41 | INFO | train_inner | epoch 173:    130 / 282 loss=3.12, nll_loss=1.208, glat_accu=0.538, glat_context_p=0.468, word_ins=3.001, length=3.009, ppl=8.69, wps=131967, ups=2.17, wpb=60808.7, bsz=2161.5, num_updates=48600, lr=0.000143444, gnorm=0.507, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:46:47 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 18:47:27 | INFO | train_inner | epoch 173:    231 / 282 loss=3.119, nll_loss=1.206, glat_accu=0.531, glat_context_p=0.468, word_ins=3, length=3.044, ppl=8.69, wps=130581, ups=2.16, wpb=60574.5, bsz=2171.8, num_updates=48700, lr=0.000143296, gnorm=0.498, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:47:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:47:54 | INFO | valid | epoch 173 | valid on 'valid' subset | loss 12.294 | nll_loss 11.11 | word_ins 12.063 | length 4.593 | ppl 5020.22 | bleu 30.59 | wps 90054.6 | wpb 21176.3 | bsz 666.3 | num_updates 48751 | best_bleu 30.63
2023-06-13 18:47:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:48:03 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint173.pt (epoch 173 @ 48751 updates, score 30.59) (writing took 9.433004390448332 seconds)
2023-06-13 18:48:03 | INFO | fairseq_cli.train | end of epoch 173 (average epoch stats below)
2023-06-13 18:48:03 | INFO | train | epoch 173 | loss 3.121 | nll_loss 1.209 | glat_accu 0.534 | glat_context_p 0.468 | word_ins 3.002 | length 3.036 | ppl 8.7 | wps 113738 | ups 1.88 | wpb 60416.7 | bsz 2156.9 | num_updates 48751 | lr 0.000143222 | gnorm 0.504 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:48:04 | INFO | fairseq.trainer | begin training epoch 174
2023-06-13 18:48:32 | INFO | train_inner | epoch 174:     49 / 282 loss=3.114, nll_loss=1.201, glat_accu=0.529, glat_context_p=0.468, word_ins=2.995, length=3.035, ppl=8.66, wps=93370.2, ups=1.56, wpb=60023.1, bsz=2156.9, num_updates=48800, lr=0.00014315, gnorm=0.499, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:49:18 | INFO | train_inner | epoch 174:    149 / 282 loss=3.11, nll_loss=1.198, glat_accu=0.522, glat_context_p=0.467, word_ins=2.992, length=3.021, ppl=8.63, wps=131291, ups=2.16, wpb=60665.7, bsz=2156, num_updates=48900, lr=0.000143003, gnorm=0.498, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:50:04 | INFO | train_inner | epoch 174:    249 / 282 loss=3.111, nll_loss=1.199, glat_accu=0.531, glat_context_p=0.467, word_ins=2.992, length=3.028, ppl=8.64, wps=132360, ups=2.19, wpb=60499.1, bsz=2182.3, num_updates=49000, lr=0.000142857, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:50:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:50:22 | INFO | valid | epoch 174 | valid on 'valid' subset | loss 12.361 | nll_loss 11.186 | word_ins 12.133 | length 4.569 | ppl 5261.15 | bleu 30.24 | wps 87172.9 | wpb 21176.3 | bsz 666.3 | num_updates 49033 | best_bleu 30.63
2023-06-13 18:50:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:50:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint174.pt (epoch 174 @ 49033 updates, score 30.24) (writing took 10.523918021470308 seconds)
2023-06-13 18:50:32 | INFO | fairseq_cli.train | end of epoch 174 (average epoch stats below)
2023-06-13 18:50:32 | INFO | train | epoch 174 | loss 3.111 | nll_loss 1.198 | glat_accu 0.524 | glat_context_p 0.467 | word_ins 2.992 | length 3.03 | ppl 8.64 | wps 114496 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 49033 | lr 0.000142809 | gnorm 0.501 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:50:32 | INFO | fairseq.trainer | begin training epoch 175
2023-06-13 18:51:11 | INFO | train_inner | epoch 175:     67 / 282 loss=3.114, nll_loss=1.201, glat_accu=0.521, glat_context_p=0.467, word_ins=2.995, length=3.052, ppl=8.66, wps=89060.5, ups=1.49, wpb=59970.8, bsz=2107.4, num_updates=49100, lr=0.000142712, gnorm=0.504, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:51:57 | INFO | train_inner | epoch 175:    167 / 282 loss=3.113, nll_loss=1.201, glat_accu=0.53, glat_context_p=0.467, word_ins=2.994, length=3.032, ppl=8.65, wps=132129, ups=2.18, wpb=60663.3, bsz=2186.1, num_updates=49200, lr=0.000142566, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:52:43 | INFO | train_inner | epoch 175:    267 / 282 loss=3.113, nll_loss=1.201, glat_accu=0.524, glat_context_p=0.467, word_ins=2.995, length=3.025, ppl=8.65, wps=131162, ups=2.17, wpb=60573.9, bsz=2158, num_updates=49300, lr=0.000142422, gnorm=0.499, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:52:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:52:53 | INFO | valid | epoch 175 | valid on 'valid' subset | loss 12.27 | nll_loss 11.089 | word_ins 12.044 | length 4.515 | ppl 4939.87 | bleu 30.82 | wps 87493.2 | wpb 21176.3 | bsz 666.3 | num_updates 49315 | best_bleu 30.82
2023-06-13 18:52:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:53:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint175.pt (epoch 175 @ 49315 updates, score 30.82) (writing took 18.835460919886827 seconds)
2023-06-13 18:53:12 | INFO | fairseq_cli.train | end of epoch 175 (average epoch stats below)
2023-06-13 18:53:12 | INFO | train | epoch 175 | loss 3.112 | nll_loss 1.2 | glat_accu 0.527 | glat_context_p 0.467 | word_ins 2.994 | length 3.031 | ppl 8.65 | wps 106741 | ups 1.77 | wpb 60413.8 | bsz 2157.2 | num_updates 49315 | lr 0.0001424 | gnorm 0.503 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:53:12 | INFO | fairseq.trainer | begin training epoch 176
2023-06-13 18:53:58 | INFO | train_inner | epoch 176:     85 / 282 loss=3.118, nll_loss=1.205, glat_accu=0.529, glat_context_p=0.467, word_ins=2.999, length=3.05, ppl=8.68, wps=80418.1, ups=1.34, wpb=60152.2, bsz=2121.9, num_updates=49400, lr=0.000142278, gnorm=0.528, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 18:54:44 | INFO | train_inner | epoch 176:    185 / 282 loss=3.103, nll_loss=1.19, glat_accu=0.515, glat_context_p=0.467, word_ins=2.985, length=3.04, ppl=8.59, wps=130341, ups=2.15, wpb=60559.9, bsz=2117, num_updates=49500, lr=0.000142134, gnorm=0.505, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:55:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:55:32 | INFO | valid | epoch 176 | valid on 'valid' subset | loss 12.615 | nll_loss 11.479 | word_ins 12.39 | length 4.501 | ppl 6273.29 | bleu 29.83 | wps 89010.7 | wpb 21176.3 | bsz 666.3 | num_updates 49597 | best_bleu 30.82
2023-06-13 18:55:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:55:43 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint176.pt (epoch 176 @ 49597 updates, score 29.83) (writing took 10.848311021924019 seconds)
2023-06-13 18:55:43 | INFO | fairseq_cli.train | end of epoch 176 (average epoch stats below)
2023-06-13 18:55:43 | INFO | train | epoch 176 | loss 3.105 | nll_loss 1.192 | glat_accu 0.52 | glat_context_p 0.467 | word_ins 2.987 | length 3.033 | ppl 8.6 | wps 112877 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 49597 | lr 0.000141995 | gnorm 0.508 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:55:43 | INFO | fairseq.trainer | begin training epoch 177
2023-06-13 18:55:50 | INFO | train_inner | epoch 177:      3 / 282 loss=3.094, nll_loss=1.182, glat_accu=0.516, glat_context_p=0.467, word_ins=2.977, length=3.011, ppl=8.54, wps=91114.3, ups=1.52, wpb=60096.7, bsz=2208.8, num_updates=49600, lr=0.00014199, gnorm=0.498, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:56:08 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 18:56:36 | INFO | train_inner | epoch 177:    104 / 282 loss=3.101, nll_loss=1.189, glat_accu=0.522, glat_context_p=0.467, word_ins=2.984, length=3.014, ppl=8.58, wps=131110, ups=2.16, wpb=60630.7, bsz=2157.3, num_updates=49700, lr=0.000141848, gnorm=0.494, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:57:22 | INFO | train_inner | epoch 177:    204 / 282 loss=3.103, nll_loss=1.19, glat_accu=0.526, glat_context_p=0.467, word_ins=2.985, length=3.032, ppl=8.59, wps=131844, ups=2.18, wpb=60580.8, bsz=2198.7, num_updates=49800, lr=0.000141705, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:57:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 18:58:01 | INFO | valid | epoch 177 | valid on 'valid' subset | loss 12.499 | nll_loss 11.343 | word_ins 12.275 | length 4.496 | ppl 5787.08 | bleu 30.61 | wps 89296.6 | wpb 21176.3 | bsz 666.3 | num_updates 49878 | best_bleu 30.82
2023-06-13 18:58:01 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 18:58:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint177.pt (epoch 177 @ 49878 updates, score 30.61) (writing took 13.980998061597347 seconds)
2023-06-13 18:58:15 | INFO | fairseq_cli.train | end of epoch 177 (average epoch stats below)
2023-06-13 18:58:15 | INFO | train | epoch 177 | loss 3.106 | nll_loss 1.193 | glat_accu 0.524 | glat_context_p 0.467 | word_ins 2.987 | length 3.028 | ppl 8.61 | wps 111353 | ups 1.84 | wpb 60423.9 | bsz 2159.1 | num_updates 49878 | lr 0.000141594 | gnorm 0.504 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 18:58:15 | INFO | fairseq.trainer | begin training epoch 178
2023-06-13 18:58:32 | INFO | train_inner | epoch 178:     22 / 282 loss=3.115, nll_loss=1.203, glat_accu=0.525, glat_context_p=0.467, word_ins=2.997, length=3.041, ppl=8.67, wps=86951.7, ups=1.45, wpb=60134, bsz=2113.8, num_updates=49900, lr=0.000141563, gnorm=0.515, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 18:59:18 | INFO | train_inner | epoch 178:    122 / 282 loss=3.104, nll_loss=1.19, glat_accu=0.53, glat_context_p=0.467, word_ins=2.985, length=3.032, ppl=8.6, wps=129392, ups=2.14, wpb=60560.5, bsz=2179.1, num_updates=50000, lr=0.000141421, gnorm=0.502, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 19:00:04 | INFO | train_inner | epoch 178:    222 / 282 loss=3.106, nll_loss=1.194, glat_accu=0.53, glat_context_p=0.467, word_ins=2.988, length=2.997, ppl=8.61, wps=133435, ups=2.2, wpb=60714.8, bsz=2216.6, num_updates=50100, lr=0.00014128, gnorm=0.492, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:00:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:00:34 | INFO | valid | epoch 178 | valid on 'valid' subset | loss 12.465 | nll_loss 11.304 | word_ins 12.239 | length 4.514 | ppl 5653.98 | bleu 31 | wps 89118.6 | wpb 21176.3 | bsz 666.3 | num_updates 50160 | best_bleu 31
2023-06-13 19:00:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:00:52 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint178.pt (epoch 178 @ 50160 updates, score 31.0) (writing took 17.93034280464053 seconds)
2023-06-13 19:00:52 | INFO | fairseq_cli.train | end of epoch 178 (average epoch stats below)
2023-06-13 19:00:52 | INFO | train | epoch 178 | loss 3.107 | nll_loss 1.194 | glat_accu 0.526 | glat_context_p 0.467 | word_ins 2.989 | length 3.03 | ppl 8.62 | wps 108456 | ups 1.8 | wpb 60413.8 | bsz 2157.2 | num_updates 50160 | lr 0.000141196 | gnorm 0.501 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:00:53 | INFO | fairseq.trainer | begin training epoch 179
2023-06-13 19:01:17 | INFO | train_inner | epoch 179:     40 / 282 loss=3.111, nll_loss=1.198, glat_accu=0.527, glat_context_p=0.467, word_ins=2.992, length=3.047, ppl=8.64, wps=82331.7, ups=1.37, wpb=59942.6, bsz=2095.5, num_updates=50200, lr=0.000141139, gnorm=0.511, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:02:02 | INFO | train_inner | epoch 179:    140 / 282 loss=3.11, nll_loss=1.198, glat_accu=0.532, glat_context_p=0.467, word_ins=2.991, length=3.024, ppl=8.64, wps=132886, ups=2.19, wpb=60705.5, bsz=2170.1, num_updates=50300, lr=0.000140999, gnorm=0.51, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:02:49 | INFO | train_inner | epoch 179:    240 / 282 loss=3.092, nll_loss=1.18, glat_accu=0.492, glat_context_p=0.466, word_ins=2.976, length=3.047, ppl=8.53, wps=129620, ups=2.14, wpb=60457.4, bsz=2133.2, num_updates=50400, lr=0.000140859, gnorm=0.482, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:03:08 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:03:11 | INFO | valid | epoch 179 | valid on 'valid' subset | loss 12.535 | nll_loss 11.389 | word_ins 12.309 | length 4.511 | ppl 5935.39 | bleu 29.88 | wps 88891.9 | wpb 21176.3 | bsz 666.3 | num_updates 50442 | best_bleu 31
2023-06-13 19:03:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:03:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint179.pt (epoch 179 @ 50442 updates, score 29.88) (writing took 8.719514943659306 seconds)
2023-06-13 19:03:20 | INFO | fairseq_cli.train | end of epoch 179 (average epoch stats below)
2023-06-13 19:03:20 | INFO | train | epoch 179 | loss 3.102 | nll_loss 1.189 | glat_accu 0.516 | glat_context_p 0.466 | word_ins 2.984 | length 3.028 | ppl 8.58 | wps 115678 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 50442 | lr 0.0001408 | gnorm 0.499 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:03:20 | INFO | fairseq.trainer | begin training epoch 180
2023-06-13 19:03:52 | INFO | train_inner | epoch 180:     58 / 282 loss=3.091, nll_loss=1.178, glat_accu=0.498, glat_context_p=0.466, word_ins=2.974, length=3.036, ppl=8.52, wps=96043.9, ups=1.6, wpb=60190.6, bsz=2103.3, num_updates=50500, lr=0.00014072, gnorm=0.502, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:04:38 | INFO | train_inner | epoch 180:    158 / 282 loss=3.105, nll_loss=1.192, glat_accu=0.531, glat_context_p=0.466, word_ins=2.986, length=3.033, ppl=8.61, wps=132303, ups=2.18, wpb=60623.4, bsz=2158.6, num_updates=50600, lr=0.00014058, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:05:06 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 19:05:24 | INFO | train_inner | epoch 180:    259 / 282 loss=3.096, nll_loss=1.183, glat_accu=0.52, glat_context_p=0.466, word_ins=2.978, length=3.021, ppl=8.55, wps=130696, ups=2.16, wpb=60539.8, bsz=2223.9, num_updates=50700, lr=0.000140442, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:05:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:05:37 | INFO | valid | epoch 180 | valid on 'valid' subset | loss 12.481 | nll_loss 11.329 | word_ins 12.255 | length 4.517 | ppl 5716.75 | bleu 30.56 | wps 88875 | wpb 21176.3 | bsz 666.3 | num_updates 50723 | best_bleu 31
2023-06-13 19:05:37 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:05:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint180.pt (epoch 180 @ 50723 updates, score 30.56) (writing took 10.742833476513624 seconds)
2023-06-13 19:05:48 | INFO | fairseq_cli.train | end of epoch 180 (average epoch stats below)
2023-06-13 19:05:48 | INFO | train | epoch 180 | loss 3.097 | nll_loss 1.184 | glat_accu 0.516 | glat_context_p 0.466 | word_ins 2.979 | length 3.033 | ppl 8.56 | wps 114221 | ups 1.89 | wpb 60409.9 | bsz 2157.6 | num_updates 50723 | lr 0.00014041 | gnorm 0.501 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:05:48 | INFO | fairseq.trainer | begin training epoch 181
2023-06-13 19:06:30 | INFO | train_inner | epoch 181:     77 / 282 loss=3.095, nll_loss=1.181, glat_accu=0.521, glat_context_p=0.466, word_ins=2.977, length=3.042, ppl=8.55, wps=90632.2, ups=1.52, wpb=59781.9, bsz=2157.1, num_updates=50800, lr=0.000140303, gnorm=0.494, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:07:16 | INFO | train_inner | epoch 181:    177 / 282 loss=3.112, nll_loss=1.2, glat_accu=0.53, glat_context_p=0.466, word_ins=2.993, length=3.022, ppl=8.65, wps=132494, ups=2.18, wpb=60685, bsz=2145.8, num_updates=50900, lr=0.000140165, gnorm=0.515, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:08:01 | INFO | train_inner | epoch 181:    277 / 282 loss=3.121, nll_loss=1.209, glat_accu=0.541, glat_context_p=0.466, word_ins=3.001, length=3.028, ppl=8.7, wps=132700, ups=2.18, wpb=60732.6, bsz=2176, num_updates=51000, lr=0.000140028, gnorm=0.52, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:08:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:08:07 | INFO | valid | epoch 181 | valid on 'valid' subset | loss 12.42 | nll_loss 11.257 | word_ins 12.197 | length 4.46 | ppl 5480.96 | bleu 30.73 | wps 89138.7 | wpb 21176.3 | bsz 666.3 | num_updates 51005 | best_bleu 31
2023-06-13 19:08:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:08:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint181.pt (epoch 181 @ 51005 updates, score 30.73) (writing took 8.307707380503416 seconds)
2023-06-13 19:08:15 | INFO | fairseq_cli.train | end of epoch 181 (average epoch stats below)
2023-06-13 19:08:15 | INFO | train | epoch 181 | loss 3.111 | nll_loss 1.198 | glat_accu 0.533 | glat_context_p 0.466 | word_ins 2.992 | length 3.029 | ppl 8.64 | wps 115702 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 51005 | lr 0.000140021 | gnorm 0.513 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:08:16 | INFO | fairseq.trainer | begin training epoch 182
2023-06-13 19:09:05 | INFO | train_inner | epoch 182:     95 / 282 loss=3.106, nll_loss=1.193, glat_accu=0.537, glat_context_p=0.466, word_ins=2.987, length=3.02, ppl=8.61, wps=95207.5, ups=1.58, wpb=60094.4, bsz=2137, num_updates=51100, lr=0.000139891, gnorm=0.512, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:09:50 | INFO | train_inner | epoch 182:    195 / 282 loss=3.109, nll_loss=1.196, glat_accu=0.537, glat_context_p=0.466, word_ins=2.99, length=3.024, ppl=8.63, wps=131968, ups=2.18, wpb=60471.4, bsz=2184.2, num_updates=51200, lr=0.000139754, gnorm=0.513, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:10:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:10:33 | INFO | valid | epoch 182 | valid on 'valid' subset | loss 12.382 | nll_loss 11.211 | word_ins 12.154 | length 4.56 | ppl 5336.74 | bleu 31.01 | wps 87639.5 | wpb 21176.3 | bsz 666.3 | num_updates 51287 | best_bleu 31.01
2023-06-13 19:10:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:10:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint182.pt (epoch 182 @ 51287 updates, score 31.01) (writing took 14.734554391354322 seconds)
2023-06-13 19:10:48 | INFO | fairseq_cli.train | end of epoch 182 (average epoch stats below)
2023-06-13 19:10:48 | INFO | train | epoch 182 | loss 3.111 | nll_loss 1.198 | glat_accu 0.536 | glat_context_p 0.466 | word_ins 2.991 | length 3.026 | ppl 8.64 | wps 111682 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 51287 | lr 0.000139636 | gnorm 0.512 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:10:48 | INFO | fairseq.trainer | begin training epoch 183
2023-06-13 19:11:00 | INFO | train_inner | epoch 183:     13 / 282 loss=3.115, nll_loss=1.203, glat_accu=0.532, glat_context_p=0.466, word_ins=2.996, length=3.032, ppl=8.67, wps=86372.2, ups=1.43, wpb=60215.6, bsz=2126.1, num_updates=51300, lr=0.000139618, gnorm=0.514, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:11:46 | INFO | train_inner | epoch 183:    113 / 282 loss=3.103, nll_loss=1.19, glat_accu=0.524, glat_context_p=0.466, word_ins=2.984, length=3.038, ppl=8.59, wps=131438, ups=2.17, wpb=60662.7, bsz=2139.8, num_updates=51400, lr=0.000139482, gnorm=0.5, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:12:32 | INFO | train_inner | epoch 183:    213 / 282 loss=3.116, nll_loss=1.204, glat_accu=0.54, glat_context_p=0.466, word_ins=2.997, length=3.024, ppl=8.67, wps=133364, ups=2.2, wpb=60538, bsz=2158.8, num_updates=51500, lr=0.000139347, gnorm=0.513, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:13:03 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:13:06 | INFO | valid | epoch 183 | valid on 'valid' subset | loss 12.395 | nll_loss 11.232 | word_ins 12.169 | length 4.523 | ppl 5385.91 | bleu 30.51 | wps 88476.6 | wpb 21176.3 | bsz 666.3 | num_updates 51569 | best_bleu 31.01
2023-06-13 19:13:06 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:13:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint183.pt (epoch 183 @ 51569 updates, score 30.51) (writing took 12.9590365588665 seconds)
2023-06-13 19:13:19 | INFO | fairseq_cli.train | end of epoch 183 (average epoch stats below)
2023-06-13 19:13:19 | INFO | train | epoch 183 | loss 3.109 | nll_loss 1.196 | glat_accu 0.533 | glat_context_p 0.466 | word_ins 2.99 | length 3.025 | ppl 8.63 | wps 112744 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 51569 | lr 0.000139253 | gnorm 0.513 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:13:19 | INFO | fairseq.trainer | begin training epoch 184
2023-06-13 19:13:39 | INFO | train_inner | epoch 184:     31 / 282 loss=3.105, nll_loss=1.192, glat_accu=0.53, glat_context_p=0.466, word_ins=2.986, length=3.015, ppl=8.6, wps=88744.5, ups=1.48, wpb=60066.1, bsz=2179.8, num_updates=51600, lr=0.000139212, gnorm=0.532, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:14:19 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 19:14:26 | INFO | train_inner | epoch 184:    132 / 282 loss=3.099, nll_loss=1.186, glat_accu=0.521, glat_context_p=0.466, word_ins=2.981, length=3.021, ppl=8.57, wps=130596, ups=2.15, wpb=60650.4, bsz=2162.5, num_updates=51700, lr=0.000139077, gnorm=0.501, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:15:12 | INFO | train_inner | epoch 184:    232 / 282 loss=3.096, nll_loss=1.183, glat_accu=0.526, glat_context_p=0.466, word_ins=2.978, length=3.019, ppl=8.55, wps=131102, ups=2.17, wpb=60452.7, bsz=2174, num_updates=51800, lr=0.000138943, gnorm=0.493, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:15:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:15:38 | INFO | valid | epoch 184 | valid on 'valid' subset | loss 12.378 | nll_loss 11.207 | word_ins 12.153 | length 4.482 | ppl 5323.17 | bleu 30.65 | wps 85826.4 | wpb 21176.3 | bsz 666.3 | num_updates 51850 | best_bleu 31.01
2023-06-13 19:15:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:15:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint184.pt (epoch 184 @ 51850 updates, score 30.65) (writing took 11.050988469272852 seconds)
2023-06-13 19:15:49 | INFO | fairseq_cli.train | end of epoch 184 (average epoch stats below)
2023-06-13 19:15:49 | INFO | train | epoch 184 | loss 3.101 | nll_loss 1.188 | glat_accu 0.524 | glat_context_p 0.466 | word_ins 2.982 | length 3.023 | ppl 8.58 | wps 113387 | ups 1.88 | wpb 60410 | bsz 2154.8 | num_updates 51850 | lr 0.000138875 | gnorm 0.504 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:15:49 | INFO | fairseq.trainer | begin training epoch 185
2023-06-13 19:16:18 | INFO | train_inner | epoch 185:     50 / 282 loss=3.112, nll_loss=1.199, glat_accu=0.535, glat_context_p=0.465, word_ins=2.992, length=3.039, ppl=8.64, wps=91134, ups=1.52, wpb=60151.8, bsz=2137.8, num_updates=51900, lr=0.000138809, gnorm=0.514, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:17:04 | INFO | train_inner | epoch 185:    150 / 282 loss=3.116, nll_loss=1.202, glat_accu=0.538, glat_context_p=0.465, word_ins=2.995, length=3.04, ppl=8.67, wps=130916, ups=2.16, wpb=60539.3, bsz=2111.8, num_updates=52000, lr=0.000138675, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:17:50 | INFO | train_inner | epoch 185:    250 / 282 loss=3.114, nll_loss=1.201, glat_accu=0.54, glat_context_p=0.465, word_ins=2.994, length=3.025, ppl=8.66, wps=130877, ups=2.16, wpb=60568.9, bsz=2179.6, num_updates=52100, lr=0.000138542, gnorm=0.514, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:18:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:18:08 | INFO | valid | epoch 185 | valid on 'valid' subset | loss 12.322 | nll_loss 11.151 | word_ins 12.098 | length 4.485 | ppl 5120.98 | bleu 30.75 | wps 87546.2 | wpb 21176.3 | bsz 666.3 | num_updates 52132 | best_bleu 31.01
2023-06-13 19:18:08 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:18:18 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint185.pt (epoch 185 @ 52132 updates, score 30.75) (writing took 9.856984414160252 seconds)
2023-06-13 19:18:18 | INFO | fairseq_cli.train | end of epoch 185 (average epoch stats below)
2023-06-13 19:18:18 | INFO | train | epoch 185 | loss 3.114 | nll_loss 1.201 | glat_accu 0.54 | glat_context_p 0.465 | word_ins 2.994 | length 3.029 | ppl 8.66 | wps 114455 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 52132 | lr 0.000138499 | gnorm 0.515 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:18:18 | INFO | fairseq.trainer | begin training epoch 186
2023-06-13 19:18:55 | INFO | train_inner | epoch 186:     68 / 282 loss=3.111, nll_loss=1.198, glat_accu=0.54, glat_context_p=0.465, word_ins=2.992, length=3.019, ppl=8.64, wps=93238.6, ups=1.55, wpb=60154.2, bsz=2158.4, num_updates=52200, lr=0.000138409, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:19:41 | INFO | train_inner | epoch 186:    168 / 282 loss=3.105, nll_loss=1.192, glat_accu=0.542, glat_context_p=0.465, word_ins=2.986, length=2.997, ppl=8.6, wps=132341, ups=2.18, wpb=60713.7, bsz=2184.8, num_updates=52300, lr=0.000138277, gnorm=0.512, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:20:26 | INFO | train_inner | epoch 186:    268 / 282 loss=3.108, nll_loss=1.195, glat_accu=0.528, glat_context_p=0.465, word_ins=2.989, length=3.041, ppl=8.62, wps=132794, ups=2.2, wpb=60496, bsz=2142.6, num_updates=52400, lr=0.000138145, gnorm=0.532, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:20:33 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:20:36 | INFO | valid | epoch 186 | valid on 'valid' subset | loss 12.341 | nll_loss 11.16 | word_ins 12.111 | length 4.605 | ppl 5188.39 | bleu 30.64 | wps 88534.6 | wpb 21176.3 | bsz 666.3 | num_updates 52414 | best_bleu 31.01
2023-06-13 19:20:36 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:20:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint186.pt (epoch 186 @ 52414 updates, score 30.64) (writing took 12.745520025491714 seconds)
2023-06-13 19:20:49 | INFO | fairseq_cli.train | end of epoch 186 (average epoch stats below)
2023-06-13 19:20:49 | INFO | train | epoch 186 | loss 3.108 | nll_loss 1.195 | glat_accu 0.536 | glat_context_p 0.465 | word_ins 2.989 | length 3.021 | ppl 8.62 | wps 112944 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 52414 | lr 0.000138126 | gnorm 0.521 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:20:49 | INFO | fairseq.trainer | begin training epoch 187
2023-06-13 19:21:34 | INFO | train_inner | epoch 187:     86 / 282 loss=3.11, nll_loss=1.196, glat_accu=0.538, glat_context_p=0.465, word_ins=2.99, length=3.025, ppl=8.63, wps=89219.2, ups=1.49, wpb=60045.1, bsz=2149, num_updates=52500, lr=0.000138013, gnorm=0.546, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:22:20 | INFO | train_inner | epoch 187:    186 / 282 loss=3.112, nll_loss=1.2, glat_accu=0.529, glat_context_p=0.465, word_ins=2.993, length=3.025, ppl=8.65, wps=131706, ups=2.17, wpb=60638.6, bsz=2160.2, num_updates=52600, lr=0.000137882, gnorm=0.514, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:23:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:23:07 | INFO | valid | epoch 187 | valid on 'valid' subset | loss 12.381 | nll_loss 11.211 | word_ins 12.148 | length 4.641 | ppl 5334.93 | bleu 31.08 | wps 87894.6 | wpb 21176.3 | bsz 666.3 | num_updates 52696 | best_bleu 31.08
2023-06-13 19:23:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:23:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint187.pt (epoch 187 @ 52696 updates, score 31.08) (writing took 18.65853478386998 seconds)
2023-06-13 19:23:26 | INFO | fairseq_cli.train | end of epoch 187 (average epoch stats below)
2023-06-13 19:23:26 | INFO | train | epoch 187 | loss 3.113 | nll_loss 1.201 | glat_accu 0.535 | glat_context_p 0.465 | word_ins 2.994 | length 3.026 | ppl 8.65 | wps 108181 | ups 1.79 | wpb 60413.8 | bsz 2157.2 | num_updates 52696 | lr 0.000137756 | gnorm 0.527 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:23:26 | INFO | fairseq.trainer | begin training epoch 188
2023-06-13 19:23:34 | INFO | train_inner | epoch 188:      4 / 282 loss=3.117, nll_loss=1.205, glat_accu=0.54, glat_context_p=0.465, word_ins=2.997, length=3.026, ppl=8.68, wps=80810.2, ups=1.34, wpb=60097.2, bsz=2164, num_updates=52700, lr=0.000137751, gnorm=0.522, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:23:38 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 19:24:20 | INFO | train_inner | epoch 188:    105 / 282 loss=3.107, nll_loss=1.193, glat_accu=0.536, glat_context_p=0.465, word_ins=2.987, length=3.018, ppl=8.61, wps=131411, ups=2.17, wpb=60671.6, bsz=2181.3, num_updates=52800, lr=0.00013762, gnorm=0.514, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:25:06 | INFO | train_inner | epoch 188:    205 / 282 loss=3.11, nll_loss=1.197, glat_accu=0.542, glat_context_p=0.465, word_ins=2.99, length=3.023, ppl=8.63, wps=131783, ups=2.18, wpb=60500.4, bsz=2182, num_updates=52900, lr=0.00013749, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:25:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:25:45 | INFO | valid | epoch 188 | valid on 'valid' subset | loss 12.392 | nll_loss 11.212 | word_ins 12.158 | length 4.665 | ppl 5373.12 | bleu 30.67 | wps 87220.9 | wpb 21176.3 | bsz 666.3 | num_updates 52977 | best_bleu 31.08
2023-06-13 19:25:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:25:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint188.pt (epoch 188 @ 52977 updates, score 30.67) (writing took 13.003574840724468 seconds)
2023-06-13 19:25:58 | INFO | fairseq_cli.train | end of epoch 188 (average epoch stats below)
2023-06-13 19:25:58 | INFO | train | epoch 188 | loss 3.11 | nll_loss 1.197 | glat_accu 0.537 | glat_context_p 0.465 | word_ins 2.99 | length 3.027 | ppl 8.63 | wps 112058 | ups 1.85 | wpb 60411.6 | bsz 2154.4 | num_updates 52977 | lr 0.00013739 | gnorm 0.514 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:25:58 | INFO | fairseq.trainer | begin training epoch 189
2023-06-13 19:26:14 | INFO | train_inner | epoch 189:     23 / 282 loss=3.111, nll_loss=1.197, glat_accu=0.534, glat_context_p=0.465, word_ins=2.991, length=3.048, ppl=8.64, wps=88455, ups=1.47, wpb=60095.4, bsz=2114.2, num_updates=53000, lr=0.000137361, gnorm=0.514, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:27:00 | INFO | train_inner | epoch 189:    123 / 282 loss=3.103, nll_loss=1.19, glat_accu=0.536, glat_context_p=0.465, word_ins=2.984, length=3.01, ppl=8.59, wps=132787, ups=2.19, wpb=60630.4, bsz=2164.2, num_updates=53100, lr=0.000137231, gnorm=0.511, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:27:46 | INFO | train_inner | epoch 189:    223 / 282 loss=3.11, nll_loss=1.197, glat_accu=0.542, glat_context_p=0.465, word_ins=2.99, length=3.031, ppl=8.64, wps=131135, ups=2.17, wpb=60493.4, bsz=2135.2, num_updates=53200, lr=0.000137102, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:28:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:28:16 | INFO | valid | epoch 189 | valid on 'valid' subset | loss 12.337 | nll_loss 11.172 | word_ins 12.116 | length 4.437 | ppl 5174.31 | bleu 30.79 | wps 89081.4 | wpb 21176.3 | bsz 666.3 | num_updates 53259 | best_bleu 31.08
2023-06-13 19:28:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:28:29 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint189.pt (epoch 189 @ 53259 updates, score 30.79) (writing took 13.575789235532284 seconds)
2023-06-13 19:28:29 | INFO | fairseq_cli.train | end of epoch 189 (average epoch stats below)
2023-06-13 19:28:29 | INFO | train | epoch 189 | loss 3.106 | nll_loss 1.193 | glat_accu 0.54 | glat_context_p 0.465 | word_ins 2.987 | length 3.026 | ppl 8.61 | wps 112223 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 53259 | lr 0.000137026 | gnorm 0.516 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:28:29 | INFO | fairseq.trainer | begin training epoch 190
2023-06-13 19:28:54 | INFO | train_inner | epoch 190:     41 / 282 loss=3.104, nll_loss=1.19, glat_accu=0.544, glat_context_p=0.465, word_ins=2.984, length=3.005, ppl=8.6, wps=88195.3, ups=1.46, wpb=60313.5, bsz=2182.7, num_updates=53300, lr=0.000136973, gnorm=0.52, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:29:40 | INFO | train_inner | epoch 190:    141 / 282 loss=3.108, nll_loss=1.194, glat_accu=0.547, glat_context_p=0.464, word_ins=2.988, length=3.016, ppl=8.62, wps=131638, ups=2.17, wpb=60616.5, bsz=2184.3, num_updates=53400, lr=0.000136845, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:30:26 | INFO | train_inner | epoch 190:    241 / 282 loss=3.112, nll_loss=1.199, glat_accu=0.537, glat_context_p=0.464, word_ins=2.992, length=3.046, ppl=8.65, wps=130936, ups=2.17, wpb=60327.2, bsz=2111.5, num_updates=53500, lr=0.000136717, gnorm=0.527, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:30:45 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:30:48 | INFO | valid | epoch 190 | valid on 'valid' subset | loss 12.208 | nll_loss 11.021 | word_ins 11.976 | length 4.661 | ppl 4731.31 | bleu 30.92 | wps 81936.4 | wpb 21176.3 | bsz 666.3 | num_updates 53541 | best_bleu 31.08
2023-06-13 19:30:48 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:31:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint190.pt (epoch 190 @ 53541 updates, score 30.92) (writing took 11.521532714366913 seconds)
2023-06-13 19:31:00 | INFO | fairseq_cli.train | end of epoch 190 (average epoch stats below)
2023-06-13 19:31:00 | INFO | train | epoch 190 | loss 3.108 | nll_loss 1.195 | glat_accu 0.543 | glat_context_p 0.464 | word_ins 2.988 | length 3.019 | ppl 8.62 | wps 113212 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 53541 | lr 0.000136665 | gnorm 0.521 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:31:00 | INFO | fairseq.trainer | begin training epoch 191
2023-06-13 19:31:33 | INFO | train_inner | epoch 191:     59 / 282 loss=3.104, nll_loss=1.191, glat_accu=0.537, glat_context_p=0.464, word_ins=2.985, length=3.016, ppl=8.6, wps=90320.4, ups=1.5, wpb=60219.4, bsz=2161.9, num_updates=53600, lr=0.00013659, gnorm=0.519, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:32:19 | INFO | train_inner | epoch 191:    159 / 282 loss=3.096, nll_loss=1.182, glat_accu=0.527, glat_context_p=0.464, word_ins=2.977, length=3.027, ppl=8.55, wps=132423, ups=2.19, wpb=60449.8, bsz=2171.2, num_updates=53700, lr=0.000136462, gnorm=0.515, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:32:34 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 19:33:05 | INFO | train_inner | epoch 191:    260 / 282 loss=3.1, nll_loss=1.188, glat_accu=0.529, glat_context_p=0.464, word_ins=2.982, length=3.009, ppl=8.57, wps=131020, ups=2.16, wpb=60700.3, bsz=2150.8, num_updates=53800, lr=0.000136335, gnorm=0.499, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:33:15 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:33:18 | INFO | valid | epoch 191 | valid on 'valid' subset | loss 12.268 | nll_loss 11.095 | word_ins 12.047 | length 4.46 | ppl 4933.17 | bleu 31.08 | wps 88529.3 | wpb 21176.3 | bsz 666.3 | num_updates 53822 | best_bleu 31.08
2023-06-13 19:33:18 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:33:33 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint191.pt (epoch 191 @ 53822 updates, score 31.08) (writing took 14.54585212096572 seconds)
2023-06-13 19:33:33 | INFO | fairseq_cli.train | end of epoch 191 (average epoch stats below)
2023-06-13 19:33:33 | INFO | train | epoch 191 | loss 3.099 | nll_loss 1.186 | glat_accu 0.529 | glat_context_p 0.464 | word_ins 2.98 | length 3.018 | ppl 8.57 | wps 110959 | ups 1.84 | wpb 60424.7 | bsz 2155.8 | num_updates 53822 | lr 0.000136308 | gnorm 0.51 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:33:33 | INFO | fairseq.trainer | begin training epoch 192
2023-06-13 19:34:15 | INFO | train_inner | epoch 192:     78 / 282 loss=3.101, nll_loss=1.188, glat_accu=0.531, glat_context_p=0.464, word_ins=2.982, length=3.023, ppl=8.58, wps=86231.6, ups=1.44, wpb=59939.7, bsz=2140, num_updates=53900, lr=0.000136209, gnorm=0.517, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:35:00 | INFO | train_inner | epoch 192:    178 / 282 loss=3.107, nll_loss=1.195, glat_accu=0.543, glat_context_p=0.464, word_ins=2.988, length=3, ppl=8.61, wps=132831, ups=2.19, wpb=60703.7, bsz=2209.9, num_updates=54000, lr=0.000136083, gnorm=0.515, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:35:46 | INFO | train_inner | epoch 192:    278 / 282 loss=3.108, nll_loss=1.196, glat_accu=0.531, glat_context_p=0.464, word_ins=2.989, length=3.026, ppl=8.62, wps=132185, ups=2.18, wpb=60655.2, bsz=2117.4, num_updates=54100, lr=0.000135957, gnorm=0.507, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:35:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:35:51 | INFO | valid | epoch 192 | valid on 'valid' subset | loss 12.371 | nll_loss 11.195 | word_ins 12.141 | length 4.576 | ppl 5297.91 | bleu 30.88 | wps 88517.3 | wpb 21176.3 | bsz 666.3 | num_updates 54104 | best_bleu 31.08
2023-06-13 19:35:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:36:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint192.pt (epoch 192 @ 54104 updates, score 30.88) (writing took 11.03962242975831 seconds)
2023-06-13 19:36:02 | INFO | fairseq_cli.train | end of epoch 192 (average epoch stats below)
2023-06-13 19:36:02 | INFO | train | epoch 192 | loss 3.106 | nll_loss 1.193 | glat_accu 0.536 | glat_context_p 0.464 | word_ins 2.987 | length 3.016 | ppl 8.61 | wps 114218 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 54104 | lr 0.000135952 | gnorm 0.514 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:36:02 | INFO | fairseq.trainer | begin training epoch 193
2023-06-13 19:36:52 | INFO | train_inner | epoch 193:     96 / 282 loss=3.094, nll_loss=1.179, glat_accu=0.528, glat_context_p=0.464, word_ins=2.975, length=3.029, ppl=8.54, wps=90651.4, ups=1.51, wpb=59878.6, bsz=2112.6, num_updates=54200, lr=0.000135831, gnorm=0.517, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:37:38 | INFO | train_inner | epoch 193:    196 / 282 loss=3.092, nll_loss=1.179, glat_accu=0.521, glat_context_p=0.464, word_ins=2.974, length=3.014, ppl=8.53, wps=132814, ups=2.19, wpb=60733.3, bsz=2136.5, num_updates=54300, lr=0.000135706, gnorm=0.502, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:38:17 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:38:20 | INFO | valid | epoch 193 | valid on 'valid' subset | loss 12.328 | nll_loss 11.154 | word_ins 12.103 | length 4.516 | ppl 5142.77 | bleu 30.59 | wps 87340 | wpb 21176.3 | bsz 666.3 | num_updates 54386 | best_bleu 31.08
2023-06-13 19:38:20 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:38:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint193.pt (epoch 193 @ 54386 updates, score 30.59) (writing took 11.747901298105717 seconds)
2023-06-13 19:38:32 | INFO | fairseq_cli.train | end of epoch 193 (average epoch stats below)
2023-06-13 19:38:32 | INFO | train | epoch 193 | loss 3.091 | nll_loss 1.177 | glat_accu 0.524 | glat_context_p 0.464 | word_ins 2.972 | length 3.016 | ppl 8.52 | wps 113932 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 54386 | lr 0.000135599 | gnorm 0.504 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:38:32 | INFO | fairseq.trainer | begin training epoch 194
2023-06-13 19:38:45 | INFO | train_inner | epoch 194:     14 / 282 loss=3.088, nll_loss=1.174, glat_accu=0.523, glat_context_p=0.464, word_ins=2.97, length=3.02, ppl=8.5, wps=89635.8, ups=1.49, wpb=60102.9, bsz=2188.8, num_updates=54400, lr=0.000135582, gnorm=0.497, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:39:30 | INFO | train_inner | epoch 194:    114 / 282 loss=3.08, nll_loss=1.166, glat_accu=0.521, glat_context_p=0.464, word_ins=2.962, length=3.001, ppl=8.46, wps=133184, ups=2.2, wpb=60556.7, bsz=2171.5, num_updates=54500, lr=0.000135457, gnorm=0.496, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:40:16 | INFO | train_inner | epoch 194:    214 / 282 loss=3.094, nll_loss=1.181, glat_accu=0.53, glat_context_p=0.464, word_ins=2.975, length=3.018, ppl=8.54, wps=132575, ups=2.19, wpb=60587.5, bsz=2184.7, num_updates=54600, lr=0.000135333, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:40:47 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:40:51 | INFO | valid | epoch 194 | valid on 'valid' subset | loss 12.381 | nll_loss 11.217 | word_ins 12.157 | length 4.472 | ppl 5334.03 | bleu 30.68 | wps 86714.5 | wpb 21176.3 | bsz 666.3 | num_updates 54668 | best_bleu 31.08
2023-06-13 19:40:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:41:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint194.pt (epoch 194 @ 54668 updates, score 30.68) (writing took 13.419524978846312 seconds)
2023-06-13 19:41:04 | INFO | fairseq_cli.train | end of epoch 194 (average epoch stats below)
2023-06-13 19:41:04 | INFO | train | epoch 194 | loss 3.089 | nll_loss 1.175 | glat_accu 0.524 | glat_context_p 0.464 | word_ins 2.971 | length 3.016 | ppl 8.51 | wps 111720 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 54668 | lr 0.000135249 | gnorm 0.51 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 19:41:04 | INFO | fairseq.trainer | begin training epoch 195
2023-06-13 19:41:25 | INFO | train_inner | epoch 195:     32 / 282 loss=3.097, nll_loss=1.184, glat_accu=0.521, glat_context_p=0.464, word_ins=2.979, length=3.026, ppl=8.56, wps=87735.4, ups=1.46, wpb=60143.4, bsz=2115.6, num_updates=54700, lr=0.000135209, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:41:51 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 19:42:11 | INFO | train_inner | epoch 195:    133 / 282 loss=3.102, nll_loss=1.188, glat_accu=0.533, glat_context_p=0.464, word_ins=2.982, length=3.027, ppl=8.58, wps=129589, ups=2.14, wpb=60513.2, bsz=2125.1, num_updates=54800, lr=0.000135086, gnorm=0.508, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 19:42:57 | INFO | train_inner | epoch 195:    233 / 282 loss=3.099, nll_loss=1.186, glat_accu=0.539, glat_context_p=0.463, word_ins=2.98, length=3.006, ppl=8.57, wps=132975, ups=2.19, wpb=60709.7, bsz=2201.9, num_updates=54900, lr=0.000134963, gnorm=0.516, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:43:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:43:23 | INFO | valid | epoch 195 | valid on 'valid' subset | loss 12.282 | nll_loss 11.109 | word_ins 12.061 | length 4.441 | ppl 4979.46 | bleu 31.06 | wps 88808.1 | wpb 21176.3 | bsz 666.3 | num_updates 54949 | best_bleu 31.08
2023-06-13 19:43:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:43:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint195.pt (epoch 195 @ 54949 updates, score 31.06) (writing took 13.621829513460398 seconds)
2023-06-13 19:43:36 | INFO | fairseq_cli.train | end of epoch 195 (average epoch stats below)
2023-06-13 19:43:36 | INFO | train | epoch 195 | loss 3.1 | nll_loss 1.187 | glat_accu 0.535 | glat_context_p 0.463 | word_ins 2.981 | length 3.019 | ppl 8.58 | wps 111561 | ups 1.85 | wpb 60412.9 | bsz 2155.9 | num_updates 54949 | lr 0.000134903 | gnorm 0.515 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:43:36 | INFO | fairseq.trainer | begin training epoch 196
2023-06-13 19:44:06 | INFO | train_inner | epoch 196:     51 / 282 loss=3.102, nll_loss=1.189, glat_accu=0.541, glat_context_p=0.463, word_ins=2.983, length=3.002, ppl=8.58, wps=86803.6, ups=1.44, wpb=60304.8, bsz=2178.2, num_updates=55000, lr=0.00013484, gnorm=0.519, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:44:53 | INFO | train_inner | epoch 196:    151 / 282 loss=3.102, nll_loss=1.189, glat_accu=0.545, glat_context_p=0.463, word_ins=2.983, length=3.005, ppl=8.59, wps=131276, ups=2.17, wpb=60588.1, bsz=2153.4, num_updates=55100, lr=0.000134718, gnorm=0.526, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:45:39 | INFO | train_inner | epoch 196:    251 / 282 loss=3.101, nll_loss=1.187, glat_accu=0.538, glat_context_p=0.463, word_ins=2.981, length=3.021, ppl=8.58, wps=131620, ups=2.18, wpb=60411.9, bsz=2159.3, num_updates=55200, lr=0.000134595, gnorm=0.529, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:45:53 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:45:56 | INFO | valid | epoch 196 | valid on 'valid' subset | loss 12.268 | nll_loss 11.088 | word_ins 12.039 | length 4.591 | ppl 4933.17 | bleu 30.42 | wps 88643.2 | wpb 21176.3 | bsz 666.3 | num_updates 55231 | best_bleu 31.08
2023-06-13 19:45:56 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:46:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint196.pt (epoch 196 @ 55231 updates, score 30.42) (writing took 9.438651531934738 seconds)
2023-06-13 19:46:05 | INFO | fairseq_cli.train | end of epoch 196 (average epoch stats below)
2023-06-13 19:46:05 | INFO | train | epoch 196 | loss 3.101 | nll_loss 1.188 | glat_accu 0.54 | glat_context_p 0.463 | word_ins 2.982 | length 3.016 | ppl 8.58 | wps 114372 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 55231 | lr 0.000134558 | gnorm 0.526 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:46:05 | INFO | fairseq.trainer | begin training epoch 197
2023-06-13 19:46:42 | INFO | train_inner | epoch 197:     69 / 282 loss=3.094, nll_loss=1.18, glat_accu=0.537, glat_context_p=0.463, word_ins=2.975, length=3.022, ppl=8.54, wps=94837.5, ups=1.58, wpb=60091.3, bsz=2165.3, num_updates=55300, lr=0.000134474, gnorm=0.517, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:47:28 | INFO | train_inner | epoch 197:    169 / 282 loss=3.093, nll_loss=1.18, glat_accu=0.535, glat_context_p=0.463, word_ins=2.975, length=3.004, ppl=8.53, wps=131778, ups=2.17, wpb=60681.7, bsz=2181.3, num_updates=55400, lr=0.000134352, gnorm=0.51, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:48:14 | INFO | train_inner | epoch 197:    269 / 282 loss=3.099, nll_loss=1.186, glat_accu=0.531, glat_context_p=0.463, word_ins=2.98, length=3.03, ppl=8.57, wps=131311, ups=2.17, wpb=60402.4, bsz=2122, num_updates=55500, lr=0.000134231, gnorm=0.509, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 19:48:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:48:23 | INFO | valid | epoch 197 | valid on 'valid' subset | loss 12.393 | nll_loss 11.215 | word_ins 12.155 | length 4.751 | ppl 5379.51 | bleu 30.11 | wps 88563.6 | wpb 21176.3 | bsz 666.3 | num_updates 55513 | best_bleu 31.08
2023-06-13 19:48:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:48:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint197.pt (epoch 197 @ 55513 updates, score 30.11) (writing took 8.354625463485718 seconds)
2023-06-13 19:48:31 | INFO | fairseq_cli.train | end of epoch 197 (average epoch stats below)
2023-06-13 19:48:31 | INFO | train | epoch 197 | loss 3.096 | nll_loss 1.183 | glat_accu 0.535 | glat_context_p 0.463 | word_ins 2.977 | length 3.013 | ppl 8.55 | wps 116847 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 55513 | lr 0.000134215 | gnorm 0.511 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:48:31 | INFO | fairseq.trainer | begin training epoch 198
2023-06-13 19:49:16 | INFO | train_inner | epoch 198:     87 / 282 loss=3.082, nll_loss=1.168, glat_accu=0.519, glat_context_p=0.463, word_ins=2.965, length=3.017, ppl=8.47, wps=96595, ups=1.6, wpb=60271.3, bsz=2132.6, num_updates=55600, lr=0.00013411, gnorm=0.522, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:49:19 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 19:50:03 | INFO | train_inner | epoch 198:    188 / 282 loss=3.093, nll_loss=1.179, glat_accu=0.535, glat_context_p=0.463, word_ins=2.974, length=3.013, ppl=8.53, wps=130596, ups=2.16, wpb=60534.6, bsz=2132.6, num_updates=55700, lr=0.00013399, gnorm=0.511, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:50:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:50:49 | INFO | valid | epoch 198 | valid on 'valid' subset | loss 12.366 | nll_loss 11.186 | word_ins 12.128 | length 4.76 | ppl 5279.05 | bleu 29.94 | wps 87064.7 | wpb 21176.3 | bsz 666.3 | num_updates 55794 | best_bleu 31.08
2023-06-13 19:50:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:50:59 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint198.pt (epoch 198 @ 55794 updates, score 29.94) (writing took 9.461304806172848 seconds)
2023-06-13 19:50:59 | INFO | fairseq_cli.train | end of epoch 198 (average epoch stats below)
2023-06-13 19:50:59 | INFO | train | epoch 198 | loss 3.086 | nll_loss 1.172 | glat_accu 0.525 | glat_context_p 0.463 | word_ins 2.968 | length 3.008 | ppl 8.49 | wps 115057 | ups 1.9 | wpb 60417.3 | bsz 2158.1 | num_updates 55794 | lr 0.000133877 | gnorm 0.511 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 19:50:59 | INFO | fairseq.trainer | begin training epoch 199
2023-06-13 19:51:07 | INFO | train_inner | epoch 199:      6 / 282 loss=3.082, nll_loss=1.169, glat_accu=0.521, glat_context_p=0.463, word_ins=2.965, length=3.004, ppl=8.47, wps=93444.8, ups=1.56, wpb=60078.1, bsz=2186.4, num_updates=55800, lr=0.00013387, gnorm=0.506, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:51:53 | INFO | train_inner | epoch 199:    106 / 282 loss=3.08, nll_loss=1.165, glat_accu=0.521, glat_context_p=0.463, word_ins=2.961, length=3.024, ppl=8.45, wps=131001, ups=2.17, wpb=60457.6, bsz=2136.6, num_updates=55900, lr=0.00013375, gnorm=0.503, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:52:39 | INFO | train_inner | epoch 199:    206 / 282 loss=3.073, nll_loss=1.159, glat_accu=0.517, glat_context_p=0.463, word_ins=2.956, length=3.01, ppl=8.42, wps=132968, ups=2.19, wpb=60629.7, bsz=2188.5, num_updates=56000, lr=0.000133631, gnorm=0.499, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 19:53:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:53:16 | INFO | valid | epoch 199 | valid on 'valid' subset | loss 12.62 | nll_loss 11.477 | word_ins 12.392 | length 4.549 | ppl 6294.63 | bleu 29.95 | wps 88360.4 | wpb 21176.3 | bsz 666.3 | num_updates 56076 | best_bleu 31.08
2023-06-13 19:53:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:53:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint199.pt (epoch 199 @ 56076 updates, score 29.95) (writing took 10.107264708727598 seconds)
2023-06-13 19:53:26 | INFO | fairseq_cli.train | end of epoch 199 (average epoch stats below)
2023-06-13 19:53:26 | INFO | train | epoch 199 | loss 3.077 | nll_loss 1.163 | glat_accu 0.518 | glat_context_p 0.463 | word_ins 2.959 | length 3.012 | ppl 8.44 | wps 115147 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 56076 | lr 0.00013354 | gnorm 0.5 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 19:53:27 | INFO | fairseq.trainer | begin training epoch 200
2023-06-13 19:53:44 | INFO | train_inner | epoch 200:     24 / 282 loss=3.075, nll_loss=1.161, glat_accu=0.513, glat_context_p=0.463, word_ins=2.958, length=2.998, ppl=8.43, wps=92667, ups=1.54, wpb=60144.2, bsz=2150.2, num_updates=56100, lr=0.000133511, gnorm=0.497, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 19:54:30 | INFO | train_inner | epoch 200:    124 / 282 loss=3.085, nll_loss=1.17, glat_accu=0.534, glat_context_p=0.463, word_ins=2.966, length=3.019, ppl=8.48, wps=130703, ups=2.16, wpb=60418.7, bsz=2171.1, num_updates=56200, lr=0.000133393, gnorm=0.511, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:55:15 | INFO | train_inner | epoch 200:    224 / 282 loss=3.099, nll_loss=1.187, glat_accu=0.545, glat_context_p=0.463, word_ins=2.981, length=2.993, ppl=8.57, wps=133262, ups=2.19, wpb=60823, bsz=2196, num_updates=56300, lr=0.000133274, gnorm=0.526, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 19:55:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:55:46 | INFO | valid | epoch 200 | valid on 'valid' subset | loss 12.361 | nll_loss 11.192 | word_ins 12.135 | length 4.518 | ppl 5261.15 | bleu 30.7 | wps 88893 | wpb 21176.3 | bsz 666.3 | num_updates 56358 | best_bleu 31.08
2023-06-13 19:55:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:55:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint200.pt (epoch 200 @ 56358 updates, score 30.7) (writing took 10.688778016716242 seconds)
2023-06-13 19:55:56 | INFO | fairseq_cli.train | end of epoch 200 (average epoch stats below)
2023-06-13 19:55:56 | INFO | train | epoch 200 | loss 3.09 | nll_loss 1.176 | glat_accu 0.533 | glat_context_p 0.463 | word_ins 2.971 | length 3.015 | ppl 8.52 | wps 113799 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 56358 | lr 0.000133206 | gnorm 0.517 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 19:55:56 | INFO | fairseq.trainer | begin training epoch 201
2023-06-13 19:56:22 | INFO | train_inner | epoch 201:     42 / 282 loss=3.091, nll_loss=1.176, glat_accu=0.526, glat_context_p=0.462, word_ins=2.972, length=3.04, ppl=8.52, wps=90383.5, ups=1.51, wpb=60041.8, bsz=2124.2, num_updates=56400, lr=0.000133156, gnorm=0.516, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:57:08 | INFO | train_inner | epoch 201:    142 / 282 loss=3.07, nll_loss=1.156, glat_accu=0.511, glat_context_p=0.462, word_ins=2.953, length=3.006, ppl=8.4, wps=132936, ups=2.19, wpb=60710.6, bsz=2172, num_updates=56500, lr=0.000133038, gnorm=0.494, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:57:54 | INFO | train_inner | epoch 201:    242 / 282 loss=3.084, nll_loss=1.17, glat_accu=0.532, glat_context_p=0.462, word_ins=2.966, length=3.014, ppl=8.48, wps=131367, ups=2.17, wpb=60495.6, bsz=2168.4, num_updates=56600, lr=0.00013292, gnorm=0.516, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 19:58:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 19:58:15 | INFO | valid | epoch 201 | valid on 'valid' subset | loss 12.319 | nll_loss 11.138 | word_ins 12.082 | length 4.727 | ppl 5110.55 | bleu 30.2 | wps 89448.5 | wpb 21176.3 | bsz 666.3 | num_updates 56640 | best_bleu 31.08
2023-06-13 19:58:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 19:58:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint201.pt (epoch 201 @ 56640 updates, score 30.2) (writing took 10.177093997597694 seconds)
2023-06-13 19:58:25 | INFO | fairseq_cli.train | end of epoch 201 (average epoch stats below)
2023-06-13 19:58:25 | INFO | train | epoch 201 | loss 3.081 | nll_loss 1.167 | glat_accu 0.524 | glat_context_p 0.462 | word_ins 2.963 | length 3.016 | ppl 8.46 | wps 114402 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 56640 | lr 0.000132874 | gnorm 0.509 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 19:58:25 | INFO | fairseq.trainer | begin training epoch 202
2023-06-13 19:58:59 | INFO | train_inner | epoch 202:     60 / 282 loss=3.086, nll_loss=1.172, glat_accu=0.526, glat_context_p=0.462, word_ins=2.968, length=3.031, ppl=8.49, wps=92229.4, ups=1.53, wpb=60155.9, bsz=2106.3, num_updates=56700, lr=0.000132803, gnorm=0.525, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 19:59:44 | INFO | train_inner | epoch 202:    160 / 282 loss=3.079, nll_loss=1.165, glat_accu=0.535, glat_context_p=0.462, word_ins=2.961, length=3.003, ppl=8.45, wps=133604, ups=2.21, wpb=60507.3, bsz=2184.2, num_updates=56800, lr=0.000132686, gnorm=0.508, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:00:30 | INFO | train_inner | epoch 202:    260 / 282 loss=3.075, nll_loss=1.161, glat_accu=0.519, glat_context_p=0.462, word_ins=2.957, length=3.004, ppl=8.43, wps=131832, ups=2.18, wpb=60558.2, bsz=2151.5, num_updates=56900, lr=0.00013257, gnorm=0.51, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:00:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:00:43 | INFO | valid | epoch 202 | valid on 'valid' subset | loss 12.347 | nll_loss 11.18 | word_ins 12.122 | length 4.483 | ppl 5210.46 | bleu 30.33 | wps 88321.3 | wpb 21176.3 | bsz 666.3 | num_updates 56922 | best_bleu 31.08
2023-06-13 20:00:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:00:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint202.pt (epoch 202 @ 56922 updates, score 30.33) (writing took 11.980876341462135 seconds)
2023-06-13 20:00:55 | INFO | fairseq_cli.train | end of epoch 202 (average epoch stats below)
2023-06-13 20:00:55 | INFO | train | epoch 202 | loss 3.078 | nll_loss 1.164 | glat_accu 0.525 | glat_context_p 0.462 | word_ins 2.96 | length 3.007 | ppl 8.45 | wps 113778 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 56922 | lr 0.000132544 | gnorm 0.512 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:00:55 | INFO | fairseq.trainer | begin training epoch 203
2023-06-13 20:01:37 | INFO | train_inner | epoch 203:     78 / 282 loss=3.082, nll_loss=1.168, glat_accu=0.527, glat_context_p=0.462, word_ins=2.964, length=3.01, ppl=8.47, wps=89756.5, ups=1.49, wpb=60198.2, bsz=2147.7, num_updates=57000, lr=0.000132453, gnorm=0.517, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:02:23 | INFO | train_inner | epoch 203:    178 / 282 loss=3.083, nll_loss=1.169, glat_accu=0.532, glat_context_p=0.462, word_ins=2.965, length=3.005, ppl=8.48, wps=132676, ups=2.19, wpb=60482.4, bsz=2156.9, num_updates=57100, lr=0.000132337, gnorm=0.508, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:03:09 | INFO | train_inner | epoch 203:    278 / 282 loss=3.093, nll_loss=1.18, glat_accu=0.537, glat_context_p=0.462, word_ins=2.974, length=3.014, ppl=8.53, wps=129983, ups=2.14, wpb=60601, bsz=2172.3, num_updates=57200, lr=0.000132221, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:03:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:03:14 | INFO | valid | epoch 203 | valid on 'valid' subset | loss 12.239 | nll_loss 11.049 | word_ins 12.008 | length 4.625 | ppl 4832.82 | bleu 30.77 | wps 88880.2 | wpb 21176.3 | bsz 666.3 | num_updates 57204 | best_bleu 31.08
2023-06-13 20:03:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:03:29 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint203.pt (epoch 203 @ 57204 updates, score 30.77) (writing took 14.311622086912394 seconds)
2023-06-13 20:03:29 | INFO | fairseq_cli.train | end of epoch 203 (average epoch stats below)
2023-06-13 20:03:29 | INFO | train | epoch 203 | loss 3.087 | nll_loss 1.173 | glat_accu 0.533 | glat_context_p 0.462 | word_ins 2.968 | length 3.009 | ppl 8.5 | wps 110724 | ups 1.83 | wpb 60413.8 | bsz 2157.2 | num_updates 57204 | lr 0.000132217 | gnorm 0.521 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:03:29 | INFO | fairseq.trainer | begin training epoch 204
2023-06-13 20:04:18 | INFO | train_inner | epoch 204:     96 / 282 loss=3.101, nll_loss=1.187, glat_accu=0.551, glat_context_p=0.462, word_ins=2.981, length=3.011, ppl=8.58, wps=86881.8, ups=1.45, wpb=59961.2, bsz=2138.9, num_updates=57300, lr=0.000132106, gnorm=0.533, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:05:04 | INFO | train_inner | epoch 204:    196 / 282 loss=3.094, nll_loss=1.18, glat_accu=0.546, glat_context_p=0.462, word_ins=2.975, length=3.013, ppl=8.54, wps=132554, ups=2.19, wpb=60542.1, bsz=2168.2, num_updates=57400, lr=0.000131991, gnorm=0.522, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:05:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:05:47 | INFO | valid | epoch 204 | valid on 'valid' subset | loss 12.478 | nll_loss 11.316 | word_ins 12.248 | length 4.591 | ppl 5706.08 | bleu 30.45 | wps 88116.1 | wpb 21176.3 | bsz 666.3 | num_updates 57486 | best_bleu 31.08
2023-06-13 20:05:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:06:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint204.pt (epoch 204 @ 57486 updates, score 30.45) (writing took 13.487670946866274 seconds)
2023-06-13 20:06:01 | INFO | fairseq_cli.train | end of epoch 204 (average epoch stats below)
2023-06-13 20:06:01 | INFO | train | epoch 204 | loss 3.096 | nll_loss 1.182 | glat_accu 0.545 | glat_context_p 0.462 | word_ins 2.976 | length 3.009 | ppl 8.55 | wps 112184 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 57486 | lr 0.000131892 | gnorm 0.523 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:06:01 | INFO | fairseq.trainer | begin training epoch 205
2023-06-13 20:06:13 | INFO | train_inner | epoch 205:     14 / 282 loss=3.09, nll_loss=1.177, glat_accu=0.539, glat_context_p=0.462, word_ins=2.972, length=2.99, ppl=8.52, wps=87611.2, ups=1.45, wpb=60267.8, bsz=2173.7, num_updates=57500, lr=0.000131876, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:06:59 | INFO | train_inner | epoch 205:    114 / 282 loss=3.088, nll_loss=1.173, glat_accu=0.532, glat_context_p=0.462, word_ins=2.969, length=3.024, ppl=8.5, wps=131901, ups=2.18, wpb=60634, bsz=2131.1, num_updates=57600, lr=0.000131762, gnorm=0.521, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:07:24 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 20:07:45 | INFO | train_inner | epoch 205:    215 / 282 loss=3.085, nll_loss=1.171, glat_accu=0.534, glat_context_p=0.462, word_ins=2.967, length=3, ppl=8.49, wps=131273, ups=2.17, wpb=60570.7, bsz=2163.6, num_updates=57700, lr=0.000131647, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:08:15 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:08:18 | INFO | valid | epoch 205 | valid on 'valid' subset | loss 12.644 | nll_loss 11.505 | word_ins 12.418 | length 4.538 | ppl 6401.36 | bleu 29.81 | wps 88419.6 | wpb 21176.3 | bsz 666.3 | num_updates 57767 | best_bleu 31.08
2023-06-13 20:08:18 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:08:29 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint205.pt (epoch 205 @ 57767 updates, score 29.81) (writing took 10.982125781476498 seconds)
2023-06-13 20:08:29 | INFO | fairseq_cli.train | end of epoch 205 (average epoch stats below)
2023-06-13 20:08:29 | INFO | train | epoch 205 | loss 3.083 | nll_loss 1.169 | glat_accu 0.53 | glat_context_p 0.462 | word_ins 2.965 | length 3.011 | ppl 8.47 | wps 114184 | ups 1.89 | wpb 60412.3 | bsz 2157.8 | num_updates 57767 | lr 0.000131571 | gnorm 0.518 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:08:29 | INFO | fairseq.trainer | begin training epoch 206
2023-06-13 20:08:50 | INFO | train_inner | epoch 206:     33 / 282 loss=3.076, nll_loss=1.161, glat_accu=0.516, glat_context_p=0.462, word_ins=2.958, length=3.024, ppl=8.43, wps=92028.5, ups=1.53, wpb=60148.4, bsz=2147.1, num_updates=57800, lr=0.000131533, gnorm=0.517, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:09:36 | INFO | train_inner | epoch 206:    133 / 282 loss=3.085, nll_loss=1.17, glat_accu=0.533, glat_context_p=0.461, word_ins=2.966, length=3.015, ppl=8.48, wps=132709, ups=2.19, wpb=60488.9, bsz=2176.2, num_updates=57900, lr=0.00013142, gnorm=0.512, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:10:22 | INFO | train_inner | epoch 206:    233 / 282 loss=3.099, nll_loss=1.186, glat_accu=0.537, glat_context_p=0.461, word_ins=2.98, length=3.014, ppl=8.57, wps=132787, ups=2.19, wpb=60651.3, bsz=2133.5, num_updates=58000, lr=0.000131306, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:10:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:10:47 | INFO | valid | epoch 206 | valid on 'valid' subset | loss 12.337 | nll_loss 11.151 | word_ins 12.101 | length 4.714 | ppl 5174.31 | bleu 30.86 | wps 90974 | wpb 21176.3 | bsz 666.3 | num_updates 58049 | best_bleu 31.08
2023-06-13 20:10:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:10:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint206.pt (epoch 206 @ 58049 updates, score 30.86) (writing took 10.612159952521324 seconds)
2023-06-13 20:10:58 | INFO | fairseq_cli.train | end of epoch 206 (average epoch stats below)
2023-06-13 20:10:58 | INFO | train | epoch 206 | loss 3.091 | nll_loss 1.177 | glat_accu 0.537 | glat_context_p 0.461 | word_ins 2.972 | length 3.013 | ppl 8.52 | wps 114713 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 58049 | lr 0.000131251 | gnorm 0.522 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:10:58 | INFO | fairseq.trainer | begin training epoch 207
2023-06-13 20:11:28 | INFO | train_inner | epoch 207:     51 / 282 loss=3.096, nll_loss=1.183, glat_accu=0.547, glat_context_p=0.461, word_ins=2.977, length=2.999, ppl=8.55, wps=90925.1, ups=1.51, wpb=60102, bsz=2144.9, num_updates=58100, lr=0.000131193, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:12:14 | INFO | train_inner | epoch 207:    151 / 282 loss=3.089, nll_loss=1.174, glat_accu=0.54, glat_context_p=0.461, word_ins=2.969, length=3.011, ppl=8.51, wps=131347, ups=2.17, wpb=60484.8, bsz=2171.6, num_updates=58200, lr=0.000131081, gnorm=0.522, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:12:40 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 20:13:00 | INFO | train_inner | epoch 207:    252 / 282 loss=3.085, nll_loss=1.171, glat_accu=0.536, glat_context_p=0.461, word_ins=2.966, length=2.993, ppl=8.48, wps=132157, ups=2.18, wpb=60628.5, bsz=2191.8, num_updates=58300, lr=0.000130968, gnorm=0.518, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:13:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:13:16 | INFO | valid | epoch 207 | valid on 'valid' subset | loss 12.323 | nll_loss 11.159 | word_ins 12.099 | length 4.477 | ppl 5123.59 | bleu 30.94 | wps 87528.2 | wpb 21176.3 | bsz 666.3 | num_updates 58330 | best_bleu 31.08
2023-06-13 20:13:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:13:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint207.pt (epoch 207 @ 58330 updates, score 30.94) (writing took 10.880822233855724 seconds)
2023-06-13 20:13:27 | INFO | fairseq_cli.train | end of epoch 207 (average epoch stats below)
2023-06-13 20:13:27 | INFO | train | epoch 207 | loss 3.088 | nll_loss 1.174 | glat_accu 0.536 | glat_context_p 0.461 | word_ins 2.969 | length 3.006 | ppl 8.5 | wps 113761 | ups 1.88 | wpb 60416 | bsz 2157.5 | num_updates 58330 | lr 0.000130934 | gnorm 0.527 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 20:13:27 | INFO | fairseq.trainer | begin training epoch 208
2023-06-13 20:14:05 | INFO | train_inner | epoch 208:     70 / 282 loss=3.091, nll_loss=1.177, glat_accu=0.532, glat_context_p=0.461, word_ins=2.972, length=3.012, ppl=8.52, wps=91199.2, ups=1.52, wpb=60044.2, bsz=2139.7, num_updates=58400, lr=0.000130856, gnorm=0.537, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 20:14:51 | INFO | train_inner | epoch 208:    170 / 282 loss=3.092, nll_loss=1.178, glat_accu=0.548, glat_context_p=0.461, word_ins=2.973, length=2.997, ppl=8.53, wps=133020, ups=2.2, wpb=60577.2, bsz=2176.9, num_updates=58500, lr=0.000130744, gnorm=0.517, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 20:15:38 | INFO | train_inner | epoch 208:    270 / 282 loss=3.099, nll_loss=1.185, glat_accu=0.54, glat_context_p=0.461, word_ins=2.979, length=3.023, ppl=8.57, wps=130355, ups=2.15, wpb=60646.2, bsz=2130.7, num_updates=58600, lr=0.000130632, gnorm=0.52, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:15:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:15:46 | INFO | valid | epoch 208 | valid on 'valid' subset | loss 12.402 | nll_loss 11.233 | word_ins 12.171 | length 4.624 | ppl 5411.59 | bleu 30.74 | wps 88232.5 | wpb 21176.3 | bsz 666.3 | num_updates 58612 | best_bleu 31.08
2023-06-13 20:15:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:15:59 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint208.pt (epoch 208 @ 58612 updates, score 30.74) (writing took 12.774059500545263 seconds)
2023-06-13 20:15:59 | INFO | fairseq_cli.train | end of epoch 208 (average epoch stats below)
2023-06-13 20:15:59 | INFO | train | epoch 208 | loss 3.094 | nll_loss 1.181 | glat_accu 0.544 | glat_context_p 0.461 | word_ins 2.975 | length 3.006 | ppl 8.54 | wps 112211 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 58612 | lr 0.000130619 | gnorm 0.523 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 20:15:59 | INFO | fairseq.trainer | begin training epoch 209
2023-06-13 20:16:45 | INFO | train_inner | epoch 209:     88 / 282 loss=3.079, nll_loss=1.164, glat_accu=0.537, glat_context_p=0.461, word_ins=2.96, length=2.998, ppl=8.45, wps=89408.2, ups=1.49, wpb=60090.7, bsz=2154.6, num_updates=58700, lr=0.000130521, gnorm=0.526, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 20:17:30 | INFO | train_inner | epoch 209:    188 / 282 loss=3.097, nll_loss=1.183, glat_accu=0.546, glat_context_p=0.461, word_ins=2.977, length=3.013, ppl=8.56, wps=132479, ups=2.18, wpb=60637.6, bsz=2143.2, num_updates=58800, lr=0.00013041, gnorm=0.523, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:18:14 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:18:17 | INFO | valid | epoch 209 | valid on 'valid' subset | loss 12.379 | nll_loss 11.209 | word_ins 12.151 | length 4.566 | ppl 5327.69 | bleu 31.2 | wps 88503.7 | wpb 21176.3 | bsz 666.3 | num_updates 58894 | best_bleu 31.2
2023-06-13 20:18:17 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:18:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint209.pt (epoch 209 @ 58894 updates, score 31.2) (writing took 16.97312283143401 seconds)
2023-06-13 20:18:34 | INFO | fairseq_cli.train | end of epoch 209 (average epoch stats below)
2023-06-13 20:18:34 | INFO | train | epoch 209 | loss 3.092 | nll_loss 1.177 | glat_accu 0.543 | glat_context_p 0.461 | word_ins 2.972 | length 3.008 | ppl 8.52 | wps 109790 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 58894 | lr 0.000130306 | gnorm 0.524 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 20:18:34 | INFO | fairseq.trainer | begin training epoch 210
2023-06-13 20:18:43 | INFO | train_inner | epoch 210:      6 / 282 loss=3.096, nll_loss=1.183, glat_accu=0.551, glat_context_p=0.461, word_ins=2.977, length=3.004, ppl=8.55, wps=82982.5, ups=1.38, wpb=60099.9, bsz=2180, num_updates=58900, lr=0.000130299, gnorm=0.527, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:19:29 | INFO | train_inner | epoch 210:    106 / 282 loss=3.083, nll_loss=1.168, glat_accu=0.54, glat_context_p=0.461, word_ins=2.964, length=3.006, ppl=8.47, wps=130423, ups=2.15, wpb=60611.9, bsz=2174, num_updates=59000, lr=0.000130189, gnorm=0.517, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:20:15 | INFO | train_inner | epoch 210:    206 / 282 loss=3.092, nll_loss=1.178, glat_accu=0.542, glat_context_p=0.461, word_ins=2.972, length=3.017, ppl=8.53, wps=132364, ups=2.19, wpb=60491.1, bsz=2113.8, num_updates=59100, lr=0.000130079, gnorm=0.524, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:20:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:20:53 | INFO | valid | epoch 210 | valid on 'valid' subset | loss 12.392 | nll_loss 11.223 | word_ins 12.166 | length 4.521 | ppl 5375.86 | bleu 31.16 | wps 88747.6 | wpb 21176.3 | bsz 666.3 | num_updates 59176 | best_bleu 31.2
2023-06-13 20:20:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:21:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint210.pt (epoch 210 @ 59176 updates, score 31.16) (writing took 8.676337648183107 seconds)
2023-06-13 20:21:02 | INFO | fairseq_cli.train | end of epoch 210 (average epoch stats below)
2023-06-13 20:21:02 | INFO | train | epoch 210 | loss 3.087 | nll_loss 1.173 | glat_accu 0.542 | glat_context_p 0.461 | word_ins 2.968 | length 3.005 | ppl 8.5 | wps 115208 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 59176 | lr 0.000129995 | gnorm 0.522 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 20:21:02 | INFO | fairseq.trainer | begin training epoch 211
2023-06-13 20:21:18 | INFO | train_inner | epoch 211:     24 / 282 loss=3.086, nll_loss=1.172, glat_accu=0.542, glat_context_p=0.461, word_ins=2.967, length=2.992, ppl=8.49, wps=95527.4, ups=1.59, wpb=60231.5, bsz=2170.5, num_updates=59200, lr=0.000129969, gnorm=0.522, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 20:22:04 | INFO | train_inner | epoch 211:    124 / 282 loss=3.084, nll_loss=1.169, glat_accu=0.543, glat_context_p=0.461, word_ins=2.965, length=3.002, ppl=8.48, wps=132331, ups=2.19, wpb=60441.2, bsz=2165.8, num_updates=59300, lr=0.000129859, gnorm=0.512, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:22:49 | INFO | train_inner | epoch 211:    224 / 282 loss=3.085, nll_loss=1.171, glat_accu=0.537, glat_context_p=0.46, word_ins=2.966, length=2.992, ppl=8.48, wps=133308, ups=2.19, wpb=60856.3, bsz=2171, num_updates=59400, lr=0.00012975, gnorm=0.514, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:23:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:23:20 | INFO | valid | epoch 211 | valid on 'valid' subset | loss 12.435 | nll_loss 11.28 | word_ins 12.211 | length 4.489 | ppl 5538.98 | bleu 30.82 | wps 88657.9 | wpb 21176.3 | bsz 666.3 | num_updates 59458 | best_bleu 31.2
2023-06-13 20:23:20 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:23:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint211.pt (epoch 211 @ 59458 updates, score 30.82) (writing took 10.818200778216124 seconds)
2023-06-13 20:23:31 | INFO | fairseq_cli.train | end of epoch 211 (average epoch stats below)
2023-06-13 20:23:31 | INFO | train | epoch 211 | loss 3.085 | nll_loss 1.171 | glat_accu 0.539 | glat_context_p 0.46 | word_ins 2.966 | length 3.004 | ppl 8.49 | wps 114590 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 59458 | lr 0.000129687 | gnorm 0.517 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:23:31 | INFO | fairseq.trainer | begin training epoch 212
2023-06-13 20:23:56 | INFO | train_inner | epoch 212:     42 / 282 loss=3.087, nll_loss=1.172, glat_accu=0.532, glat_context_p=0.46, word_ins=2.967, length=3.037, ppl=8.5, wps=89775.7, ups=1.5, wpb=59857, bsz=2117.9, num_updates=59500, lr=0.000129641, gnorm=0.527, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:24:42 | INFO | train_inner | epoch 212:    142 / 282 loss=3.086, nll_loss=1.171, glat_accu=0.541, glat_context_p=0.46, word_ins=2.967, length=3.007, ppl=8.49, wps=133345, ups=2.2, wpb=60740, bsz=2186.1, num_updates=59600, lr=0.000129532, gnorm=0.518, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:25:28 | INFO | train_inner | epoch 212:    242 / 282 loss=3.071, nll_loss=1.156, glat_accu=0.528, glat_context_p=0.46, word_ins=2.953, length=3.009, ppl=8.4, wps=130907, ups=2.16, wpb=60469.1, bsz=2123.4, num_updates=59700, lr=0.000129423, gnorm=0.525, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:25:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:25:50 | INFO | valid | epoch 212 | valid on 'valid' subset | loss 12.43 | nll_loss 11.268 | word_ins 12.205 | length 4.504 | ppl 5519.26 | bleu 30.97 | wps 89481.9 | wpb 21176.3 | bsz 666.3 | num_updates 59740 | best_bleu 31.2
2023-06-13 20:25:50 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:26:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint212.pt (epoch 212 @ 59740 updates, score 30.97) (writing took 10.936437703669071 seconds)
2023-06-13 20:26:01 | INFO | fairseq_cli.train | end of epoch 212 (average epoch stats below)
2023-06-13 20:26:01 | INFO | train | epoch 212 | loss 3.08 | nll_loss 1.165 | glat_accu 0.533 | glat_context_p 0.46 | word_ins 2.961 | length 3.01 | ppl 8.46 | wps 113608 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 59740 | lr 0.00012938 | gnorm 0.521 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:26:01 | INFO | fairseq.trainer | begin training epoch 213
2023-06-13 20:26:34 | INFO | train_inner | epoch 213:     60 / 282 loss=3.08, nll_loss=1.165, glat_accu=0.538, glat_context_p=0.46, word_ins=2.961, length=2.994, ppl=8.45, wps=91144.6, ups=1.51, wpb=60175.3, bsz=2188.1, num_updates=59800, lr=0.000129315, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:27:20 | INFO | train_inner | epoch 213:    160 / 282 loss=3.076, nll_loss=1.162, glat_accu=0.535, glat_context_p=0.46, word_ins=2.958, length=2.992, ppl=8.43, wps=131142, ups=2.16, wpb=60588.2, bsz=2150, num_updates=59900, lr=0.000129207, gnorm=0.511, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:28:06 | INFO | train_inner | epoch 213:    260 / 282 loss=3.072, nll_loss=1.158, glat_accu=0.527, glat_context_p=0.46, word_ins=2.954, length=2.999, ppl=8.41, wps=133304, ups=2.2, wpb=60534.1, bsz=2190.1, num_updates=60000, lr=0.000129099, gnorm=0.51, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:28:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:28:19 | INFO | valid | epoch 213 | valid on 'valid' subset | loss 12.425 | nll_loss 11.261 | word_ins 12.195 | length 4.6 | ppl 5498.67 | bleu 30.63 | wps 88731.7 | wpb 21176.3 | bsz 666.3 | num_updates 60022 | best_bleu 31.2
2023-06-13 20:28:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:28:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint213.pt (epoch 213 @ 60022 updates, score 30.63) (writing took 8.914722766727209 seconds)
2023-06-13 20:28:28 | INFO | fairseq_cli.train | end of epoch 213 (average epoch stats below)
2023-06-13 20:28:28 | INFO | train | epoch 213 | loss 3.076 | nll_loss 1.161 | glat_accu 0.533 | glat_context_p 0.46 | word_ins 2.957 | length 2.999 | ppl 8.43 | wps 115651 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 60022 | lr 0.000129076 | gnorm 0.512 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:28:28 | INFO | fairseq.trainer | begin training epoch 214
2023-06-13 20:29:09 | INFO | train_inner | epoch 214:     78 / 282 loss=3.08, nll_loss=1.165, glat_accu=0.532, glat_context_p=0.46, word_ins=2.961, length=3.01, ppl=8.46, wps=95168.9, ups=1.59, wpb=59985.1, bsz=2126.2, num_updates=60100, lr=0.000128992, gnorm=0.524, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:29:54 | INFO | train_inner | epoch 214:    178 / 282 loss=3.086, nll_loss=1.172, glat_accu=0.548, glat_context_p=0.46, word_ins=2.967, length=2.994, ppl=8.49, wps=132890, ups=2.19, wpb=60707.6, bsz=2171.3, num_updates=60200, lr=0.000128885, gnorm=0.528, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:30:40 | INFO | train_inner | epoch 214:    278 / 282 loss=3.079, nll_loss=1.164, glat_accu=0.528, glat_context_p=0.46, word_ins=2.96, length=3.018, ppl=8.45, wps=132404, ups=2.19, wpb=60562.2, bsz=2145, num_updates=60300, lr=0.000128778, gnorm=0.513, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:30:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:30:45 | INFO | valid | epoch 214 | valid on 'valid' subset | loss 12.508 | nll_loss 11.354 | word_ins 12.278 | length 4.597 | ppl 5823.56 | bleu 30.2 | wps 89442.4 | wpb 21176.3 | bsz 666.3 | num_updates 60304 | best_bleu 31.2
2023-06-13 20:30:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:30:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint214.pt (epoch 214 @ 60304 updates, score 30.2) (writing took 10.848178215324879 seconds)
2023-06-13 20:30:56 | INFO | fairseq_cli.train | end of epoch 214 (average epoch stats below)
2023-06-13 20:30:56 | INFO | train | epoch 214 | loss 3.081 | nll_loss 1.167 | glat_accu 0.539 | glat_context_p 0.46 | word_ins 2.962 | length 3.002 | ppl 8.46 | wps 115248 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 60304 | lr 0.000128774 | gnorm 0.522 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:30:56 | INFO | fairseq.trainer | begin training epoch 215
2023-06-13 20:31:03 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 20:31:47 | INFO | train_inner | epoch 215:     97 / 282 loss=3.075, nll_loss=1.159, glat_accu=0.541, glat_context_p=0.46, word_ins=2.956, length=2.998, ppl=8.43, wps=89962, ups=1.5, wpb=60010.4, bsz=2164.2, num_updates=60400, lr=0.000128671, gnorm=0.526, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:32:32 | INFO | train_inner | epoch 215:    197 / 282 loss=3.075, nll_loss=1.16, glat_accu=0.533, glat_context_p=0.46, word_ins=2.956, length=3.006, ppl=8.43, wps=132428, ups=2.19, wpb=60540.3, bsz=2189.3, num_updates=60500, lr=0.000128565, gnorm=0.52, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:33:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:33:14 | INFO | valid | epoch 215 | valid on 'valid' subset | loss 12.538 | nll_loss 11.393 | word_ins 12.312 | length 4.524 | ppl 5947.5 | bleu 30.54 | wps 87871.5 | wpb 21176.3 | bsz 666.3 | num_updates 60585 | best_bleu 31.2
2023-06-13 20:33:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:33:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint215.pt (epoch 215 @ 60585 updates, score 30.54) (writing took 11.353703521192074 seconds)
2023-06-13 20:33:25 | INFO | fairseq_cli.train | end of epoch 215 (average epoch stats below)
2023-06-13 20:33:25 | INFO | train | epoch 215 | loss 3.077 | nll_loss 1.162 | glat_accu 0.534 | glat_context_p 0.46 | word_ins 2.958 | length 3.005 | ppl 8.44 | wps 113419 | ups 1.88 | wpb 60411.2 | bsz 2157.7 | num_updates 60585 | lr 0.000128475 | gnorm 0.522 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:33:25 | INFO | fairseq.trainer | begin training epoch 216
2023-06-13 20:33:39 | INFO | train_inner | epoch 216:     15 / 282 loss=3.08, nll_loss=1.166, glat_accu=0.527, glat_context_p=0.46, word_ins=2.962, length=3.015, ppl=8.46, wps=90301.3, ups=1.5, wpb=60274.4, bsz=2098.3, num_updates=60600, lr=0.000128459, gnorm=0.525, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:34:25 | INFO | train_inner | epoch 216:    115 / 282 loss=3.07, nll_loss=1.155, glat_accu=0.527, glat_context_p=0.46, word_ins=2.952, length=3.011, ppl=8.4, wps=132380, ups=2.19, wpb=60526.2, bsz=2153.4, num_updates=60700, lr=0.000128353, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:35:11 | INFO | train_inner | epoch 216:    215 / 282 loss=3.079, nll_loss=1.165, glat_accu=0.54, glat_context_p=0.46, word_ins=2.96, length=3.002, ppl=8.45, wps=132007, ups=2.18, wpb=60568.2, bsz=2171.2, num_updates=60800, lr=0.000128247, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:35:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:35:45 | INFO | valid | epoch 216 | valid on 'valid' subset | loss 12.43 | nll_loss 11.262 | word_ins 12.195 | length 4.722 | ppl 5518.32 | bleu 30.12 | wps 88960.2 | wpb 21176.3 | bsz 666.3 | num_updates 60867 | best_bleu 31.2
2023-06-13 20:35:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:35:54 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint216.pt (epoch 216 @ 60867 updates, score 30.12) (writing took 9.005828093737364 seconds)
2023-06-13 20:35:54 | INFO | fairseq_cli.train | end of epoch 216 (average epoch stats below)
2023-06-13 20:35:54 | INFO | train | epoch 216 | loss 3.073 | nll_loss 1.158 | glat_accu 0.53 | glat_context_p 0.46 | word_ins 2.955 | length 3 | ppl 8.41 | wps 114986 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 60867 | lr 0.000128177 | gnorm 0.516 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:35:54 | INFO | fairseq.trainer | begin training epoch 217
2023-06-13 20:36:14 | INFO | train_inner | epoch 217:     33 / 282 loss=3.067, nll_loss=1.153, glat_accu=0.529, glat_context_p=0.459, word_ins=2.95, length=2.981, ppl=8.38, wps=94936.6, ups=1.58, wpb=60130.1, bsz=2172.8, num_updates=60900, lr=0.000128142, gnorm=0.508, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:37:00 | INFO | train_inner | epoch 217:    133 / 282 loss=3.077, nll_loss=1.163, glat_accu=0.537, glat_context_p=0.459, word_ins=2.958, length=2.993, ppl=8.44, wps=130876, ups=2.16, wpb=60636.3, bsz=2153.8, num_updates=61000, lr=0.000128037, gnorm=0.521, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:37:47 | INFO | train_inner | epoch 217:    233 / 282 loss=3.079, nll_loss=1.165, glat_accu=0.533, glat_context_p=0.459, word_ins=2.96, length=2.996, ppl=8.45, wps=131296, ups=2.17, wpb=60556.4, bsz=2144.1, num_updates=61100, lr=0.000127932, gnorm=0.515, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:38:09 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:38:12 | INFO | valid | epoch 217 | valid on 'valid' subset | loss 12.404 | nll_loss 11.231 | word_ins 12.172 | length 4.664 | ppl 5418.94 | bleu 30.79 | wps 89265.3 | wpb 21176.3 | bsz 666.3 | num_updates 61149 | best_bleu 31.2
2023-06-13 20:38:12 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:38:22 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint217.pt (epoch 217 @ 61149 updates, score 30.79) (writing took 9.708811156451702 seconds)
2023-06-13 20:38:22 | INFO | fairseq_cli.train | end of epoch 217 (average epoch stats below)
2023-06-13 20:38:22 | INFO | train | epoch 217 | loss 3.077 | nll_loss 1.162 | glat_accu 0.536 | glat_context_p 0.459 | word_ins 2.958 | length 2.996 | ppl 8.44 | wps 114942 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 61149 | lr 0.000127881 | gnorm 0.523 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:38:22 | INFO | fairseq.trainer | begin training epoch 218
2023-06-13 20:38:51 | INFO | train_inner | epoch 218:     51 / 282 loss=3.073, nll_loss=1.158, glat_accu=0.534, glat_context_p=0.459, word_ins=2.955, length=2.995, ppl=8.41, wps=93986.8, ups=1.56, wpb=60139.7, bsz=2188.9, num_updates=61200, lr=0.000127827, gnorm=0.533, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:39:36 | INFO | train_inner | epoch 218:    151 / 282 loss=3.076, nll_loss=1.162, glat_accu=0.538, glat_context_p=0.459, word_ins=2.958, length=2.982, ppl=8.43, wps=133164, ups=2.19, wpb=60704.3, bsz=2187.7, num_updates=61300, lr=0.000127723, gnorm=0.514, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:39:51 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 20:40:23 | INFO | train_inner | epoch 218:    252 / 282 loss=3.074, nll_loss=1.159, glat_accu=0.531, glat_context_p=0.459, word_ins=2.955, length=3.007, ppl=8.42, wps=128816, ups=2.13, wpb=60430.5, bsz=2129.9, num_updates=61400, lr=0.000127619, gnorm=0.513, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 20:40:37 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:40:40 | INFO | valid | epoch 218 | valid on 'valid' subset | loss 12.51 | nll_loss 11.36 | word_ins 12.284 | length 4.501 | ppl 5832.47 | bleu 30.7 | wps 87300.3 | wpb 21176.3 | bsz 666.3 | num_updates 61430 | best_bleu 31.2
2023-06-13 20:40:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:40:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint218.pt (epoch 218 @ 61430 updates, score 30.7) (writing took 10.64849841222167 seconds)
2023-06-13 20:40:51 | INFO | fairseq_cli.train | end of epoch 218 (average epoch stats below)
2023-06-13 20:40:51 | INFO | train | epoch 218 | loss 3.075 | nll_loss 1.161 | glat_accu 0.534 | glat_context_p 0.459 | word_ins 2.957 | length 2.996 | ppl 8.43 | wps 114048 | ups 1.89 | wpb 60409.4 | bsz 2157.8 | num_updates 61430 | lr 0.000127588 | gnorm 0.516 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:40:51 | INFO | fairseq.trainer | begin training epoch 219
2023-06-13 20:41:29 | INFO | train_inner | epoch 219:     70 / 282 loss=3.075, nll_loss=1.159, glat_accu=0.534, glat_context_p=0.459, word_ins=2.956, length=3.007, ppl=8.42, wps=91661.3, ups=1.52, wpb=60179.2, bsz=2144.4, num_updates=61500, lr=0.000127515, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:42:14 | INFO | train_inner | epoch 219:    170 / 282 loss=3.062, nll_loss=1.147, glat_accu=0.529, glat_context_p=0.459, word_ins=2.944, length=2.988, ppl=8.35, wps=132462, ups=2.19, wpb=60451.5, bsz=2207.6, num_updates=61600, lr=0.000127412, gnorm=0.511, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:43:00 | INFO | train_inner | epoch 219:    270 / 282 loss=3.081, nll_loss=1.166, glat_accu=0.533, glat_context_p=0.459, word_ins=2.962, length=3.009, ppl=8.46, wps=131834, ups=2.17, wpb=60627.1, bsz=2112.8, num_updates=61700, lr=0.000127309, gnorm=0.518, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:43:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:43:09 | INFO | valid | epoch 219 | valid on 'valid' subset | loss 12.382 | nll_loss 11.218 | word_ins 12.158 | length 4.494 | ppl 5338.56 | bleu 30.79 | wps 88715.8 | wpb 21176.3 | bsz 666.3 | num_updates 61712 | best_bleu 31.2
2023-06-13 20:43:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:43:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint219.pt (epoch 219 @ 61712 updates, score 30.79) (writing took 11.431082658469677 seconds)
2023-06-13 20:43:20 | INFO | fairseq_cli.train | end of epoch 219 (average epoch stats below)
2023-06-13 20:43:20 | INFO | train | epoch 219 | loss 3.071 | nll_loss 1.156 | glat_accu 0.531 | glat_context_p 0.459 | word_ins 2.953 | length 3.002 | ppl 8.41 | wps 114017 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 61712 | lr 0.000127296 | gnorm 0.522 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:43:20 | INFO | fairseq.trainer | begin training epoch 220
2023-06-13 20:44:07 | INFO | train_inner | epoch 220:     88 / 282 loss=3.079, nll_loss=1.163, glat_accu=0.535, glat_context_p=0.459, word_ins=2.959, length=3.028, ppl=8.45, wps=90282.4, ups=1.5, wpb=60017.3, bsz=2099.7, num_updates=61800, lr=0.000127205, gnorm=0.535, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:44:53 | INFO | train_inner | epoch 220:    188 / 282 loss=3.07, nll_loss=1.155, glat_accu=0.536, glat_context_p=0.459, word_ins=2.951, length=2.998, ppl=8.4, wps=131666, ups=2.17, wpb=60615.7, bsz=2172.6, num_updates=61900, lr=0.000127103, gnorm=0.518, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:45:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:45:40 | INFO | valid | epoch 220 | valid on 'valid' subset | loss 12.564 | nll_loss 11.422 | word_ins 12.339 | length 4.47 | ppl 6054.52 | bleu 30.53 | wps 89155 | wpb 21176.3 | bsz 666.3 | num_updates 61994 | best_bleu 31.2
2023-06-13 20:45:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:45:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint220.pt (epoch 220 @ 61994 updates, score 30.53) (writing took 11.376273710280657 seconds)
2023-06-13 20:45:51 | INFO | fairseq_cli.train | end of epoch 220 (average epoch stats below)
2023-06-13 20:45:51 | INFO | train | epoch 220 | loss 3.074 | nll_loss 1.159 | glat_accu 0.537 | glat_context_p 0.459 | word_ins 2.955 | length 3.002 | ppl 8.42 | wps 112590 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 61994 | lr 0.000127006 | gnorm 0.522 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:45:51 | INFO | fairseq.trainer | begin training epoch 221
2023-06-13 20:46:00 | INFO | train_inner | epoch 221:      6 / 282 loss=3.076, nll_loss=1.162, glat_accu=0.54, glat_context_p=0.459, word_ins=2.958, length=2.986, ppl=8.43, wps=89547.2, ups=1.49, wpb=60120.6, bsz=2173.3, num_updates=62000, lr=0.000127, gnorm=0.52, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:46:46 | INFO | train_inner | epoch 221:    106 / 282 loss=3.067, nll_loss=1.152, glat_accu=0.533, glat_context_p=0.459, word_ins=2.949, length=2.979, ppl=8.38, wps=131292, ups=2.16, wpb=60749.4, bsz=2167.1, num_updates=62100, lr=0.000126898, gnorm=0.516, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:47:32 | INFO | train_inner | epoch 221:    206 / 282 loss=3.072, nll_loss=1.158, glat_accu=0.531, glat_context_p=0.459, word_ins=2.954, length=3.003, ppl=8.41, wps=131667, ups=2.17, wpb=60565.6, bsz=2147.9, num_updates=62200, lr=0.000126796, gnorm=0.509, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:48:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:48:10 | INFO | valid | epoch 221 | valid on 'valid' subset | loss 12.446 | nll_loss 11.291 | word_ins 12.219 | length 4.516 | ppl 5578.63 | bleu 31 | wps 89344.4 | wpb 21176.3 | bsz 666.3 | num_updates 62276 | best_bleu 31.2
2023-06-13 20:48:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:48:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint221.pt (epoch 221 @ 62276 updates, score 31.0) (writing took 12.707207005470991 seconds)
2023-06-13 20:48:23 | INFO | fairseq_cli.train | end of epoch 221 (average epoch stats below)
2023-06-13 20:48:23 | INFO | train | epoch 221 | loss 3.074 | nll_loss 1.159 | glat_accu 0.537 | glat_context_p 0.459 | word_ins 2.956 | length 2.995 | ppl 8.42 | wps 112657 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 62276 | lr 0.000126718 | gnorm 0.523 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:48:23 | INFO | fairseq.trainer | begin training epoch 222
2023-06-13 20:48:40 | INFO | train_inner | epoch 222:     24 / 282 loss=3.083, nll_loss=1.168, glat_accu=0.551, glat_context_p=0.459, word_ins=2.963, length=2.992, ppl=8.47, wps=89096.2, ups=1.49, wpb=59979.5, bsz=2173.8, num_updates=62300, lr=0.000126694, gnorm=0.545, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:49:05 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 20:49:26 | INFO | train_inner | epoch 222:    125 / 282 loss=3.072, nll_loss=1.157, glat_accu=0.539, glat_context_p=0.458, word_ins=2.954, length=2.989, ppl=8.41, wps=130395, ups=2.14, wpb=60800.4, bsz=2144.7, num_updates=62400, lr=0.000126592, gnorm=0.521, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:50:12 | INFO | train_inner | epoch 222:    225 / 282 loss=3.083, nll_loss=1.169, glat_accu=0.542, glat_context_p=0.458, word_ins=2.964, length=3.008, ppl=8.48, wps=130803, ups=2.16, wpb=60428.7, bsz=2131.3, num_updates=62500, lr=0.000126491, gnorm=0.533, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:50:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:50:41 | INFO | valid | epoch 222 | valid on 'valid' subset | loss 12.454 | nll_loss 11.298 | word_ins 12.226 | length 4.555 | ppl 5611.89 | bleu 30.68 | wps 87600.6 | wpb 21176.3 | bsz 666.3 | num_updates 62557 | best_bleu 31.2
2023-06-13 20:50:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:50:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint222.pt (epoch 222 @ 62557 updates, score 30.68) (writing took 9.928947478532791 seconds)
2023-06-13 20:50:51 | INFO | fairseq_cli.train | end of epoch 222 (average epoch stats below)
2023-06-13 20:50:51 | INFO | train | epoch 222 | loss 3.079 | nll_loss 1.164 | glat_accu 0.544 | glat_context_p 0.458 | word_ins 2.96 | length 2.995 | ppl 8.45 | wps 114120 | ups 1.89 | wpb 60414.9 | bsz 2157.2 | num_updates 62557 | lr 0.000126433 | gnorm 0.532 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:50:51 | INFO | fairseq.trainer | begin training epoch 223
2023-06-13 20:51:16 | INFO | train_inner | epoch 223:     43 / 282 loss=3.087, nll_loss=1.172, glat_accu=0.541, glat_context_p=0.458, word_ins=2.967, length=3.005, ppl=8.49, wps=94220.6, ups=1.56, wpb=60221.4, bsz=2142.6, num_updates=62600, lr=0.00012639, gnorm=0.549, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:52:02 | INFO | train_inner | epoch 223:    143 / 282 loss=3.067, nll_loss=1.152, glat_accu=0.535, glat_context_p=0.458, word_ins=2.949, length=2.994, ppl=8.38, wps=132129, ups=2.19, wpb=60432.2, bsz=2185.3, num_updates=62700, lr=0.000126289, gnorm=0.508, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:52:48 | INFO | train_inner | epoch 223:    243 / 282 loss=3.077, nll_loss=1.162, glat_accu=0.539, glat_context_p=0.458, word_ins=2.958, length=3.006, ppl=8.44, wps=132807, ups=2.19, wpb=60536, bsz=2164, num_updates=62800, lr=0.000126189, gnorm=0.526, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:53:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:53:08 | INFO | valid | epoch 223 | valid on 'valid' subset | loss 12.489 | nll_loss 11.331 | word_ins 12.257 | length 4.658 | ppl 5747.9 | bleu 30.7 | wps 89327.3 | wpb 21176.3 | bsz 666.3 | num_updates 62839 | best_bleu 31.2
2023-06-13 20:53:08 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:53:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint223.pt (epoch 223 @ 62839 updates, score 30.7) (writing took 11.569599319249392 seconds)
2023-06-13 20:53:20 | INFO | fairseq_cli.train | end of epoch 223 (average epoch stats below)
2023-06-13 20:53:20 | INFO | train | epoch 223 | loss 3.073 | nll_loss 1.158 | glat_accu 0.537 | glat_context_p 0.458 | word_ins 2.955 | length 3 | ppl 8.42 | wps 114782 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 62839 | lr 0.000126149 | gnorm 0.523 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 20:53:20 | INFO | fairseq.trainer | begin training epoch 224
2023-06-13 20:53:54 | INFO | train_inner | epoch 224:     61 / 282 loss=3.07, nll_loss=1.154, glat_accu=0.541, glat_context_p=0.458, word_ins=2.951, length=3.003, ppl=8.4, wps=91062.9, ups=1.51, wpb=60113, bsz=2136.2, num_updates=62900, lr=0.000126088, gnorm=0.523, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:54:40 | INFO | train_inner | epoch 224:    161 / 282 loss=3.074, nll_loss=1.158, glat_accu=0.543, glat_context_p=0.458, word_ins=2.954, length=3.009, ppl=8.42, wps=130500, ups=2.16, wpb=60365.2, bsz=2159.9, num_updates=63000, lr=0.000125988, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:55:25 | INFO | train_inner | epoch 224:    261 / 282 loss=3.077, nll_loss=1.163, glat_accu=0.541, glat_context_p=0.458, word_ins=2.958, length=2.977, ppl=8.44, wps=133442, ups=2.19, wpb=60825.1, bsz=2187.2, num_updates=63100, lr=0.000125888, gnorm=0.538, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:55:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:55:38 | INFO | valid | epoch 224 | valid on 'valid' subset | loss 12.579 | nll_loss 11.437 | word_ins 12.351 | length 4.556 | ppl 6118.61 | bleu 30.55 | wps 89514.2 | wpb 21176.3 | bsz 666.3 | num_updates 63121 | best_bleu 31.2
2023-06-13 20:55:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:55:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint224.pt (epoch 224 @ 63121 updates, score 30.55) (writing took 11.241239976137877 seconds)
2023-06-13 20:55:49 | INFO | fairseq_cli.train | end of epoch 224 (average epoch stats below)
2023-06-13 20:55:49 | INFO | train | epoch 224 | loss 3.074 | nll_loss 1.159 | glat_accu 0.539 | glat_context_p 0.458 | word_ins 2.955 | length 2.997 | ppl 8.42 | wps 113785 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 63121 | lr 0.000125867 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:55:50 | INFO | fairseq.trainer | begin training epoch 225
2023-06-13 20:56:32 | INFO | train_inner | epoch 225:     79 / 282 loss=3.066, nll_loss=1.151, glat_accu=0.528, glat_context_p=0.458, word_ins=2.948, length=3.003, ppl=8.38, wps=90692.6, ups=1.51, wpb=60133.9, bsz=2131.8, num_updates=63200, lr=0.000125789, gnorm=0.523, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 20:57:17 | INFO | train_inner | epoch 225:    179 / 282 loss=3.084, nll_loss=1.168, glat_accu=0.546, glat_context_p=0.458, word_ins=2.964, length=3.015, ppl=8.48, wps=132549, ups=2.19, wpb=60548.6, bsz=2151, num_updates=63300, lr=0.000125689, gnorm=0.529, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:57:54 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 20:58:04 | INFO | train_inner | epoch 225:    280 / 282 loss=3.086, nll_loss=1.172, glat_accu=0.555, glat_context_p=0.458, word_ins=2.967, length=2.984, ppl=8.49, wps=129387, ups=2.14, wpb=60578, bsz=2195, num_updates=63400, lr=0.00012559, gnorm=0.538, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 20:58:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 20:58:08 | INFO | valid | epoch 225 | valid on 'valid' subset | loss 12.344 | nll_loss 11.179 | word_ins 12.115 | length 4.568 | ppl 5198.97 | bleu 31.05 | wps 88382.9 | wpb 21176.3 | bsz 666.3 | num_updates 63402 | best_bleu 31.2
2023-06-13 20:58:08 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 20:58:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint225.pt (epoch 225 @ 63402 updates, score 31.05) (writing took 11.814627408981323 seconds)
2023-06-13 20:58:20 | INFO | fairseq_cli.train | end of epoch 225 (average epoch stats below)
2023-06-13 20:58:20 | INFO | train | epoch 225 | loss 3.08 | nll_loss 1.165 | glat_accu 0.544 | glat_context_p 0.458 | word_ins 2.96 | length 3.002 | ppl 8.46 | wps 112733 | ups 1.87 | wpb 60411.9 | bsz 2155.9 | num_updates 63402 | lr 0.000125588 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 20:58:20 | INFO | fairseq.trainer | begin training epoch 226
2023-06-13 20:59:11 | INFO | train_inner | epoch 226:     98 / 282 loss=3.087, nll_loss=1.173, glat_accu=0.546, glat_context_p=0.458, word_ins=2.967, length=3.009, ppl=8.5, wps=89582.9, ups=1.49, wpb=60164.4, bsz=2100.9, num_updates=63500, lr=0.000125491, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 20:59:57 | INFO | train_inner | epoch 226:    198 / 282 loss=3.065, nll_loss=1.15, glat_accu=0.542, glat_context_p=0.458, word_ins=2.947, length=2.976, ppl=8.37, wps=133467, ups=2.2, wpb=60575.6, bsz=2196.5, num_updates=63600, lr=0.000125392, gnorm=0.522, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:00:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:00:39 | INFO | valid | epoch 226 | valid on 'valid' subset | loss 12.442 | nll_loss 11.28 | word_ins 12.212 | length 4.593 | ppl 5562.55 | bleu 30.73 | wps 88840 | wpb 21176.3 | bsz 666.3 | num_updates 63684 | best_bleu 31.2
2023-06-13 21:00:39 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:00:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint226.pt (epoch 226 @ 63684 updates, score 30.73) (writing took 12.912821840494871 seconds)
2023-06-13 21:00:51 | INFO | fairseq_cli.train | end of epoch 226 (average epoch stats below)
2023-06-13 21:00:51 | INFO | train | epoch 226 | loss 3.073 | nll_loss 1.158 | glat_accu 0.541 | glat_context_p 0.458 | word_ins 2.955 | length 2.994 | ppl 8.42 | wps 112532 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 63684 | lr 0.00012531 | gnorm 0.525 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:00:52 | INFO | fairseq.trainer | begin training epoch 227
2023-06-13 21:01:06 | INFO | train_inner | epoch 227:     16 / 282 loss=3.065, nll_loss=1.15, glat_accu=0.533, glat_context_p=0.458, word_ins=2.947, length=2.997, ppl=8.37, wps=86857.2, ups=1.45, wpb=59992.8, bsz=2162.6, num_updates=63700, lr=0.000125294, gnorm=0.522, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:01:52 | INFO | train_inner | epoch 227:    116 / 282 loss=3.069, nll_loss=1.154, glat_accu=0.547, glat_context_p=0.458, word_ins=2.95, length=2.981, ppl=8.39, wps=131491, ups=2.17, wpb=60533.4, bsz=2166.6, num_updates=63800, lr=0.000125196, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:02:38 | INFO | train_inner | epoch 227:    216 / 282 loss=3.077, nll_loss=1.161, glat_accu=0.551, glat_context_p=0.457, word_ins=2.957, length=2.992, ppl=8.44, wps=131777, ups=2.17, wpb=60594.9, bsz=2163.2, num_updates=63900, lr=0.000125098, gnorm=0.529, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:03:08 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:03:11 | INFO | valid | epoch 227 | valid on 'valid' subset | loss 12.33 | nll_loss 11.154 | word_ins 12.098 | length 4.628 | ppl 5148.88 | bleu 31.18 | wps 89487.6 | wpb 21176.3 | bsz 666.3 | num_updates 63966 | best_bleu 31.2
2023-06-13 21:03:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:03:24 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint227.pt (epoch 227 @ 63966 updates, score 31.18) (writing took 13.129946522414684 seconds)
2023-06-13 21:03:24 | INFO | fairseq_cli.train | end of epoch 227 (average epoch stats below)
2023-06-13 21:03:24 | INFO | train | epoch 227 | loss 3.073 | nll_loss 1.158 | glat_accu 0.546 | glat_context_p 0.457 | word_ins 2.954 | length 2.992 | ppl 8.42 | wps 111530 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 63966 | lr 0.000125033 | gnorm 0.526 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:03:24 | INFO | fairseq.trainer | begin training epoch 228
2023-06-13 21:03:45 | INFO | train_inner | epoch 228:     34 / 282 loss=3.08, nll_loss=1.165, glat_accu=0.544, glat_context_p=0.457, word_ins=2.96, length=3.017, ppl=8.46, wps=88952.2, ups=1.48, wpb=60057.6, bsz=2122.3, num_updates=64000, lr=0.000125, gnorm=0.532, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:04:32 | INFO | train_inner | epoch 228:    134 / 282 loss=3.074, nll_loss=1.159, glat_accu=0.544, glat_context_p=0.457, word_ins=2.955, length=2.991, ppl=8.42, wps=130617, ups=2.15, wpb=60733.9, bsz=2161.4, num_updates=64100, lr=0.000124902, gnorm=0.521, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:05:18 | INFO | train_inner | epoch 228:    234 / 282 loss=3.071, nll_loss=1.156, glat_accu=0.535, glat_context_p=0.457, word_ins=2.952, length=3.002, ppl=8.4, wps=131979, ups=2.18, wpb=60603.6, bsz=2149, num_updates=64200, lr=0.000124805, gnorm=0.528, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:05:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:05:43 | INFO | valid | epoch 228 | valid on 'valid' subset | loss 12.253 | nll_loss 11.06 | word_ins 12.019 | length 4.677 | ppl 4881.49 | bleu 30.72 | wps 88698.6 | wpb 21176.3 | bsz 666.3 | num_updates 64248 | best_bleu 31.2
2023-06-13 21:05:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:05:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint228.pt (epoch 228 @ 64248 updates, score 30.72) (writing took 12.911090187728405 seconds)
2023-06-13 21:05:56 | INFO | fairseq_cli.train | end of epoch 228 (average epoch stats below)
2023-06-13 21:05:56 | INFO | train | epoch 228 | loss 3.075 | nll_loss 1.16 | glat_accu 0.544 | glat_context_p 0.457 | word_ins 2.955 | length 2.996 | ppl 8.43 | wps 112152 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 64248 | lr 0.000124759 | gnorm 0.527 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:05:56 | INFO | fairseq.trainer | begin training epoch 229
2023-06-13 21:06:27 | INFO | train_inner | epoch 229:     52 / 282 loss=3.076, nll_loss=1.161, glat_accu=0.555, glat_context_p=0.457, word_ins=2.957, length=2.978, ppl=8.43, wps=86920.2, ups=1.45, wpb=60139, bsz=2184.5, num_updates=64300, lr=0.000124708, gnorm=0.542, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:07:12 | INFO | train_inner | epoch 229:    152 / 282 loss=3.072, nll_loss=1.156, glat_accu=0.554, glat_context_p=0.457, word_ins=2.953, length=2.984, ppl=8.41, wps=133254, ups=2.2, wpb=60528.3, bsz=2186.6, num_updates=64400, lr=0.000124611, gnorm=0.528, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:07:14 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:07:59 | INFO | train_inner | epoch 229:    253 / 282 loss=3.087, nll_loss=1.172, glat_accu=0.548, glat_context_p=0.457, word_ins=2.967, length=3.001, ppl=8.5, wps=129525, ups=2.14, wpb=60546.1, bsz=2122.1, num_updates=64500, lr=0.000124515, gnorm=0.545, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 21:08:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:08:16 | INFO | valid | epoch 229 | valid on 'valid' subset | loss 12.392 | nll_loss 11.225 | word_ins 12.165 | length 4.535 | ppl 5374.03 | bleu 31.08 | wps 86980.7 | wpb 21176.3 | bsz 666.3 | num_updates 64529 | best_bleu 31.2
2023-06-13 21:08:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:08:30 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint229.pt (epoch 229 @ 64529 updates, score 31.08) (writing took 13.834845162928104 seconds)
2023-06-13 21:08:30 | INFO | fairseq_cli.train | end of epoch 229 (average epoch stats below)
2023-06-13 21:08:30 | INFO | train | epoch 229 | loss 3.079 | nll_loss 1.164 | glat_accu 0.552 | glat_context_p 0.457 | word_ins 2.959 | length 2.994 | ppl 8.45 | wps 110654 | ups 1.83 | wpb 60405.7 | bsz 2156.6 | num_updates 64529 | lr 0.000124487 | gnorm 0.538 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:08:30 | INFO | fairseq.trainer | begin training epoch 230
2023-06-13 21:09:08 | INFO | train_inner | epoch 230:     71 / 282 loss=3.071, nll_loss=1.155, glat_accu=0.548, glat_context_p=0.457, word_ins=2.952, length=2.995, ppl=8.4, wps=87282.8, ups=1.45, wpb=60020.4, bsz=2153.5, num_updates=64600, lr=0.000124418, gnorm=0.522, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:09:55 | INFO | train_inner | epoch 230:    171 / 282 loss=3.075, nll_loss=1.159, glat_accu=0.544, glat_context_p=0.457, word_ins=2.955, length=3.012, ppl=8.43, wps=129684, ups=2.15, wpb=60449.4, bsz=2156.6, num_updates=64700, lr=0.000124322, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:10:41 | INFO | train_inner | epoch 230:    271 / 282 loss=3.074, nll_loss=1.16, glat_accu=0.551, glat_context_p=0.457, word_ins=2.956, length=2.971, ppl=8.42, wps=131994, ups=2.17, wpb=60876.3, bsz=2172.8, num_updates=64800, lr=0.000124226, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:10:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:10:49 | INFO | valid | epoch 230 | valid on 'valid' subset | loss 12.357 | nll_loss 11.192 | word_ins 12.132 | length 4.524 | ppl 5246.87 | bleu 31.08 | wps 88652.3 | wpb 21176.3 | bsz 666.3 | num_updates 64811 | best_bleu 31.2
2023-06-13 21:10:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:11:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint230.pt (epoch 230 @ 64811 updates, score 31.08) (writing took 13.48108047991991 seconds)
2023-06-13 21:11:02 | INFO | fairseq_cli.train | end of epoch 230 (average epoch stats below)
2023-06-13 21:11:02 | INFO | train | epoch 230 | loss 3.073 | nll_loss 1.157 | glat_accu 0.547 | glat_context_p 0.457 | word_ins 2.954 | length 2.992 | ppl 8.41 | wps 111455 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 64811 | lr 0.000124215 | gnorm 0.524 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:11:02 | INFO | fairseq.trainer | begin training epoch 231
2023-06-13 21:11:50 | INFO | train_inner | epoch 231:     89 / 282 loss=3.074, nll_loss=1.159, glat_accu=0.556, glat_context_p=0.457, word_ins=2.955, length=2.968, ppl=8.42, wps=86928.9, ups=1.45, wpb=60116, bsz=2190.5, num_updates=64900, lr=0.00012413, gnorm=0.538, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:12:36 | INFO | train_inner | epoch 231:    189 / 282 loss=3.073, nll_loss=1.157, glat_accu=0.545, glat_context_p=0.457, word_ins=2.953, length=3.011, ppl=8.42, wps=131587, ups=2.17, wpb=60631.1, bsz=2131.6, num_updates=65000, lr=0.000124035, gnorm=0.54, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:13:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:13:22 | INFO | valid | epoch 231 | valid on 'valid' subset | loss 12.505 | nll_loss 11.347 | word_ins 12.276 | length 4.582 | ppl 5814.67 | bleu 30.99 | wps 88882.3 | wpb 21176.3 | bsz 666.3 | num_updates 65093 | best_bleu 31.2
2023-06-13 21:13:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:13:35 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint231.pt (epoch 231 @ 65093 updates, score 30.99) (writing took 12.612241223454475 seconds)
2023-06-13 21:13:35 | INFO | fairseq_cli.train | end of epoch 231 (average epoch stats below)
2023-06-13 21:13:35 | INFO | train | epoch 231 | loss 3.074 | nll_loss 1.159 | glat_accu 0.547 | glat_context_p 0.457 | word_ins 2.955 | length 2.989 | ppl 8.42 | wps 111773 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 65093 | lr 0.000123946 | gnorm 0.537 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:13:35 | INFO | fairseq.trainer | begin training epoch 232
2023-06-13 21:13:45 | INFO | train_inner | epoch 232:      7 / 282 loss=3.076, nll_loss=1.161, glat_accu=0.541, glat_context_p=0.457, word_ins=2.957, length=2.998, ppl=8.43, wps=87035.1, ups=1.45, wpb=59932.7, bsz=2125, num_updates=65100, lr=0.000123939, gnorm=0.542, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:14:31 | INFO | train_inner | epoch 232:    107 / 282 loss=3.065, nll_loss=1.15, glat_accu=0.545, glat_context_p=0.457, word_ins=2.947, length=2.981, ppl=8.37, wps=131191, ups=2.16, wpb=60714.8, bsz=2202.4, num_updates=65200, lr=0.000123844, gnorm=0.534, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:15:18 | INFO | train_inner | epoch 232:    207 / 282 loss=3.064, nll_loss=1.148, glat_accu=0.53, glat_context_p=0.457, word_ins=2.946, length=3.011, ppl=8.36, wps=129485, ups=2.14, wpb=60429.3, bsz=2143.8, num_updates=65300, lr=0.000123749, gnorm=0.52, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 21:15:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:15:55 | INFO | valid | epoch 232 | valid on 'valid' subset | loss 12.601 | nll_loss 11.453 | word_ins 12.368 | length 4.659 | ppl 6210.74 | bleu 30.59 | wps 86842.5 | wpb 21176.3 | bsz 666.3 | num_updates 65375 | best_bleu 31.2
2023-06-13 21:15:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:16:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint232.pt (epoch 232 @ 65375 updates, score 30.59) (writing took 13.056882247328758 seconds)
2023-06-13 21:16:08 | INFO | fairseq_cli.train | end of epoch 232 (average epoch stats below)
2023-06-13 21:16:08 | INFO | train | epoch 232 | loss 3.068 | nll_loss 1.152 | glat_accu 0.539 | glat_context_p 0.457 | word_ins 2.949 | length 2.995 | ppl 8.38 | wps 111214 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 65375 | lr 0.000123678 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-13 21:16:08 | INFO | fairseq.trainer | begin training epoch 233
2023-06-13 21:16:26 | INFO | train_inner | epoch 233:     25 / 282 loss=3.074, nll_loss=1.159, glat_accu=0.544, glat_context_p=0.456, word_ins=2.955, length=2.984, ppl=8.42, wps=88522.5, ups=1.47, wpb=60221.7, bsz=2124.9, num_updates=65400, lr=0.000123655, gnorm=0.542, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:16:38 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:17:12 | INFO | train_inner | epoch 233:    126 / 282 loss=3.062, nll_loss=1.147, glat_accu=0.548, glat_context_p=0.456, word_ins=2.944, length=2.963, ppl=8.35, wps=130886, ups=2.16, wpb=60652.7, bsz=2215.3, num_updates=65500, lr=0.00012356, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:17:58 | INFO | train_inner | epoch 233:    226 / 282 loss=3.073, nll_loss=1.157, glat_accu=0.546, glat_context_p=0.456, word_ins=2.953, length=3.008, ppl=8.42, wps=131144, ups=2.17, wpb=60434.6, bsz=2143, num_updates=65600, lr=0.000123466, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:18:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:18:27 | INFO | valid | epoch 233 | valid on 'valid' subset | loss 12.425 | nll_loss 11.258 | word_ins 12.191 | length 4.696 | ppl 5498.67 | bleu 31.19 | wps 88229.5 | wpb 21176.3 | bsz 666.3 | num_updates 65656 | best_bleu 31.2
2023-06-13 21:18:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:18:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint233.pt (epoch 233 @ 65656 updates, score 31.19) (writing took 13.112415101379156 seconds)
2023-06-13 21:18:40 | INFO | fairseq_cli.train | end of epoch 233 (average epoch stats below)
2023-06-13 21:18:40 | INFO | train | epoch 233 | loss 3.071 | nll_loss 1.156 | glat_accu 0.546 | glat_context_p 0.456 | word_ins 2.952 | length 2.994 | ppl 8.4 | wps 111567 | ups 1.85 | wpb 60409.7 | bsz 2155.3 | num_updates 65656 | lr 0.000123414 | gnorm 0.535 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:18:40 | INFO | fairseq.trainer | begin training epoch 234
2023-06-13 21:19:07 | INFO | train_inner | epoch 234:     44 / 282 loss=3.077, nll_loss=1.161, glat_accu=0.55, glat_context_p=0.456, word_ins=2.957, length=3.005, ppl=8.44, wps=87646.3, ups=1.46, wpb=60127.5, bsz=2134.9, num_updates=65700, lr=0.000123372, gnorm=0.537, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:19:52 | INFO | train_inner | epoch 234:    144 / 282 loss=3.072, nll_loss=1.156, glat_accu=0.552, glat_context_p=0.456, word_ins=2.953, length=2.995, ppl=8.41, wps=132889, ups=2.2, wpb=60500.8, bsz=2134, num_updates=65800, lr=0.000123278, gnorm=0.539, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:20:39 | INFO | train_inner | epoch 234:    244 / 282 loss=3.08, nll_loss=1.166, glat_accu=0.553, glat_context_p=0.456, word_ins=2.961, length=2.989, ppl=8.46, wps=130769, ups=2.16, wpb=60601.4, bsz=2170.3, num_updates=65900, lr=0.000123185, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:20:56 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:20:59 | INFO | valid | epoch 234 | valid on 'valid' subset | loss 12.374 | nll_loss 11.208 | word_ins 12.147 | length 4.539 | ppl 5309.62 | bleu 31.03 | wps 89315.3 | wpb 21176.3 | bsz 666.3 | num_updates 65938 | best_bleu 31.2
2023-06-13 21:20:59 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:21:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint234.pt (epoch 234 @ 65938 updates, score 31.03) (writing took 12.990432489663363 seconds)
2023-06-13 21:21:12 | INFO | fairseq_cli.train | end of epoch 234 (average epoch stats below)
2023-06-13 21:21:12 | INFO | train | epoch 234 | loss 3.074 | nll_loss 1.159 | glat_accu 0.552 | glat_context_p 0.456 | word_ins 2.955 | length 2.99 | ppl 8.42 | wps 111960 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 65938 | lr 0.000123149 | gnorm 0.536 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:21:12 | INFO | fairseq.trainer | begin training epoch 235
2023-06-13 21:21:48 | INFO | train_inner | epoch 235:     62 / 282 loss=3.071, nll_loss=1.155, glat_accu=0.554, glat_context_p=0.456, word_ins=2.951, length=2.98, ppl=8.4, wps=87145.7, ups=1.45, wpb=60136.7, bsz=2192.2, num_updates=66000, lr=0.000123091, gnorm=0.539, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:22:34 | INFO | train_inner | epoch 235:    162 / 282 loss=3.078, nll_loss=1.163, glat_accu=0.549, glat_context_p=0.456, word_ins=2.958, length=2.999, ppl=8.44, wps=131935, ups=2.18, wpb=60635.9, bsz=2124.6, num_updates=66100, lr=0.000122998, gnorm=0.53, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:23:20 | INFO | train_inner | epoch 235:    262 / 282 loss=3.063, nll_loss=1.147, glat_accu=0.542, glat_context_p=0.456, word_ins=2.944, length=2.978, ppl=8.36, wps=131713, ups=2.18, wpb=60501.2, bsz=2161.8, num_updates=66200, lr=0.000122905, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:23:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:23:32 | INFO | valid | epoch 235 | valid on 'valid' subset | loss 12.36 | nll_loss 11.189 | word_ins 12.129 | length 4.606 | ppl 5255.79 | bleu 31.02 | wps 86768.6 | wpb 21176.3 | bsz 666.3 | num_updates 66220 | best_bleu 31.2
2023-06-13 21:23:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:23:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint235.pt (epoch 235 @ 66220 updates, score 31.02) (writing took 12.286422785371542 seconds)
2023-06-13 21:23:44 | INFO | fairseq_cli.train | end of epoch 235 (average epoch stats below)
2023-06-13 21:23:44 | INFO | train | epoch 235 | loss 3.071 | nll_loss 1.156 | glat_accu 0.548 | glat_context_p 0.456 | word_ins 2.952 | length 2.986 | ppl 8.4 | wps 112278 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 66220 | lr 0.000122887 | gnorm 0.529 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:23:44 | INFO | fairseq.trainer | begin training epoch 236
2023-06-13 21:24:27 | INFO | train_inner | epoch 236:     80 / 282 loss=3.079, nll_loss=1.164, glat_accu=0.549, glat_context_p=0.456, word_ins=2.959, length=2.993, ppl=8.45, wps=89359.9, ups=1.49, wpb=60174.2, bsz=2142.3, num_updates=66300, lr=0.000122813, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:25:13 | INFO | train_inner | epoch 236:    180 / 282 loss=3.068, nll_loss=1.153, glat_accu=0.553, glat_context_p=0.456, word_ins=2.949, length=2.97, ppl=8.39, wps=131572, ups=2.17, wpb=60583.4, bsz=2180.4, num_updates=66400, lr=0.00012272, gnorm=0.536, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:25:36 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:25:59 | INFO | train_inner | epoch 236:    281 / 282 loss=3.071, nll_loss=1.156, glat_accu=0.544, glat_context_p=0.456, word_ins=2.952, length=2.998, ppl=8.4, wps=130851, ups=2.16, wpb=60580.4, bsz=2147.2, num_updates=66500, lr=0.000122628, gnorm=0.529, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:26:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:26:03 | INFO | valid | epoch 236 | valid on 'valid' subset | loss 12.368 | nll_loss 11.21 | word_ins 12.145 | length 4.454 | ppl 5286.23 | bleu 31.21 | wps 88584.8 | wpb 21176.3 | bsz 666.3 | num_updates 66501 | best_bleu 31.21
2023-06-13 21:26:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:26:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint236.pt (epoch 236 @ 66501 updates, score 31.21) (writing took 16.461084462702274 seconds)
2023-06-13 21:26:19 | INFO | fairseq_cli.train | end of epoch 236 (average epoch stats below)
2023-06-13 21:26:19 | INFO | train | epoch 236 | loss 3.072 | nll_loss 1.157 | glat_accu 0.548 | glat_context_p 0.456 | word_ins 2.953 | length 2.986 | ppl 8.41 | wps 109349 | ups 1.81 | wpb 60410.1 | bsz 2156.4 | num_updates 66501 | lr 0.000122627 | gnorm 0.534 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:26:19 | INFO | fairseq.trainer | begin training epoch 237
2023-06-13 21:27:11 | INFO | train_inner | epoch 237:     99 / 282 loss=3.063, nll_loss=1.148, glat_accu=0.542, glat_context_p=0.456, word_ins=2.945, length=2.974, ppl=8.36, wps=84484.8, ups=1.4, wpb=60221.2, bsz=2150.6, num_updates=66600, lr=0.000122536, gnorm=0.532, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:27:57 | INFO | train_inner | epoch 237:    199 / 282 loss=3.064, nll_loss=1.148, glat_accu=0.536, glat_context_p=0.456, word_ins=2.945, length=3.003, ppl=8.36, wps=130289, ups=2.15, wpb=60507.9, bsz=2130.6, num_updates=66700, lr=0.000122444, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:28:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:28:38 | INFO | valid | epoch 237 | valid on 'valid' subset | loss 12.573 | nll_loss 11.422 | word_ins 12.336 | length 4.74 | ppl 6092.68 | bleu 30.45 | wps 88679.2 | wpb 21176.3 | bsz 666.3 | num_updates 66783 | best_bleu 31.21
2023-06-13 21:28:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:28:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint237.pt (epoch 237 @ 66783 updates, score 30.45) (writing took 11.149711955338717 seconds)
2023-06-13 21:28:49 | INFO | fairseq_cli.train | end of epoch 237 (average epoch stats below)
2023-06-13 21:28:49 | INFO | train | epoch 237 | loss 3.065 | nll_loss 1.15 | glat_accu 0.543 | glat_context_p 0.456 | word_ins 2.946 | length 2.983 | ppl 8.37 | wps 113539 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 66783 | lr 0.000122368 | gnorm 0.528 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:28:49 | INFO | fairseq.trainer | begin training epoch 238
2023-06-13 21:29:03 | INFO | train_inner | epoch 238:     17 / 282 loss=3.067, nll_loss=1.152, glat_accu=0.55, glat_context_p=0.456, word_ins=2.948, length=2.975, ppl=8.38, wps=90847.9, ups=1.51, wpb=60063, bsz=2169, num_updates=66800, lr=0.000122352, gnorm=0.537, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:29:48 | INFO | train_inner | epoch 238:    117 / 282 loss=3.059, nll_loss=1.143, glat_accu=0.543, glat_context_p=0.455, word_ins=2.94, length=2.975, ppl=8.33, wps=134520, ups=2.22, wpb=60687.9, bsz=2195.8, num_updates=66900, lr=0.000122261, gnorm=0.52, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:30:34 | INFO | train_inner | epoch 238:    217 / 282 loss=3.069, nll_loss=1.153, glat_accu=0.548, glat_context_p=0.455, word_ins=2.949, length=3.008, ppl=8.39, wps=130576, ups=2.16, wpb=60400.5, bsz=2157.2, num_updates=67000, lr=0.000122169, gnorm=0.539, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:31:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:31:07 | INFO | valid | epoch 238 | valid on 'valid' subset | loss 12.326 | nll_loss 11.155 | word_ins 12.098 | length 4.558 | ppl 5133.17 | bleu 31.27 | wps 88287.2 | wpb 21176.3 | bsz 666.3 | num_updates 67065 | best_bleu 31.27
2023-06-13 21:31:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:31:24 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint238.pt (epoch 238 @ 67065 updates, score 31.27) (writing took 17.046481136232615 seconds)
2023-06-13 21:31:24 | INFO | fairseq_cli.train | end of epoch 238 (average epoch stats below)
2023-06-13 21:31:24 | INFO | train | epoch 238 | loss 3.068 | nll_loss 1.152 | glat_accu 0.547 | glat_context_p 0.455 | word_ins 2.949 | length 2.989 | ppl 8.39 | wps 109836 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 67065 | lr 0.00012211 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:31:25 | INFO | fairseq.trainer | begin training epoch 239
2023-06-13 21:31:47 | INFO | train_inner | epoch 239:     35 / 282 loss=3.08, nll_loss=1.166, glat_accu=0.559, glat_context_p=0.455, word_ins=2.961, length=2.977, ppl=8.46, wps=82670.7, ups=1.37, wpb=60247, bsz=2128.9, num_updates=67100, lr=0.000122078, gnorm=0.553, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:32:33 | INFO | train_inner | epoch 239:    135 / 282 loss=3.069, nll_loss=1.154, glat_accu=0.55, glat_context_p=0.455, word_ins=2.95, length=2.979, ppl=8.39, wps=132040, ups=2.18, wpb=60682.2, bsz=2187.8, num_updates=67200, lr=0.000121988, gnorm=0.528, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:33:19 | INFO | train_inner | epoch 239:    235 / 282 loss=3.067, nll_loss=1.151, glat_accu=0.549, glat_context_p=0.455, word_ins=2.948, length=2.994, ppl=8.38, wps=131639, ups=2.18, wpb=60370.2, bsz=2150.6, num_updates=67300, lr=0.000121897, gnorm=0.537, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:33:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:33:44 | INFO | valid | epoch 239 | valid on 'valid' subset | loss 12.435 | nll_loss 11.274 | word_ins 12.205 | length 4.601 | ppl 5536.16 | bleu 31 | wps 89166.7 | wpb 21176.3 | bsz 666.3 | num_updates 67347 | best_bleu 31.27
2023-06-13 21:33:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:33:54 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint239.pt (epoch 239 @ 67347 updates, score 31.0) (writing took 9.880988638848066 seconds)
2023-06-13 21:33:54 | INFO | fairseq_cli.train | end of epoch 239 (average epoch stats below)
2023-06-13 21:33:54 | INFO | train | epoch 239 | loss 3.07 | nll_loss 1.154 | glat_accu 0.551 | glat_context_p 0.455 | word_ins 2.95 | length 2.988 | ppl 8.4 | wps 114033 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 67347 | lr 0.000121854 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:33:54 | INFO | fairseq.trainer | begin training epoch 240
2023-06-13 21:34:25 | INFO | train_inner | epoch 240:     53 / 282 loss=3.071, nll_loss=1.156, glat_accu=0.551, glat_context_p=0.455, word_ins=2.952, length=2.987, ppl=8.4, wps=91328.8, ups=1.52, wpb=60065.3, bsz=2154.9, num_updates=67400, lr=0.000121806, gnorm=0.554, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:34:59 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:35:11 | INFO | train_inner | epoch 240:    154 / 282 loss=3.064, nll_loss=1.148, glat_accu=0.547, glat_context_p=0.455, word_ins=2.945, length=2.987, ppl=8.36, wps=130948, ups=2.16, wpb=60693.2, bsz=2181, num_updates=67500, lr=0.000121716, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:35:58 | INFO | train_inner | epoch 240:    254 / 282 loss=3.064, nll_loss=1.149, glat_accu=0.536, glat_context_p=0.455, word_ins=2.945, length=3.003, ppl=8.36, wps=130171, ups=2.15, wpb=60509.2, bsz=2126.4, num_updates=67600, lr=0.000121626, gnorm=0.527, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:36:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:36:14 | INFO | valid | epoch 240 | valid on 'valid' subset | loss 12.368 | nll_loss 11.202 | word_ins 12.139 | length 4.576 | ppl 5287.12 | bleu 31.07 | wps 88660 | wpb 21176.3 | bsz 666.3 | num_updates 67628 | best_bleu 31.27
2023-06-13 21:36:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:36:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint240.pt (epoch 240 @ 67628 updates, score 31.07) (writing took 11.891325697302818 seconds)
2023-06-13 21:36:26 | INFO | fairseq_cli.train | end of epoch 240 (average epoch stats below)
2023-06-13 21:36:26 | INFO | train | epoch 240 | loss 3.065 | nll_loss 1.149 | glat_accu 0.544 | glat_context_p 0.455 | word_ins 2.946 | length 2.988 | ppl 8.37 | wps 111830 | ups 1.85 | wpb 60409.6 | bsz 2156 | num_updates 67628 | lr 0.000121601 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:36:26 | INFO | fairseq.trainer | begin training epoch 241
2023-06-13 21:37:06 | INFO | train_inner | epoch 241:     72 / 282 loss=3.06, nll_loss=1.143, glat_accu=0.539, glat_context_p=0.455, word_ins=2.94, length=3.003, ppl=8.34, wps=87936.3, ups=1.47, wpb=59976.1, bsz=2109.9, num_updates=67700, lr=0.000121536, gnorm=0.544, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:37:52 | INFO | train_inner | epoch 241:    172 / 282 loss=3.058, nll_loss=1.143, glat_accu=0.542, glat_context_p=0.455, word_ins=2.94, length=2.982, ppl=8.33, wps=132372, ups=2.19, wpb=60537, bsz=2172.4, num_updates=67800, lr=0.000121447, gnorm=0.519, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:38:37 | INFO | train_inner | epoch 241:    272 / 282 loss=3.064, nll_loss=1.149, glat_accu=0.551, glat_context_p=0.455, word_ins=2.945, length=2.978, ppl=8.36, wps=133112, ups=2.19, wpb=60796.4, bsz=2184.5, num_updates=67900, lr=0.000121357, gnorm=0.528, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:38:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:38:45 | INFO | valid | epoch 241 | valid on 'valid' subset | loss 12.47 | nll_loss 11.312 | word_ins 12.244 | length 4.53 | ppl 5674.18 | bleu 31.18 | wps 88735.3 | wpb 21176.3 | bsz 666.3 | num_updates 67910 | best_bleu 31.27
2023-06-13 21:38:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:38:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint241.pt (epoch 241 @ 67910 updates, score 31.18) (writing took 7.926344089210033 seconds)
2023-06-13 21:38:53 | INFO | fairseq_cli.train | end of epoch 241 (average epoch stats below)
2023-06-13 21:38:53 | INFO | train | epoch 241 | loss 3.061 | nll_loss 1.145 | glat_accu 0.545 | glat_context_p 0.455 | word_ins 2.942 | length 2.989 | ppl 8.35 | wps 115719 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 67910 | lr 0.000121348 | gnorm 0.532 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:38:53 | INFO | fairseq.trainer | begin training epoch 242
2023-06-13 21:39:39 | INFO | train_inner | epoch 242:     90 / 282 loss=3.067, nll_loss=1.15, glat_accu=0.545, glat_context_p=0.455, word_ins=2.947, length=3.007, ppl=8.38, wps=96934.3, ups=1.61, wpb=60098.1, bsz=2109.6, num_updates=68000, lr=0.000121268, gnorm=0.536, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:40:25 | INFO | train_inner | epoch 242:    190 / 282 loss=3.062, nll_loss=1.146, glat_accu=0.551, glat_context_p=0.455, word_ins=2.943, length=2.975, ppl=8.35, wps=132314, ups=2.18, wpb=60662.1, bsz=2195.2, num_updates=68100, lr=0.000121179, gnorm=0.534, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:41:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:41:11 | INFO | valid | epoch 242 | valid on 'valid' subset | loss 12.473 | nll_loss 11.305 | word_ins 12.237 | length 4.73 | ppl 5684.79 | bleu 31.01 | wps 87643.7 | wpb 21176.3 | bsz 666.3 | num_updates 68192 | best_bleu 31.27
2023-06-13 21:41:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:41:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint242.pt (epoch 242 @ 68192 updates, score 31.01) (writing took 12.200516194105148 seconds)
2023-06-13 21:41:23 | INFO | fairseq_cli.train | end of epoch 242 (average epoch stats below)
2023-06-13 21:41:23 | INFO | train | epoch 242 | loss 3.061 | nll_loss 1.145 | glat_accu 0.547 | glat_context_p 0.455 | word_ins 2.942 | length 2.987 | ppl 8.35 | wps 113637 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 68192 | lr 0.000121097 | gnorm 0.533 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:41:23 | INFO | fairseq.trainer | begin training epoch 243
2023-06-13 21:41:33 | INFO | train_inner | epoch 243:      8 / 282 loss=3.058, nll_loss=1.142, glat_accu=0.544, glat_context_p=0.455, word_ins=2.939, length=2.985, ppl=8.33, wps=88352.1, ups=1.47, wpb=60009.8, bsz=2143.5, num_updates=68200, lr=0.00012109, gnorm=0.536, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:42:19 | INFO | train_inner | epoch 243:    108 / 282 loss=3.057, nll_loss=1.141, glat_accu=0.538, glat_context_p=0.455, word_ins=2.938, length=2.979, ppl=8.32, wps=133427, ups=2.2, wpb=60638.7, bsz=2152.4, num_updates=68300, lr=0.000121001, gnorm=0.529, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:43:04 | INFO | train_inner | epoch 243:    208 / 282 loss=3.067, nll_loss=1.151, glat_accu=0.552, glat_context_p=0.454, word_ins=2.947, length=2.985, ppl=8.38, wps=132335, ups=2.19, wpb=60530.5, bsz=2196.6, num_updates=68400, lr=0.000120913, gnorm=0.527, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:43:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:43:42 | INFO | valid | epoch 243 | valid on 'valid' subset | loss 12.362 | nll_loss 11.184 | word_ins 12.127 | length 4.701 | ppl 5264.72 | bleu 30.9 | wps 89393.9 | wpb 21176.3 | bsz 666.3 | num_updates 68474 | best_bleu 31.27
2023-06-13 21:43:42 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:43:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint243.pt (epoch 243 @ 68474 updates, score 30.9) (writing took 9.68847781792283 seconds)
2023-06-13 21:43:51 | INFO | fairseq_cli.train | end of epoch 243 (average epoch stats below)
2023-06-13 21:43:51 | INFO | train | epoch 243 | loss 3.065 | nll_loss 1.149 | glat_accu 0.545 | glat_context_p 0.454 | word_ins 2.946 | length 2.986 | ppl 8.37 | wps 114694 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 68474 | lr 0.000120847 | gnorm 0.528 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:43:51 | INFO | fairseq.trainer | begin training epoch 244
2023-06-13 21:44:09 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:44:10 | INFO | train_inner | epoch 244:     27 / 282 loss=3.07, nll_loss=1.155, glat_accu=0.548, glat_context_p=0.454, word_ins=2.951, length=2.991, ppl=8.4, wps=91754.9, ups=1.53, wpb=60111.4, bsz=2134.7, num_updates=68500, lr=0.000120824, gnorm=0.526, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:44:56 | INFO | train_inner | epoch 244:    127 / 282 loss=3.066, nll_loss=1.151, glat_accu=0.558, glat_context_p=0.454, word_ins=2.947, length=2.959, ppl=8.38, wps=132025, ups=2.18, wpb=60586.3, bsz=2173, num_updates=68600, lr=0.000120736, gnorm=0.541, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:45:42 | INFO | train_inner | epoch 244:    227 / 282 loss=3.076, nll_loss=1.16, glat_accu=0.553, glat_context_p=0.454, word_ins=2.956, length=2.998, ppl=8.43, wps=131215, ups=2.17, wpb=60552.1, bsz=2153.3, num_updates=68700, lr=0.000120648, gnorm=0.539, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:46:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:46:10 | INFO | valid | epoch 244 | valid on 'valid' subset | loss 12.476 | nll_loss 11.323 | word_ins 12.251 | length 4.516 | ppl 5698.33 | bleu 31.15 | wps 88577.3 | wpb 21176.3 | bsz 666.3 | num_updates 68755 | best_bleu 31.27
2023-06-13 21:46:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:46:17 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint244.pt (epoch 244 @ 68755 updates, score 31.15) (writing took 7.093511361628771 seconds)
2023-06-13 21:46:17 | INFO | fairseq_cli.train | end of epoch 244 (average epoch stats below)
2023-06-13 21:46:17 | INFO | train | epoch 244 | loss 3.071 | nll_loss 1.155 | glat_accu 0.553 | glat_context_p 0.454 | word_ins 2.951 | length 2.985 | ppl 8.4 | wps 116155 | ups 1.92 | wpb 60413.5 | bsz 2157.3 | num_updates 68755 | lr 0.0001206 | gnorm 0.541 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:46:18 | INFO | fairseq.trainer | begin training epoch 245
2023-06-13 21:46:43 | INFO | train_inner | epoch 245:     45 / 282 loss=3.071, nll_loss=1.155, glat_accu=0.544, glat_context_p=0.454, word_ins=2.951, length=3.012, ppl=8.41, wps=97883.3, ups=1.63, wpb=60085.1, bsz=2121.6, num_updates=68800, lr=0.000120561, gnorm=0.553, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:47:29 | INFO | train_inner | epoch 245:    145 / 282 loss=3.069, nll_loss=1.155, glat_accu=0.554, glat_context_p=0.454, word_ins=2.951, length=2.958, ppl=8.39, wps=134252, ups=2.21, wpb=60825, bsz=2201.7, num_updates=68900, lr=0.000120473, gnorm=0.538, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:48:14 | INFO | train_inner | epoch 245:    245 / 282 loss=3.073, nll_loss=1.157, glat_accu=0.552, glat_context_p=0.454, word_ins=2.953, length=2.994, ppl=8.42, wps=131782, ups=2.18, wpb=60415, bsz=2122.1, num_updates=69000, lr=0.000120386, gnorm=0.548, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:48:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:48:35 | INFO | valid | epoch 245 | valid on 'valid' subset | loss 12.534 | nll_loss 11.389 | word_ins 12.307 | length 4.521 | ppl 5930.36 | bleu 30.71 | wps 88338.5 | wpb 21176.3 | bsz 666.3 | num_updates 69037 | best_bleu 31.27
2023-06-13 21:48:35 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:48:46 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint245.pt (epoch 245 @ 69037 updates, score 30.71) (writing took 11.67389077320695 seconds)
2023-06-13 21:48:46 | INFO | fairseq_cli.train | end of epoch 245 (average epoch stats below)
2023-06-13 21:48:46 | INFO | train | epoch 245 | loss 3.07 | nll_loss 1.155 | glat_accu 0.553 | glat_context_p 0.454 | word_ins 2.951 | length 2.983 | ppl 8.4 | wps 114423 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 69037 | lr 0.000120354 | gnorm 0.544 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:48:46 | INFO | fairseq.trainer | begin training epoch 246
2023-06-13 21:49:22 | INFO | train_inner | epoch 246:     63 / 282 loss=3.059, nll_loss=1.144, glat_accu=0.547, glat_context_p=0.454, word_ins=2.941, length=2.972, ppl=8.34, wps=89518.8, ups=1.49, wpb=60054.2, bsz=2182.7, num_updates=69100, lr=0.000120299, gnorm=0.53, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:50:07 | INFO | train_inner | epoch 246:    163 / 282 loss=3.066, nll_loss=1.15, glat_accu=0.555, glat_context_p=0.454, word_ins=2.946, length=2.986, ppl=8.38, wps=131801, ups=2.18, wpb=60541.9, bsz=2168.1, num_updates=69200, lr=0.000120212, gnorm=0.537, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:50:53 | INFO | train_inner | epoch 246:    263 / 282 loss=3.077, nll_loss=1.162, glat_accu=0.555, glat_context_p=0.454, word_ins=2.957, length=2.987, ppl=8.44, wps=132632, ups=2.19, wpb=60646.2, bsz=2134.1, num_updates=69300, lr=0.000120125, gnorm=0.541, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:51:02 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:51:05 | INFO | valid | epoch 246 | valid on 'valid' subset | loss 12.478 | nll_loss 11.33 | word_ins 12.254 | length 4.447 | ppl 5704.14 | bleu 31.39 | wps 87952.8 | wpb 21176.3 | bsz 666.3 | num_updates 69319 | best_bleu 31.39
2023-06-13 21:51:05 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:51:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint246.pt (epoch 246 @ 69319 updates, score 31.39) (writing took 17.817635241895914 seconds)
2023-06-13 21:51:23 | INFO | fairseq_cli.train | end of epoch 246 (average epoch stats below)
2023-06-13 21:51:23 | INFO | train | epoch 246 | loss 3.068 | nll_loss 1.152 | glat_accu 0.551 | glat_context_p 0.454 | word_ins 2.948 | length 2.983 | ppl 8.39 | wps 108801 | ups 1.8 | wpb 60413.8 | bsz 2157.2 | num_updates 69319 | lr 0.000120109 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:51:23 | INFO | fairseq.trainer | begin training epoch 247
2023-06-13 21:52:07 | INFO | train_inner | epoch 247:     81 / 282 loss=3.065, nll_loss=1.149, glat_accu=0.542, glat_context_p=0.454, word_ins=2.946, length=2.994, ppl=8.37, wps=81553.9, ups=1.36, wpb=60177.7, bsz=2154.2, num_updates=69400, lr=0.000120038, gnorm=0.537, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:52:52 | INFO | train_inner | epoch 247:    181 / 282 loss=3.066, nll_loss=1.15, glat_accu=0.555, glat_context_p=0.454, word_ins=2.947, length=2.975, ppl=8.38, wps=133802, ups=2.21, wpb=60670, bsz=2177.4, num_updates=69500, lr=0.000119952, gnorm=0.544, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:53:03 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 21:53:38 | INFO | train_inner | epoch 247:    282 / 282 loss=3.07, nll_loss=1.154, glat_accu=0.556, glat_context_p=0.454, word_ins=2.95, length=2.989, ppl=8.4, wps=129913, ups=2.17, wpb=59949.9, bsz=2131.2, num_updates=69600, lr=0.000119866, gnorm=0.542, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:53:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:53:42 | INFO | valid | epoch 247 | valid on 'valid' subset | loss 12.431 | nll_loss 11.271 | word_ins 12.201 | length 4.585 | ppl 5522.07 | bleu 31.41 | wps 89285.9 | wpb 21176.3 | bsz 666.3 | num_updates 69600 | best_bleu 31.41
2023-06-13 21:53:42 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:53:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint247.pt (epoch 247 @ 69600 updates, score 31.41) (writing took 13.847046088427305 seconds)
2023-06-13 21:53:55 | INFO | fairseq_cli.train | end of epoch 247 (average epoch stats below)
2023-06-13 21:53:55 | INFO | train | epoch 247 | loss 3.066 | nll_loss 1.151 | glat_accu 0.551 | glat_context_p 0.454 | word_ins 2.947 | length 2.985 | ppl 8.38 | wps 111296 | ups 1.84 | wpb 60407 | bsz 2157.2 | num_updates 69600 | lr 0.000119866 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:53:56 | INFO | fairseq.trainer | begin training epoch 248
2023-06-13 21:54:47 | INFO | train_inner | epoch 248:    100 / 282 loss=3.061, nll_loss=1.145, glat_accu=0.544, glat_context_p=0.454, word_ins=2.942, length=2.981, ppl=8.34, wps=87566.1, ups=1.45, wpb=60438.4, bsz=2149.6, num_updates=69700, lr=0.00011978, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:55:33 | INFO | train_inner | epoch 248:    200 / 282 loss=3.065, nll_loss=1.149, glat_accu=0.553, glat_context_p=0.454, word_ins=2.945, length=2.971, ppl=8.37, wps=133181, ups=2.2, wpb=60671.5, bsz=2194.4, num_updates=69800, lr=0.000119694, gnorm=0.529, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:56:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:56:13 | INFO | valid | epoch 248 | valid on 'valid' subset | loss 12.408 | nll_loss 11.244 | word_ins 12.177 | length 4.619 | ppl 5436.46 | bleu 31.27 | wps 88466.8 | wpb 21176.3 | bsz 666.3 | num_updates 69882 | best_bleu 31.41
2023-06-13 21:56:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:56:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint248.pt (epoch 248 @ 69882 updates, score 31.27) (writing took 9.239027563482523 seconds)
2023-06-13 21:56:23 | INFO | fairseq_cli.train | end of epoch 248 (average epoch stats below)
2023-06-13 21:56:23 | INFO | train | epoch 248 | loss 3.065 | nll_loss 1.149 | glat_accu 0.55 | glat_context_p 0.454 | word_ins 2.946 | length 2.976 | ppl 8.37 | wps 115701 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 69882 | lr 0.000119624 | gnorm 0.537 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 21:56:23 | INFO | fairseq.trainer | begin training epoch 249
2023-06-13 21:56:37 | INFO | train_inner | epoch 249:     18 / 282 loss=3.068, nll_loss=1.153, glat_accu=0.553, glat_context_p=0.453, word_ins=2.949, length=2.974, ppl=8.39, wps=93945.8, ups=1.56, wpb=60143.5, bsz=2152.8, num_updates=69900, lr=0.000119608, gnorm=0.541, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 21:57:24 | INFO | train_inner | epoch 249:    118 / 282 loss=3.071, nll_loss=1.155, glat_accu=0.558, glat_context_p=0.453, word_ins=2.951, length=2.984, ppl=8.4, wps=130344, ups=2.15, wpb=60576.1, bsz=2141.4, num_updates=70000, lr=0.000119523, gnorm=0.535, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:58:09 | INFO | train_inner | epoch 249:    218 / 282 loss=3.072, nll_loss=1.157, glat_accu=0.547, glat_context_p=0.453, word_ins=2.952, length=2.992, ppl=8.41, wps=132260, ups=2.19, wpb=60523.8, bsz=2113.8, num_updates=70100, lr=0.000119438, gnorm=0.557, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 21:58:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 21:58:41 | INFO | valid | epoch 249 | valid on 'valid' subset | loss 12.493 | nll_loss 11.34 | word_ins 12.264 | length 4.604 | ppl 5763.54 | bleu 30.74 | wps 88821.1 | wpb 21176.3 | bsz 666.3 | num_updates 70164 | best_bleu 31.41
2023-06-13 21:58:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 21:58:52 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint249.pt (epoch 249 @ 70164 updates, score 30.74) (writing took 10.703399669378996 seconds)
2023-06-13 21:58:52 | INFO | fairseq_cli.train | end of epoch 249 (average epoch stats below)
2023-06-13 21:58:52 | INFO | train | epoch 249 | loss 3.068 | nll_loss 1.153 | glat_accu 0.555 | glat_context_p 0.453 | word_ins 2.949 | length 2.979 | ppl 8.39 | wps 114086 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 70164 | lr 0.000119383 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 21:58:52 | INFO | fairseq.trainer | begin training epoch 250
2023-06-13 21:59:14 | INFO | train_inner | epoch 250:     36 / 282 loss=3.061, nll_loss=1.145, glat_accu=0.556, glat_context_p=0.453, word_ins=2.942, length=2.963, ppl=8.35, wps=92410.7, ups=1.53, wpb=60250.3, bsz=2192.6, num_updates=70200, lr=0.000119352, gnorm=0.535, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:00:00 | INFO | train_inner | epoch 250:    136 / 282 loss=3.065, nll_loss=1.15, glat_accu=0.561, glat_context_p=0.453, word_ins=2.946, length=2.966, ppl=8.37, wps=133627, ups=2.21, wpb=60544.9, bsz=2163.5, num_updates=70300, lr=0.000119268, gnorm=0.531, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:00:46 | INFO | train_inner | epoch 250:    236 / 282 loss=3.068, nll_loss=1.153, glat_accu=0.553, glat_context_p=0.453, word_ins=2.949, length=2.98, ppl=8.39, wps=130323, ups=2.15, wpb=60530.1, bsz=2164.9, num_updates=70400, lr=0.000119183, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:01:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:01:10 | INFO | valid | epoch 250 | valid on 'valid' subset | loss 12.477 | nll_loss 11.312 | word_ins 12.24 | length 4.735 | ppl 5700.26 | bleu 31.18 | wps 86281.4 | wpb 21176.3 | bsz 666.3 | num_updates 70446 | best_bleu 31.41
2023-06-13 22:01:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:01:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint250.pt (epoch 250 @ 70446 updates, score 31.18) (writing took 9.89597013220191 seconds)
2023-06-13 22:01:20 | INFO | fairseq_cli.train | end of epoch 250 (average epoch stats below)
2023-06-13 22:01:20 | INFO | train | epoch 250 | loss 3.067 | nll_loss 1.151 | glat_accu 0.556 | glat_context_p 0.453 | word_ins 2.947 | length 2.977 | ppl 8.38 | wps 115229 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 70446 | lr 0.000119144 | gnorm 0.538 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:01:20 | INFO | fairseq.trainer | begin training epoch 251
2023-06-13 22:01:51 | INFO | train_inner | epoch 251:     54 / 282 loss=3.075, nll_loss=1.159, glat_accu=0.559, glat_context_p=0.453, word_ins=2.954, length=3, ppl=8.43, wps=93078.6, ups=1.55, wpb=60167.1, bsz=2138.3, num_updates=70500, lr=0.000119098, gnorm=0.568, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:02:13 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:02:37 | INFO | train_inner | epoch 251:    155 / 282 loss=3.07, nll_loss=1.155, glat_accu=0.567, glat_context_p=0.453, word_ins=2.95, length=2.967, ppl=8.4, wps=130567, ups=2.16, wpb=60522.1, bsz=2204.4, num_updates=70600, lr=0.000119014, gnorm=0.545, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:03:23 | INFO | train_inner | epoch 251:    255 / 282 loss=3.075, nll_loss=1.16, glat_accu=0.558, glat_context_p=0.453, word_ins=2.955, length=2.972, ppl=8.43, wps=133829, ups=2.2, wpb=60730.8, bsz=2148.2, num_updates=70700, lr=0.00011893, gnorm=0.544, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:03:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:03:38 | INFO | valid | epoch 251 | valid on 'valid' subset | loss 12.328 | nll_loss 11.155 | word_ins 12.101 | length 4.561 | ppl 5141.89 | bleu 31.53 | wps 88393.8 | wpb 21176.3 | bsz 666.3 | num_updates 70727 | best_bleu 31.53
2023-06-13 22:03:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:03:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint251.pt (epoch 251 @ 70727 updates, score 31.53) (writing took 17.043819550424814 seconds)
2023-06-13 22:03:55 | INFO | fairseq_cli.train | end of epoch 251 (average epoch stats below)
2023-06-13 22:03:55 | INFO | train | epoch 251 | loss 3.074 | nll_loss 1.159 | glat_accu 0.561 | glat_context_p 0.453 | word_ins 2.954 | length 2.98 | ppl 8.42 | wps 109326 | ups 1.81 | wpb 60414.2 | bsz 2156.4 | num_updates 70727 | lr 0.000118907 | gnorm 0.549 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:03:55 | INFO | fairseq.trainer | begin training epoch 252
2023-06-13 22:04:35 | INFO | train_inner | epoch 252:     73 / 282 loss=3.074, nll_loss=1.159, glat_accu=0.556, glat_context_p=0.453, word_ins=2.954, length=2.984, ppl=8.42, wps=83301.4, ups=1.39, wpb=60091.2, bsz=2104.9, num_updates=70800, lr=0.000118846, gnorm=0.553, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:05:20 | INFO | train_inner | epoch 252:    173 / 282 loss=3.064, nll_loss=1.148, glat_accu=0.556, glat_context_p=0.453, word_ins=2.944, length=2.979, ppl=8.36, wps=133681, ups=2.21, wpb=60618.5, bsz=2173, num_updates=70900, lr=0.000118762, gnorm=0.535, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:06:06 | INFO | train_inner | epoch 252:    273 / 282 loss=3.066, nll_loss=1.15, glat_accu=0.555, glat_context_p=0.453, word_ins=2.946, length=2.972, ppl=8.37, wps=131521, ups=2.17, wpb=60472.9, bsz=2178.6, num_updates=71000, lr=0.000118678, gnorm=0.547, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:06:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:06:13 | INFO | valid | epoch 252 | valid on 'valid' subset | loss 12.382 | nll_loss 11.214 | word_ins 12.151 | length 4.62 | ppl 5336.74 | bleu 30.93 | wps 88620.1 | wpb 21176.3 | bsz 666.3 | num_updates 71009 | best_bleu 31.53
2023-06-13 22:06:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:06:24 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint252.pt (epoch 252 @ 71009 updates, score 30.93) (writing took 10.700412575155497 seconds)
2023-06-13 22:06:24 | INFO | fairseq_cli.train | end of epoch 252 (average epoch stats below)
2023-06-13 22:06:24 | INFO | train | epoch 252 | loss 3.067 | nll_loss 1.151 | glat_accu 0.555 | glat_context_p 0.453 | word_ins 2.947 | length 2.976 | ppl 8.38 | wps 114623 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 71009 | lr 0.000118671 | gnorm 0.545 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:06:24 | INFO | fairseq.trainer | begin training epoch 253
2023-06-13 22:07:12 | INFO | train_inner | epoch 253:     91 / 282 loss=3.062, nll_loss=1.146, glat_accu=0.548, glat_context_p=0.453, word_ins=2.942, length=2.985, ppl=8.35, wps=91478.5, ups=1.52, wpb=60118.8, bsz=2129.3, num_updates=71100, lr=0.000118595, gnorm=0.539, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:07:58 | INFO | train_inner | epoch 253:    191 / 282 loss=3.06, nll_loss=1.144, glat_accu=0.542, glat_context_p=0.453, word_ins=2.941, length=2.986, ppl=8.34, wps=132102, ups=2.18, wpb=60598.2, bsz=2146.3, num_updates=71200, lr=0.000118511, gnorm=0.544, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:08:39 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:08:42 | INFO | valid | epoch 253 | valid on 'valid' subset | loss 12.515 | nll_loss 11.355 | word_ins 12.279 | length 4.72 | ppl 5854.3 | bleu 30.95 | wps 88502.1 | wpb 21176.3 | bsz 666.3 | num_updates 71291 | best_bleu 31.53
2023-06-13 22:08:42 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:08:54 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint253.pt (epoch 253 @ 71291 updates, score 30.95) (writing took 12.01032618060708 seconds)
2023-06-13 22:08:54 | INFO | fairseq_cli.train | end of epoch 253 (average epoch stats below)
2023-06-13 22:08:54 | INFO | train | epoch 253 | loss 3.061 | nll_loss 1.145 | glat_accu 0.547 | glat_context_p 0.453 | word_ins 2.942 | length 2.982 | ppl 8.35 | wps 113228 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 71291 | lr 0.000118436 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:08:54 | INFO | fairseq.trainer | begin training epoch 254
2023-06-13 22:09:04 | INFO | train_inner | epoch 254:      9 / 282 loss=3.062, nll_loss=1.146, glat_accu=0.552, glat_context_p=0.453, word_ins=2.943, length=2.973, ppl=8.35, wps=90312.9, ups=1.5, wpb=60093, bsz=2166.6, num_updates=71300, lr=0.000118428, gnorm=0.556, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:09:50 | INFO | train_inner | epoch 254:    109 / 282 loss=3.066, nll_loss=1.15, glat_accu=0.553, glat_context_p=0.452, word_ins=2.946, length=2.975, ppl=8.37, wps=133633, ups=2.2, wpb=60626.4, bsz=2120, num_updates=71400, lr=0.000118345, gnorm=0.548, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:10:36 | INFO | train_inner | epoch 254:    209 / 282 loss=3.054, nll_loss=1.138, glat_accu=0.546, glat_context_p=0.452, word_ins=2.935, length=2.984, ppl=8.31, wps=131339, ups=2.17, wpb=60409.8, bsz=2193, num_updates=71500, lr=0.000118262, gnorm=0.539, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:11:08 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:11:09 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:11:12 | INFO | valid | epoch 254 | valid on 'valid' subset | loss 12.41 | nll_loss 11.249 | word_ins 12.18 | length 4.591 | ppl 5441.08 | bleu 30.91 | wps 87572.6 | wpb 21176.3 | bsz 666.3 | num_updates 71572 | best_bleu 31.53
2023-06-13 22:11:12 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:11:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint254.pt (epoch 254 @ 71572 updates, score 30.91) (writing took 10.533708423376083 seconds)
2023-06-13 22:11:23 | INFO | fairseq_cli.train | end of epoch 254 (average epoch stats below)
2023-06-13 22:11:23 | INFO | train | epoch 254 | loss 3.059 | nll_loss 1.143 | glat_accu 0.549 | glat_context_p 0.452 | word_ins 2.94 | length 2.975 | ppl 8.33 | wps 114414 | ups 1.89 | wpb 60413.2 | bsz 2157.9 | num_updates 71572 | lr 0.000118203 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:11:23 | INFO | fairseq.trainer | begin training epoch 255
2023-06-13 22:11:43 | INFO | train_inner | epoch 255:     28 / 282 loss=3.056, nll_loss=1.14, glat_accu=0.543, glat_context_p=0.452, word_ins=2.937, length=2.981, ppl=8.32, wps=89284.9, ups=1.48, wpb=60235.4, bsz=2147.8, num_updates=71600, lr=0.00011818, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:12:28 | INFO | train_inner | epoch 255:    128 / 282 loss=3.059, nll_loss=1.143, glat_accu=0.556, glat_context_p=0.452, word_ins=2.94, length=2.976, ppl=8.34, wps=133044, ups=2.2, wpb=60460.6, bsz=2155.2, num_updates=71700, lr=0.000118097, gnorm=0.544, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:13:14 | INFO | train_inner | epoch 255:    228 / 282 loss=3.06, nll_loss=1.144, glat_accu=0.546, glat_context_p=0.452, word_ins=2.941, length=2.974, ppl=8.34, wps=132241, ups=2.18, wpb=60738, bsz=2144.9, num_updates=71800, lr=0.000118015, gnorm=0.535, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:13:39 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:13:42 | INFO | valid | epoch 255 | valid on 'valid' subset | loss 12.342 | nll_loss 11.163 | word_ins 12.107 | length 4.72 | ppl 5191.91 | bleu 31.38 | wps 87293.2 | wpb 21176.3 | bsz 666.3 | num_updates 71854 | best_bleu 31.53
2023-06-13 22:13:42 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:13:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint255.pt (epoch 255 @ 71854 updates, score 31.38) (writing took 11.13891926035285 seconds)
2023-06-13 22:13:53 | INFO | fairseq_cli.train | end of epoch 255 (average epoch stats below)
2023-06-13 22:13:53 | INFO | train | epoch 255 | loss 3.06 | nll_loss 1.144 | glat_accu 0.552 | glat_context_p 0.452 | word_ins 2.941 | length 2.976 | ppl 8.34 | wps 113017 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 71854 | lr 0.000117971 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:13:54 | INFO | fairseq.trainer | begin training epoch 256
2023-06-13 22:14:21 | INFO | train_inner | epoch 256:     46 / 282 loss=3.06, nll_loss=1.145, glat_accu=0.563, glat_context_p=0.452, word_ins=2.941, length=2.956, ppl=8.34, wps=90744.3, ups=1.51, wpb=60066.3, bsz=2205.9, num_updates=71900, lr=0.000117933, gnorm=0.539, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:15:06 | INFO | train_inner | epoch 256:    146 / 282 loss=3.064, nll_loss=1.148, glat_accu=0.557, glat_context_p=0.452, word_ins=2.944, length=2.971, ppl=8.36, wps=133679, ups=2.21, wpb=60514.1, bsz=2177.2, num_updates=72000, lr=0.000117851, gnorm=0.544, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:15:52 | INFO | train_inner | epoch 256:    246 / 282 loss=3.055, nll_loss=1.139, glat_accu=0.541, glat_context_p=0.452, word_ins=2.936, length=2.987, ppl=8.31, wps=131672, ups=2.17, wpb=60645.8, bsz=2136.7, num_updates=72100, lr=0.000117769, gnorm=0.526, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:16:09 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:16:12 | INFO | valid | epoch 256 | valid on 'valid' subset | loss 12.463 | nll_loss 11.302 | word_ins 12.229 | length 4.708 | ppl 5646.31 | bleu 31.17 | wps 88862.7 | wpb 21176.3 | bsz 666.3 | num_updates 72136 | best_bleu 31.53
2023-06-13 22:16:12 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:16:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint256.pt (epoch 256 @ 72136 updates, score 31.17) (writing took 8.75673769786954 seconds)
2023-06-13 22:16:21 | INFO | fairseq_cli.train | end of epoch 256 (average epoch stats below)
2023-06-13 22:16:21 | INFO | train | epoch 256 | loss 3.059 | nll_loss 1.143 | glat_accu 0.55 | glat_context_p 0.452 | word_ins 2.94 | length 2.976 | ppl 8.33 | wps 115761 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 72136 | lr 0.00011774 | gnorm 0.536 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:16:21 | INFO | fairseq.trainer | begin training epoch 257
2023-06-13 22:16:55 | INFO | train_inner | epoch 257:     64 / 282 loss=3.058, nll_loss=1.142, glat_accu=0.551, glat_context_p=0.452, word_ins=2.939, length=2.97, ppl=8.33, wps=94950.8, ups=1.58, wpb=60158.4, bsz=2144.9, num_updates=72200, lr=0.000117688, gnorm=0.532, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:17:41 | INFO | train_inner | epoch 257:    164 / 282 loss=3.062, nll_loss=1.145, glat_accu=0.557, glat_context_p=0.452, word_ins=2.942, length=2.983, ppl=8.35, wps=132263, ups=2.19, wpb=60426.3, bsz=2163, num_updates=72300, lr=0.000117606, gnorm=0.545, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:18:27 | INFO | train_inner | epoch 257:    264 / 282 loss=3.066, nll_loss=1.15, glat_accu=0.557, glat_context_p=0.452, word_ins=2.946, length=2.971, ppl=8.37, wps=131361, ups=2.17, wpb=60653, bsz=2153.8, num_updates=72400, lr=0.000117525, gnorm=0.534, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:18:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:18:38 | INFO | valid | epoch 257 | valid on 'valid' subset | loss 12.303 | nll_loss 11.129 | word_ins 12.071 | length 4.631 | ppl 5051.87 | bleu 31.32 | wps 87036.7 | wpb 21176.3 | bsz 666.3 | num_updates 72418 | best_bleu 31.53
2023-06-13 22:18:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:18:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint257.pt (epoch 257 @ 72418 updates, score 31.32) (writing took 12.011670060455799 seconds)
2023-06-13 22:18:50 | INFO | fairseq_cli.train | end of epoch 257 (average epoch stats below)
2023-06-13 22:18:50 | INFO | train | epoch 257 | loss 3.062 | nll_loss 1.146 | glat_accu 0.556 | glat_context_p 0.452 | word_ins 2.943 | length 2.974 | ppl 8.35 | wps 113742 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 72418 | lr 0.000117511 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:18:50 | INFO | fairseq.trainer | begin training epoch 258
2023-06-13 22:19:34 | INFO | train_inner | epoch 258:     82 / 282 loss=3.056, nll_loss=1.14, glat_accu=0.56, glat_context_p=0.452, word_ins=2.937, length=2.953, ppl=8.32, wps=89429.9, ups=1.49, wpb=60181.1, bsz=2197.8, num_updates=72500, lr=0.000117444, gnorm=0.54, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:20:19 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:20:21 | INFO | train_inner | epoch 258:    183 / 282 loss=3.057, nll_loss=1.14, glat_accu=0.549, glat_context_p=0.452, word_ins=2.937, length=2.992, ppl=8.32, wps=129163, ups=2.14, wpb=60494, bsz=2144.7, num_updates=72600, lr=0.000117363, gnorm=0.537, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 22:21:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:21:10 | INFO | valid | epoch 258 | valid on 'valid' subset | loss 12.385 | nll_loss 11.22 | word_ins 12.153 | length 4.648 | ppl 5347.63 | bleu 31.06 | wps 86218.7 | wpb 21176.3 | bsz 666.3 | num_updates 72699 | best_bleu 31.53
2023-06-13 22:21:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:21:22 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint258.pt (epoch 258 @ 72699 updates, score 31.06) (writing took 12.63316622376442 seconds)
2023-06-13 22:21:22 | INFO | fairseq_cli.train | end of epoch 258 (average epoch stats below)
2023-06-13 22:21:22 | INFO | train | epoch 258 | loss 3.058 | nll_loss 1.142 | glat_accu 0.552 | glat_context_p 0.452 | word_ins 2.938 | length 2.978 | ppl 8.33 | wps 111784 | ups 1.85 | wpb 60413.4 | bsz 2156.2 | num_updates 72699 | lr 0.000117283 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:21:22 | INFO | fairseq.trainer | begin training epoch 259
2023-06-13 22:21:29 | INFO | train_inner | epoch 259:      1 / 282 loss=3.062, nll_loss=1.146, glat_accu=0.55, glat_context_p=0.452, word_ins=2.942, length=2.987, ppl=8.35, wps=89236.3, ups=1.48, wpb=60129.8, bsz=2120.5, num_updates=72700, lr=0.000117282, gnorm=0.557, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:22:15 | INFO | train_inner | epoch 259:    101 / 282 loss=3.057, nll_loss=1.14, glat_accu=0.551, glat_context_p=0.452, word_ins=2.937, length=2.977, ppl=8.32, wps=131140, ups=2.17, wpb=60478.5, bsz=2141.4, num_updates=72800, lr=0.000117202, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:23:01 | INFO | train_inner | epoch 259:    201 / 282 loss=3.056, nll_loss=1.14, glat_accu=0.553, glat_context_p=0.451, word_ins=2.937, length=2.97, ppl=8.32, wps=132050, ups=2.17, wpb=60809.5, bsz=2142.6, num_updates=72900, lr=0.000117121, gnorm=0.541, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:23:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:23:41 | INFO | valid | epoch 259 | valid on 'valid' subset | loss 12.322 | nll_loss 11.145 | word_ins 12.084 | length 4.759 | ppl 5121.85 | bleu 31.11 | wps 88905.3 | wpb 21176.3 | bsz 666.3 | num_updates 72981 | best_bleu 31.53
2023-06-13 22:23:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:23:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint259.pt (epoch 259 @ 72981 updates, score 31.11) (writing took 10.055820491164923 seconds)
2023-06-13 22:23:51 | INFO | fairseq_cli.train | end of epoch 259 (average epoch stats below)
2023-06-13 22:23:51 | INFO | train | epoch 259 | loss 3.059 | nll_loss 1.143 | glat_accu 0.555 | glat_context_p 0.451 | word_ins 2.94 | length 2.971 | ppl 8.34 | wps 114574 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 72981 | lr 0.000117056 | gnorm 0.54 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:23:51 | INFO | fairseq.trainer | begin training epoch 260
2023-06-13 22:24:06 | INFO | train_inner | epoch 260:     19 / 282 loss=3.065, nll_loss=1.15, glat_accu=0.564, glat_context_p=0.451, word_ins=2.945, length=2.966, ppl=8.37, wps=92681.8, ups=1.55, wpb=59963.3, bsz=2190.3, num_updates=73000, lr=0.000117041, gnorm=0.541, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:24:51 | INFO | train_inner | epoch 260:    119 / 282 loss=3.051, nll_loss=1.134, glat_accu=0.556, glat_context_p=0.451, word_ins=2.932, length=2.947, ppl=8.29, wps=133983, ups=2.2, wpb=60767.1, bsz=2195.6, num_updates=73100, lr=0.000116961, gnorm=0.532, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:25:37 | INFO | train_inner | epoch 260:    219 / 282 loss=3.057, nll_loss=1.14, glat_accu=0.543, glat_context_p=0.451, word_ins=2.937, length=2.997, ppl=8.32, wps=130439, ups=2.16, wpb=60394.1, bsz=2125, num_updates=73200, lr=0.000116881, gnorm=0.539, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:26:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:26:09 | INFO | valid | epoch 260 | valid on 'valid' subset | loss 12.44 | nll_loss 11.28 | word_ins 12.206 | length 4.698 | ppl 5556.88 | bleu 30.78 | wps 87228.4 | wpb 21176.3 | bsz 666.3 | num_updates 73263 | best_bleu 31.53
2023-06-13 22:26:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:26:18 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint260.pt (epoch 260 @ 73263 updates, score 30.78) (writing took 9.246714070439339 seconds)
2023-06-13 22:26:18 | INFO | fairseq_cli.train | end of epoch 260 (average epoch stats below)
2023-06-13 22:26:18 | INFO | train | epoch 260 | loss 3.056 | nll_loss 1.14 | glat_accu 0.552 | glat_context_p 0.451 | word_ins 2.937 | length 2.971 | ppl 8.32 | wps 115511 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 73263 | lr 0.000116831 | gnorm 0.535 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:26:19 | INFO | fairseq.trainer | begin training epoch 261
2023-06-13 22:26:40 | INFO | train_inner | epoch 261:     37 / 282 loss=3.055, nll_loss=1.14, glat_accu=0.548, glat_context_p=0.451, word_ins=2.937, length=2.967, ppl=8.31, wps=95352.9, ups=1.58, wpb=60206.8, bsz=2165.8, num_updates=73300, lr=0.000116801, gnorm=0.527, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:27:26 | INFO | train_inner | epoch 261:    137 / 282 loss=3.055, nll_loss=1.138, glat_accu=0.554, glat_context_p=0.451, word_ins=2.936, length=2.981, ppl=8.31, wps=132822, ups=2.2, wpb=60315.9, bsz=2147.5, num_updates=73400, lr=0.000116722, gnorm=0.543, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:28:11 | INFO | train_inner | epoch 261:    237 / 282 loss=3.056, nll_loss=1.14, glat_accu=0.55, glat_context_p=0.451, word_ins=2.937, length=2.973, ppl=8.32, wps=132950, ups=2.19, wpb=60646, bsz=2179.5, num_updates=73500, lr=0.000116642, gnorm=0.542, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:28:32 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:28:35 | INFO | valid | epoch 261 | valid on 'valid' subset | loss 12.388 | nll_loss 11.217 | word_ins 12.154 | length 4.687 | ppl 5361.27 | bleu 31.15 | wps 88394.4 | wpb 21176.3 | bsz 666.3 | num_updates 73545 | best_bleu 31.53
2023-06-13 22:28:35 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:28:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint261.pt (epoch 261 @ 73545 updates, score 31.15) (writing took 12.256948921829462 seconds)
2023-06-13 22:28:48 | INFO | fairseq_cli.train | end of epoch 261 (average epoch stats below)
2023-06-13 22:28:48 | INFO | train | epoch 261 | loss 3.054 | nll_loss 1.138 | glat_accu 0.549 | glat_context_p 0.451 | word_ins 2.935 | length 2.972 | ppl 8.3 | wps 114278 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 73545 | lr 0.000116607 | gnorm 0.54 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:28:48 | INFO | fairseq.trainer | begin training epoch 262
2023-06-13 22:29:19 | INFO | train_inner | epoch 262:     55 / 282 loss=3.048, nll_loss=1.132, glat_accu=0.554, glat_context_p=0.451, word_ins=2.93, length=2.951, ppl=8.27, wps=88899.7, ups=1.48, wpb=60240.3, bsz=2183.1, num_updates=73600, lr=0.000116563, gnorm=0.541, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:29:28 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:30:05 | INFO | train_inner | epoch 262:    156 / 282 loss=3.044, nll_loss=1.128, glat_accu=0.546, glat_context_p=0.451, word_ins=2.926, length=2.96, ppl=8.25, wps=132518, ups=2.18, wpb=60662.5, bsz=2182.1, num_updates=73700, lr=0.000116484, gnorm=0.531, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:30:51 | INFO | train_inner | epoch 262:    256 / 282 loss=3.056, nll_loss=1.139, glat_accu=0.542, glat_context_p=0.451, word_ins=2.936, length=2.995, ppl=8.32, wps=131616, ups=2.18, wpb=60511.2, bsz=2089.4, num_updates=73800, lr=0.000116405, gnorm=0.542, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:31:03 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 22:31:03 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:31:06 | INFO | valid | epoch 262 | valid on 'valid' subset | loss 12.433 | nll_loss 11.274 | word_ins 12.202 | length 4.628 | ppl 5528.64 | bleu 31.3 | wps 86056.6 | wpb 21176.3 | bsz 666.3 | num_updates 73825 | best_bleu 31.53
2023-06-13 22:31:06 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:31:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint262.pt (epoch 262 @ 73825 updates, score 31.3) (writing took 9.957095123827457 seconds)
2023-06-13 22:31:16 | INFO | fairseq_cli.train | end of epoch 262 (average epoch stats below)
2023-06-13 22:31:16 | INFO | train | epoch 262 | loss 3.05 | nll_loss 1.133 | glat_accu 0.547 | glat_context_p 0.451 | word_ins 2.931 | length 2.972 | ppl 8.28 | wps 114350 | ups 1.89 | wpb 60591.2 | bsz 2164.4 | num_updates 73825 | lr 0.000116385 | gnorm 0.537 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 22:31:16 | INFO | fairseq.trainer | begin training epoch 263
2023-06-13 22:31:56 | INFO | train_inner | epoch 263:     75 / 282 loss=3.053, nll_loss=1.136, glat_accu=0.557, glat_context_p=0.451, word_ins=2.934, length=2.963, ppl=8.3, wps=92572.4, ups=1.53, wpb=60548.8, bsz=2176.8, num_updates=73900, lr=0.000116326, gnorm=0.549, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:32:42 | INFO | train_inner | epoch 263:    175 / 282 loss=3.052, nll_loss=1.136, glat_accu=0.557, glat_context_p=0.451, word_ins=2.933, length=2.964, ppl=8.3, wps=132853, ups=2.19, wpb=60624.1, bsz=2225.1, num_updates=74000, lr=0.000116248, gnorm=0.538, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 22:33:28 | INFO | train_inner | epoch 263:    275 / 282 loss=3.063, nll_loss=1.147, glat_accu=0.544, glat_context_p=0.451, word_ins=2.943, length=2.999, ppl=8.36, wps=131065, ups=2.16, wpb=60626.8, bsz=2093.2, num_updates=74100, lr=0.000116169, gnorm=0.538, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:33:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:33:34 | INFO | valid | epoch 263 | valid on 'valid' subset | loss 12.42 | nll_loss 11.254 | word_ins 12.186 | length 4.678 | ppl 5481.89 | bleu 31.16 | wps 88836.6 | wpb 21176.3 | bsz 666.3 | num_updates 74107 | best_bleu 31.53
2023-06-13 22:33:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:33:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint263.pt (epoch 263 @ 74107 updates, score 31.16) (writing took 10.403464313596487 seconds)
2023-06-13 22:33:45 | INFO | fairseq_cli.train | end of epoch 263 (average epoch stats below)
2023-06-13 22:33:45 | INFO | train | epoch 263 | loss 3.056 | nll_loss 1.14 | glat_accu 0.553 | glat_context_p 0.451 | word_ins 2.937 | length 2.974 | ppl 8.32 | wps 114376 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 74107 | lr 0.000116164 | gnorm 0.544 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 22:33:45 | INFO | fairseq.trainer | begin training epoch 264
2023-06-13 22:34:34 | INFO | train_inner | epoch 264:     93 / 282 loss=3.046, nll_loss=1.13, glat_accu=0.553, glat_context_p=0.451, word_ins=2.928, length=2.958, ppl=8.26, wps=91053.9, ups=1.52, wpb=59913.2, bsz=2166.6, num_updates=74200, lr=0.000116091, gnorm=0.546, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:35:19 | INFO | train_inner | epoch 264:    193 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.552, glat_context_p=0.451, word_ins=2.93, length=2.963, ppl=8.28, wps=133608, ups=2.21, wpb=60525.8, bsz=2144.9, num_updates=74300, lr=0.000116013, gnorm=0.534, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 22:36:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:36:03 | INFO | valid | epoch 264 | valid on 'valid' subset | loss 12.437 | nll_loss 11.273 | word_ins 12.202 | length 4.679 | ppl 5544.62 | bleu 31.13 | wps 89882.8 | wpb 21176.3 | bsz 666.3 | num_updates 74389 | best_bleu 31.53
2023-06-13 22:36:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:36:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint264.pt (epoch 264 @ 74389 updates, score 31.13) (writing took 12.696188542991877 seconds)
2023-06-13 22:36:16 | INFO | fairseq_cli.train | end of epoch 264 (average epoch stats below)
2023-06-13 22:36:16 | INFO | train | epoch 264 | loss 3.05 | nll_loss 1.133 | glat_accu 0.55 | glat_context_p 0.451 | word_ins 2.931 | length 2.967 | ppl 8.28 | wps 112748 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 74389 | lr 0.000115943 | gnorm 0.541 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 22:36:16 | INFO | fairseq.trainer | begin training epoch 265
2023-06-13 22:36:27 | INFO | train_inner | epoch 265:     11 / 282 loss=3.053, nll_loss=1.137, glat_accu=0.545, glat_context_p=0.45, word_ins=2.934, length=2.973, ppl=8.3, wps=89250.8, ups=1.48, wpb=60353.5, bsz=2145.5, num_updates=74400, lr=0.000115935, gnorm=0.55, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:37:13 | INFO | train_inner | epoch 265:    111 / 282 loss=3.051, nll_loss=1.134, glat_accu=0.556, glat_context_p=0.45, word_ins=2.932, length=2.956, ppl=8.29, wps=131540, ups=2.18, wpb=60469.9, bsz=2148.4, num_updates=74500, lr=0.000115857, gnorm=0.545, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:37:59 | INFO | train_inner | epoch 265:    211 / 282 loss=3.056, nll_loss=1.14, glat_accu=0.544, glat_context_p=0.45, word_ins=2.937, length=2.978, ppl=8.31, wps=132157, ups=2.18, wpb=60619.3, bsz=2153.8, num_updates=74600, lr=0.000115779, gnorm=0.541, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:38:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:38:34 | INFO | valid | epoch 265 | valid on 'valid' subset | loss 12.39 | nll_loss 11.224 | word_ins 12.158 | length 4.64 | ppl 5365.83 | bleu 30.99 | wps 88653.2 | wpb 21176.3 | bsz 666.3 | num_updates 74671 | best_bleu 31.53
2023-06-13 22:38:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:38:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint265.pt (epoch 265 @ 74671 updates, score 30.99) (writing took 10.133034858852625 seconds)
2023-06-13 22:38:44 | INFO | fairseq_cli.train | end of epoch 265 (average epoch stats below)
2023-06-13 22:38:44 | INFO | train | epoch 265 | loss 3.053 | nll_loss 1.137 | glat_accu 0.552 | glat_context_p 0.45 | word_ins 2.934 | length 2.966 | ppl 8.3 | wps 115094 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 74671 | lr 0.000115724 | gnorm 0.544 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 22:38:44 | INFO | fairseq.trainer | begin training epoch 266
2023-06-13 22:39:03 | INFO | train_inner | epoch 266:     29 / 282 loss=3.054, nll_loss=1.137, glat_accu=0.554, glat_context_p=0.45, word_ins=2.934, length=2.967, ppl=8.3, wps=93121.3, ups=1.55, wpb=60249, bsz=2156.6, num_updates=74700, lr=0.000115702, gnorm=0.544, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 22:39:49 | INFO | train_inner | epoch 266:    129 / 282 loss=3.053, nll_loss=1.135, glat_accu=0.552, glat_context_p=0.45, word_ins=2.933, length=2.988, ppl=8.3, wps=131516, ups=2.18, wpb=60413.4, bsz=2145.6, num_updates=74800, lr=0.000115624, gnorm=0.542, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 22:40:35 | INFO | train_inner | epoch 266:    229 / 282 loss=3.055, nll_loss=1.14, glat_accu=0.559, glat_context_p=0.45, word_ins=2.936, length=2.95, ppl=8.31, wps=134077, ups=2.21, wpb=60617.7, bsz=2208.2, num_updates=74900, lr=0.000115547, gnorm=0.547, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:40:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:41:02 | INFO | valid | epoch 266 | valid on 'valid' subset | loss 12.495 | nll_loss 11.339 | word_ins 12.263 | length 4.652 | ppl 5774.32 | bleu 30.97 | wps 87810.9 | wpb 21176.3 | bsz 666.3 | num_updates 74953 | best_bleu 31.53
2023-06-13 22:41:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:41:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint266.pt (epoch 266 @ 74953 updates, score 30.97) (writing took 8.791966006159782 seconds)
2023-06-13 22:41:11 | INFO | fairseq_cli.train | end of epoch 266 (average epoch stats below)
2023-06-13 22:41:11 | INFO | train | epoch 266 | loss 3.053 | nll_loss 1.137 | glat_accu 0.553 | glat_context_p 0.45 | word_ins 2.934 | length 2.968 | ppl 8.3 | wps 116199 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 74953 | lr 0.000115506 | gnorm 0.544 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:41:11 | INFO | fairseq.trainer | begin training epoch 267
2023-06-13 22:41:38 | INFO | train_inner | epoch 267:     47 / 282 loss=3.049, nll_loss=1.133, glat_accu=0.542, glat_context_p=0.45, word_ins=2.93, length=2.974, ppl=8.28, wps=95618, ups=1.59, wpb=60186.3, bsz=2127.4, num_updates=75000, lr=0.00011547, gnorm=0.545, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:42:23 | INFO | train_inner | epoch 267:    147 / 282 loss=3.041, nll_loss=1.125, glat_accu=0.545, glat_context_p=0.45, word_ins=2.924, length=2.939, ppl=8.23, wps=133938, ups=2.2, wpb=60826.9, bsz=2189.4, num_updates=75100, lr=0.000115393, gnorm=0.533, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:43:09 | INFO | train_inner | epoch 267:    247 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.518, glat_context_p=0.45, word_ins=2.917, length=2.992, ppl=8.2, wps=130976, ups=2.17, wpb=60382.1, bsz=2148.2, num_updates=75200, lr=0.000115316, gnorm=0.525, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:43:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:43:28 | INFO | valid | epoch 267 | valid on 'valid' subset | loss 12.509 | nll_loss 11.362 | word_ins 12.28 | length 4.613 | ppl 5829.5 | bleu 30.41 | wps 88473.1 | wpb 21176.3 | bsz 666.3 | num_updates 75235 | best_bleu 31.53
2023-06-13 22:43:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:43:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint267.pt (epoch 267 @ 75235 updates, score 30.41) (writing took 8.667273793369532 seconds)
2023-06-13 22:43:37 | INFO | fairseq_cli.train | end of epoch 267 (average epoch stats below)
2023-06-13 22:43:37 | INFO | train | epoch 267 | loss 3.041 | nll_loss 1.124 | glat_accu 0.535 | glat_context_p 0.45 | word_ins 2.923 | length 2.972 | ppl 8.23 | wps 116557 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 75235 | lr 0.00011529 | gnorm 0.535 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:43:37 | INFO | fairseq.trainer | begin training epoch 268
2023-06-13 22:44:12 | INFO | train_inner | epoch 268:     65 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.536, glat_context_p=0.45, word_ins=2.923, length=2.977, ppl=8.23, wps=94974.2, ups=1.58, wpb=60006.8, bsz=2141.8, num_updates=75300, lr=0.00011524, gnorm=0.542, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:44:58 | INFO | train_inner | epoch 268:    165 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.531, glat_context_p=0.45, word_ins=2.917, length=2.965, ppl=8.2, wps=132360, ups=2.18, wpb=60643.2, bsz=2165.3, num_updates=75400, lr=0.000115163, gnorm=0.517, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:45:43 | INFO | train_inner | epoch 268:    265 / 282 loss=3.056, nll_loss=1.139, glat_accu=0.554, glat_context_p=0.45, word_ins=2.936, length=2.978, ppl=8.31, wps=133422, ups=2.2, wpb=60515.8, bsz=2152, num_updates=75500, lr=0.000115087, gnorm=0.561, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:45:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:45:54 | INFO | valid | epoch 268 | valid on 'valid' subset | loss 12.418 | nll_loss 11.258 | word_ins 12.19 | length 4.549 | ppl 5471.66 | bleu 31.24 | wps 87647.9 | wpb 21176.3 | bsz 666.3 | num_updates 75517 | best_bleu 31.53
2023-06-13 22:45:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:46:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint268.pt (epoch 268 @ 75517 updates, score 31.24) (writing took 12.031474761664867 seconds)
2023-06-13 22:46:06 | INFO | fairseq_cli.train | end of epoch 268 (average epoch stats below)
2023-06-13 22:46:06 | INFO | train | epoch 268 | loss 3.045 | nll_loss 1.128 | glat_accu 0.542 | glat_context_p 0.45 | word_ins 2.926 | length 2.971 | ppl 8.25 | wps 114141 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 75517 | lr 0.000115074 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:46:06 | INFO | fairseq.trainer | begin training epoch 269
2023-06-13 22:46:50 | INFO | train_inner | epoch 269:     83 / 282 loss=3.053, nll_loss=1.137, glat_accu=0.561, glat_context_p=0.45, word_ins=2.934, length=2.953, ppl=8.3, wps=90091, ups=1.5, wpb=60103.6, bsz=2196.9, num_updates=75600, lr=0.000115011, gnorm=0.544, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:47:36 | INFO | train_inner | epoch 269:    183 / 282 loss=3.05, nll_loss=1.134, glat_accu=0.547, glat_context_p=0.45, word_ins=2.931, length=2.969, ppl=8.28, wps=131068, ups=2.17, wpb=60502.8, bsz=2113, num_updates=75700, lr=0.000114935, gnorm=0.538, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:48:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:48:25 | INFO | valid | epoch 269 | valid on 'valid' subset | loss 12.453 | nll_loss 11.302 | word_ins 12.225 | length 4.574 | ppl 5607.13 | bleu 30.87 | wps 87161.1 | wpb 21176.3 | bsz 666.3 | num_updates 75799 | best_bleu 31.53
2023-06-13 22:48:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:48:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint269.pt (epoch 269 @ 75799 updates, score 30.87) (writing took 11.043639119714499 seconds)
2023-06-13 22:48:36 | INFO | fairseq_cli.train | end of epoch 269 (average epoch stats below)
2023-06-13 22:48:36 | INFO | train | epoch 269 | loss 3.053 | nll_loss 1.137 | glat_accu 0.553 | glat_context_p 0.45 | word_ins 2.934 | length 2.966 | ppl 8.3 | wps 113906 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 75799 | lr 0.00011486 | gnorm 0.549 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:48:36 | INFO | fairseq.trainer | begin training epoch 270
2023-06-13 22:48:42 | INFO | train_inner | epoch 270:      1 / 282 loss=3.057, nll_loss=1.141, glat_accu=0.55, glat_context_p=0.45, word_ins=2.937, length=2.981, ppl=8.32, wps=91392.1, ups=1.52, wpb=60197.3, bsz=2150.2, num_updates=75800, lr=0.000114859, gnorm=0.569, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:49:15 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:49:28 | INFO | train_inner | epoch 270:    102 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.544, glat_context_p=0.449, word_ins=2.917, length=2.956, ppl=8.2, wps=132713, ups=2.19, wpb=60667.9, bsz=2187, num_updates=75900, lr=0.000114783, gnorm=0.537, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:50:14 | INFO | train_inner | epoch 270:    202 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.541, glat_context_p=0.449, word_ins=2.93, length=2.977, ppl=8.27, wps=131119, ups=2.17, wpb=60477.3, bsz=2133.5, num_updates=76000, lr=0.000114708, gnorm=0.544, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:50:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:50:53 | INFO | valid | epoch 270 | valid on 'valid' subset | loss 12.423 | nll_loss 11.255 | word_ins 12.183 | length 4.815 | ppl 5491.21 | bleu 30.32 | wps 88962.7 | wpb 21176.3 | bsz 666.3 | num_updates 76080 | best_bleu 31.53
2023-06-13 22:50:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:51:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint270.pt (epoch 270 @ 76080 updates, score 30.32) (writing took 10.2963925935328 seconds)
2023-06-13 22:51:04 | INFO | fairseq_cli.train | end of epoch 270 (average epoch stats below)
2023-06-13 22:51:04 | INFO | train | epoch 270 | loss 3.041 | nll_loss 1.125 | glat_accu 0.542 | glat_context_p 0.449 | word_ins 2.923 | length 2.965 | ppl 8.23 | wps 114593 | ups 1.9 | wpb 60407.7 | bsz 2156.7 | num_updates 76080 | lr 0.000114648 | gnorm 0.541 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:51:04 | INFO | fairseq.trainer | begin training epoch 271
2023-06-13 22:51:19 | INFO | train_inner | epoch 271:     20 / 282 loss=3.041, nll_loss=1.125, glat_accu=0.541, glat_context_p=0.449, word_ins=2.923, length=2.965, ppl=8.23, wps=92477.2, ups=1.54, wpb=60014.2, bsz=2147.3, num_updates=76100, lr=0.000114632, gnorm=0.542, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:52:04 | INFO | train_inner | epoch 271:    120 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.548, glat_context_p=0.449, word_ins=2.923, length=2.96, ppl=8.24, wps=133194, ups=2.2, wpb=60625.2, bsz=2177.8, num_updates=76200, lr=0.000114557, gnorm=0.534, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:52:50 | INFO | train_inner | epoch 271:    220 / 282 loss=3.041, nll_loss=1.124, glat_accu=0.544, glat_context_p=0.449, word_ins=2.923, length=2.968, ppl=8.23, wps=132217, ups=2.18, wpb=60566.4, bsz=2163.6, num_updates=76300, lr=0.000114482, gnorm=0.534, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:53:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:53:22 | INFO | valid | epoch 271 | valid on 'valid' subset | loss 12.59 | nll_loss 11.45 | word_ins 12.358 | length 4.659 | ppl 6166.6 | bleu 30.7 | wps 89029.6 | wpb 21176.3 | bsz 666.3 | num_updates 76362 | best_bleu 31.53
2023-06-13 22:53:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:53:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint271.pt (epoch 271 @ 76362 updates, score 30.7) (writing took 6.533741421997547 seconds)
2023-06-13 22:53:28 | INFO | fairseq_cli.train | end of epoch 271 (average epoch stats below)
2023-06-13 22:53:28 | INFO | train | epoch 271 | loss 3.044 | nll_loss 1.128 | glat_accu 0.545 | glat_context_p 0.449 | word_ins 2.926 | length 2.972 | ppl 8.25 | wps 117757 | ups 1.95 | wpb 60413.8 | bsz 2157.2 | num_updates 76362 | lr 0.000114436 | gnorm 0.538 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 22:53:29 | INFO | fairseq.trainer | begin training epoch 272
2023-06-13 22:53:51 | INFO | train_inner | epoch 272:     38 / 282 loss=3.048, nll_loss=1.131, glat_accu=0.544, glat_context_p=0.449, word_ins=2.929, length=2.983, ppl=8.27, wps=99153.5, ups=1.65, wpb=60111.1, bsz=2139.8, num_updates=76400, lr=0.000114407, gnorm=0.545, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:54:36 | INFO | train_inner | epoch 272:    138 / 282 loss=3.051, nll_loss=1.135, glat_accu=0.552, glat_context_p=0.449, word_ins=2.932, length=2.974, ppl=8.29, wps=132869, ups=2.19, wpb=60645.8, bsz=2155.2, num_updates=76500, lr=0.000114332, gnorm=0.541, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 22:55:23 | INFO | train_inner | epoch 272:    238 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.55, glat_context_p=0.449, word_ins=2.93, length=2.968, ppl=8.28, wps=131556, ups=2.17, wpb=60583.4, bsz=2169.9, num_updates=76600, lr=0.000114258, gnorm=0.535, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:55:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:55:46 | INFO | valid | epoch 272 | valid on 'valid' subset | loss 12.395 | nll_loss 11.23 | word_ins 12.164 | length 4.627 | ppl 5385 | bleu 31.07 | wps 88424.8 | wpb 21176.3 | bsz 666.3 | num_updates 76644 | best_bleu 31.53
2023-06-13 22:55:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:55:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint272.pt (epoch 272 @ 76644 updates, score 31.07) (writing took 6.6729685217142105 seconds)
2023-06-13 22:55:53 | INFO | fairseq_cli.train | end of epoch 272 (average epoch stats below)
2023-06-13 22:55:53 | INFO | train | epoch 272 | loss 3.048 | nll_loss 1.131 | glat_accu 0.551 | glat_context_p 0.449 | word_ins 2.929 | length 2.969 | ppl 8.27 | wps 117580 | ups 1.95 | wpb 60413.8 | bsz 2157.2 | num_updates 76644 | lr 0.000114225 | gnorm 0.543 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:55:53 | INFO | fairseq.trainer | begin training epoch 273
2023-06-13 22:56:25 | INFO | train_inner | epoch 273:     56 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.554, glat_context_p=0.449, word_ins=2.929, length=2.967, ppl=8.27, wps=96696.3, ups=1.61, wpb=59990.1, bsz=2120.1, num_updates=76700, lr=0.000114183, gnorm=0.551, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:57:10 | INFO | train_inner | epoch 273:    156 / 282 loss=3.05, nll_loss=1.134, glat_accu=0.561, glat_context_p=0.449, word_ins=2.931, length=2.944, ppl=8.28, wps=132278, ups=2.18, wpb=60676, bsz=2202.5, num_updates=76800, lr=0.000114109, gnorm=0.543, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:57:56 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 22:57:57 | INFO | train_inner | epoch 273:    257 / 282 loss=3.048, nll_loss=1.131, glat_accu=0.543, glat_context_p=0.449, word_ins=2.929, length=2.976, ppl=8.27, wps=129916, ups=2.14, wpb=60697.6, bsz=2134.5, num_updates=76900, lr=0.000114035, gnorm=0.531, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 22:58:09 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 22:58:12 | INFO | valid | epoch 273 | valid on 'valid' subset | loss 12.664 | nll_loss 11.523 | word_ins 12.427 | length 4.722 | ppl 6488.93 | bleu 30.59 | wps 88075.3 | wpb 21176.3 | bsz 666.3 | num_updates 76925 | best_bleu 31.53
2023-06-13 22:58:12 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 22:58:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint273.pt (epoch 273 @ 76925 updates, score 30.59) (writing took 15.206368278712034 seconds)
2023-06-13 22:58:27 | INFO | fairseq_cli.train | end of epoch 273 (average epoch stats below)
2023-06-13 22:58:27 | INFO | train | epoch 273 | loss 3.048 | nll_loss 1.131 | glat_accu 0.552 | glat_context_p 0.449 | word_ins 2.929 | length 2.961 | ppl 8.27 | wps 110300 | ups 1.83 | wpb 60416.9 | bsz 2155.7 | num_updates 76925 | lr 0.000114016 | gnorm 0.539 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 22:58:27 | INFO | fairseq.trainer | begin training epoch 274
2023-06-13 22:59:07 | INFO | train_inner | epoch 274:     75 / 282 loss=3.033, nll_loss=1.116, glat_accu=0.54, glat_context_p=0.449, word_ins=2.915, length=2.95, ppl=8.18, wps=85716.3, ups=1.43, wpb=60073.3, bsz=2154.5, num_updates=77000, lr=0.000113961, gnorm=0.536, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 22:59:54 | INFO | train_inner | epoch 274:    175 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.545, glat_context_p=0.449, word_ins=2.923, length=2.974, ppl=8.24, wps=130753, ups=2.16, wpb=60578.2, bsz=2159.2, num_updates=77100, lr=0.000113887, gnorm=0.543, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:00:39 | INFO | train_inner | epoch 274:    275 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.545, glat_context_p=0.449, word_ins=2.923, length=2.963, ppl=8.24, wps=132762, ups=2.19, wpb=60546.6, bsz=2176.6, num_updates=77200, lr=0.000113813, gnorm=0.533, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:00:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:00:46 | INFO | valid | epoch 274 | valid on 'valid' subset | loss 12.482 | nll_loss 11.323 | word_ins 12.252 | length 4.595 | ppl 5720.63 | bleu 30.48 | wps 89079.8 | wpb 21176.3 | bsz 666.3 | num_updates 77207 | best_bleu 31.53
2023-06-13 23:00:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:00:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint274.pt (epoch 274 @ 77207 updates, score 30.48) (writing took 12.789805602282286 seconds)
2023-06-13 23:00:58 | INFO | fairseq_cli.train | end of epoch 274 (average epoch stats below)
2023-06-13 23:00:58 | INFO | train | epoch 274 | loss 3.039 | nll_loss 1.123 | glat_accu 0.543 | glat_context_p 0.449 | word_ins 2.921 | length 2.964 | ppl 8.22 | wps 112757 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 77207 | lr 0.000113808 | gnorm 0.537 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:00:58 | INFO | fairseq.trainer | begin training epoch 275
2023-06-13 23:01:47 | INFO | train_inner | epoch 275:     93 / 282 loss=3.038, nll_loss=1.122, glat_accu=0.539, glat_context_p=0.449, word_ins=2.92, length=2.958, ppl=8.21, wps=88513.8, ups=1.47, wpb=60226.2, bsz=2126.7, num_updates=77300, lr=0.000113739, gnorm=0.555, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:02:33 | INFO | train_inner | epoch 275:    193 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.548, glat_context_p=0.448, word_ins=2.917, length=2.959, ppl=8.2, wps=133288, ups=2.2, wpb=60561.4, bsz=2178.2, num_updates=77400, lr=0.000113666, gnorm=0.556, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:03:14 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:03:17 | INFO | valid | epoch 275 | valid on 'valid' subset | loss 12.392 | nll_loss 11.222 | word_ins 12.154 | length 4.759 | ppl 5374.03 | bleu 31.13 | wps 83753.6 | wpb 21176.3 | bsz 666.3 | num_updates 77489 | best_bleu 31.53
2023-06-13 23:03:17 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:03:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint275.pt (epoch 275 @ 77489 updates, score 31.13) (writing took 10.965254548937082 seconds)
2023-06-13 23:03:28 | INFO | fairseq_cli.train | end of epoch 275 (average epoch stats below)
2023-06-13 23:03:28 | INFO | train | epoch 275 | loss 3.041 | nll_loss 1.124 | glat_accu 0.546 | glat_context_p 0.448 | word_ins 2.923 | length 2.967 | ppl 8.23 | wps 113776 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 77489 | lr 0.0001136 | gnorm 0.552 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 23:03:28 | INFO | fairseq.trainer | begin training epoch 276
2023-06-13 23:03:40 | INFO | train_inner | epoch 276:     11 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.55, glat_context_p=0.448, word_ins=2.931, length=2.982, ppl=8.28, wps=89541.6, ups=1.49, wpb=59981.6, bsz=2134.2, num_updates=77500, lr=0.000113592, gnorm=0.549, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:04:25 | INFO | train_inner | epoch 276:    111 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.551, glat_context_p=0.448, word_ins=2.929, length=2.976, ppl=8.28, wps=132816, ups=2.19, wpb=60575.6, bsz=2141.2, num_updates=77600, lr=0.000113519, gnorm=0.546, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:04:40 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-13 23:05:11 | INFO | train_inner | epoch 276:    212 / 282 loss=3.035, nll_loss=1.119, glat_accu=0.549, glat_context_p=0.448, word_ins=2.918, length=2.936, ppl=8.2, wps=131783, ups=2.17, wpb=60592.8, bsz=2212.8, num_updates=77700, lr=0.000113446, gnorm=0.544, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:05:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:05:46 | INFO | valid | epoch 276 | valid on 'valid' subset | loss 12.498 | nll_loss 11.343 | word_ins 12.266 | length 4.639 | ppl 5783.15 | bleu 30.92 | wps 88849.4 | wpb 21176.3 | bsz 666.3 | num_updates 77770 | best_bleu 31.53
2023-06-13 23:05:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:05:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint276.pt (epoch 276 @ 77770 updates, score 30.92) (writing took 11.36776727065444 seconds)
2023-06-13 23:05:58 | INFO | fairseq_cli.train | end of epoch 276 (average epoch stats below)
2023-06-13 23:05:58 | INFO | train | epoch 276 | loss 3.044 | nll_loss 1.127 | glat_accu 0.549 | glat_context_p 0.448 | word_ins 2.925 | length 2.962 | ppl 8.25 | wps 113431 | ups 1.88 | wpb 60414.3 | bsz 2156.4 | num_updates 77770 | lr 0.000113395 | gnorm 0.547 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:05:58 | INFO | fairseq.trainer | begin training epoch 277
2023-06-13 23:06:18 | INFO | train_inner | epoch 277:     30 / 282 loss=3.047, nll_loss=1.131, glat_accu=0.538, glat_context_p=0.448, word_ins=2.929, length=2.985, ppl=8.27, wps=89795.5, ups=1.49, wpb=60145.8, bsz=2103.4, num_updates=77800, lr=0.000113373, gnorm=0.549, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:07:04 | INFO | train_inner | epoch 277:    130 / 282 loss=3.041, nll_loss=1.124, glat_accu=0.537, glat_context_p=0.448, word_ins=2.922, length=2.978, ppl=8.23, wps=132360, ups=2.18, wpb=60777.2, bsz=2162.8, num_updates=77900, lr=0.0001133, gnorm=0.543, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:07:50 | INFO | train_inner | epoch 277:    230 / 282 loss=3.044, nll_loss=1.128, glat_accu=0.551, glat_context_p=0.448, word_ins=2.926, length=2.942, ppl=8.25, wps=131588, ups=2.17, wpb=60514.4, bsz=2169.4, num_updates=78000, lr=0.000113228, gnorm=0.542, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:08:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:08:16 | INFO | valid | epoch 277 | valid on 'valid' subset | loss 12.406 | nll_loss 11.239 | word_ins 12.168 | length 4.772 | ppl 5426.31 | bleu 30.74 | wps 87480.9 | wpb 21176.3 | bsz 666.3 | num_updates 78052 | best_bleu 31.53
2023-06-13 23:08:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:08:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint277.pt (epoch 277 @ 78052 updates, score 30.74) (writing took 11.232444908469915 seconds)
2023-06-13 23:08:27 | INFO | fairseq_cli.train | end of epoch 277 (average epoch stats below)
2023-06-13 23:08:27 | INFO | train | epoch 277 | loss 3.042 | nll_loss 1.126 | glat_accu 0.543 | glat_context_p 0.448 | word_ins 2.924 | length 2.96 | ppl 8.24 | wps 113785 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 78052 | lr 0.00011319 | gnorm 0.547 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:08:28 | INFO | fairseq.trainer | begin training epoch 278
2023-06-13 23:08:56 | INFO | train_inner | epoch 278:     48 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.548, glat_context_p=0.448, word_ins=2.925, length=2.97, ppl=8.25, wps=91658.2, ups=1.53, wpb=59966.8, bsz=2152.2, num_updates=78100, lr=0.000113155, gnorm=0.56, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:09:41 | INFO | train_inner | epoch 278:    148 / 282 loss=3.046, nll_loss=1.13, glat_accu=0.557, glat_context_p=0.448, word_ins=2.928, length=2.94, ppl=8.26, wps=134123, ups=2.21, wpb=60558.1, bsz=2205, num_updates=78200, lr=0.000113083, gnorm=0.553, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:10:27 | INFO | train_inner | epoch 278:    248 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.542, glat_context_p=0.448, word_ins=2.927, length=2.977, ppl=8.26, wps=130875, ups=2.16, wpb=60572, bsz=2124.2, num_updates=78300, lr=0.000113011, gnorm=0.536, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:10:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:10:45 | INFO | valid | epoch 278 | valid on 'valid' subset | loss 12.441 | nll_loss 11.275 | word_ins 12.204 | length 4.726 | ppl 5561.6 | bleu 30.69 | wps 85192 | wpb 21176.3 | bsz 666.3 | num_updates 78334 | best_bleu 31.53
2023-06-13 23:10:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:10:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint278.pt (epoch 278 @ 78334 updates, score 30.69) (writing took 10.008832532912493 seconds)
2023-06-13 23:10:55 | INFO | fairseq_cli.train | end of epoch 278 (average epoch stats below)
2023-06-13 23:10:55 | INFO | train | epoch 278 | loss 3.047 | nll_loss 1.13 | glat_accu 0.55 | glat_context_p 0.448 | word_ins 2.928 | length 2.964 | ppl 8.26 | wps 115104 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 78334 | lr 0.000112986 | gnorm 0.549 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:10:56 | INFO | fairseq.trainer | begin training epoch 279
2023-06-13 23:11:32 | INFO | train_inner | epoch 279:     66 / 282 loss=3.046, nll_loss=1.13, glat_accu=0.552, glat_context_p=0.448, word_ins=2.927, length=2.965, ppl=8.26, wps=93054.1, ups=1.55, wpb=60137.5, bsz=2151.8, num_updates=78400, lr=0.000112938, gnorm=0.559, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:12:17 | INFO | train_inner | epoch 279:    166 / 282 loss=3.039, nll_loss=1.122, glat_accu=0.549, glat_context_p=0.448, word_ins=2.921, length=2.954, ppl=8.22, wps=133106, ups=2.19, wpb=60836.9, bsz=2153.7, num_updates=78500, lr=0.000112867, gnorm=0.533, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:13:03 | INFO | train_inner | epoch 279:    266 / 282 loss=3.049, nll_loss=1.131, glat_accu=0.557, glat_context_p=0.448, word_ins=2.929, length=2.974, ppl=8.27, wps=132736, ups=2.2, wpb=60421.8, bsz=2181, num_updates=78600, lr=0.000112795, gnorm=0.549, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:13:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:13:13 | INFO | valid | epoch 279 | valid on 'valid' subset | loss 12.542 | nll_loss 11.399 | word_ins 12.313 | length 4.583 | ppl 5964.7 | bleu 30.58 | wps 84820 | wpb 21176.3 | bsz 666.3 | num_updates 78616 | best_bleu 31.53
2023-06-13 23:13:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:13:22 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint279.pt (epoch 279 @ 78616 updates, score 30.58) (writing took 8.619690287858248 seconds)
2023-06-13 23:13:22 | INFO | fairseq_cli.train | end of epoch 279 (average epoch stats below)
2023-06-13 23:13:22 | INFO | train | epoch 279 | loss 3.044 | nll_loss 1.127 | glat_accu 0.552 | glat_context_p 0.448 | word_ins 2.925 | length 2.969 | ppl 8.25 | wps 116321 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 78616 | lr 0.000112783 | gnorm 0.546 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:13:22 | INFO | fairseq.trainer | begin training epoch 280
2023-06-13 23:13:45 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 8192.0
2023-06-13 23:14:07 | INFO | train_inner | epoch 280:     85 / 282 loss=3.039, nll_loss=1.122, glat_accu=0.548, glat_context_p=0.448, word_ins=2.921, length=2.969, ppl=8.22, wps=94107.5, ups=1.57, wpb=60034.3, bsz=2137.2, num_updates=78700, lr=0.000112723, gnorm=0.544, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:14:52 | INFO | train_inner | epoch 280:    185 / 282 loss=3.042, nll_loss=1.126, glat_accu=0.557, glat_context_p=0.448, word_ins=2.924, length=2.946, ppl=8.24, wps=132523, ups=2.19, wpb=60563.4, bsz=2174.6, num_updates=78800, lr=0.000112651, gnorm=0.542, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:15:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:15:40 | INFO | valid | epoch 280 | valid on 'valid' subset | loss 12.488 | nll_loss 11.323 | word_ins 12.252 | length 4.712 | ppl 5744 | bleu 31.2 | wps 88810.5 | wpb 21176.3 | bsz 666.3 | num_updates 78897 | best_bleu 31.53
2023-06-13 23:15:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:15:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint280.pt (epoch 280 @ 78897 updates, score 31.2) (writing took 11.310737166553736 seconds)
2023-06-13 23:15:51 | INFO | fairseq_cli.train | end of epoch 280 (average epoch stats below)
2023-06-13 23:15:51 | INFO | train | epoch 280 | loss 3.042 | nll_loss 1.125 | glat_accu 0.551 | glat_context_p 0.447 | word_ins 2.923 | length 2.957 | ppl 8.24 | wps 113938 | ups 1.89 | wpb 60417.5 | bsz 2158.2 | num_updates 78897 | lr 0.000112582 | gnorm 0.542 | clip 0 | loss_scale 8192 | train_wall 128 | wall 0
2023-06-13 23:15:51 | INFO | fairseq.trainer | begin training epoch 281
2023-06-13 23:15:59 | INFO | train_inner | epoch 281:      3 / 282 loss=3.046, nll_loss=1.13, glat_accu=0.548, glat_context_p=0.447, word_ins=2.927, length=2.968, ppl=8.26, wps=90778.4, ups=1.51, wpb=60105.6, bsz=2135.5, num_updates=78900, lr=0.00011258, gnorm=0.545, clip=0, loss_scale=8192, train_wall=45, wall=0
2023-06-13 23:16:44 | INFO | train_inner | epoch 281:    103 / 282 loss=3.037, nll_loss=1.121, glat_accu=0.56, glat_context_p=0.447, word_ins=2.919, length=2.919, ppl=8.21, wps=133351, ups=2.2, wpb=60579.2, bsz=2206.7, num_updates=79000, lr=0.000112509, gnorm=0.536, clip=0, loss_scale=8192, train_wall=45, wall=0
2023-06-13 23:17:30 | INFO | train_inner | epoch 281:    203 / 282 loss=3.045, nll_loss=1.128, glat_accu=0.556, glat_context_p=0.447, word_ins=2.926, length=2.96, ppl=8.25, wps=131745, ups=2.17, wpb=60580.8, bsz=2166.2, num_updates=79100, lr=0.000112438, gnorm=0.556, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:18:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:18:11 | INFO | valid | epoch 281 | valid on 'valid' subset | loss 12.48 | nll_loss 11.319 | word_ins 12.244 | length 4.717 | ppl 5712.86 | bleu 30.83 | wps 88781 | wpb 21176.3 | bsz 666.3 | num_updates 79179 | best_bleu 31.53
2023-06-13 23:18:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:18:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint281.pt (epoch 281 @ 79179 updates, score 30.83) (writing took 9.207592941820621 seconds)
2023-06-13 23:18:20 | INFO | fairseq_cli.train | end of epoch 281 (average epoch stats below)
2023-06-13 23:18:20 | INFO | train | epoch 281 | loss 3.042 | nll_loss 1.125 | glat_accu 0.55 | glat_context_p 0.447 | word_ins 2.923 | length 2.957 | ppl 8.24 | wps 114403 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 79179 | lr 0.000112382 | gnorm 0.545 | clip 0 | loss_scale 8192 | train_wall 129 | wall 0
2023-06-13 23:18:20 | INFO | fairseq.trainer | begin training epoch 282
2023-06-13 23:18:35 | INFO | train_inner | epoch 282:     21 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.532, glat_context_p=0.447, word_ins=2.923, length=2.991, ppl=8.24, wps=92708.5, ups=1.54, wpb=60089.5, bsz=2106.9, num_updates=79200, lr=0.000112367, gnorm=0.54, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:19:21 | INFO | train_inner | epoch 282:    121 / 282 loss=3.038, nll_loss=1.12, glat_accu=0.549, glat_context_p=0.447, word_ins=2.919, length=2.968, ppl=8.21, wps=132497, ups=2.19, wpb=60618.9, bsz=2159.8, num_updates=79300, lr=0.000112296, gnorm=0.534, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:20:06 | INFO | train_inner | epoch 282:    221 / 282 loss=3.05, nll_loss=1.134, glat_accu=0.558, glat_context_p=0.447, word_ins=2.931, length=2.951, ppl=8.28, wps=132510, ups=2.18, wpb=60685, bsz=2143.8, num_updates=79400, lr=0.000112225, gnorm=0.553, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:20:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:20:37 | INFO | valid | epoch 282 | valid on 'valid' subset | loss 12.464 | nll_loss 11.294 | word_ins 12.219 | length 4.903 | ppl 5649.18 | bleu 30.78 | wps 88540.8 | wpb 21176.3 | bsz 666.3 | num_updates 79461 | best_bleu 31.53
2023-06-13 23:20:37 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:20:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint282.pt (epoch 282 @ 79461 updates, score 30.78) (writing took 12.337058894336224 seconds)
2023-06-13 23:20:50 | INFO | fairseq_cli.train | end of epoch 282 (average epoch stats below)
2023-06-13 23:20:50 | INFO | train | epoch 282 | loss 3.044 | nll_loss 1.128 | glat_accu 0.553 | glat_context_p 0.447 | word_ins 2.925 | length 2.958 | ppl 8.25 | wps 113674 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 79461 | lr 0.000112182 | gnorm 0.543 | clip 0 | loss_scale 8192 | train_wall 129 | wall 0
2023-06-13 23:20:50 | INFO | fairseq.trainer | begin training epoch 283
2023-06-13 23:21:14 | INFO | train_inner | epoch 283:     39 / 282 loss=3.045, nll_loss=1.129, glat_accu=0.553, glat_context_p=0.447, word_ins=2.926, length=2.952, ppl=8.25, wps=89061.6, ups=1.48, wpb=60006.6, bsz=2141.3, num_updates=79500, lr=0.000112154, gnorm=0.544, clip=0, loss_scale=8192, train_wall=45, wall=0
2023-06-13 23:22:00 | INFO | train_inner | epoch 283:    139 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.541, glat_context_p=0.447, word_ins=2.917, length=2.959, ppl=8.2, wps=132125, ups=2.18, wpb=60574.7, bsz=2170.5, num_updates=79600, lr=0.000112084, gnorm=0.55, clip=0, loss_scale=8192, train_wall=46, wall=0
2023-06-13 23:22:45 | INFO | train_inner | epoch 283:    239 / 282 loss=3.033, nll_loss=1.115, glat_accu=0.547, glat_context_p=0.447, word_ins=2.914, length=2.958, ppl=8.18, wps=133360, ups=2.2, wpb=60585.4, bsz=2182.6, num_updates=79700, lr=0.000112014, gnorm=0.535, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:23:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:23:07 | INFO | valid | epoch 283 | valid on 'valid' subset | loss 12.415 | nll_loss 11.247 | word_ins 12.178 | length 4.74 | ppl 5459.59 | bleu 30.92 | wps 88020.4 | wpb 21176.3 | bsz 666.3 | num_updates 79743 | best_bleu 31.53
2023-06-13 23:23:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:23:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint283.pt (epoch 283 @ 79743 updates, score 30.92) (writing took 11.279234949499369 seconds)
2023-06-13 23:23:19 | INFO | fairseq_cli.train | end of epoch 283 (average epoch stats below)
2023-06-13 23:23:19 | INFO | train | epoch 283 | loss 3.036 | nll_loss 1.119 | glat_accu 0.547 | glat_context_p 0.447 | word_ins 2.918 | length 2.962 | ppl 8.2 | wps 114364 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 79743 | lr 0.000111983 | gnorm 0.544 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:23:19 | INFO | fairseq.trainer | begin training epoch 284
2023-06-13 23:23:50 | INFO | train_inner | epoch 284:     57 / 282 loss=3.045, nll_loss=1.127, glat_accu=0.555, glat_context_p=0.447, word_ins=2.925, length=2.984, ppl=8.25, wps=91931.1, ups=1.53, wpb=60139.7, bsz=2160.1, num_updates=79800, lr=0.000111943, gnorm=0.551, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:24:36 | INFO | train_inner | epoch 284:    157 / 282 loss=3.042, nll_loss=1.126, glat_accu=0.553, glat_context_p=0.447, word_ins=2.924, length=2.954, ppl=8.24, wps=132305, ups=2.18, wpb=60671.5, bsz=2137, num_updates=79900, lr=0.000111873, gnorm=0.545, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:25:22 | INFO | train_inner | epoch 284:    257 / 282 loss=3.051, nll_loss=1.135, glat_accu=0.563, glat_context_p=0.447, word_ins=2.932, length=2.956, ppl=8.29, wps=132109, ups=2.18, wpb=60538.3, bsz=2178.9, num_updates=80000, lr=0.000111803, gnorm=0.548, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:25:33 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:25:36 | INFO | valid | epoch 284 | valid on 'valid' subset | loss 12.348 | nll_loss 11.179 | word_ins 12.116 | length 4.65 | ppl 5214.89 | bleu 31.16 | wps 89138.1 | wpb 21176.3 | bsz 666.3 | num_updates 80025 | best_bleu 31.53
2023-06-13 23:25:36 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:25:47 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint284.pt (epoch 284 @ 80025 updates, score 31.16) (writing took 10.240588258951902 seconds)
2023-06-13 23:25:47 | INFO | fairseq_cli.train | end of epoch 284 (average epoch stats below)
2023-06-13 23:25:47 | INFO | train | epoch 284 | loss 3.047 | nll_loss 1.131 | glat_accu 0.558 | glat_context_p 0.447 | word_ins 2.928 | length 2.96 | ppl 8.27 | wps 115121 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 80025 | lr 0.000111786 | gnorm 0.55 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-13 23:25:47 | INFO | fairseq.trainer | begin training epoch 285
2023-06-13 23:26:27 | INFO | train_inner | epoch 285:     75 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.568, glat_context_p=0.447, word_ins=2.93, length=2.94, ppl=8.28, wps=91808, ups=1.53, wpb=60057.5, bsz=2138.4, num_updates=80100, lr=0.000111734, gnorm=0.564, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:27:13 | INFO | train_inner | epoch 285:    175 / 282 loss=3.056, nll_loss=1.139, glat_accu=0.566, glat_context_p=0.447, word_ins=2.935, length=2.976, ppl=8.32, wps=132418, ups=2.18, wpb=60644.2, bsz=2154.5, num_updates=80200, lr=0.000111664, gnorm=0.563, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:27:59 | INFO | train_inner | epoch 285:    275 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.554, glat_context_p=0.447, word_ins=2.93, length=2.974, ppl=8.28, wps=131075, ups=2.17, wpb=60501.3, bsz=2157.8, num_updates=80300, lr=0.000111594, gnorm=0.565, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:28:02 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:28:06 | INFO | valid | epoch 285 | valid on 'valid' subset | loss 12.291 | nll_loss 11.114 | word_ins 12.058 | length 4.677 | ppl 5012.56 | bleu 30.9 | wps 88604.6 | wpb 21176.3 | bsz 666.3 | num_updates 80307 | best_bleu 31.53
2023-06-13 23:28:06 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:28:17 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint285.pt (epoch 285 @ 80307 updates, score 30.9) (writing took 11.1681212708354 seconds)
2023-06-13 23:28:17 | INFO | fairseq_cli.train | end of epoch 285 (average epoch stats below)
2023-06-13 23:28:17 | INFO | train | epoch 285 | loss 3.052 | nll_loss 1.135 | glat_accu 0.564 | glat_context_p 0.447 | word_ins 2.932 | length 2.962 | ppl 8.29 | wps 113556 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 80307 | lr 0.000111589 | gnorm 0.563 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 23:28:17 | INFO | fairseq.trainer | begin training epoch 286
2023-06-13 23:29:07 | INFO | train_inner | epoch 286:     93 / 282 loss=3.045, nll_loss=1.128, glat_accu=0.562, glat_context_p=0.446, word_ins=2.926, length=2.95, ppl=8.26, wps=89444, ups=1.49, wpb=60141.5, bsz=2161.4, num_updates=80400, lr=0.000111525, gnorm=0.552, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:29:52 | INFO | train_inner | epoch 286:    193 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.56, glat_context_p=0.446, word_ins=2.928, length=2.962, ppl=8.27, wps=132591, ups=2.19, wpb=60649.8, bsz=2157.9, num_updates=80500, lr=0.000111456, gnorm=0.55, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-13 23:30:33 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:30:36 | INFO | valid | epoch 286 | valid on 'valid' subset | loss 12.373 | nll_loss 11.2 | word_ins 12.137 | length 4.717 | ppl 5305.11 | bleu 31.31 | wps 87574.6 | wpb 21176.3 | bsz 666.3 | num_updates 80589 | best_bleu 31.53
2023-06-13 23:30:36 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:30:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint286.pt (epoch 286 @ 80589 updates, score 31.31) (writing took 15.275876585394144 seconds)
2023-06-13 23:30:51 | INFO | fairseq_cli.train | end of epoch 286 (average epoch stats below)
2023-06-13 23:30:51 | INFO | train | epoch 286 | loss 3.047 | nll_loss 1.13 | glat_accu 0.56 | glat_context_p 0.446 | word_ins 2.927 | length 2.96 | ppl 8.26 | wps 110069 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 80589 | lr 0.000111394 | gnorm 0.558 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-13 23:30:52 | INFO | fairseq.trainer | begin training epoch 287
2023-06-13 23:31:03 | INFO | train_inner | epoch 287:     11 / 282 loss=3.048, nll_loss=1.131, glat_accu=0.564, glat_context_p=0.446, word_ins=2.928, length=2.958, ppl=8.27, wps=85331.8, ups=1.42, wpb=60004.8, bsz=2168.8, num_updates=80600, lr=0.000111386, gnorm=0.576, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:31:48 | INFO | train_inner | epoch 287:    111 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.559, glat_context_p=0.446, word_ins=2.927, length=2.957, ppl=8.26, wps=132918, ups=2.19, wpb=60633.3, bsz=2154.2, num_updates=80700, lr=0.000111317, gnorm=0.56, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-13 23:32:34 | INFO | train_inner | epoch 287:    211 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.557, glat_context_p=0.446, word_ins=2.929, length=2.973, ppl=8.28, wps=131960, ups=2.18, wpb=60578.3, bsz=2147.1, num_updates=80800, lr=0.000111249, gnorm=0.55, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:33:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:33:10 | INFO | valid | epoch 287 | valid on 'valid' subset | loss 12.483 | nll_loss 11.318 | word_ins 12.243 | length 4.792 | ppl 5723.55 | bleu 31.12 | wps 88842.8 | wpb 21176.3 | bsz 666.3 | num_updates 80871 | best_bleu 31.53
2023-06-13 23:33:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:33:24 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint287.pt (epoch 287 @ 80871 updates, score 31.12) (writing took 14.181212190538645 seconds)
2023-06-13 23:33:24 | INFO | fairseq_cli.train | end of epoch 287 (average epoch stats below)
2023-06-13 23:33:24 | INFO | train | epoch 287 | loss 3.048 | nll_loss 1.132 | glat_accu 0.561 | glat_context_p 0.446 | word_ins 2.929 | length 2.959 | ppl 8.27 | wps 111372 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 80871 | lr 0.0001112 | gnorm 0.561 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:33:25 | INFO | fairseq.trainer | begin training epoch 288
2023-06-13 23:33:44 | INFO | train_inner | epoch 288:     29 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.564, glat_context_p=0.446, word_ins=2.927, length=2.951, ppl=8.26, wps=86246.3, ups=1.44, wpb=60073.4, bsz=2167.4, num_updates=80900, lr=0.00011118, gnorm=0.566, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:34:30 | INFO | train_inner | epoch 288:    129 / 282 loss=3.052, nll_loss=1.135, glat_accu=0.567, glat_context_p=0.446, word_ins=2.932, length=2.957, ppl=8.29, wps=131924, ups=2.18, wpb=60623, bsz=2159.3, num_updates=81000, lr=0.000111111, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:35:16 | INFO | train_inner | epoch 288:    229 / 282 loss=3.05, nll_loss=1.134, glat_accu=0.569, glat_context_p=0.446, word_ins=2.931, length=2.948, ppl=8.28, wps=132730, ups=2.19, wpb=60582.1, bsz=2169.6, num_updates=81100, lr=0.000111043, gnorm=0.563, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:35:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:35:43 | INFO | valid | epoch 288 | valid on 'valid' subset | loss 12.396 | nll_loss 11.233 | word_ins 12.165 | length 4.598 | ppl 5389.57 | bleu 31.18 | wps 87696.4 | wpb 21176.3 | bsz 666.3 | num_updates 81153 | best_bleu 31.53
2023-06-13 23:35:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:35:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint288.pt (epoch 288 @ 81153 updates, score 31.18) (writing took 12.743327889591455 seconds)
2023-06-13 23:35:56 | INFO | fairseq_cli.train | end of epoch 288 (average epoch stats below)
2023-06-13 23:35:56 | INFO | train | epoch 288 | loss 3.049 | nll_loss 1.132 | glat_accu 0.561 | glat_context_p 0.446 | word_ins 2.929 | length 2.959 | ppl 8.28 | wps 112372 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 81153 | lr 0.000111006 | gnorm 0.557 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:35:56 | INFO | fairseq.trainer | begin training epoch 289
2023-06-13 23:36:24 | INFO | train_inner | epoch 289:     47 / 282 loss=3.045, nll_loss=1.128, glat_accu=0.551, glat_context_p=0.446, word_ins=2.926, length=2.969, ppl=8.25, wps=87108.9, ups=1.45, wpb=59970.4, bsz=2135.8, num_updates=81200, lr=0.000110974, gnorm=0.558, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:37:10 | INFO | train_inner | epoch 289:    147 / 282 loss=3.04, nll_loss=1.123, glat_accu=0.567, glat_context_p=0.446, word_ins=2.921, length=2.94, ppl=8.23, wps=133580, ups=2.21, wpb=60555.1, bsz=2206.6, num_updates=81300, lr=0.000110906, gnorm=0.551, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:37:56 | INFO | train_inner | epoch 289:    247 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.559, glat_context_p=0.446, word_ins=2.927, length=2.965, ppl=8.27, wps=132034, ups=2.18, wpb=60633.6, bsz=2148, num_updates=81400, lr=0.000110838, gnorm=0.555, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:38:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:38:15 | INFO | valid | epoch 289 | valid on 'valid' subset | loss 12.541 | nll_loss 11.385 | word_ins 12.3 | length 4.787 | ppl 5958.63 | bleu 30.87 | wps 86423.3 | wpb 21176.3 | bsz 666.3 | num_updates 81435 | best_bleu 31.53
2023-06-13 23:38:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:38:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint289.pt (epoch 289 @ 81435 updates, score 30.87) (writing took 12.278245627880096 seconds)
2023-06-13 23:38:27 | INFO | fairseq_cli.train | end of epoch 289 (average epoch stats below)
2023-06-13 23:38:27 | INFO | train | epoch 289 | loss 3.044 | nll_loss 1.127 | glat_accu 0.56 | glat_context_p 0.446 | word_ins 2.925 | length 2.956 | ppl 8.25 | wps 112931 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 81435 | lr 0.000110814 | gnorm 0.552 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:38:27 | INFO | fairseq.trainer | begin training epoch 290
2023-06-13 23:39:04 | INFO | train_inner | epoch 290:     65 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.556, glat_context_p=0.446, word_ins=2.929, length=2.967, ppl=8.27, wps=87838.3, ups=1.46, wpb=60256, bsz=2114.3, num_updates=81500, lr=0.00011077, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:39:50 | INFO | train_inner | epoch 290:    165 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.562, glat_context_p=0.446, word_ins=2.923, length=2.96, ppl=8.24, wps=131912, ups=2.18, wpb=60475.2, bsz=2167.4, num_updates=81600, lr=0.000110702, gnorm=0.543, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:40:35 | INFO | train_inner | epoch 290:    265 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.564, glat_context_p=0.446, word_ins=2.925, length=2.951, ppl=8.25, wps=133590, ups=2.21, wpb=60564.6, bsz=2164.4, num_updates=81700, lr=0.000110634, gnorm=0.561, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:40:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:40:46 | INFO | valid | epoch 290 | valid on 'valid' subset | loss 12.331 | nll_loss 11.155 | word_ins 12.097 | length 4.694 | ppl 5150.63 | bleu 31.04 | wps 89997 | wpb 21176.3 | bsz 666.3 | num_updates 81717 | best_bleu 31.53
2023-06-13 23:40:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:40:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint290.pt (epoch 290 @ 81717 updates, score 31.04) (writing took 11.678300105035305 seconds)
2023-06-13 23:40:58 | INFO | fairseq_cli.train | end of epoch 290 (average epoch stats below)
2023-06-13 23:40:58 | INFO | train | epoch 290 | loss 3.045 | nll_loss 1.128 | glat_accu 0.563 | glat_context_p 0.446 | word_ins 2.926 | length 2.955 | ppl 8.25 | wps 112952 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 81717 | lr 0.000110623 | gnorm 0.554 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 23:40:58 | INFO | fairseq.trainer | begin training epoch 291
2023-06-13 23:41:09 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 23:41:43 | INFO | train_inner | epoch 291:     84 / 282 loss=3.054, nll_loss=1.137, glat_accu=0.558, glat_context_p=0.446, word_ins=2.934, length=2.984, ppl=8.31, wps=89029, ups=1.48, wpb=59975, bsz=2068.6, num_updates=81800, lr=0.000110566, gnorm=0.57, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:42:28 | INFO | train_inner | epoch 291:    184 / 282 loss=3.044, nll_loss=1.129, glat_accu=0.575, glat_context_p=0.445, word_ins=2.926, length=2.915, ppl=8.25, wps=133178, ups=2.19, wpb=60730.4, bsz=2251.2, num_updates=81900, lr=0.000110499, gnorm=0.543, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:43:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:43:17 | INFO | valid | epoch 291 | valid on 'valid' subset | loss 12.336 | nll_loss 11.165 | word_ins 12.104 | length 4.625 | ppl 5169.04 | bleu 31.33 | wps 88134.4 | wpb 21176.3 | bsz 666.3 | num_updates 81998 | best_bleu 31.53
2023-06-13 23:43:17 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:43:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint291.pt (epoch 291 @ 81998 updates, score 31.33) (writing took 11.66888527199626 seconds)
2023-06-13 23:43:28 | INFO | fairseq_cli.train | end of epoch 291 (average epoch stats below)
2023-06-13 23:43:28 | INFO | train | epoch 291 | loss 3.049 | nll_loss 1.133 | glat_accu 0.567 | glat_context_p 0.445 | word_ins 2.93 | length 2.954 | ppl 8.28 | wps 112712 | ups 1.87 | wpb 60407.6 | bsz 2158.3 | num_updates 81998 | lr 0.000110433 | gnorm 0.555 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:43:28 | INFO | fairseq.trainer | begin training epoch 292
2023-06-13 23:43:36 | INFO | train_inner | epoch 292:      2 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.568, glat_context_p=0.445, word_ins=2.929, length=2.959, ppl=8.28, wps=88793.9, ups=1.48, wpb=60086.9, bsz=2130, num_updates=82000, lr=0.000110432, gnorm=0.556, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:44:22 | INFO | train_inner | epoch 292:    102 / 282 loss=3.054, nll_loss=1.138, glat_accu=0.57, glat_context_p=0.445, word_ins=2.934, length=2.945, ppl=8.31, wps=132906, ups=2.19, wpb=60625.5, bsz=2172.1, num_updates=82100, lr=0.000110364, gnorm=0.562, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:45:08 | INFO | train_inner | epoch 292:    202 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.569, glat_context_p=0.445, word_ins=2.924, length=2.948, ppl=8.25, wps=131671, ups=2.18, wpb=60519.2, bsz=2159.7, num_updates=82200, lr=0.000110297, gnorm=0.558, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:45:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:45:47 | INFO | valid | epoch 292 | valid on 'valid' subset | loss 12.283 | nll_loss 11.108 | word_ins 12.054 | length 4.589 | ppl 4985.39 | bleu 31.66 | wps 88480.7 | wpb 21176.3 | bsz 666.3 | num_updates 82280 | best_bleu 31.66
2023-06-13 23:45:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:46:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint292.pt (epoch 292 @ 82280 updates, score 31.66) (writing took 17.965789631009102 seconds)
2023-06-13 23:46:05 | INFO | fairseq_cli.train | end of epoch 292 (average epoch stats below)
2023-06-13 23:46:05 | INFO | train | epoch 292 | loss 3.049 | nll_loss 1.132 | glat_accu 0.567 | glat_context_p 0.445 | word_ins 2.929 | length 2.953 | ppl 8.28 | wps 108611 | ups 1.8 | wpb 60413.8 | bsz 2157.2 | num_updates 82280 | lr 0.000110243 | gnorm 0.566 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 23:46:05 | INFO | fairseq.trainer | begin training epoch 293
2023-06-13 23:46:21 | INFO | train_inner | epoch 293:     20 / 282 loss=3.049, nll_loss=1.132, glat_accu=0.563, glat_context_p=0.445, word_ins=2.929, length=2.966, ppl=8.28, wps=82191, ups=1.37, wpb=60076.1, bsz=2141.7, num_updates=82300, lr=0.00011023, gnorm=0.578, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:47:07 | INFO | train_inner | epoch 293:    120 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.56, glat_context_p=0.445, word_ins=2.925, length=2.966, ppl=8.25, wps=130864, ups=2.16, wpb=60619.5, bsz=2123.1, num_updates=82400, lr=0.000110163, gnorm=0.556, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:47:53 | INFO | train_inner | epoch 293:    220 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.566, glat_context_p=0.445, word_ins=2.923, length=2.952, ppl=8.24, wps=133240, ups=2.2, wpb=60648.6, bsz=2180.4, num_updates=82500, lr=0.000110096, gnorm=0.55, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:48:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:48:24 | INFO | valid | epoch 293 | valid on 'valid' subset | loss 12.348 | nll_loss 11.181 | word_ins 12.118 | length 4.624 | ppl 5213.12 | bleu 31.58 | wps 88646.2 | wpb 21176.3 | bsz 666.3 | num_updates 82562 | best_bleu 31.66
2023-06-13 23:48:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:48:35 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint293.pt (epoch 293 @ 82562 updates, score 31.58) (writing took 10.9375434294343 seconds)
2023-06-13 23:48:35 | INFO | fairseq_cli.train | end of epoch 293 (average epoch stats below)
2023-06-13 23:48:35 | INFO | train | epoch 293 | loss 3.045 | nll_loss 1.127 | glat_accu 0.566 | glat_context_p 0.445 | word_ins 2.925 | length 2.957 | ppl 8.25 | wps 113832 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 82562 | lr 0.000110055 | gnorm 0.555 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:48:35 | INFO | fairseq.trainer | begin training epoch 294
2023-06-13 23:48:58 | INFO | train_inner | epoch 294:     38 / 282 loss=3.043, nll_loss=1.125, glat_accu=0.573, glat_context_p=0.445, word_ins=2.923, length=2.939, ppl=8.24, wps=91572.9, ups=1.53, wpb=60031.3, bsz=2181.7, num_updates=82600, lr=0.00011003, gnorm=0.553, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:49:44 | INFO | train_inner | epoch 294:    138 / 282 loss=3.045, nll_loss=1.128, glat_accu=0.564, glat_context_p=0.445, word_ins=2.926, length=2.955, ppl=8.26, wps=132067, ups=2.17, wpb=60794.2, bsz=2147.8, num_updates=82700, lr=0.000109963, gnorm=0.556, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:50:08 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 23:50:31 | INFO | train_inner | epoch 294:    239 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.565, glat_context_p=0.445, word_ins=2.926, length=2.956, ppl=8.26, wps=128965, ups=2.13, wpb=60439.4, bsz=2158.2, num_updates=82800, lr=0.000109897, gnorm=0.561, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-13 23:50:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:50:54 | INFO | valid | epoch 294 | valid on 'valid' subset | loss 12.3 | nll_loss 11.131 | word_ins 12.073 | length 4.552 | ppl 5044.15 | bleu 31.56 | wps 88392.7 | wpb 21176.3 | bsz 666.3 | num_updates 82843 | best_bleu 31.66
2023-06-13 23:50:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:51:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint294.pt (epoch 294 @ 82843 updates, score 31.56) (writing took 10.851539719849825 seconds)
2023-06-13 23:51:05 | INFO | fairseq_cli.train | end of epoch 294 (average epoch stats below)
2023-06-13 23:51:05 | INFO | train | epoch 294 | loss 3.045 | nll_loss 1.128 | glat_accu 0.566 | glat_context_p 0.445 | word_ins 2.926 | length 2.953 | ppl 8.25 | wps 113190 | ups 1.87 | wpb 60413.5 | bsz 2157.7 | num_updates 82843 | lr 0.000109868 | gnorm 0.558 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:51:05 | INFO | fairseq.trainer | begin training epoch 295
2023-06-13 23:51:39 | INFO | train_inner | epoch 295:     57 / 282 loss=3.039, nll_loss=1.121, glat_accu=0.56, glat_context_p=0.445, word_ins=2.919, length=2.969, ppl=8.22, wps=88604.9, ups=1.48, wpb=59898, bsz=2163.4, num_updates=82900, lr=0.00010983, gnorm=0.556, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:52:25 | INFO | train_inner | epoch 295:    157 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.557, glat_context_p=0.445, word_ins=2.917, length=2.964, ppl=8.2, wps=131496, ups=2.17, wpb=60519.8, bsz=2166.5, num_updates=83000, lr=0.000109764, gnorm=0.545, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:53:11 | INFO | train_inner | epoch 295:    257 / 282 loss=3.039, nll_loss=1.123, glat_accu=0.562, glat_context_p=0.445, word_ins=2.921, length=2.935, ppl=8.22, wps=131906, ups=2.17, wpb=60795.6, bsz=2160, num_updates=83100, lr=0.000109698, gnorm=0.559, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:53:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:53:25 | INFO | valid | epoch 295 | valid on 'valid' subset | loss 12.449 | nll_loss 11.286 | word_ins 12.211 | length 4.766 | ppl 5591.91 | bleu 30.79 | wps 88597.7 | wpb 21176.3 | bsz 666.3 | num_updates 83125 | best_bleu 31.66
2023-06-13 23:53:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:53:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint295.pt (epoch 295 @ 83125 updates, score 30.79) (writing took 10.270485889166594 seconds)
2023-06-13 23:53:36 | INFO | fairseq_cli.train | end of epoch 295 (average epoch stats below)
2023-06-13 23:53:36 | INFO | train | epoch 295 | loss 3.037 | nll_loss 1.12 | glat_accu 0.559 | glat_context_p 0.445 | word_ins 2.918 | length 2.954 | ppl 8.21 | wps 112963 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 83125 | lr 0.000109682 | gnorm 0.553 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:53:36 | INFO | fairseq.trainer | begin training epoch 296
2023-06-13 23:54:16 | INFO | train_inner | epoch 296:     75 / 282 loss=3.041, nll_loss=1.123, glat_accu=0.561, glat_context_p=0.445, word_ins=2.921, length=2.956, ppl=8.23, wps=91476.9, ups=1.52, wpb=60080.6, bsz=2136.1, num_updates=83200, lr=0.000109632, gnorm=0.561, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:55:03 | INFO | train_inner | epoch 296:    175 / 282 loss=3.038, nll_loss=1.121, glat_accu=0.561, glat_context_p=0.445, word_ins=2.919, length=2.947, ppl=8.21, wps=131492, ups=2.17, wpb=60723.6, bsz=2177, num_updates=83300, lr=0.000109566, gnorm=0.548, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:55:49 | INFO | train_inner | epoch 296:    275 / 282 loss=3.051, nll_loss=1.134, glat_accu=0.567, glat_context_p=0.444, word_ins=2.93, length=2.955, ppl=8.29, wps=131682, ups=2.18, wpb=60510.6, bsz=2144.2, num_updates=83400, lr=0.000109501, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-13 23:55:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:55:55 | INFO | valid | epoch 296 | valid on 'valid' subset | loss 12.463 | nll_loss 11.306 | word_ins 12.231 | length 4.642 | ppl 5646.31 | bleu 31.02 | wps 89373.4 | wpb 21176.3 | bsz 666.3 | num_updates 83407 | best_bleu 31.66
2023-06-13 23:55:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:56:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint296.pt (epoch 296 @ 83407 updates, score 31.02) (writing took 9.31594830751419 seconds)
2023-06-13 23:56:04 | INFO | fairseq_cli.train | end of epoch 296 (average epoch stats below)
2023-06-13 23:56:04 | INFO | train | epoch 296 | loss 3.043 | nll_loss 1.126 | glat_accu 0.563 | glat_context_p 0.444 | word_ins 2.923 | length 2.951 | ppl 8.24 | wps 114936 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 83407 | lr 0.000109496 | gnorm 0.555 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-13 23:56:04 | INFO | fairseq.trainer | begin training epoch 297
2023-06-13 23:56:52 | INFO | train_inner | epoch 297:     93 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.563, glat_context_p=0.444, word_ins=2.926, length=2.938, ppl=8.26, wps=94924.9, ups=1.58, wpb=60107.8, bsz=2165.4, num_updates=83500, lr=0.000109435, gnorm=0.562, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:57:37 | INFO | train_inner | epoch 297:    193 / 282 loss=3.048, nll_loss=1.131, glat_accu=0.569, glat_context_p=0.444, word_ins=2.928, length=2.94, ppl=8.27, wps=133609, ups=2.2, wpb=60712, bsz=2188.3, num_updates=83600, lr=0.00010937, gnorm=0.567, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:58:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-13 23:58:21 | INFO | valid | epoch 297 | valid on 'valid' subset | loss 12.358 | nll_loss 11.195 | word_ins 12.132 | length 4.548 | ppl 5249.55 | bleu 31.59 | wps 87490.8 | wpb 21176.3 | bsz 666.3 | num_updates 83689 | best_bleu 31.66
2023-06-13 23:58:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-13 23:58:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint297.pt (epoch 297 @ 83689 updates, score 31.59) (writing took 11.170998346060514 seconds)
2023-06-13 23:58:32 | INFO | fairseq_cli.train | end of epoch 297 (average epoch stats below)
2023-06-13 23:58:32 | INFO | train | epoch 297 | loss 3.051 | nll_loss 1.134 | glat_accu 0.565 | glat_context_p 0.444 | word_ins 2.931 | length 2.952 | ppl 8.29 | wps 114732 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 83689 | lr 0.000109311 | gnorm 0.566 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-13 23:58:33 | INFO | fairseq.trainer | begin training epoch 298
2023-06-13 23:58:44 | INFO | train_inner | epoch 298:     11 / 282 loss=3.057, nll_loss=1.14, glat_accu=0.564, glat_context_p=0.444, word_ins=2.937, length=2.971, ppl=8.32, wps=90480.8, ups=1.51, wpb=59963.9, bsz=2103.6, num_updates=83700, lr=0.000109304, gnorm=0.575, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-13 23:59:18 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-13 23:59:30 | INFO | train_inner | epoch 298:    112 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.576, glat_context_p=0.444, word_ins=2.926, length=2.934, ppl=8.26, wps=131288, ups=2.16, wpb=60685, bsz=2248, num_updates=83800, lr=0.000109239, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:00:16 | INFO | train_inner | epoch 298:    212 / 282 loss=3.052, nll_loss=1.135, glat_accu=0.568, glat_context_p=0.444, word_ins=2.932, length=2.963, ppl=8.29, wps=132341, ups=2.18, wpb=60573.4, bsz=2129.7, num_updates=83900, lr=0.000109174, gnorm=0.566, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:00:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:00:52 | INFO | valid | epoch 298 | valid on 'valid' subset | loss 12.531 | nll_loss 11.381 | word_ins 12.301 | length 4.62 | ppl 5920.29 | bleu 31.52 | wps 88145.7 | wpb 21176.3 | bsz 666.3 | num_updates 83970 | best_bleu 31.66
2023-06-14 00:00:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:01:03 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint298.pt (epoch 298 @ 83970 updates, score 31.52) (writing took 11.45357083529234 seconds)
2023-06-14 00:01:03 | INFO | fairseq_cli.train | end of epoch 298 (average epoch stats below)
2023-06-14 00:01:03 | INFO | train | epoch 298 | loss 3.052 | nll_loss 1.135 | glat_accu 0.569 | glat_context_p 0.444 | word_ins 2.931 | length 2.955 | ppl 8.29 | wps 112535 | ups 1.86 | wpb 60412.1 | bsz 2157 | num_updates 83970 | lr 0.000109128 | gnorm 0.565 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:01:03 | INFO | fairseq.trainer | begin training epoch 299
2023-06-14 00:01:23 | INFO | train_inner | epoch 299:     30 / 282 loss=3.059, nll_loss=1.142, glat_accu=0.559, glat_context_p=0.444, word_ins=2.938, length=2.984, ppl=8.33, wps=88531.3, ups=1.47, wpb=60022.1, bsz=2077.9, num_updates=84000, lr=0.000109109, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:02:09 | INFO | train_inner | epoch 299:    130 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.575, glat_context_p=0.444, word_ins=2.925, length=2.927, ppl=8.25, wps=133281, ups=2.2, wpb=60556.2, bsz=2201.8, num_updates=84100, lr=0.000109044, gnorm=0.566, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:02:55 | INFO | train_inner | epoch 299:    230 / 282 loss=3.058, nll_loss=1.14, glat_accu=0.569, glat_context_p=0.444, word_ins=2.936, length=2.974, ppl=8.33, wps=131825, ups=2.17, wpb=60711, bsz=2122.7, num_updates=84200, lr=0.000108979, gnorm=0.569, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:03:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:03:22 | INFO | valid | epoch 299 | valid on 'valid' subset | loss 12.371 | nll_loss 11.204 | word_ins 12.136 | length 4.714 | ppl 5297.91 | bleu 31.18 | wps 88115.8 | wpb 21176.3 | bsz 666.3 | num_updates 84252 | best_bleu 31.66
2023-06-14 00:03:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:03:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint299.pt (epoch 299 @ 84252 updates, score 31.18) (writing took 9.699880745261908 seconds)
2023-06-14 00:03:32 | INFO | fairseq_cli.train | end of epoch 299 (average epoch stats below)
2023-06-14 00:03:32 | INFO | train | epoch 299 | loss 3.051 | nll_loss 1.134 | glat_accu 0.57 | glat_context_p 0.444 | word_ins 2.931 | length 2.953 | ppl 8.29 | wps 114809 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 84252 | lr 0.000108946 | gnorm 0.566 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:03:32 | INFO | fairseq.trainer | begin training epoch 300
2023-06-14 00:03:59 | INFO | train_inner | epoch 300:     48 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.568, glat_context_p=0.444, word_ins=2.924, length=2.935, ppl=8.24, wps=93561.7, ups=1.56, wpb=60064.7, bsz=2166.6, num_updates=84300, lr=0.000108915, gnorm=0.565, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:04:45 | INFO | train_inner | epoch 300:    148 / 282 loss=3.05, nll_loss=1.132, glat_accu=0.563, glat_context_p=0.444, word_ins=2.929, length=2.968, ppl=8.28, wps=131111, ups=2.17, wpb=60407.1, bsz=2149.7, num_updates=84400, lr=0.00010885, gnorm=0.548, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:05:31 | INFO | train_inner | epoch 300:    248 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.571, glat_context_p=0.444, word_ins=2.926, length=2.932, ppl=8.26, wps=133096, ups=2.19, wpb=60655.2, bsz=2194.3, num_updates=84500, lr=0.000108786, gnorm=0.557, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:05:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:05:49 | INFO | valid | epoch 300 | valid on 'valid' subset | loss 12.221 | nll_loss 11.039 | word_ins 11.993 | length 4.577 | ppl 4775.71 | bleu 31.57 | wps 87773.5 | wpb 21176.3 | bsz 666.3 | num_updates 84534 | best_bleu 31.66
2023-06-14 00:05:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:06:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint300.pt (epoch 300 @ 84534 updates, score 31.57) (writing took 11.87349433451891 seconds)
2023-06-14 00:06:01 | INFO | fairseq_cli.train | end of epoch 300 (average epoch stats below)
2023-06-14 00:06:01 | INFO | train | epoch 300 | loss 3.047 | nll_loss 1.13 | glat_accu 0.566 | glat_context_p 0.444 | word_ins 2.927 | length 2.956 | ppl 8.27 | wps 114284 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 84534 | lr 0.000108764 | gnorm 0.558 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:06:01 | INFO | fairseq.trainer | begin training epoch 301
2023-06-14 00:06:37 | INFO | train_inner | epoch 301:     66 / 282 loss=3.049, nll_loss=1.131, glat_accu=0.575, glat_context_p=0.444, word_ins=2.928, length=2.961, ppl=8.28, wps=90303.6, ups=1.51, wpb=59990.7, bsz=2152.9, num_updates=84600, lr=0.000108721, gnorm=0.565, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:07:23 | INFO | train_inner | epoch 301:    166 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.561, glat_context_p=0.444, word_ins=2.926, length=2.957, ppl=8.26, wps=131866, ups=2.18, wpb=60581.7, bsz=2156.9, num_updates=84700, lr=0.000108657, gnorm=0.556, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:08:08 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 00:08:09 | INFO | train_inner | epoch 301:    267 / 282 loss=3.045, nll_loss=1.128, glat_accu=0.564, glat_context_p=0.444, word_ins=2.925, length=2.94, ppl=8.25, wps=132110, ups=2.18, wpb=60675.2, bsz=2165.4, num_updates=84800, lr=0.000108593, gnorm=0.551, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:08:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:08:19 | INFO | valid | epoch 301 | valid on 'valid' subset | loss 12.427 | nll_loss 11.26 | word_ins 12.19 | length 4.735 | ppl 5506.15 | bleu 31.17 | wps 86372.1 | wpb 21176.3 | bsz 666.3 | num_updates 84815 | best_bleu 31.66
2023-06-14 00:08:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:08:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint301.pt (epoch 301 @ 84815 updates, score 31.17) (writing took 13.178721033036709 seconds)
2023-06-14 00:08:32 | INFO | fairseq_cli.train | end of epoch 301 (average epoch stats below)
2023-06-14 00:08:32 | INFO | train | epoch 301 | loss 3.046 | nll_loss 1.129 | glat_accu 0.567 | glat_context_p 0.444 | word_ins 2.926 | length 2.943 | ppl 8.26 | wps 111949 | ups 1.85 | wpb 60413.9 | bsz 2157.5 | num_updates 84815 | lr 0.000108583 | gnorm 0.559 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:08:32 | INFO | fairseq.trainer | begin training epoch 302
2023-06-14 00:09:17 | INFO | train_inner | epoch 302:     85 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.572, glat_context_p=0.443, word_ins=2.926, length=2.927, ppl=8.26, wps=88225.4, ups=1.46, wpb=60267.7, bsz=2146.2, num_updates=84900, lr=0.000108529, gnorm=0.569, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:10:03 | INFO | train_inner | epoch 302:    185 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.567, glat_context_p=0.443, word_ins=2.924, length=2.942, ppl=8.25, wps=131400, ups=2.17, wpb=60631.1, bsz=2182.8, num_updates=85000, lr=0.000108465, gnorm=0.554, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:10:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:10:51 | INFO | valid | epoch 302 | valid on 'valid' subset | loss 12.475 | nll_loss 11.318 | word_ins 12.242 | length 4.678 | ppl 5694.46 | bleu 31.07 | wps 87513.8 | wpb 21176.3 | bsz 666.3 | num_updates 85097 | best_bleu 31.66
2023-06-14 00:10:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:11:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint302.pt (epoch 302 @ 85097 updates, score 31.07) (writing took 11.060291070491076 seconds)
2023-06-14 00:11:02 | INFO | fairseq_cli.train | end of epoch 302 (average epoch stats below)
2023-06-14 00:11:02 | INFO | train | epoch 302 | loss 3.046 | nll_loss 1.129 | glat_accu 0.567 | glat_context_p 0.443 | word_ins 2.926 | length 2.947 | ppl 8.26 | wps 113613 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 85097 | lr 0.000108403 | gnorm 0.561 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:11:02 | INFO | fairseq.trainer | begin training epoch 303
2023-06-14 00:11:10 | INFO | train_inner | epoch 303:      3 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.56, glat_context_p=0.443, word_ins=2.929, length=2.967, ppl=8.28, wps=90163.2, ups=1.5, wpb=59970.6, bsz=2111.7, num_updates=85100, lr=0.000108401, gnorm=0.569, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:11:56 | INFO | train_inner | epoch 303:    103 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.56, glat_context_p=0.443, word_ins=2.914, length=2.951, ppl=8.19, wps=132556, ups=2.2, wpb=60351.2, bsz=2152.6, num_updates=85200, lr=0.000108338, gnorm=0.571, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:12:41 | INFO | train_inner | epoch 303:    203 / 282 loss=3.04, nll_loss=1.124, glat_accu=0.556, glat_context_p=0.443, word_ins=2.922, length=2.944, ppl=8.23, wps=132477, ups=2.18, wpb=60831, bsz=2176.4, num_updates=85300, lr=0.000108274, gnorm=0.558, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:13:17 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:13:21 | INFO | valid | epoch 303 | valid on 'valid' subset | loss 12.473 | nll_loss 11.323 | word_ins 12.242 | length 4.619 | ppl 5685.76 | bleu 30.76 | wps 87525.9 | wpb 21176.3 | bsz 666.3 | num_updates 85379 | best_bleu 31.66
2023-06-14 00:13:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:13:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint303.pt (epoch 303 @ 85379 updates, score 30.76) (writing took 10.744830448180437 seconds)
2023-06-14 00:13:31 | INFO | fairseq_cli.train | end of epoch 303 (average epoch stats below)
2023-06-14 00:13:31 | INFO | train | epoch 303 | loss 3.039 | nll_loss 1.122 | glat_accu 0.559 | glat_context_p 0.443 | word_ins 2.92 | length 2.95 | ppl 8.22 | wps 114387 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 85379 | lr 0.000108224 | gnorm 0.569 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:13:31 | INFO | fairseq.trainer | begin training epoch 304
2023-06-14 00:13:48 | INFO | train_inner | epoch 304:     21 / 282 loss=3.04, nll_loss=1.122, glat_accu=0.56, glat_context_p=0.443, word_ins=2.92, length=2.952, ppl=8.22, wps=90464.6, ups=1.51, wpb=60100, bsz=2146.2, num_updates=85400, lr=0.000108211, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:14:34 | INFO | train_inner | epoch 304:    121 / 282 loss=3.034, nll_loss=1.117, glat_accu=0.559, glat_context_p=0.443, word_ins=2.915, length=2.939, ppl=8.19, wps=130629, ups=2.16, wpb=60476.8, bsz=2213.6, num_updates=85500, lr=0.000108148, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:15:20 | INFO | train_inner | epoch 304:    221 / 282 loss=3.051, nll_loss=1.134, glat_accu=0.572, glat_context_p=0.443, word_ins=2.93, length=2.958, ppl=8.29, wps=133155, ups=2.19, wpb=60728.1, bsz=2134.5, num_updates=85600, lr=0.000108084, gnorm=0.568, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:15:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:15:51 | INFO | valid | epoch 304 | valid on 'valid' subset | loss 12.413 | nll_loss 11.239 | word_ins 12.177 | length 4.735 | ppl 5453.1 | bleu 31.39 | wps 88824.1 | wpb 21176.3 | bsz 666.3 | num_updates 85661 | best_bleu 31.66
2023-06-14 00:15:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:16:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint304.pt (epoch 304 @ 85661 updates, score 31.39) (writing took 11.111887589097023 seconds)
2023-06-14 00:16:02 | INFO | fairseq_cli.train | end of epoch 304 (average epoch stats below)
2023-06-14 00:16:02 | INFO | train | epoch 304 | loss 3.043 | nll_loss 1.125 | glat_accu 0.565 | glat_context_p 0.443 | word_ins 2.923 | length 2.952 | ppl 8.24 | wps 112977 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 85661 | lr 0.000108046 | gnorm 0.566 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:16:02 | INFO | fairseq.trainer | begin training epoch 305
2023-06-14 00:16:27 | INFO | train_inner | epoch 305:     39 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.568, glat_context_p=0.443, word_ins=2.927, length=2.962, ppl=8.27, wps=88908.1, ups=1.48, wpb=60102.3, bsz=2111.8, num_updates=85700, lr=0.000108021, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:17:13 | INFO | train_inner | epoch 305:    139 / 282 loss=3.043, nll_loss=1.124, glat_accu=0.57, glat_context_p=0.443, word_ins=2.922, length=2.955, ppl=8.24, wps=132055, ups=2.18, wpb=60457.3, bsz=2166.5, num_updates=85800, lr=0.000107958, gnorm=0.562, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:17:24 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 00:17:59 | INFO | train_inner | epoch 305:    240 / 282 loss=3.038, nll_loss=1.121, glat_accu=0.568, glat_context_p=0.443, word_ins=2.919, length=2.937, ppl=8.22, wps=130707, ups=2.16, wpb=60567.8, bsz=2193.8, num_updates=85900, lr=0.000107896, gnorm=0.55, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:18:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:18:22 | INFO | valid | epoch 305 | valid on 'valid' subset | loss 12.318 | nll_loss 11.149 | word_ins 12.088 | length 4.598 | ppl 5106.21 | bleu 31.35 | wps 88766.9 | wpb 21176.3 | bsz 666.3 | num_updates 85942 | best_bleu 31.66
2023-06-14 00:18:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:18:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint305.pt (epoch 305 @ 85942 updates, score 31.35) (writing took 13.766596760600805 seconds)
2023-06-14 00:18:36 | INFO | fairseq_cli.train | end of epoch 305 (average epoch stats below)
2023-06-14 00:18:36 | INFO | train | epoch 305 | loss 3.042 | nll_loss 1.124 | glat_accu 0.568 | glat_context_p 0.443 | word_ins 2.922 | length 2.946 | ppl 8.24 | wps 110597 | ups 1.83 | wpb 60412.9 | bsz 2157.5 | num_updates 85942 | lr 0.000107869 | gnorm 0.562 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:18:36 | INFO | fairseq.trainer | begin training epoch 306
2023-06-14 00:19:09 | INFO | train_inner | epoch 306:     58 / 282 loss=3.045, nll_loss=1.127, glat_accu=0.557, glat_context_p=0.443, word_ins=2.925, length=2.956, ppl=8.25, wps=87205.6, ups=1.45, wpb=60210.7, bsz=2092.8, num_updates=86000, lr=0.000107833, gnorm=0.594, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:19:54 | INFO | train_inner | epoch 306:    158 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.573, glat_context_p=0.443, word_ins=2.923, length=2.943, ppl=8.25, wps=132037, ups=2.18, wpb=60606.5, bsz=2174, num_updates=86100, lr=0.00010777, gnorm=0.565, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:20:40 | INFO | train_inner | epoch 306:    258 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.567, glat_context_p=0.443, word_ins=2.923, length=2.946, ppl=8.24, wps=131849, ups=2.18, wpb=60539.7, bsz=2186.3, num_updates=86200, lr=0.000107708, gnorm=0.551, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:20:51 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:20:54 | INFO | valid | epoch 306 | valid on 'valid' subset | loss 12.34 | nll_loss 11.158 | word_ins 12.093 | length 4.951 | ppl 5184.86 | bleu 31.35 | wps 89586.6 | wpb 21176.3 | bsz 666.3 | num_updates 86224 | best_bleu 31.66
2023-06-14 00:20:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:21:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint306.pt (epoch 306 @ 86224 updates, score 31.35) (writing took 9.87716905400157 seconds)
2023-06-14 00:21:04 | INFO | fairseq_cli.train | end of epoch 306 (average epoch stats below)
2023-06-14 00:21:04 | INFO | train | epoch 306 | loss 3.044 | nll_loss 1.127 | glat_accu 0.568 | glat_context_p 0.443 | word_ins 2.924 | length 2.949 | ppl 8.25 | wps 114665 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 86224 | lr 0.000107693 | gnorm 0.569 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:21:04 | INFO | fairseq.trainer | begin training epoch 307
2023-06-14 00:21:45 | INFO | train_inner | epoch 307:     76 / 282 loss=3.048, nll_loss=1.13, glat_accu=0.573, glat_context_p=0.443, word_ins=2.927, length=2.943, ppl=8.27, wps=92449.3, ups=1.54, wpb=60123.9, bsz=2138.2, num_updates=86300, lr=0.000107645, gnorm=0.576, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:22:31 | INFO | train_inner | epoch 307:    176 / 282 loss=3.045, nll_loss=1.127, glat_accu=0.568, glat_context_p=0.442, word_ins=2.924, length=2.962, ppl=8.25, wps=132828, ups=2.19, wpb=60584.4, bsz=2149.4, num_updates=86400, lr=0.000107583, gnorm=0.555, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:23:17 | INFO | train_inner | epoch 307:    276 / 282 loss=3.041, nll_loss=1.124, glat_accu=0.569, glat_context_p=0.442, word_ins=2.922, length=2.941, ppl=8.23, wps=131070, ups=2.16, wpb=60572.1, bsz=2169.4, num_updates=86500, lr=0.000107521, gnorm=0.555, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:23:20 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:23:23 | INFO | valid | epoch 307 | valid on 'valid' subset | loss 12.376 | nll_loss 11.2 | word_ins 12.135 | length 4.82 | ppl 5314.13 | bleu 31.32 | wps 88230.5 | wpb 21176.3 | bsz 666.3 | num_updates 86506 | best_bleu 31.66
2023-06-14 00:23:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:23:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint307.pt (epoch 307 @ 86506 updates, score 31.32) (writing took 9.126535590738058 seconds)
2023-06-14 00:23:32 | INFO | fairseq_cli.train | end of epoch 307 (average epoch stats below)
2023-06-14 00:23:32 | INFO | train | epoch 307 | loss 3.044 | nll_loss 1.126 | glat_accu 0.569 | glat_context_p 0.442 | word_ins 2.924 | length 2.948 | ppl 8.25 | wps 115046 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 86506 | lr 0.000107517 | gnorm 0.562 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:23:32 | INFO | fairseq.trainer | begin training epoch 308
2023-06-14 00:23:54 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 00:24:22 | INFO | train_inner | epoch 308:     95 / 282 loss=3.037, nll_loss=1.12, glat_accu=0.565, glat_context_p=0.442, word_ins=2.918, length=2.931, ppl=8.21, wps=93331.6, ups=1.55, wpb=60084.6, bsz=2149.6, num_updates=86600, lr=0.000107459, gnorm=0.57, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:25:07 | INFO | train_inner | epoch 308:    195 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.567, glat_context_p=0.442, word_ins=2.923, length=2.948, ppl=8.24, wps=133306, ups=2.2, wpb=60572.3, bsz=2179.1, num_updates=86700, lr=0.000107397, gnorm=0.551, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 00:25:47 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:25:50 | INFO | valid | epoch 308 | valid on 'valid' subset | loss 12.498 | nll_loss 11.341 | word_ins 12.258 | length 4.821 | ppl 5785.11 | bleu 31.12 | wps 86747.5 | wpb 21176.3 | bsz 666.3 | num_updates 86787 | best_bleu 31.66
2023-06-14 00:25:50 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:26:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint308.pt (epoch 308 @ 86787 updates, score 31.12) (writing took 12.111953627318144 seconds)
2023-06-14 00:26:02 | INFO | fairseq_cli.train | end of epoch 308 (average epoch stats below)
2023-06-14 00:26:02 | INFO | train | epoch 308 | loss 3.044 | nll_loss 1.127 | glat_accu 0.567 | glat_context_p 0.442 | word_ins 2.924 | length 2.945 | ppl 8.25 | wps 113276 | ups 1.87 | wpb 60416.6 | bsz 2158.1 | num_updates 86787 | lr 0.000107343 | gnorm 0.562 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 00:26:02 | INFO | fairseq.trainer | begin training epoch 309
2023-06-14 00:26:14 | INFO | train_inner | epoch 309:     13 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.571, glat_context_p=0.442, word_ins=2.93, length=2.947, ppl=8.28, wps=89544, ups=1.49, wpb=60146.4, bsz=2154.8, num_updates=86800, lr=0.000107335, gnorm=0.57, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 00:27:00 | INFO | train_inner | epoch 309:    113 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.567, glat_context_p=0.442, word_ins=2.924, length=2.951, ppl=8.25, wps=132432, ups=2.18, wpb=60719.1, bsz=2122.6, num_updates=86900, lr=0.000107273, gnorm=0.562, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:27:46 | INFO | train_inner | epoch 309:    213 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.575, glat_context_p=0.442, word_ins=2.923, length=2.95, ppl=8.25, wps=130711, ups=2.16, wpb=60455.8, bsz=2168.9, num_updates=87000, lr=0.000107211, gnorm=0.559, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:28:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:28:21 | INFO | valid | epoch 309 | valid on 'valid' subset | loss 12.374 | nll_loss 11.202 | word_ins 12.136 | length 4.779 | ppl 5307.82 | bleu 31.16 | wps 87170.9 | wpb 21176.3 | bsz 666.3 | num_updates 87069 | best_bleu 31.66
2023-06-14 00:28:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:28:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint309.pt (epoch 309 @ 87069 updates, score 31.16) (writing took 12.947274528443813 seconds)
2023-06-14 00:28:34 | INFO | fairseq_cli.train | end of epoch 309 (average epoch stats below)
2023-06-14 00:28:34 | INFO | train | epoch 309 | loss 3.044 | nll_loss 1.126 | glat_accu 0.57 | glat_context_p 0.442 | word_ins 2.923 | length 2.944 | ppl 8.24 | wps 112043 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 87069 | lr 0.000107169 | gnorm 0.56 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 00:28:34 | INFO | fairseq.trainer | begin training epoch 310
2023-06-14 00:28:55 | INFO | train_inner | epoch 310:     31 / 282 loss=3.041, nll_loss=1.124, glat_accu=0.572, glat_context_p=0.442, word_ins=2.921, length=2.938, ppl=8.23, wps=87620.2, ups=1.46, wpb=59981.3, bsz=2177.6, num_updates=87100, lr=0.00010715, gnorm=0.561, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:29:40 | INFO | train_inner | epoch 310:    131 / 282 loss=3.043, nll_loss=1.125, glat_accu=0.574, glat_context_p=0.442, word_ins=2.922, length=2.937, ppl=8.24, wps=132982, ups=2.2, wpb=60558.6, bsz=2171.6, num_updates=87200, lr=0.000107088, gnorm=0.56, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 00:30:26 | INFO | train_inner | epoch 310:    231 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.568, glat_context_p=0.442, word_ins=2.927, length=2.942, ppl=8.26, wps=132278, ups=2.18, wpb=60652.9, bsz=2148.6, num_updates=87300, lr=0.000107027, gnorm=0.566, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:30:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:30:53 | INFO | valid | epoch 310 | valid on 'valid' subset | loss 12.324 | nll_loss 11.152 | word_ins 12.089 | length 4.693 | ppl 5127.07 | bleu 31.72 | wps 87656.8 | wpb 21176.3 | bsz 666.3 | num_updates 87351 | best_bleu 31.72
2023-06-14 00:30:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:31:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint310.pt (epoch 310 @ 87351 updates, score 31.72) (writing took 18.745652094483376 seconds)
2023-06-14 00:31:11 | INFO | fairseq_cli.train | end of epoch 310 (average epoch stats below)
2023-06-14 00:31:11 | INFO | train | epoch 310 | loss 3.045 | nll_loss 1.127 | glat_accu 0.571 | glat_context_p 0.442 | word_ins 2.925 | length 2.944 | ppl 8.25 | wps 108365 | ups 1.79 | wpb 60413.8 | bsz 2157.2 | num_updates 87351 | lr 0.000106996 | gnorm 0.566 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 00:31:11 | INFO | fairseq.trainer | begin training epoch 311
2023-06-14 00:31:41 | INFO | train_inner | epoch 311:     49 / 282 loss=3.046, nll_loss=1.128, glat_accu=0.571, glat_context_p=0.442, word_ins=2.925, length=2.949, ppl=8.26, wps=80869.4, ups=1.34, wpb=60149.9, bsz=2143.9, num_updates=87400, lr=0.000106966, gnorm=0.574, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:32:26 | INFO | train_inner | epoch 311:    149 / 282 loss=3.047, nll_loss=1.129, glat_accu=0.571, glat_context_p=0.442, word_ins=2.926, length=2.95, ppl=8.26, wps=132672, ups=2.19, wpb=60658.6, bsz=2155.1, num_updates=87500, lr=0.000106904, gnorm=0.57, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 00:33:12 | INFO | train_inner | epoch 311:    249 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.568, glat_context_p=0.442, word_ins=2.917, length=2.943, ppl=8.21, wps=131308, ups=2.17, wpb=60472.9, bsz=2167, num_updates=87600, lr=0.000106843, gnorm=0.545, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:33:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:33:31 | INFO | valid | epoch 311 | valid on 'valid' subset | loss 12.38 | nll_loss 11.206 | word_ins 12.143 | length 4.745 | ppl 5328.59 | bleu 31.34 | wps 88615.1 | wpb 21176.3 | bsz 666.3 | num_updates 87633 | best_bleu 31.72
2023-06-14 00:33:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:33:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint311.pt (epoch 311 @ 87633 updates, score 31.34) (writing took 13.591516237705946 seconds)
2023-06-14 00:33:44 | INFO | fairseq_cli.train | end of epoch 311 (average epoch stats below)
2023-06-14 00:33:44 | INFO | train | epoch 311 | loss 3.042 | nll_loss 1.125 | glat_accu 0.57 | glat_context_p 0.442 | word_ins 2.922 | length 2.944 | ppl 8.24 | wps 111552 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 87633 | lr 0.000106823 | gnorm 0.562 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:33:44 | INFO | fairseq.trainer | begin training epoch 312
2023-06-14 00:34:21 | INFO | train_inner | epoch 312:     67 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.571, glat_context_p=0.442, word_ins=2.923, length=2.925, ppl=8.24, wps=87772.2, ups=1.46, wpb=60219.4, bsz=2171.8, num_updates=87700, lr=0.000106783, gnorm=0.57, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:35:07 | INFO | train_inner | epoch 312:    167 / 282 loss=3.043, nll_loss=1.125, glat_accu=0.57, glat_context_p=0.442, word_ins=2.922, length=2.949, ppl=8.24, wps=131152, ups=2.17, wpb=60499.6, bsz=2139.4, num_updates=87800, lr=0.000106722, gnorm=0.567, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:35:53 | INFO | train_inner | epoch 312:    267 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.569, glat_context_p=0.441, word_ins=2.924, length=2.944, ppl=8.25, wps=132545, ups=2.19, wpb=60591, bsz=2169.3, num_updates=87900, lr=0.000106661, gnorm=0.552, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:35:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:36:02 | INFO | valid | epoch 312 | valid on 'valid' subset | loss 12.345 | nll_loss 11.178 | word_ins 12.115 | length 4.61 | ppl 5204.27 | bleu 31.39 | wps 87209.2 | wpb 21176.3 | bsz 666.3 | num_updates 87915 | best_bleu 31.72
2023-06-14 00:36:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:36:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint312.pt (epoch 312 @ 87915 updates, score 31.39) (writing took 13.32653557881713 seconds)
2023-06-14 00:36:16 | INFO | fairseq_cli.train | end of epoch 312 (average epoch stats below)
2023-06-14 00:36:16 | INFO | train | epoch 312 | loss 3.042 | nll_loss 1.125 | glat_accu 0.57 | glat_context_p 0.441 | word_ins 2.922 | length 2.942 | ppl 8.24 | wps 112364 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 87915 | lr 0.000106652 | gnorm 0.564 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:36:16 | INFO | fairseq.trainer | begin training epoch 313
2023-06-14 00:37:02 | INFO | train_inner | epoch 313:     85 / 282 loss=3.04, nll_loss=1.122, glat_accu=0.57, glat_context_p=0.441, word_ins=2.92, length=2.938, ppl=8.22, wps=87182.2, ups=1.45, wpb=60293.4, bsz=2125.5, num_updates=88000, lr=0.0001066, gnorm=0.575, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:37:48 | INFO | train_inner | epoch 313:    185 / 282 loss=3.046, nll_loss=1.128, glat_accu=0.575, glat_context_p=0.441, word_ins=2.925, length=2.953, ppl=8.26, wps=131797, ups=2.18, wpb=60493, bsz=2178.3, num_updates=88100, lr=0.00010654, gnorm=0.576, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:38:32 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:38:36 | INFO | valid | epoch 313 | valid on 'valid' subset | loss 12.449 | nll_loss 11.281 | word_ins 12.213 | length 4.729 | ppl 5590.01 | bleu 31.5 | wps 87560.9 | wpb 21176.3 | bsz 666.3 | num_updates 88197 | best_bleu 31.72
2023-06-14 00:38:36 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:38:47 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint313.pt (epoch 313 @ 88197 updates, score 31.5) (writing took 10.920349106192589 seconds)
2023-06-14 00:38:47 | INFO | fairseq_cli.train | end of epoch 313 (average epoch stats below)
2023-06-14 00:38:47 | INFO | train | epoch 313 | loss 3.043 | nll_loss 1.126 | glat_accu 0.572 | glat_context_p 0.441 | word_ins 2.923 | length 2.943 | ppl 8.24 | wps 112964 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 88197 | lr 0.000106481 | gnorm 0.567 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:38:47 | INFO | fairseq.trainer | begin training epoch 314
2023-06-14 00:38:54 | INFO | train_inner | epoch 314:      3 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.57, glat_context_p=0.441, word_ins=2.922, length=2.941, ppl=8.24, wps=90264.4, ups=1.5, wpb=60038.4, bsz=2156.9, num_updates=88200, lr=0.000106479, gnorm=0.559, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:39:40 | INFO | train_inner | epoch 314:    103 / 282 loss=3.042, nll_loss=1.124, glat_accu=0.573, glat_context_p=0.441, word_ins=2.921, length=2.941, ppl=8.23, wps=132565, ups=2.19, wpb=60620.6, bsz=2141.8, num_updates=88300, lr=0.000106419, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:40:26 | INFO | train_inner | epoch 314:    203 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.575, glat_context_p=0.441, word_ins=2.927, length=2.938, ppl=8.27, wps=132284, ups=2.18, wpb=60630.4, bsz=2131.2, num_updates=88400, lr=0.000106359, gnorm=0.567, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:41:02 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:41:05 | INFO | valid | epoch 314 | valid on 'valid' subset | loss 12.31 | nll_loss 11.129 | word_ins 12.075 | length 4.699 | ppl 5079.4 | bleu 31.55 | wps 86304.4 | wpb 21176.3 | bsz 666.3 | num_updates 88479 | best_bleu 31.72
2023-06-14 00:41:05 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:41:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint314.pt (epoch 314 @ 88479 updates, score 31.55) (writing took 9.948931686580181 seconds)
2023-06-14 00:41:15 | INFO | fairseq_cli.train | end of epoch 314 (average epoch stats below)
2023-06-14 00:41:15 | INFO | train | epoch 314 | loss 3.047 | nll_loss 1.129 | glat_accu 0.575 | glat_context_p 0.441 | word_ins 2.926 | length 2.944 | ppl 8.26 | wps 114724 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 88479 | lr 0.000106311 | gnorm 0.573 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:41:15 | INFO | fairseq.trainer | begin training epoch 315
2023-06-14 00:41:30 | INFO | train_inner | epoch 315:     21 / 282 loss=3.052, nll_loss=1.135, glat_accu=0.578, glat_context_p=0.441, word_ins=2.931, length=2.956, ppl=8.3, wps=92916.9, ups=1.55, wpb=59966, bsz=2189.7, num_updates=88500, lr=0.000106299, gnorm=0.58, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:42:11 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 00:42:16 | INFO | train_inner | epoch 315:    122 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.575, glat_context_p=0.441, word_ins=2.926, length=2.942, ppl=8.26, wps=132126, ups=2.18, wpb=60690.9, bsz=2145.9, num_updates=88600, lr=0.000106239, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:43:02 | INFO | train_inner | epoch 315:    222 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.569, glat_context_p=0.441, word_ins=2.923, length=2.937, ppl=8.24, wps=131352, ups=2.17, wpb=60517.8, bsz=2170.6, num_updates=88700, lr=0.000106179, gnorm=0.561, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:43:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:43:33 | INFO | valid | epoch 315 | valid on 'valid' subset | loss 12.313 | nll_loss 11.137 | word_ins 12.079 | length 4.683 | ppl 5088.9 | bleu 31.48 | wps 88960.7 | wpb 21176.3 | bsz 666.3 | num_updates 88760 | best_bleu 31.72
2023-06-14 00:43:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:43:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint315.pt (epoch 315 @ 88760 updates, score 31.48) (writing took 10.422660637646914 seconds)
2023-06-14 00:43:44 | INFO | fairseq_cli.train | end of epoch 315 (average epoch stats below)
2023-06-14 00:43:44 | INFO | train | epoch 315 | loss 3.044 | nll_loss 1.127 | glat_accu 0.573 | glat_context_p 0.441 | word_ins 2.924 | length 2.94 | ppl 8.25 | wps 114138 | ups 1.89 | wpb 60412.6 | bsz 2157.1 | num_updates 88760 | lr 0.000106143 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:43:44 | INFO | fairseq.trainer | begin training epoch 316
2023-06-14 00:44:08 | INFO | train_inner | epoch 316:     40 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.571, glat_context_p=0.441, word_ins=2.923, length=2.955, ppl=8.25, wps=91320.4, ups=1.52, wpb=60175.5, bsz=2132.2, num_updates=88800, lr=0.000106119, gnorm=0.578, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:44:54 | INFO | train_inner | epoch 316:    140 / 282 loss=3.039, nll_loss=1.121, glat_accu=0.576, glat_context_p=0.441, word_ins=2.919, length=2.932, ppl=8.22, wps=133312, ups=2.2, wpb=60651.7, bsz=2182.4, num_updates=88900, lr=0.000106059, gnorm=0.569, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:45:39 | INFO | train_inner | epoch 316:    240 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.571, glat_context_p=0.441, word_ins=2.922, length=2.94, ppl=8.24, wps=132438, ups=2.19, wpb=60429.3, bsz=2167.9, num_updates=89000, lr=0.000106, gnorm=0.57, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:45:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:46:02 | INFO | valid | epoch 316 | valid on 'valid' subset | loss 12.369 | nll_loss 11.2 | word_ins 12.135 | length 4.685 | ppl 5291.62 | bleu 31.54 | wps 87866.7 | wpb 21176.3 | bsz 666.3 | num_updates 89042 | best_bleu 31.72
2023-06-14 00:46:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:46:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint316.pt (epoch 316 @ 89042 updates, score 31.54) (writing took 10.614940647035837 seconds)
2023-06-14 00:46:12 | INFO | fairseq_cli.train | end of epoch 316 (average epoch stats below)
2023-06-14 00:46:12 | INFO | train | epoch 316 | loss 3.042 | nll_loss 1.125 | glat_accu 0.573 | glat_context_p 0.441 | word_ins 2.922 | length 2.943 | ppl 8.24 | wps 114843 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 89042 | lr 0.000105975 | gnorm 0.57 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:46:12 | INFO | fairseq.trainer | begin training epoch 317
2023-06-14 00:46:45 | INFO | train_inner | epoch 317:     58 / 282 loss=3.042, nll_loss=1.125, glat_accu=0.579, glat_context_p=0.441, word_ins=2.922, length=2.919, ppl=8.24, wps=91878.6, ups=1.53, wpb=60231, bsz=2181.8, num_updates=89100, lr=0.00010594, gnorm=0.567, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:47:31 | INFO | train_inner | epoch 317:    158 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.575, glat_context_p=0.441, word_ins=2.923, length=2.946, ppl=8.25, wps=131428, ups=2.17, wpb=60532, bsz=2171.6, num_updates=89200, lr=0.000105881, gnorm=0.58, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:48:17 | INFO | train_inner | epoch 317:    258 / 282 loss=3.049, nll_loss=1.131, glat_accu=0.575, glat_context_p=0.441, word_ins=2.928, length=2.961, ppl=8.28, wps=131765, ups=2.18, wpb=60568.7, bsz=2116.8, num_updates=89300, lr=0.000105822, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:48:28 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:48:31 | INFO | valid | epoch 317 | valid on 'valid' subset | loss 12.335 | nll_loss 11.161 | word_ins 12.104 | length 4.609 | ppl 5166.4 | bleu 31.72 | wps 88399.2 | wpb 21176.3 | bsz 666.3 | num_updates 89324 | best_bleu 31.72
2023-06-14 00:48:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:48:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint317.pt (epoch 317 @ 89324 updates, score 31.72) (writing took 18.69034769758582 seconds)
2023-06-14 00:48:50 | INFO | fairseq_cli.train | end of epoch 317 (average epoch stats below)
2023-06-14 00:48:50 | INFO | train | epoch 317 | loss 3.045 | nll_loss 1.128 | glat_accu 0.577 | glat_context_p 0.441 | word_ins 2.925 | length 2.944 | ppl 8.26 | wps 108109 | ups 1.79 | wpb 60413.8 | bsz 2157.2 | num_updates 89324 | lr 0.000105807 | gnorm 0.579 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:48:50 | INFO | fairseq.trainer | begin training epoch 318
2023-06-14 00:49:31 | INFO | train_inner | epoch 318:     76 / 282 loss=3.042, nll_loss=1.124, glat_accu=0.577, glat_context_p=0.44, word_ins=2.921, length=2.945, ppl=8.23, wps=80637.5, ups=1.34, wpb=59976.9, bsz=2152.6, num_updates=89400, lr=0.000105762, gnorm=0.576, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:50:18 | INFO | train_inner | epoch 318:    176 / 282 loss=3.045, nll_loss=1.127, glat_accu=0.575, glat_context_p=0.44, word_ins=2.924, length=2.945, ppl=8.25, wps=131483, ups=2.17, wpb=60621.5, bsz=2134.5, num_updates=89500, lr=0.000105703, gnorm=0.568, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:51:03 | INFO | train_inner | epoch 318:    276 / 282 loss=3.05, nll_loss=1.133, glat_accu=0.583, glat_context_p=0.44, word_ins=2.93, length=2.941, ppl=8.28, wps=133303, ups=2.2, wpb=60633.9, bsz=2185.3, num_updates=89600, lr=0.000105644, gnorm=0.575, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:51:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:51:09 | INFO | valid | epoch 318 | valid on 'valid' subset | loss 12.308 | nll_loss 11.132 | word_ins 12.074 | length 4.672 | ppl 5070.78 | bleu 31.5 | wps 88517.9 | wpb 21176.3 | bsz 666.3 | num_updates 89606 | best_bleu 31.72
2023-06-14 00:51:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:51:22 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint318.pt (epoch 318 @ 89606 updates, score 31.5) (writing took 12.728915877640247 seconds)
2023-06-14 00:51:22 | INFO | fairseq_cli.train | end of epoch 318 (average epoch stats below)
2023-06-14 00:51:22 | INFO | train | epoch 318 | loss 3.045 | nll_loss 1.127 | glat_accu 0.578 | glat_context_p 0.44 | word_ins 2.924 | length 2.942 | ppl 8.25 | wps 112192 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 89606 | lr 0.000105641 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:51:22 | INFO | fairseq.trainer | begin training epoch 319
2023-06-14 00:51:31 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 00:52:11 | INFO | train_inner | epoch 319:     95 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.577, glat_context_p=0.44, word_ins=2.923, length=2.937, ppl=8.25, wps=88515.2, ups=1.47, wpb=60243.8, bsz=2141.4, num_updates=89700, lr=0.000105585, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:52:57 | INFO | train_inner | epoch 319:    195 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.58, glat_context_p=0.44, word_ins=2.924, length=2.939, ppl=8.25, wps=131835, ups=2.18, wpb=60610.8, bsz=2178.2, num_updates=89800, lr=0.000105527, gnorm=0.573, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:53:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:53:40 | INFO | valid | epoch 319 | valid on 'valid' subset | loss 12.323 | nll_loss 11.142 | word_ins 12.087 | length 4.719 | ppl 5123.59 | bleu 31.69 | wps 87876.7 | wpb 21176.3 | bsz 666.3 | num_updates 89887 | best_bleu 31.72
2023-06-14 00:53:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:53:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint319.pt (epoch 319 @ 89887 updates, score 31.69) (writing took 17.475382398813963 seconds)
2023-06-14 00:53:57 | INFO | fairseq_cli.train | end of epoch 319 (average epoch stats below)
2023-06-14 00:53:57 | INFO | train | epoch 319 | loss 3.044 | nll_loss 1.127 | glat_accu 0.578 | glat_context_p 0.44 | word_ins 2.924 | length 2.941 | ppl 8.25 | wps 109235 | ups 1.81 | wpb 60414.4 | bsz 2157.2 | num_updates 89887 | lr 0.000105475 | gnorm 0.578 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:53:57 | INFO | fairseq.trainer | begin training epoch 320
2023-06-14 00:54:08 | INFO | train_inner | epoch 320:     13 / 282 loss=3.042, nll_loss=1.124, glat_accu=0.576, glat_context_p=0.44, word_ins=2.921, length=2.937, ppl=8.23, wps=84214.3, ups=1.4, wpb=59952.4, bsz=2136.2, num_updates=89900, lr=0.000105468, gnorm=0.572, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:54:54 | INFO | train_inner | epoch 320:    113 / 282 loss=3.041, nll_loss=1.123, glat_accu=0.572, glat_context_p=0.44, word_ins=2.92, length=2.953, ppl=8.23, wps=133199, ups=2.2, wpb=60607.9, bsz=2160.2, num_updates=90000, lr=0.000105409, gnorm=0.562, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:55:40 | INFO | train_inner | epoch 320:    213 / 282 loss=3.042, nll_loss=1.124, glat_accu=0.577, glat_context_p=0.44, word_ins=2.921, length=2.946, ppl=8.24, wps=132346, ups=2.18, wpb=60617, bsz=2144.2, num_updates=90100, lr=0.000105351, gnorm=0.569, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:56:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:56:14 | INFO | valid | epoch 320 | valid on 'valid' subset | loss 12.26 | nll_loss 11.073 | word_ins 12.021 | length 4.788 | ppl 4905.6 | bleu 31.62 | wps 88151 | wpb 21176.3 | bsz 666.3 | num_updates 90169 | best_bleu 31.72
2023-06-14 00:56:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:56:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint320.pt (epoch 320 @ 90169 updates, score 31.62) (writing took 13.288223151117563 seconds)
2023-06-14 00:56:27 | INFO | fairseq_cli.train | end of epoch 320 (average epoch stats below)
2023-06-14 00:56:27 | INFO | train | epoch 320 | loss 3.043 | nll_loss 1.126 | glat_accu 0.578 | glat_context_p 0.44 | word_ins 2.923 | length 2.938 | ppl 8.24 | wps 113345 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 90169 | lr 0.00010531 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 00:56:27 | INFO | fairseq.trainer | begin training epoch 321
2023-06-14 00:56:48 | INFO | train_inner | epoch 321:     31 / 282 loss=3.052, nll_loss=1.136, glat_accu=0.584, glat_context_p=0.44, word_ins=2.932, length=2.923, ppl=8.29, wps=87499.7, ups=1.46, wpb=60030.4, bsz=2168, num_updates=90200, lr=0.000105292, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 00:57:34 | INFO | train_inner | epoch 321:    131 / 282 loss=3.043, nll_loss=1.125, glat_accu=0.581, glat_context_p=0.44, word_ins=2.922, length=2.94, ppl=8.24, wps=131565, ups=2.18, wpb=60371, bsz=2182.9, num_updates=90300, lr=0.000105234, gnorm=0.579, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:58:20 | INFO | train_inner | epoch 321:    231 / 282 loss=3.043, nll_loss=1.127, glat_accu=0.575, glat_context_p=0.44, word_ins=2.924, length=2.918, ppl=8.24, wps=132978, ups=2.19, wpb=60802.1, bsz=2184.2, num_updates=90400, lr=0.000105176, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 00:58:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 00:58:47 | INFO | valid | epoch 321 | valid on 'valid' subset | loss 12.312 | nll_loss 11.142 | word_ins 12.083 | length 4.607 | ppl 5086.31 | bleu 31.78 | wps 88464.1 | wpb 21176.3 | bsz 666.3 | num_updates 90451 | best_bleu 31.78
2023-06-14 00:58:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 00:59:07 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint321.pt (epoch 321 @ 90451 updates, score 31.78) (writing took 19.179108012467623 seconds)
2023-06-14 00:59:07 | INFO | fairseq_cli.train | end of epoch 321 (average epoch stats below)
2023-06-14 00:59:07 | INFO | train | epoch 321 | loss 3.046 | nll_loss 1.129 | glat_accu 0.577 | glat_context_p 0.44 | word_ins 2.926 | length 2.937 | ppl 8.26 | wps 107023 | ups 1.77 | wpb 60413.8 | bsz 2157.2 | num_updates 90451 | lr 0.000105146 | gnorm 0.583 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 00:59:07 | INFO | fairseq.trainer | begin training epoch 322
2023-06-14 00:59:35 | INFO | train_inner | epoch 322:     49 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.574, glat_context_p=0.44, word_ins=2.926, length=2.951, ppl=8.27, wps=79680.9, ups=1.32, wpb=60246.6, bsz=2124.6, num_updates=90500, lr=0.000105118, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:00:21 | INFO | train_inner | epoch 322:    149 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.575, glat_context_p=0.44, word_ins=2.925, length=2.954, ppl=8.26, wps=131273, ups=2.17, wpb=60397.7, bsz=2146.9, num_updates=90600, lr=0.00010506, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:00:38 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 01:01:08 | INFO | train_inner | epoch 322:    250 / 282 loss=3.047, nll_loss=1.13, glat_accu=0.582, glat_context_p=0.44, word_ins=2.927, length=2.935, ppl=8.27, wps=131351, ups=2.16, wpb=60675.5, bsz=2161.6, num_updates=90700, lr=0.000105002, gnorm=0.564, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:01:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:01:25 | INFO | valid | epoch 322 | valid on 'valid' subset | loss 12.425 | nll_loss 11.251 | word_ins 12.185 | length 4.807 | ppl 5498.67 | bleu 31.51 | wps 89315.4 | wpb 21176.3 | bsz 666.3 | num_updates 90732 | best_bleu 31.78
2023-06-14 01:01:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:01:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint322.pt (epoch 322 @ 90732 updates, score 31.51) (writing took 11.9674127176404 seconds)
2023-06-14 01:01:37 | INFO | fairseq_cli.train | end of epoch 322 (average epoch stats below)
2023-06-14 01:01:37 | INFO | train | epoch 322 | loss 3.046 | nll_loss 1.129 | glat_accu 0.579 | glat_context_p 0.44 | word_ins 2.926 | length 2.941 | ppl 8.26 | wps 112984 | ups 1.87 | wpb 60421.6 | bsz 2157.2 | num_updates 90732 | lr 0.000104983 | gnorm 0.573 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 01:01:37 | INFO | fairseq.trainer | begin training epoch 323
2023-06-14 01:02:15 | INFO | train_inner | epoch 323:     68 / 282 loss=3.043, nll_loss=1.125, glat_accu=0.581, glat_context_p=0.44, word_ins=2.922, length=2.947, ppl=8.24, wps=89450.1, ups=1.49, wpb=59971.4, bsz=2123.5, num_updates=90800, lr=0.000104944, gnorm=0.58, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:03:00 | INFO | train_inner | epoch 323:    168 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.586, glat_context_p=0.439, word_ins=2.926, length=2.927, ppl=8.26, wps=133373, ups=2.2, wpb=60587.7, bsz=2187.2, num_updates=90900, lr=0.000104886, gnorm=0.567, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:03:46 | INFO | train_inner | epoch 323:    268 / 282 loss=3.051, nll_loss=1.134, glat_accu=0.582, glat_context_p=0.439, word_ins=2.93, length=2.941, ppl=8.29, wps=132010, ups=2.18, wpb=60681.5, bsz=2158.6, num_updates=91000, lr=0.000104828, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:03:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:03:56 | INFO | valid | epoch 323 | valid on 'valid' subset | loss 12.317 | nll_loss 11.132 | word_ins 12.078 | length 4.75 | ppl 5101.88 | bleu 31.44 | wps 87803.7 | wpb 21176.3 | bsz 666.3 | num_updates 91014 | best_bleu 31.78
2023-06-14 01:03:56 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:04:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint323.pt (epoch 323 @ 91014 updates, score 31.44) (writing took 10.561921872198582 seconds)
2023-06-14 01:04:06 | INFO | fairseq_cli.train | end of epoch 323 (average epoch stats below)
2023-06-14 01:04:06 | INFO | train | epoch 323 | loss 3.046 | nll_loss 1.129 | glat_accu 0.582 | glat_context_p 0.439 | word_ins 2.925 | length 2.939 | ppl 8.26 | wps 113983 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 91014 | lr 0.00010482 | gnorm 0.576 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:04:06 | INFO | fairseq.trainer | begin training epoch 324
2023-06-14 01:04:52 | INFO | train_inner | epoch 324:     86 / 282 loss=3.046, nll_loss=1.129, glat_accu=0.578, glat_context_p=0.439, word_ins=2.926, length=2.942, ppl=8.26, wps=90816.4, ups=1.51, wpb=59960.3, bsz=2150.2, num_updates=91100, lr=0.000104771, gnorm=0.57, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:05:17 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 01:05:38 | INFO | train_inner | epoch 324:    187 / 282 loss=3.036, nll_loss=1.119, glat_accu=0.57, glat_context_p=0.439, word_ins=2.917, length=2.933, ppl=8.2, wps=131226, ups=2.16, wpb=60684.2, bsz=2168.1, num_updates=91200, lr=0.000104713, gnorm=0.559, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:06:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:06:25 | INFO | valid | epoch 324 | valid on 'valid' subset | loss 12.423 | nll_loss 11.259 | word_ins 12.186 | length 4.752 | ppl 5492.14 | bleu 30.96 | wps 87712 | wpb 21176.3 | bsz 666.3 | num_updates 91295 | best_bleu 31.78
2023-06-14 01:06:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:06:33 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint324.pt (epoch 324 @ 91295 updates, score 30.96) (writing took 8.447189711034298 seconds)
2023-06-14 01:06:33 | INFO | fairseq_cli.train | end of epoch 324 (average epoch stats below)
2023-06-14 01:06:33 | INFO | train | epoch 324 | loss 3.04 | nll_loss 1.123 | glat_accu 0.574 | glat_context_p 0.439 | word_ins 2.92 | length 2.941 | ppl 8.23 | wps 115495 | ups 1.91 | wpb 60416.7 | bsz 2157 | num_updates 91295 | lr 0.000104659 | gnorm 0.567 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 01:06:33 | INFO | fairseq.trainer | begin training epoch 325
2023-06-14 01:06:41 | INFO | train_inner | epoch 325:      5 / 282 loss=3.039, nll_loss=1.121, glat_accu=0.576, glat_context_p=0.439, word_ins=2.919, length=2.947, ppl=8.22, wps=95420.6, ups=1.59, wpb=60116.1, bsz=2150.2, num_updates=91300, lr=0.000104656, gnorm=0.578, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:07:27 | INFO | train_inner | epoch 325:    105 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.569, glat_context_p=0.439, word_ins=2.917, length=2.949, ppl=8.21, wps=131446, ups=2.17, wpb=60608.5, bsz=2165.1, num_updates=91400, lr=0.000104599, gnorm=0.573, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:08:13 | INFO | train_inner | epoch 325:    205 / 282 loss=3.047, nll_loss=1.129, glat_accu=0.575, glat_context_p=0.439, word_ins=2.926, length=2.952, ppl=8.27, wps=131911, ups=2.17, wpb=60693.6, bsz=2112.9, num_updates=91500, lr=0.000104542, gnorm=0.576, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:08:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:08:51 | INFO | valid | epoch 325 | valid on 'valid' subset | loss 12.363 | nll_loss 11.197 | word_ins 12.134 | length 4.579 | ppl 5268.3 | bleu 31.57 | wps 86353.1 | wpb 21176.3 | bsz 666.3 | num_updates 91577 | best_bleu 31.78
2023-06-14 01:08:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:09:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint325.pt (epoch 325 @ 91577 updates, score 31.57) (writing took 10.374279361218214 seconds)
2023-06-14 01:09:02 | INFO | fairseq_cli.train | end of epoch 325 (average epoch stats below)
2023-06-14 01:09:02 | INFO | train | epoch 325 | loss 3.04 | nll_loss 1.123 | glat_accu 0.575 | glat_context_p 0.439 | word_ins 2.92 | length 2.94 | ppl 8.23 | wps 114839 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 91577 | lr 0.000104498 | gnorm 0.575 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 01:09:02 | INFO | fairseq.trainer | begin training epoch 326
2023-06-14 01:09:19 | INFO | train_inner | epoch 326:     23 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.581, glat_context_p=0.439, word_ins=2.915, length=2.921, ppl=8.2, wps=91905.4, ups=1.53, wpb=59981.3, bsz=2191.4, num_updates=91600, lr=0.000104485, gnorm=0.577, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 01:10:04 | INFO | train_inner | epoch 326:    123 / 282 loss=3.041, nll_loss=1.123, glat_accu=0.573, glat_context_p=0.439, word_ins=2.92, length=2.941, ppl=8.23, wps=133151, ups=2.2, wpb=60627.9, bsz=2144.5, num_updates=91700, lr=0.000104428, gnorm=0.568, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 01:10:51 | INFO | train_inner | epoch 326:    223 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.57, glat_context_p=0.439, word_ins=2.924, length=2.944, ppl=8.25, wps=130696, ups=2.16, wpb=60630.4, bsz=2150.6, num_updates=91800, lr=0.000104371, gnorm=0.574, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:11:17 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:11:21 | INFO | valid | epoch 326 | valid on 'valid' subset | loss 12.382 | nll_loss 11.205 | word_ins 12.139 | length 4.865 | ppl 5337.65 | bleu 31.36 | wps 87578.2 | wpb 21176.3 | bsz 666.3 | num_updates 91859 | best_bleu 31.78
2023-06-14 01:11:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:11:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint326.pt (epoch 326 @ 91859 updates, score 31.36) (writing took 10.659065343439579 seconds)
2023-06-14 01:11:31 | INFO | fairseq_cli.train | end of epoch 326 (average epoch stats below)
2023-06-14 01:11:31 | INFO | train | epoch 326 | loss 3.04 | nll_loss 1.122 | glat_accu 0.574 | glat_context_p 0.439 | word_ins 2.92 | length 2.939 | ppl 8.22 | wps 113827 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 91859 | lr 0.000104337 | gnorm 0.573 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 01:11:31 | INFO | fairseq.trainer | begin training epoch 327
2023-06-14 01:11:57 | INFO | train_inner | epoch 327:     41 / 282 loss=3.034, nll_loss=1.115, glat_accu=0.577, glat_context_p=0.439, word_ins=2.913, length=2.939, ppl=8.19, wps=90150.7, ups=1.51, wpb=59886.9, bsz=2187.1, num_updates=91900, lr=0.000104314, gnorm=0.575, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:12:43 | INFO | train_inner | epoch 327:    141 / 282 loss=3.036, nll_loss=1.119, glat_accu=0.573, glat_context_p=0.439, word_ins=2.917, length=2.923, ppl=8.2, wps=131415, ups=2.17, wpb=60688.5, bsz=2154.4, num_updates=92000, lr=0.000104257, gnorm=0.573, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 01:13:29 | INFO | train_inner | epoch 327:    241 / 282 loss=3.044, nll_loss=1.127, glat_accu=0.574, glat_context_p=0.439, word_ins=2.924, length=2.947, ppl=8.25, wps=133778, ups=2.2, wpb=60685.6, bsz=2155.3, num_updates=92100, lr=0.000104201, gnorm=0.576, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 01:13:47 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:13:51 | INFO | valid | epoch 327 | valid on 'valid' subset | loss 12.272 | nll_loss 11.09 | word_ins 12.041 | length 4.614 | ppl 4945.75 | bleu 31.95 | wps 88148.2 | wpb 21176.3 | bsz 666.3 | num_updates 92141 | best_bleu 31.95
2023-06-14 01:13:51 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:14:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint327.pt (epoch 327 @ 92141 updates, score 31.95) (writing took 18.023523304611444 seconds)
2023-06-14 01:14:09 | INFO | fairseq_cli.train | end of epoch 327 (average epoch stats below)
2023-06-14 01:14:09 | INFO | train | epoch 327 | loss 3.04 | nll_loss 1.122 | glat_accu 0.574 | glat_context_p 0.439 | word_ins 2.919 | length 2.939 | ppl 8.22 | wps 108348 | ups 1.79 | wpb 60413.8 | bsz 2157.2 | num_updates 92141 | lr 0.000104177 | gnorm 0.575 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 01:14:09 | INFO | fairseq.trainer | begin training epoch 328
2023-06-14 01:14:43 | INFO | train_inner | epoch 328:     59 / 282 loss=3.039, nll_loss=1.122, glat_accu=0.577, glat_context_p=0.439, word_ins=2.92, length=2.925, ppl=8.22, wps=80549.7, ups=1.34, wpb=59989.8, bsz=2170.4, num_updates=92200, lr=0.000104144, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:15:29 | INFO | train_inner | epoch 328:    159 / 282 loss=3.04, nll_loss=1.122, glat_accu=0.581, glat_context_p=0.439, word_ins=2.919, length=2.93, ppl=8.22, wps=132736, ups=2.19, wpb=60634.1, bsz=2179.6, num_updates=92300, lr=0.000104088, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:16:14 | INFO | train_inner | epoch 328:    259 / 282 loss=3.041, nll_loss=1.124, glat_accu=0.575, glat_context_p=0.438, word_ins=2.921, length=2.944, ppl=8.23, wps=133000, ups=2.19, wpb=60639.1, bsz=2141, num_updates=92400, lr=0.000104031, gnorm=0.571, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:16:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:16:28 | INFO | valid | epoch 328 | valid on 'valid' subset | loss 12.386 | nll_loss 11.221 | word_ins 12.154 | length 4.643 | ppl 5351.27 | bleu 31.55 | wps 88128.5 | wpb 21176.3 | bsz 666.3 | num_updates 92423 | best_bleu 31.95
2023-06-14 01:16:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:16:39 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint328.pt (epoch 328 @ 92423 updates, score 31.55) (writing took 10.314740672707558 seconds)
2023-06-14 01:16:39 | INFO | fairseq_cli.train | end of epoch 328 (average epoch stats below)
2023-06-14 01:16:39 | INFO | train | epoch 328 | loss 3.04 | nll_loss 1.122 | glat_accu 0.577 | glat_context_p 0.438 | word_ins 2.92 | length 2.936 | ppl 8.23 | wps 113605 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 92423 | lr 0.000104018 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:16:39 | INFO | fairseq.trainer | begin training epoch 329
2023-06-14 01:17:20 | INFO | train_inner | epoch 329:     77 / 282 loss=3.037, nll_loss=1.118, glat_accu=0.578, glat_context_p=0.438, word_ins=2.916, length=2.941, ppl=8.21, wps=91276.6, ups=1.52, wpb=60094, bsz=2135.3, num_updates=92500, lr=0.000103975, gnorm=0.573, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:18:06 | INFO | train_inner | epoch 329:    177 / 282 loss=3.038, nll_loss=1.12, glat_accu=0.574, glat_context_p=0.438, word_ins=2.918, length=2.932, ppl=8.21, wps=130808, ups=2.16, wpb=60509.3, bsz=2179.6, num_updates=92600, lr=0.000103919, gnorm=0.567, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:18:52 | INFO | train_inner | epoch 329:    277 / 282 loss=3.048, nll_loss=1.13, glat_accu=0.576, glat_context_p=0.438, word_ins=2.926, length=2.961, ppl=8.27, wps=131820, ups=2.17, wpb=60626.7, bsz=2142.6, num_updates=92700, lr=0.000103863, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:18:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:18:58 | INFO | valid | epoch 329 | valid on 'valid' subset | loss 12.321 | nll_loss 11.138 | word_ins 12.082 | length 4.779 | ppl 5115.76 | bleu 31.73 | wps 84139.6 | wpb 21176.3 | bsz 666.3 | num_updates 92705 | best_bleu 31.95
2023-06-14 01:18:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:19:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint329.pt (epoch 329 @ 92705 updates, score 31.73) (writing took 10.38939069956541 seconds)
2023-06-14 01:19:08 | INFO | fairseq_cli.train | end of epoch 329 (average epoch stats below)
2023-06-14 01:19:08 | INFO | train | epoch 329 | loss 3.04 | nll_loss 1.123 | glat_accu 0.576 | glat_context_p 0.438 | word_ins 2.92 | length 2.941 | ppl 8.23 | wps 113809 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 92705 | lr 0.00010386 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:19:08 | INFO | fairseq.trainer | begin training epoch 330
2023-06-14 01:19:58 | INFO | train_inner | epoch 330:     95 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.58, glat_context_p=0.438, word_ins=2.916, length=2.926, ppl=8.2, wps=91268.5, ups=1.52, wpb=60191.8, bsz=2135, num_updates=92800, lr=0.000103807, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:20:44 | INFO | train_inner | epoch 330:    195 / 282 loss=3.043, nll_loss=1.124, glat_accu=0.58, glat_context_p=0.438, word_ins=2.922, length=2.949, ppl=8.24, wps=133050, ups=2.2, wpb=60511.3, bsz=2186.9, num_updates=92900, lr=0.000103751, gnorm=0.577, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:21:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:21:27 | INFO | valid | epoch 330 | valid on 'valid' subset | loss 12.35 | nll_loss 11.169 | word_ins 12.111 | length 4.79 | ppl 5220.21 | bleu 31.56 | wps 86396.1 | wpb 21176.3 | bsz 666.3 | num_updates 92987 | best_bleu 31.95
2023-06-14 01:21:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:21:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint330.pt (epoch 330 @ 92987 updates, score 31.56) (writing took 12.920001916587353 seconds)
2023-06-14 01:21:40 | INFO | fairseq_cli.train | end of epoch 330 (average epoch stats below)
2023-06-14 01:21:40 | INFO | train | epoch 330 | loss 3.039 | nll_loss 1.121 | glat_accu 0.579 | glat_context_p 0.438 | word_ins 2.919 | length 2.934 | ppl 8.22 | wps 112183 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 92987 | lr 0.000103702 | gnorm 0.578 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:21:40 | INFO | fairseq.trainer | begin training epoch 331
2023-06-14 01:21:52 | INFO | train_inner | epoch 331:     13 / 282 loss=3.038, nll_loss=1.121, glat_accu=0.578, glat_context_p=0.438, word_ins=2.918, length=2.919, ppl=8.21, wps=87812.6, ups=1.46, wpb=60081.5, bsz=2143.8, num_updates=93000, lr=0.000103695, gnorm=0.573, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:22:38 | INFO | train_inner | epoch 331:    113 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.58, glat_context_p=0.438, word_ins=2.917, length=2.929, ppl=8.21, wps=132903, ups=2.19, wpb=60591.8, bsz=2177.4, num_updates=93100, lr=0.000103639, gnorm=0.584, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:23:24 | INFO | train_inner | epoch 331:    213 / 282 loss=3.037, nll_loss=1.118, glat_accu=0.571, glat_context_p=0.438, word_ins=2.916, length=2.946, ppl=8.21, wps=132111, ups=2.18, wpb=60588.8, bsz=2110.8, num_updates=93200, lr=0.000103584, gnorm=0.579, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:23:25 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 01:23:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:23:58 | INFO | valid | epoch 331 | valid on 'valid' subset | loss 12.42 | nll_loss 11.253 | word_ins 12.184 | length 4.733 | ppl 5481.89 | bleu 31.5 | wps 88720.1 | wpb 21176.3 | bsz 666.3 | num_updates 93268 | best_bleu 31.95
2023-06-14 01:23:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:24:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint331.pt (epoch 331 @ 93268 updates, score 31.5) (writing took 11.22639674693346 seconds)
2023-06-14 01:24:09 | INFO | fairseq_cli.train | end of epoch 331 (average epoch stats below)
2023-06-14 01:24:09 | INFO | train | epoch 331 | loss 3.037 | nll_loss 1.12 | glat_accu 0.575 | glat_context_p 0.438 | word_ins 2.917 | length 2.934 | ppl 8.21 | wps 113681 | ups 1.88 | wpb 60411.3 | bsz 2157.1 | num_updates 93268 | lr 0.000103546 | gnorm 0.581 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 01:24:09 | INFO | fairseq.trainer | begin training epoch 332
2023-06-14 01:24:31 | INFO | train_inner | epoch 332:     32 / 282 loss=3.035, nll_loss=1.118, glat_accu=0.569, glat_context_p=0.438, word_ins=2.916, length=2.931, ppl=8.2, wps=88800.8, ups=1.48, wpb=60145.1, bsz=2177.4, num_updates=93300, lr=0.000103528, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:25:18 | INFO | train_inner | epoch 332:    132 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.572, glat_context_p=0.438, word_ins=2.908, length=2.94, ppl=8.16, wps=131084, ups=2.17, wpb=60533.2, bsz=2151.6, num_updates=93400, lr=0.000103473, gnorm=0.578, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:26:03 | INFO | train_inner | epoch 332:    232 / 282 loss=3.039, nll_loss=1.121, glat_accu=0.577, glat_context_p=0.438, word_ins=2.919, length=2.928, ppl=8.22, wps=132780, ups=2.19, wpb=60723.6, bsz=2148.9, num_updates=93500, lr=0.000103418, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:26:26 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:26:29 | INFO | valid | epoch 332 | valid on 'valid' subset | loss 12.294 | nll_loss 11.115 | word_ins 12.062 | length 4.62 | ppl 5020.22 | bleu 31.67 | wps 87705.4 | wpb 21176.3 | bsz 666.3 | num_updates 93550 | best_bleu 31.95
2023-06-14 01:26:29 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:26:43 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint332.pt (epoch 332 @ 93550 updates, score 31.67) (writing took 14.088639035820961 seconds)
2023-06-14 01:26:43 | INFO | fairseq_cli.train | end of epoch 332 (average epoch stats below)
2023-06-14 01:26:43 | INFO | train | epoch 332 | loss 3.035 | nll_loss 1.117 | glat_accu 0.575 | glat_context_p 0.438 | word_ins 2.915 | length 2.934 | ppl 8.2 | wps 110842 | ups 1.83 | wpb 60413.8 | bsz 2157.2 | num_updates 93550 | lr 0.00010339 | gnorm 0.577 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:26:43 | INFO | fairseq.trainer | begin training epoch 333
2023-06-14 01:27:12 | INFO | train_inner | epoch 333:     50 / 282 loss=3.043, nll_loss=1.126, glat_accu=0.581, glat_context_p=0.438, word_ins=2.923, length=2.93, ppl=8.24, wps=86884.8, ups=1.45, wpb=60000.1, bsz=2179.2, num_updates=93600, lr=0.000103362, gnorm=0.58, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:27:58 | INFO | train_inner | epoch 333:    150 / 282 loss=3.04, nll_loss=1.121, glat_accu=0.574, glat_context_p=0.438, word_ins=2.919, length=2.943, ppl=8.22, wps=132445, ups=2.18, wpb=60633.7, bsz=2125.6, num_updates=93700, lr=0.000103307, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:28:44 | INFO | train_inner | epoch 333:    250 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.579, glat_context_p=0.438, word_ins=2.914, length=2.928, ppl=8.19, wps=132590, ups=2.19, wpb=60558.6, bsz=2198.6, num_updates=93800, lr=0.000103252, gnorm=0.569, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:28:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:29:02 | INFO | valid | epoch 333 | valid on 'valid' subset | loss 12.266 | nll_loss 11.091 | word_ins 12.036 | length 4.617 | ppl 4924.8 | bleu 31.55 | wps 88355.5 | wpb 21176.3 | bsz 666.3 | num_updates 93832 | best_bleu 31.95
2023-06-14 01:29:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:29:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint333.pt (epoch 333 @ 93832 updates, score 31.55) (writing took 12.582796767354012 seconds)
2023-06-14 01:29:15 | INFO | fairseq_cli.train | end of epoch 333 (average epoch stats below)
2023-06-14 01:29:15 | INFO | train | epoch 333 | loss 3.038 | nll_loss 1.12 | glat_accu 0.579 | glat_context_p 0.438 | word_ins 2.917 | length 2.934 | ppl 8.21 | wps 112420 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 93832 | lr 0.000103234 | gnorm 0.576 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:29:15 | INFO | fairseq.trainer | begin training epoch 334
2023-06-14 01:29:53 | INFO | train_inner | epoch 334:     68 / 282 loss=3.044, nll_loss=1.126, glat_accu=0.579, glat_context_p=0.437, word_ins=2.923, length=2.954, ppl=8.25, wps=86322.3, ups=1.44, wpb=60001.6, bsz=2095.2, num_updates=93900, lr=0.000103197, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:30:39 | INFO | train_inner | epoch 334:    168 / 282 loss=3.038, nll_loss=1.12, glat_accu=0.579, glat_context_p=0.437, word_ins=2.918, length=2.929, ppl=8.21, wps=131712, ups=2.17, wpb=60745.6, bsz=2186.3, num_updates=94000, lr=0.000103142, gnorm=0.585, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:31:24 | INFO | train_inner | epoch 334:    268 / 282 loss=3.031, nll_loss=1.112, glat_accu=0.574, glat_context_p=0.437, word_ins=2.911, length=2.929, ppl=8.17, wps=134582, ups=2.22, wpb=60488.9, bsz=2179.8, num_updates=94100, lr=0.000103087, gnorm=0.575, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:31:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:31:34 | INFO | valid | epoch 334 | valid on 'valid' subset | loss 12.374 | nll_loss 11.205 | word_ins 12.143 | length 4.623 | ppl 5306.92 | bleu 31.55 | wps 88472.1 | wpb 21176.3 | bsz 666.3 | num_updates 94114 | best_bleu 31.95
2023-06-14 01:31:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:31:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint334.pt (epoch 334 @ 94114 updates, score 31.55) (writing took 13.764942698180676 seconds)
2023-06-14 01:31:48 | INFO | fairseq_cli.train | end of epoch 334 (average epoch stats below)
2023-06-14 01:31:48 | INFO | train | epoch 334 | loss 3.036 | nll_loss 1.118 | glat_accu 0.576 | glat_context_p 0.437 | word_ins 2.916 | length 2.936 | ppl 8.2 | wps 111410 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 94114 | lr 0.00010308 | gnorm 0.581 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 01:31:48 | INFO | fairseq.trainer | begin training epoch 335
2023-06-14 01:32:33 | INFO | train_inner | epoch 335:     86 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.571, glat_context_p=0.437, word_ins=2.909, length=2.933, ppl=8.16, wps=87479.8, ups=1.46, wpb=60048.1, bsz=2189.2, num_updates=94200, lr=0.000103033, gnorm=0.576, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:32:45 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 01:33:20 | INFO | train_inner | epoch 335:    187 / 282 loss=3.036, nll_loss=1.117, glat_accu=0.575, glat_context_p=0.437, word_ins=2.915, length=2.946, ppl=8.2, wps=130082, ups=2.15, wpb=60498.7, bsz=2137.8, num_updates=94300, lr=0.000102978, gnorm=0.568, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:34:03 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:34:07 | INFO | valid | epoch 335 | valid on 'valid' subset | loss 12.4 | nll_loss 11.225 | word_ins 12.161 | length 4.763 | ppl 5403.32 | bleu 31.34 | wps 85874.8 | wpb 21176.3 | bsz 666.3 | num_updates 94395 | best_bleu 31.95
2023-06-14 01:34:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:34:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint335.pt (epoch 335 @ 94395 updates, score 31.34) (writing took 12.887585509568453 seconds)
2023-06-14 01:34:19 | INFO | fairseq_cli.train | end of epoch 335 (average epoch stats below)
2023-06-14 01:34:19 | INFO | train | epoch 335 | loss 3.035 | nll_loss 1.117 | glat_accu 0.573 | glat_context_p 0.437 | word_ins 2.915 | length 2.937 | ppl 8.2 | wps 111774 | ups 1.85 | wpb 60412.3 | bsz 2156.6 | num_updates 94395 | lr 0.000102926 | gnorm 0.577 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:34:20 | INFO | fairseq.trainer | begin training epoch 336
2023-06-14 01:34:28 | INFO | train_inner | epoch 336:      5 / 282 loss=3.038, nll_loss=1.121, glat_accu=0.575, glat_context_p=0.437, word_ins=2.918, length=2.93, ppl=8.21, wps=87554.7, ups=1.45, wpb=60193.7, bsz=2136.8, num_updates=94400, lr=0.000102923, gnorm=0.591, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:35:15 | INFO | train_inner | epoch 336:    105 / 282 loss=3.033, nll_loss=1.115, glat_accu=0.57, glat_context_p=0.437, word_ins=2.913, length=2.945, ppl=8.19, wps=131144, ups=2.16, wpb=60617.4, bsz=2152, num_updates=94500, lr=0.000102869, gnorm=0.57, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:36:00 | INFO | train_inner | epoch 336:    205 / 282 loss=3.03, nll_loss=1.111, glat_accu=0.56, glat_context_p=0.437, word_ins=2.91, length=2.952, ppl=8.17, wps=132295, ups=2.19, wpb=60398.2, bsz=2157.7, num_updates=94600, lr=0.000102815, gnorm=0.564, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:36:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:36:39 | INFO | valid | epoch 336 | valid on 'valid' subset | loss 12.392 | nll_loss 11.229 | word_ins 12.157 | length 4.7 | ppl 5375.86 | bleu 31.62 | wps 88137.5 | wpb 21176.3 | bsz 666.3 | num_updates 94677 | best_bleu 31.95
2023-06-14 01:36:39 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:36:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint336.pt (epoch 336 @ 94677 updates, score 31.62) (writing took 12.332432299852371 seconds)
2023-06-14 01:36:51 | INFO | fairseq_cli.train | end of epoch 336 (average epoch stats below)
2023-06-14 01:36:51 | INFO | train | epoch 336 | loss 3.031 | nll_loss 1.113 | glat_accu 0.566 | glat_context_p 0.437 | word_ins 2.911 | length 2.935 | ppl 8.17 | wps 112454 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 94677 | lr 0.000102773 | gnorm 0.568 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:36:51 | INFO | fairseq.trainer | begin training epoch 337
2023-06-14 01:37:08 | INFO | train_inner | epoch 337:     23 / 282 loss=3.028, nll_loss=1.112, glat_accu=0.57, glat_context_p=0.437, word_ins=2.91, length=2.904, ppl=8.16, wps=89074.3, ups=1.48, wpb=60355.9, bsz=2170.2, num_updates=94700, lr=0.00010276, gnorm=0.569, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:37:53 | INFO | train_inner | epoch 337:    123 / 282 loss=3.033, nll_loss=1.115, glat_accu=0.573, glat_context_p=0.437, word_ins=2.913, length=2.937, ppl=8.19, wps=133204, ups=2.2, wpb=60613.7, bsz=2150.3, num_updates=94800, lr=0.000102706, gnorm=0.575, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:38:39 | INFO | train_inner | epoch 337:    223 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.566, glat_context_p=0.437, word_ins=2.911, length=2.936, ppl=8.17, wps=132232, ups=2.18, wpb=60552.4, bsz=2167.9, num_updates=94900, lr=0.000102652, gnorm=0.562, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:39:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:39:10 | INFO | valid | epoch 337 | valid on 'valid' subset | loss 12.549 | nll_loss 11.392 | word_ins 12.312 | length 4.746 | ppl 5993.13 | bleu 31.35 | wps 88156.5 | wpb 21176.3 | bsz 666.3 | num_updates 94959 | best_bleu 31.95
2023-06-14 01:39:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:39:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint337.pt (epoch 337 @ 94959 updates, score 31.35) (writing took 11.617128569632769 seconds)
2023-06-14 01:39:21 | INFO | fairseq_cli.train | end of epoch 337 (average epoch stats below)
2023-06-14 01:39:21 | INFO | train | epoch 337 | loss 3.033 | nll_loss 1.115 | glat_accu 0.572 | glat_context_p 0.437 | word_ins 2.913 | length 2.93 | ppl 8.18 | wps 113392 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 94959 | lr 0.00010262 | gnorm 0.573 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:39:21 | INFO | fairseq.trainer | begin training epoch 338
2023-06-14 01:39:46 | INFO | train_inner | epoch 338:     41 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.576, glat_context_p=0.437, word_ins=2.917, length=2.932, ppl=8.21, wps=89568.4, ups=1.5, wpb=59908.3, bsz=2140.9, num_updates=95000, lr=0.000102598, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:40:32 | INFO | train_inner | epoch 338:    141 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.571, glat_context_p=0.437, word_ins=2.911, length=2.92, ppl=8.17, wps=132618, ups=2.18, wpb=60772.2, bsz=2137.1, num_updates=95100, lr=0.000102544, gnorm=0.57, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:41:18 | INFO | train_inner | epoch 338:    241 / 282 loss=3.033, nll_loss=1.116, glat_accu=0.581, glat_context_p=0.437, word_ins=2.913, length=2.913, ppl=8.19, wps=133031, ups=2.2, wpb=60572.3, bsz=2211.4, num_updates=95200, lr=0.00010249, gnorm=0.578, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:41:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:41:39 | INFO | valid | epoch 338 | valid on 'valid' subset | loss 12.497 | nll_loss 11.341 | word_ins 12.263 | length 4.673 | ppl 5779.22 | bleu 31.19 | wps 87163.6 | wpb 21176.3 | bsz 666.3 | num_updates 95241 | best_bleu 31.95
2023-06-14 01:41:39 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:41:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint338.pt (epoch 338 @ 95241 updates, score 31.19) (writing took 13.431057915091515 seconds)
2023-06-14 01:41:53 | INFO | fairseq_cli.train | end of epoch 338 (average epoch stats below)
2023-06-14 01:41:53 | INFO | train | epoch 338 | loss 3.033 | nll_loss 1.115 | glat_accu 0.573 | glat_context_p 0.437 | word_ins 2.913 | length 2.931 | ppl 8.19 | wps 112387 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 95241 | lr 0.000102468 | gnorm 0.579 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 01:41:53 | INFO | fairseq.trainer | begin training epoch 339
2023-06-14 01:42:03 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 01:42:27 | INFO | train_inner | epoch 339:     60 / 282 loss=3.032, nll_loss=1.113, glat_accu=0.567, glat_context_p=0.437, word_ins=2.911, length=2.945, ppl=8.18, wps=87074.8, ups=1.45, wpb=60120.5, bsz=2149.5, num_updates=95300, lr=0.000102436, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:43:13 | INFO | train_inner | epoch 339:    160 / 282 loss=3.03, nll_loss=1.113, glat_accu=0.574, glat_context_p=0.436, word_ins=2.911, length=2.913, ppl=8.17, wps=131139, ups=2.16, wpb=60766.6, bsz=2166.3, num_updates=95400, lr=0.000102383, gnorm=0.567, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:43:59 | INFO | train_inner | epoch 339:    260 / 282 loss=3.035, nll_loss=1.117, glat_accu=0.567, glat_context_p=0.436, word_ins=2.914, length=2.946, ppl=8.19, wps=131578, ups=2.18, wpb=60374.6, bsz=2119.1, num_updates=95500, lr=0.000102329, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:44:09 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:44:12 | INFO | valid | epoch 339 | valid on 'valid' subset | loss 12.414 | nll_loss 11.246 | word_ins 12.174 | length 4.811 | ppl 5457.74 | bleu 31.66 | wps 88177.6 | wpb 21176.3 | bsz 666.3 | num_updates 95522 | best_bleu 31.95
2023-06-14 01:44:12 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:44:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint339.pt (epoch 339 @ 95522 updates, score 31.66) (writing took 12.757972311228514 seconds)
2023-06-14 01:44:25 | INFO | fairseq_cli.train | end of epoch 339 (average epoch stats below)
2023-06-14 01:44:25 | INFO | train | epoch 339 | loss 3.032 | nll_loss 1.114 | glat_accu 0.573 | glat_context_p 0.436 | word_ins 2.912 | length 2.926 | ppl 8.18 | wps 111744 | ups 1.85 | wpb 60417 | bsz 2157.5 | num_updates 95522 | lr 0.000102317 | gnorm 0.573 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:44:25 | INFO | fairseq.trainer | begin training epoch 340
2023-06-14 01:45:08 | INFO | train_inner | epoch 340:     78 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.574, glat_context_p=0.436, word_ins=2.908, length=2.931, ppl=8.16, wps=87223.7, ups=1.45, wpb=60021.5, bsz=2141.8, num_updates=95600, lr=0.000102275, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:45:53 | INFO | train_inner | epoch 340:    178 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.579, glat_context_p=0.436, word_ins=2.916, length=2.93, ppl=8.21, wps=132864, ups=2.19, wpb=60589.8, bsz=2182.5, num_updates=95700, lr=0.000102222, gnorm=0.58, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:46:39 | INFO | train_inner | epoch 340:    278 / 282 loss=3.034, nll_loss=1.117, glat_accu=0.567, glat_context_p=0.436, word_ins=2.914, length=2.928, ppl=8.19, wps=132251, ups=2.18, wpb=60614.9, bsz=2155.6, num_updates=95800, lr=0.000102169, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:46:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:46:44 | INFO | valid | epoch 340 | valid on 'valid' subset | loss 12.449 | nll_loss 11.289 | word_ins 12.212 | length 4.714 | ppl 5590.96 | bleu 31.34 | wps 88285.9 | wpb 21176.3 | bsz 666.3 | num_updates 95804 | best_bleu 31.95
2023-06-14 01:46:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:46:54 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint340.pt (epoch 340 @ 95804 updates, score 31.34) (writing took 10.726568885147572 seconds)
2023-06-14 01:46:54 | INFO | fairseq_cli.train | end of epoch 340 (average epoch stats below)
2023-06-14 01:46:54 | INFO | train | epoch 340 | loss 3.033 | nll_loss 1.115 | glat_accu 0.573 | glat_context_p 0.436 | word_ins 2.913 | length 2.929 | ppl 8.18 | wps 113752 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 95804 | lr 0.000102166 | gnorm 0.574 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:46:55 | INFO | fairseq.trainer | begin training epoch 341
2023-06-14 01:47:45 | INFO | train_inner | epoch 341:     96 / 282 loss=3.03, nll_loss=1.111, glat_accu=0.573, glat_context_p=0.436, word_ins=2.91, length=2.932, ppl=8.17, wps=90748.2, ups=1.51, wpb=60152.2, bsz=2125.5, num_updates=95900, lr=0.000102115, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:48:31 | INFO | train_inner | epoch 341:    196 / 282 loss=3.039, nll_loss=1.121, glat_accu=0.58, glat_context_p=0.436, word_ins=2.918, length=2.929, ppl=8.22, wps=133384, ups=2.19, wpb=60782.8, bsz=2150.2, num_updates=96000, lr=0.000102062, gnorm=0.574, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:49:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:49:13 | INFO | valid | epoch 341 | valid on 'valid' subset | loss 12.45 | nll_loss 11.296 | word_ins 12.222 | length 4.558 | ppl 5594.76 | bleu 31.69 | wps 85275.5 | wpb 21176.3 | bsz 666.3 | num_updates 96086 | best_bleu 31.95
2023-06-14 01:49:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:49:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint341.pt (epoch 341 @ 96086 updates, score 31.69) (writing took 11.809369038790464 seconds)
2023-06-14 01:49:25 | INFO | fairseq_cli.train | end of epoch 341 (average epoch stats below)
2023-06-14 01:49:25 | INFO | train | epoch 341 | loss 3.035 | nll_loss 1.117 | glat_accu 0.578 | glat_context_p 0.436 | word_ins 2.915 | length 2.931 | ppl 8.2 | wps 113104 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 96086 | lr 0.000102016 | gnorm 0.58 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:49:25 | INFO | fairseq.trainer | begin training epoch 342
2023-06-14 01:49:38 | INFO | train_inner | epoch 342:     14 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.583, glat_context_p=0.436, word_ins=2.915, length=2.923, ppl=8.2, wps=89740.5, ups=1.5, wpb=59838.4, bsz=2193.8, num_updates=96100, lr=0.000102009, gnorm=0.588, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:50:23 | INFO | train_inner | epoch 342:    114 / 282 loss=3.023, nll_loss=1.105, glat_accu=0.565, glat_context_p=0.436, word_ins=2.904, length=2.925, ppl=8.13, wps=132163, ups=2.18, wpb=60538.6, bsz=2161.8, num_updates=96200, lr=0.000101956, gnorm=0.559, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:50:57 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 01:51:10 | INFO | train_inner | epoch 342:    215 / 282 loss=3.038, nll_loss=1.12, glat_accu=0.579, glat_context_p=0.436, word_ins=2.917, length=2.933, ppl=8.21, wps=130434, ups=2.15, wpb=60632.6, bsz=2170.3, num_updates=96300, lr=0.000101903, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:51:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:51:44 | INFO | valid | epoch 342 | valid on 'valid' subset | loss 12.483 | nll_loss 11.318 | word_ins 12.245 | length 4.749 | ppl 5724.52 | bleu 31.22 | wps 89271.6 | wpb 21176.3 | bsz 666.3 | num_updates 96367 | best_bleu 31.95
2023-06-14 01:51:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:51:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint342.pt (epoch 342 @ 96367 updates, score 31.22) (writing took 11.888657454401255 seconds)
2023-06-14 01:51:56 | INFO | fairseq_cli.train | end of epoch 342 (average epoch stats below)
2023-06-14 01:51:56 | INFO | train | epoch 342 | loss 3.029 | nll_loss 1.111 | glat_accu 0.569 | glat_context_p 0.436 | word_ins 2.91 | length 2.929 | ppl 8.16 | wps 112623 | ups 1.86 | wpb 60415.6 | bsz 2157.6 | num_updates 96367 | lr 0.000101868 | gnorm 0.571 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:51:56 | INFO | fairseq.trainer | begin training epoch 343
2023-06-14 01:52:18 | INFO | train_inner | epoch 343:     33 / 282 loss=3.024, nll_loss=1.106, glat_accu=0.561, glat_context_p=0.436, word_ins=2.905, length=2.931, ppl=8.13, wps=88734.9, ups=1.47, wpb=60193.9, bsz=2132.4, num_updates=96400, lr=0.00010185, gnorm=0.576, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:53:04 | INFO | train_inner | epoch 343:    133 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.562, glat_context_p=0.436, word_ins=2.916, length=2.953, ppl=8.2, wps=131296, ups=2.16, wpb=60675.2, bsz=2096.9, num_updates=96500, lr=0.000101797, gnorm=0.578, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:53:49 | INFO | train_inner | epoch 343:    233 / 282 loss=3.03, nll_loss=1.112, glat_accu=0.588, glat_context_p=0.436, word_ins=2.91, length=2.911, ppl=8.17, wps=133317, ups=2.21, wpb=60399.2, bsz=2251.3, num_updates=96600, lr=0.000101745, gnorm=0.579, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:54:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:54:15 | INFO | valid | epoch 343 | valid on 'valid' subset | loss 12.398 | nll_loss 11.235 | word_ins 12.165 | length 4.671 | ppl 5398.73 | bleu 31.59 | wps 88189 | wpb 21176.3 | bsz 666.3 | num_updates 96649 | best_bleu 31.95
2023-06-14 01:54:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:54:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint343.pt (epoch 343 @ 96649 updates, score 31.59) (writing took 10.157529819756746 seconds)
2023-06-14 01:54:25 | INFO | fairseq_cli.train | end of epoch 343 (average epoch stats below)
2023-06-14 01:54:25 | INFO | train | epoch 343 | loss 3.031 | nll_loss 1.113 | glat_accu 0.574 | glat_context_p 0.436 | word_ins 2.911 | length 2.933 | ppl 8.17 | wps 114317 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 96649 | lr 0.000101719 | gnorm 0.581 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:54:25 | INFO | fairseq.trainer | begin training epoch 344
2023-06-14 01:54:54 | INFO | train_inner | epoch 344:     51 / 282 loss=3.03, nll_loss=1.111, glat_accu=0.578, glat_context_p=0.436, word_ins=2.91, length=2.932, ppl=8.17, wps=92459.5, ups=1.54, wpb=60150.7, bsz=2148.6, num_updates=96700, lr=0.000101692, gnorm=0.578, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 01:55:40 | INFO | train_inner | epoch 344:    151 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.577, glat_context_p=0.436, word_ins=2.913, length=2.932, ppl=8.19, wps=131220, ups=2.17, wpb=60582.3, bsz=2132.2, num_updates=96800, lr=0.000101639, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:56:27 | INFO | train_inner | epoch 344:    251 / 282 loss=3.037, nll_loss=1.119, glat_accu=0.585, glat_context_p=0.435, word_ins=2.916, length=2.921, ppl=8.21, wps=130937, ups=2.16, wpb=60533.1, bsz=2163.1, num_updates=96900, lr=0.000101587, gnorm=0.578, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:56:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:56:45 | INFO | valid | epoch 344 | valid on 'valid' subset | loss 12.375 | nll_loss 11.203 | word_ins 12.139 | length 4.724 | ppl 5312.33 | bleu 31.69 | wps 88466.1 | wpb 21176.3 | bsz 666.3 | num_updates 96931 | best_bleu 31.95
2023-06-14 01:56:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:57:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint344.pt (epoch 344 @ 96931 updates, score 31.69) (writing took 16.4471637532115 seconds)
2023-06-14 01:57:01 | INFO | fairseq_cli.train | end of epoch 344 (average epoch stats below)
2023-06-14 01:57:01 | INFO | train | epoch 344 | loss 3.034 | nll_loss 1.116 | glat_accu 0.58 | glat_context_p 0.435 | word_ins 2.913 | length 2.927 | ppl 8.19 | wps 108926 | ups 1.8 | wpb 60413.8 | bsz 2157.2 | num_updates 96931 | lr 0.000101571 | gnorm 0.58 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:57:01 | INFO | fairseq.trainer | begin training epoch 345
2023-06-14 01:57:39 | INFO | train_inner | epoch 345:     69 / 282 loss=3.03, nll_loss=1.112, glat_accu=0.573, glat_context_p=0.435, word_ins=2.91, length=2.925, ppl=8.17, wps=83352, ups=1.38, wpb=60238.3, bsz=2152.7, num_updates=97000, lr=0.000101535, gnorm=0.583, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:58:25 | INFO | train_inner | epoch 345:    169 / 282 loss=3.035, nll_loss=1.117, glat_accu=0.583, glat_context_p=0.435, word_ins=2.915, length=2.911, ppl=8.2, wps=132540, ups=2.19, wpb=60594.5, bsz=2179.3, num_updates=97100, lr=0.000101482, gnorm=0.581, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:59:11 | INFO | train_inner | epoch 345:    269 / 282 loss=3.028, nll_loss=1.108, glat_accu=0.569, glat_context_p=0.435, word_ins=2.907, length=2.953, ppl=8.15, wps=131548, ups=2.18, wpb=60454.7, bsz=2146.3, num_updates=97200, lr=0.00010143, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 01:59:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 01:59:20 | INFO | valid | epoch 345 | valid on 'valid' subset | loss 12.448 | nll_loss 11.278 | word_ins 12.205 | length 4.842 | ppl 5587.16 | bleu 31.42 | wps 89147.4 | wpb 21176.3 | bsz 666.3 | num_updates 97213 | best_bleu 31.95
2023-06-14 01:59:20 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 01:59:32 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint345.pt (epoch 345 @ 97213 updates, score 31.42) (writing took 12.236015655100346 seconds)
2023-06-14 01:59:32 | INFO | fairseq_cli.train | end of epoch 345 (average epoch stats below)
2023-06-14 01:59:32 | INFO | train | epoch 345 | loss 3.03 | nll_loss 1.112 | glat_accu 0.575 | glat_context_p 0.435 | word_ins 2.91 | length 2.926 | ppl 8.17 | wps 113122 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 97213 | lr 0.000101423 | gnorm 0.577 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 01:59:32 | INFO | fairseq.trainer | begin training epoch 346
2023-06-14 02:00:17 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 02:00:18 | INFO | train_inner | epoch 346:     88 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.581, glat_context_p=0.435, word_ins=2.907, length=2.909, ppl=8.15, wps=88558.6, ups=1.48, wpb=59940.3, bsz=2173.8, num_updates=97300, lr=0.000101378, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:01:04 | INFO | train_inner | epoch 346:    188 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.576, glat_context_p=0.435, word_ins=2.908, length=2.932, ppl=8.16, wps=131908, ups=2.17, wpb=60670.6, bsz=2182.8, num_updates=97400, lr=0.000101326, gnorm=0.58, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:01:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:01:52 | INFO | valid | epoch 346 | valid on 'valid' subset | loss 12.47 | nll_loss 11.305 | word_ins 12.23 | length 4.803 | ppl 5673.22 | bleu 31.54 | wps 85953.1 | wpb 21176.3 | bsz 666.3 | num_updates 97494 | best_bleu 31.95
2023-06-14 02:01:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:02:02 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint346.pt (epoch 346 @ 97494 updates, score 31.54) (writing took 10.140476427972317 seconds)
2023-06-14 02:02:02 | INFO | fairseq_cli.train | end of epoch 346 (average epoch stats below)
2023-06-14 02:02:02 | INFO | train | epoch 346 | loss 3.029 | nll_loss 1.111 | glat_accu 0.575 | glat_context_p 0.435 | word_ins 2.909 | length 2.929 | ppl 8.16 | wps 113316 | ups 1.88 | wpb 60411.7 | bsz 2157.1 | num_updates 97494 | lr 0.000101277 | gnorm 0.584 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:02:02 | INFO | fairseq.trainer | begin training epoch 347
2023-06-14 02:02:12 | INFO | train_inner | epoch 347:      6 / 282 loss=3.032, nll_loss=1.114, glat_accu=0.567, glat_context_p=0.435, word_ins=2.912, length=2.947, ppl=8.18, wps=89142.8, ups=1.48, wpb=60217.6, bsz=2098.8, num_updates=97500, lr=0.000101274, gnorm=0.587, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:02:57 | INFO | train_inner | epoch 347:    106 / 282 loss=3.028, nll_loss=1.109, glat_accu=0.572, glat_context_p=0.435, word_ins=2.908, length=2.933, ppl=8.16, wps=133967, ups=2.21, wpb=60571.6, bsz=2147.4, num_updates=97600, lr=0.000101222, gnorm=0.576, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:03:43 | INFO | train_inner | epoch 347:    206 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.578, glat_context_p=0.435, word_ins=2.913, length=2.932, ppl=8.19, wps=132404, ups=2.18, wpb=60601.9, bsz=2162.5, num_updates=97700, lr=0.00010117, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:04:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:04:21 | INFO | valid | epoch 347 | valid on 'valid' subset | loss 12.418 | nll_loss 11.241 | word_ins 12.176 | length 4.841 | ppl 5473.52 | bleu 31.6 | wps 88482.9 | wpb 21176.3 | bsz 666.3 | num_updates 97776 | best_bleu 31.95
2023-06-14 02:04:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:04:30 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint347.pt (epoch 347 @ 97776 updates, score 31.6) (writing took 8.432008042931557 seconds)
2023-06-14 02:04:30 | INFO | fairseq_cli.train | end of epoch 347 (average epoch stats below)
2023-06-14 02:04:30 | INFO | train | epoch 347 | loss 3.03 | nll_loss 1.112 | glat_accu 0.576 | glat_context_p 0.435 | word_ins 2.91 | length 2.928 | ppl 8.17 | wps 115265 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 97776 | lr 0.000101131 | gnorm 0.58 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:04:30 | INFO | fairseq.trainer | begin training epoch 348
2023-06-14 02:04:46 | INFO | train_inner | epoch 348:     24 / 282 loss=3.03, nll_loss=1.112, glat_accu=0.581, glat_context_p=0.435, word_ins=2.91, length=2.903, ppl=8.17, wps=95805.9, ups=1.59, wpb=60149.8, bsz=2179.9, num_updates=97800, lr=0.000101118, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:05:31 | INFO | train_inner | epoch 348:    124 / 282 loss=3.035, nll_loss=1.117, glat_accu=0.581, glat_context_p=0.435, word_ins=2.914, length=2.925, ppl=8.19, wps=132443, ups=2.18, wpb=60757.9, bsz=2153, num_updates=97900, lr=0.000101067, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:06:17 | INFO | train_inner | epoch 348:    224 / 282 loss=3.032, nll_loss=1.114, glat_accu=0.576, glat_context_p=0.435, word_ins=2.912, length=2.936, ppl=8.18, wps=131455, ups=2.18, wpb=60428.6, bsz=2152.2, num_updates=98000, lr=0.000101015, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:06:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:06:47 | INFO | valid | epoch 348 | valid on 'valid' subset | loss 12.628 | nll_loss 11.481 | word_ins 12.386 | length 4.836 | ppl 6327.86 | bleu 31.01 | wps 85434 | wpb 21176.3 | bsz 666.3 | num_updates 98058 | best_bleu 31.95
2023-06-14 02:06:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:06:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint348.pt (epoch 348 @ 98058 updates, score 31.01) (writing took 10.439130611717701 seconds)
2023-06-14 02:06:57 | INFO | fairseq_cli.train | end of epoch 348 (average epoch stats below)
2023-06-14 02:06:57 | INFO | train | epoch 348 | loss 3.032 | nll_loss 1.114 | glat_accu 0.577 | glat_context_p 0.435 | word_ins 2.912 | length 2.924 | ppl 8.18 | wps 115213 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 98058 | lr 0.000100985 | gnorm 0.578 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:06:57 | INFO | fairseq.trainer | begin training epoch 349
2023-06-14 02:07:23 | INFO | train_inner | epoch 349:     42 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.565, glat_context_p=0.435, word_ins=2.909, length=2.924, ppl=8.16, wps=92060.6, ups=1.53, wpb=60104.7, bsz=2127.3, num_updates=98100, lr=0.000100964, gnorm=0.586, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:08:09 | INFO | train_inner | epoch 349:    142 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.575, glat_context_p=0.435, word_ins=2.907, length=2.926, ppl=8.15, wps=132152, ups=2.18, wpb=60639.3, bsz=2160.4, num_updates=98200, lr=0.000100912, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:08:55 | INFO | train_inner | epoch 349:    242 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.579, glat_context_p=0.435, word_ins=2.909, length=2.93, ppl=8.16, wps=131378, ups=2.17, wpb=60413.3, bsz=2179, num_updates=98300, lr=0.000100861, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:09:05 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 02:09:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:09:16 | INFO | valid | epoch 349 | valid on 'valid' subset | loss 12.334 | nll_loss 11.15 | word_ins 12.091 | length 4.845 | ppl 5162.02 | bleu 31.62 | wps 84785.6 | wpb 21176.3 | bsz 666.3 | num_updates 98339 | best_bleu 31.95
2023-06-14 02:09:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:09:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint349.pt (epoch 349 @ 98339 updates, score 31.62) (writing took 10.07427228614688 seconds)
2023-06-14 02:09:26 | INFO | fairseq_cli.train | end of epoch 349 (average epoch stats below)
2023-06-14 02:09:26 | INFO | train | epoch 349 | loss 3.028 | nll_loss 1.109 | glat_accu 0.574 | glat_context_p 0.435 | word_ins 2.908 | length 2.924 | ppl 8.16 | wps 113970 | ups 1.89 | wpb 60420.6 | bsz 2156.8 | num_updates 98339 | lr 0.000100841 | gnorm 0.579 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:09:26 | INFO | fairseq.trainer | begin training epoch 350
2023-06-14 02:10:00 | INFO | train_inner | epoch 350:     61 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.569, glat_context_p=0.434, word_ins=2.905, length=2.923, ppl=8.14, wps=92137.6, ups=1.53, wpb=60143, bsz=2132.9, num_updates=98400, lr=0.00010081, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:10:46 | INFO | train_inner | epoch 350:    161 / 282 loss=3.026, nll_loss=1.109, glat_accu=0.579, glat_context_p=0.434, word_ins=2.907, length=2.899, ppl=8.15, wps=132392, ups=2.18, wpb=60726.2, bsz=2182.6, num_updates=98500, lr=0.000100759, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:11:32 | INFO | train_inner | epoch 350:    261 / 282 loss=3.037, nll_loss=1.118, glat_accu=0.581, glat_context_p=0.434, word_ins=2.915, length=2.94, ppl=8.21, wps=132012, ups=2.18, wpb=60435.2, bsz=2146.2, num_updates=98600, lr=0.000100707, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:11:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:11:44 | INFO | valid | epoch 350 | valid on 'valid' subset | loss 12.339 | nll_loss 11.163 | word_ins 12.106 | length 4.652 | ppl 5179.58 | bleu 31.93 | wps 88175 | wpb 21176.3 | bsz 666.3 | num_updates 98621 | best_bleu 31.95
2023-06-14 02:11:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:11:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint350.pt (epoch 350 @ 98621 updates, score 31.93) (writing took 11.090905498713255 seconds)
2023-06-14 02:11:55 | INFO | fairseq_cli.train | end of epoch 350 (average epoch stats below)
2023-06-14 02:11:55 | INFO | train | epoch 350 | loss 3.03 | nll_loss 1.112 | glat_accu 0.578 | glat_context_p 0.434 | word_ins 2.91 | length 2.922 | ppl 8.17 | wps 114445 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 98621 | lr 0.000100697 | gnorm 0.583 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:11:55 | INFO | fairseq.trainer | begin training epoch 351
2023-06-14 02:12:38 | INFO | train_inner | epoch 351:     79 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.577, glat_context_p=0.434, word_ins=2.908, length=2.934, ppl=8.16, wps=90653.8, ups=1.51, wpb=60066.7, bsz=2178.7, num_updates=98700, lr=0.000100656, gnorm=0.584, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:13:23 | INFO | train_inner | epoch 351:    179 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.583, glat_context_p=0.434, word_ins=2.91, length=2.909, ppl=8.17, wps=133413, ups=2.2, wpb=60642.7, bsz=2129.8, num_updates=98800, lr=0.000100605, gnorm=0.578, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:14:09 | INFO | train_inner | epoch 351:    279 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.578, glat_context_p=0.434, word_ins=2.905, length=2.914, ppl=8.14, wps=132145, ups=2.18, wpb=60615.2, bsz=2197.6, num_updates=98900, lr=0.000100555, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:14:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:14:14 | INFO | valid | epoch 351 | valid on 'valid' subset | loss 12.361 | nll_loss 11.194 | word_ins 12.13 | length 4.612 | ppl 5259.36 | bleu 31.93 | wps 88119.7 | wpb 21176.3 | bsz 666.3 | num_updates 98903 | best_bleu 31.95
2023-06-14 02:14:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:14:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint351.pt (epoch 351 @ 98903 updates, score 31.93) (writing took 7.4228870049119 seconds)
2023-06-14 02:14:21 | INFO | fairseq_cli.train | end of epoch 351 (average epoch stats below)
2023-06-14 02:14:21 | INFO | train | epoch 351 | loss 3.028 | nll_loss 1.11 | glat_accu 0.578 | glat_context_p 0.434 | word_ins 2.908 | length 2.923 | ppl 8.16 | wps 116892 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 98903 | lr 0.000100553 | gnorm 0.579 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:14:21 | INFO | fairseq.trainer | begin training epoch 352
2023-06-14 02:15:10 | INFO | train_inner | epoch 352:     97 / 282 loss=3.032, nll_loss=1.113, glat_accu=0.582, glat_context_p=0.434, word_ins=2.911, length=2.928, ppl=8.18, wps=98631.9, ups=1.64, wpb=60122.6, bsz=2129.4, num_updates=99000, lr=0.000100504, gnorm=0.583, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:15:55 | INFO | train_inner | epoch 352:    197 / 282 loss=3.033, nll_loss=1.114, glat_accu=0.581, glat_context_p=0.434, word_ins=2.912, length=2.929, ppl=8.18, wps=133352, ups=2.2, wpb=60546.2, bsz=2177, num_updates=99100, lr=0.000100453, gnorm=0.567, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:16:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:16:38 | INFO | valid | epoch 352 | valid on 'valid' subset | loss 12.428 | nll_loss 11.267 | word_ins 12.192 | length 4.705 | ppl 5510.83 | bleu 31.69 | wps 86897.4 | wpb 21176.3 | bsz 666.3 | num_updates 99185 | best_bleu 31.95
2023-06-14 02:16:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:16:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint352.pt (epoch 352 @ 99185 updates, score 31.69) (writing took 11.634598806500435 seconds)
2023-06-14 02:16:49 | INFO | fairseq_cli.train | end of epoch 352 (average epoch stats below)
2023-06-14 02:16:49 | INFO | train | epoch 352 | loss 3.032 | nll_loss 1.114 | glat_accu 0.58 | glat_context_p 0.434 | word_ins 2.912 | length 2.927 | ppl 8.18 | wps 114878 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 99185 | lr 0.00010041 | gnorm 0.576 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:16:49 | INFO | fairseq.trainer | begin training epoch 353
2023-06-14 02:17:03 | INFO | train_inner | epoch 353:     15 / 282 loss=3.035, nll_loss=1.117, glat_accu=0.575, glat_context_p=0.434, word_ins=2.915, length=2.936, ppl=8.2, wps=89005.6, ups=1.48, wpb=60078.7, bsz=2130.3, num_updates=99200, lr=0.000100402, gnorm=0.587, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:17:49 | INFO | train_inner | epoch 353:    115 / 282 loss=3.035, nll_loss=1.117, glat_accu=0.579, glat_context_p=0.434, word_ins=2.915, length=2.934, ppl=8.2, wps=132506, ups=2.19, wpb=60599.7, bsz=2132.9, num_updates=99300, lr=0.000100352, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:18:10 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 02:18:35 | INFO | train_inner | epoch 353:    216 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.583, glat_context_p=0.434, word_ins=2.908, length=2.919, ppl=8.16, wps=131554, ups=2.17, wpb=60663.6, bsz=2156.3, num_updates=99400, lr=0.000100301, gnorm=0.576, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:19:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:19:09 | INFO | valid | epoch 353 | valid on 'valid' subset | loss 12.41 | nll_loss 11.225 | word_ins 12.162 | length 4.964 | ppl 5443.85 | bleu 31.53 | wps 87883.2 | wpb 21176.3 | bsz 666.3 | num_updates 99466 | best_bleu 31.95
2023-06-14 02:19:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:19:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint353.pt (epoch 353 @ 99466 updates, score 31.53) (writing took 7.893811706453562 seconds)
2023-06-14 02:19:16 | INFO | fairseq_cli.train | end of epoch 353 (average epoch stats below)
2023-06-14 02:19:16 | INFO | train | epoch 353 | loss 3.032 | nll_loss 1.114 | glat_accu 0.58 | glat_context_p 0.434 | word_ins 2.912 | length 2.923 | ppl 8.18 | wps 115346 | ups 1.91 | wpb 60409.6 | bsz 2155 | num_updates 99466 | lr 0.000100268 | gnorm 0.575 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:19:17 | INFO | fairseq.trainer | begin training epoch 354
2023-06-14 02:19:38 | INFO | train_inner | epoch 354:     34 / 282 loss=3.03, nll_loss=1.113, glat_accu=0.58, glat_context_p=0.434, word_ins=2.911, length=2.904, ppl=8.17, wps=95920, ups=1.6, wpb=60123.5, bsz=2185.3, num_updates=99500, lr=0.000100251, gnorm=0.578, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:20:24 | INFO | train_inner | epoch 354:    134 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.578, glat_context_p=0.434, word_ins=2.904, length=2.924, ppl=8.14, wps=130182, ups=2.15, wpb=60532, bsz=2149.7, num_updates=99600, lr=0.000100201, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:21:10 | INFO | train_inner | epoch 354:    234 / 282 loss=3.032, nll_loss=1.114, glat_accu=0.577, glat_context_p=0.434, word_ins=2.911, length=2.935, ppl=8.18, wps=131947, ups=2.18, wpb=60585.6, bsz=2161.8, num_updates=99700, lr=0.00010015, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:21:32 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:21:35 | INFO | valid | epoch 354 | valid on 'valid' subset | loss 12.465 | nll_loss 11.289 | word_ins 12.22 | length 4.868 | ppl 5652.06 | bleu 31.52 | wps 88548.3 | wpb 21176.3 | bsz 666.3 | num_updates 99748 | best_bleu 31.95
2023-06-14 02:21:35 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:21:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint354.pt (epoch 354 @ 99748 updates, score 31.52) (writing took 10.411475993692875 seconds)
2023-06-14 02:21:45 | INFO | fairseq_cli.train | end of epoch 354 (average epoch stats below)
2023-06-14 02:21:45 | INFO | train | epoch 354 | loss 3.028 | nll_loss 1.109 | glat_accu 0.578 | glat_context_p 0.434 | word_ins 2.908 | length 2.922 | ppl 8.16 | wps 114313 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 99748 | lr 0.000100126 | gnorm 0.582 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-14 02:21:46 | INFO | fairseq.trainer | begin training epoch 355
2023-06-14 02:22:15 | INFO | train_inner | epoch 355:     52 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.581, glat_context_p=0.434, word_ins=2.906, length=2.912, ppl=8.15, wps=91668.6, ups=1.53, wpb=60062, bsz=2172.4, num_updates=99800, lr=0.0001001, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:23:01 | INFO | train_inner | epoch 355:    152 / 282 loss=3.029, nll_loss=1.111, glat_accu=0.586, glat_context_p=0.433, word_ins=2.909, length=2.912, ppl=8.16, wps=133069, ups=2.2, wpb=60565.2, bsz=2179.9, num_updates=99900, lr=0.00010005, gnorm=0.588, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:23:46 | INFO | train_inner | epoch 355:    252 / 282 loss=3.031, nll_loss=1.112, glat_accu=0.575, glat_context_p=0.433, word_ins=2.91, length=2.938, ppl=8.17, wps=132973, ups=2.2, wpb=60537.5, bsz=2127.6, num_updates=100000, lr=0.0001, gnorm=0.573, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:24:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:24:03 | INFO | valid | epoch 355 | valid on 'valid' subset | loss 12.389 | nll_loss 11.21 | word_ins 12.146 | length 4.858 | ppl 5362.18 | bleu 31.51 | wps 86038.1 | wpb 21176.3 | bsz 666.3 | num_updates 100030 | best_bleu 31.95
2023-06-14 02:24:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:24:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint355.pt (epoch 355 @ 100030 updates, score 31.51) (writing took 8.816321734338999 seconds)
2023-06-14 02:24:12 | INFO | fairseq_cli.train | end of epoch 355 (average epoch stats below)
2023-06-14 02:24:12 | INFO | train | epoch 355 | loss 3.029 | nll_loss 1.111 | glat_accu 0.579 | glat_context_p 0.433 | word_ins 2.909 | length 2.922 | ppl 8.16 | wps 116049 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 100030 | lr 9.9985e-05 | gnorm 0.583 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:24:12 | INFO | fairseq.trainer | begin training epoch 356
2023-06-14 02:24:50 | INFO | train_inner | epoch 356:     70 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.58, glat_context_p=0.433, word_ins=2.908, length=2.907, ppl=8.15, wps=95209.4, ups=1.58, wpb=60266.2, bsz=2158.2, num_updates=100100, lr=9.995e-05, gnorm=0.584, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:25:35 | INFO | train_inner | epoch 356:    170 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.578, glat_context_p=0.433, word_ins=2.906, length=2.915, ppl=8.14, wps=132761, ups=2.19, wpb=60541.7, bsz=2183.2, num_updates=100200, lr=9.99001e-05, gnorm=0.581, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:26:22 | INFO | train_inner | epoch 356:    270 / 282 loss=3.03, nll_loss=1.111, glat_accu=0.576, glat_context_p=0.433, word_ins=2.909, length=2.932, ppl=8.17, wps=130996, ups=2.17, wpb=60494.1, bsz=2132.8, num_updates=100300, lr=9.98503e-05, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:26:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:26:30 | INFO | valid | epoch 356 | valid on 'valid' subset | loss 12.435 | nll_loss 11.271 | word_ins 12.198 | length 4.732 | ppl 5538.04 | bleu 31.94 | wps 88183.9 | wpb 21176.3 | bsz 666.3 | num_updates 100312 | best_bleu 31.95
2023-06-14 02:26:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:26:38 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint356.pt (epoch 356 @ 100312 updates, score 31.94) (writing took 8.014351900666952 seconds)
2023-06-14 02:26:38 | INFO | fairseq_cli.train | end of epoch 356 (average epoch stats below)
2023-06-14 02:26:38 | INFO | train | epoch 356 | loss 3.029 | nll_loss 1.111 | glat_accu 0.578 | glat_context_p 0.433 | word_ins 2.909 | length 2.92 | ppl 8.16 | wps 116769 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 100312 | lr 9.98444e-05 | gnorm 0.578 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:26:38 | INFO | fairseq.trainer | begin training epoch 357
2023-06-14 02:27:10 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 02:27:24 | INFO | train_inner | epoch 357:     89 / 282 loss=3.032, nll_loss=1.113, glat_accu=0.579, glat_context_p=0.433, word_ins=2.911, length=2.932, ppl=8.18, wps=95651.4, ups=1.59, wpb=60128.3, bsz=2128.5, num_updates=100400, lr=9.98006e-05, gnorm=0.589, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:28:10 | INFO | train_inner | epoch 357:    189 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.576, glat_context_p=0.433, word_ins=2.907, length=2.921, ppl=8.15, wps=131681, ups=2.18, wpb=60398.9, bsz=2140.2, num_updates=100500, lr=9.97509e-05, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:28:53 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:28:56 | INFO | valid | epoch 357 | valid on 'valid' subset | loss 12.466 | nll_loss 11.302 | word_ins 12.227 | length 4.761 | ppl 5656.86 | bleu 31.72 | wps 86856.5 | wpb 21176.3 | bsz 666.3 | num_updates 100593 | best_bleu 31.95
2023-06-14 02:28:56 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:29:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint357.pt (epoch 357 @ 100593 updates, score 31.72) (writing took 10.370976623147726 seconds)
2023-06-14 02:29:06 | INFO | fairseq_cli.train | end of epoch 357 (average epoch stats below)
2023-06-14 02:29:06 | INFO | train | epoch 357 | loss 3.027 | nll_loss 1.108 | glat_accu 0.577 | glat_context_p 0.433 | word_ins 2.906 | length 2.923 | ppl 8.15 | wps 114618 | ups 1.9 | wpb 60407.1 | bsz 2156.2 | num_updates 100593 | lr 9.97048e-05 | gnorm 0.58 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:29:06 | INFO | fairseq.trainer | begin training epoch 358
2023-06-14 02:29:16 | INFO | train_inner | epoch 358:      7 / 282 loss=3.022, nll_loss=1.104, glat_accu=0.574, glat_context_p=0.433, word_ins=2.902, length=2.914, ppl=8.12, wps=91968, ups=1.52, wpb=60327.3, bsz=2194.6, num_updates=100600, lr=9.97013e-05, gnorm=0.571, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:30:02 | INFO | train_inner | epoch 358:    107 / 282 loss=3.03, nll_loss=1.111, glat_accu=0.58, glat_context_p=0.433, word_ins=2.909, length=2.928, ppl=8.17, wps=131579, ups=2.18, wpb=60443.4, bsz=2151.9, num_updates=100700, lr=9.96518e-05, gnorm=0.572, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:30:47 | INFO | train_inner | epoch 358:    207 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.576, glat_context_p=0.433, word_ins=2.907, length=2.915, ppl=8.15, wps=133002, ups=2.19, wpb=60628.2, bsz=2162.2, num_updates=100800, lr=9.96024e-05, gnorm=0.572, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:31:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:31:25 | INFO | valid | epoch 358 | valid on 'valid' subset | loss 12.459 | nll_loss 11.296 | word_ins 12.222 | length 4.727 | ppl 5631.94 | bleu 31.63 | wps 87902.5 | wpb 21176.3 | bsz 666.3 | num_updates 100875 | best_bleu 31.95
2023-06-14 02:31:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:31:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint358.pt (epoch 358 @ 100875 updates, score 31.63) (writing took 9.371694456785917 seconds)
2023-06-14 02:31:34 | INFO | fairseq_cli.train | end of epoch 358 (average epoch stats below)
2023-06-14 02:31:34 | INFO | train | epoch 358 | loss 3.027 | nll_loss 1.109 | glat_accu 0.577 | glat_context_p 0.433 | word_ins 2.907 | length 2.919 | ppl 8.15 | wps 115095 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 100875 | lr 9.95654e-05 | gnorm 0.578 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:31:34 | INFO | fairseq.trainer | begin training epoch 359
2023-06-14 02:31:51 | INFO | train_inner | epoch 359:     25 / 282 loss=3.023, nll_loss=1.105, glat_accu=0.574, glat_context_p=0.433, word_ins=2.903, length=2.922, ppl=8.13, wps=94430.7, ups=1.57, wpb=60059, bsz=2153.8, num_updates=100900, lr=9.9553e-05, gnorm=0.593, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:32:37 | INFO | train_inner | epoch 359:    125 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.577, glat_context_p=0.433, word_ins=2.907, length=2.913, ppl=8.15, wps=131965, ups=2.17, wpb=60773.8, bsz=2150.3, num_updates=101000, lr=9.95037e-05, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:33:23 | INFO | train_inner | epoch 359:    225 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.579, glat_context_p=0.433, word_ins=2.905, length=2.915, ppl=8.14, wps=132636, ups=2.19, wpb=60591.7, bsz=2190.5, num_updates=101100, lr=9.94545e-05, gnorm=0.571, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:33:48 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:33:52 | INFO | valid | epoch 359 | valid on 'valid' subset | loss 12.352 | nll_loss 11.186 | word_ins 12.12 | length 4.648 | ppl 5227.3 | bleu 31.6 | wps 84228 | wpb 21176.3 | bsz 666.3 | num_updates 101157 | best_bleu 31.95
2023-06-14 02:33:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:34:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint359.pt (epoch 359 @ 101157 updates, score 31.6) (writing took 12.114899791777134 seconds)
2023-06-14 02:34:04 | INFO | fairseq_cli.train | end of epoch 359 (average epoch stats below)
2023-06-14 02:34:04 | INFO | train | epoch 359 | loss 3.026 | nll_loss 1.108 | glat_accu 0.577 | glat_context_p 0.433 | word_ins 2.906 | length 2.922 | ppl 8.15 | wps 113991 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 101157 | lr 9.94265e-05 | gnorm 0.579 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:34:04 | INFO | fairseq.trainer | begin training epoch 360
2023-06-14 02:34:29 | INFO | train_inner | epoch 360:     43 / 282 loss=3.031, nll_loss=1.112, glat_accu=0.576, glat_context_p=0.433, word_ins=2.91, length=2.933, ppl=8.17, wps=90320.3, ups=1.5, wpb=60096.1, bsz=2134.3, num_updates=101200, lr=9.94053e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:35:15 | INFO | train_inner | epoch 360:    143 / 282 loss=3.023, nll_loss=1.105, glat_accu=0.579, glat_context_p=0.433, word_ins=2.903, length=2.917, ppl=8.13, wps=132115, ups=2.18, wpb=60551.7, bsz=2174.2, num_updates=101300, lr=9.93563e-05, gnorm=0.585, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:35:58 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 02:36:02 | INFO | train_inner | epoch 360:    244 / 282 loss=3.024, nll_loss=1.106, glat_accu=0.576, glat_context_p=0.432, word_ins=2.904, length=2.911, ppl=8.13, wps=130743, ups=2.16, wpb=60668.4, bsz=2155.3, num_updates=101400, lr=9.93073e-05, gnorm=0.579, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:36:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:36:22 | INFO | valid | epoch 360 | valid on 'valid' subset | loss 12.483 | nll_loss 11.326 | word_ins 12.248 | length 4.695 | ppl 5724.52 | bleu 31.58 | wps 87878.8 | wpb 21176.3 | bsz 666.3 | num_updates 101438 | best_bleu 31.95
2023-06-14 02:36:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:36:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint360.pt (epoch 360 @ 101438 updates, score 31.58) (writing took 8.888727523386478 seconds)
2023-06-14 02:36:31 | INFO | fairseq_cli.train | end of epoch 360 (average epoch stats below)
2023-06-14 02:36:31 | INFO | train | epoch 360 | loss 3.026 | nll_loss 1.108 | glat_accu 0.576 | glat_context_p 0.432 | word_ins 2.906 | length 2.921 | ppl 8.15 | wps 115310 | ups 1.91 | wpb 60413.2 | bsz 2152.2 | num_updates 101438 | lr 9.92887e-05 | gnorm 0.584 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:36:31 | INFO | fairseq.trainer | begin training epoch 361
2023-06-14 02:37:05 | INFO | train_inner | epoch 361:     62 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.578, glat_context_p=0.432, word_ins=2.906, length=2.924, ppl=8.15, wps=94871.5, ups=1.58, wpb=59907.1, bsz=2148.2, num_updates=101500, lr=9.92583e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:37:51 | INFO | train_inner | epoch 361:    162 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.575, glat_context_p=0.432, word_ins=2.913, length=2.934, ppl=8.19, wps=132272, ups=2.18, wpb=60691.1, bsz=2098.9, num_updates=101600, lr=9.92095e-05, gnorm=0.585, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:38:37 | INFO | train_inner | epoch 361:    262 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.576, glat_context_p=0.432, word_ins=2.9, length=2.914, ppl=8.11, wps=131126, ups=2.17, wpb=60490.4, bsz=2200.2, num_updates=101700, lr=9.91607e-05, gnorm=0.565, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:38:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:38:49 | INFO | valid | epoch 361 | valid on 'valid' subset | loss 12.412 | nll_loss 11.235 | word_ins 12.169 | length 4.857 | ppl 5449.4 | bleu 31.66 | wps 88025.9 | wpb 21176.3 | bsz 666.3 | num_updates 101720 | best_bleu 31.95
2023-06-14 02:38:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:38:59 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint361.pt (epoch 361 @ 101720 updates, score 31.66) (writing took 10.45452180877328 seconds)
2023-06-14 02:38:59 | INFO | fairseq_cli.train | end of epoch 361 (average epoch stats below)
2023-06-14 02:38:59 | INFO | train | epoch 361 | loss 3.026 | nll_loss 1.107 | glat_accu 0.578 | glat_context_p 0.432 | word_ins 2.906 | length 2.921 | ppl 8.14 | wps 114849 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 101720 | lr 9.91509e-05 | gnorm 0.577 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:38:59 | INFO | fairseq.trainer | begin training epoch 362
2023-06-14 02:39:42 | INFO | train_inner | epoch 362:     80 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.582, glat_context_p=0.432, word_ins=2.903, length=2.914, ppl=8.13, wps=92377.1, ups=1.54, wpb=60064.2, bsz=2193.7, num_updates=101800, lr=9.9112e-05, gnorm=0.584, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:40:28 | INFO | train_inner | epoch 362:    180 / 282 loss=3.029, nll_loss=1.111, glat_accu=0.576, glat_context_p=0.432, word_ins=2.909, length=2.926, ppl=8.16, wps=131289, ups=2.17, wpb=60549.9, bsz=2132, num_updates=101900, lr=9.90633e-05, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:41:14 | INFO | train_inner | epoch 362:    280 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.58, glat_context_p=0.432, word_ins=2.907, length=2.922, ppl=8.15, wps=132187, ups=2.18, wpb=60663.5, bsz=2156.5, num_updates=102000, lr=9.90148e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:41:14 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:41:18 | INFO | valid | epoch 362 | valid on 'valid' subset | loss 12.377 | nll_loss 11.195 | word_ins 12.129 | length 4.96 | ppl 5320.45 | bleu 31.7 | wps 87638.8 | wpb 21176.3 | bsz 666.3 | num_updates 102002 | best_bleu 31.95
2023-06-14 02:41:18 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:41:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint362.pt (epoch 362 @ 102002 updates, score 31.7) (writing took 10.342727083712816 seconds)
2023-06-14 02:41:28 | INFO | fairseq_cli.train | end of epoch 362 (average epoch stats below)
2023-06-14 02:41:28 | INFO | train | epoch 362 | loss 3.026 | nll_loss 1.108 | glat_accu 0.579 | glat_context_p 0.432 | word_ins 2.906 | length 2.919 | ppl 8.15 | wps 114540 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 102002 | lr 9.90138e-05 | gnorm 0.589 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:41:28 | INFO | fairseq.trainer | begin training epoch 363
2023-06-14 02:42:20 | INFO | train_inner | epoch 363:     98 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.574, glat_context_p=0.432, word_ins=2.908, length=2.933, ppl=8.16, wps=90074.9, ups=1.5, wpb=60029.6, bsz=2101.9, num_updates=102100, lr=9.89663e-05, gnorm=0.584, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:42:36 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 02:43:06 | INFO | train_inner | epoch 363:    199 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.592, glat_context_p=0.432, word_ins=2.903, length=2.901, ppl=8.13, wps=132830, ups=2.19, wpb=60646.8, bsz=2188.8, num_updates=102200, lr=9.89178e-05, gnorm=0.581, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:43:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:43:47 | INFO | valid | epoch 363 | valid on 'valid' subset | loss 12.413 | nll_loss 11.239 | word_ins 12.171 | length 4.85 | ppl 5454.96 | bleu 31.72 | wps 89924.3 | wpb 21176.3 | bsz 666.3 | num_updates 102283 | best_bleu 31.95
2023-06-14 02:43:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:43:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint363.pt (epoch 363 @ 102283 updates, score 31.72) (writing took 8.738220386207104 seconds)
2023-06-14 02:43:56 | INFO | fairseq_cli.train | end of epoch 363 (average epoch stats below)
2023-06-14 02:43:56 | INFO | train | epoch 363 | loss 3.025 | nll_loss 1.107 | glat_accu 0.58 | glat_context_p 0.432 | word_ins 2.905 | length 2.92 | ppl 8.14 | wps 114820 | ups 1.9 | wpb 60416.9 | bsz 2157.7 | num_updates 102283 | lr 9.88777e-05 | gnorm 0.58 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 02:43:56 | INFO | fairseq.trainer | begin training epoch 364
2023-06-14 02:44:09 | INFO | train_inner | epoch 364:     17 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.574, glat_context_p=0.432, word_ins=2.904, length=2.93, ppl=8.14, wps=95960.1, ups=1.6, wpb=60131.8, bsz=2154.1, num_updates=102300, lr=9.88695e-05, gnorm=0.583, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 02:44:54 | INFO | train_inner | epoch 364:    117 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.586, glat_context_p=0.432, word_ins=2.899, length=2.902, ppl=8.1, wps=132527, ups=2.19, wpb=60567.2, bsz=2199.7, num_updates=102400, lr=9.88212e-05, gnorm=0.578, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:45:40 | INFO | train_inner | epoch 364:    217 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.578, glat_context_p=0.432, word_ins=2.905, length=2.924, ppl=8.14, wps=132388, ups=2.18, wpb=60675.9, bsz=2165.3, num_updates=102500, lr=9.8773e-05, gnorm=0.576, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:46:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:46:13 | INFO | valid | epoch 364 | valid on 'valid' subset | loss 12.421 | nll_loss 11.252 | word_ins 12.18 | length 4.828 | ppl 5483.75 | bleu 31.41 | wps 88456.1 | wpb 21176.3 | bsz 666.3 | num_updates 102565 | best_bleu 31.95
2023-06-14 02:46:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:46:24 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint364.pt (epoch 364 @ 102565 updates, score 31.41) (writing took 10.577609527856112 seconds)
2023-06-14 02:46:24 | INFO | fairseq_cli.train | end of epoch 364 (average epoch stats below)
2023-06-14 02:46:24 | INFO | train | epoch 364 | loss 3.025 | nll_loss 1.106 | glat_accu 0.58 | glat_context_p 0.432 | word_ins 2.904 | length 2.922 | ppl 8.14 | wps 115002 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 102565 | lr 9.87417e-05 | gnorm 0.583 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 02:46:24 | INFO | fairseq.trainer | begin training epoch 365
2023-06-14 02:46:46 | INFO | train_inner | epoch 365:     35 / 282 loss=3.028, nll_loss=1.109, glat_accu=0.574, glat_context_p=0.432, word_ins=2.908, length=2.937, ppl=8.16, wps=91642.2, ups=1.53, wpb=59991.4, bsz=2100.9, num_updates=102600, lr=9.87248e-05, gnorm=0.595, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:47:32 | INFO | train_inner | epoch 365:    135 / 282 loss=3.022, nll_loss=1.104, glat_accu=0.574, glat_context_p=0.432, word_ins=2.903, length=2.92, ppl=8.13, wps=132142, ups=2.18, wpb=60715.7, bsz=2147.6, num_updates=102700, lr=9.86767e-05, gnorm=0.588, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:48:17 | INFO | train_inner | epoch 365:    235 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.581, glat_context_p=0.432, word_ins=2.902, length=2.912, ppl=8.12, wps=132944, ups=2.2, wpb=60456.7, bsz=2198.9, num_updates=102800, lr=9.86287e-05, gnorm=0.59, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 02:48:39 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:48:42 | INFO | valid | epoch 365 | valid on 'valid' subset | loss 12.337 | nll_loss 11.161 | word_ins 12.099 | length 4.749 | ppl 5172.55 | bleu 31.53 | wps 88377.1 | wpb 21176.3 | bsz 666.3 | num_updates 102847 | best_bleu 31.95
2023-06-14 02:48:42 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:48:51 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint365.pt (epoch 365 @ 102847 updates, score 31.53) (writing took 9.07465236261487 seconds)
2023-06-14 02:48:51 | INFO | fairseq_cli.train | end of epoch 365 (average epoch stats below)
2023-06-14 02:48:51 | INFO | train | epoch 365 | loss 3.024 | nll_loss 1.105 | glat_accu 0.579 | glat_context_p 0.432 | word_ins 2.904 | length 2.92 | ppl 8.13 | wps 116035 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 102847 | lr 9.86062e-05 | gnorm 0.592 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 02:48:51 | INFO | fairseq.trainer | begin training epoch 366
2023-06-14 02:49:21 | INFO | train_inner | epoch 366:     53 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.585, glat_context_p=0.431, word_ins=2.905, length=2.915, ppl=8.14, wps=94770.7, ups=1.57, wpb=60255.8, bsz=2171.8, num_updates=102900, lr=9.85808e-05, gnorm=0.591, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:50:07 | INFO | train_inner | epoch 366:    153 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.576, glat_context_p=0.431, word_ins=2.908, length=2.93, ppl=8.16, wps=131837, ups=2.17, wpb=60619.1, bsz=2109.6, num_updates=103000, lr=9.85329e-05, gnorm=0.589, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 02:50:52 | INFO | train_inner | epoch 366:    253 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.588, glat_context_p=0.431, word_ins=2.906, length=2.915, ppl=8.15, wps=133002, ups=2.2, wpb=60505.1, bsz=2197, num_updates=103100, lr=9.84851e-05, gnorm=0.603, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 02:51:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:51:09 | INFO | valid | epoch 366 | valid on 'valid' subset | loss 12.399 | nll_loss 11.237 | word_ins 12.163 | length 4.706 | ppl 5400.57 | bleu 31.82 | wps 87803.2 | wpb 21176.3 | bsz 666.3 | num_updates 103129 | best_bleu 31.95
2023-06-14 02:51:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:51:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint366.pt (epoch 366 @ 103129 updates, score 31.82) (writing took 12.57161270827055 seconds)
2023-06-14 02:51:21 | INFO | fairseq_cli.train | end of epoch 366 (average epoch stats below)
2023-06-14 02:51:21 | INFO | train | epoch 366 | loss 3.027 | nll_loss 1.108 | glat_accu 0.583 | glat_context_p 0.431 | word_ins 2.906 | length 2.918 | ppl 8.15 | wps 113421 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 103129 | lr 9.84713e-05 | gnorm 0.593 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 02:51:21 | INFO | fairseq.trainer | begin training epoch 367
2023-06-14 02:51:59 | INFO | train_inner | epoch 367:     71 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.589, glat_context_p=0.431, word_ins=2.904, length=2.911, ppl=8.14, wps=89532.8, ups=1.49, wpb=59979.2, bsz=2155.3, num_updates=103200, lr=9.84374e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:52:45 | INFO | train_inner | epoch 367:    171 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.585, glat_context_p=0.431, word_ins=2.905, length=2.921, ppl=8.14, wps=132846, ups=2.19, wpb=60648.4, bsz=2181.1, num_updates=103300, lr=9.83897e-05, gnorm=0.582, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:53:31 | INFO | train_inner | epoch 367:    271 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.583, glat_context_p=0.431, word_ins=2.911, length=2.93, ppl=8.18, wps=130995, ups=2.16, wpb=60583.9, bsz=2128.3, num_updates=103400, lr=9.83422e-05, gnorm=0.6, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:53:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:53:40 | INFO | valid | epoch 367 | valid on 'valid' subset | loss 12.346 | nll_loss 11.157 | word_ins 12.103 | length 4.874 | ppl 5207.81 | bleu 31.71 | wps 89083.7 | wpb 21176.3 | bsz 666.3 | num_updates 103411 | best_bleu 31.95
2023-06-14 02:53:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:53:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint367.pt (epoch 367 @ 103411 updates, score 31.71) (writing took 10.423452086746693 seconds)
2023-06-14 02:53:50 | INFO | fairseq_cli.train | end of epoch 367 (average epoch stats below)
2023-06-14 02:53:50 | INFO | train | epoch 367 | loss 3.027 | nll_loss 1.108 | glat_accu 0.585 | glat_context_p 0.431 | word_ins 2.906 | length 2.92 | ppl 8.15 | wps 114080 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 103411 | lr 9.83369e-05 | gnorm 0.593 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 02:53:51 | INFO | fairseq.trainer | begin training epoch 368
2023-06-14 02:54:38 | INFO | train_inner | epoch 368:     89 / 282 loss=3.029, nll_loss=1.111, glat_accu=0.588, glat_context_p=0.431, word_ins=2.909, length=2.901, ppl=8.16, wps=90336.1, ups=1.5, wpb=60033.1, bsz=2147, num_updates=103500, lr=9.82946e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:55:24 | INFO | train_inner | epoch 368:    189 / 282 loss=3.031, nll_loss=1.112, glat_accu=0.581, glat_context_p=0.431, word_ins=2.91, length=2.934, ppl=8.17, wps=131722, ups=2.17, wpb=60599.8, bsz=2152.6, num_updates=103600, lr=9.82472e-05, gnorm=0.589, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:56:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:56:09 | INFO | valid | epoch 368 | valid on 'valid' subset | loss 12.39 | nll_loss 11.216 | word_ins 12.149 | length 4.812 | ppl 5365.83 | bleu 31.66 | wps 88547.5 | wpb 21176.3 | bsz 666.3 | num_updates 103693 | best_bleu 31.95
2023-06-14 02:56:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:56:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint368.pt (epoch 368 @ 103693 updates, score 31.66) (writing took 9.839724987745285 seconds)
2023-06-14 02:56:19 | INFO | fairseq_cli.train | end of epoch 368 (average epoch stats below)
2023-06-14 02:56:19 | INFO | train | epoch 368 | loss 3.029 | nll_loss 1.11 | glat_accu 0.586 | glat_context_p 0.431 | word_ins 2.908 | length 2.918 | ppl 8.16 | wps 114607 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 103693 | lr 9.82031e-05 | gnorm 0.588 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:56:19 | INFO | fairseq.trainer | begin training epoch 369
2023-06-14 02:56:28 | INFO | train_inner | epoch 369:      7 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.591, glat_context_p=0.431, word_ins=2.906, length=2.92, ppl=8.15, wps=93791.7, ups=1.56, wpb=60166.5, bsz=2158.3, num_updates=103700, lr=9.81998e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 02:57:14 | INFO | train_inner | epoch 369:    107 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.589, glat_context_p=0.431, word_ins=2.904, length=2.913, ppl=8.14, wps=131389, ups=2.17, wpb=60425, bsz=2181.7, num_updates=103800, lr=9.81525e-05, gnorm=0.578, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:58:00 | INFO | train_inner | epoch 369:    207 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.575, glat_context_p=0.431, word_ins=2.904, length=2.927, ppl=8.14, wps=131706, ups=2.17, wpb=60638.1, bsz=2143.4, num_updates=103900, lr=9.81052e-05, gnorm=0.581, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:58:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 02:58:38 | INFO | valid | epoch 369 | valid on 'valid' subset | loss 12.383 | nll_loss 11.203 | word_ins 12.142 | length 4.835 | ppl 5343.09 | bleu 31.28 | wps 86934.4 | wpb 21176.3 | bsz 666.3 | num_updates 103975 | best_bleu 31.95
2023-06-14 02:58:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 02:58:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint369.pt (epoch 369 @ 103975 updates, score 31.28) (writing took 12.292043656110764 seconds)
2023-06-14 02:58:50 | INFO | fairseq_cli.train | end of epoch 369 (average epoch stats below)
2023-06-14 02:58:50 | INFO | train | epoch 369 | loss 3.024 | nll_loss 1.105 | glat_accu 0.58 | glat_context_p 0.431 | word_ins 2.903 | length 2.919 | ppl 8.13 | wps 112926 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 103975 | lr 9.80699e-05 | gnorm 0.582 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 02:58:50 | INFO | fairseq.trainer | begin training epoch 370
2023-06-14 02:59:07 | INFO | train_inner | epoch 370:     25 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.573, glat_context_p=0.431, word_ins=2.902, length=2.922, ppl=8.12, wps=88887.3, ups=1.48, wpb=60237.3, bsz=2127.9, num_updates=104000, lr=9.80581e-05, gnorm=0.584, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 02:59:53 | INFO | train_inner | epoch 370:    125 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.579, glat_context_p=0.431, word_ins=2.899, length=2.916, ppl=8.11, wps=132247, ups=2.18, wpb=60540.3, bsz=2162.9, num_updates=104100, lr=9.8011e-05, gnorm=0.574, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:00:31 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:00:40 | INFO | train_inner | epoch 370:    226 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.584, glat_context_p=0.431, word_ins=2.903, length=2.909, ppl=8.13, wps=130102, ups=2.15, wpb=60493.5, bsz=2158.8, num_updates=104200, lr=9.79639e-05, gnorm=0.593, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:01:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:01:08 | INFO | valid | epoch 370 | valid on 'valid' subset | loss 12.409 | nll_loss 11.241 | word_ins 12.169 | length 4.79 | ppl 5437.38 | bleu 31.68 | wps 86213 | wpb 21176.3 | bsz 666.3 | num_updates 104256 | best_bleu 31.95
2023-06-14 03:01:08 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:01:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint370.pt (epoch 370 @ 104256 updates, score 31.68) (writing took 10.310750711709261 seconds)
2023-06-14 03:01:19 | INFO | fairseq_cli.train | end of epoch 370 (average epoch stats below)
2023-06-14 03:01:19 | INFO | train | epoch 370 | loss 3.022 | nll_loss 1.103 | glat_accu 0.579 | glat_context_p 0.431 | word_ins 2.902 | length 2.915 | ppl 8.12 | wps 114143 | ups 1.89 | wpb 60407.4 | bsz 2156.9 | num_updates 104256 | lr 9.79376e-05 | gnorm 0.584 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:01:19 | INFO | fairseq.trainer | begin training epoch 371
2023-06-14 03:01:45 | INFO | train_inner | epoch 371:     44 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.582, glat_context_p=0.431, word_ins=2.903, length=2.909, ppl=8.13, wps=92344, ups=1.53, wpb=60222.8, bsz=2170.8, num_updates=104300, lr=9.79169e-05, gnorm=0.596, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:02:30 | INFO | train_inner | epoch 371:    144 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.589, glat_context_p=0.43, word_ins=2.9, length=2.91, ppl=8.11, wps=132982, ups=2.2, wpb=60523.5, bsz=2203.8, num_updates=104400, lr=9.787e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:03:16 | INFO | train_inner | epoch 371:    244 / 282 loss=3.031, nll_loss=1.112, glat_accu=0.584, glat_context_p=0.43, word_ins=2.91, length=2.928, ppl=8.17, wps=131822, ups=2.18, wpb=60588, bsz=2132.7, num_updates=104500, lr=9.78232e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:03:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:03:37 | INFO | valid | epoch 371 | valid on 'valid' subset | loss 12.316 | nll_loss 11.138 | word_ins 12.078 | length 4.753 | ppl 5100.15 | bleu 31.99 | wps 88948.9 | wpb 21176.3 | bsz 666.3 | num_updates 104538 | best_bleu 31.99
2023-06-14 03:03:37 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:03:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint371.pt (epoch 371 @ 104538 updates, score 31.99) (writing took 16.322424679994583 seconds)
2023-06-14 03:03:53 | INFO | fairseq_cli.train | end of epoch 371 (average epoch stats below)
2023-06-14 03:03:53 | INFO | train | epoch 371 | loss 3.026 | nll_loss 1.107 | glat_accu 0.587 | glat_context_p 0.43 | word_ins 2.905 | length 2.915 | ppl 8.14 | wps 110094 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 104538 | lr 9.78054e-05 | gnorm 0.593 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:03:54 | INFO | fairseq.trainer | begin training epoch 372
2023-06-14 03:04:28 | INFO | train_inner | epoch 372:     62 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.589, glat_context_p=0.43, word_ins=2.907, length=2.904, ppl=8.15, wps=83427.8, ups=1.39, wpb=60041.3, bsz=2157, num_updates=104600, lr=9.77764e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:05:14 | INFO | train_inner | epoch 372:    162 / 282 loss=3.024, nll_loss=1.106, glat_accu=0.586, glat_context_p=0.43, word_ins=2.904, length=2.906, ppl=8.14, wps=132813, ups=2.19, wpb=60598.4, bsz=2181.3, num_updates=104700, lr=9.77297e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:06:00 | INFO | train_inner | epoch 372:    262 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.583, glat_context_p=0.43, word_ins=2.907, length=2.921, ppl=8.15, wps=130499, ups=2.15, wpb=60570.9, bsz=2133.3, num_updates=104800, lr=9.76831e-05, gnorm=0.591, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:06:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:06:13 | INFO | valid | epoch 372 | valid on 'valid' subset | loss 12.411 | nll_loss 11.24 | word_ins 12.169 | length 4.856 | ppl 5446.62 | bleu 31.35 | wps 88977.2 | wpb 21176.3 | bsz 666.3 | num_updates 104820 | best_bleu 31.99
2023-06-14 03:06:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:06:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint372.pt (epoch 372 @ 104820 updates, score 31.35) (writing took 8.643682930618525 seconds)
2023-06-14 03:06:21 | INFO | fairseq_cli.train | end of epoch 372 (average epoch stats below)
2023-06-14 03:06:21 | INFO | train | epoch 372 | loss 3.026 | nll_loss 1.107 | glat_accu 0.584 | glat_context_p 0.43 | word_ins 2.906 | length 2.912 | ppl 8.14 | wps 115162 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 104820 | lr 9.76738e-05 | gnorm 0.589 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:06:21 | INFO | fairseq.trainer | begin training epoch 373
2023-06-14 03:07:03 | INFO | train_inner | epoch 373:     80 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.578, glat_context_p=0.43, word_ins=2.899, length=2.908, ppl=8.11, wps=96728.8, ups=1.6, wpb=60284.7, bsz=2137.5, num_updates=104900, lr=9.76365e-05, gnorm=0.586, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:07:49 | INFO | train_inner | epoch 373:    180 / 282 loss=3.028, nll_loss=1.108, glat_accu=0.581, glat_context_p=0.43, word_ins=2.907, length=2.93, ppl=8.15, wps=130334, ups=2.15, wpb=60696.4, bsz=2133.9, num_updates=105000, lr=9.759e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:08:35 | INFO | train_inner | epoch 373:    280 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.586, glat_context_p=0.43, word_ins=2.903, length=2.918, ppl=8.13, wps=133016, ups=2.2, wpb=60389.5, bsz=2190.5, num_updates=105100, lr=9.75436e-05, gnorm=0.608, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:08:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:08:39 | INFO | valid | epoch 373 | valid on 'valid' subset | loss 12.36 | nll_loss 11.192 | word_ins 12.121 | length 4.778 | ppl 5257.58 | bleu 31.64 | wps 88366.8 | wpb 21176.3 | bsz 666.3 | num_updates 105102 | best_bleu 31.99
2023-06-14 03:08:39 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:08:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint373.pt (epoch 373 @ 105102 updates, score 31.64) (writing took 11.174901586025953 seconds)
2023-06-14 03:08:50 | INFO | fairseq_cli.train | end of epoch 373 (average epoch stats below)
2023-06-14 03:08:50 | INFO | train | epoch 373 | loss 3.023 | nll_loss 1.104 | glat_accu 0.584 | glat_context_p 0.43 | word_ins 2.903 | length 2.917 | ppl 8.13 | wps 114745 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 105102 | lr 9.75426e-05 | gnorm 0.6 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:08:50 | INFO | fairseq.trainer | begin training epoch 374
2023-06-14 03:09:42 | INFO | train_inner | epoch 374:     98 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.578, glat_context_p=0.43, word_ins=2.901, length=2.921, ppl=8.12, wps=89764, ups=1.5, wpb=60040.9, bsz=2131.4, num_updates=105200, lr=9.74972e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:09:45 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:10:28 | INFO | train_inner | epoch 374:    199 / 282 loss=3.018, nll_loss=1.099, glat_accu=0.586, glat_context_p=0.43, word_ins=2.898, length=2.902, ppl=8.1, wps=131765, ups=2.17, wpb=60644.7, bsz=2192.3, num_updates=105300, lr=9.74509e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:11:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:11:09 | INFO | valid | epoch 374 | valid on 'valid' subset | loss 12.319 | nll_loss 11.141 | word_ins 12.08 | length 4.773 | ppl 5107.95 | bleu 31.69 | wps 88184.3 | wpb 21176.3 | bsz 666.3 | num_updates 105383 | best_bleu 31.99
2023-06-14 03:11:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:11:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint374.pt (epoch 374 @ 105383 updates, score 31.69) (writing took 11.587782721966505 seconds)
2023-06-14 03:11:20 | INFO | fairseq_cli.train | end of epoch 374 (average epoch stats below)
2023-06-14 03:11:20 | INFO | train | epoch 374 | loss 3.022 | nll_loss 1.103 | glat_accu 0.581 | glat_context_p 0.43 | word_ins 2.902 | length 2.917 | ppl 8.12 | wps 112884 | ups 1.87 | wpb 60414.6 | bsz 2155.3 | num_updates 105383 | lr 9.74125e-05 | gnorm 0.592 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:11:20 | INFO | fairseq.trainer | begin training epoch 375
2023-06-14 03:11:34 | INFO | train_inner | epoch 375:     17 / 282 loss=3.027, nll_loss=1.107, glat_accu=0.581, glat_context_p=0.43, word_ins=2.906, length=2.929, ppl=8.15, wps=90093.3, ups=1.5, wpb=60027.2, bsz=2128.6, num_updates=105400, lr=9.74047e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:12:20 | INFO | train_inner | epoch 375:    117 / 282 loss=3.027, nll_loss=1.109, glat_accu=0.584, glat_context_p=0.43, word_ins=2.907, length=2.904, ppl=8.15, wps=132003, ups=2.17, wpb=60868.5, bsz=2136.2, num_updates=105500, lr=9.73585e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:13:06 | INFO | train_inner | epoch 375:    217 / 282 loss=3.016, nll_loss=1.096, glat_accu=0.575, glat_context_p=0.43, word_ins=2.896, length=2.925, ppl=8.09, wps=131407, ups=2.17, wpb=60422.8, bsz=2164.5, num_updates=105600, lr=9.73124e-05, gnorm=0.575, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:13:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:13:39 | INFO | valid | epoch 375 | valid on 'valid' subset | loss 12.394 | nll_loss 11.229 | word_ins 12.16 | length 4.698 | ppl 5383.17 | bleu 31.43 | wps 88580.2 | wpb 21176.3 | bsz 666.3 | num_updates 105665 | best_bleu 31.99
2023-06-14 03:13:39 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:13:50 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint375.pt (epoch 375 @ 105665 updates, score 31.43) (writing took 11.270646259188652 seconds)
2023-06-14 03:13:50 | INFO | fairseq_cli.train | end of epoch 375 (average epoch stats below)
2023-06-14 03:13:50 | INFO | train | epoch 375 | loss 3.019 | nll_loss 1.1 | glat_accu 0.577 | glat_context_p 0.43 | word_ins 2.899 | length 2.916 | ppl 8.11 | wps 113631 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 105665 | lr 9.72824e-05 | gnorm 0.585 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:13:50 | INFO | fairseq.trainer | begin training epoch 376
2023-06-14 03:14:12 | INFO | train_inner | epoch 376:     35 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.564, glat_context_p=0.43, word_ins=2.893, length=2.926, ppl=8.07, wps=90856.1, ups=1.51, wpb=60012.4, bsz=2146.9, num_updates=105700, lr=9.72663e-05, gnorm=0.591, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:14:58 | INFO | train_inner | epoch 376:    135 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.578, glat_context_p=0.43, word_ins=2.896, length=2.919, ppl=8.09, wps=132864, ups=2.19, wpb=60642.2, bsz=2151.8, num_updates=105800, lr=9.72203e-05, gnorm=0.582, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:15:44 | INFO | train_inner | epoch 376:    235 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.585, glat_context_p=0.429, word_ins=2.903, length=2.911, ppl=8.13, wps=132028, ups=2.18, wpb=60497.4, bsz=2184.6, num_updates=105900, lr=9.71744e-05, gnorm=0.586, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:16:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:16:08 | INFO | valid | epoch 376 | valid on 'valid' subset | loss 12.332 | nll_loss 11.153 | word_ins 12.096 | length 4.74 | ppl 5156.76 | bleu 31.93 | wps 85130.2 | wpb 21176.3 | bsz 666.3 | num_updates 105947 | best_bleu 31.99
2023-06-14 03:16:08 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:16:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint376.pt (epoch 376 @ 105947 updates, score 31.93) (writing took 11.0084731541574 seconds)
2023-06-14 03:16:19 | INFO | fairseq_cli.train | end of epoch 376 (average epoch stats below)
2023-06-14 03:16:19 | INFO | train | epoch 376 | loss 3.021 | nll_loss 1.102 | glat_accu 0.58 | glat_context_p 0.429 | word_ins 2.901 | length 2.917 | ppl 8.12 | wps 114152 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 105947 | lr 9.71529e-05 | gnorm 0.588 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:16:20 | INFO | fairseq.trainer | begin training epoch 377
2023-06-14 03:16:50 | INFO | train_inner | epoch 377:     53 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.592, glat_context_p=0.429, word_ins=2.903, length=2.894, ppl=8.13, wps=91312.2, ups=1.52, wpb=60224.3, bsz=2170.8, num_updates=106000, lr=9.71286e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:17:36 | INFO | train_inner | epoch 377:    153 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.573, glat_context_p=0.429, word_ins=2.901, length=2.933, ppl=8.12, wps=131004, ups=2.16, wpb=60558.9, bsz=2106.4, num_updates=106100, lr=9.70828e-05, gnorm=0.582, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:18:22 | INFO | train_inner | epoch 377:    253 / 282 loss=3.018, nll_loss=1.099, glat_accu=0.582, glat_context_p=0.429, word_ins=2.898, length=2.916, ppl=8.1, wps=132996, ups=2.2, wpb=60445.6, bsz=2221, num_updates=106200, lr=9.70371e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:18:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:18:38 | INFO | valid | epoch 377 | valid on 'valid' subset | loss 12.424 | nll_loss 11.258 | word_ins 12.186 | length 4.762 | ppl 5494.01 | bleu 31.66 | wps 87629.5 | wpb 21176.3 | bsz 666.3 | num_updates 106229 | best_bleu 31.99
2023-06-14 03:18:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:18:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint377.pt (epoch 377 @ 106229 updates, score 31.66) (writing took 10.272884253412485 seconds)
2023-06-14 03:18:48 | INFO | fairseq_cli.train | end of epoch 377 (average epoch stats below)
2023-06-14 03:18:48 | INFO | train | epoch 377 | loss 3.02 | nll_loss 1.101 | glat_accu 0.581 | glat_context_p 0.429 | word_ins 2.9 | length 2.915 | ppl 8.11 | wps 114359 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 106229 | lr 9.70238e-05 | gnorm 0.587 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:18:48 | INFO | fairseq.trainer | begin training epoch 378
2023-06-14 03:18:56 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:19:28 | INFO | train_inner | epoch 378:     72 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.586, glat_context_p=0.429, word_ins=2.903, length=2.913, ppl=8.13, wps=90412.5, ups=1.5, wpb=60365.2, bsz=2136.7, num_updates=106300, lr=9.69914e-05, gnorm=0.591, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:20:14 | INFO | train_inner | epoch 378:    172 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.571, glat_context_p=0.429, word_ins=2.897, length=2.938, ppl=8.1, wps=131933, ups=2.18, wpb=60554.2, bsz=2124.6, num_updates=106400, lr=9.69458e-05, gnorm=0.583, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:21:00 | INFO | train_inner | epoch 378:    272 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.586, glat_context_p=0.429, word_ins=2.893, length=2.894, ppl=8.07, wps=130941, ups=2.17, wpb=60436.1, bsz=2194.4, num_updates=106500, lr=9.69003e-05, gnorm=0.589, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:21:05 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:21:09 | INFO | valid | epoch 378 | valid on 'valid' subset | loss 12.37 | nll_loss 11.194 | word_ins 12.126 | length 4.892 | ppl 5294.31 | bleu 31.38 | wps 88455.9 | wpb 21176.3 | bsz 666.3 | num_updates 106510 | best_bleu 31.99
2023-06-14 03:21:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:21:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint378.pt (epoch 378 @ 106510 updates, score 31.38) (writing took 12.656085029244423 seconds)
2023-06-14 03:21:21 | INFO | fairseq_cli.train | end of epoch 378 (average epoch stats below)
2023-06-14 03:21:21 | INFO | train | epoch 378 | loss 3.018 | nll_loss 1.099 | glat_accu 0.581 | glat_context_p 0.429 | word_ins 2.898 | length 2.915 | ppl 8.1 | wps 110904 | ups 1.84 | wpb 60414.5 | bsz 2155.7 | num_updates 106510 | lr 9.68958e-05 | gnorm 0.587 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:21:22 | INFO | fairseq.trainer | begin training epoch 379
2023-06-14 03:22:10 | INFO | train_inner | epoch 379:     90 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.58, glat_context_p=0.429, word_ins=2.899, length=2.908, ppl=8.1, wps=86584.6, ups=1.44, wpb=60054.9, bsz=2147, num_updates=106600, lr=9.68549e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:22:56 | INFO | train_inner | epoch 379:    190 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.579, glat_context_p=0.429, word_ins=2.895, length=2.916, ppl=8.09, wps=132027, ups=2.18, wpb=60664.1, bsz=2166.1, num_updates=106700, lr=9.68095e-05, gnorm=0.577, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:23:37 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:23:41 | INFO | valid | epoch 379 | valid on 'valid' subset | loss 12.35 | nll_loss 11.178 | word_ins 12.112 | length 4.743 | ppl 5219.32 | bleu 31.65 | wps 88168.7 | wpb 21176.3 | bsz 666.3 | num_updates 106792 | best_bleu 31.99
2023-06-14 03:23:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:23:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint379.pt (epoch 379 @ 106792 updates, score 31.65) (writing took 14.655964225530624 seconds)
2023-06-14 03:23:55 | INFO | fairseq_cli.train | end of epoch 379 (average epoch stats below)
2023-06-14 03:23:55 | INFO | train | epoch 379 | loss 3.018 | nll_loss 1.099 | glat_accu 0.581 | glat_context_p 0.429 | word_ins 2.898 | length 2.91 | ppl 8.1 | wps 110693 | ups 1.83 | wpb 60413.8 | bsz 2157.2 | num_updates 106792 | lr 9.67677e-05 | gnorm 0.584 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:23:56 | INFO | fairseq.trainer | begin training epoch 380
2023-06-14 03:24:05 | INFO | train_inner | epoch 380:      8 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.585, glat_context_p=0.429, word_ins=2.9, length=2.906, ppl=8.11, wps=86191.5, ups=1.44, wpb=60047.9, bsz=2140.3, num_updates=106800, lr=9.67641e-05, gnorm=0.591, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:24:51 | INFO | train_inner | epoch 380:    108 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.579, glat_context_p=0.429, word_ins=2.893, length=2.916, ppl=8.07, wps=132872, ups=2.19, wpb=60580.2, bsz=2177, num_updates=106900, lr=9.67189e-05, gnorm=0.585, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:25:37 | INFO | train_inner | epoch 380:    208 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.571, glat_context_p=0.429, word_ins=2.892, length=2.912, ppl=8.07, wps=131959, ups=2.18, wpb=60548.2, bsz=2163.6, num_updates=107000, lr=9.66736e-05, gnorm=0.564, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:26:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:26:15 | INFO | valid | epoch 380 | valid on 'valid' subset | loss 12.425 | nll_loss 11.259 | word_ins 12.188 | length 4.733 | ppl 5497.74 | bleu 31.51 | wps 88167.9 | wpb 21176.3 | bsz 666.3 | num_updates 107074 | best_bleu 31.99
2023-06-14 03:26:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:26:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint380.pt (epoch 380 @ 107074 updates, score 31.51) (writing took 11.184282213449478 seconds)
2023-06-14 03:26:26 | INFO | fairseq_cli.train | end of epoch 380 (average epoch stats below)
2023-06-14 03:26:26 | INFO | train | epoch 380 | loss 3.014 | nll_loss 1.095 | glat_accu 0.573 | glat_context_p 0.429 | word_ins 2.894 | length 2.916 | ppl 8.08 | wps 113192 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 107074 | lr 9.66402e-05 | gnorm 0.582 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:26:26 | INFO | fairseq.trainer | begin training epoch 381
2023-06-14 03:26:44 | INFO | train_inner | epoch 381:     26 / 282 loss=3.017, nll_loss=1.099, glat_accu=0.571, glat_context_p=0.429, word_ins=2.898, length=2.912, ppl=8.09, wps=89086.6, ups=1.48, wpb=60197.9, bsz=2134.6, num_updates=107100, lr=9.66285e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:27:31 | INFO | train_inner | epoch 381:    126 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.576, glat_context_p=0.429, word_ins=2.903, length=2.918, ppl=8.13, wps=130343, ups=2.15, wpb=60583.2, bsz=2148.7, num_updates=107200, lr=9.65834e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:27:56 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:28:17 | INFO | train_inner | epoch 381:    227 / 282 loss=3.023, nll_loss=1.105, glat_accu=0.583, glat_context_p=0.429, word_ins=2.903, length=2.893, ppl=8.13, wps=131968, ups=2.18, wpb=60524.7, bsz=2195.6, num_updates=107300, lr=9.65384e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:28:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:28:45 | INFO | valid | epoch 381 | valid on 'valid' subset | loss 12.376 | nll_loss 11.195 | word_ins 12.136 | length 4.799 | ppl 5315.94 | bleu 31.74 | wps 87883.8 | wpb 21176.3 | bsz 666.3 | num_updates 107355 | best_bleu 31.99
2023-06-14 03:28:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:28:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint381.pt (epoch 381 @ 107355 updates, score 31.74) (writing took 12.178001657128334 seconds)
2023-06-14 03:28:57 | INFO | fairseq_cli.train | end of epoch 381 (average epoch stats below)
2023-06-14 03:28:57 | INFO | train | epoch 381 | loss 3.025 | nll_loss 1.107 | glat_accu 0.58 | glat_context_p 0.429 | word_ins 2.905 | length 2.909 | ppl 8.14 | wps 112318 | ups 1.86 | wpb 60415.4 | bsz 2157.8 | num_updates 107355 | lr 9.65137e-05 | gnorm 0.593 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:28:57 | INFO | fairseq.trainer | begin training epoch 382
2023-06-14 03:29:24 | INFO | train_inner | epoch 382:     45 / 282 loss=3.033, nll_loss=1.114, glat_accu=0.582, glat_context_p=0.428, word_ins=2.912, length=2.93, ppl=8.18, wps=89926.4, ups=1.5, wpb=60123.5, bsz=2137.1, num_updates=107400, lr=9.64935e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:30:10 | INFO | train_inner | epoch 382:    145 / 282 loss=3.034, nll_loss=1.116, glat_accu=0.587, glat_context_p=0.428, word_ins=2.913, length=2.905, ppl=8.19, wps=131977, ups=2.17, wpb=60712.2, bsz=2157.1, num_updates=107500, lr=9.64486e-05, gnorm=0.597, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:30:55 | INFO | train_inner | epoch 382:    245 / 282 loss=3.036, nll_loss=1.118, glat_accu=0.588, glat_context_p=0.428, word_ins=2.915, length=2.925, ppl=8.2, wps=132248, ups=2.18, wpb=60571.2, bsz=2146.8, num_updates=107600, lr=9.64037e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:31:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:31:15 | INFO | valid | epoch 382 | valid on 'valid' subset | loss 12.337 | nll_loss 11.143 | word_ins 12.085 | length 5.027 | ppl 5174.31 | bleu 31.62 | wps 87959.2 | wpb 21176.3 | bsz 666.3 | num_updates 107637 | best_bleu 31.99
2023-06-14 03:31:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:31:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint382.pt (epoch 382 @ 107637 updates, score 31.62) (writing took 15.69981848821044 seconds)
2023-06-14 03:31:31 | INFO | fairseq_cli.train | end of epoch 382 (average epoch stats below)
2023-06-14 03:31:31 | INFO | train | epoch 382 | loss 3.034 | nll_loss 1.116 | glat_accu 0.589 | glat_context_p 0.428 | word_ins 2.913 | length 2.912 | ppl 8.19 | wps 110436 | ups 1.83 | wpb 60413.8 | bsz 2157.2 | num_updates 107637 | lr 9.63872e-05 | gnorm 0.602 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:31:31 | INFO | fairseq.trainer | begin training epoch 383
2023-06-14 03:32:06 | INFO | train_inner | epoch 383:     63 / 282 loss=3.036, nll_loss=1.119, glat_accu=0.594, glat_context_p=0.428, word_ins=2.915, length=2.897, ppl=8.2, wps=85398.2, ups=1.42, wpb=60166, bsz=2150.4, num_updates=107700, lr=9.6359e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:32:52 | INFO | train_inner | epoch 383:    163 / 282 loss=3.028, nll_loss=1.109, glat_accu=0.581, glat_context_p=0.428, word_ins=2.907, length=2.924, ppl=8.16, wps=131170, ups=2.17, wpb=60467.6, bsz=2133, num_updates=107800, lr=9.63143e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:33:38 | INFO | train_inner | epoch 383:    263 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.584, glat_context_p=0.428, word_ins=2.905, length=2.918, ppl=8.15, wps=132656, ups=2.19, wpb=60546.2, bsz=2193.2, num_updates=107900, lr=9.62696e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:33:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:33:49 | INFO | valid | epoch 383 | valid on 'valid' subset | loss 12.489 | nll_loss 11.332 | word_ins 12.25 | length 4.785 | ppl 5748.88 | bleu 31.43 | wps 87222.3 | wpb 21176.3 | bsz 666.3 | num_updates 107919 | best_bleu 31.99
2023-06-14 03:33:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:34:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint383.pt (epoch 383 @ 107919 updates, score 31.43) (writing took 15.908399868756533 seconds)
2023-06-14 03:34:05 | INFO | fairseq_cli.train | end of epoch 383 (average epoch stats below)
2023-06-14 03:34:05 | INFO | train | epoch 383 | loss 3.029 | nll_loss 1.11 | glat_accu 0.584 | glat_context_p 0.428 | word_ins 2.908 | length 2.914 | ppl 8.16 | wps 110678 | ups 1.83 | wpb 60413.8 | bsz 2157.2 | num_updates 107919 | lr 9.62611e-05 | gnorm 0.599 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:34:05 | INFO | fairseq.trainer | begin training epoch 384
2023-06-14 03:34:50 | INFO | train_inner | epoch 384:     81 / 282 loss=3.021, nll_loss=1.103, glat_accu=0.586, glat_context_p=0.428, word_ins=2.901, length=2.884, ppl=8.12, wps=82726.5, ups=1.38, wpb=60090.5, bsz=2188.2, num_updates=108000, lr=9.6225e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:35:36 | INFO | train_inner | epoch 384:    181 / 282 loss=3.03, nll_loss=1.112, glat_accu=0.579, glat_context_p=0.428, word_ins=2.909, length=2.93, ppl=8.17, wps=130921, ups=2.16, wpb=60488.4, bsz=2118.8, num_updates=108100, lr=9.61805e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:36:21 | INFO | train_inner | epoch 384:    281 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.59, glat_context_p=0.428, word_ins=2.904, length=2.913, ppl=8.14, wps=135045, ups=2.22, wpb=60725.9, bsz=2176, num_updates=108200, lr=9.61361e-05, gnorm=0.589, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:36:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:36:25 | INFO | valid | epoch 384 | valid on 'valid' subset | loss 12.378 | nll_loss 11.203 | word_ins 12.14 | length 4.758 | ppl 5321.36 | bleu 31.63 | wps 85147 | wpb 21176.3 | bsz 666.3 | num_updates 108201 | best_bleu 31.99
2023-06-14 03:36:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:36:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint384.pt (epoch 384 @ 108201 updates, score 31.63) (writing took 14.871465232223272 seconds)
2023-06-14 03:36:40 | INFO | fairseq_cli.train | end of epoch 384 (average epoch stats below)
2023-06-14 03:36:40 | INFO | train | epoch 384 | loss 3.026 | nll_loss 1.107 | glat_accu 0.585 | glat_context_p 0.428 | word_ins 2.905 | length 2.911 | ppl 8.15 | wps 110222 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 108201 | lr 9.61356e-05 | gnorm 0.597 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:36:40 | INFO | fairseq.trainer | begin training epoch 385
2023-06-14 03:37:22 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:37:32 | INFO | train_inner | epoch 385:    100 / 282 loss=3.022, nll_loss=1.104, glat_accu=0.583, glat_context_p=0.428, word_ins=2.902, length=2.901, ppl=8.13, wps=85351.2, ups=1.42, wpb=60128.5, bsz=2177.1, num_updates=108300, lr=9.60917e-05, gnorm=0.604, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:38:18 | INFO | train_inner | epoch 385:    200 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.587, glat_context_p=0.428, word_ins=2.907, length=2.919, ppl=8.16, wps=132109, ups=2.18, wpb=60556.8, bsz=2153.1, num_updates=108400, lr=9.60473e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:38:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:38:58 | INFO | valid | epoch 385 | valid on 'valid' subset | loss 12.451 | nll_loss 11.277 | word_ins 12.204 | length 4.949 | ppl 5599.51 | bleu 31.34 | wps 87951.5 | wpb 21176.3 | bsz 666.3 | num_updates 108482 | best_bleu 31.99
2023-06-14 03:38:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:39:10 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint385.pt (epoch 385 @ 108482 updates, score 31.34) (writing took 12.295309506356716 seconds)
2023-06-14 03:39:10 | INFO | fairseq_cli.train | end of epoch 385 (average epoch stats below)
2023-06-14 03:39:10 | INFO | train | epoch 385 | loss 3.028 | nll_loss 1.109 | glat_accu 0.585 | glat_context_p 0.428 | word_ins 2.907 | length 2.914 | ppl 8.16 | wps 112697 | ups 1.87 | wpb 60414.2 | bsz 2157.1 | num_updates 108482 | lr 9.6011e-05 | gnorm 0.6 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:39:11 | INFO | fairseq.trainer | begin training epoch 386
2023-06-14 03:39:26 | INFO | train_inner | epoch 386:     18 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.583, glat_context_p=0.428, word_ins=2.91, length=2.921, ppl=8.18, wps=88036.3, ups=1.47, wpb=60028.4, bsz=2134.4, num_updates=108500, lr=9.60031e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:40:12 | INFO | train_inner | epoch 386:    118 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.585, glat_context_p=0.428, word_ins=2.908, length=2.921, ppl=8.16, wps=132248, ups=2.18, wpb=60574.9, bsz=2190.3, num_updates=108600, lr=9.59589e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:40:58 | INFO | train_inner | epoch 386:    218 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.582, glat_context_p=0.428, word_ins=2.911, length=2.917, ppl=8.18, wps=131895, ups=2.17, wpb=60786.6, bsz=2113.4, num_updates=108700, lr=9.59147e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:41:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:41:30 | INFO | valid | epoch 386 | valid on 'valid' subset | loss 12.331 | nll_loss 11.166 | word_ins 12.099 | length 4.652 | ppl 5150.63 | bleu 31.82 | wps 86744.5 | wpb 21176.3 | bsz 666.3 | num_updates 108764 | best_bleu 31.99
2023-06-14 03:41:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:41:42 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint386.pt (epoch 386 @ 108764 updates, score 31.82) (writing took 12.026164695620537 seconds)
2023-06-14 03:41:42 | INFO | fairseq_cli.train | end of epoch 386 (average epoch stats below)
2023-06-14 03:41:42 | INFO | train | epoch 386 | loss 3.028 | nll_loss 1.109 | glat_accu 0.583 | glat_context_p 0.428 | word_ins 2.907 | length 2.913 | ppl 8.16 | wps 112339 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 108764 | lr 9.58865e-05 | gnorm 0.606 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:41:42 | INFO | fairseq.trainer | begin training epoch 387
2023-06-14 03:42:06 | INFO | train_inner | epoch 387:     36 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.585, glat_context_p=0.428, word_ins=2.904, length=2.909, ppl=8.14, wps=87849.4, ups=1.47, wpb=59946.9, bsz=2144.9, num_updates=108800, lr=9.58706e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:42:52 | INFO | train_inner | epoch 387:    136 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.592, glat_context_p=0.427, word_ins=2.903, length=2.9, ppl=8.13, wps=131847, ups=2.18, wpb=60583.2, bsz=2164.9, num_updates=108900, lr=9.58266e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:43:38 | INFO | train_inner | epoch 387:    236 / 282 loss=3.032, nll_loss=1.113, glat_accu=0.588, glat_context_p=0.427, word_ins=2.911, length=2.922, ppl=8.18, wps=132970, ups=2.19, wpb=60634.5, bsz=2172.2, num_updates=109000, lr=9.57826e-05, gnorm=0.599, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:43:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:44:03 | INFO | valid | epoch 387 | valid on 'valid' subset | loss 12.506 | nll_loss 11.348 | word_ins 12.269 | length 4.74 | ppl 5817.63 | bleu 31.6 | wps 88217.9 | wpb 21176.3 | bsz 666.3 | num_updates 109046 | best_bleu 31.99
2023-06-14 03:44:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:44:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint387.pt (epoch 387 @ 109046 updates, score 31.6) (writing took 12.283853601664305 seconds)
2023-06-14 03:44:15 | INFO | fairseq_cli.train | end of epoch 387 (average epoch stats below)
2023-06-14 03:44:15 | INFO | train | epoch 387 | loss 3.028 | nll_loss 1.109 | glat_accu 0.587 | glat_context_p 0.427 | word_ins 2.907 | length 2.912 | ppl 8.16 | wps 111544 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 109046 | lr 9.57624e-05 | gnorm 0.602 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 03:44:15 | INFO | fairseq.trainer | begin training epoch 388
2023-06-14 03:44:46 | INFO | train_inner | epoch 388:     54 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.578, glat_context_p=0.427, word_ins=2.905, length=2.919, ppl=8.14, wps=87152.2, ups=1.45, wpb=59989.9, bsz=2120.9, num_updates=109100, lr=9.57387e-05, gnorm=0.609, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:45:31 | INFO | train_inner | epoch 388:    154 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.585, glat_context_p=0.427, word_ins=2.903, length=2.909, ppl=8.13, wps=134547, ups=2.22, wpb=60594.9, bsz=2194.8, num_updates=109200, lr=9.56949e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:46:18 | INFO | train_inner | epoch 388:    254 / 282 loss=3.028, nll_loss=1.11, glat_accu=0.586, glat_context_p=0.427, word_ins=2.908, length=2.896, ppl=8.16, wps=130831, ups=2.16, wpb=60668.4, bsz=2191.9, num_updates=109300, lr=9.56511e-05, gnorm=0.592, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:46:19 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:46:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:46:34 | INFO | valid | epoch 388 | valid on 'valid' subset | loss 12.369 | nll_loss 11.199 | word_ins 12.132 | length 4.746 | ppl 5291.62 | bleu 31.64 | wps 88461.3 | wpb 21176.3 | bsz 666.3 | num_updates 109327 | best_bleu 31.99
2023-06-14 03:46:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:46:46 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint388.pt (epoch 388 @ 109327 updates, score 31.64) (writing took 11.744566980749369 seconds)
2023-06-14 03:46:46 | INFO | fairseq_cli.train | end of epoch 388 (average epoch stats below)
2023-06-14 03:46:46 | INFO | train | epoch 388 | loss 3.026 | nll_loss 1.107 | glat_accu 0.583 | glat_context_p 0.427 | word_ins 2.905 | length 2.91 | ppl 8.14 | wps 112316 | ups 1.86 | wpb 60418 | bsz 2157.1 | num_updates 109327 | lr 9.56393e-05 | gnorm 0.602 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:46:46 | INFO | fairseq.trainer | begin training epoch 389
2023-06-14 03:47:26 | INFO | train_inner | epoch 389:     73 / 282 loss=3.024, nll_loss=1.106, glat_accu=0.584, glat_context_p=0.427, word_ins=2.904, length=2.896, ppl=8.14, wps=88983.1, ups=1.48, wpb=60285.6, bsz=2115.8, num_updates=109400, lr=9.56074e-05, gnorm=0.6, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:48:11 | INFO | train_inner | epoch 389:    173 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.591, glat_context_p=0.427, word_ins=2.901, length=2.909, ppl=8.13, wps=134105, ups=2.21, wpb=60652.7, bsz=2215.8, num_updates=109500, lr=9.55637e-05, gnorm=0.591, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:48:57 | INFO | train_inner | epoch 389:    273 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.571, glat_context_p=0.427, word_ins=2.902, length=2.924, ppl=8.13, wps=131608, ups=2.18, wpb=60424.1, bsz=2116.9, num_updates=109600, lr=9.55201e-05, gnorm=0.584, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:49:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:49:04 | INFO | valid | epoch 389 | valid on 'valid' subset | loss 12.391 | nll_loss 11.227 | word_ins 12.154 | length 4.736 | ppl 5372.21 | bleu 31.79 | wps 87147.6 | wpb 21176.3 | bsz 666.3 | num_updates 109609 | best_bleu 31.99
2023-06-14 03:49:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:49:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint389.pt (epoch 389 @ 109609 updates, score 31.79) (writing took 11.752502374351025 seconds)
2023-06-14 03:49:16 | INFO | fairseq_cli.train | end of epoch 389 (average epoch stats below)
2023-06-14 03:49:16 | INFO | train | epoch 389 | loss 3.023 | nll_loss 1.104 | glat_accu 0.583 | glat_context_p 0.427 | word_ins 2.902 | length 2.91 | ppl 8.13 | wps 113862 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 109609 | lr 9.55162e-05 | gnorm 0.592 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:49:16 | INFO | fairseq.trainer | begin training epoch 390
2023-06-14 03:50:03 | INFO | train_inner | epoch 390:     91 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.587, glat_context_p=0.427, word_ins=2.903, length=2.91, ppl=8.13, wps=90359, ups=1.5, wpb=60126.2, bsz=2145.9, num_updates=109700, lr=9.54765e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:50:49 | INFO | train_inner | epoch 390:    191 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.582, glat_context_p=0.427, word_ins=2.9, length=2.912, ppl=8.11, wps=131749, ups=2.18, wpb=60549.4, bsz=2168.2, num_updates=109800, lr=9.54331e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:51:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:51:34 | INFO | valid | epoch 390 | valid on 'valid' subset | loss 12.495 | nll_loss 11.334 | word_ins 12.255 | length 4.783 | ppl 5773.34 | bleu 31.56 | wps 87311.9 | wpb 21176.3 | bsz 666.3 | num_updates 109891 | best_bleu 31.99
2023-06-14 03:51:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:51:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint390.pt (epoch 390 @ 109891 updates, score 31.56) (writing took 11.398506581783295 seconds)
2023-06-14 03:51:45 | INFO | fairseq_cli.train | end of epoch 390 (average epoch stats below)
2023-06-14 03:51:45 | INFO | train | epoch 390 | loss 3.024 | nll_loss 1.105 | glat_accu 0.584 | glat_context_p 0.427 | word_ins 2.903 | length 2.91 | ppl 8.13 | wps 113887 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 109891 | lr 9.53935e-05 | gnorm 0.594 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:51:45 | INFO | fairseq.trainer | begin training epoch 391
2023-06-14 03:51:56 | INFO | train_inner | epoch 391:      9 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.584, glat_context_p=0.427, word_ins=2.905, length=2.909, ppl=8.14, wps=89583.6, ups=1.49, wpb=60045.9, bsz=2134.6, num_updates=109900, lr=9.53896e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:52:42 | INFO | train_inner | epoch 391:    109 / 282 loss=3.021, nll_loss=1.103, glat_accu=0.575, glat_context_p=0.427, word_ins=2.901, length=2.914, ppl=8.12, wps=131154, ups=2.17, wpb=60522.7, bsz=2158.6, num_updates=110000, lr=9.53463e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:53:28 | INFO | train_inner | epoch 391:    209 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.582, glat_context_p=0.427, word_ins=2.901, length=2.91, ppl=8.12, wps=133949, ups=2.21, wpb=60740.9, bsz=2152.2, num_updates=110100, lr=9.53029e-05, gnorm=0.59, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:54:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:54:04 | INFO | valid | epoch 391 | valid on 'valid' subset | loss 12.418 | nll_loss 11.254 | word_ins 12.18 | length 4.74 | ppl 5470.73 | bleu 31.52 | wps 87812.5 | wpb 21176.3 | bsz 666.3 | num_updates 110173 | best_bleu 31.99
2023-06-14 03:54:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:54:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint391.pt (epoch 391 @ 110173 updates, score 31.52) (writing took 10.41749395057559 seconds)
2023-06-14 03:54:15 | INFO | fairseq_cli.train | end of epoch 391 (average epoch stats below)
2023-06-14 03:54:15 | INFO | train | epoch 391 | loss 3.022 | nll_loss 1.103 | glat_accu 0.582 | glat_context_p 0.427 | word_ins 2.901 | length 2.91 | ppl 8.12 | wps 114056 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 110173 | lr 9.52714e-05 | gnorm 0.591 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:54:15 | INFO | fairseq.trainer | begin training epoch 392
2023-06-14 03:54:33 | INFO | train_inner | epoch 392:     27 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.59, glat_context_p=0.427, word_ins=2.902, length=2.917, ppl=8.13, wps=91481.6, ups=1.53, wpb=59958.4, bsz=2156.2, num_updates=110200, lr=9.52597e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:55:19 | INFO | train_inner | epoch 392:    127 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.585, glat_context_p=0.427, word_ins=2.905, length=2.9, ppl=8.14, wps=132261, ups=2.18, wpb=60687.9, bsz=2144.5, num_updates=110300, lr=9.52165e-05, gnorm=0.605, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:55:31 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 03:56:05 | INFO | train_inner | epoch 392:    228 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.581, glat_context_p=0.426, word_ins=2.9, length=2.911, ppl=8.12, wps=131371, ups=2.17, wpb=60589.9, bsz=2171.1, num_updates=110400, lr=9.51734e-05, gnorm=0.604, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:56:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:56:33 | INFO | valid | epoch 392 | valid on 'valid' subset | loss 12.411 | nll_loss 11.236 | word_ins 12.169 | length 4.852 | ppl 5447.55 | bleu 31.54 | wps 88065.9 | wpb 21176.3 | bsz 666.3 | num_updates 110454 | best_bleu 31.99
2023-06-14 03:56:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:56:43 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint392.pt (epoch 392 @ 110454 updates, score 31.54) (writing took 10.167370337992907 seconds)
2023-06-14 03:56:43 | INFO | fairseq_cli.train | end of epoch 392 (average epoch stats below)
2023-06-14 03:56:43 | INFO | train | epoch 392 | loss 3.023 | nll_loss 1.105 | glat_accu 0.584 | glat_context_p 0.426 | word_ins 2.903 | length 2.909 | ppl 8.13 | wps 114239 | ups 1.89 | wpb 60407.6 | bsz 2157.2 | num_updates 110454 | lr 9.51501e-05 | gnorm 0.612 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:56:43 | INFO | fairseq.trainer | begin training epoch 393
2023-06-14 03:57:11 | INFO | train_inner | epoch 393:     46 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.591, glat_context_p=0.426, word_ins=2.902, length=2.897, ppl=8.13, wps=92057.2, ups=1.53, wpb=60100.6, bsz=2214, num_updates=110500, lr=9.51303e-05, gnorm=0.609, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:57:56 | INFO | train_inner | epoch 393:    146 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.59, glat_context_p=0.426, word_ins=2.903, length=2.91, ppl=8.14, wps=133345, ups=2.2, wpb=60551.9, bsz=2161.4, num_updates=110600, lr=9.50873e-05, gnorm=0.592, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 03:58:42 | INFO | train_inner | epoch 393:    246 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.58, glat_context_p=0.426, word_ins=2.905, length=2.918, ppl=8.15, wps=132332, ups=2.18, wpb=60584.8, bsz=2141.1, num_updates=110700, lr=9.50443e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 03:58:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 03:59:01 | INFO | valid | epoch 393 | valid on 'valid' subset | loss 12.278 | nll_loss 11.09 | word_ins 12.036 | length 4.838 | ppl 4965.11 | bleu 31.77 | wps 87044.3 | wpb 21176.3 | bsz 666.3 | num_updates 110736 | best_bleu 31.99
2023-06-14 03:59:01 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 03:59:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint393.pt (epoch 393 @ 110736 updates, score 31.77) (writing took 14.148549783974886 seconds)
2023-06-14 03:59:16 | INFO | fairseq_cli.train | end of epoch 393 (average epoch stats below)
2023-06-14 03:59:16 | INFO | train | epoch 393 | loss 3.025 | nll_loss 1.106 | glat_accu 0.586 | glat_context_p 0.426 | word_ins 2.904 | length 2.91 | ppl 8.14 | wps 111801 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 110736 | lr 9.50289e-05 | gnorm 0.602 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 03:59:16 | INFO | fairseq.trainer | begin training epoch 394
2023-06-14 03:59:51 | INFO | train_inner | epoch 394:     64 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.589, glat_context_p=0.426, word_ins=2.91, length=2.904, ppl=8.18, wps=86634.1, ups=1.44, wpb=60046.1, bsz=2140.3, num_updates=110800, lr=9.50014e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:00:37 | INFO | train_inner | epoch 394:    164 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.594, glat_context_p=0.426, word_ins=2.906, length=2.9, ppl=8.15, wps=132563, ups=2.19, wpb=60591.5, bsz=2175.7, num_updates=110900, lr=9.49586e-05, gnorm=0.597, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:01:23 | INFO | train_inner | epoch 394:    264 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.583, glat_context_p=0.426, word_ins=2.903, length=2.917, ppl=8.13, wps=131641, ups=2.17, wpb=60614.3, bsz=2150.6, num_updates=111000, lr=9.49158e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:01:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:01:35 | INFO | valid | epoch 394 | valid on 'valid' subset | loss 12.309 | nll_loss 11.128 | word_ins 12.068 | length 4.815 | ppl 5075.09 | bleu 31.73 | wps 87959 | wpb 21176.3 | bsz 666.3 | num_updates 111018 | best_bleu 31.99
2023-06-14 04:01:35 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:01:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint394.pt (epoch 394 @ 111018 updates, score 31.73) (writing took 14.19779548048973 seconds)
2023-06-14 04:01:49 | INFO | fairseq_cli.train | end of epoch 394 (average epoch stats below)
2023-06-14 04:01:49 | INFO | train | epoch 394 | loss 3.027 | nll_loss 1.109 | glat_accu 0.589 | glat_context_p 0.426 | word_ins 2.906 | length 2.907 | ppl 8.15 | wps 111238 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 111018 | lr 9.49081e-05 | gnorm 0.607 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:01:49 | INFO | fairseq.trainer | begin training epoch 395
2023-06-14 04:02:32 | INFO | train_inner | epoch 395:     82 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.588, glat_context_p=0.426, word_ins=2.9, length=2.893, ppl=8.11, wps=86322.1, ups=1.44, wpb=60013.8, bsz=2142.5, num_updates=111100, lr=9.48731e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:03:18 | INFO | train_inner | epoch 395:    182 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.583, glat_context_p=0.426, word_ins=2.904, length=2.909, ppl=8.14, wps=131656, ups=2.17, wpb=60678.9, bsz=2130.3, num_updates=111200, lr=9.48304e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:04:04 | INFO | train_inner | epoch 395:    282 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.579, glat_context_p=0.426, word_ins=2.902, length=2.918, ppl=8.13, wps=132266, ups=2.2, wpb=60070.7, bsz=2159.4, num_updates=111300, lr=9.47878e-05, gnorm=0.592, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:04:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:04:07 | INFO | valid | epoch 395 | valid on 'valid' subset | loss 12.498 | nll_loss 11.345 | word_ins 12.266 | length 4.626 | ppl 5785.11 | bleu 31.27 | wps 87791.4 | wpb 21176.3 | bsz 666.3 | num_updates 111300 | best_bleu 31.99
2023-06-14 04:04:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:04:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint395.pt (epoch 395 @ 111300 updates, score 31.27) (writing took 12.90304370969534 seconds)
2023-06-14 04:04:20 | INFO | fairseq_cli.train | end of epoch 395 (average epoch stats below)
2023-06-14 04:04:20 | INFO | train | epoch 395 | loss 3.022 | nll_loss 1.103 | glat_accu 0.584 | glat_context_p 0.426 | word_ins 2.901 | length 2.906 | ppl 8.12 | wps 112452 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 111300 | lr 9.47878e-05 | gnorm 0.599 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:04:20 | INFO | fairseq.trainer | begin training epoch 396
2023-06-14 04:04:50 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 04:05:13 | INFO | train_inner | epoch 396:    101 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.575, glat_context_p=0.426, word_ins=2.894, length=2.901, ppl=8.08, wps=87232.3, ups=1.44, wpb=60721.4, bsz=2179.2, num_updates=111400, lr=9.47452e-05, gnorm=0.588, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:05:59 | INFO | train_inner | epoch 396:    201 / 282 loss=3.018, nll_loss=1.099, glat_accu=0.575, glat_context_p=0.426, word_ins=2.898, length=2.91, ppl=8.1, wps=131763, ups=2.17, wpb=60593.4, bsz=2143.8, num_updates=111500, lr=9.47027e-05, gnorm=0.596, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:06:37 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:06:40 | INFO | valid | epoch 396 | valid on 'valid' subset | loss 12.323 | nll_loss 11.144 | word_ins 12.084 | length 4.786 | ppl 5125.33 | bleu 31.93 | wps 86662.4 | wpb 21176.3 | bsz 666.3 | num_updates 111581 | best_bleu 31.99
2023-06-14 04:06:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:06:52 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint396.pt (epoch 396 @ 111581 updates, score 31.93) (writing took 12.190707948058844 seconds)
2023-06-14 04:06:52 | INFO | fairseq_cli.train | end of epoch 396 (average epoch stats below)
2023-06-14 04:06:52 | INFO | train | epoch 396 | loss 3.019 | nll_loss 1.101 | glat_accu 0.581 | glat_context_p 0.426 | word_ins 2.899 | length 2.904 | ppl 8.11 | wps 111826 | ups 1.85 | wpb 60413.7 | bsz 2157.7 | num_updates 111581 | lr 9.46684e-05 | gnorm 0.597 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:06:52 | INFO | fairseq.trainer | begin training epoch 397
2023-06-14 04:07:06 | INFO | train_inner | epoch 397:     19 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.594, glat_context_p=0.426, word_ins=2.906, length=2.903, ppl=8.15, wps=89481.2, ups=1.49, wpb=59999.2, bsz=2151.8, num_updates=111600, lr=9.46603e-05, gnorm=0.608, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:07:17 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 04:07:53 | INFO | train_inner | epoch 397:    120 / 282 loss=3.022, nll_loss=1.104, glat_accu=0.589, glat_context_p=0.426, word_ins=2.903, length=2.886, ppl=8.13, wps=130540, ups=2.16, wpb=60498.4, bsz=2174.5, num_updates=111700, lr=9.46179e-05, gnorm=0.61, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 04:08:39 | INFO | train_inner | epoch 397:    220 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.59, glat_context_p=0.426, word_ins=2.905, length=2.915, ppl=8.15, wps=131337, ups=2.17, wpb=60609.7, bsz=2156.3, num_updates=111800, lr=9.45756e-05, gnorm=0.604, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 04:09:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:09:10 | INFO | valid | epoch 397 | valid on 'valid' subset | loss 12.385 | nll_loss 11.213 | word_ins 12.147 | length 4.769 | ppl 5348.54 | bleu 31.99 | wps 88154.5 | wpb 21176.3 | bsz 666.3 | num_updates 111862 | best_bleu 31.99
2023-06-14 04:09:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:09:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint397.pt (epoch 397 @ 111862 updates, score 31.99) (writing took 16.15947538241744 seconds)
2023-06-14 04:09:26 | INFO | fairseq_cli.train | end of epoch 397 (average epoch stats below)
2023-06-14 04:09:26 | INFO | train | epoch 397 | loss 3.025 | nll_loss 1.106 | glat_accu 0.589 | glat_context_p 0.426 | word_ins 2.904 | length 2.904 | ppl 8.14 | wps 110162 | ups 1.82 | wpb 60412 | bsz 2156.5 | num_updates 111862 | lr 9.45494e-05 | gnorm 0.608 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 04:09:26 | INFO | fairseq.trainer | begin training epoch 398
2023-06-14 04:09:50 | INFO | train_inner | epoch 398:     38 / 282 loss=3.027, nll_loss=1.108, glat_accu=0.589, glat_context_p=0.425, word_ins=2.906, length=2.909, ppl=8.15, wps=85391.7, ups=1.42, wpb=60234.6, bsz=2136.6, num_updates=111900, lr=9.45333e-05, gnorm=0.613, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 04:10:35 | INFO | train_inner | epoch 398:    138 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.593, glat_context_p=0.425, word_ins=2.905, length=2.89, ppl=8.14, wps=132444, ups=2.18, wpb=60635.6, bsz=2175, num_updates=112000, lr=9.44911e-05, gnorm=0.593, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 04:11:21 | INFO | train_inner | epoch 398:    238 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.591, glat_context_p=0.425, word_ins=2.904, length=2.906, ppl=8.14, wps=132688, ups=2.19, wpb=60574, bsz=2181, num_updates=112100, lr=9.4449e-05, gnorm=0.6, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 04:11:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:11:44 | INFO | valid | epoch 398 | valid on 'valid' subset | loss 12.36 | nll_loss 11.189 | word_ins 12.124 | length 4.745 | ppl 5258.47 | bleu 32 | wps 88516.6 | wpb 21176.3 | bsz 666.3 | num_updates 112144 | best_bleu 32
2023-06-14 04:11:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:11:59 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint398.pt (epoch 398 @ 112144 updates, score 32.0) (writing took 15.242252763360739 seconds)
2023-06-14 04:11:59 | INFO | fairseq_cli.train | end of epoch 398 (average epoch stats below)
2023-06-14 04:11:59 | INFO | train | epoch 398 | loss 3.027 | nll_loss 1.108 | glat_accu 0.591 | glat_context_p 0.425 | word_ins 2.906 | length 2.905 | ppl 8.15 | wps 111442 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 112144 | lr 9.44304e-05 | gnorm 0.604 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 04:11:59 | INFO | fairseq.trainer | begin training epoch 399
2023-06-14 04:12:31 | INFO | train_inner | epoch 399:     56 / 282 loss=3.029, nll_loss=1.109, glat_accu=0.589, glat_context_p=0.425, word_ins=2.907, length=2.92, ppl=8.16, wps=85130.4, ups=1.42, wpb=59874.4, bsz=2113.8, num_updates=112200, lr=9.44069e-05, gnorm=0.61, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 04:13:17 | INFO | train_inner | epoch 399:    156 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.588, glat_context_p=0.425, word_ins=2.899, length=2.897, ppl=8.11, wps=132869, ups=2.19, wpb=60587.3, bsz=2192, num_updates=112300, lr=9.43648e-05, gnorm=0.608, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 04:14:03 | INFO | train_inner | epoch 399:    256 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.584, glat_context_p=0.425, word_ins=2.904, length=2.908, ppl=8.14, wps=132702, ups=2.19, wpb=60614.8, bsz=2144.9, num_updates=112400, lr=9.43228e-05, gnorm=0.599, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 04:14:14 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:14:17 | INFO | valid | epoch 399 | valid on 'valid' subset | loss 12.381 | nll_loss 11.212 | word_ins 12.142 | length 4.792 | ppl 5334.03 | bleu 31.63 | wps 87755 | wpb 21176.3 | bsz 666.3 | num_updates 112426 | best_bleu 32
2023-06-14 04:14:17 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:14:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint399.pt (epoch 399 @ 112426 updates, score 31.63) (writing took 7.841191444545984 seconds)
2023-06-14 04:14:25 | INFO | fairseq_cli.train | end of epoch 399 (average epoch stats below)
2023-06-14 04:14:25 | INFO | train | epoch 399 | loss 3.023 | nll_loss 1.105 | glat_accu 0.586 | glat_context_p 0.425 | word_ins 2.903 | length 2.903 | ppl 8.13 | wps 116420 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 112426 | lr 9.43119e-05 | gnorm 0.601 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 04:14:26 | INFO | fairseq.trainer | begin training epoch 400
2023-06-14 04:15:05 | INFO | train_inner | epoch 400:     74 / 282 loss=3.021, nll_loss=1.103, glat_accu=0.588, glat_context_p=0.425, word_ins=2.901, length=2.899, ppl=8.12, wps=96955.6, ups=1.61, wpb=60331.6, bsz=2181.6, num_updates=112500, lr=9.42809e-05, gnorm=0.605, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 04:15:51 | INFO | train_inner | epoch 400:    174 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.582, glat_context_p=0.425, word_ins=2.904, length=2.921, ppl=8.14, wps=130941, ups=2.17, wpb=60441.2, bsz=2121.5, num_updates=112600, lr=9.4239e-05, gnorm=0.592, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 04:16:37 | INFO | train_inner | epoch 400:    274 / 282 loss=3.017, nll_loss=1.098, glat_accu=0.579, glat_context_p=0.425, word_ins=2.897, length=2.905, ppl=8.09, wps=131957, ups=2.18, wpb=60534.4, bsz=2159.4, num_updates=112700, lr=9.41972e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:16:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:16:44 | INFO | valid | epoch 400 | valid on 'valid' subset | loss 12.446 | nll_loss 11.289 | word_ins 12.212 | length 4.702 | ppl 5580.52 | bleu 31.57 | wps 87369.7 | wpb 21176.3 | bsz 666.3 | num_updates 112708 | best_bleu 32
2023-06-14 04:16:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:16:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint400.pt (epoch 400 @ 112708 updates, score 31.57) (writing took 12.639789961278439 seconds)
2023-06-14 04:16:56 | INFO | fairseq_cli.train | end of epoch 400 (average epoch stats below)
2023-06-14 04:16:56 | INFO | train | epoch 400 | loss 3.02 | nll_loss 1.101 | glat_accu 0.583 | glat_context_p 0.425 | word_ins 2.9 | length 2.907 | ppl 8.11 | wps 112899 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 112708 | lr 9.41939e-05 | gnorm 0.601 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:16:56 | INFO | fairseq.trainer | begin training epoch 401
2023-06-14 04:17:45 | INFO | train_inner | epoch 401:     92 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.588, glat_context_p=0.425, word_ins=2.898, length=2.908, ppl=8.11, wps=88256, ups=1.47, wpb=59995.2, bsz=2138.6, num_updates=112800, lr=9.41554e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:18:31 | INFO | train_inner | epoch 401:    192 / 282 loss=3.023, nll_loss=1.105, glat_accu=0.591, glat_context_p=0.425, word_ins=2.903, length=2.889, ppl=8.13, wps=132897, ups=2.19, wpb=60779.4, bsz=2178.1, num_updates=112900, lr=9.41137e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:19:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:19:15 | INFO | valid | epoch 401 | valid on 'valid' subset | loss 12.43 | nll_loss 11.262 | word_ins 12.195 | length 4.698 | ppl 5517.38 | bleu 31.73 | wps 83785 | wpb 21176.3 | bsz 666.3 | num_updates 112990 | best_bleu 32
2023-06-14 04:19:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:19:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint401.pt (epoch 401 @ 112990 updates, score 31.73) (writing took 9.894113563001156 seconds)
2023-06-14 04:19:25 | INFO | fairseq_cli.train | end of epoch 401 (average epoch stats below)
2023-06-14 04:19:25 | INFO | train | epoch 401 | loss 3.022 | nll_loss 1.103 | glat_accu 0.59 | glat_context_p 0.425 | word_ins 2.902 | length 2.903 | ppl 8.12 | wps 114420 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 112990 | lr 9.40762e-05 | gnorm 0.6 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:19:25 | INFO | fairseq.trainer | begin training epoch 402
2023-06-14 04:19:35 | INFO | train_inner | epoch 402:     10 / 282 loss=3.025, nll_loss=1.105, glat_accu=0.587, glat_context_p=0.425, word_ins=2.904, length=2.917, ppl=8.14, wps=93452.2, ups=1.56, wpb=60080.3, bsz=2127.2, num_updates=113000, lr=9.40721e-05, gnorm=0.608, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:20:20 | INFO | train_inner | epoch 402:    110 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.588, glat_context_p=0.425, word_ins=2.901, length=2.909, ppl=8.12, wps=132787, ups=2.19, wpb=60618, bsz=2154, num_updates=113100, lr=9.40305e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:21:06 | INFO | train_inner | epoch 402:    210 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.59, glat_context_p=0.425, word_ins=2.902, length=2.897, ppl=8.12, wps=132096, ups=2.18, wpb=60565.7, bsz=2138.6, num_updates=113200, lr=9.39889e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:21:39 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:21:43 | INFO | valid | epoch 402 | valid on 'valid' subset | loss 12.303 | nll_loss 11.119 | word_ins 12.062 | length 4.806 | ppl 5051.87 | bleu 32.07 | wps 87769 | wpb 21176.3 | bsz 666.3 | num_updates 113272 | best_bleu 32.07
2023-06-14 04:21:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:22:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint402.pt (epoch 402 @ 113272 updates, score 32.07) (writing took 17.147191021591425 seconds)
2023-06-14 04:22:00 | INFO | fairseq_cli.train | end of epoch 402 (average epoch stats below)
2023-06-14 04:22:00 | INFO | train | epoch 402 | loss 3.02 | nll_loss 1.101 | glat_accu 0.587 | glat_context_p 0.425 | word_ins 2.899 | length 2.904 | ppl 8.11 | wps 110220 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 113272 | lr 9.39591e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:22:00 | INFO | fairseq.trainer | begin training epoch 403
2023-06-14 04:22:19 | INFO | train_inner | epoch 403:     28 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.581, glat_context_p=0.425, word_ins=2.892, length=2.909, ppl=8.07, wps=82685.7, ups=1.38, wpb=59921.6, bsz=2175.4, num_updates=113300, lr=9.39475e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:23:05 | INFO | train_inner | epoch 403:    128 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.582, glat_context_p=0.424, word_ins=2.896, length=2.904, ppl=8.09, wps=130817, ups=2.16, wpb=60484.8, bsz=2142.8, num_updates=113400, lr=9.3906e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:23:51 | INFO | train_inner | epoch 403:    228 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.587, glat_context_p=0.424, word_ins=2.896, length=2.902, ppl=8.09, wps=132585, ups=2.19, wpb=60641.2, bsz=2184.2, num_updates=113500, lr=9.38647e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:24:15 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:24:18 | INFO | valid | epoch 403 | valid on 'valid' subset | loss 12.267 | nll_loss 11.08 | word_ins 12.022 | length 4.887 | ppl 4928.14 | bleu 31.64 | wps 88243.1 | wpb 21176.3 | bsz 666.3 | num_updates 113554 | best_bleu 32.07
2023-06-14 04:24:18 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:24:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint403.pt (epoch 403 @ 113554 updates, score 31.64) (writing took 7.175742752850056 seconds)
2023-06-14 04:24:26 | INFO | fairseq_cli.train | end of epoch 403 (average epoch stats below)
2023-06-14 04:24:26 | INFO | train | epoch 403 | loss 3.017 | nll_loss 1.098 | glat_accu 0.585 | glat_context_p 0.424 | word_ins 2.897 | length 2.901 | ppl 8.09 | wps 116725 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 113554 | lr 9.38423e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:24:26 | INFO | fairseq.trainer | begin training epoch 404
2023-06-14 04:24:52 | INFO | train_inner | epoch 404:     46 / 282 loss=3.02, nll_loss=1.102, glat_accu=0.594, glat_context_p=0.424, word_ins=2.9, length=2.885, ppl=8.11, wps=98607.3, ups=1.64, wpb=60282.6, bsz=2174.9, num_updates=113600, lr=9.38233e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:25:25 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 04:25:38 | INFO | train_inner | epoch 404:    147 / 282 loss=3.021, nll_loss=1.101, glat_accu=0.593, glat_context_p=0.424, word_ins=2.9, length=2.9, ppl=8.11, wps=131155, ups=2.17, wpb=60562.2, bsz=2151.5, num_updates=113700, lr=9.37821e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:26:24 | INFO | train_inner | epoch 404:    247 / 282 loss=3.031, nll_loss=1.113, glat_accu=0.587, glat_context_p=0.424, word_ins=2.91, length=2.903, ppl=8.17, wps=131759, ups=2.17, wpb=60677.6, bsz=2176.6, num_updates=113800, lr=9.37408e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:26:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:26:43 | INFO | valid | epoch 404 | valid on 'valid' subset | loss 12.328 | nll_loss 11.158 | word_ins 12.093 | length 4.683 | ppl 5141.02 | bleu 31.87 | wps 88118 | wpb 21176.3 | bsz 666.3 | num_updates 113835 | best_bleu 32.07
2023-06-14 04:26:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:26:52 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint404.pt (epoch 404 @ 113835 updates, score 31.87) (writing took 8.52456060424447 seconds)
2023-06-14 04:26:52 | INFO | fairseq_cli.train | end of epoch 404 (average epoch stats below)
2023-06-14 04:26:52 | INFO | train | epoch 404 | loss 3.025 | nll_loss 1.106 | glat_accu 0.59 | glat_context_p 0.424 | word_ins 2.904 | length 2.904 | ppl 8.14 | wps 116220 | ups 1.92 | wpb 60410.8 | bsz 2157.3 | num_updates 113835 | lr 9.37264e-05 | gnorm 0.614 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:26:52 | INFO | fairseq.trainer | begin training epoch 405
2023-06-14 04:27:27 | INFO | train_inner | epoch 405:     65 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.588, glat_context_p=0.424, word_ins=2.897, length=2.906, ppl=8.1, wps=95593.8, ups=1.59, wpb=60105.1, bsz=2156.1, num_updates=113900, lr=9.36997e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:28:13 | INFO | train_inner | epoch 405:    165 / 282 loss=3.021, nll_loss=1.103, glat_accu=0.594, glat_context_p=0.424, word_ins=2.901, length=2.889, ppl=8.12, wps=132782, ups=2.19, wpb=60704.6, bsz=2197.6, num_updates=114000, lr=9.36586e-05, gnorm=0.596, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:28:58 | INFO | train_inner | epoch 405:    265 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.591, glat_context_p=0.424, word_ins=2.907, length=2.925, ppl=8.16, wps=132971, ups=2.2, wpb=60384.7, bsz=2106.2, num_updates=114100, lr=9.36175e-05, gnorm=0.61, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:29:06 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:29:09 | INFO | valid | epoch 405 | valid on 'valid' subset | loss 12.314 | nll_loss 11.138 | word_ins 12.076 | length 4.743 | ppl 5090.63 | bleu 31.74 | wps 87509.6 | wpb 21176.3 | bsz 666.3 | num_updates 114117 | best_bleu 32.07
2023-06-14 04:29:09 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:29:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint405.pt (epoch 405 @ 114117 updates, score 31.74) (writing took 10.973467469215393 seconds)
2023-06-14 04:29:20 | INFO | fairseq_cli.train | end of epoch 405 (average epoch stats below)
2023-06-14 04:29:20 | INFO | train | epoch 405 | loss 3.023 | nll_loss 1.104 | glat_accu 0.592 | glat_context_p 0.424 | word_ins 2.902 | length 2.904 | ppl 8.13 | wps 114953 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 114117 | lr 9.36106e-05 | gnorm 0.602 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:29:20 | INFO | fairseq.trainer | begin training epoch 406
2023-06-14 04:30:04 | INFO | train_inner | epoch 406:     83 / 282 loss=3.029, nll_loss=1.11, glat_accu=0.599, glat_context_p=0.424, word_ins=2.908, length=2.896, ppl=8.16, wps=91391.5, ups=1.52, wpb=60179.2, bsz=2144.3, num_updates=114200, lr=9.35765e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:30:50 | INFO | train_inner | epoch 406:    183 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.596, glat_context_p=0.424, word_ins=2.902, length=2.9, ppl=8.13, wps=131952, ups=2.18, wpb=60444.5, bsz=2177.1, num_updates=114300, lr=9.35356e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:31:35 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:31:38 | INFO | valid | epoch 406 | valid on 'valid' subset | loss 12.406 | nll_loss 11.227 | word_ins 12.156 | length 4.989 | ppl 5426.31 | bleu 31.61 | wps 88021.5 | wpb 21176.3 | bsz 666.3 | num_updates 114399 | best_bleu 32.07
2023-06-14 04:31:38 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:31:49 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint406.pt (epoch 406 @ 114399 updates, score 31.61) (writing took 10.73037926480174 seconds)
2023-06-14 04:31:49 | INFO | fairseq_cli.train | end of epoch 406 (average epoch stats below)
2023-06-14 04:31:49 | INFO | train | epoch 406 | loss 3.026 | nll_loss 1.107 | glat_accu 0.595 | glat_context_p 0.424 | word_ins 2.905 | length 2.901 | ppl 8.14 | wps 114212 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 114399 | lr 9.34951e-05 | gnorm 0.605 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:31:49 | INFO | fairseq.trainer | begin training epoch 407
2023-06-14 04:31:56 | INFO | train_inner | epoch 407:      1 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.592, glat_context_p=0.424, word_ins=2.904, length=2.908, ppl=8.14, wps=91048.3, ups=1.51, wpb=60150.4, bsz=2140.6, num_updates=114400, lr=9.34947e-05, gnorm=0.604, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:32:41 | INFO | train_inner | epoch 407:    101 / 282 loss=3.024, nll_loss=1.104, glat_accu=0.593, glat_context_p=0.424, word_ins=2.902, length=2.912, ppl=8.13, wps=133305, ups=2.2, wpb=60526.9, bsz=2140.8, num_updates=114500, lr=9.34539e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:33:27 | INFO | train_inner | epoch 407:    201 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.591, glat_context_p=0.424, word_ins=2.903, length=2.894, ppl=8.13, wps=131219, ups=2.17, wpb=60564.8, bsz=2173.4, num_updates=114600, lr=9.34131e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:34:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:34:07 | INFO | valid | epoch 407 | valid on 'valid' subset | loss 12.339 | nll_loss 11.162 | word_ins 12.103 | length 4.719 | ppl 5179.58 | bleu 32.11 | wps 88187.8 | wpb 21176.3 | bsz 666.3 | num_updates 114681 | best_bleu 32.11
2023-06-14 04:34:07 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:34:22 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint407.pt (epoch 407 @ 114681 updates, score 32.11) (writing took 15.112970918416977 seconds)
2023-06-14 04:34:22 | INFO | fairseq_cli.train | end of epoch 407 (average epoch stats below)
2023-06-14 04:34:22 | INFO | train | epoch 407 | loss 3.024 | nll_loss 1.105 | glat_accu 0.592 | glat_context_p 0.424 | word_ins 2.903 | length 2.901 | ppl 8.14 | wps 111097 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 114681 | lr 9.33801e-05 | gnorm 0.618 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:34:23 | INFO | fairseq.trainer | begin training epoch 408
2023-06-14 04:34:36 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 04:34:38 | INFO | train_inner | epoch 408:     20 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.593, glat_context_p=0.424, word_ins=2.905, length=2.902, ppl=8.14, wps=85490.4, ups=1.42, wpb=60161.7, bsz=2157.4, num_updates=114700, lr=9.33724e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:35:24 | INFO | train_inner | epoch 408:    120 / 282 loss=3.023, nll_loss=1.103, glat_accu=0.592, glat_context_p=0.424, word_ins=2.902, length=2.908, ppl=8.13, wps=132318, ups=2.18, wpb=60582.4, bsz=2163.6, num_updates=114800, lr=9.33317e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:36:09 | INFO | train_inner | epoch 408:    220 / 282 loss=3.026, nll_loss=1.108, glat_accu=0.597, glat_context_p=0.423, word_ins=2.905, length=2.888, ppl=8.15, wps=133096, ups=2.19, wpb=60738.2, bsz=2165.5, num_updates=114900, lr=9.32911e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:36:38 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:36:41 | INFO | valid | epoch 408 | valid on 'valid' subset | loss 12.3 | nll_loss 11.113 | word_ins 12.059 | length 4.836 | ppl 5041.58 | bleu 31.52 | wps 88585.1 | wpb 21176.3 | bsz 666.3 | num_updates 114962 | best_bleu 32.11
2023-06-14 04:36:41 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:36:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint408.pt (epoch 408 @ 114962 updates, score 31.52) (writing took 11.96385670080781 seconds)
2023-06-14 04:36:53 | INFO | fairseq_cli.train | end of epoch 408 (average epoch stats below)
2023-06-14 04:36:53 | INFO | train | epoch 408 | loss 3.025 | nll_loss 1.106 | glat_accu 0.594 | glat_context_p 0.423 | word_ins 2.904 | length 2.902 | ppl 8.14 | wps 112880 | ups 1.87 | wpb 60425 | bsz 2157.9 | num_updates 114962 | lr 9.32659e-05 | gnorm 0.615 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:36:53 | INFO | fairseq.trainer | begin training epoch 409
2023-06-14 04:37:17 | INFO | train_inner | epoch 409:     38 / 282 loss=3.025, nll_loss=1.106, glat_accu=0.593, glat_context_p=0.423, word_ins=2.904, length=2.896, ppl=8.14, wps=89346.3, ups=1.48, wpb=60175.3, bsz=2148.3, num_updates=115000, lr=9.32505e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:38:03 | INFO | train_inner | epoch 409:    138 / 282 loss=3.02, nll_loss=1.1, glat_accu=0.591, glat_context_p=0.423, word_ins=2.899, length=2.902, ppl=8.11, wps=131173, ups=2.17, wpb=60444.2, bsz=2140.7, num_updates=115100, lr=9.321e-05, gnorm=0.597, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:38:48 | INFO | train_inner | epoch 409:    238 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.596, glat_context_p=0.423, word_ins=2.896, length=2.891, ppl=8.09, wps=133986, ups=2.21, wpb=60600.3, bsz=2209.8, num_updates=115200, lr=9.31695e-05, gnorm=0.587, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:39:08 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:39:11 | INFO | valid | epoch 409 | valid on 'valid' subset | loss 12.488 | nll_loss 11.322 | word_ins 12.243 | length 4.891 | ppl 5744 | bleu 31.56 | wps 87370 | wpb 21176.3 | bsz 666.3 | num_updates 115244 | best_bleu 32.11
2023-06-14 04:39:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:39:19 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint409.pt (epoch 409 @ 115244 updates, score 31.56) (writing took 7.970848083496094 seconds)
2023-06-14 04:39:19 | INFO | fairseq_cli.train | end of epoch 409 (average epoch stats below)
2023-06-14 04:39:19 | INFO | train | epoch 409 | loss 3.02 | nll_loss 1.1 | glat_accu 0.592 | glat_context_p 0.423 | word_ins 2.899 | length 2.9 | ppl 8.11 | wps 116473 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 115244 | lr 9.31517e-05 | gnorm 0.597 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:39:19 | INFO | fairseq.trainer | begin training epoch 410
2023-06-14 04:39:50 | INFO | train_inner | epoch 410:     56 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.584, glat_context_p=0.423, word_ins=2.897, length=2.918, ppl=8.1, wps=96029.9, ups=1.6, wpb=59981, bsz=2116.2, num_updates=115300, lr=9.31291e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:40:36 | INFO | train_inner | epoch 410:    156 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.593, glat_context_p=0.423, word_ins=2.899, length=2.899, ppl=8.11, wps=132746, ups=2.2, wpb=60416.5, bsz=2189.4, num_updates=115400, lr=9.30887e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:41:22 | INFO | train_inner | epoch 410:    256 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.587, glat_context_p=0.423, word_ins=2.901, length=2.898, ppl=8.12, wps=132537, ups=2.18, wpb=60792, bsz=2136.2, num_updates=115500, lr=9.30484e-05, gnorm=0.597, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:41:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:41:37 | INFO | valid | epoch 410 | valid on 'valid' subset | loss 12.33 | nll_loss 11.153 | word_ins 12.092 | length 4.754 | ppl 5148.88 | bleu 31.7 | wps 87385.5 | wpb 21176.3 | bsz 666.3 | num_updates 115526 | best_bleu 32.11
2023-06-14 04:41:37 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:41:46 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint410.pt (epoch 410 @ 115526 updates, score 31.7) (writing took 8.577743325382471 seconds)
2023-06-14 04:41:46 | INFO | fairseq_cli.train | end of epoch 410 (average epoch stats below)
2023-06-14 04:41:46 | INFO | train | epoch 410 | loss 3.019 | nll_loss 1.1 | glat_accu 0.589 | glat_context_p 0.423 | word_ins 2.898 | length 2.9 | ppl 8.11 | wps 116343 | ups 1.93 | wpb 60413.8 | bsz 2157.2 | num_updates 115526 | lr 9.30379e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:41:46 | INFO | fairseq.trainer | begin training epoch 411
2023-06-14 04:42:25 | INFO | train_inner | epoch 411:     74 / 282 loss=3.016, nll_loss=1.098, glat_accu=0.589, glat_context_p=0.423, word_ins=2.897, length=2.879, ppl=8.09, wps=95225.8, ups=1.59, wpb=60074.7, bsz=2142.6, num_updates=115600, lr=9.30082e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:43:11 | INFO | train_inner | epoch 411:    174 / 282 loss=3.023, nll_loss=1.104, glat_accu=0.589, glat_context_p=0.423, word_ins=2.902, length=2.901, ppl=8.13, wps=132170, ups=2.18, wpb=60692.5, bsz=2162.1, num_updates=115700, lr=9.2968e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:43:20 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 04:43:57 | INFO | train_inner | epoch 411:    275 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.592, glat_context_p=0.423, word_ins=2.901, length=2.908, ppl=8.12, wps=130019, ups=2.15, wpb=60605.9, bsz=2176.2, num_updates=115800, lr=9.29278e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:44:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:44:04 | INFO | valid | epoch 411 | valid on 'valid' subset | loss 12.386 | nll_loss 11.219 | word_ins 12.151 | length 4.693 | ppl 5352.17 | bleu 31.81 | wps 88156 | wpb 21176.3 | bsz 666.3 | num_updates 115807 | best_bleu 32.11
2023-06-14 04:44:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:44:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint411.pt (epoch 411 @ 115807 updates, score 31.81) (writing took 11.75427545234561 seconds)
2023-06-14 04:44:15 | INFO | fairseq_cli.train | end of epoch 411 (average epoch stats below)
2023-06-14 04:44:15 | INFO | train | epoch 411 | loss 3.02 | nll_loss 1.101 | glat_accu 0.591 | glat_context_p 0.423 | word_ins 2.9 | length 2.897 | ppl 8.11 | wps 113378 | ups 1.88 | wpb 60415.8 | bsz 2157.8 | num_updates 115807 | lr 9.2925e-05 | gnorm 0.607 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:44:15 | INFO | fairseq.trainer | begin training epoch 412
2023-06-14 04:45:04 | INFO | train_inner | epoch 412:     93 / 282 loss=3.019, nll_loss=1.099, glat_accu=0.588, glat_context_p=0.423, word_ins=2.898, length=2.91, ppl=8.11, wps=90177.9, ups=1.51, wpb=59917, bsz=2118.9, num_updates=115900, lr=9.28877e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:45:50 | INFO | train_inner | epoch 412:    193 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.585, glat_context_p=0.423, word_ins=2.899, length=2.905, ppl=8.11, wps=129902, ups=2.15, wpb=60538.5, bsz=2166.9, num_updates=116000, lr=9.28477e-05, gnorm=0.604, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:46:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:46:34 | INFO | valid | epoch 412 | valid on 'valid' subset | loss 12.381 | nll_loss 11.208 | word_ins 12.137 | length 4.869 | ppl 5334.03 | bleu 31.81 | wps 88209.3 | wpb 21176.3 | bsz 666.3 | num_updates 116089 | best_bleu 32.11
2023-06-14 04:46:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:46:47 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint412.pt (epoch 412 @ 116089 updates, score 31.81) (writing took 13.281799741089344 seconds)
2023-06-14 04:46:47 | INFO | fairseq_cli.train | end of epoch 412 (average epoch stats below)
2023-06-14 04:46:47 | INFO | train | epoch 412 | loss 3.019 | nll_loss 1.099 | glat_accu 0.589 | glat_context_p 0.423 | word_ins 2.898 | length 2.9 | ppl 8.1 | wps 112154 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 116089 | lr 9.28121e-05 | gnorm 0.606 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:46:47 | INFO | fairseq.trainer | begin training epoch 413
2023-06-14 04:46:59 | INFO | train_inner | epoch 413:     11 / 282 loss=3.017, nll_loss=1.098, glat_accu=0.598, glat_context_p=0.423, word_ins=2.896, length=2.882, ppl=8.09, wps=88300.7, ups=1.47, wpb=60271.8, bsz=2188.2, num_updates=116100, lr=9.28077e-05, gnorm=0.608, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:47:45 | INFO | train_inner | epoch 413:    111 / 282 loss=3.021, nll_loss=1.101, glat_accu=0.593, glat_context_p=0.423, word_ins=2.899, length=2.906, ppl=8.12, wps=131696, ups=2.17, wpb=60564.3, bsz=2171.1, num_updates=116200, lr=9.27677e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:48:30 | INFO | train_inner | epoch 413:    211 / 282 loss=3.022, nll_loss=1.102, glat_accu=0.595, glat_context_p=0.423, word_ins=2.901, length=2.899, ppl=8.12, wps=132367, ups=2.18, wpb=60674.5, bsz=2130.6, num_updates=116300, lr=9.27278e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:49:03 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:49:06 | INFO | valid | epoch 413 | valid on 'valid' subset | loss 12.404 | nll_loss 11.228 | word_ins 12.16 | length 4.882 | ppl 5419.86 | bleu 31.8 | wps 88604.6 | wpb 21176.3 | bsz 666.3 | num_updates 116371 | best_bleu 32.11
2023-06-14 04:49:06 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:49:17 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint413.pt (epoch 413 @ 116371 updates, score 31.8) (writing took 10.505216676741838 seconds)
2023-06-14 04:49:17 | INFO | fairseq_cli.train | end of epoch 413 (average epoch stats below)
2023-06-14 04:49:17 | INFO | train | epoch 413 | loss 3.023 | nll_loss 1.104 | glat_accu 0.595 | glat_context_p 0.423 | word_ins 2.902 | length 2.901 | ppl 8.13 | wps 114044 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 116371 | lr 9.26995e-05 | gnorm 0.617 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:49:17 | INFO | fairseq.trainer | begin training epoch 414
2023-06-14 04:49:36 | INFO | train_inner | epoch 414:     29 / 282 loss=3.025, nll_loss=1.107, glat_accu=0.596, glat_context_p=0.422, word_ins=2.905, length=2.885, ppl=8.14, wps=91175, ups=1.52, wpb=60061.9, bsz=2177.8, num_updates=116400, lr=9.2688e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:50:22 | INFO | train_inner | epoch 414:    129 / 282 loss=3.026, nll_loss=1.107, glat_accu=0.597, glat_context_p=0.422, word_ins=2.905, length=2.903, ppl=8.15, wps=131862, ups=2.18, wpb=60415.2, bsz=2177, num_updates=116500, lr=9.26482e-05, gnorm=0.604, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:51:08 | INFO | train_inner | epoch 414:    229 / 282 loss=3.024, nll_loss=1.105, glat_accu=0.588, glat_context_p=0.422, word_ins=2.903, length=2.91, ppl=8.14, wps=133583, ups=2.2, wpb=60635.3, bsz=2134.4, num_updates=116600, lr=9.26085e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:51:32 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:51:35 | INFO | valid | epoch 414 | valid on 'valid' subset | loss 12.435 | nll_loss 11.268 | word_ins 12.195 | length 4.814 | ppl 5538.98 | bleu 31.7 | wps 88506.9 | wpb 21176.3 | bsz 666.3 | num_updates 116653 | best_bleu 32.11
2023-06-14 04:51:35 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:51:47 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint414.pt (epoch 414 @ 116653 updates, score 31.7) (writing took 11.320235040038824 seconds)
2023-06-14 04:51:47 | INFO | fairseq_cli.train | end of epoch 414 (average epoch stats below)
2023-06-14 04:51:47 | INFO | train | epoch 414 | loss 3.023 | nll_loss 1.105 | glat_accu 0.592 | glat_context_p 0.422 | word_ins 2.903 | length 2.899 | ppl 8.13 | wps 113652 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 116653 | lr 9.25874e-05 | gnorm 0.61 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:51:47 | INFO | fairseq.trainer | begin training epoch 415
2023-06-14 04:52:14 | INFO | train_inner | epoch 415:     47 / 282 loss=3.017, nll_loss=1.099, glat_accu=0.585, glat_context_p=0.422, word_ins=2.897, length=2.888, ppl=8.1, wps=90784.3, ups=1.51, wpb=60288.1, bsz=2157.4, num_updates=116700, lr=9.25688e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:52:34 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 04:53:00 | INFO | train_inner | epoch 415:    148 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.594, glat_context_p=0.422, word_ins=2.9, length=2.901, ppl=8.12, wps=130860, ups=2.16, wpb=60562.8, bsz=2161.6, num_updates=116800, lr=9.25292e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:53:46 | INFO | train_inner | epoch 415:    248 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.585, glat_context_p=0.422, word_ins=2.901, length=2.921, ppl=8.12, wps=131384, ups=2.17, wpb=60455.6, bsz=2159, num_updates=116900, lr=9.24896e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:54:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:54:05 | INFO | valid | epoch 415 | valid on 'valid' subset | loss 12.433 | nll_loss 11.273 | word_ins 12.196 | length 4.736 | ppl 5531.46 | bleu 31.77 | wps 89089.2 | wpb 21176.3 | bsz 666.3 | num_updates 116934 | best_bleu 32.11
2023-06-14 04:54:05 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:54:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint415.pt (epoch 415 @ 116934 updates, score 31.77) (writing took 11.067591104656458 seconds)
2023-06-14 04:54:16 | INFO | fairseq_cli.train | end of epoch 415 (average epoch stats below)
2023-06-14 04:54:16 | INFO | train | epoch 415 | loss 3.02 | nll_loss 1.101 | glat_accu 0.589 | glat_context_p 0.422 | word_ins 2.899 | length 2.905 | ppl 8.11 | wps 113695 | ups 1.88 | wpb 60412.6 | bsz 2158.3 | num_updates 116934 | lr 9.24761e-05 | gnorm 0.616 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 04:54:16 | INFO | fairseq.trainer | begin training epoch 416
2023-06-14 04:54:52 | INFO | train_inner | epoch 416:     66 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.585, glat_context_p=0.422, word_ins=2.896, length=2.899, ppl=8.09, wps=91570.7, ups=1.52, wpb=60217.8, bsz=2114.2, num_updates=117000, lr=9.245e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:55:38 | INFO | train_inner | epoch 416:    166 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.589, glat_context_p=0.422, word_ins=2.893, length=2.902, ppl=8.08, wps=132267, ups=2.19, wpb=60509.4, bsz=2183.6, num_updates=117100, lr=9.24105e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:56:24 | INFO | train_inner | epoch 416:    266 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.587, glat_context_p=0.422, word_ins=2.899, length=2.906, ppl=8.11, wps=131298, ups=2.17, wpb=60580.3, bsz=2155.3, num_updates=117200, lr=9.23711e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:56:31 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:56:34 | INFO | valid | epoch 416 | valid on 'valid' subset | loss 12.338 | nll_loss 11.152 | word_ins 12.094 | length 4.862 | ppl 5176.06 | bleu 31.79 | wps 87895 | wpb 21176.3 | bsz 666.3 | num_updates 117216 | best_bleu 32.11
2023-06-14 04:56:34 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:56:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint416.pt (epoch 416 @ 117216 updates, score 31.79) (writing took 10.935961335897446 seconds)
2023-06-14 04:56:45 | INFO | fairseq_cli.train | end of epoch 416 (average epoch stats below)
2023-06-14 04:56:45 | INFO | train | epoch 416 | loss 3.016 | nll_loss 1.097 | glat_accu 0.588 | glat_context_p 0.422 | word_ins 2.896 | length 2.897 | ppl 8.09 | wps 113969 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 117216 | lr 9.23648e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 04:56:45 | INFO | fairseq.trainer | begin training epoch 417
2023-06-14 04:57:29 | INFO | train_inner | epoch 417:     84 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.594, glat_context_p=0.422, word_ins=2.895, length=2.886, ppl=8.09, wps=92110.2, ups=1.53, wpb=60255.3, bsz=2189.8, num_updates=117300, lr=9.23317e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 04:58:15 | INFO | train_inner | epoch 417:    184 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.595, glat_context_p=0.422, word_ins=2.901, length=2.906, ppl=8.12, wps=131657, ups=2.18, wpb=60264.2, bsz=2156.1, num_updates=117400, lr=9.22924e-05, gnorm=0.605, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 04:58:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 04:59:02 | INFO | valid | epoch 417 | valid on 'valid' subset | loss 12.417 | nll_loss 11.247 | word_ins 12.178 | length 4.785 | ppl 5467.94 | bleu 31.86 | wps 88177.6 | wpb 21176.3 | bsz 666.3 | num_updates 117498 | best_bleu 32.11
2023-06-14 04:59:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 04:59:13 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint417.pt (epoch 417 @ 117498 updates, score 31.86) (writing took 10.37972616776824 seconds)
2023-06-14 04:59:13 | INFO | fairseq_cli.train | end of epoch 417 (average epoch stats below)
2023-06-14 04:59:13 | INFO | train | epoch 417 | loss 3.02 | nll_loss 1.101 | glat_accu 0.594 | glat_context_p 0.422 | word_ins 2.899 | length 2.898 | ppl 8.11 | wps 115533 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 117498 | lr 9.22539e-05 | gnorm 0.61 | clip 0 | loss_scale 32768 | train_wall 127 | wall 0
2023-06-14 04:59:13 | INFO | fairseq.trainer | begin training epoch 418
2023-06-14 04:59:20 | INFO | train_inner | epoch 418:      2 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.591, glat_context_p=0.422, word_ins=2.901, length=2.897, ppl=8.12, wps=92651, ups=1.54, wpb=60280.2, bsz=2109.1, num_updates=117500, lr=9.22531e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:00:06 | INFO | train_inner | epoch 418:    102 / 282 loss=3.02, nll_loss=1.1, glat_accu=0.592, glat_context_p=0.422, word_ins=2.899, length=2.906, ppl=8.11, wps=132087, ups=2.18, wpb=60630.5, bsz=2162.5, num_updates=117600, lr=9.22139e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:00:52 | INFO | train_inner | epoch 418:    202 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.592, glat_context_p=0.422, word_ins=2.9, length=2.902, ppl=8.12, wps=131960, ups=2.18, wpb=60531.1, bsz=2137.3, num_updates=117700, lr=9.21747e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:01:23 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:01:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:01:32 | INFO | valid | epoch 418 | valid on 'valid' subset | loss 12.408 | nll_loss 11.244 | word_ins 12.172 | length 4.713 | ppl 5436.46 | bleu 31.78 | wps 88053.6 | wpb 21176.3 | bsz 666.3 | num_updates 117779 | best_bleu 32.11
2023-06-14 05:01:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:01:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint418.pt (epoch 418 @ 117779 updates, score 31.78) (writing took 12.717820897698402 seconds)
2023-06-14 05:01:45 | INFO | fairseq_cli.train | end of epoch 418 (average epoch stats below)
2023-06-14 05:01:45 | INFO | train | epoch 418 | loss 3.019 | nll_loss 1.1 | glat_accu 0.591 | glat_context_p 0.422 | word_ins 2.898 | length 2.898 | ppl 8.11 | wps 111716 | ups 1.85 | wpb 60412.7 | bsz 2156.4 | num_updates 117779 | lr 9.21438e-05 | gnorm 0.613 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:01:45 | INFO | fairseq.trainer | begin training epoch 419
2023-06-14 05:02:01 | INFO | train_inner | epoch 419:     21 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.591, glat_context_p=0.422, word_ins=2.896, length=2.884, ppl=8.09, wps=86990.3, ups=1.45, wpb=60089.1, bsz=2172.5, num_updates=117800, lr=9.21356e-05, gnorm=0.608, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:02:47 | INFO | train_inner | epoch 419:    121 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.579, glat_context_p=0.421, word_ins=2.887, length=2.893, ppl=8.04, wps=130651, ups=2.16, wpb=60418.4, bsz=2162.2, num_updates=117900, lr=9.20965e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:03:33 | INFO | train_inner | epoch 419:    221 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.575, glat_context_p=0.421, word_ins=2.891, length=2.892, ppl=8.06, wps=133840, ups=2.2, wpb=60793.1, bsz=2151.7, num_updates=118000, lr=9.20575e-05, gnorm=0.593, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:04:00 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:04:03 | INFO | valid | epoch 419 | valid on 'valid' subset | loss 12.536 | nll_loss 11.374 | word_ins 12.295 | length 4.805 | ppl 5939.43 | bleu 31.5 | wps 84923.3 | wpb 21176.3 | bsz 666.3 | num_updates 118061 | best_bleu 32.11
2023-06-14 05:04:03 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:04:13 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint419.pt (epoch 419 @ 118061 updates, score 31.5) (writing took 10.179764293134212 seconds)
2023-06-14 05:04:13 | INFO | fairseq_cli.train | end of epoch 419 (average epoch stats below)
2023-06-14 05:04:13 | INFO | train | epoch 419 | loss 3.011 | nll_loss 1.091 | glat_accu 0.581 | glat_context_p 0.421 | word_ins 2.891 | length 2.893 | ppl 8.06 | wps 114524 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 118061 | lr 9.20337e-05 | gnorm 0.597 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:04:14 | INFO | fairseq.trainer | begin training epoch 420
2023-06-14 05:04:37 | INFO | train_inner | epoch 420:     39 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.586, glat_context_p=0.421, word_ins=2.895, length=2.896, ppl=8.08, wps=92746.5, ups=1.54, wpb=60101.3, bsz=2157, num_updates=118100, lr=9.20185e-05, gnorm=0.6, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:05:23 | INFO | train_inner | epoch 420:    139 / 282 loss=3.012, nll_loss=1.091, glat_accu=0.589, glat_context_p=0.421, word_ins=2.891, length=2.902, ppl=8.06, wps=132846, ups=2.19, wpb=60533.1, bsz=2176.1, num_updates=118200, lr=9.19795e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:06:09 | INFO | train_inner | epoch 420:    239 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.59, glat_context_p=0.421, word_ins=2.899, length=2.899, ppl=8.11, wps=130972, ups=2.16, wpb=60634, bsz=2144.2, num_updates=118300, lr=9.19407e-05, gnorm=0.599, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:06:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:06:32 | INFO | valid | epoch 420 | valid on 'valid' subset | loss 12.37 | nll_loss 11.198 | word_ins 12.132 | length 4.744 | ppl 5293.41 | bleu 32.05 | wps 88032.1 | wpb 21176.3 | bsz 666.3 | num_updates 118343 | best_bleu 32.11
2023-06-14 05:06:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:06:42 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint420.pt (epoch 420 @ 118343 updates, score 32.05) (writing took 10.136168103665113 seconds)
2023-06-14 05:06:42 | INFO | fairseq_cli.train | end of epoch 420 (average epoch stats below)
2023-06-14 05:06:42 | INFO | train | epoch 420 | loss 3.016 | nll_loss 1.097 | glat_accu 0.589 | glat_context_p 0.421 | word_ins 2.895 | length 2.897 | ppl 8.09 | wps 114853 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 118343 | lr 9.1924e-05 | gnorm 0.61 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:06:42 | INFO | fairseq.trainer | begin training epoch 421
2023-06-14 05:07:14 | INFO | train_inner | epoch 421:     57 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.588, glat_context_p=0.421, word_ins=2.901, length=2.908, ppl=8.12, wps=92581.5, ups=1.54, wpb=59994.5, bsz=2114.6, num_updates=118400, lr=9.19018e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:07:59 | INFO | train_inner | epoch 421:    157 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.596, glat_context_p=0.421, word_ins=2.899, length=2.892, ppl=8.11, wps=133618, ups=2.21, wpb=60562.2, bsz=2146.3, num_updates=118500, lr=9.1863e-05, gnorm=0.605, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:08:45 | INFO | train_inner | epoch 421:    257 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.593, glat_context_p=0.421, word_ins=2.895, length=2.884, ppl=8.09, wps=132400, ups=2.18, wpb=60721.1, bsz=2210.1, num_updates=118600, lr=9.18243e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:08:57 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:09:00 | INFO | valid | epoch 421 | valid on 'valid' subset | loss 12.332 | nll_loss 11.153 | word_ins 12.093 | length 4.773 | ppl 5155.88 | bleu 31.99 | wps 87181.8 | wpb 21176.3 | bsz 666.3 | num_updates 118625 | best_bleu 32.11
2023-06-14 05:09:00 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:09:10 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint421.pt (epoch 421 @ 118625 updates, score 31.99) (writing took 9.423560082912445 seconds)
2023-06-14 05:09:10 | INFO | fairseq_cli.train | end of epoch 421 (average epoch stats below)
2023-06-14 05:09:10 | INFO | train | epoch 421 | loss 3.019 | nll_loss 1.1 | glat_accu 0.592 | glat_context_p 0.421 | word_ins 2.899 | length 2.894 | ppl 8.11 | wps 115318 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 118625 | lr 9.18146e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:09:10 | INFO | fairseq.trainer | begin training epoch 422
2023-06-14 05:09:51 | INFO | train_inner | epoch 422:     75 / 282 loss=3.017, nll_loss=1.098, glat_accu=0.587, glat_context_p=0.421, word_ins=2.897, length=2.894, ppl=8.1, wps=92524.1, ups=1.53, wpb=60368.1, bsz=2153.7, num_updates=118700, lr=9.17856e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:10:32 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:10:36 | INFO | train_inner | epoch 422:    176 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.594, glat_context_p=0.421, word_ins=2.894, length=2.905, ppl=8.09, wps=132917, ups=2.2, wpb=60461.4, bsz=2183, num_updates=118800, lr=9.1747e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:11:22 | INFO | train_inner | epoch 422:    276 / 282 loss=3.018, nll_loss=1.099, glat_accu=0.591, glat_context_p=0.421, word_ins=2.897, length=2.896, ppl=8.1, wps=131966, ups=2.18, wpb=60460.5, bsz=2136.2, num_updates=118900, lr=9.17084e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:11:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:11:27 | INFO | valid | epoch 422 | valid on 'valid' subset | loss 12.316 | nll_loss 11.122 | word_ins 12.065 | length 4.994 | ppl 5099.28 | bleu 31.57 | wps 88244.2 | wpb 21176.3 | bsz 666.3 | num_updates 118906 | best_bleu 32.11
2023-06-14 05:11:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:11:39 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint422.pt (epoch 422 @ 118906 updates, score 31.57) (writing took 11.022175494581461 seconds)
2023-06-14 05:11:39 | INFO | fairseq_cli.train | end of epoch 422 (average epoch stats below)
2023-06-14 05:11:39 | INFO | train | epoch 422 | loss 3.017 | nll_loss 1.097 | glat_accu 0.591 | glat_context_p 0.421 | word_ins 2.896 | length 2.899 | ppl 8.09 | wps 113975 | ups 1.89 | wpb 60414.2 | bsz 2156.9 | num_updates 118906 | lr 9.17061e-05 | gnorm 0.617 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:11:39 | INFO | fairseq.trainer | begin training epoch 423
2023-06-14 05:12:28 | INFO | train_inner | epoch 423:     94 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.596, glat_context_p=0.421, word_ins=2.892, length=2.875, ppl=8.07, wps=91094.4, ups=1.52, wpb=60112.4, bsz=2179, num_updates=119000, lr=9.16698e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:13:14 | INFO | train_inner | epoch 423:    194 / 282 loss=3.009, nll_loss=1.09, glat_accu=0.574, glat_context_p=0.421, word_ins=2.889, length=2.903, ppl=8.05, wps=131277, ups=2.17, wpb=60486.6, bsz=2155.6, num_updates=119100, lr=9.16314e-05, gnorm=0.592, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:13:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:13:58 | INFO | valid | epoch 423 | valid on 'valid' subset | loss 12.395 | nll_loss 11.227 | word_ins 12.157 | length 4.778 | ppl 5386.83 | bleu 32.06 | wps 88111.6 | wpb 21176.3 | bsz 666.3 | num_updates 119188 | best_bleu 32.11
2023-06-14 05:13:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:14:07 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint423.pt (epoch 423 @ 119188 updates, score 32.06) (writing took 8.54098353907466 seconds)
2023-06-14 05:14:07 | INFO | fairseq_cli.train | end of epoch 423 (average epoch stats below)
2023-06-14 05:14:07 | INFO | train | epoch 423 | loss 3.013 | nll_loss 1.093 | glat_accu 0.586 | glat_context_p 0.421 | word_ins 2.892 | length 2.894 | ppl 8.07 | wps 114969 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 119188 | lr 9.15975e-05 | gnorm 0.607 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:14:07 | INFO | fairseq.trainer | begin training epoch 424
2023-06-14 05:14:17 | INFO | train_inner | epoch 424:     12 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.59, glat_context_p=0.421, word_ins=2.896, length=2.901, ppl=8.09, wps=94907, ups=1.58, wpb=60204.2, bsz=2132.3, num_updates=119200, lr=9.15929e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:15:03 | INFO | train_inner | epoch 424:    112 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.589, glat_context_p=0.421, word_ins=2.888, length=2.897, ppl=8.05, wps=133539, ups=2.2, wpb=60572, bsz=2168.2, num_updates=119300, lr=9.15545e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:15:49 | INFO | train_inner | epoch 424:    212 / 282 loss=3.02, nll_loss=1.101, glat_accu=0.593, glat_context_p=0.42, word_ins=2.899, length=2.891, ppl=8.11, wps=130817, ups=2.16, wpb=60673.1, bsz=2184.3, num_updates=119400, lr=9.15162e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:16:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:16:25 | INFO | valid | epoch 424 | valid on 'valid' subset | loss 12.494 | nll_loss 11.328 | word_ins 12.248 | length 4.928 | ppl 5768.44 | bleu 31.45 | wps 88176.5 | wpb 21176.3 | bsz 666.3 | num_updates 119470 | best_bleu 32.11
2023-06-14 05:16:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:16:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint424.pt (epoch 424 @ 119470 updates, score 31.45) (writing took 11.039247091859579 seconds)
2023-06-14 05:16:36 | INFO | fairseq_cli.train | end of epoch 424 (average epoch stats below)
2023-06-14 05:16:36 | INFO | train | epoch 424 | loss 3.015 | nll_loss 1.095 | glat_accu 0.591 | glat_context_p 0.42 | word_ins 2.894 | length 2.896 | ppl 8.08 | wps 114403 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 119470 | lr 9.14894e-05 | gnorm 0.613 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:16:36 | INFO | fairseq.trainer | begin training epoch 425
2023-06-14 05:16:56 | INFO | train_inner | epoch 425:     30 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.582, glat_context_p=0.42, word_ins=2.89, length=2.91, ppl=8.06, wps=90055.9, ups=1.51, wpb=59804, bsz=2124.6, num_updates=119500, lr=9.14779e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:17:41 | INFO | train_inner | epoch 425:    130 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.581, glat_context_p=0.42, word_ins=2.887, length=2.888, ppl=8.04, wps=133537, ups=2.2, wpb=60665.7, bsz=2156.9, num_updates=119600, lr=9.14396e-05, gnorm=0.597, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:18:27 | INFO | train_inner | epoch 425:    230 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.59, glat_context_p=0.42, word_ins=2.892, length=2.885, ppl=8.07, wps=132289, ups=2.18, wpb=60651.9, bsz=2188, num_updates=119700, lr=9.14014e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:18:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:18:54 | INFO | valid | epoch 425 | valid on 'valid' subset | loss 12.41 | nll_loss 11.235 | word_ins 12.165 | length 4.908 | ppl 5442.93 | bleu 31.76 | wps 87209.6 | wpb 21176.3 | bsz 666.3 | num_updates 119752 | best_bleu 32.11
2023-06-14 05:18:54 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:19:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint425.pt (epoch 425 @ 119752 updates, score 31.76) (writing took 13.787431169301271 seconds)
2023-06-14 05:19:08 | INFO | fairseq_cli.train | end of epoch 425 (average epoch stats below)
2023-06-14 05:19:08 | INFO | train | epoch 425 | loss 3.01 | nll_loss 1.091 | glat_accu 0.584 | glat_context_p 0.42 | word_ins 2.89 | length 2.894 | ppl 8.06 | wps 112124 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 119752 | lr 9.13816e-05 | gnorm 0.6 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:19:08 | INFO | fairseq.trainer | begin training epoch 426
2023-06-14 05:19:35 | INFO | train_inner | epoch 426:     48 / 282 loss=3.02, nll_loss=1.1, glat_accu=0.59, glat_context_p=0.42, word_ins=2.899, length=2.908, ppl=8.11, wps=87659.2, ups=1.46, wpb=60119.2, bsz=2106.8, num_updates=119800, lr=9.13633e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:19:42 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:20:21 | INFO | train_inner | epoch 426:    149 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.592, glat_context_p=0.42, word_ins=2.892, length=2.892, ppl=8.07, wps=131625, ups=2.17, wpb=60637.3, bsz=2178.1, num_updates=119900, lr=9.13252e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:21:08 | INFO | train_inner | epoch 426:    249 / 282 loss=3.017, nll_loss=1.099, glat_accu=0.589, glat_context_p=0.42, word_ins=2.897, length=2.896, ppl=8.1, wps=131210, ups=2.17, wpb=60551.8, bsz=2166.4, num_updates=120000, lr=9.12871e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:21:23 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:21:26 | INFO | valid | epoch 426 | valid on 'valid' subset | loss 12.271 | nll_loss 11.083 | word_ins 12.032 | length 4.779 | ppl 4941.55 | bleu 32.01 | wps 88359.1 | wpb 21176.3 | bsz 666.3 | num_updates 120033 | best_bleu 32.11
2023-06-14 05:21:26 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:21:38 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint426.pt (epoch 426 @ 120033 updates, score 32.01) (writing took 12.05582346022129 seconds)
2023-06-14 05:21:38 | INFO | fairseq_cli.train | end of epoch 426 (average epoch stats below)
2023-06-14 05:21:38 | INFO | train | epoch 426 | loss 3.016 | nll_loss 1.096 | glat_accu 0.591 | glat_context_p 0.42 | word_ins 2.895 | length 2.896 | ppl 8.09 | wps 112893 | ups 1.87 | wpb 60412.9 | bsz 2154.2 | num_updates 120033 | lr 9.12745e-05 | gnorm 0.614 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:21:38 | INFO | fairseq.trainer | begin training epoch 427
2023-06-14 05:22:15 | INFO | train_inner | epoch 427:     67 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.587, glat_context_p=0.42, word_ins=2.894, length=2.906, ppl=8.08, wps=88910, ups=1.48, wpb=59897.6, bsz=2105.8, num_updates=120100, lr=9.12491e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:23:01 | INFO | train_inner | epoch 427:    167 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.591, glat_context_p=0.42, word_ins=2.893, length=2.892, ppl=8.08, wps=132544, ups=2.19, wpb=60602.2, bsz=2183.6, num_updates=120200, lr=9.12111e-05, gnorm=0.62, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:23:47 | INFO | train_inner | epoch 427:    267 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.592, glat_context_p=0.42, word_ins=2.894, length=2.897, ppl=8.09, wps=132405, ups=2.18, wpb=60670.5, bsz=2166.9, num_updates=120300, lr=9.11732e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:23:53 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:23:56 | INFO | valid | epoch 427 | valid on 'valid' subset | loss 12.384 | nll_loss 11.205 | word_ins 12.139 | length 4.886 | ppl 5344.91 | bleu 31.54 | wps 84907.7 | wpb 21176.3 | bsz 666.3 | num_updates 120315 | best_bleu 32.11
2023-06-14 05:23:56 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:24:07 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint427.pt (epoch 427 @ 120315 updates, score 31.54) (writing took 10.473390754312277 seconds)
2023-06-14 05:24:07 | INFO | fairseq_cli.train | end of epoch 427 (average epoch stats below)
2023-06-14 05:24:07 | INFO | train | epoch 427 | loss 3.015 | nll_loss 1.095 | glat_accu 0.591 | glat_context_p 0.42 | word_ins 2.894 | length 2.897 | ppl 8.08 | wps 114470 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 120315 | lr 9.11675e-05 | gnorm 0.622 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:24:07 | INFO | fairseq.trainer | begin training epoch 428
2023-06-14 05:24:52 | INFO | train_inner | epoch 428:     85 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.598, glat_context_p=0.42, word_ins=2.896, length=2.873, ppl=8.09, wps=92361.8, ups=1.53, wpb=60271, bsz=2176.6, num_updates=120400, lr=9.11353e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:25:38 | INFO | train_inner | epoch 428:    185 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.59, glat_context_p=0.42, word_ins=2.897, length=2.903, ppl=8.1, wps=131115, ups=2.17, wpb=60554.4, bsz=2104, num_updates=120500, lr=9.10975e-05, gnorm=0.621, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:26:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:26:25 | INFO | valid | epoch 428 | valid on 'valid' subset | loss 12.303 | nll_loss 11.117 | word_ins 12.061 | length 4.843 | ppl 5052.73 | bleu 31.83 | wps 85843.4 | wpb 21176.3 | bsz 666.3 | num_updates 120597 | best_bleu 32.11
2023-06-14 05:26:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:26:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint428.pt (epoch 428 @ 120597 updates, score 31.83) (writing took 11.283662494271994 seconds)
2023-06-14 05:26:37 | INFO | fairseq_cli.train | end of epoch 428 (average epoch stats below)
2023-06-14 05:26:37 | INFO | train | epoch 428 | loss 3.016 | nll_loss 1.097 | glat_accu 0.593 | glat_context_p 0.42 | word_ins 2.895 | length 2.891 | ppl 8.09 | wps 113658 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 120597 | lr 9.10609e-05 | gnorm 0.612 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:26:37 | INFO | fairseq.trainer | begin training epoch 429
2023-06-14 05:26:44 | INFO | train_inner | epoch 429:      3 / 282 loss=3.016, nll_loss=1.096, glat_accu=0.592, glat_context_p=0.42, word_ins=2.895, length=2.894, ppl=8.09, wps=90611, ups=1.51, wpb=60022.7, bsz=2180.7, num_updates=120600, lr=9.10597e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:27:30 | INFO | train_inner | epoch 429:    103 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.588, glat_context_p=0.42, word_ins=2.893, length=2.913, ppl=8.08, wps=133128, ups=2.2, wpb=60541.5, bsz=2131.4, num_updates=120700, lr=9.1022e-05, gnorm=0.621, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:28:16 | INFO | train_inner | epoch 429:    203 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.581, glat_context_p=0.42, word_ins=2.891, length=2.888, ppl=8.06, wps=130652, ups=2.16, wpb=60626.1, bsz=2167.9, num_updates=120800, lr=9.09843e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:28:34 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:28:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:28:55 | INFO | valid | epoch 429 | valid on 'valid' subset | loss 12.442 | nll_loss 11.278 | word_ins 12.207 | length 4.711 | ppl 5565.38 | bleu 31.9 | wps 88056.6 | wpb 21176.3 | bsz 666.3 | num_updates 120878 | best_bleu 32.11
2023-06-14 05:28:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:29:03 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint429.pt (epoch 429 @ 120878 updates, score 31.9) (writing took 8.458055946975946 seconds)
2023-06-14 05:29:03 | INFO | fairseq_cli.train | end of epoch 429 (average epoch stats below)
2023-06-14 05:29:03 | INFO | train | epoch 429 | loss 3.012 | nll_loss 1.092 | glat_accu 0.586 | glat_context_p 0.42 | word_ins 2.891 | length 2.895 | ppl 8.06 | wps 115798 | ups 1.92 | wpb 60408.4 | bsz 2157.3 | num_updates 120878 | lr 9.0955e-05 | gnorm 0.608 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:29:03 | INFO | fairseq.trainer | begin training epoch 430
2023-06-14 05:29:19 | INFO | train_inner | epoch 430:     22 / 282 loss=3.009, nll_loss=1.09, glat_accu=0.589, glat_context_p=0.419, word_ins=2.889, length=2.876, ppl=8.05, wps=96107.2, ups=1.6, wpb=60209.8, bsz=2184.2, num_updates=120900, lr=9.09467e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:30:04 | INFO | train_inner | epoch 430:    122 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.589, glat_context_p=0.419, word_ins=2.89, length=2.885, ppl=8.06, wps=132675, ups=2.19, wpb=60572.1, bsz=2177.8, num_updates=121000, lr=9.09091e-05, gnorm=0.605, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:30:50 | INFO | train_inner | epoch 430:    222 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.594, glat_context_p=0.419, word_ins=2.89, length=2.887, ppl=8.06, wps=133744, ups=2.21, wpb=60500.3, bsz=2162.6, num_updates=121100, lr=9.08715e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:31:17 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:31:20 | INFO | valid | epoch 430 | valid on 'valid' subset | loss 12.365 | nll_loss 11.193 | word_ins 12.128 | length 4.746 | ppl 5274.57 | bleu 31.9 | wps 87408.8 | wpb 21176.3 | bsz 666.3 | num_updates 121160 | best_bleu 32.11
2023-06-14 05:31:20 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:31:29 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint430.pt (epoch 430 @ 121160 updates, score 31.9) (writing took 8.627723198384047 seconds)
2023-06-14 05:31:29 | INFO | fairseq_cli.train | end of epoch 430 (average epoch stats below)
2023-06-14 05:31:29 | INFO | train | epoch 430 | loss 3.011 | nll_loss 1.092 | glat_accu 0.587 | glat_context_p 0.419 | word_ins 2.891 | length 2.891 | ppl 8.06 | wps 117099 | ups 1.94 | wpb 60413.8 | bsz 2157.2 | num_updates 121160 | lr 9.0849e-05 | gnorm 0.604 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:31:29 | INFO | fairseq.trainer | begin training epoch 431
2023-06-14 05:31:53 | INFO | train_inner | epoch 431:     40 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.573, glat_context_p=0.419, word_ins=2.894, length=2.897, ppl=8.08, wps=95443.1, ups=1.59, wpb=60207.5, bsz=2088.4, num_updates=121200, lr=9.08341e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:32:38 | INFO | train_inner | epoch 431:    140 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.589, glat_context_p=0.419, word_ins=2.888, length=2.876, ppl=8.04, wps=132970, ups=2.19, wpb=60716.2, bsz=2220.6, num_updates=121300, lr=9.07966e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:33:24 | INFO | train_inner | epoch 431:    240 / 282 loss=3.009, nll_loss=1.088, glat_accu=0.583, glat_context_p=0.419, word_ins=2.888, length=2.917, ppl=8.05, wps=133481, ups=2.21, wpb=60370.4, bsz=2148.9, num_updates=121400, lr=9.07592e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:33:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:33:46 | INFO | valid | epoch 431 | valid on 'valid' subset | loss 12.435 | nll_loss 11.273 | word_ins 12.196 | length 4.775 | ppl 5538.04 | bleu 32.02 | wps 87875.6 | wpb 21176.3 | bsz 666.3 | num_updates 121442 | best_bleu 32.11
2023-06-14 05:33:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:33:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint431.pt (epoch 431 @ 121442 updates, score 32.02) (writing took 11.128164365887642 seconds)
2023-06-14 05:33:57 | INFO | fairseq_cli.train | end of epoch 431 (average epoch stats below)
2023-06-14 05:33:57 | INFO | train | epoch 431 | loss 3.01 | nll_loss 1.09 | glat_accu 0.584 | glat_context_p 0.419 | word_ins 2.889 | length 2.896 | ppl 8.05 | wps 114924 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 121442 | lr 9.07435e-05 | gnorm 0.606 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:33:57 | INFO | fairseq.trainer | begin training epoch 432
2023-06-14 05:34:31 | INFO | train_inner | epoch 432:     58 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.58, glat_context_p=0.419, word_ins=2.887, length=2.899, ppl=8.04, wps=89049.8, ups=1.48, wpb=60000.1, bsz=2123.3, num_updates=121500, lr=9.07218e-05, gnorm=0.61, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:35:16 | INFO | train_inner | epoch 432:    158 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.592, glat_context_p=0.419, word_ins=2.893, length=2.891, ppl=8.08, wps=133574, ups=2.2, wpb=60675.2, bsz=2163.1, num_updates=121600, lr=9.06845e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:36:02 | INFO | train_inner | epoch 432:    258 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.588, glat_context_p=0.419, word_ins=2.891, length=2.893, ppl=8.06, wps=132698, ups=2.19, wpb=60613, bsz=2191, num_updates=121700, lr=9.06473e-05, gnorm=0.598, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:36:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:36:16 | INFO | valid | epoch 432 | valid on 'valid' subset | loss 12.316 | nll_loss 11.134 | word_ins 12.073 | length 4.88 | ppl 5100.15 | bleu 31.87 | wps 88636.8 | wpb 21176.3 | bsz 666.3 | num_updates 121724 | best_bleu 32.11
2023-06-14 05:36:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:36:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint432.pt (epoch 432 @ 121724 updates, score 31.87) (writing took 10.962578993290663 seconds)
2023-06-14 05:36:27 | INFO | fairseq_cli.train | end of epoch 432 (average epoch stats below)
2023-06-14 05:36:27 | INFO | train | epoch 432 | loss 3.011 | nll_loss 1.092 | glat_accu 0.587 | glat_context_p 0.419 | word_ins 2.891 | length 2.894 | ppl 8.06 | wps 113494 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 121724 | lr 9.06383e-05 | gnorm 0.611 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:36:27 | INFO | fairseq.trainer | begin training epoch 433
2023-06-14 05:37:08 | INFO | train_inner | epoch 433:     76 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.595, glat_context_p=0.419, word_ins=2.894, length=2.886, ppl=8.08, wps=91779, ups=1.53, wpb=60175.8, bsz=2158.6, num_updates=121800, lr=9.061e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:37:37 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:37:54 | INFO | train_inner | epoch 433:    177 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.592, glat_context_p=0.419, word_ins=2.893, length=2.899, ppl=8.08, wps=130108, ups=2.15, wpb=60402.1, bsz=2146, num_updates=121900, lr=9.05729e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:38:40 | INFO | train_inner | epoch 433:    277 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.59, glat_context_p=0.419, word_ins=2.896, length=2.898, ppl=8.09, wps=133032, ups=2.19, wpb=60718.4, bsz=2158.3, num_updates=122000, lr=9.05357e-05, gnorm=0.609, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:38:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:38:45 | INFO | valid | epoch 433 | valid on 'valid' subset | loss 12.373 | nll_loss 11.186 | word_ins 12.116 | length 5.16 | ppl 5306.02 | bleu 31.53 | wps 85416.9 | wpb 21176.3 | bsz 666.3 | num_updates 122005 | best_bleu 32.11
2023-06-14 05:38:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:38:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint433.pt (epoch 433 @ 122005 updates, score 31.53) (writing took 10.69306119158864 seconds)
2023-06-14 05:38:56 | INFO | fairseq_cli.train | end of epoch 433 (average epoch stats below)
2023-06-14 05:38:56 | INFO | train | epoch 433 | loss 3.014 | nll_loss 1.095 | glat_accu 0.593 | glat_context_p 0.419 | word_ins 2.894 | length 2.892 | ppl 8.08 | wps 114258 | ups 1.89 | wpb 60412 | bsz 2157.5 | num_updates 122005 | lr 9.05339e-05 | gnorm 0.613 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:38:56 | INFO | fairseq.trainer | begin training epoch 434
2023-06-14 05:39:47 | INFO | train_inner | epoch 434:     95 / 282 loss=3.019, nll_loss=1.099, glat_accu=0.598, glat_context_p=0.419, word_ins=2.897, length=2.902, ppl=8.1, wps=89627.2, ups=1.49, wpb=60010.9, bsz=2135, num_updates=122100, lr=9.04987e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:40:33 | INFO | train_inner | epoch 434:    195 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.588, glat_context_p=0.419, word_ins=2.9, length=2.908, ppl=8.12, wps=131280, ups=2.17, wpb=60607.3, bsz=2117.1, num_updates=122200, lr=9.04616e-05, gnorm=0.599, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:41:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:41:16 | INFO | valid | epoch 434 | valid on 'valid' subset | loss 12.342 | nll_loss 11.169 | word_ins 12.104 | length 4.767 | ppl 5192.79 | bleu 31.99 | wps 87417.9 | wpb 21176.3 | bsz 666.3 | num_updates 122287 | best_bleu 32.11
2023-06-14 05:41:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:41:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint434.pt (epoch 434 @ 122287 updates, score 31.99) (writing took 12.558075465261936 seconds)
2023-06-14 05:41:28 | INFO | fairseq_cli.train | end of epoch 434 (average epoch stats below)
2023-06-14 05:41:28 | INFO | train | epoch 434 | loss 3.019 | nll_loss 1.1 | glat_accu 0.595 | glat_context_p 0.419 | word_ins 2.898 | length 2.894 | ppl 8.11 | wps 111789 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 122287 | lr 9.04294e-05 | gnorm 0.616 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:41:28 | INFO | fairseq.trainer | begin training epoch 435
2023-06-14 05:41:40 | INFO | train_inner | epoch 435:     13 / 282 loss=3.018, nll_loss=1.1, glat_accu=0.601, glat_context_p=0.419, word_ins=2.898, length=2.869, ppl=8.1, wps=89261.2, ups=1.49, wpb=60086.8, bsz=2198.3, num_updates=122300, lr=9.04246e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:42:26 | INFO | train_inner | epoch 435:    113 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.599, glat_context_p=0.418, word_ins=2.892, length=2.878, ppl=8.07, wps=132113, ups=2.19, wpb=60453.4, bsz=2186.5, num_updates=122400, lr=9.03877e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:43:11 | INFO | train_inner | epoch 435:    213 / 282 loss=3.018, nll_loss=1.1, glat_accu=0.593, glat_context_p=0.418, word_ins=2.898, length=2.888, ppl=8.1, wps=133263, ups=2.2, wpb=60637.3, bsz=2149, num_updates=122500, lr=9.03508e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:43:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:43:46 | INFO | valid | epoch 435 | valid on 'valid' subset | loss 12.428 | nll_loss 11.257 | word_ins 12.187 | length 4.78 | ppl 5509.89 | bleu 32.05 | wps 85397.5 | wpb 21176.3 | bsz 666.3 | num_updates 122569 | best_bleu 32.11
2023-06-14 05:43:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:43:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint435.pt (epoch 435 @ 122569 updates, score 32.05) (writing took 11.187978204339743 seconds)
2023-06-14 05:43:57 | INFO | fairseq_cli.train | end of epoch 435 (average epoch stats below)
2023-06-14 05:43:57 | INFO | train | epoch 435 | loss 3.016 | nll_loss 1.097 | glat_accu 0.594 | glat_context_p 0.418 | word_ins 2.895 | length 2.886 | ppl 8.09 | wps 114197 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 122569 | lr 9.03254e-05 | gnorm 0.61 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:43:57 | INFO | fairseq.trainer | begin training epoch 436
2023-06-14 05:44:18 | INFO | train_inner | epoch 436:     31 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.59, glat_context_p=0.418, word_ins=2.896, length=2.905, ppl=8.09, wps=90778.5, ups=1.51, wpb=60204.4, bsz=2117.4, num_updates=122600, lr=9.03139e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:45:03 | INFO | train_inner | epoch 436:    131 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.597, glat_context_p=0.418, word_ins=2.89, length=2.88, ppl=8.06, wps=133160, ups=2.19, wpb=60723.7, bsz=2182.2, num_updates=122700, lr=9.02771e-05, gnorm=0.603, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:45:49 | INFO | train_inner | epoch 436:    231 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.583, glat_context_p=0.418, word_ins=2.892, length=2.901, ppl=8.07, wps=131693, ups=2.18, wpb=60537.3, bsz=2126.4, num_updates=122800, lr=9.02404e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:46:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:46:16 | INFO | valid | epoch 436 | valid on 'valid' subset | loss 12.44 | nll_loss 11.271 | word_ins 12.199 | length 4.805 | ppl 5556.88 | bleu 31.39 | wps 88360.1 | wpb 21176.3 | bsz 666.3 | num_updates 122851 | best_bleu 32.11
2023-06-14 05:46:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:46:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint436.pt (epoch 436 @ 122851 updates, score 31.39) (writing took 10.314943127334118 seconds)
2023-06-14 05:46:26 | INFO | fairseq_cli.train | end of epoch 436 (average epoch stats below)
2023-06-14 05:46:26 | INFO | train | epoch 436 | loss 3.011 | nll_loss 1.091 | glat_accu 0.59 | glat_context_p 0.418 | word_ins 2.89 | length 2.888 | ppl 8.06 | wps 114251 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 122851 | lr 9.02216e-05 | gnorm 0.606 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:46:26 | INFO | fairseq.trainer | begin training epoch 437
2023-06-14 05:46:50 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:46:56 | INFO | train_inner | epoch 437:     50 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.584, glat_context_p=0.418, word_ins=2.884, length=2.889, ppl=8.02, wps=89633.5, ups=1.5, wpb=59881.4, bsz=2186.9, num_updates=122900, lr=9.02036e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:47:42 | INFO | train_inner | epoch 437:    150 / 282 loss=3.008, nll_loss=1.089, glat_accu=0.587, glat_context_p=0.418, word_ins=2.888, length=2.876, ppl=8.04, wps=131740, ups=2.17, wpb=60805.4, bsz=2165.6, num_updates=123000, lr=9.0167e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:48:29 | INFO | train_inner | epoch 437:    250 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.587, glat_context_p=0.418, word_ins=2.89, length=2.888, ppl=8.06, wps=130526, ups=2.15, wpb=60570.6, bsz=2140.6, num_updates=123100, lr=9.01303e-05, gnorm=0.594, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:48:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:48:46 | INFO | valid | epoch 437 | valid on 'valid' subset | loss 12.438 | nll_loss 11.269 | word_ins 12.196 | length 4.829 | ppl 5547.45 | bleu 31.8 | wps 88287 | wpb 21176.3 | bsz 666.3 | num_updates 123132 | best_bleu 32.11
2023-06-14 05:48:46 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:48:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint437.pt (epoch 437 @ 123132 updates, score 31.8) (writing took 9.054824937134981 seconds)
2023-06-14 05:48:56 | INFO | fairseq_cli.train | end of epoch 437 (average epoch stats below)
2023-06-14 05:48:56 | INFO | train | epoch 437 | loss 3.008 | nll_loss 1.089 | glat_accu 0.587 | glat_context_p 0.418 | word_ins 2.888 | length 2.888 | ppl 8.05 | wps 113837 | ups 1.88 | wpb 60416.8 | bsz 2155.9 | num_updates 123132 | lr 9.01186e-05 | gnorm 0.603 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:48:56 | INFO | fairseq.trainer | begin training epoch 438
2023-06-14 05:49:33 | INFO | train_inner | epoch 438:     68 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.58, glat_context_p=0.418, word_ins=2.882, length=2.892, ppl=8.01, wps=92898.1, ups=1.55, wpb=59923.8, bsz=2128.5, num_updates=123200, lr=9.00937e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:50:19 | INFO | train_inner | epoch 438:    168 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.586, glat_context_p=0.418, word_ins=2.884, length=2.886, ppl=8.02, wps=133787, ups=2.21, wpb=60665.6, bsz=2178.2, num_updates=123300, lr=9.00572e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:51:04 | INFO | train_inner | epoch 438:    268 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.595, glat_context_p=0.418, word_ins=2.893, length=2.887, ppl=8.08, wps=131926, ups=2.18, wpb=60652.4, bsz=2161.1, num_updates=123400, lr=9.00207e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:51:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:51:14 | INFO | valid | epoch 438 | valid on 'valid' subset | loss 12.458 | nll_loss 11.298 | word_ins 12.218 | length 4.808 | ppl 5626.2 | bleu 31.79 | wps 87464 | wpb 21176.3 | bsz 666.3 | num_updates 123414 | best_bleu 32.11
2023-06-14 05:51:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:51:23 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint438.pt (epoch 438 @ 123414 updates, score 31.79) (writing took 8.370327901095152 seconds)
2023-06-14 05:51:23 | INFO | fairseq_cli.train | end of epoch 438 (average epoch stats below)
2023-06-14 05:51:23 | INFO | train | epoch 438 | loss 3.007 | nll_loss 1.087 | glat_accu 0.587 | glat_context_p 0.418 | word_ins 2.887 | length 2.889 | ppl 8.04 | wps 115893 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 123414 | lr 9.00156e-05 | gnorm 0.612 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 05:51:23 | INFO | fairseq.trainer | begin training epoch 439
2023-06-14 05:52:07 | INFO | train_inner | epoch 439:     86 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.595, glat_context_p=0.418, word_ins=2.892, length=2.892, ppl=8.07, wps=96382.7, ups=1.6, wpb=60065.2, bsz=2125.8, num_updates=123500, lr=8.99843e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:52:52 | INFO | train_inner | epoch 439:    186 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.602, glat_context_p=0.418, word_ins=2.894, length=2.87, ppl=8.08, wps=134788, ups=2.22, wpb=60792.5, bsz=2226.6, num_updates=123600, lr=8.99478e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 05:53:36 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:53:40 | INFO | valid | epoch 439 | valid on 'valid' subset | loss 12.372 | nll_loss 11.204 | word_ins 12.138 | length 4.692 | ppl 5302.41 | bleu 31.78 | wps 87062.4 | wpb 21176.3 | bsz 666.3 | num_updates 123696 | best_bleu 32.11
2023-06-14 05:53:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:53:52 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint439.pt (epoch 439 @ 123696 updates, score 31.78) (writing took 11.907052483409643 seconds)
2023-06-14 05:53:52 | INFO | fairseq_cli.train | end of epoch 439 (average epoch stats below)
2023-06-14 05:53:52 | INFO | train | epoch 439 | loss 3.016 | nll_loss 1.097 | glat_accu 0.596 | glat_context_p 0.418 | word_ins 2.895 | length 2.89 | ppl 8.09 | wps 114326 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 123696 | lr 8.99129e-05 | gnorm 0.627 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 05:53:52 | INFO | fairseq.trainer | begin training epoch 440
2023-06-14 05:54:00 | INFO | train_inner | epoch 440:      4 / 282 loss=3.021, nll_loss=1.101, glat_accu=0.591, glat_context_p=0.418, word_ins=2.9, length=2.909, ppl=8.12, wps=88231.2, ups=1.47, wpb=59925.3, bsz=2127.1, num_updates=123700, lr=8.99115e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:54:46 | INFO | train_inner | epoch 440:    104 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.592, glat_context_p=0.418, word_ins=2.898, length=2.9, ppl=8.11, wps=130930, ups=2.16, wpb=60511.1, bsz=2114.2, num_updates=123800, lr=8.98752e-05, gnorm=0.602, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:55:32 | INFO | train_inner | epoch 440:    204 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.595, glat_context_p=0.417, word_ins=2.891, length=2.891, ppl=8.07, wps=132157, ups=2.18, wpb=60587.7, bsz=2153.1, num_updates=123900, lr=8.98389e-05, gnorm=0.607, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 05:55:37 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 05:56:06 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 05:56:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:56:10 | INFO | valid | epoch 440 | valid on 'valid' subset | loss 12.341 | nll_loss 11.167 | word_ins 12.105 | length 4.707 | ppl 5186.62 | bleu 31.98 | wps 87729.7 | wpb 21176.3 | bsz 666.3 | num_updates 123976 | best_bleu 32.11
2023-06-14 05:56:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:56:20 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint440.pt (epoch 440 @ 123976 updates, score 31.98) (writing took 9.779548548161983 seconds)
2023-06-14 05:56:20 | INFO | fairseq_cli.train | end of epoch 440 (average epoch stats below)
2023-06-14 05:56:20 | INFO | train | epoch 440 | loss 3.014 | nll_loss 1.094 | glat_accu 0.596 | glat_context_p 0.417 | word_ins 2.893 | length 2.886 | ppl 8.08 | wps 113951 | ups 1.89 | wpb 60432 | bsz 2158.5 | num_updates 123976 | lr 8.98113e-05 | gnorm 0.608 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 05:56:20 | INFO | fairseq.trainer | begin training epoch 441
2023-06-14 05:56:36 | INFO | train_inner | epoch 441:     24 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.599, glat_context_p=0.417, word_ins=2.89, length=2.87, ppl=8.06, wps=93269.7, ups=1.55, wpb=60124.9, bsz=2188.2, num_updates=124000, lr=8.98027e-05, gnorm=0.615, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 05:57:22 | INFO | train_inner | epoch 441:    124 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.591, glat_context_p=0.417, word_ins=2.892, length=2.887, ppl=8.07, wps=132154, ups=2.18, wpb=60525, bsz=2124.6, num_updates=124100, lr=8.97665e-05, gnorm=0.612, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 05:58:08 | INFO | train_inner | epoch 441:    224 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.596, glat_context_p=0.417, word_ins=2.887, length=2.871, ppl=8.04, wps=132667, ups=2.19, wpb=60573.9, bsz=2206.9, num_updates=124200, lr=8.97303e-05, gnorm=0.603, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 05:58:34 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 05:58:37 | INFO | valid | epoch 441 | valid on 'valid' subset | loss 12.342 | nll_loss 11.159 | word_ins 12.096 | length 4.903 | ppl 5190.15 | bleu 31.84 | wps 85521.2 | wpb 21176.3 | bsz 666.3 | num_updates 124258 | best_bleu 32.11
2023-06-14 05:58:37 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 05:58:48 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint441.pt (epoch 441 @ 124258 updates, score 31.84) (writing took 10.734019197523594 seconds)
2023-06-14 05:58:48 | INFO | fairseq_cli.train | end of epoch 441 (average epoch stats below)
2023-06-14 05:58:48 | INFO | train | epoch 441 | loss 3.012 | nll_loss 1.093 | glat_accu 0.594 | glat_context_p 0.417 | word_ins 2.892 | length 2.884 | ppl 8.07 | wps 115103 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 124258 | lr 8.97094e-05 | gnorm 0.611 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 05:58:48 | INFO | fairseq.trainer | begin training epoch 442
2023-06-14 05:59:13 | INFO | train_inner | epoch 442:     42 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.594, glat_context_p=0.417, word_ins=2.896, length=2.898, ppl=8.1, wps=92373.2, ups=1.53, wpb=60290.6, bsz=2148, num_updates=124300, lr=8.96942e-05, gnorm=0.617, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 05:59:59 | INFO | train_inner | epoch 442:    142 / 282 loss=3.009, nll_loss=1.088, glat_accu=0.591, glat_context_p=0.417, word_ins=2.888, length=2.898, ppl=8.05, wps=132194, ups=2.18, wpb=60577.2, bsz=2139.3, num_updates=124400, lr=8.96582e-05, gnorm=0.614, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:00:45 | INFO | train_inner | epoch 442:    242 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.596, glat_context_p=0.417, word_ins=2.886, length=2.882, ppl=8.04, wps=132477, ups=2.19, wpb=60576.7, bsz=2186.8, num_updates=124500, lr=8.96221e-05, gnorm=0.611, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:01:03 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:01:06 | INFO | valid | epoch 442 | valid on 'valid' subset | loss 12.5 | nll_loss 11.335 | word_ins 12.255 | length 4.891 | ppl 5793.96 | bleu 31.95 | wps 87824.3 | wpb 21176.3 | bsz 666.3 | num_updates 124540 | best_bleu 32.11
2023-06-14 06:01:06 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:01:16 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint442.pt (epoch 442 @ 124540 updates, score 31.95) (writing took 10.089997999370098 seconds)
2023-06-14 06:01:16 | INFO | fairseq_cli.train | end of epoch 442 (average epoch stats below)
2023-06-14 06:01:16 | INFO | train | epoch 442 | loss 3.01 | nll_loss 1.09 | glat_accu 0.593 | glat_context_p 0.417 | word_ins 2.889 | length 2.891 | ppl 8.05 | wps 115112 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 124540 | lr 8.96077e-05 | gnorm 0.616 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 06:01:16 | INFO | fairseq.trainer | begin training epoch 443
2023-06-14 06:01:50 | INFO | train_inner | epoch 443:     60 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.595, glat_context_p=0.417, word_ins=2.891, length=2.888, ppl=8.06, wps=91652, ups=1.53, wpb=60068.9, bsz=2152.8, num_updates=124600, lr=8.95862e-05, gnorm=0.625, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 06:02:36 | INFO | train_inner | epoch 443:    160 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.594, glat_context_p=0.417, word_ins=2.892, length=2.882, ppl=8.07, wps=132278, ups=2.18, wpb=60731.7, bsz=2185.1, num_updates=124700, lr=8.95502e-05, gnorm=0.615, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:03:22 | INFO | train_inner | epoch 443:    260 / 282 loss=3.009, nll_loss=1.09, glat_accu=0.591, glat_context_p=0.417, word_ins=2.889, length=2.874, ppl=8.05, wps=130714, ups=2.16, wpb=60409.5, bsz=2160.5, num_updates=124800, lr=8.95144e-05, gnorm=0.612, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:03:32 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:03:36 | INFO | valid | epoch 443 | valid on 'valid' subset | loss 12.424 | nll_loss 11.249 | word_ins 12.178 | length 4.895 | ppl 5495.87 | bleu 31.59 | wps 87598.8 | wpb 21176.3 | bsz 666.3 | num_updates 124822 | best_bleu 32.11
2023-06-14 06:03:36 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:03:46 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint443.pt (epoch 443 @ 124822 updates, score 31.59) (writing took 10.24519481882453 seconds)
2023-06-14 06:03:46 | INFO | fairseq_cli.train | end of epoch 443 (average epoch stats below)
2023-06-14 06:03:46 | INFO | train | epoch 443 | loss 3.011 | nll_loss 1.091 | glat_accu 0.593 | glat_context_p 0.417 | word_ins 2.89 | length 2.882 | ppl 8.06 | wps 113728 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 124822 | lr 8.95065e-05 | gnorm 0.616 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 06:03:46 | INFO | fairseq.trainer | begin training epoch 444
2023-06-14 06:04:28 | INFO | train_inner | epoch 444:     78 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.589, glat_context_p=0.417, word_ins=2.887, length=2.885, ppl=8.04, wps=91571.9, ups=1.53, wpb=60027.5, bsz=2153.5, num_updates=124900, lr=8.94785e-05, gnorm=0.623, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 06:05:14 | INFO | train_inner | epoch 444:    178 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.596, glat_context_p=0.417, word_ins=2.892, length=2.898, ppl=8.07, wps=132601, ups=2.19, wpb=60554.4, bsz=2134.5, num_updates=125000, lr=8.94427e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:05:59 | INFO | train_inner | epoch 444:    278 / 282 loss=3.016, nll_loss=1.098, glat_accu=0.593, glat_context_p=0.417, word_ins=2.896, length=2.889, ppl=8.09, wps=132404, ups=2.18, wpb=60719.9, bsz=2167, num_updates=125100, lr=8.9407e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:06:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:06:04 | INFO | valid | epoch 444 | valid on 'valid' subset | loss 12.432 | nll_loss 11.264 | word_ins 12.188 | length 4.878 | ppl 5525.82 | bleu 31.91 | wps 86994.3 | wpb 21176.3 | bsz 666.3 | num_updates 125104 | best_bleu 32.11
2023-06-14 06:06:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:06:15 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint444.pt (epoch 444 @ 125104 updates, score 31.91) (writing took 10.230708073824644 seconds)
2023-06-14 06:06:15 | INFO | fairseq_cli.train | end of epoch 444 (average epoch stats below)
2023-06-14 06:06:15 | INFO | train | epoch 444 | loss 3.012 | nll_loss 1.093 | glat_accu 0.594 | glat_context_p 0.417 | word_ins 2.891 | length 2.889 | ppl 8.07 | wps 114526 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 125104 | lr 8.94055e-05 | gnorm 0.621 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:06:15 | INFO | fairseq.trainer | begin training epoch 445
2023-06-14 06:07:05 | INFO | train_inner | epoch 445:     96 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.602, glat_context_p=0.417, word_ins=2.892, length=2.866, ppl=8.07, wps=91355.5, ups=1.52, wpb=60104.4, bsz=2162.2, num_updates=125200, lr=8.93713e-05, gnorm=0.621, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:07:51 | INFO | train_inner | epoch 445:    196 / 282 loss=3.022, nll_loss=1.103, glat_accu=0.594, glat_context_p=0.417, word_ins=2.901, length=2.896, ppl=8.12, wps=131248, ups=2.17, wpb=60602.3, bsz=2148.9, num_updates=125300, lr=8.93356e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:08:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:08:33 | INFO | valid | epoch 445 | valid on 'valid' subset | loss 12.475 | nll_loss 11.299 | word_ins 12.231 | length 4.89 | ppl 5693.49 | bleu 31.95 | wps 88146.4 | wpb 21176.3 | bsz 666.3 | num_updates 125386 | best_bleu 32.11
2023-06-14 06:08:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:08:44 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint445.pt (epoch 445 @ 125386 updates, score 31.95) (writing took 10.053358469158411 seconds)
2023-06-14 06:08:44 | INFO | fairseq_cli.train | end of epoch 445 (average epoch stats below)
2023-06-14 06:08:44 | INFO | train | epoch 445 | loss 3.017 | nll_loss 1.098 | glat_accu 0.598 | glat_context_p 0.417 | word_ins 2.896 | length 2.887 | ppl 8.1 | wps 114404 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 125386 | lr 8.93049e-05 | gnorm 0.619 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:08:44 | INFO | fairseq.trainer | begin training epoch 446
2023-06-14 06:08:56 | INFO | train_inner | epoch 446:     14 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.597, glat_context_p=0.416, word_ins=2.895, length=2.898, ppl=8.09, wps=92961.5, ups=1.55, wpb=60112.6, bsz=2145, num_updates=125400, lr=8.93e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:09:42 | INFO | train_inner | epoch 446:    114 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.597, glat_context_p=0.416, word_ins=2.894, length=2.892, ppl=8.08, wps=132480, ups=2.19, wpb=60557.3, bsz=2126.2, num_updates=125500, lr=8.92644e-05, gnorm=0.601, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:10:28 | INFO | train_inner | epoch 446:    214 / 282 loss=3.017, nll_loss=1.097, glat_accu=0.592, glat_context_p=0.416, word_ins=2.895, length=2.895, ppl=8.09, wps=131510, ups=2.18, wpb=60370.2, bsz=2162.1, num_updates=125600, lr=8.92288e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:10:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:11:02 | INFO | valid | epoch 446 | valid on 'valid' subset | loss 12.442 | nll_loss 11.274 | word_ins 12.201 | length 4.806 | ppl 5565.38 | bleu 31.88 | wps 86707.1 | wpb 21176.3 | bsz 666.3 | num_updates 125668 | best_bleu 32.11
2023-06-14 06:11:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:11:13 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint446.pt (epoch 446 @ 125668 updates, score 31.88) (writing took 11.171672489494085 seconds)
2023-06-14 06:11:13 | INFO | fairseq_cli.train | end of epoch 446 (average epoch stats below)
2023-06-14 06:11:13 | INFO | train | epoch 446 | loss 3.015 | nll_loss 1.096 | glat_accu 0.595 | glat_context_p 0.416 | word_ins 2.894 | length 2.887 | ppl 8.08 | wps 113976 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 125668 | lr 8.92047e-05 | gnorm 0.613 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:11:13 | INFO | fairseq.trainer | begin training epoch 447
2023-06-14 06:11:34 | INFO | train_inner | epoch 447:     32 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.593, glat_context_p=0.416, word_ins=2.894, length=2.872, ppl=8.08, wps=90685.3, ups=1.5, wpb=60307.3, bsz=2174.6, num_updates=125700, lr=8.91933e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:12:20 | INFO | train_inner | epoch 447:    132 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.602, glat_context_p=0.416, word_ins=2.887, length=2.864, ppl=8.04, wps=133400, ups=2.2, wpb=60771.9, bsz=2187.4, num_updates=125800, lr=8.91579e-05, gnorm=0.609, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:13:05 | INFO | train_inner | epoch 447:    232 / 282 loss=3.022, nll_loss=1.102, glat_accu=0.593, glat_context_p=0.416, word_ins=2.9, length=2.91, ppl=8.12, wps=133124, ups=2.2, wpb=60460.2, bsz=2131.4, num_updates=125900, lr=8.91225e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:13:28 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:13:31 | INFO | valid | epoch 447 | valid on 'valid' subset | loss 12.408 | nll_loss 11.234 | word_ins 12.165 | length 4.869 | ppl 5433.69 | bleu 31.61 | wps 87517.8 | wpb 21176.3 | bsz 666.3 | num_updates 125950 | best_bleu 32.11
2023-06-14 06:13:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:13:43 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint447.pt (epoch 447 @ 125950 updates, score 31.61) (writing took 11.897318221628666 seconds)
2023-06-14 06:13:43 | INFO | fairseq_cli.train | end of epoch 447 (average epoch stats below)
2023-06-14 06:13:43 | INFO | train | epoch 447 | loss 3.015 | nll_loss 1.096 | glat_accu 0.597 | glat_context_p 0.416 | word_ins 2.894 | length 2.884 | ppl 8.09 | wps 113617 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 125950 | lr 8.91048e-05 | gnorm 0.618 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:13:43 | INFO | fairseq.trainer | begin training epoch 448
2023-06-14 06:14:12 | INFO | train_inner | epoch 448:     50 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.596, glat_context_p=0.416, word_ins=2.894, length=2.872, ppl=8.08, wps=89703.1, ups=1.49, wpb=60161.6, bsz=2156.2, num_updates=126000, lr=8.90871e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:14:22 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 06:14:59 | INFO | train_inner | epoch 448:    151 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.59, glat_context_p=0.416, word_ins=2.891, length=2.892, ppl=8.07, wps=130078, ups=2.15, wpb=60629.9, bsz=2151.3, num_updates=126100, lr=8.90517e-05, gnorm=0.605, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:15:45 | INFO | train_inner | epoch 448:    251 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.59, glat_context_p=0.416, word_ins=2.89, length=2.881, ppl=8.06, wps=131931, ups=2.18, wpb=60502.5, bsz=2161.8, num_updates=126200, lr=8.90165e-05, gnorm=0.595, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:15:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:16:02 | INFO | valid | epoch 448 | valid on 'valid' subset | loss 12.514 | nll_loss 11.346 | word_ins 12.265 | length 4.966 | ppl 5848.34 | bleu 31.68 | wps 88404.7 | wpb 21176.3 | bsz 666.3 | num_updates 126231 | best_bleu 32.11
2023-06-14 06:16:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:16:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint448.pt (epoch 448 @ 126231 updates, score 31.68) (writing took 10.599636770784855 seconds)
2023-06-14 06:16:12 | INFO | fairseq_cli.train | end of epoch 448 (average epoch stats below)
2023-06-14 06:16:12 | INFO | train | epoch 448 | loss 3.011 | nll_loss 1.091 | glat_accu 0.592 | glat_context_p 0.416 | word_ins 2.89 | length 2.883 | ppl 8.06 | wps 113682 | ups 1.88 | wpb 60411.4 | bsz 2156.6 | num_updates 126231 | lr 8.90055e-05 | gnorm 0.608 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:16:12 | INFO | fairseq.trainer | begin training epoch 449
2023-06-14 06:16:50 | INFO | train_inner | epoch 449:     69 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.601, glat_context_p=0.416, word_ins=2.883, length=2.861, ppl=8.02, wps=92044.7, ups=1.54, wpb=59954.2, bsz=2211.7, num_updates=126300, lr=8.89812e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:17:36 | INFO | train_inner | epoch 449:    169 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.592, glat_context_p=0.416, word_ins=2.894, length=2.895, ppl=8.08, wps=130813, ups=2.15, wpb=60730.6, bsz=2151.1, num_updates=126400, lr=8.8946e-05, gnorm=0.612, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:18:22 | INFO | train_inner | epoch 449:    269 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.585, glat_context_p=0.416, word_ins=2.893, length=2.903, ppl=8.08, wps=131335, ups=2.17, wpb=60467.5, bsz=2118.6, num_updates=126500, lr=8.89108e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:18:28 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:18:31 | INFO | valid | epoch 449 | valid on 'valid' subset | loss 12.436 | nll_loss 11.26 | word_ins 12.19 | length 4.911 | ppl 5541.8 | bleu 31.48 | wps 86863.8 | wpb 21176.3 | bsz 666.3 | num_updates 126513 | best_bleu 32.11
2023-06-14 06:18:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:18:41 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint449.pt (epoch 449 @ 126513 updates, score 31.48) (writing took 9.929821066558361 seconds)
2023-06-14 06:18:41 | INFO | fairseq_cli.train | end of epoch 449 (average epoch stats below)
2023-06-14 06:18:41 | INFO | train | epoch 449 | loss 3.011 | nll_loss 1.091 | glat_accu 0.592 | glat_context_p 0.416 | word_ins 2.89 | length 2.886 | ppl 8.06 | wps 114423 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 126513 | lr 8.89063e-05 | gnorm 0.616 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:18:41 | INFO | fairseq.trainer | begin training epoch 450
2023-06-14 06:19:27 | INFO | train_inner | epoch 450:     87 / 282 loss=3.006, nll_loss=1.087, glat_accu=0.596, glat_context_p=0.416, word_ins=2.886, length=2.862, ppl=8.03, wps=93516.9, ups=1.55, wpb=60191.7, bsz=2203.3, num_updates=126600, lr=8.88757e-05, gnorm=0.62, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:20:13 | INFO | train_inner | epoch 450:    187 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.589, glat_context_p=0.416, word_ins=2.889, length=2.89, ppl=8.05, wps=131141, ups=2.16, wpb=60614.6, bsz=2133.4, num_updates=126700, lr=8.88406e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:20:56 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:20:59 | INFO | valid | epoch 450 | valid on 'valid' subset | loss 12.408 | nll_loss 11.245 | word_ins 12.173 | length 4.673 | ppl 5434.61 | bleu 32.01 | wps 88415.8 | wpb 21176.3 | bsz 666.3 | num_updates 126795 | best_bleu 32.11
2023-06-14 06:20:59 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:21:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint450.pt (epoch 450 @ 126795 updates, score 32.01) (writing took 9.844599567353725 seconds)
2023-06-14 06:21:09 | INFO | fairseq_cli.train | end of epoch 450 (average epoch stats below)
2023-06-14 06:21:09 | INFO | train | epoch 450 | loss 3.011 | nll_loss 1.091 | glat_accu 0.594 | glat_context_p 0.416 | word_ins 2.89 | length 2.883 | ppl 8.06 | wps 115086 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 126795 | lr 8.88074e-05 | gnorm 0.615 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:21:09 | INFO | fairseq.trainer | begin training epoch 451
2023-06-14 06:21:17 | INFO | train_inner | epoch 451:      5 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.597, glat_context_p=0.416, word_ins=2.894, length=2.892, ppl=8.08, wps=93048.4, ups=1.55, wpb=60056.1, bsz=2123.4, num_updates=126800, lr=8.88056e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:22:03 | INFO | train_inner | epoch 451:    105 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.597, glat_context_p=0.415, word_ins=2.892, length=2.886, ppl=8.08, wps=131575, ups=2.17, wpb=60548.1, bsz=2138.6, num_updates=126900, lr=8.87706e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:22:49 | INFO | train_inner | epoch 451:    205 / 282 loss=3.021, nll_loss=1.102, glat_accu=0.599, glat_context_p=0.415, word_ins=2.9, length=2.877, ppl=8.12, wps=133560, ups=2.2, wpb=60742.6, bsz=2181.4, num_updates=127000, lr=8.87357e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:23:10 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 06:23:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:23:27 | INFO | valid | epoch 451 | valid on 'valid' subset | loss 12.366 | nll_loss 11.194 | word_ins 12.126 | length 4.781 | ppl 5277.26 | bleu 31.85 | wps 87618.3 | wpb 21176.3 | bsz 666.3 | num_updates 127076 | best_bleu 32.11
2023-06-14 06:23:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:23:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint451.pt (epoch 451 @ 127076 updates, score 31.85) (writing took 8.646842818707228 seconds)
2023-06-14 06:23:36 | INFO | fairseq_cli.train | end of epoch 451 (average epoch stats below)
2023-06-14 06:23:36 | INFO | train | epoch 451 | loss 3.016 | nll_loss 1.096 | glat_accu 0.599 | glat_context_p 0.415 | word_ins 2.895 | length 2.884 | ppl 8.09 | wps 115839 | ups 1.92 | wpb 60412.1 | bsz 2157.7 | num_updates 127076 | lr 8.87091e-05 | gnorm 0.616 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:23:36 | INFO | fairseq.trainer | begin training epoch 452
2023-06-14 06:23:52 | INFO | train_inner | epoch 452:     24 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.601, glat_context_p=0.415, word_ins=2.893, length=2.887, ppl=8.08, wps=95078, ups=1.59, wpb=59922.5, bsz=2166.8, num_updates=127100, lr=8.87007e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:24:38 | INFO | train_inner | epoch 452:    124 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.597, glat_context_p=0.415, word_ins=2.895, length=2.885, ppl=8.09, wps=131121, ups=2.16, wpb=60624.1, bsz=2125.3, num_updates=127200, lr=8.86659e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:25:24 | INFO | train_inner | epoch 452:    224 / 282 loss=3.019, nll_loss=1.099, glat_accu=0.591, glat_context_p=0.415, word_ins=2.897, length=2.901, ppl=8.1, wps=132040, ups=2.18, wpb=60683.5, bsz=2153.2, num_updates=127300, lr=8.8631e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:25:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:25:53 | INFO | valid | epoch 452 | valid on 'valid' subset | loss 12.369 | nll_loss 11.192 | word_ins 12.127 | length 4.864 | ppl 5291.62 | bleu 31.84 | wps 84738.4 | wpb 21176.3 | bsz 666.3 | num_updates 127358 | best_bleu 32.11
2023-06-14 06:25:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:26:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint452.pt (epoch 452 @ 127358 updates, score 31.84) (writing took 11.161207344383001 seconds)
2023-06-14 06:26:05 | INFO | fairseq_cli.train | end of epoch 452 (average epoch stats below)
2023-06-14 06:26:05 | INFO | train | epoch 452 | loss 3.017 | nll_loss 1.097 | glat_accu 0.598 | glat_context_p 0.415 | word_ins 2.896 | length 2.885 | ppl 8.09 | wps 114503 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 127358 | lr 8.86108e-05 | gnorm 0.623 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:26:05 | INFO | fairseq.trainer | begin training epoch 453
2023-06-14 06:26:30 | INFO | train_inner | epoch 453:     42 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.604, glat_context_p=0.415, word_ins=2.895, length=2.867, ppl=8.09, wps=90502.5, ups=1.51, wpb=60041.9, bsz=2190.3, num_updates=127400, lr=8.85962e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:27:16 | INFO | train_inner | epoch 453:    142 / 282 loss=3.015, nll_loss=1.097, glat_accu=0.594, glat_context_p=0.415, word_ins=2.895, length=2.876, ppl=8.09, wps=132048, ups=2.18, wpb=60607.3, bsz=2182.8, num_updates=127500, lr=8.85615e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:28:02 | INFO | train_inner | epoch 453:    242 / 282 loss=3.013, nll_loss=1.092, glat_accu=0.595, glat_context_p=0.415, word_ins=2.891, length=2.897, ppl=8.07, wps=131132, ups=2.17, wpb=60501.3, bsz=2128.1, num_updates=127600, lr=8.85268e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:28:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:28:24 | INFO | valid | epoch 453 | valid on 'valid' subset | loss 12.387 | nll_loss 11.214 | word_ins 12.149 | length 4.76 | ppl 5354.9 | bleu 31.96 | wps 88532.6 | wpb 21176.3 | bsz 666.3 | num_updates 127640 | best_bleu 32.11
2023-06-14 06:28:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:28:35 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint453.pt (epoch 453 @ 127640 updates, score 31.96) (writing took 10.675386883318424 seconds)
2023-06-14 06:28:35 | INFO | fairseq_cli.train | end of epoch 453 (average epoch stats below)
2023-06-14 06:28:35 | INFO | train | epoch 453 | loss 3.014 | nll_loss 1.095 | glat_accu 0.596 | glat_context_p 0.415 | word_ins 2.893 | length 2.885 | ppl 8.08 | wps 113592 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 127640 | lr 8.85129e-05 | gnorm 0.627 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:28:35 | INFO | fairseq.trainer | begin training epoch 454
2023-06-14 06:29:08 | INFO | train_inner | epoch 454:     60 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.593, glat_context_p=0.415, word_ins=2.887, length=2.896, ppl=8.05, wps=91499.9, ups=1.53, wpb=59965.2, bsz=2137.9, num_updates=127700, lr=8.84921e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:29:54 | INFO | train_inner | epoch 454:    160 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.596, glat_context_p=0.415, word_ins=2.89, length=2.864, ppl=8.06, wps=133550, ups=2.2, wpb=60786.8, bsz=2184.1, num_updates=127800, lr=8.84575e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:30:39 | INFO | train_inner | epoch 454:    260 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.599, glat_context_p=0.415, word_ins=2.89, length=2.878, ppl=8.06, wps=133957, ups=2.21, wpb=60621.2, bsz=2189.9, num_updates=127900, lr=8.84229e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:30:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:30:52 | INFO | valid | epoch 454 | valid on 'valid' subset | loss 12.493 | nll_loss 11.337 | word_ins 12.253 | length 4.799 | ppl 5763.54 | bleu 31.69 | wps 87698.5 | wpb 21176.3 | bsz 666.3 | num_updates 127922 | best_bleu 32.11
2023-06-14 06:30:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:31:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint454.pt (epoch 454 @ 127922 updates, score 31.69) (writing took 13.741169024258852 seconds)
2023-06-14 06:31:06 | INFO | fairseq_cli.train | end of epoch 454 (average epoch stats below)
2023-06-14 06:31:06 | INFO | train | epoch 454 | loss 3.011 | nll_loss 1.091 | glat_accu 0.593 | glat_context_p 0.415 | word_ins 2.89 | length 2.884 | ppl 8.06 | wps 112462 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 127922 | lr 8.84153e-05 | gnorm 0.622 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:31:06 | INFO | fairseq.trainer | begin training epoch 455
2023-06-14 06:31:48 | INFO | train_inner | epoch 455:     78 / 282 loss=3.011, nll_loss=1.09, glat_accu=0.589, glat_context_p=0.415, word_ins=2.89, length=2.894, ppl=8.06, wps=86731.1, ups=1.45, wpb=59890.4, bsz=2136.6, num_updates=128000, lr=8.83883e-05, gnorm=0.621, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:32:20 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 06:32:34 | INFO | train_inner | epoch 455:    179 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.589, glat_context_p=0.415, word_ins=2.891, length=2.901, ppl=8.07, wps=131688, ups=2.17, wpb=60656.2, bsz=2116.7, num_updates=128100, lr=8.83538e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:33:20 | INFO | train_inner | epoch 455:    279 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.598, glat_context_p=0.415, word_ins=2.891, length=2.868, ppl=8.06, wps=133003, ups=2.19, wpb=60643.3, bsz=2186.3, num_updates=128200, lr=8.83194e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:33:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:33:24 | INFO | valid | epoch 455 | valid on 'valid' subset | loss 12.454 | nll_loss 11.289 | word_ins 12.213 | length 4.806 | ppl 5610.94 | bleu 31.71 | wps 88891.6 | wpb 21176.3 | bsz 666.3 | num_updates 128203 | best_bleu 32.11
2023-06-14 06:33:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:33:39 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint455.pt (epoch 455 @ 128203 updates, score 31.71) (writing took 15.333263870328665 seconds)
2023-06-14 06:33:39 | INFO | fairseq_cli.train | end of epoch 455 (average epoch stats below)
2023-06-14 06:33:39 | INFO | train | epoch 455 | loss 3.01 | nll_loss 1.091 | glat_accu 0.594 | glat_context_p 0.415 | word_ins 2.889 | length 2.883 | ppl 8.06 | wps 110867 | ups 1.84 | wpb 60408.3 | bsz 2157.2 | num_updates 128203 | lr 8.83183e-05 | gnorm 0.62 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:33:39 | INFO | fairseq.trainer | begin training epoch 456
2023-06-14 06:34:29 | INFO | train_inner | epoch 456:     97 / 282 loss=3.006, nll_loss=1.087, glat_accu=0.596, glat_context_p=0.415, word_ins=2.886, length=2.862, ppl=8.03, wps=86560.2, ups=1.44, wpb=60284.3, bsz=2190.2, num_updates=128300, lr=8.82849e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:35:15 | INFO | train_inner | epoch 456:    197 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.585, glat_context_p=0.414, word_ins=2.891, length=2.892, ppl=8.06, wps=131535, ups=2.17, wpb=60551, bsz=2147.6, num_updates=128400, lr=8.82506e-05, gnorm=0.61, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:35:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:35:57 | INFO | valid | epoch 456 | valid on 'valid' subset | loss 12.519 | nll_loss 11.347 | word_ins 12.268 | length 5.019 | ppl 5869.24 | bleu 31.5 | wps 86663.4 | wpb 21176.3 | bsz 666.3 | num_updates 128485 | best_bleu 32.11
2023-06-14 06:35:57 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:36:09 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint456.pt (epoch 456 @ 128485 updates, score 31.5) (writing took 11.744025480002165 seconds)
2023-06-14 06:36:09 | INFO | fairseq_cli.train | end of epoch 456 (average epoch stats below)
2023-06-14 06:36:09 | INFO | train | epoch 456 | loss 3.009 | nll_loss 1.09 | glat_accu 0.588 | glat_context_p 0.414 | word_ins 2.889 | length 2.883 | ppl 8.05 | wps 113734 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 128485 | lr 8.82214e-05 | gnorm 0.619 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:36:09 | INFO | fairseq.trainer | begin training epoch 457
2023-06-14 06:36:22 | INFO | train_inner | epoch 457:     15 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.584, glat_context_p=0.414, word_ins=2.888, length=2.893, ppl=8.05, wps=89564.8, ups=1.49, wpb=59975.5, bsz=2123.3, num_updates=128500, lr=8.82162e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:37:08 | INFO | train_inner | epoch 457:    115 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.584, glat_context_p=0.414, word_ins=2.883, length=2.88, ppl=8.01, wps=131320, ups=2.16, wpb=60673.7, bsz=2137.8, num_updates=128600, lr=8.81819e-05, gnorm=0.615, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:37:55 | INFO | train_inner | epoch 457:    215 / 282 loss=3.004, nll_loss=1.085, glat_accu=0.586, glat_context_p=0.414, word_ins=2.885, length=2.868, ppl=8.02, wps=129720, ups=2.14, wpb=60621.6, bsz=2170.5, num_updates=128700, lr=8.81476e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-14 06:38:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:38:28 | INFO | valid | epoch 457 | valid on 'valid' subset | loss 12.351 | nll_loss 11.173 | word_ins 12.111 | length 4.79 | ppl 5225.53 | bleu 32.03 | wps 88247.7 | wpb 21176.3 | bsz 666.3 | num_updates 128767 | best_bleu 32.11
2023-06-14 06:38:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:38:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint457.pt (epoch 457 @ 128767 updates, score 32.03) (writing took 11.420102499425411 seconds)
2023-06-14 06:38:40 | INFO | fairseq_cli.train | end of epoch 457 (average epoch stats below)
2023-06-14 06:38:40 | INFO | train | epoch 457 | loss 3.005 | nll_loss 1.086 | glat_accu 0.587 | glat_context_p 0.414 | word_ins 2.885 | length 2.882 | ppl 8.03 | wps 112847 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 128767 | lr 8.81247e-05 | gnorm 0.619 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:38:40 | INFO | fairseq.trainer | begin training epoch 458
2023-06-14 06:39:01 | INFO | train_inner | epoch 458:     33 / 282 loss=3.01, nll_loss=1.089, glat_accu=0.588, glat_context_p=0.414, word_ins=2.888, length=2.908, ppl=8.06, wps=90674.6, ups=1.51, wpb=59988.8, bsz=2138.6, num_updates=128800, lr=8.81134e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:39:47 | INFO | train_inner | epoch 458:    133 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.589, glat_context_p=0.414, word_ins=2.882, length=2.879, ppl=8.01, wps=130975, ups=2.16, wpb=60560.9, bsz=2172.2, num_updates=128900, lr=8.80792e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:40:33 | INFO | train_inner | epoch 458:    233 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.588, glat_context_p=0.414, word_ins=2.883, length=2.87, ppl=8.01, wps=133213, ups=2.2, wpb=60594.1, bsz=2176.6, num_updates=129000, lr=8.80451e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:40:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:40:59 | INFO | valid | epoch 458 | valid on 'valid' subset | loss 12.34 | nll_loss 11.162 | word_ins 12.099 | length 4.808 | ppl 5183.98 | bleu 31.58 | wps 85879 | wpb 21176.3 | bsz 666.3 | num_updates 129049 | best_bleu 32.11
2023-06-14 06:40:59 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:41:10 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint458.pt (epoch 458 @ 129049 updates, score 31.58) (writing took 11.197467889636755 seconds)
2023-06-14 06:41:10 | INFO | fairseq_cli.train | end of epoch 458 (average epoch stats below)
2023-06-14 06:41:10 | INFO | train | epoch 458 | loss 3.005 | nll_loss 1.085 | glat_accu 0.588 | glat_context_p 0.414 | word_ins 2.885 | length 2.886 | ppl 8.03 | wps 113662 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 129049 | lr 8.80284e-05 | gnorm 0.616 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:41:10 | INFO | fairseq.trainer | begin training epoch 459
2023-06-14 06:41:37 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 06:41:40 | INFO | train_inner | epoch 459:     52 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.582, glat_context_p=0.414, word_ins=2.888, length=2.91, ppl=8.05, wps=89617.8, ups=1.49, wpb=60065.6, bsz=2122.6, num_updates=129100, lr=8.8011e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:42:25 | INFO | train_inner | epoch 459:    152 / 282 loss=3.005, nll_loss=1.086, glat_accu=0.595, glat_context_p=0.414, word_ins=2.885, length=2.869, ppl=8.03, wps=133342, ups=2.2, wpb=60567.2, bsz=2205.5, num_updates=129200, lr=8.79769e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:43:12 | INFO | train_inner | epoch 459:    252 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.589, glat_context_p=0.414, word_ins=2.893, length=2.904, ppl=8.08, wps=130954, ups=2.16, wpb=60499.1, bsz=2126.1, num_updates=129300, lr=8.79429e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:43:21 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 06:43:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:43:28 | INFO | valid | epoch 459 | valid on 'valid' subset | loss 12.48 | nll_loss 11.311 | word_ins 12.237 | length 4.871 | ppl 5714.8 | bleu 31.72 | wps 85767 | wpb 21176.3 | bsz 666.3 | num_updates 129329 | best_bleu 32.11
2023-06-14 06:43:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:43:39 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint459.pt (epoch 459 @ 129329 updates, score 31.72) (writing took 10.61849058419466 seconds)
2023-06-14 06:43:39 | INFO | fairseq_cli.train | end of epoch 459 (average epoch stats below)
2023-06-14 06:43:39 | INFO | train | epoch 459 | loss 3.008 | nll_loss 1.089 | glat_accu 0.59 | glat_context_p 0.414 | word_ins 2.888 | length 2.883 | ppl 8.05 | wps 113459 | ups 1.88 | wpb 60413.9 | bsz 2156.2 | num_updates 129329 | lr 8.7933e-05 | gnorm 0.624 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 06:43:39 | INFO | fairseq.trainer | begin training epoch 460
2023-06-14 06:44:17 | INFO | train_inner | epoch 460:     71 / 282 loss=3.003, nll_loss=1.084, glat_accu=0.591, glat_context_p=0.414, word_ins=2.884, length=2.859, ppl=8.02, wps=91782.8, ups=1.52, wpb=60188.5, bsz=2173.9, num_updates=129400, lr=8.79089e-05, gnorm=0.627, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:45:03 | INFO | train_inner | epoch 460:    171 / 282 loss=3.008, nll_loss=1.089, glat_accu=0.601, glat_context_p=0.414, word_ins=2.888, length=2.871, ppl=8.05, wps=132041, ups=2.18, wpb=60623.2, bsz=2228.6, num_updates=129500, lr=8.7875e-05, gnorm=0.638, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:45:49 | INFO | train_inner | epoch 460:    271 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.588, glat_context_p=0.414, word_ins=2.891, length=2.891, ppl=8.06, wps=131830, ups=2.18, wpb=60589.2, bsz=2090.5, num_updates=129600, lr=8.7841e-05, gnorm=0.626, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:45:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:45:57 | INFO | valid | epoch 460 | valid on 'valid' subset | loss 12.456 | nll_loss 11.288 | word_ins 12.213 | length 4.846 | ppl 5618.56 | bleu 31.48 | wps 88118.9 | wpb 21176.3 | bsz 666.3 | num_updates 129611 | best_bleu 32.11
2023-06-14 06:45:57 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:46:10 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint460.pt (epoch 460 @ 129611 updates, score 31.48) (writing took 12.259238056838512 seconds)
2023-06-14 06:46:10 | INFO | fairseq_cli.train | end of epoch 460 (average epoch stats below)
2023-06-14 06:46:10 | INFO | train | epoch 460 | loss 3.008 | nll_loss 1.088 | glat_accu 0.592 | glat_context_p 0.414 | word_ins 2.887 | length 2.879 | ppl 8.04 | wps 113049 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 129611 | lr 8.78373e-05 | gnorm 0.631 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 06:46:10 | INFO | fairseq.trainer | begin training epoch 461
2023-06-14 06:46:56 | INFO | train_inner | epoch 461:     89 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.596, glat_context_p=0.414, word_ins=2.885, length=2.867, ppl=8.03, wps=89155.1, ups=1.48, wpb=60086.6, bsz=2160.1, num_updates=129700, lr=8.78072e-05, gnorm=0.617, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 06:47:43 | INFO | train_inner | epoch 461:    189 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.591, glat_context_p=0.414, word_ins=2.893, length=2.896, ppl=8.08, wps=131013, ups=2.16, wpb=60588.7, bsz=2137.5, num_updates=129800, lr=8.77733e-05, gnorm=0.623, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:48:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:48:28 | INFO | valid | epoch 461 | valid on 'valid' subset | loss 12.451 | nll_loss 11.285 | word_ins 12.211 | length 4.804 | ppl 5600.46 | bleu 31.91 | wps 87128 | wpb 21176.3 | bsz 666.3 | num_updates 129893 | best_bleu 32.11
2023-06-14 06:48:28 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:48:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint461.pt (epoch 461 @ 129893 updates, score 31.91) (writing took 16.145190600305796 seconds)
2023-06-14 06:48:45 | INFO | fairseq_cli.train | end of epoch 461 (average epoch stats below)
2023-06-14 06:48:45 | INFO | train | epoch 461 | loss 3.01 | nll_loss 1.09 | glat_accu 0.595 | glat_context_p 0.413 | word_ins 2.889 | length 2.88 | ppl 8.06 | wps 109728 | ups 1.82 | wpb 60413.8 | bsz 2157.2 | num_updates 129893 | lr 8.77419e-05 | gnorm 0.618 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 06:48:45 | INFO | fairseq.trainer | begin training epoch 462
2023-06-14 06:48:54 | INFO | train_inner | epoch 462:      7 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.596, glat_context_p=0.413, word_ins=2.89, length=2.878, ppl=8.06, wps=84664.4, ups=1.41, wpb=60034.5, bsz=2157, num_updates=129900, lr=8.77396e-05, gnorm=0.619, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:49:39 | INFO | train_inner | epoch 462:    107 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.594, glat_context_p=0.413, word_ins=2.89, length=2.884, ppl=8.06, wps=132411, ups=2.18, wpb=60605.5, bsz=2177.9, num_updates=130000, lr=8.77058e-05, gnorm=0.635, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:50:25 | INFO | train_inner | epoch 462:    207 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.6, glat_context_p=0.413, word_ins=2.89, length=2.882, ppl=8.06, wps=133415, ups=2.2, wpb=60613.3, bsz=2161.8, num_updates=130100, lr=8.76721e-05, gnorm=0.638, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 06:50:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:51:02 | INFO | valid | epoch 462 | valid on 'valid' subset | loss 12.315 | nll_loss 11.124 | word_ins 12.066 | length 4.978 | ppl 5095.82 | bleu 31.83 | wps 88028.1 | wpb 21176.3 | bsz 666.3 | num_updates 130175 | best_bleu 32.11
2023-06-14 06:51:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:51:14 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint462.pt (epoch 462 @ 130175 updates, score 31.83) (writing took 11.890050940215588 seconds)
2023-06-14 06:51:14 | INFO | fairseq_cli.train | end of epoch 462 (average epoch stats below)
2023-06-14 06:51:14 | INFO | train | epoch 462 | loss 3.011 | nll_loss 1.091 | glat_accu 0.595 | glat_context_p 0.413 | word_ins 2.89 | length 2.885 | ppl 8.06 | wps 113995 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 130175 | lr 8.76468e-05 | gnorm 0.632 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 06:51:14 | INFO | fairseq.trainer | begin training epoch 463
2023-06-14 06:51:33 | INFO | train_inner | epoch 463:     25 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.597, glat_context_p=0.413, word_ins=2.888, length=2.881, ppl=8.05, wps=88644.4, ups=1.48, wpb=60069.9, bsz=2154.6, num_updates=130200, lr=8.76384e-05, gnorm=0.623, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 06:52:18 | INFO | train_inner | epoch 463:    125 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.595, glat_context_p=0.413, word_ins=2.896, length=2.894, ppl=8.1, wps=133036, ups=2.2, wpb=60550.5, bsz=2141, num_updates=130300, lr=8.76048e-05, gnorm=0.629, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 06:53:04 | INFO | train_inner | epoch 463:    225 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.602, glat_context_p=0.413, word_ins=2.889, length=2.862, ppl=8.05, wps=133069, ups=2.19, wpb=60710.2, bsz=2175.2, num_updates=130400, lr=8.75712e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:53:30 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:53:33 | INFO | valid | epoch 463 | valid on 'valid' subset | loss 12.347 | nll_loss 11.169 | word_ins 12.105 | length 4.818 | ppl 5209.58 | bleu 32.17 | wps 88459.7 | wpb 21176.3 | bsz 666.3 | num_updates 130457 | best_bleu 32.17
2023-06-14 06:53:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:53:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint463.pt (epoch 463 @ 130457 updates, score 32.17) (writing took 21.71580522507429 seconds)
2023-06-14 06:53:55 | INFO | fairseq_cli.train | end of epoch 463 (average epoch stats below)
2023-06-14 06:53:55 | INFO | train | epoch 463 | loss 3.014 | nll_loss 1.094 | glat_accu 0.599 | glat_context_p 0.413 | word_ins 2.893 | length 2.881 | ppl 8.08 | wps 106093 | ups 1.76 | wpb 60413.8 | bsz 2157.2 | num_updates 130457 | lr 8.7552e-05 | gnorm 0.629 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:53:55 | INFO | fairseq.trainer | begin training epoch 464
2023-06-14 06:54:21 | INFO | train_inner | epoch 464:     43 / 282 loss=3.016, nll_loss=1.096, glat_accu=0.597, glat_context_p=0.413, word_ins=2.895, length=2.885, ppl=8.09, wps=78094.3, ups=1.3, wpb=60141.2, bsz=2136.4, num_updates=130500, lr=8.75376e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:55:06 | INFO | train_inner | epoch 464:    143 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.604, glat_context_p=0.413, word_ins=2.886, length=2.865, ppl=8.04, wps=133167, ups=2.2, wpb=60623.6, bsz=2216, num_updates=130600, lr=8.75041e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:55:52 | INFO | train_inner | epoch 464:    243 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.597, glat_context_p=0.413, word_ins=2.892, length=2.899, ppl=8.08, wps=131511, ups=2.17, wpb=60513.6, bsz=2140, num_updates=130700, lr=8.74706e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:56:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:56:14 | INFO | valid | epoch 464 | valid on 'valid' subset | loss 12.356 | nll_loss 11.175 | word_ins 12.116 | length 4.772 | ppl 5241.53 | bleu 32.12 | wps 88157.3 | wpb 21176.3 | bsz 666.3 | num_updates 130739 | best_bleu 32.17
2023-06-14 06:56:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:56:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint464.pt (epoch 464 @ 130739 updates, score 32.12) (writing took 11.062054142355919 seconds)
2023-06-14 06:56:25 | INFO | fairseq_cli.train | end of epoch 464 (average epoch stats below)
2023-06-14 06:56:25 | INFO | train | epoch 464 | loss 3.012 | nll_loss 1.092 | glat_accu 0.599 | glat_context_p 0.413 | word_ins 2.891 | length 2.883 | ppl 8.07 | wps 113689 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 130739 | lr 8.74576e-05 | gnorm 0.629 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 06:56:25 | INFO | fairseq.trainer | begin training epoch 465
2023-06-14 06:56:59 | INFO | train_inner | epoch 465:     61 / 282 loss=3.019, nll_loss=1.1, glat_accu=0.598, glat_context_p=0.413, word_ins=2.897, length=2.898, ppl=8.11, wps=90041.7, ups=1.5, wpb=60139.3, bsz=2116.4, num_updates=130800, lr=8.74372e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 06:57:45 | INFO | train_inner | epoch 465:    161 / 282 loss=3.019, nll_loss=1.099, glat_accu=0.602, glat_context_p=0.413, word_ins=2.897, length=2.889, ppl=8.11, wps=132170, ups=2.19, wpb=60339, bsz=2158.7, num_updates=130900, lr=8.74038e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:58:30 | INFO | train_inner | epoch 465:    261 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.6, glat_context_p=0.413, word_ins=2.888, length=2.86, ppl=8.05, wps=134042, ups=2.21, wpb=60746.6, bsz=2192, num_updates=131000, lr=8.73704e-05, gnorm=0.62, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:58:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 06:58:43 | INFO | valid | epoch 465 | valid on 'valid' subset | loss 12.389 | nll_loss 11.215 | word_ins 12.149 | length 4.788 | ppl 5363.09 | bleu 32 | wps 84428.5 | wpb 21176.3 | bsz 666.3 | num_updates 131021 | best_bleu 32.17
2023-06-14 06:58:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 06:58:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint465.pt (epoch 465 @ 131021 updates, score 32.0) (writing took 9.340540409088135 seconds)
2023-06-14 06:58:53 | INFO | fairseq_cli.train | end of epoch 465 (average epoch stats below)
2023-06-14 06:58:53 | INFO | train | epoch 465 | loss 3.015 | nll_loss 1.095 | glat_accu 0.6 | glat_context_p 0.413 | word_ins 2.893 | length 2.879 | ppl 8.08 | wps 115261 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 131021 | lr 8.73634e-05 | gnorm 0.623 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 06:58:53 | INFO | fairseq.trainer | begin training epoch 466
2023-06-14 06:59:34 | INFO | train_inner | epoch 466:     79 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.598, glat_context_p=0.413, word_ins=2.892, length=2.896, ppl=8.08, wps=94014.8, ups=1.57, wpb=60016.2, bsz=2098.2, num_updates=131100, lr=8.73371e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 06:59:45 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 16384.0
2023-06-14 07:00:20 | INFO | train_inner | epoch 466:    180 / 282 loss=3.015, nll_loss=1.096, glat_accu=0.605, glat_context_p=0.413, word_ins=2.894, length=2.874, ppl=8.08, wps=130398, ups=2.15, wpb=60528.2, bsz=2177.3, num_updates=131200, lr=8.73038e-05, gnorm=0.626, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:01:06 | INFO | train_inner | epoch 466:    280 / 282 loss=3.02, nll_loss=1.102, glat_accu=0.605, glat_context_p=0.413, word_ins=2.899, length=2.861, ppl=8.11, wps=131817, ups=2.17, wpb=60742.1, bsz=2176.9, num_updates=131300, lr=8.72705e-05, gnorm=0.631, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:01:07 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:01:10 | INFO | valid | epoch 466 | valid on 'valid' subset | loss 12.355 | nll_loss 11.176 | word_ins 12.113 | length 4.824 | ppl 5237.97 | bleu 31.98 | wps 87892.1 | wpb 21176.3 | bsz 666.3 | num_updates 131302 | best_bleu 32.17
2023-06-14 07:01:10 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:01:21 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint466.pt (epoch 466 @ 131302 updates, score 31.98) (writing took 10.895446915179491 seconds)
2023-06-14 07:01:21 | INFO | fairseq_cli.train | end of epoch 466 (average epoch stats below)
2023-06-14 07:01:21 | INFO | train | epoch 466 | loss 3.016 | nll_loss 1.097 | glat_accu 0.603 | glat_context_p 0.413 | word_ins 2.895 | length 2.875 | ppl 8.09 | wps 114161 | ups 1.89 | wpb 60419.8 | bsz 2157.7 | num_updates 131302 | lr 8.72699e-05 | gnorm 0.633 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 07:01:21 | INFO | fairseq.trainer | begin training epoch 467
2023-06-14 07:02:13 | INFO | train_inner | epoch 467:     98 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.598, glat_context_p=0.412, word_ins=2.892, length=2.883, ppl=8.08, wps=90379.9, ups=1.51, wpb=60000.1, bsz=2151.1, num_updates=131400, lr=8.72373e-05, gnorm=0.628, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 07:02:59 | INFO | train_inner | epoch 467:    198 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.605, glat_context_p=0.412, word_ins=2.891, length=2.867, ppl=8.07, wps=132533, ups=2.18, wpb=60739.7, bsz=2180.4, num_updates=131500, lr=8.72041e-05, gnorm=0.614, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:03:37 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:03:40 | INFO | valid | epoch 467 | valid on 'valid' subset | loss 12.41 | nll_loss 11.239 | word_ins 12.172 | length 4.75 | ppl 5441.08 | bleu 31.74 | wps 88594.5 | wpb 21176.3 | bsz 666.3 | num_updates 131584 | best_bleu 32.17
2023-06-14 07:03:40 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:03:53 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint467.pt (epoch 467 @ 131584 updates, score 31.74) (writing took 12.551454931497574 seconds)
2023-06-14 07:03:53 | INFO | fairseq_cli.train | end of epoch 467 (average epoch stats below)
2023-06-14 07:03:53 | INFO | train | epoch 467 | loss 3.012 | nll_loss 1.093 | glat_accu 0.6 | glat_context_p 0.412 | word_ins 2.891 | length 2.878 | ppl 8.07 | wps 112352 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 131584 | lr 8.71763e-05 | gnorm 0.621 | clip 0 | loss_scale 16384 | train_wall 129 | wall 0
2023-06-14 07:03:53 | INFO | fairseq.trainer | begin training epoch 468
2023-06-14 07:04:06 | INFO | train_inner | epoch 468:     16 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.595, glat_context_p=0.412, word_ins=2.89, length=2.888, ppl=8.06, wps=88571.8, ups=1.48, wpb=60047.1, bsz=2116.5, num_updates=131600, lr=8.7171e-05, gnorm=0.631, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:04:52 | INFO | train_inner | epoch 468:    116 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.599, glat_context_p=0.412, word_ins=2.888, length=2.878, ppl=8.05, wps=132242, ups=2.18, wpb=60534.8, bsz=2136.8, num_updates=131700, lr=8.71379e-05, gnorm=0.619, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:05:38 | INFO | train_inner | epoch 468:    216 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.605, glat_context_p=0.412, word_ins=2.891, length=2.863, ppl=8.06, wps=132824, ups=2.19, wpb=60741.1, bsz=2181.7, num_updates=131800, lr=8.71048e-05, gnorm=0.638, clip=0, loss_scale=16384, train_wall=46, wall=0
2023-06-14 07:06:08 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:06:11 | INFO | valid | epoch 468 | valid on 'valid' subset | loss 12.408 | nll_loss 11.232 | word_ins 12.162 | length 4.937 | ppl 5435.54 | bleu 31.7 | wps 83362.4 | wpb 21176.3 | bsz 666.3 | num_updates 131866 | best_bleu 32.17
2023-06-14 07:06:11 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:06:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint468.pt (epoch 468 @ 131866 updates, score 31.7) (writing took 13.540533524006605 seconds)
2023-06-14 07:06:25 | INFO | fairseq_cli.train | end of epoch 468 (average epoch stats below)
2023-06-14 07:06:25 | INFO | train | epoch 468 | loss 3.01 | nll_loss 1.09 | glat_accu 0.601 | glat_context_p 0.412 | word_ins 2.889 | length 2.875 | ppl 8.05 | wps 112279 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 131866 | lr 8.7083e-05 | gnorm 0.631 | clip 0 | loss_scale 16384 | train_wall 128 | wall 0
2023-06-14 07:06:25 | INFO | fairseq.trainer | begin training epoch 469
2023-06-14 07:06:46 | INFO | train_inner | epoch 469:     34 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.6, glat_context_p=0.412, word_ins=2.886, length=2.873, ppl=8.04, wps=87493.9, ups=1.46, wpb=60002.1, bsz=2159.8, num_updates=131900, lr=8.70718e-05, gnorm=0.633, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 07:07:32 | INFO | train_inner | epoch 469:    134 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.594, glat_context_p=0.412, word_ins=2.885, length=2.886, ppl=8.03, wps=133054, ups=2.2, wpb=60575.6, bsz=2162.1, num_updates=132000, lr=8.70388e-05, gnorm=0.63, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 07:08:18 | INFO | train_inner | epoch 469:    234 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.601, glat_context_p=0.412, word_ins=2.887, length=2.872, ppl=8.04, wps=132919, ups=2.19, wpb=60582.9, bsz=2182.9, num_updates=132100, lr=8.70059e-05, gnorm=0.627, clip=0, loss_scale=16384, train_wall=45, wall=0
2023-06-14 07:08:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:08:43 | INFO | valid | epoch 469 | valid on 'valid' subset | loss 12.437 | nll_loss 11.264 | word_ins 12.192 | length 4.895 | ppl 5546.51 | bleu 31.56 | wps 88314.9 | wpb 21176.3 | bsz 666.3 | num_updates 132148 | best_bleu 32.17
2023-06-14 07:08:43 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:08:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint469.pt (epoch 469 @ 132148 updates, score 31.56) (writing took 11.999533545225859 seconds)
2023-06-14 07:08:55 | INFO | fairseq_cli.train | end of epoch 469 (average epoch stats below)
2023-06-14 07:08:55 | INFO | train | epoch 469 | loss 3.008 | nll_loss 1.088 | glat_accu 0.597 | glat_context_p 0.412 | word_ins 2.887 | length 2.88 | ppl 8.05 | wps 113487 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 132148 | lr 8.69901e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:08:55 | INFO | fairseq.trainer | begin training epoch 470
2023-06-14 07:09:25 | INFO | train_inner | epoch 470:     52 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.594, glat_context_p=0.412, word_ins=2.888, length=2.889, ppl=8.05, wps=89382.5, ups=1.48, wpb=60204.1, bsz=2122.5, num_updates=132200, lr=8.6973e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:10:11 | INFO | train_inner | epoch 470:    152 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.595, glat_context_p=0.412, word_ins=2.889, length=2.884, ppl=8.05, wps=131437, ups=2.17, wpb=60563, bsz=2143.7, num_updates=132300, lr=8.69401e-05, gnorm=0.617, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:10:57 | INFO | train_inner | epoch 470:    252 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.597, glat_context_p=0.412, word_ins=2.886, length=2.871, ppl=8.04, wps=132144, ups=2.18, wpb=60625.3, bsz=2186.2, num_updates=132400, lr=8.69072e-05, gnorm=0.62, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:11:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:11:14 | INFO | valid | epoch 470 | valid on 'valid' subset | loss 12.4 | nll_loss 11.235 | word_ins 12.156 | length 4.883 | ppl 5404.24 | bleu 31.58 | wps 87174.3 | wpb 21176.3 | bsz 666.3 | num_updates 132430 | best_bleu 32.17
2023-06-14 07:11:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:11:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint470.pt (epoch 470 @ 132430 updates, score 31.58) (writing took 12.830831483006477 seconds)
2023-06-14 07:11:26 | INFO | fairseq_cli.train | end of epoch 470 (average epoch stats below)
2023-06-14 07:11:26 | INFO | train | epoch 470 | loss 3.007 | nll_loss 1.087 | glat_accu 0.595 | glat_context_p 0.412 | word_ins 2.886 | length 2.879 | ppl 8.04 | wps 112289 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 132430 | lr 8.68974e-05 | gnorm 0.619 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:11:27 | INFO | fairseq.trainer | begin training epoch 471
2023-06-14 07:12:04 | INFO | train_inner | epoch 471:     70 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.593, glat_context_p=0.412, word_ins=2.889, length=2.894, ppl=8.06, wps=88876.1, ups=1.48, wpb=60028.2, bsz=2122.9, num_updates=132500, lr=8.68744e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:12:50 | INFO | train_inner | epoch 471:    170 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.599, glat_context_p=0.412, word_ins=2.883, length=2.876, ppl=8.02, wps=131945, ups=2.18, wpb=60405, bsz=2183.1, num_updates=132600, lr=8.68417e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:13:36 | INFO | train_inner | epoch 471:    270 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.597, glat_context_p=0.412, word_ins=2.888, length=2.87, ppl=8.05, wps=132541, ups=2.18, wpb=60710.3, bsz=2159.2, num_updates=132700, lr=8.6809e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:13:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:13:44 | INFO | valid | epoch 471 | valid on 'valid' subset | loss 12.4 | nll_loss 11.231 | word_ins 12.156 | length 4.858 | ppl 5403.32 | bleu 32.06 | wps 87719.1 | wpb 21176.3 | bsz 666.3 | num_updates 132712 | best_bleu 32.17
2023-06-14 07:13:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:13:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint471.pt (epoch 471 @ 132712 updates, score 32.06) (writing took 12.536866195499897 seconds)
2023-06-14 07:13:57 | INFO | fairseq_cli.train | end of epoch 471 (average epoch stats below)
2023-06-14 07:13:57 | INFO | train | epoch 471 | loss 3.008 | nll_loss 1.088 | glat_accu 0.598 | glat_context_p 0.412 | word_ins 2.887 | length 2.877 | ppl 8.04 | wps 113196 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 132712 | lr 8.6805e-05 | gnorm 0.619 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:13:57 | INFO | fairseq.trainer | begin training epoch 472
2023-06-14 07:14:43 | INFO | train_inner | epoch 472:     88 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.601, glat_context_p=0.412, word_ins=2.887, length=2.871, ppl=8.04, wps=89852.7, ups=1.49, wpb=60178.2, bsz=2179.5, num_updates=132800, lr=8.67763e-05, gnorm=0.644, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:15:29 | INFO | train_inner | epoch 472:    188 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.594, glat_context_p=0.411, word_ins=2.885, length=2.872, ppl=8.03, wps=132318, ups=2.19, wpb=60547.6, bsz=2131.1, num_updates=132900, lr=8.67436e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:16:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:16:15 | INFO | valid | epoch 472 | valid on 'valid' subset | loss 12.429 | nll_loss 11.256 | word_ins 12.184 | length 4.909 | ppl 5515.51 | bleu 31.52 | wps 89064.6 | wpb 21176.3 | bsz 666.3 | num_updates 132994 | best_bleu 32.17
2023-06-14 07:16:15 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:16:28 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint472.pt (epoch 472 @ 132994 updates, score 31.52) (writing took 13.079911265522242 seconds)
2023-06-14 07:16:28 | INFO | fairseq_cli.train | end of epoch 472 (average epoch stats below)
2023-06-14 07:16:28 | INFO | train | epoch 472 | loss 3.006 | nll_loss 1.086 | glat_accu 0.592 | glat_context_p 0.411 | word_ins 2.885 | length 2.876 | ppl 8.03 | wps 112542 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 132994 | lr 8.6713e-05 | gnorm 0.627 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:16:28 | INFO | fairseq.trainer | begin training epoch 473
2023-06-14 07:16:38 | INFO | train_inner | epoch 473:      6 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.583, glat_context_p=0.411, word_ins=2.884, length=2.879, ppl=8.02, wps=86947.8, ups=1.45, wpb=60084.2, bsz=2153.5, num_updates=133000, lr=8.6711e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:17:23 | INFO | train_inner | epoch 473:    106 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.604, glat_context_p=0.411, word_ins=2.892, length=2.868, ppl=8.07, wps=133628, ups=2.2, wpb=60645.5, bsz=2170.3, num_updates=133100, lr=8.66784e-05, gnorm=0.656, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:17:56 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 07:18:09 | INFO | train_inner | epoch 473:    207 / 282 loss=3.015, nll_loss=1.095, glat_accu=0.597, glat_context_p=0.411, word_ins=2.894, length=2.886, ppl=8.08, wps=131222, ups=2.17, wpb=60597.1, bsz=2138.8, num_updates=133200, lr=8.66459e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:18:44 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:18:47 | INFO | valid | epoch 473 | valid on 'valid' subset | loss 12.44 | nll_loss 11.268 | word_ins 12.196 | length 4.891 | ppl 5557.82 | bleu 31.86 | wps 87703.7 | wpb 21176.3 | bsz 666.3 | num_updates 133275 | best_bleu 32.17
2023-06-14 07:18:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:19:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint473.pt (epoch 473 @ 133275 updates, score 31.86) (writing took 12.628061678260565 seconds)
2023-06-14 07:19:00 | INFO | fairseq_cli.train | end of epoch 473 (average epoch stats below)
2023-06-14 07:19:00 | INFO | train | epoch 473 | loss 3.015 | nll_loss 1.095 | glat_accu 0.601 | glat_context_p 0.411 | word_ins 2.893 | length 2.876 | ppl 8.08 | wps 112286 | ups 1.86 | wpb 60415 | bsz 2157.4 | num_updates 133275 | lr 8.66215e-05 | gnorm 0.642 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:19:00 | INFO | fairseq.trainer | begin training epoch 474
2023-06-14 07:19:18 | INFO | train_inner | epoch 474:     25 / 282 loss=3.018, nll_loss=1.098, glat_accu=0.602, glat_context_p=0.411, word_ins=2.896, length=2.887, ppl=8.1, wps=86944.8, ups=1.45, wpb=60047.2, bsz=2150.7, num_updates=133300, lr=8.66134e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:20:04 | INFO | train_inner | epoch 474:    125 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.601, glat_context_p=0.411, word_ins=2.889, length=2.876, ppl=8.06, wps=133235, ups=2.19, wpb=60712.8, bsz=2149.8, num_updates=133400, lr=8.65809e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:20:50 | INFO | train_inner | epoch 474:    225 / 282 loss=3.013, nll_loss=1.094, glat_accu=0.606, glat_context_p=0.411, word_ins=2.892, length=2.868, ppl=8.07, wps=132879, ups=2.19, wpb=60660.6, bsz=2199.7, num_updates=133500, lr=8.65485e-05, gnorm=0.618, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:21:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:21:19 | INFO | valid | epoch 474 | valid on 'valid' subset | loss 12.437 | nll_loss 11.256 | word_ins 12.19 | length 4.944 | ppl 5544.62 | bleu 31.96 | wps 87213.2 | wpb 21176.3 | bsz 666.3 | num_updates 133557 | best_bleu 32.17
2023-06-14 07:21:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:21:30 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint474.pt (epoch 474 @ 133557 updates, score 31.96) (writing took 11.195363473147154 seconds)
2023-06-14 07:21:30 | INFO | fairseq_cli.train | end of epoch 474 (average epoch stats below)
2023-06-14 07:21:30 | INFO | train | epoch 474 | loss 3.013 | nll_loss 1.093 | glat_accu 0.603 | glat_context_p 0.411 | word_ins 2.892 | length 2.878 | ppl 8.07 | wps 112983 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 133557 | lr 8.653e-05 | gnorm 0.625 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:21:30 | INFO | fairseq.trainer | begin training epoch 475
2023-06-14 07:21:57 | INFO | train_inner | epoch 475:     43 / 282 loss=3.013, nll_loss=1.092, glat_accu=0.6, glat_context_p=0.411, word_ins=2.891, length=2.885, ppl=8.07, wps=89239.9, ups=1.49, wpb=59934.2, bsz=2137.3, num_updates=133600, lr=8.65161e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:22:43 | INFO | train_inner | epoch 475:    143 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.595, glat_context_p=0.411, word_ins=2.892, length=2.888, ppl=8.07, wps=132025, ups=2.18, wpb=60664, bsz=2125.9, num_updates=133700, lr=8.64837e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:23:28 | INFO | train_inner | epoch 475:    243 / 282 loss=3.008, nll_loss=1.089, glat_accu=0.607, glat_context_p=0.411, word_ins=2.888, length=2.856, ppl=8.05, wps=134710, ups=2.22, wpb=60587.2, bsz=2188.4, num_updates=133800, lr=8.64514e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:23:45 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:23:49 | INFO | valid | epoch 475 | valid on 'valid' subset | loss 12.428 | nll_loss 11.255 | word_ins 12.185 | length 4.866 | ppl 5509.89 | bleu 32.26 | wps 85433.5 | wpb 21176.3 | bsz 666.3 | num_updates 133839 | best_bleu 32.26
2023-06-14 07:23:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:24:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint475.pt (epoch 475 @ 133839 updates, score 32.26) (writing took 19.25927644968033 seconds)
2023-06-14 07:24:08 | INFO | fairseq_cli.train | end of epoch 475 (average epoch stats below)
2023-06-14 07:24:08 | INFO | train | epoch 475 | loss 3.01 | nll_loss 1.09 | glat_accu 0.599 | glat_context_p 0.411 | word_ins 2.889 | length 2.879 | ppl 8.06 | wps 108056 | ups 1.79 | wpb 60413.8 | bsz 2157.2 | num_updates 133839 | lr 8.64388e-05 | gnorm 0.627 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:24:08 | INFO | fairseq.trainer | begin training epoch 476
2023-06-14 07:24:43 | INFO | train_inner | epoch 476:     61 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.594, glat_context_p=0.411, word_ins=2.883, length=2.887, ppl=8.02, wps=80014.2, ups=1.34, wpb=59930.4, bsz=2161.6, num_updates=133900, lr=8.64191e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:25:29 | INFO | train_inner | epoch 476:    161 / 282 loss=3.003, nll_loss=1.082, glat_accu=0.597, glat_context_p=0.411, word_ins=2.882, length=2.878, ppl=8.02, wps=132122, ups=2.18, wpb=60604.4, bsz=2176.7, num_updates=134000, lr=8.63868e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:26:15 | INFO | train_inner | epoch 476:    261 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.597, glat_context_p=0.411, word_ins=2.89, length=2.864, ppl=8.06, wps=131510, ups=2.17, wpb=60668.2, bsz=2161.3, num_updates=134100, lr=8.63546e-05, gnorm=0.611, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:26:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:26:27 | INFO | valid | epoch 476 | valid on 'valid' subset | loss 12.421 | nll_loss 11.248 | word_ins 12.178 | length 4.874 | ppl 5484.68 | bleu 32.08 | wps 88541.7 | wpb 21176.3 | bsz 666.3 | num_updates 134121 | best_bleu 32.26
2023-06-14 07:26:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:26:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint476.pt (epoch 476 @ 134121 updates, score 32.08) (writing took 12.401722367852926 seconds)
2023-06-14 07:26:40 | INFO | fairseq_cli.train | end of epoch 476 (average epoch stats below)
2023-06-14 07:26:40 | INFO | train | epoch 476 | loss 3.006 | nll_loss 1.086 | glat_accu 0.596 | glat_context_p 0.411 | word_ins 2.885 | length 2.874 | ppl 8.03 | wps 112229 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 134121 | lr 8.63479e-05 | gnorm 0.618 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:26:40 | INFO | fairseq.trainer | begin training epoch 477
2023-06-14 07:27:21 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 07:27:23 | INFO | train_inner | epoch 477:     80 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.6, glat_context_p=0.411, word_ins=2.886, length=2.875, ppl=8.04, wps=87851.7, ups=1.46, wpb=60019.9, bsz=2165.5, num_updates=134200, lr=8.63224e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:28:08 | INFO | train_inner | epoch 477:    180 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.599, glat_context_p=0.411, word_ins=2.888, length=2.875, ppl=8.05, wps=133479, ups=2.2, wpb=60711.8, bsz=2156.7, num_updates=134300, lr=8.62903e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:28:54 | INFO | train_inner | epoch 477:    280 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.59, glat_context_p=0.41, word_ins=2.885, length=2.882, ppl=8.03, wps=132011, ups=2.18, wpb=60489.1, bsz=2129, num_updates=134400, lr=8.62582e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:28:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:28:58 | INFO | valid | epoch 477 | valid on 'valid' subset | loss 12.35 | nll_loss 11.175 | word_ins 12.108 | length 4.861 | ppl 5220.21 | bleu 31.92 | wps 87185.5 | wpb 21176.3 | bsz 666.3 | num_updates 134402 | best_bleu 32.26
2023-06-14 07:28:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:29:08 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint477.pt (epoch 477 @ 134402 updates, score 31.92) (writing took 9.798144724220037 seconds)
2023-06-14 07:29:08 | INFO | fairseq_cli.train | end of epoch 477 (average epoch stats below)
2023-06-14 07:29:08 | INFO | train | epoch 477 | loss 3.007 | nll_loss 1.087 | glat_accu 0.597 | glat_context_p 0.41 | word_ins 2.886 | length 2.875 | ppl 8.04 | wps 114599 | ups 1.9 | wpb 60409.9 | bsz 2156.8 | num_updates 134402 | lr 8.62576e-05 | gnorm 0.632 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:29:08 | INFO | fairseq.trainer | begin training epoch 478
2023-06-14 07:30:00 | INFO | train_inner | epoch 478:     98 / 282 loss=3.001, nll_loss=1.081, glat_accu=0.6, glat_context_p=0.41, word_ins=2.881, length=2.864, ppl=8.01, wps=91977.7, ups=1.53, wpb=60014.6, bsz=2160.9, num_updates=134500, lr=8.62261e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:30:46 | INFO | train_inner | epoch 478:    198 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.587, glat_context_p=0.41, word_ins=2.886, length=2.888, ppl=8.04, wps=131375, ups=2.16, wpb=60726.2, bsz=2149.6, num_updates=134600, lr=8.61941e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:31:24 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:31:27 | INFO | valid | epoch 478 | valid on 'valid' subset | loss 12.344 | nll_loss 11.171 | word_ins 12.104 | length 4.809 | ppl 5199.85 | bleu 32.14 | wps 87636.4 | wpb 21176.3 | bsz 666.3 | num_updates 134684 | best_bleu 32.26
2023-06-14 07:31:27 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:31:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint478.pt (epoch 478 @ 134684 updates, score 32.14) (writing took 9.726986173540354 seconds)
2023-06-14 07:31:37 | INFO | fairseq_cli.train | end of epoch 478 (average epoch stats below)
2023-06-14 07:31:37 | INFO | train | epoch 478 | loss 3.006 | nll_loss 1.086 | glat_accu 0.594 | glat_context_p 0.41 | word_ins 2.885 | length 2.878 | ppl 8.03 | wps 114615 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 134684 | lr 8.61672e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:31:37 | INFO | fairseq.trainer | begin training epoch 479
2023-06-14 07:31:50 | INFO | train_inner | epoch 479:     16 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.596, glat_context_p=0.41, word_ins=2.887, length=2.879, ppl=8.05, wps=93435, ups=1.56, wpb=60016.5, bsz=2151, num_updates=134700, lr=8.61621e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:32:35 | INFO | train_inner | epoch 479:    116 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.604, glat_context_p=0.41, word_ins=2.883, length=2.85, ppl=8.02, wps=133383, ups=2.2, wpb=60571.8, bsz=2212.2, num_updates=134800, lr=8.61301e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:33:22 | INFO | train_inner | epoch 479:    216 / 282 loss=3.013, nll_loss=1.092, glat_accu=0.587, glat_context_p=0.41, word_ins=2.891, length=2.905, ppl=8.07, wps=130840, ups=2.16, wpb=60560.9, bsz=2089.8, num_updates=134900, lr=8.60982e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:33:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:33:55 | INFO | valid | epoch 479 | valid on 'valid' subset | loss 12.343 | nll_loss 11.159 | word_ins 12.099 | length 4.901 | ppl 5195.44 | bleu 32.12 | wps 87865.7 | wpb 21176.3 | bsz 666.3 | num_updates 134966 | best_bleu 32.26
2023-06-14 07:33:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:34:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint479.pt (epoch 479 @ 134966 updates, score 32.12) (writing took 9.366536244750023 seconds)
2023-06-14 07:34:04 | INFO | fairseq_cli.train | end of epoch 479 (average epoch stats below)
2023-06-14 07:34:04 | INFO | train | epoch 479 | loss 3.008 | nll_loss 1.088 | glat_accu 0.598 | glat_context_p 0.41 | word_ins 2.887 | length 2.875 | ppl 8.04 | wps 115508 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 134966 | lr 8.60771e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:34:04 | INFO | fairseq.trainer | begin training epoch 480
2023-06-14 07:34:25 | INFO | train_inner | epoch 480:     34 / 282 loss=3.009, nll_loss=1.09, glat_accu=0.603, glat_context_p=0.41, word_ins=2.888, length=2.87, ppl=8.05, wps=95678.4, ups=1.59, wpb=60292.1, bsz=2170, num_updates=135000, lr=8.60663e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:35:11 | INFO | train_inner | epoch 480:    134 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.606, glat_context_p=0.41, word_ins=2.888, length=2.86, ppl=8.05, wps=131644, ups=2.17, wpb=60587.4, bsz=2197.1, num_updates=135100, lr=8.60344e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:35:56 | INFO | train_inner | epoch 480:    234 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.602, glat_context_p=0.41, word_ins=2.892, length=2.874, ppl=8.08, wps=132288, ups=2.19, wpb=60490, bsz=2153.8, num_updates=135200, lr=8.60026e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:36:06 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 07:36:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:36:22 | INFO | valid | epoch 480 | valid on 'valid' subset | loss 12.361 | nll_loss 11.18 | word_ins 12.115 | length 4.899 | ppl 5260.25 | bleu 32.06 | wps 88285.4 | wpb 21176.3 | bsz 666.3 | num_updates 135247 | best_bleu 32.26
2023-06-14 07:36:22 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:36:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint480.pt (epoch 480 @ 135247 updates, score 32.06) (writing took 11.937042735517025 seconds)
2023-06-14 07:36:34 | INFO | fairseq_cli.train | end of epoch 480 (average epoch stats below)
2023-06-14 07:36:34 | INFO | train | epoch 480 | loss 3.013 | nll_loss 1.093 | glat_accu 0.603 | glat_context_p 0.41 | word_ins 2.891 | length 2.876 | ppl 8.07 | wps 113340 | ups 1.88 | wpb 60416.9 | bsz 2159.1 | num_updates 135247 | lr 8.59877e-05 | gnorm 0.638 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:36:34 | INFO | fairseq.trainer | begin training epoch 481
2023-06-14 07:37:05 | INFO | train_inner | epoch 481:     53 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.608, glat_context_p=0.41, word_ins=2.893, length=2.877, ppl=8.08, wps=88203.6, ups=1.47, wpb=60165.1, bsz=2143.4, num_updates=135300, lr=8.59708e-05, gnorm=0.654, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:37:51 | INFO | train_inner | epoch 481:    153 / 282 loss=3.017, nll_loss=1.098, glat_accu=0.605, glat_context_p=0.41, word_ins=2.896, length=2.868, ppl=8.1, wps=132339, ups=2.18, wpb=60764.5, bsz=2160.1, num_updates=135400, lr=8.59391e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:38:37 | INFO | train_inner | epoch 481:    253 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.595, glat_context_p=0.41, word_ins=2.889, length=2.887, ppl=8.05, wps=130188, ups=2.16, wpb=60401.1, bsz=2124.8, num_updates=135500, lr=8.59074e-05, gnorm=0.616, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:38:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:38:53 | INFO | valid | epoch 481 | valid on 'valid' subset | loss 12.34 | nll_loss 11.156 | word_ins 12.095 | length 4.881 | ppl 5183.98 | bleu 32.06 | wps 87650.8 | wpb 21176.3 | bsz 666.3 | num_updates 135529 | best_bleu 32.26
2023-06-14 07:38:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:39:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint481.pt (epoch 481 @ 135529 updates, score 32.06) (writing took 13.021161407232285 seconds)
2023-06-14 07:39:06 | INFO | fairseq_cli.train | end of epoch 481 (average epoch stats below)
2023-06-14 07:39:06 | INFO | train | epoch 481 | loss 3.012 | nll_loss 1.093 | glat_accu 0.603 | glat_context_p 0.41 | word_ins 2.891 | length 2.869 | ppl 8.07 | wps 111656 | ups 1.85 | wpb 60413.8 | bsz 2157.2 | num_updates 135529 | lr 8.58982e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:39:07 | INFO | fairseq.trainer | begin training epoch 482
2023-06-14 07:39:46 | INFO | train_inner | epoch 482:     71 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.607, glat_context_p=0.41, word_ins=2.89, length=2.866, ppl=8.07, wps=87515.6, ups=1.46, wpb=59976.1, bsz=2148.5, num_updates=135600, lr=8.58757e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:40:31 | INFO | train_inner | epoch 482:    171 / 282 loss=3.011, nll_loss=1.092, glat_accu=0.6, glat_context_p=0.41, word_ins=2.89, length=2.875, ppl=8.06, wps=132433, ups=2.19, wpb=60539.1, bsz=2162.4, num_updates=135700, lr=8.5844e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:41:17 | INFO | train_inner | epoch 482:    271 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.6, glat_context_p=0.41, word_ins=2.886, length=2.865, ppl=8.04, wps=131681, ups=2.17, wpb=60667.8, bsz=2191.9, num_updates=135800, lr=8.58124e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:41:22 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:41:26 | INFO | valid | epoch 482 | valid on 'valid' subset | loss 12.401 | nll_loss 11.228 | word_ins 12.158 | length 4.871 | ppl 5408.83 | bleu 31.48 | wps 88826.9 | wpb 21176.3 | bsz 666.3 | num_updates 135811 | best_bleu 32.26
2023-06-14 07:41:26 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:41:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint482.pt (epoch 482 @ 135811 updates, score 31.48) (writing took 14.064434695988894 seconds)
2023-06-14 07:41:40 | INFO | fairseq_cli.train | end of epoch 482 (average epoch stats below)
2023-06-14 07:41:40 | INFO | train | epoch 482 | loss 3.01 | nll_loss 1.09 | glat_accu 0.602 | glat_context_p 0.41 | word_ins 2.889 | length 2.872 | ppl 8.06 | wps 111054 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 135811 | lr 8.58089e-05 | gnorm 0.634 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:41:40 | INFO | fairseq.trainer | begin training epoch 483
2023-06-14 07:42:28 | INFO | train_inner | epoch 483:     89 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.602, glat_context_p=0.409, word_ins=2.886, length=2.874, ppl=8.04, wps=85198.7, ups=1.42, wpb=59905.3, bsz=2150.2, num_updates=135900, lr=8.57808e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:43:13 | INFO | train_inner | epoch 483:    189 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.604, glat_context_p=0.409, word_ins=2.889, length=2.863, ppl=8.06, wps=132785, ups=2.19, wpb=60667.5, bsz=2146, num_updates=136000, lr=8.57493e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:43:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:43:59 | INFO | valid | epoch 483 | valid on 'valid' subset | loss 12.362 | nll_loss 11.185 | word_ins 12.121 | length 4.813 | ppl 5265.62 | bleu 31.73 | wps 87747.5 | wpb 21176.3 | bsz 666.3 | num_updates 136093 | best_bleu 32.26
2023-06-14 07:43:59 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:44:10 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint483.pt (epoch 483 @ 136093 updates, score 31.73) (writing took 11.539259403944016 seconds)
2023-06-14 07:44:10 | INFO | fairseq_cli.train | end of epoch 483 (average epoch stats below)
2023-06-14 07:44:10 | INFO | train | epoch 483 | loss 3.008 | nll_loss 1.088 | glat_accu 0.6 | glat_context_p 0.409 | word_ins 2.887 | length 2.873 | ppl 8.04 | wps 113230 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 136093 | lr 8.572e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:44:10 | INFO | fairseq.trainer | begin training epoch 484
2023-06-14 07:44:20 | INFO | train_inner | epoch 484:      7 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.595, glat_context_p=0.409, word_ins=2.885, length=2.885, ppl=8.03, wps=90656.5, ups=1.51, wpb=60225.3, bsz=2163.4, num_updates=136100, lr=8.57178e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:45:06 | INFO | train_inner | epoch 484:    107 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.6, glat_context_p=0.409, word_ins=2.883, length=2.863, ppl=8.02, wps=132110, ups=2.18, wpb=60591.6, bsz=2170.9, num_updates=136200, lr=8.56863e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:45:26 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 07:45:52 | INFO | train_inner | epoch 484:    208 / 282 loss=3.014, nll_loss=1.094, glat_accu=0.598, glat_context_p=0.409, word_ins=2.893, length=2.887, ppl=8.08, wps=130802, ups=2.16, wpb=60519.6, bsz=2149.5, num_updates=136300, lr=8.56549e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:46:26 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:46:30 | INFO | valid | epoch 484 | valid on 'valid' subset | loss 12.411 | nll_loss 11.244 | word_ins 12.174 | length 4.754 | ppl 5445.7 | bleu 31.8 | wps 86769.4 | wpb 21176.3 | bsz 666.3 | num_updates 136374 | best_bleu 32.26
2023-06-14 07:46:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:46:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint484.pt (epoch 484 @ 136374 updates, score 31.8) (writing took 7.295797944068909 seconds)
2023-06-14 07:46:37 | INFO | fairseq_cli.train | end of epoch 484 (average epoch stats below)
2023-06-14 07:46:37 | INFO | train | epoch 484 | loss 3.01 | nll_loss 1.09 | glat_accu 0.599 | glat_context_p 0.409 | word_ins 2.889 | length 2.879 | ppl 8.06 | wps 115407 | ups 1.91 | wpb 60414.7 | bsz 2156.4 | num_updates 136374 | lr 8.56316e-05 | gnorm 0.634 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:46:38 | INFO | fairseq.trainer | begin training epoch 485
2023-06-14 07:46:54 | INFO | train_inner | epoch 485:     26 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.601, glat_context_p=0.409, word_ins=2.888, length=2.877, ppl=8.05, wps=96252.2, ups=1.6, wpb=60237.6, bsz=2141.2, num_updates=136400, lr=8.56235e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:47:40 | INFO | train_inner | epoch 485:    126 / 282 loss=3.007, nll_loss=1.086, glat_accu=0.588, glat_context_p=0.409, word_ins=2.886, length=2.889, ppl=8.04, wps=132007, ups=2.18, wpb=60610.9, bsz=2122.2, num_updates=136500, lr=8.55921e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:48:27 | INFO | train_inner | epoch 485:    226 / 282 loss=3.006, nll_loss=1.087, glat_accu=0.6, glat_context_p=0.409, word_ins=2.885, length=2.857, ppl=8.03, wps=130808, ups=2.16, wpb=60485, bsz=2203.8, num_updates=136600, lr=8.55608e-05, gnorm=0.606, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:48:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:48:55 | INFO | valid | epoch 485 | valid on 'valid' subset | loss 12.412 | nll_loss 11.246 | word_ins 12.171 | length 4.808 | ppl 5451.25 | bleu 31.81 | wps 86918.4 | wpb 21176.3 | bsz 666.3 | num_updates 136656 | best_bleu 32.26
2023-06-14 07:48:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:49:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint485.pt (epoch 485 @ 136656 updates, score 31.81) (writing took 10.221658419817686 seconds)
2023-06-14 07:49:05 | INFO | fairseq_cli.train | end of epoch 485 (average epoch stats below)
2023-06-14 07:49:05 | INFO | train | epoch 485 | loss 3.006 | nll_loss 1.086 | glat_accu 0.599 | glat_context_p 0.409 | word_ins 2.885 | length 2.87 | ppl 8.03 | wps 115155 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 136656 | lr 8.55432e-05 | gnorm 0.631 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:49:06 | INFO | fairseq.trainer | begin training epoch 486
2023-06-14 07:49:31 | INFO | train_inner | epoch 486:     44 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.612, glat_context_p=0.409, word_ins=2.885, length=2.856, ppl=8.03, wps=92919.4, ups=1.54, wpb=60172.7, bsz=2174.8, num_updates=136700, lr=8.55295e-05, gnorm=0.658, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:50:17 | INFO | train_inner | epoch 486:    144 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.591, glat_context_p=0.409, word_ins=2.886, length=2.892, ppl=8.04, wps=131553, ups=2.17, wpb=60553.4, bsz=2113, num_updates=136800, lr=8.54982e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:51:03 | INFO | train_inner | epoch 486:    244 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.596, glat_context_p=0.409, word_ins=2.888, length=2.878, ppl=8.05, wps=132980, ups=2.19, wpb=60594.8, bsz=2160.2, num_updates=136900, lr=8.5467e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:51:20 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:51:24 | INFO | valid | epoch 486 | valid on 'valid' subset | loss 12.513 | nll_loss 11.354 | word_ins 12.274 | length 4.783 | ppl 5843.38 | bleu 31.78 | wps 87030.4 | wpb 21176.3 | bsz 666.3 | num_updates 136938 | best_bleu 32.26
2023-06-14 07:51:24 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:51:33 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint486.pt (epoch 486 @ 136938 updates, score 31.78) (writing took 9.775698445737362 seconds)
2023-06-14 07:51:33 | INFO | fairseq_cli.train | end of epoch 486 (average epoch stats below)
2023-06-14 07:51:33 | INFO | train | epoch 486 | loss 3.005 | nll_loss 1.085 | glat_accu 0.596 | glat_context_p 0.409 | word_ins 2.885 | length 2.872 | ppl 8.03 | wps 115160 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 136938 | lr 8.54551e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:51:33 | INFO | fairseq.trainer | begin training epoch 487
2023-06-14 07:52:08 | INFO | train_inner | epoch 487:     62 / 282 loss=3, nll_loss=1.079, glat_accu=0.594, glat_context_p=0.409, word_ins=2.879, length=2.868, ppl=8, wps=91803.9, ups=1.53, wpb=59885.9, bsz=2170.2, num_updates=137000, lr=8.54358e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:52:54 | INFO | train_inner | epoch 487:    162 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.598, glat_context_p=0.409, word_ins=2.882, length=2.863, ppl=8.01, wps=132025, ups=2.17, wpb=60716.2, bsz=2157.8, num_updates=137100, lr=8.54046e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:53:40 | INFO | train_inner | epoch 487:    262 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.598, glat_context_p=0.409, word_ins=2.887, length=2.872, ppl=8.04, wps=132133, ups=2.18, wpb=60578.6, bsz=2173.2, num_updates=137200, lr=8.53735e-05, gnorm=0.621, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:53:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:53:52 | INFO | valid | epoch 487 | valid on 'valid' subset | loss 12.405 | nll_loss 11.236 | word_ins 12.165 | length 4.798 | ppl 5421.71 | bleu 31.75 | wps 88355.8 | wpb 21176.3 | bsz 666.3 | num_updates 137220 | best_bleu 32.26
2023-06-14 07:53:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:54:03 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint487.pt (epoch 487 @ 137220 updates, score 31.75) (writing took 10.203705944120884 seconds)
2023-06-14 07:54:03 | INFO | fairseq_cli.train | end of epoch 487 (average epoch stats below)
2023-06-14 07:54:03 | INFO | train | epoch 487 | loss 3.005 | nll_loss 1.084 | glat_accu 0.598 | glat_context_p 0.409 | word_ins 2.884 | length 2.87 | ppl 8.03 | wps 114200 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 137220 | lr 8.53673e-05 | gnorm 0.623 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:54:03 | INFO | fairseq.trainer | begin training epoch 488
2023-06-14 07:54:31 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 07:54:46 | INFO | train_inner | epoch 488:     81 / 282 loss=3.008, nll_loss=1.087, glat_accu=0.594, glat_context_p=0.409, word_ins=2.886, length=2.883, ppl=8.04, wps=91204.8, ups=1.52, wpb=60064.2, bsz=2090.9, num_updates=137300, lr=8.53424e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:55:32 | INFO | train_inner | epoch 488:    181 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.602, glat_context_p=0.408, word_ins=2.888, length=2.872, ppl=8.05, wps=132623, ups=2.19, wpb=60634.7, bsz=2190.8, num_updates=137400, lr=8.53113e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:56:18 | INFO | train_inner | epoch 488:    281 / 282 loss=3.005, nll_loss=1.086, glat_accu=0.603, glat_context_p=0.408, word_ins=2.885, length=2.855, ppl=8.03, wps=131903, ups=2.18, wpb=60614, bsz=2182, num_updates=137500, lr=8.52803e-05, gnorm=0.614, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:56:18 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:56:21 | INFO | valid | epoch 488 | valid on 'valid' subset | loss 12.336 | nll_loss 11.15 | word_ins 12.088 | length 4.959 | ppl 5169.04 | bleu 31.87 | wps 87590 | wpb 21176.3 | bsz 666.3 | num_updates 137501 | best_bleu 32.26
2023-06-14 07:56:21 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:56:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint488.pt (epoch 488 @ 137501 updates, score 31.87) (writing took 10.145896878093481 seconds)
2023-06-14 07:56:31 | INFO | fairseq_cli.train | end of epoch 488 (average epoch stats below)
2023-06-14 07:56:31 | INFO | train | epoch 488 | loss 3.007 | nll_loss 1.088 | glat_accu 0.6 | glat_context_p 0.408 | word_ins 2.886 | length 2.87 | ppl 8.04 | wps 114152 | ups 1.89 | wpb 60411.3 | bsz 2156.2 | num_updates 137501 | lr 8.528e-05 | gnorm 0.627 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 07:56:31 | INFO | fairseq.trainer | begin training epoch 489
2023-06-14 07:57:23 | INFO | train_inner | epoch 489:     99 / 282 loss=3.007, nll_loss=1.086, glat_accu=0.603, glat_context_p=0.408, word_ins=2.885, length=2.872, ppl=8.04, wps=92375.8, ups=1.54, wpb=60138.3, bsz=2156.7, num_updates=137600, lr=8.52493e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:58:09 | INFO | train_inner | epoch 489:    199 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.598, glat_context_p=0.408, word_ins=2.89, length=2.867, ppl=8.06, wps=130649, ups=2.16, wpb=60587.1, bsz=2160, num_updates=137700, lr=8.52183e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 07:58:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 07:58:50 | INFO | valid | epoch 489 | valid on 'valid' subset | loss 12.371 | nll_loss 11.203 | word_ins 12.134 | length 4.766 | ppl 5298.81 | bleu 32.12 | wps 86478.7 | wpb 21176.3 | bsz 666.3 | num_updates 137783 | best_bleu 32.26
2023-06-14 07:58:50 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 07:58:59 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint489.pt (epoch 489 @ 137783 updates, score 32.12) (writing took 9.193798296153545 seconds)
2023-06-14 07:58:59 | INFO | fairseq_cli.train | end of epoch 489 (average epoch stats below)
2023-06-14 07:58:59 | INFO | train | epoch 489 | loss 3.008 | nll_loss 1.088 | glat_accu 0.599 | glat_context_p 0.408 | word_ins 2.887 | length 2.87 | ppl 8.04 | wps 115480 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 137783 | lr 8.51927e-05 | gnorm 0.629 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 07:58:59 | INFO | fairseq.trainer | begin training epoch 490
2023-06-14 07:59:12 | INFO | train_inner | epoch 490:     17 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.599, glat_context_p=0.408, word_ins=2.882, length=2.86, ppl=8.02, wps=95977.9, ups=1.6, wpb=60065.9, bsz=2156.1, num_updates=137800, lr=8.51874e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 07:59:57 | INFO | train_inner | epoch 490:    117 / 282 loss=3.004, nll_loss=1.085, glat_accu=0.6, glat_context_p=0.408, word_ins=2.884, length=2.855, ppl=8.02, wps=132663, ups=2.19, wpb=60618.2, bsz=2178.6, num_updates=137900, lr=8.51565e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:00:43 | INFO | train_inner | epoch 490:    217 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.59, glat_context_p=0.408, word_ins=2.881, length=2.885, ppl=8.01, wps=132494, ups=2.19, wpb=60625.8, bsz=2136, num_updates=138000, lr=8.51257e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:01:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:01:16 | INFO | valid | epoch 490 | valid on 'valid' subset | loss 12.332 | nll_loss 11.149 | word_ins 12.086 | length 4.898 | ppl 5156.76 | bleu 31.96 | wps 87555.8 | wpb 21176.3 | bsz 666.3 | num_updates 138065 | best_bleu 32.26
2023-06-14 08:01:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:01:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint490.pt (epoch 490 @ 138065 updates, score 31.96) (writing took 10.663199063390493 seconds)
2023-06-14 08:01:27 | INFO | fairseq_cli.train | end of epoch 490 (average epoch stats below)
2023-06-14 08:01:27 | INFO | train | epoch 490 | loss 3.004 | nll_loss 1.084 | glat_accu 0.596 | glat_context_p 0.408 | word_ins 2.883 | length 2.871 | ppl 8.02 | wps 115167 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 138065 | lr 8.51056e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:01:27 | INFO | fairseq.trainer | begin training epoch 491
2023-06-14 08:01:49 | INFO | train_inner | epoch 491:     35 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.598, glat_context_p=0.408, word_ins=2.886, length=2.88, ppl=8.04, wps=91398.9, ups=1.52, wpb=60015.7, bsz=2148.1, num_updates=138100, lr=8.50948e-05, gnorm=0.648, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:02:34 | INFO | train_inner | epoch 491:    135 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.598, glat_context_p=0.408, word_ins=2.884, length=2.875, ppl=8.03, wps=132781, ups=2.19, wpb=60639.8, bsz=2148.6, num_updates=138200, lr=8.5064e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:03:17 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:03:21 | INFO | train_inner | epoch 491:    236 / 282 loss=3, nll_loss=1.08, glat_accu=0.597, glat_context_p=0.408, word_ins=2.88, length=2.86, ppl=8, wps=131398, ups=2.17, wpb=60583.7, bsz=2187.6, num_updates=138300, lr=8.50333e-05, gnorm=0.613, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:03:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:03:45 | INFO | valid | epoch 491 | valid on 'valid' subset | loss 12.334 | nll_loss 11.158 | word_ins 12.095 | length 4.774 | ppl 5162.02 | bleu 32.12 | wps 87753.5 | wpb 21176.3 | bsz 666.3 | num_updates 138346 | best_bleu 32.26
2023-06-14 08:03:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:03:58 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint491.pt (epoch 491 @ 138346 updates, score 32.12) (writing took 12.15241775661707 seconds)
2023-06-14 08:03:58 | INFO | fairseq_cli.train | end of epoch 491 (average epoch stats below)
2023-06-14 08:03:58 | INFO | train | epoch 491 | loss 3.004 | nll_loss 1.084 | glat_accu 0.599 | glat_context_p 0.408 | word_ins 2.884 | length 2.869 | ppl 8.02 | wps 112534 | ups 1.86 | wpb 60416.4 | bsz 2157.5 | num_updates 138346 | lr 8.50191e-05 | gnorm 0.628 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:03:58 | INFO | fairseq.trainer | begin training epoch 492
2023-06-14 08:04:28 | INFO | train_inner | epoch 492:     54 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.598, glat_context_p=0.408, word_ins=2.888, length=2.874, ppl=8.05, wps=89322.2, ups=1.49, wpb=60075.2, bsz=2144.2, num_updates=138400, lr=8.50026e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:05:14 | INFO | train_inner | epoch 492:    154 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.594, glat_context_p=0.408, word_ins=2.884, length=2.877, ppl=8.03, wps=129910, ups=2.15, wpb=60479.4, bsz=2145.6, num_updates=138500, lr=8.49719e-05, gnorm=0.623, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:06:00 | INFO | train_inner | epoch 492:    254 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.606, glat_context_p=0.408, word_ins=2.885, length=2.869, ppl=8.04, wps=133362, ups=2.2, wpb=60563, bsz=2173.2, num_updates=138600, lr=8.49412e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:06:12 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:06:16 | INFO | valid | epoch 492 | valid on 'valid' subset | loss 12.551 | nll_loss 11.393 | word_ins 12.308 | length 4.847 | ppl 5999.24 | bleu 32.13 | wps 87261.8 | wpb 21176.3 | bsz 666.3 | num_updates 138628 | best_bleu 32.26
2023-06-14 08:06:16 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:06:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint492.pt (epoch 492 @ 138628 updates, score 32.13) (writing took 10.605086371302605 seconds)
2023-06-14 08:06:26 | INFO | fairseq_cli.train | end of epoch 492 (average epoch stats below)
2023-06-14 08:06:26 | INFO | train | epoch 492 | loss 3.006 | nll_loss 1.086 | glat_accu 0.599 | glat_context_p 0.408 | word_ins 2.885 | length 2.87 | ppl 8.03 | wps 114599 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 138628 | lr 8.49326e-05 | gnorm 0.632 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:06:26 | INFO | fairseq.trainer | begin training epoch 493
2023-06-14 08:07:06 | INFO | train_inner | epoch 493:     72 / 282 loss=3.002, nll_loss=1.081, glat_accu=0.597, glat_context_p=0.408, word_ins=2.881, length=2.868, ppl=8.01, wps=91608.4, ups=1.52, wpb=60216.6, bsz=2124.3, num_updates=138700, lr=8.49106e-05, gnorm=0.644, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:07:51 | INFO | train_inner | epoch 493:    172 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.602, glat_context_p=0.408, word_ins=2.881, length=2.853, ppl=8.01, wps=132593, ups=2.19, wpb=60573.2, bsz=2188.3, num_updates=138800, lr=8.488e-05, gnorm=0.624, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:08:37 | INFO | train_inner | epoch 493:    272 / 282 loss=3.012, nll_loss=1.093, glat_accu=0.595, glat_context_p=0.407, word_ins=2.891, length=2.881, ppl=8.07, wps=131477, ups=2.17, wpb=60596, bsz=2144.9, num_updates=138900, lr=8.48494e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:08:42 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:08:45 | INFO | valid | epoch 493 | valid on 'valid' subset | loss 12.463 | nll_loss 11.297 | word_ins 12.219 | length 4.873 | ppl 5646.31 | bleu 31.97 | wps 87326.8 | wpb 21176.3 | bsz 666.3 | num_updates 138910 | best_bleu 32.26
2023-06-14 08:08:45 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:08:57 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint493.pt (epoch 493 @ 138910 updates, score 31.97) (writing took 12.466237261891365 seconds)
2023-06-14 08:08:57 | INFO | fairseq_cli.train | end of epoch 493 (average epoch stats below)
2023-06-14 08:08:57 | INFO | train | epoch 493 | loss 3.005 | nll_loss 1.085 | glat_accu 0.599 | glat_context_p 0.407 | word_ins 2.884 | length 2.869 | ppl 8.03 | wps 112774 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 138910 | lr 8.48464e-05 | gnorm 0.634 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:08:57 | INFO | fairseq.trainer | begin training epoch 494
2023-06-14 08:09:45 | INFO | train_inner | epoch 494:     90 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.61, glat_context_p=0.407, word_ins=2.882, length=2.854, ppl=8.02, wps=88889.1, ups=1.48, wpb=59926.7, bsz=2180.9, num_updates=139000, lr=8.48189e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:10:31 | INFO | train_inner | epoch 494:    190 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.602, glat_context_p=0.407, word_ins=2.89, length=2.877, ppl=8.06, wps=131143, ups=2.16, wpb=60760.8, bsz=2124.6, num_updates=139100, lr=8.47884e-05, gnorm=0.658, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:11:13 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:11:17 | INFO | valid | epoch 494 | valid on 'valid' subset | loss 12.327 | nll_loss 11.141 | word_ins 12.08 | length 4.927 | ppl 5139.27 | bleu 32.11 | wps 88020.1 | wpb 21176.3 | bsz 666.3 | num_updates 139192 | best_bleu 32.26
2023-06-14 08:11:17 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:11:26 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint494.pt (epoch 494 @ 139192 updates, score 32.11) (writing took 9.510385662317276 seconds)
2023-06-14 08:11:26 | INFO | fairseq_cli.train | end of epoch 494 (average epoch stats below)
2023-06-14 08:11:26 | INFO | train | epoch 494 | loss 3.009 | nll_loss 1.089 | glat_accu 0.604 | glat_context_p 0.407 | word_ins 2.888 | length 2.868 | ppl 8.05 | wps 114498 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 139192 | lr 8.47604e-05 | gnorm 0.641 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:11:26 | INFO | fairseq.trainer | begin training epoch 495
2023-06-14 08:11:35 | INFO | train_inner | epoch 495:      8 / 282 loss=3.013, nll_loss=1.093, glat_accu=0.601, glat_context_p=0.407, word_ins=2.892, length=2.878, ppl=8.07, wps=93177.7, ups=1.55, wpb=60040.5, bsz=2156.8, num_updates=139200, lr=8.47579e-05, gnorm=0.644, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:12:22 | INFO | train_inner | epoch 495:    108 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.605, glat_context_p=0.407, word_ins=2.893, length=2.87, ppl=8.08, wps=131109, ups=2.16, wpb=60630.2, bsz=2149.6, num_updates=139300, lr=8.47275e-05, gnorm=0.65, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:12:30 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:13:08 | INFO | train_inner | epoch 495:    209 / 282 loss=3.008, nll_loss=1.089, glat_accu=0.605, glat_context_p=0.407, word_ins=2.888, length=2.854, ppl=8.05, wps=132126, ups=2.18, wpb=60733.6, bsz=2180.6, num_updates=139400, lr=8.46971e-05, gnorm=0.649, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:13:40 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:13:44 | INFO | valid | epoch 495 | valid on 'valid' subset | loss 12.362 | nll_loss 11.181 | word_ins 12.12 | length 4.849 | ppl 5264.72 | bleu 32.22 | wps 87997.4 | wpb 21176.3 | bsz 666.3 | num_updates 139473 | best_bleu 32.26
2023-06-14 08:13:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:13:55 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint495.pt (epoch 495 @ 139473 updates, score 32.22) (writing took 11.405584841966629 seconds)
2023-06-14 08:13:55 | INFO | fairseq_cli.train | end of epoch 495 (average epoch stats below)
2023-06-14 08:13:55 | INFO | train | epoch 495 | loss 3.011 | nll_loss 1.092 | glat_accu 0.605 | glat_context_p 0.407 | word_ins 2.89 | length 2.868 | ppl 8.06 | wps 113935 | ups 1.89 | wpb 60420 | bsz 2158.2 | num_updates 139473 | lr 8.46749e-05 | gnorm 0.649 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:13:55 | INFO | fairseq.trainer | begin training epoch 496
2023-06-14 08:14:14 | INFO | train_inner | epoch 496:     27 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.6, glat_context_p=0.407, word_ins=2.89, length=2.882, ppl=8.06, wps=90842.9, ups=1.52, wpb=59955.7, bsz=2121, num_updates=139500, lr=8.46668e-05, gnorm=0.64, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:14:59 | INFO | train_inner | epoch 496:    127 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.605, glat_context_p=0.407, word_ins=2.881, length=2.866, ppl=8.01, wps=133510, ups=2.2, wpb=60618.7, bsz=2152, num_updates=139600, lr=8.46364e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:15:45 | INFO | train_inner | epoch 496:    227 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.607, glat_context_p=0.407, word_ins=2.888, length=2.872, ppl=8.06, wps=132557, ups=2.19, wpb=60596.9, bsz=2180.1, num_updates=139700, lr=8.46061e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:16:10 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:16:13 | INFO | valid | epoch 496 | valid on 'valid' subset | loss 12.356 | nll_loss 11.165 | word_ins 12.102 | length 5.078 | ppl 5243.31 | bleu 31.88 | wps 86947.9 | wpb 21176.3 | bsz 666.3 | num_updates 139755 | best_bleu 32.26
2023-06-14 08:16:13 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:16:25 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint496.pt (epoch 496 @ 139755 updates, score 31.88) (writing took 12.077694442123175 seconds)
2023-06-14 08:16:25 | INFO | fairseq_cli.train | end of epoch 496 (average epoch stats below)
2023-06-14 08:16:25 | INFO | train | epoch 496 | loss 3.008 | nll_loss 1.087 | glat_accu 0.604 | glat_context_p 0.407 | word_ins 2.886 | length 2.869 | ppl 8.04 | wps 113348 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 139755 | lr 8.45895e-05 | gnorm 0.634 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:16:26 | INFO | fairseq.trainer | begin training epoch 497
2023-06-14 08:16:52 | INFO | train_inner | epoch 497:     45 / 282 loss=3.009, nll_loss=1.09, glat_accu=0.599, glat_context_p=0.407, word_ins=2.888, length=2.864, ppl=8.05, wps=89339.9, ups=1.48, wpb=60169.8, bsz=2130.3, num_updates=139800, lr=8.45759e-05, gnorm=0.652, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:17:38 | INFO | train_inner | epoch 497:    145 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.606, glat_context_p=0.407, word_ins=2.883, length=2.855, ppl=8.02, wps=131251, ups=2.17, wpb=60505.7, bsz=2219.5, num_updates=139900, lr=8.45456e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:18:24 | INFO | train_inner | epoch 497:    245 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.598, glat_context_p=0.407, word_ins=2.884, length=2.876, ppl=8.03, wps=133688, ups=2.21, wpb=60538.6, bsz=2156.1, num_updates=140000, lr=8.45154e-05, gnorm=0.648, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:18:41 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:18:44 | INFO | valid | epoch 497 | valid on 'valid' subset | loss 12.314 | nll_loss 11.131 | word_ins 12.071 | length 4.854 | ppl 5091.49 | bleu 32.07 | wps 87849.6 | wpb 21176.3 | bsz 666.3 | num_updates 140037 | best_bleu 32.26
2023-06-14 08:18:44 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:18:56 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint497.pt (epoch 497 @ 140037 updates, score 32.07) (writing took 12.414989341050386 seconds)
2023-06-14 08:18:56 | INFO | fairseq_cli.train | end of epoch 497 (average epoch stats below)
2023-06-14 08:18:56 | INFO | train | epoch 497 | loss 3.005 | nll_loss 1.085 | glat_accu 0.601 | glat_context_p 0.407 | word_ins 2.884 | length 2.865 | ppl 8.03 | wps 112958 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 140037 | lr 8.45043e-05 | gnorm 0.644 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:18:56 | INFO | fairseq.trainer | begin training epoch 498
2023-06-14 08:19:31 | INFO | train_inner | epoch 498:     63 / 282 loss=3.004, nll_loss=1.085, glat_accu=0.61, glat_context_p=0.407, word_ins=2.884, length=2.852, ppl=8.02, wps=89562.9, ups=1.49, wpb=60175.2, bsz=2157.9, num_updates=140100, lr=8.44853e-05, gnorm=0.64, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:20:16 | INFO | train_inner | epoch 498:    163 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.602, glat_context_p=0.407, word_ins=2.883, length=2.863, ppl=8.02, wps=133190, ups=2.19, wpb=60760.6, bsz=2171.4, num_updates=140200, lr=8.44551e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:21:03 | INFO | train_inner | epoch 498:    263 / 282 loss=3.012, nll_loss=1.091, glat_accu=0.602, glat_context_p=0.407, word_ins=2.89, length=2.89, ppl=8.07, wps=130245, ups=2.16, wpb=60315.2, bsz=2129.2, num_updates=140300, lr=8.4425e-05, gnorm=0.653, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:21:11 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:21:14 | INFO | valid | epoch 498 | valid on 'valid' subset | loss 12.446 | nll_loss 11.275 | word_ins 12.202 | length 4.877 | ppl 5578.63 | bleu 31.89 | wps 88127.4 | wpb 21176.3 | bsz 666.3 | num_updates 140319 | best_bleu 32.26
2023-06-14 08:21:14 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:21:27 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint498.pt (epoch 498 @ 140319 updates, score 31.89) (writing took 12.534192685037851 seconds)
2023-06-14 08:21:27 | INFO | fairseq_cli.train | end of epoch 498 (average epoch stats below)
2023-06-14 08:21:27 | INFO | train | epoch 498 | loss 3.007 | nll_loss 1.086 | glat_accu 0.604 | glat_context_p 0.407 | word_ins 2.885 | length 2.87 | ppl 8.04 | wps 112992 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 140319 | lr 8.44193e-05 | gnorm 0.643 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:21:27 | INFO | fairseq.trainer | begin training epoch 499
2023-06-14 08:21:44 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:22:11 | INFO | train_inner | epoch 499:     82 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.603, glat_context_p=0.406, word_ins=2.883, length=2.869, ppl=8.02, wps=87396.7, ups=1.45, wpb=60166.9, bsz=2144.2, num_updates=140400, lr=8.43949e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:22:57 | INFO | train_inner | epoch 499:    182 / 282 loss=3.01, nll_loss=1.091, glat_accu=0.606, glat_context_p=0.406, word_ins=2.889, length=2.859, ppl=8.06, wps=132268, ups=2.18, wpb=60625.1, bsz=2191.4, num_updates=140500, lr=8.43649e-05, gnorm=0.645, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:23:43 | INFO | train_inner | epoch 499:    282 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.596, glat_context_p=0.406, word_ins=2.885, length=2.873, ppl=8.03, wps=130465, ups=2.17, wpb=60055, bsz=2123.4, num_updates=140600, lr=8.43349e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:23:43 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:23:47 | INFO | valid | epoch 499 | valid on 'valid' subset | loss 12.418 | nll_loss 11.241 | word_ins 12.171 | length 4.958 | ppl 5473.52 | bleu 32.11 | wps 87454.5 | wpb 21176.3 | bsz 666.3 | num_updates 140600 | best_bleu 32.26
2023-06-14 08:23:47 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:24:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint499.pt (epoch 499 @ 140600 updates, score 32.11) (writing took 12.866613358259201 seconds)
2023-06-14 08:24:00 | INFO | fairseq_cli.train | end of epoch 499 (average epoch stats below)
2023-06-14 08:24:00 | INFO | train | epoch 499 | loss 3.007 | nll_loss 1.087 | glat_accu 0.602 | glat_context_p 0.406 | word_ins 2.886 | length 2.866 | ppl 8.04 | wps 111287 | ups 1.84 | wpb 60410.9 | bsz 2158.6 | num_updates 140600 | lr 8.43349e-05 | gnorm 0.637 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:24:00 | INFO | fairseq.trainer | begin training epoch 500
2023-06-14 08:24:51 | INFO | train_inner | epoch 500:    100 / 282 loss=3.006, nll_loss=1.085, glat_accu=0.607, glat_context_p=0.406, word_ins=2.884, length=2.871, ppl=8.03, wps=89195.5, ups=1.47, wpb=60476.6, bsz=2133, num_updates=140700, lr=8.43049e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:25:37 | INFO | train_inner | epoch 500:    200 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.603, glat_context_p=0.406, word_ins=2.886, length=2.869, ppl=8.04, wps=132087, ups=2.18, wpb=60656.8, bsz=2182.1, num_updates=140800, lr=8.4275e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:26:15 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:26:19 | INFO | valid | epoch 500 | valid on 'valid' subset | loss 12.354 | nll_loss 11.174 | word_ins 12.114 | length 4.796 | ppl 5233.52 | bleu 32.17 | wps 87530 | wpb 21176.3 | bsz 666.3 | num_updates 140882 | best_bleu 32.26
2023-06-14 08:26:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:26:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint500.pt (epoch 500 @ 140882 updates, score 32.17) (writing took 12.015366360545158 seconds)
2023-06-14 08:26:31 | INFO | fairseq_cli.train | end of epoch 500 (average epoch stats below)
2023-06-14 08:26:31 | INFO | train | epoch 500 | loss 3.007 | nll_loss 1.087 | glat_accu 0.604 | glat_context_p 0.406 | word_ins 2.886 | length 2.867 | ppl 8.04 | wps 112783 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 140882 | lr 8.42505e-05 | gnorm 0.635 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:26:31 | INFO | fairseq.trainer | begin training epoch 501
2023-06-14 08:26:45 | INFO | train_inner | epoch 501:     18 / 282 loss=3.008, nll_loss=1.089, glat_accu=0.602, glat_context_p=0.406, word_ins=2.888, length=2.857, ppl=8.05, wps=87871.7, ups=1.46, wpb=60087.6, bsz=2152.6, num_updates=140900, lr=8.42451e-05, gnorm=0.648, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:27:32 | INFO | train_inner | epoch 501:    118 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.606, glat_context_p=0.406, word_ins=2.882, length=2.862, ppl=8.02, wps=130529, ups=2.15, wpb=60640.6, bsz=2205.4, num_updates=141000, lr=8.42152e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:28:17 | INFO | train_inner | epoch 501:    218 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.607, glat_context_p=0.406, word_ins=2.885, length=2.866, ppl=8.04, wps=133135, ups=2.2, wpb=60520.9, bsz=2148.7, num_updates=141100, lr=8.41853e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:28:46 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:28:49 | INFO | valid | epoch 501 | valid on 'valid' subset | loss 12.379 | nll_loss 11.206 | word_ins 12.137 | length 4.849 | ppl 5325.88 | bleu 32.1 | wps 87549.4 | wpb 21176.3 | bsz 666.3 | num_updates 141164 | best_bleu 32.26
2023-06-14 08:28:49 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:29:00 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint501.pt (epoch 501 @ 141164 updates, score 32.1) (writing took 11.015498124063015 seconds)
2023-06-14 08:29:00 | INFO | fairseq_cli.train | end of epoch 501 (average epoch stats below)
2023-06-14 08:29:00 | INFO | train | epoch 501 | loss 3.006 | nll_loss 1.086 | glat_accu 0.604 | glat_context_p 0.406 | word_ins 2.885 | length 2.869 | ppl 8.03 | wps 113726 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 141164 | lr 8.41663e-05 | gnorm 0.636 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:29:01 | INFO | fairseq.trainer | begin training epoch 502
2023-06-14 08:29:24 | INFO | train_inner | epoch 502:     36 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.598, glat_context_p=0.406, word_ins=2.887, length=2.888, ppl=8.05, wps=90220.8, ups=1.5, wpb=60060.8, bsz=2105.9, num_updates=141200, lr=8.41555e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:30:09 | INFO | train_inner | epoch 502:    136 / 282 loss=2.996, nll_loss=1.075, glat_accu=0.607, glat_context_p=0.406, word_ins=2.875, length=2.845, ppl=7.98, wps=133734, ups=2.2, wpb=60658.4, bsz=2229.5, num_updates=141300, lr=8.41257e-05, gnorm=0.626, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:30:39 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:30:55 | INFO | train_inner | epoch 502:    237 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.601, glat_context_p=0.406, word_ins=2.89, length=2.874, ppl=8.06, wps=132091, ups=2.18, wpb=60600.6, bsz=2139.7, num_updates=141400, lr=8.4096e-05, gnorm=0.64, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:31:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:31:19 | INFO | valid | epoch 502 | valid on 'valid' subset | loss 12.429 | nll_loss 11.255 | word_ins 12.183 | length 4.917 | ppl 5512.7 | bleu 32.28 | wps 86503.3 | wpb 21176.3 | bsz 666.3 | num_updates 141445 | best_bleu 32.28
2023-06-14 08:31:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:31:37 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint502.pt (epoch 502 @ 141445 updates, score 32.28) (writing took 18.229257866740227 seconds)
2023-06-14 08:31:37 | INFO | fairseq_cli.train | end of epoch 502 (average epoch stats below)
2023-06-14 08:31:37 | INFO | train | epoch 502 | loss 3.005 | nll_loss 1.085 | glat_accu 0.603 | glat_context_p 0.406 | word_ins 2.884 | length 2.865 | ppl 8.03 | wps 108358 | ups 1.79 | wpb 60410.2 | bsz 2159.5 | num_updates 141445 | lr 8.40826e-05 | gnorm 0.635 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:31:37 | INFO | fairseq.trainer | begin training epoch 503
2023-06-14 08:32:08 | INFO | train_inner | epoch 503:     55 / 282 loss=3.005, nll_loss=1.084, glat_accu=0.606, glat_context_p=0.406, word_ins=2.883, length=2.865, ppl=8.03, wps=82385.5, ups=1.37, wpb=60048.5, bsz=2144.9, num_updates=141500, lr=8.40663e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:32:54 | INFO | train_inner | epoch 503:    155 / 282 loss=2.998, nll_loss=1.078, glat_accu=0.597, glat_context_p=0.406, word_ins=2.878, length=2.858, ppl=7.99, wps=132096, ups=2.18, wpb=60604.3, bsz=2186.2, num_updates=141600, lr=8.40366e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:33:40 | INFO | train_inner | epoch 503:    255 / 282 loss=3.006, nll_loss=1.087, glat_accu=0.603, glat_context_p=0.406, word_ins=2.885, length=2.864, ppl=8.04, wps=132505, ups=2.19, wpb=60616.3, bsz=2156.9, num_updates=141700, lr=8.40069e-05, gnorm=0.619, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:33:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:33:55 | INFO | valid | epoch 503 | valid on 'valid' subset | loss 12.387 | nll_loss 11.211 | word_ins 12.141 | length 4.915 | ppl 5354.9 | bleu 31.89 | wps 88148.1 | wpb 21176.3 | bsz 666.3 | num_updates 141727 | best_bleu 32.28
2023-06-14 08:33:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:34:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint503.pt (epoch 503 @ 141727 updates, score 31.89) (writing took 11.110174238681793 seconds)
2023-06-14 08:34:06 | INFO | fairseq_cli.train | end of epoch 503 (average epoch stats below)
2023-06-14 08:34:06 | INFO | train | epoch 503 | loss 3.003 | nll_loss 1.083 | glat_accu 0.601 | glat_context_p 0.406 | word_ins 2.882 | length 2.866 | ppl 8.02 | wps 114228 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 141727 | lr 8.39989e-05 | gnorm 0.625 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:34:06 | INFO | fairseq.trainer | begin training epoch 504
2023-06-14 08:34:46 | INFO | train_inner | epoch 504:     73 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.6, glat_context_p=0.406, word_ins=2.884, length=2.873, ppl=8.03, wps=90690.1, ups=1.51, wpb=59979.8, bsz=2112.7, num_updates=141800, lr=8.39773e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:35:32 | INFO | train_inner | epoch 504:    173 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.608, glat_context_p=0.405, word_ins=2.884, length=2.859, ppl=8.03, wps=131649, ups=2.17, wpb=60544.2, bsz=2201.7, num_updates=141900, lr=8.39477e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:36:17 | INFO | train_inner | epoch 504:    273 / 282 loss=3.005, nll_loss=1.084, glat_accu=0.599, glat_context_p=0.405, word_ins=2.884, length=2.87, ppl=8.03, wps=133371, ups=2.2, wpb=60759.4, bsz=2146.2, num_updates=142000, lr=8.39181e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:36:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:36:25 | INFO | valid | epoch 504 | valid on 'valid' subset | loss 12.313 | nll_loss 11.132 | word_ins 12.07 | length 4.879 | ppl 5089.76 | bleu 31.88 | wps 88261.3 | wpb 21176.3 | bsz 666.3 | num_updates 142009 | best_bleu 32.28
2023-06-14 08:36:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:36:36 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint504.pt (epoch 504 @ 142009 updates, score 31.88) (writing took 11.512179974466562 seconds)
2023-06-14 08:36:36 | INFO | fairseq_cli.train | end of epoch 504 (average epoch stats below)
2023-06-14 08:36:36 | INFO | train | epoch 504 | loss 3.004 | nll_loss 1.084 | glat_accu 0.603 | glat_context_p 0.405 | word_ins 2.883 | length 2.865 | ppl 8.02 | wps 113575 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 142009 | lr 8.39155e-05 | gnorm 0.634 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:36:36 | INFO | fairseq.trainer | begin training epoch 505
2023-06-14 08:37:25 | INFO | train_inner | epoch 505:     91 / 282 loss=3.003, nll_loss=1.082, glat_accu=0.596, glat_context_p=0.405, word_ins=2.882, length=2.874, ppl=8.01, wps=89021.1, ups=1.48, wpb=60119.8, bsz=2120.7, num_updates=142100, lr=8.38886e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:38:11 | INFO | train_inner | epoch 505:    191 / 282 loss=3.003, nll_loss=1.082, glat_accu=0.605, glat_context_p=0.405, word_ins=2.882, length=2.861, ppl=8.02, wps=131533, ups=2.17, wpb=60626.9, bsz=2155, num_updates=142200, lr=8.38591e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:38:52 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:38:55 | INFO | valid | epoch 505 | valid on 'valid' subset | loss 12.42 | nll_loss 11.255 | word_ins 12.182 | length 4.758 | ppl 5480.96 | bleu 32.01 | wps 86598.9 | wpb 21176.3 | bsz 666.3 | num_updates 142291 | best_bleu 32.28
2023-06-14 08:38:55 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:39:06 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint505.pt (epoch 505 @ 142291 updates, score 32.01) (writing took 10.577951792627573 seconds)
2023-06-14 08:39:06 | INFO | fairseq_cli.train | end of epoch 505 (average epoch stats below)
2023-06-14 08:39:06 | INFO | train | epoch 505 | loss 3.005 | nll_loss 1.084 | glat_accu 0.602 | glat_context_p 0.405 | word_ins 2.884 | length 2.866 | ppl 8.03 | wps 114018 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 142291 | lr 8.38323e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:39:06 | INFO | fairseq.trainer | begin training epoch 506
2023-06-14 08:39:16 | INFO | train_inner | epoch 506:      9 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.606, glat_context_p=0.405, word_ins=2.887, length=2.859, ppl=8.05, wps=92562.6, ups=1.54, wpb=60089.5, bsz=2173.7, num_updates=142300, lr=8.38296e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:39:57 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:40:03 | INFO | train_inner | epoch 506:    110 / 282 loss=3.003, nll_loss=1.081, glat_accu=0.604, glat_context_p=0.405, word_ins=2.881, length=2.878, ppl=8.02, wps=129805, ups=2.15, wpb=60457.3, bsz=2129.2, num_updates=142400, lr=8.38002e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:40:48 | INFO | train_inner | epoch 506:    210 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.605, glat_context_p=0.405, word_ins=2.886, length=2.857, ppl=8.04, wps=132705, ups=2.19, wpb=60702.8, bsz=2165.2, num_updates=142500, lr=8.37708e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:41:21 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:41:25 | INFO | valid | epoch 506 | valid on 'valid' subset | loss 12.364 | nll_loss 11.18 | word_ins 12.114 | length 4.992 | ppl 5270.99 | bleu 32.05 | wps 88291.3 | wpb 21176.3 | bsz 666.3 | num_updates 142572 | best_bleu 32.28
2023-06-14 08:41:25 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:41:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint506.pt (epoch 506 @ 142572 updates, score 32.05) (writing took 9.24275952577591 seconds)
2023-06-14 08:41:34 | INFO | fairseq_cli.train | end of epoch 506 (average epoch stats below)
2023-06-14 08:41:34 | INFO | train | epoch 506 | loss 3.006 | nll_loss 1.086 | glat_accu 0.605 | glat_context_p 0.405 | word_ins 2.885 | length 2.863 | ppl 8.03 | wps 114192 | ups 1.89 | wpb 60412.9 | bsz 2155 | num_updates 142572 | lr 8.37496e-05 | gnorm 0.63 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:41:34 | INFO | fairseq.trainer | begin training epoch 507
2023-06-14 08:41:52 | INFO | train_inner | epoch 507:     28 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.603, glat_context_p=0.405, word_ins=2.888, length=2.867, ppl=8.05, wps=93670.7, ups=1.56, wpb=60034.7, bsz=2156.7, num_updates=142600, lr=8.37414e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:42:38 | INFO | train_inner | epoch 507:    128 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.603, glat_context_p=0.405, word_ins=2.885, length=2.875, ppl=8.04, wps=132142, ups=2.18, wpb=60592.7, bsz=2158.8, num_updates=142700, lr=8.37121e-05, gnorm=0.629, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:43:24 | INFO | train_inner | epoch 507:    228 / 282 loss=2.999, nll_loss=1.078, glat_accu=0.605, glat_context_p=0.405, word_ins=2.878, length=2.856, ppl=7.99, wps=130812, ups=2.16, wpb=60489.2, bsz=2160.2, num_updates=142800, lr=8.36827e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:43:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:43:52 | INFO | valid | epoch 507 | valid on 'valid' subset | loss 12.372 | nll_loss 11.194 | word_ins 12.128 | length 4.882 | ppl 5300.61 | bleu 32.09 | wps 87743.9 | wpb 21176.3 | bsz 666.3 | num_updates 142854 | best_bleu 32.28
2023-06-14 08:43:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:44:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint507.pt (epoch 507 @ 142854 updates, score 32.09) (writing took 11.524858374148607 seconds)
2023-06-14 08:44:04 | INFO | fairseq_cli.train | end of epoch 507 (average epoch stats below)
2023-06-14 08:44:04 | INFO | train | epoch 507 | loss 3.003 | nll_loss 1.083 | glat_accu 0.604 | glat_context_p 0.405 | word_ins 2.882 | length 2.864 | ppl 8.02 | wps 113835 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 142854 | lr 8.36669e-05 | gnorm 0.631 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:44:04 | INFO | fairseq.trainer | begin training epoch 508
2023-06-14 08:44:32 | INFO | train_inner | epoch 508:     46 / 282 loss=3.006, nll_loss=1.087, glat_accu=0.603, glat_context_p=0.405, word_ins=2.885, length=2.861, ppl=8.03, wps=89511.5, ups=1.49, wpb=60150.5, bsz=2148.3, num_updates=142900, lr=8.36535e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:45:18 | INFO | train_inner | epoch 508:    146 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.591, glat_context_p=0.405, word_ins=2.888, length=2.891, ppl=8.05, wps=130186, ups=2.15, wpb=60571.1, bsz=2101.1, num_updates=143000, lr=8.36242e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:46:04 | INFO | train_inner | epoch 508:    246 / 282 loss=3, nll_loss=1.08, glat_accu=0.612, glat_context_p=0.405, word_ins=2.879, length=2.848, ppl=8, wps=133472, ups=2.2, wpb=60644.1, bsz=2197.8, num_updates=143100, lr=8.3595e-05, gnorm=0.642, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:46:20 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:46:23 | INFO | valid | epoch 508 | valid on 'valid' subset | loss 12.446 | nll_loss 11.27 | word_ins 12.2 | length 4.91 | ppl 5578.63 | bleu 32.16 | wps 87339.6 | wpb 21176.3 | bsz 666.3 | num_updates 143136 | best_bleu 32.28
2023-06-14 08:46:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:46:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint508.pt (epoch 508 @ 143136 updates, score 32.16) (writing took 10.720394298434258 seconds)
2023-06-14 08:46:34 | INFO | fairseq_cli.train | end of epoch 508 (average epoch stats below)
2023-06-14 08:46:34 | INFO | train | epoch 508 | loss 3.005 | nll_loss 1.084 | glat_accu 0.603 | glat_context_p 0.405 | word_ins 2.884 | length 2.866 | ppl 8.03 | wps 113702 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 143136 | lr 8.35845e-05 | gnorm 0.64 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:46:34 | INFO | fairseq.trainer | begin training epoch 509
2023-06-14 08:47:09 | INFO | train_inner | epoch 509:     64 / 282 loss=3.002, nll_loss=1.081, glat_accu=0.607, glat_context_p=0.405, word_ins=2.881, length=2.862, ppl=8.01, wps=91361.5, ups=1.52, wpb=60040.5, bsz=2174.8, num_updates=143200, lr=8.35658e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:47:55 | INFO | train_inner | epoch 509:    164 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.611, glat_context_p=0.405, word_ins=2.889, length=2.86, ppl=8.06, wps=132701, ups=2.19, wpb=60637.2, bsz=2185.8, num_updates=143300, lr=8.35366e-05, gnorm=0.649, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:48:41 | INFO | train_inner | epoch 509:    264 / 282 loss=3.012, nll_loss=1.092, glat_accu=0.608, glat_context_p=0.404, word_ins=2.89, length=2.858, ppl=8.06, wps=130830, ups=2.16, wpb=60614.4, bsz=2152.9, num_updates=143400, lr=8.35075e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:48:47 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:48:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:48:53 | INFO | valid | epoch 509 | valid on 'valid' subset | loss 12.391 | nll_loss 11.213 | word_ins 12.143 | length 4.962 | ppl 5369.47 | bleu 32.28 | wps 88897.3 | wpb 21176.3 | bsz 666.3 | num_updates 143417 | best_bleu 32.28
2023-06-14 08:48:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:49:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint509.pt (epoch 509 @ 143417 updates, score 32.28) (writing took 18.339538659900427 seconds)
2023-06-14 08:49:11 | INFO | fairseq_cli.train | end of epoch 509 (average epoch stats below)
2023-06-14 08:49:11 | INFO | train | epoch 509 | loss 3.009 | nll_loss 1.089 | glat_accu 0.608 | glat_context_p 0.404 | word_ins 2.888 | length 2.863 | ppl 8.05 | wps 108028 | ups 1.79 | wpb 60421.2 | bsz 2157.6 | num_updates 143417 | lr 8.35025e-05 | gnorm 0.64 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:49:11 | INFO | fairseq.trainer | begin training epoch 510
2023-06-14 08:49:55 | INFO | train_inner | epoch 510:     83 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.611, glat_context_p=0.404, word_ins=2.886, length=2.866, ppl=8.04, wps=81417, ups=1.35, wpb=60158.8, bsz=2150.8, num_updates=143500, lr=8.34784e-05, gnorm=0.642, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:50:42 | INFO | train_inner | epoch 510:    183 / 282 loss=3.014, nll_loss=1.095, glat_accu=0.605, glat_context_p=0.404, word_ins=2.893, length=2.873, ppl=8.08, wps=129333, ups=2.13, wpb=60585.1, bsz=2140.2, num_updates=143600, lr=8.34493e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-14 08:51:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:51:30 | INFO | valid | epoch 510 | valid on 'valid' subset | loss 12.389 | nll_loss 11.204 | word_ins 12.139 | length 5.006 | ppl 5364.92 | bleu 32.22 | wps 86770.9 | wpb 21176.3 | bsz 666.3 | num_updates 143699 | best_bleu 32.28
2023-06-14 08:51:30 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:51:40 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint510.pt (epoch 510 @ 143699 updates, score 32.22) (writing took 9.439800065010786 seconds)
2023-06-14 08:51:40 | INFO | fairseq_cli.train | end of epoch 510 (average epoch stats below)
2023-06-14 08:51:40 | INFO | train | epoch 510 | loss 3.009 | nll_loss 1.089 | glat_accu 0.609 | glat_context_p 0.404 | word_ins 2.888 | length 2.866 | ppl 8.05 | wps 114560 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 143699 | lr 8.34206e-05 | gnorm 0.639 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:51:40 | INFO | fairseq.trainer | begin training epoch 511
2023-06-14 08:51:46 | INFO | train_inner | epoch 511:      1 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.612, glat_context_p=0.404, word_ins=2.885, length=2.86, ppl=8.04, wps=94102.6, ups=1.57, wpb=60046.7, bsz=2155.9, num_updates=143700, lr=8.34203e-05, gnorm=0.649, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:52:32 | INFO | train_inner | epoch 511:    101 / 282 loss=3.005, nll_loss=1.084, glat_accu=0.61, glat_context_p=0.404, word_ins=2.883, length=2.867, ppl=8.03, wps=130893, ups=2.16, wpb=60534.5, bsz=2156.6, num_updates=143800, lr=8.33913e-05, gnorm=0.633, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:53:18 | INFO | train_inner | epoch 511:    201 / 282 loss=3.007, nll_loss=1.088, glat_accu=0.608, glat_context_p=0.404, word_ins=2.886, length=2.848, ppl=8.04, wps=133352, ups=2.2, wpb=60656.5, bsz=2181.4, num_updates=143900, lr=8.33623e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:53:55 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:53:58 | INFO | valid | epoch 511 | valid on 'valid' subset | loss 12.299 | nll_loss 11.118 | word_ins 12.054 | length 4.9 | ppl 5039.87 | bleu 32.12 | wps 88312.3 | wpb 21176.3 | bsz 666.3 | num_updates 143981 | best_bleu 32.28
2023-06-14 08:53:58 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:54:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint511.pt (epoch 511 @ 143981 updates, score 32.12) (writing took 12.881628260016441 seconds)
2023-06-14 08:54:11 | INFO | fairseq_cli.train | end of epoch 511 (average epoch stats below)
2023-06-14 08:54:11 | INFO | train | epoch 511 | loss 3.009 | nll_loss 1.089 | glat_accu 0.608 | glat_context_p 0.404 | word_ins 2.888 | length 2.863 | ppl 8.05 | wps 112799 | ups 1.87 | wpb 60413.8 | bsz 2157.2 | num_updates 143981 | lr 8.33388e-05 | gnorm 0.637 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:54:11 | INFO | fairseq.trainer | begin training epoch 512
2023-06-14 08:54:26 | INFO | train_inner | epoch 512:     19 / 282 loss=3.016, nll_loss=1.097, glat_accu=0.606, glat_context_p=0.404, word_ins=2.894, length=2.87, ppl=8.09, wps=88454.3, ups=1.47, wpb=60110.2, bsz=2140.9, num_updates=144000, lr=8.33333e-05, gnorm=0.65, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:55:12 | INFO | train_inner | epoch 512:    119 / 282 loss=3.011, nll_loss=1.091, glat_accu=0.598, glat_context_p=0.404, word_ins=2.89, length=2.882, ppl=8.06, wps=131760, ups=2.18, wpb=60566.3, bsz=2098.6, num_updates=144100, lr=8.33044e-05, gnorm=0.643, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:55:57 | INFO | train_inner | epoch 512:    219 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.612, glat_context_p=0.404, word_ins=2.884, length=2.843, ppl=8.03, wps=133120, ups=2.2, wpb=60555.3, bsz=2195.8, num_updates=144200, lr=8.32755e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:56:25 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:56:29 | INFO | valid | epoch 512 | valid on 'valid' subset | loss 12.314 | nll_loss 11.129 | word_ins 12.068 | length 4.918 | ppl 5092.36 | bleu 32.33 | wps 88580.9 | wpb 21176.3 | bsz 666.3 | num_updates 144263 | best_bleu 32.33
2023-06-14 08:56:29 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:56:42 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint512.pt (epoch 512 @ 144263 updates, score 32.33) (writing took 13.55830393731594 seconds)
2023-06-14 08:56:42 | INFO | fairseq_cli.train | end of epoch 512 (average epoch stats below)
2023-06-14 08:56:42 | INFO | train | epoch 512 | loss 3.007 | nll_loss 1.088 | glat_accu 0.607 | glat_context_p 0.404 | word_ins 2.886 | length 2.861 | ppl 8.04 | wps 112495 | ups 1.86 | wpb 60413.8 | bsz 2157.2 | num_updates 144263 | lr 8.32573e-05 | gnorm 0.645 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 08:56:42 | INFO | fairseq.trainer | begin training epoch 513
2023-06-14 08:57:05 | INFO | train_inner | epoch 513:     37 / 282 loss=3.006, nll_loss=1.085, glat_accu=0.613, glat_context_p=0.404, word_ins=2.884, length=2.859, ppl=8.03, wps=87865.4, ups=1.46, wpb=60085.1, bsz=2182.6, num_updates=144300, lr=8.32467e-05, gnorm=0.645, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 08:57:51 | INFO | train_inner | epoch 513:    137 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.607, glat_context_p=0.404, word_ins=2.885, length=2.866, ppl=8.03, wps=132537, ups=2.18, wpb=60669, bsz=2147.1, num_updates=144400, lr=8.32178e-05, gnorm=0.648, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 08:58:08 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 08:58:38 | INFO | train_inner | epoch 513:    238 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.608, glat_context_p=0.404, word_ins=2.887, length=2.87, ppl=8.05, wps=129283, ups=2.14, wpb=60522.2, bsz=2157.8, num_updates=144500, lr=8.3189e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-14 08:58:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 08:59:01 | INFO | valid | epoch 513 | valid on 'valid' subset | loss 12.452 | nll_loss 11.277 | word_ins 12.207 | length 4.886 | ppl 5603.32 | bleu 32.07 | wps 87843.4 | wpb 21176.3 | bsz 666.3 | num_updates 144544 | best_bleu 32.33
2023-06-14 08:59:01 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 08:59:13 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint513.pt (epoch 513 @ 144544 updates, score 32.07) (writing took 12.055463567376137 seconds)
2023-06-14 08:59:13 | INFO | fairseq_cli.train | end of epoch 513 (average epoch stats below)
2023-06-14 08:59:13 | INFO | train | epoch 513 | loss 3.006 | nll_loss 1.086 | glat_accu 0.609 | glat_context_p 0.404 | word_ins 2.885 | length 2.864 | ppl 8.03 | wps 112218 | ups 1.86 | wpb 60411 | bsz 2156.4 | num_updates 144544 | lr 8.31764e-05 | gnorm 0.637 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 08:59:14 | INFO | fairseq.trainer | begin training epoch 514
2023-06-14 08:59:45 | INFO | train_inner | epoch 514:     56 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.61, glat_context_p=0.404, word_ins=2.882, length=2.854, ppl=8.02, wps=89267.3, ups=1.48, wpb=60135, bsz=2153, num_updates=144600, lr=8.31603e-05, gnorm=0.643, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:00:31 | INFO | train_inner | epoch 514:    156 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.603, glat_context_p=0.404, word_ins=2.881, length=2.866, ppl=8.01, wps=134183, ups=2.21, wpb=60722.4, bsz=2159, num_updates=144700, lr=8.31315e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:01:17 | INFO | train_inner | epoch 514:    256 / 282 loss=3.006, nll_loss=1.086, glat_accu=0.606, glat_context_p=0.404, word_ins=2.885, length=2.871, ppl=8.03, wps=129510, ups=2.14, wpb=60506.7, bsz=2164.2, num_updates=144800, lr=8.31028e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-14 09:01:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:01:32 | INFO | valid | epoch 514 | valid on 'valid' subset | loss 12.366 | nll_loss 11.184 | word_ins 12.119 | length 4.957 | ppl 5277.26 | bleu 32.16 | wps 88027.2 | wpb 21176.3 | bsz 666.3 | num_updates 144826 | best_bleu 32.33
2023-06-14 09:01:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:01:43 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint514.pt (epoch 514 @ 144826 updates, score 32.16) (writing took 10.913045935332775 seconds)
2023-06-14 09:01:43 | INFO | fairseq_cli.train | end of epoch 514 (average epoch stats below)
2023-06-14 09:01:43 | INFO | train | epoch 514 | loss 3.005 | nll_loss 1.084 | glat_accu 0.605 | glat_context_p 0.404 | word_ins 2.883 | length 2.864 | ppl 8.03 | wps 113672 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 144826 | lr 8.30954e-05 | gnorm 0.64 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:01:43 | INFO | fairseq.trainer | begin training epoch 515
2023-06-14 09:02:24 | INFO | train_inner | epoch 515:     74 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.605, glat_context_p=0.403, word_ins=2.881, length=2.853, ppl=8.01, wps=89824, ups=1.5, wpb=60020.4, bsz=2163.8, num_updates=144900, lr=8.30741e-05, gnorm=0.651, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:03:10 | INFO | train_inner | epoch 515:    174 / 282 loss=3.008, nll_loss=1.088, glat_accu=0.606, glat_context_p=0.403, word_ins=2.886, length=2.859, ppl=8.04, wps=131988, ups=2.17, wpb=60760.5, bsz=2162.1, num_updates=145000, lr=8.30455e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:03:56 | INFO | train_inner | epoch 515:    274 / 282 loss=3.007, nll_loss=1.086, glat_accu=0.61, glat_context_p=0.403, word_ins=2.885, length=2.866, ppl=8.04, wps=132318, ups=2.19, wpb=60427.1, bsz=2157.9, num_updates=145100, lr=8.30169e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:03:59 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:04:02 | INFO | valid | epoch 515 | valid on 'valid' subset | loss 12.442 | nll_loss 11.268 | word_ins 12.198 | length 4.886 | ppl 5564.44 | bleu 32.03 | wps 87565 | wpb 21176.3 | bsz 666.3 | num_updates 145108 | best_bleu 32.33
2023-06-14 09:04:02 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:04:13 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint515.pt (epoch 515 @ 145108 updates, score 32.03) (writing took 10.902136698365211 seconds)
2023-06-14 09:04:13 | INFO | fairseq_cli.train | end of epoch 515 (average epoch stats below)
2023-06-14 09:04:13 | INFO | train | epoch 515 | loss 3.005 | nll_loss 1.085 | glat_accu 0.607 | glat_context_p 0.403 | word_ins 2.884 | length 2.86 | ppl 8.03 | wps 113618 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 145108 | lr 8.30146e-05 | gnorm 0.638 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:04:13 | INFO | fairseq.trainer | begin training epoch 516
2023-06-14 09:05:02 | INFO | train_inner | epoch 516:     92 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.607, glat_context_p=0.403, word_ins=2.882, length=2.86, ppl=8.02, wps=91327.8, ups=1.52, wpb=60218, bsz=2105.3, num_updates=145200, lr=8.29883e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:05:48 | INFO | train_inner | epoch 516:    192 / 282 loss=3.006, nll_loss=1.085, glat_accu=0.602, glat_context_p=0.403, word_ins=2.884, length=2.881, ppl=8.03, wps=131446, ups=2.17, wpb=60616.3, bsz=2139.8, num_updates=145300, lr=8.29597e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:06:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:06:32 | INFO | valid | epoch 516 | valid on 'valid' subset | loss 12.39 | nll_loss 11.208 | word_ins 12.141 | length 4.96 | ppl 5368.56 | bleu 32.05 | wps 87609.3 | wpb 21176.3 | bsz 666.3 | num_updates 145390 | best_bleu 32.33
2023-06-14 09:06:32 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:06:41 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint516.pt (epoch 516 @ 145390 updates, score 32.05) (writing took 9.567666906863451 seconds)
2023-06-14 09:06:41 | INFO | fairseq_cli.train | end of epoch 516 (average epoch stats below)
2023-06-14 09:06:41 | INFO | train | epoch 516 | loss 3.004 | nll_loss 1.084 | glat_accu 0.607 | glat_context_p 0.403 | word_ins 2.883 | length 2.861 | ppl 8.02 | wps 114964 | ups 1.9 | wpb 60413.8 | bsz 2157.2 | num_updates 145390 | lr 8.2934e-05 | gnorm 0.638 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:06:42 | INFO | fairseq.trainer | begin training epoch 517
2023-06-14 09:06:52 | INFO | train_inner | epoch 517:     10 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.613, glat_context_p=0.403, word_ins=2.883, length=2.84, ppl=8.02, wps=93014.1, ups=1.55, wpb=59983.5, bsz=2206.7, num_updates=145400, lr=8.29312e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:07:20 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 09:07:39 | INFO | train_inner | epoch 517:    111 / 282 loss=3.004, nll_loss=1.083, glat_accu=0.614, glat_context_p=0.403, word_ins=2.882, length=2.849, ppl=8.02, wps=130656, ups=2.15, wpb=60677.3, bsz=2204.1, num_updates=145500, lr=8.29027e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:08:25 | INFO | train_inner | epoch 517:    211 / 282 loss=3.003, nll_loss=1.082, glat_accu=0.607, glat_context_p=0.403, word_ins=2.882, length=2.875, ppl=8.02, wps=131532, ups=2.17, wpb=60543.3, bsz=2138.6, num_updates=145600, lr=8.28742e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:08:58 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:09:01 | INFO | valid | epoch 517 | valid on 'valid' subset | loss 12.345 | nll_loss 11.169 | word_ins 12.104 | length 4.83 | ppl 5202.5 | bleu 32.16 | wps 88254.3 | wpb 21176.3 | bsz 666.3 | num_updates 145671 | best_bleu 32.33
2023-06-14 09:09:01 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:09:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint517.pt (epoch 517 @ 145671 updates, score 32.16) (writing took 10.796456541866064 seconds)
2023-06-14 09:09:12 | INFO | fairseq_cli.train | end of epoch 517 (average epoch stats below)
2023-06-14 09:09:12 | INFO | train | epoch 517 | loss 3.005 | nll_loss 1.084 | glat_accu 0.608 | glat_context_p 0.403 | word_ins 2.883 | length 2.865 | ppl 8.03 | wps 112838 | ups 1.87 | wpb 60411.2 | bsz 2156.3 | num_updates 145671 | lr 8.2854e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 130 | wall 0
2023-06-14 09:09:12 | INFO | fairseq.trainer | begin training epoch 518
2023-06-14 09:09:32 | INFO | train_inner | epoch 518:     29 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.605, glat_context_p=0.403, word_ins=2.884, length=2.863, ppl=8.03, wps=89866.2, ups=1.5, wpb=60019.3, bsz=2154.9, num_updates=145700, lr=8.28457e-05, gnorm=0.638, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:10:18 | INFO | train_inner | epoch 518:    129 / 282 loss=3, nll_loss=1.08, glat_accu=0.606, glat_context_p=0.403, word_ins=2.879, length=2.846, ppl=8, wps=130961, ups=2.15, wpb=60814.3, bsz=2162.7, num_updates=145800, lr=8.28173e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:11:04 | INFO | train_inner | epoch 518:    229 / 282 loss=3.008, nll_loss=1.087, glat_accu=0.608, glat_context_p=0.403, word_ins=2.886, length=2.865, ppl=8.04, wps=131314, ups=2.17, wpb=60565.1, bsz=2142.2, num_updates=145900, lr=8.27889e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:11:28 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:11:31 | INFO | valid | epoch 518 | valid on 'valid' subset | loss 12.392 | nll_loss 11.223 | word_ins 12.153 | length 4.767 | ppl 5374.03 | bleu 32.32 | wps 87951.8 | wpb 21176.3 | bsz 666.3 | num_updates 145953 | best_bleu 32.33
2023-06-14 09:11:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:11:42 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint518.pt (epoch 518 @ 145953 updates, score 32.32) (writing took 10.470973573625088 seconds)
2023-06-14 09:11:42 | INFO | fairseq_cli.train | end of epoch 518 (average epoch stats below)
2023-06-14 09:11:42 | INFO | train | epoch 518 | loss 3.004 | nll_loss 1.084 | glat_accu 0.608 | glat_context_p 0.403 | word_ins 2.883 | length 2.86 | ppl 8.02 | wps 113677 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 145953 | lr 8.27739e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:11:42 | INFO | fairseq.trainer | begin training epoch 519
2023-06-14 09:12:10 | INFO | train_inner | epoch 519:     47 / 282 loss=3.007, nll_loss=1.087, glat_accu=0.608, glat_context_p=0.403, word_ins=2.886, length=2.872, ppl=8.04, wps=91730.9, ups=1.53, wpb=59917.2, bsz=2128.6, num_updates=146000, lr=8.27606e-05, gnorm=0.646, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:12:55 | INFO | train_inner | epoch 519:    147 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.61, glat_context_p=0.403, word_ins=2.884, length=2.854, ppl=8.03, wps=131929, ups=2.18, wpb=60542.3, bsz=2148.2, num_updates=146100, lr=8.27323e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:13:42 | INFO | train_inner | epoch 519:    247 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.606, glat_context_p=0.403, word_ins=2.884, length=2.869, ppl=8.03, wps=131546, ups=2.17, wpb=60672.1, bsz=2168.9, num_updates=146200, lr=8.2704e-05, gnorm=0.642, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:13:57 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:14:00 | INFO | valid | epoch 519 | valid on 'valid' subset | loss 12.416 | nll_loss 11.244 | word_ins 12.172 | length 4.881 | ppl 5466.09 | bleu 31.87 | wps 84135.9 | wpb 21176.3 | bsz 666.3 | num_updates 146235 | best_bleu 32.33
2023-06-14 09:14:00 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:14:12 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint519.pt (epoch 519 @ 146235 updates, score 31.87) (writing took 11.151031635701656 seconds)
2023-06-14 09:14:12 | INFO | fairseq_cli.train | end of epoch 519 (average epoch stats below)
2023-06-14 09:14:12 | INFO | train | epoch 519 | loss 3.005 | nll_loss 1.085 | glat_accu 0.609 | glat_context_p 0.403 | word_ins 2.884 | length 2.859 | ppl 8.03 | wps 113709 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 146235 | lr 8.26941e-05 | gnorm 0.641 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:14:12 | INFO | fairseq.trainer | begin training epoch 520
2023-06-14 09:14:48 | INFO | train_inner | epoch 520:     65 / 282 loss=2.997, nll_loss=1.077, glat_accu=0.612, glat_context_p=0.403, word_ins=2.876, length=2.844, ppl=7.98, wps=90651.9, ups=1.51, wpb=60068.7, bsz=2198.8, num_updates=146300, lr=8.26757e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:15:34 | INFO | train_inner | epoch 520:    165 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.608, glat_context_p=0.402, word_ins=2.881, length=2.855, ppl=8.01, wps=131603, ups=2.17, wpb=60634.2, bsz=2160.3, num_updates=146400, lr=8.26475e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:16:13 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 09:16:20 | INFO | train_inner | epoch 520:    266 / 282 loss=3.009, nll_loss=1.089, glat_accu=0.606, glat_context_p=0.402, word_ins=2.888, length=2.875, ppl=8.05, wps=130911, ups=2.16, wpb=60557.8, bsz=2145.4, num_updates=146500, lr=8.26192e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:16:27 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:16:31 | INFO | valid | epoch 520 | valid on 'valid' subset | loss 12.338 | nll_loss 11.158 | word_ins 12.095 | length 4.863 | ppl 5177.82 | bleu 32.36 | wps 87568.7 | wpb 21176.3 | bsz 666.3 | num_updates 146516 | best_bleu 32.36
2023-06-14 09:16:31 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:16:45 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint520.pt (epoch 520 @ 146516 updates, score 32.36) (writing took 14.793999321758747 seconds)
2023-06-14 09:16:45 | INFO | fairseq_cli.train | end of epoch 520 (average epoch stats below)
2023-06-14 09:16:45 | INFO | train | epoch 520 | loss 3.003 | nll_loss 1.083 | glat_accu 0.608 | glat_context_p 0.402 | word_ins 2.882 | length 2.86 | ppl 8.02 | wps 110434 | ups 1.83 | wpb 60415.1 | bsz 2157.8 | num_updates 146516 | lr 8.26147e-05 | gnorm 0.635 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:16:45 | INFO | fairseq.trainer | begin training epoch 521
2023-06-14 09:17:30 | INFO | train_inner | epoch 521:     84 / 282 loss=3.001, nll_loss=1.081, glat_accu=0.614, glat_context_p=0.402, word_ins=2.88, length=2.844, ppl=8.01, wps=86511, ups=1.44, wpb=60221.8, bsz=2183.3, num_updates=146600, lr=8.25911e-05, gnorm=0.646, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:18:16 | INFO | train_inner | epoch 521:    184 / 282 loss=3.001, nll_loss=1.081, glat_accu=0.603, glat_context_p=0.402, word_ins=2.88, length=2.866, ppl=8.01, wps=131251, ups=2.17, wpb=60530, bsz=2134.2, num_updates=146700, lr=8.25629e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:19:01 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:19:04 | INFO | valid | epoch 521 | valid on 'valid' subset | loss 12.435 | nll_loss 11.261 | word_ins 12.19 | length 4.912 | ppl 5538.98 | bleu 32.07 | wps 87525.7 | wpb 21176.3 | bsz 666.3 | num_updates 146798 | best_bleu 32.36
2023-06-14 09:19:04 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:19:14 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint521.pt (epoch 521 @ 146798 updates, score 32.07) (writing took 10.18504697829485 seconds)
2023-06-14 09:19:14 | INFO | fairseq_cli.train | end of epoch 521 (average epoch stats below)
2023-06-14 09:19:14 | INFO | train | epoch 521 | loss 3.002 | nll_loss 1.082 | glat_accu 0.607 | glat_context_p 0.402 | word_ins 2.881 | length 2.863 | ppl 8.01 | wps 114278 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 146798 | lr 8.25353e-05 | gnorm 0.644 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:19:15 | INFO | fairseq.trainer | begin training epoch 522
2023-06-14 09:19:21 | INFO | train_inner | epoch 522:      2 / 282 loss=3.006, nll_loss=1.085, glat_accu=0.605, glat_context_p=0.402, word_ins=2.884, length=2.88, ppl=8.03, wps=91703.9, ups=1.53, wpb=59997, bsz=2124.9, num_updates=146800, lr=8.25348e-05, gnorm=0.66, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:20:07 | INFO | train_inner | epoch 522:    102 / 282 loss=3.001, nll_loss=1.08, glat_accu=0.599, glat_context_p=0.402, word_ins=2.88, length=2.875, ppl=8.01, wps=131450, ups=2.17, wpb=60489.9, bsz=2130.6, num_updates=146900, lr=8.25067e-05, gnorm=0.63, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:20:53 | INFO | train_inner | epoch 522:    202 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.611, glat_context_p=0.402, word_ins=2.884, length=2.858, ppl=8.03, wps=133020, ups=2.19, wpb=60611, bsz=2161.3, num_updates=147000, lr=8.24786e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:21:29 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:21:33 | INFO | valid | epoch 522 | valid on 'valid' subset | loss 12.384 | nll_loss 11.204 | word_ins 12.138 | length 4.916 | ppl 5345.82 | bleu 32.32 | wps 83129.7 | wpb 21176.3 | bsz 666.3 | num_updates 147080 | best_bleu 32.36
2023-06-14 09:21:33 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:21:42 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint522.pt (epoch 522 @ 147080 updates, score 32.32) (writing took 9.417562890797853 seconds)
2023-06-14 09:21:42 | INFO | fairseq_cli.train | end of epoch 522 (average epoch stats below)
2023-06-14 09:21:42 | INFO | train | epoch 522 | loss 3.004 | nll_loss 1.083 | glat_accu 0.607 | glat_context_p 0.402 | word_ins 2.882 | length 2.86 | ppl 8.02 | wps 115403 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 147080 | lr 8.24562e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:21:42 | INFO | fairseq.trainer | begin training epoch 523
2023-06-14 09:21:57 | INFO | train_inner | epoch 523:     20 / 282 loss=3.005, nll_loss=1.085, glat_accu=0.608, glat_context_p=0.402, word_ins=2.884, length=2.85, ppl=8.03, wps=94293.7, ups=1.57, wpb=60151.8, bsz=2169.4, num_updates=147100, lr=8.24506e-05, gnorm=0.638, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:22:42 | INFO | train_inner | epoch 523:    120 / 282 loss=2.999, nll_loss=1.079, glat_accu=0.614, glat_context_p=0.402, word_ins=2.878, length=2.843, ppl=8, wps=132433, ups=2.19, wpb=60410.5, bsz=2185.5, num_updates=147200, lr=8.24226e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:23:28 | INFO | train_inner | epoch 523:    220 / 282 loss=3.007, nll_loss=1.086, glat_accu=0.604, glat_context_p=0.402, word_ins=2.885, length=2.876, ppl=8.04, wps=132217, ups=2.18, wpb=60704.6, bsz=2146.1, num_updates=147300, lr=8.23946e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:23:57 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:24:00 | INFO | valid | epoch 523 | valid on 'valid' subset | loss 12.386 | nll_loss 11.203 | word_ins 12.142 | length 4.896 | ppl 5352.17 | bleu 32.11 | wps 88491.5 | wpb 21176.3 | bsz 666.3 | num_updates 147362 | best_bleu 32.36
2023-06-14 09:24:00 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:24:11 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint523.pt (epoch 523 @ 147362 updates, score 32.11) (writing took 11.450906686484814 seconds)
2023-06-14 09:24:11 | INFO | fairseq_cli.train | end of epoch 523 (average epoch stats below)
2023-06-14 09:24:11 | INFO | train | epoch 523 | loss 3.003 | nll_loss 1.083 | glat_accu 0.608 | glat_context_p 0.402 | word_ins 2.882 | length 2.859 | ppl 8.02 | wps 114087 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 147362 | lr 8.23772e-05 | gnorm 0.642 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:24:12 | INFO | fairseq.trainer | begin training epoch 524
2023-06-14 09:24:35 | INFO | train_inner | epoch 524:     38 / 282 loss=3, nll_loss=1.08, glat_accu=0.609, glat_context_p=0.402, word_ins=2.879, length=2.846, ppl=8, wps=90847.1, ups=1.51, wpb=60268.1, bsz=2167.8, num_updates=147400, lr=8.23666e-05, gnorm=0.643, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:25:20 | INFO | train_inner | epoch 524:    138 / 282 loss=3.001, nll_loss=1.081, glat_accu=0.604, glat_context_p=0.402, word_ins=2.88, length=2.859, ppl=8.01, wps=132016, ups=2.18, wpb=60601.9, bsz=2154.3, num_updates=147500, lr=8.23387e-05, gnorm=0.634, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:25:25 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 09:26:07 | INFO | train_inner | epoch 524:    239 / 282 loss=2.997, nll_loss=1.076, glat_accu=0.597, glat_context_p=0.402, word_ins=2.876, length=2.877, ppl=7.99, wps=131089, ups=2.17, wpb=60426.5, bsz=2124.9, num_updates=147600, lr=8.23108e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:26:26 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:26:29 | INFO | valid | epoch 524 | valid on 'valid' subset | loss 12.332 | nll_loss 11.153 | word_ins 12.093 | length 4.782 | ppl 5155.01 | bleu 31.99 | wps 87117.4 | wpb 21176.3 | bsz 666.3 | num_updates 147643 | best_bleu 32.36
2023-06-14 09:26:29 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:26:38 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint524.pt (epoch 524 @ 147643 updates, score 31.99) (writing took 9.417877551168203 seconds)
2023-06-14 09:26:38 | INFO | fairseq_cli.train | end of epoch 524 (average epoch stats below)
2023-06-14 09:26:38 | INFO | train | epoch 524 | loss 2.997 | nll_loss 1.077 | glat_accu 0.602 | glat_context_p 0.402 | word_ins 2.876 | length 2.858 | ppl 7.98 | wps 115543 | ups 1.91 | wpb 60417 | bsz 2156.6 | num_updates 147643 | lr 8.22988e-05 | gnorm 0.631 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:26:38 | INFO | fairseq.trainer | begin training epoch 525
2023-06-14 09:27:11 | INFO | train_inner | epoch 525:     57 / 282 loss=2.993, nll_loss=1.072, glat_accu=0.603, glat_context_p=0.402, word_ins=2.872, length=2.852, ppl=7.96, wps=93602.3, ups=1.56, wpb=60023, bsz=2158.7, num_updates=147700, lr=8.22829e-05, gnorm=0.628, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:27:57 | INFO | train_inner | epoch 525:    157 / 282 loss=3.003, nll_loss=1.083, glat_accu=0.604, glat_context_p=0.402, word_ins=2.882, length=2.855, ppl=8.01, wps=132289, ups=2.18, wpb=60710.9, bsz=2148.3, num_updates=147800, lr=8.22551e-05, gnorm=0.627, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:28:42 | INFO | train_inner | epoch 525:    257 / 282 loss=2.999, nll_loss=1.078, glat_accu=0.602, glat_context_p=0.401, word_ins=2.878, length=2.861, ppl=7.99, wps=132085, ups=2.18, wpb=60596.4, bsz=2198.6, num_updates=147900, lr=8.22273e-05, gnorm=0.644, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:28:54 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:28:57 | INFO | valid | epoch 525 | valid on 'valid' subset | loss 12.363 | nll_loss 11.183 | word_ins 12.114 | length 4.976 | ppl 5266.51 | bleu 32.09 | wps 87399.2 | wpb 21176.3 | bsz 666.3 | num_updates 147925 | best_bleu 32.36
2023-06-14 09:28:57 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:29:05 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint525.pt (epoch 525 @ 147925 updates, score 32.09) (writing took 8.372677013278008 seconds)
2023-06-14 09:29:05 | INFO | fairseq_cli.train | end of epoch 525 (average epoch stats below)
2023-06-14 09:29:05 | INFO | train | epoch 525 | loss 2.999 | nll_loss 1.079 | glat_accu 0.603 | glat_context_p 0.401 | word_ins 2.878 | length 2.86 | ppl 7.99 | wps 116078 | ups 1.92 | wpb 60413.8 | bsz 2157.2 | num_updates 147925 | lr 8.22203e-05 | gnorm 0.637 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:29:05 | INFO | fairseq.trainer | begin training epoch 526
2023-06-14 09:29:46 | INFO | train_inner | epoch 526:     75 / 282 loss=3, nll_loss=1.08, glat_accu=0.602, glat_context_p=0.401, word_ins=2.879, length=2.861, ppl=8, wps=95021.2, ups=1.58, wpb=60017.7, bsz=2112.5, num_updates=148000, lr=8.21995e-05, gnorm=0.658, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:30:31 | INFO | train_inner | epoch 526:    175 / 282 loss=2.999, nll_loss=1.078, glat_accu=0.603, glat_context_p=0.401, word_ins=2.877, length=2.867, ppl=7.99, wps=132176, ups=2.18, wpb=60510.8, bsz=2155.2, num_updates=148100, lr=8.21717e-05, gnorm=0.631, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:31:17 | INFO | train_inner | epoch 526:    275 / 282 loss=2.997, nll_loss=1.077, glat_accu=0.609, glat_context_p=0.401, word_ins=2.876, length=2.844, ppl=7.98, wps=133495, ups=2.2, wpb=60740.9, bsz=2206.8, num_updates=148200, lr=8.2144e-05, gnorm=0.64, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:31:20 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:31:23 | INFO | valid | epoch 526 | valid on 'valid' subset | loss 12.37 | nll_loss 11.196 | word_ins 12.128 | length 4.839 | ppl 5294.31 | bleu 32.2 | wps 87402 | wpb 21176.3 | bsz 666.3 | num_updates 148207 | best_bleu 32.36
2023-06-14 09:31:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:31:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint526.pt (epoch 526 @ 148207 updates, score 32.2) (writing took 11.281135898083448 seconds)
2023-06-14 09:31:34 | INFO | fairseq_cli.train | end of epoch 526 (average epoch stats below)
2023-06-14 09:31:34 | INFO | train | epoch 526 | loss 2.999 | nll_loss 1.078 | glat_accu 0.605 | glat_context_p 0.401 | word_ins 2.878 | length 2.857 | ppl 7.99 | wps 114127 | ups 1.89 | wpb 60413.8 | bsz 2157.2 | num_updates 148207 | lr 8.21421e-05 | gnorm 0.643 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:31:35 | INFO | fairseq.trainer | begin training epoch 527
2023-06-14 09:32:24 | INFO | train_inner | epoch 527:     93 / 282 loss=3, nll_loss=1.079, glat_accu=0.598, glat_context_p=0.401, word_ins=2.879, length=2.864, ppl=8, wps=90147.6, ups=1.5, wpb=60221.2, bsz=2101.8, num_updates=148300, lr=8.21163e-05, gnorm=0.638, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:33:09 | INFO | train_inner | epoch 527:    193 / 282 loss=3.002, nll_loss=1.081, glat_accu=0.608, glat_context_p=0.401, word_ins=2.881, length=2.856, ppl=8.01, wps=132856, ups=2.19, wpb=60556.8, bsz=2171, num_updates=148400, lr=8.20886e-05, gnorm=0.641, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:33:50 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:33:53 | INFO | valid | epoch 527 | valid on 'valid' subset | loss 12.341 | nll_loss 11.161 | word_ins 12.098 | length 4.872 | ppl 5186.62 | bleu 32.19 | wps 81227.6 | wpb 21176.3 | bsz 666.3 | num_updates 148489 | best_bleu 32.36
2023-06-14 09:33:53 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:34:04 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint527.pt (epoch 527 @ 148489 updates, score 32.19) (writing took 11.014223746955395 seconds)
2023-06-14 09:34:04 | INFO | fairseq_cli.train | end of epoch 527 (average epoch stats below)
2023-06-14 09:34:04 | INFO | train | epoch 527 | loss 3 | nll_loss 1.079 | glat_accu 0.603 | glat_context_p 0.401 | word_ins 2.879 | length 2.859 | ppl 8 | wps 113754 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 148489 | lr 8.2064e-05 | gnorm 0.636 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:34:04 | INFO | fairseq.trainer | begin training epoch 528
2023-06-14 09:34:16 | INFO | train_inner | epoch 528:     11 / 282 loss=2.997, nll_loss=1.076, glat_accu=0.606, glat_context_p=0.401, word_ins=2.876, length=2.856, ppl=7.98, wps=89651.2, ups=1.49, wpb=60049.2, bsz=2172.2, num_updates=148500, lr=8.2061e-05, gnorm=0.637, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:34:32 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 09:35:03 | INFO | train_inner | epoch 528:    112 / 282 loss=3.005, nll_loss=1.084, glat_accu=0.603, glat_context_p=0.401, word_ins=2.883, length=2.869, ppl=8.03, wps=129854, ups=2.14, wpb=60624.2, bsz=2134.2, num_updates=148600, lr=8.20334e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=47, wall=0
2023-06-14 09:35:48 | INFO | train_inner | epoch 528:    212 / 282 loss=2.995, nll_loss=1.075, glat_accu=0.611, glat_context_p=0.401, word_ins=2.875, length=2.843, ppl=7.97, wps=133712, ups=2.21, wpb=60598.1, bsz=2209.4, num_updates=148700, lr=8.20058e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:36:20 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:36:23 | INFO | valid | epoch 528 | valid on 'valid' subset | loss 12.373 | nll_loss 11.201 | word_ins 12.132 | length 4.827 | ppl 5304.21 | bleu 31.79 | wps 87169.5 | wpb 21176.3 | bsz 666.3 | num_updates 148770 | best_bleu 32.36
2023-06-14 09:36:23 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:36:33 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint528.pt (epoch 528 @ 148770 updates, score 31.79) (writing took 9.444878898561 seconds)
2023-06-14 09:36:33 | INFO | fairseq_cli.train | end of epoch 528 (average epoch stats below)
2023-06-14 09:36:33 | INFO | train | epoch 528 | loss 3 | nll_loss 1.08 | glat_accu 0.606 | glat_context_p 0.401 | word_ins 2.879 | length 2.856 | ppl 8 | wps 114108 | ups 1.89 | wpb 60408.7 | bsz 2156.6 | num_updates 148770 | lr 8.19865e-05 | gnorm 0.635 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:36:33 | INFO | fairseq.trainer | begin training epoch 529
2023-06-14 09:36:52 | INFO | train_inner | epoch 529:     30 / 282 loss=2.997, nll_loss=1.077, glat_accu=0.603, glat_context_p=0.401, word_ins=2.877, length=2.849, ppl=7.98, wps=93673.8, ups=1.56, wpb=59988.5, bsz=2150.9, num_updates=148800, lr=8.19782e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:37:38 | INFO | train_inner | epoch 529:    130 / 282 loss=2.997, nll_loss=1.076, glat_accu=0.609, glat_context_p=0.401, word_ins=2.876, length=2.861, ppl=7.99, wps=133865, ups=2.21, wpb=60535.9, bsz=2151, num_updates=148900, lr=8.19507e-05, gnorm=0.636, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:38:23 | INFO | train_inner | epoch 529:    230 / 282 loss=3.002, nll_loss=1.082, glat_accu=0.608, glat_context_p=0.401, word_ins=2.881, length=2.854, ppl=8.01, wps=131990, ups=2.18, wpb=60589.3, bsz=2177.1, num_updates=149000, lr=8.19232e-05, gnorm=0.625, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:38:47 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:38:50 | INFO | valid | epoch 529 | valid on 'valid' subset | loss 12.381 | nll_loss 11.201 | word_ins 12.134 | length 4.961 | ppl 5334.03 | bleu 32.27 | wps 88372.4 | wpb 21176.3 | bsz 666.3 | num_updates 149052 | best_bleu 32.36
2023-06-14 09:38:50 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:39:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint529.pt (epoch 529 @ 149052 updates, score 32.27) (writing took 10.381935022771358 seconds)
2023-06-14 09:39:01 | INFO | fairseq_cli.train | end of epoch 529 (average epoch stats below)
2023-06-14 09:39:01 | INFO | train | epoch 529 | loss 3 | nll_loss 1.079 | glat_accu 0.607 | glat_context_p 0.401 | word_ins 2.879 | length 2.857 | ppl 8 | wps 115353 | ups 1.91 | wpb 60413.8 | bsz 2157.2 | num_updates 149052 | lr 8.19089e-05 | gnorm 0.636 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:39:01 | INFO | fairseq.trainer | begin training epoch 530
2023-06-14 09:39:29 | INFO | train_inner | epoch 530:     48 / 282 loss=3.001, nll_loss=1.08, glat_accu=0.607, glat_context_p=0.401, word_ins=2.88, length=2.859, ppl=8, wps=92449.3, ups=1.54, wpb=60154.5, bsz=2140.1, num_updates=149100, lr=8.18957e-05, gnorm=0.664, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:40:14 | INFO | train_inner | epoch 530:    148 / 282 loss=2.997, nll_loss=1.076, glat_accu=0.607, glat_context_p=0.401, word_ins=2.876, length=2.865, ppl=7.98, wps=132414, ups=2.19, wpb=60570.9, bsz=2164.9, num_updates=149200, lr=8.18683e-05, gnorm=0.649, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:41:00 | INFO | train_inner | epoch 530:    248 / 282 loss=3.004, nll_loss=1.084, glat_accu=0.608, glat_context_p=0.401, word_ins=2.883, length=2.861, ppl=8.02, wps=131172, ups=2.17, wpb=60586.4, bsz=2144.4, num_updates=149300, lr=8.18408e-05, gnorm=0.654, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:41:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:41:19 | INFO | valid | epoch 530 | valid on 'valid' subset | loss 12.37 | nll_loss 11.195 | word_ins 12.127 | length 4.856 | ppl 5292.52 | bleu 32.38 | wps 86527.8 | wpb 21176.3 | bsz 666.3 | num_updates 149334 | best_bleu 32.38
2023-06-14 09:41:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:41:34 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint530.pt (epoch 530 @ 149334 updates, score 32.38) (writing took 15.036746598780155 seconds)
2023-06-14 09:41:34 | INFO | fairseq_cli.train | end of epoch 530 (average epoch stats below)
2023-06-14 09:41:34 | INFO | train | epoch 530 | loss 3.001 | nll_loss 1.08 | glat_accu 0.608 | glat_context_p 0.401 | word_ins 2.879 | length 2.859 | ppl 8 | wps 111052 | ups 1.84 | wpb 60413.8 | bsz 2157.2 | num_updates 149334 | lr 8.18315e-05 | gnorm 0.653 | clip 0 | loss_scale 32768 | train_wall 129 | wall 0
2023-06-14 09:41:34 | INFO | fairseq.trainer | begin training epoch 531
2023-06-14 09:42:11 | INFO | train_inner | epoch 531:     66 / 282 loss=2.999, nll_loss=1.08, glat_accu=0.613, glat_context_p=0.4, word_ins=2.879, length=2.839, ppl=8, wps=85619.8, ups=1.42, wpb=60172, bsz=2179.8, num_updates=149400, lr=8.18134e-05, gnorm=0.642, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:42:56 | INFO | train_inner | epoch 531:    166 / 282 loss=2.999, nll_loss=1.079, glat_accu=0.603, glat_context_p=0.4, word_ins=2.878, length=2.861, ppl=8, wps=132780, ups=2.2, wpb=60419.1, bsz=2156.2, num_updates=149500, lr=8.17861e-05, gnorm=0.622, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:43:22 | INFO | fairseq.trainer | NOTE: overflow detected, setting loss scale to: 32768.0
2023-06-14 09:43:42 | INFO | train_inner | epoch 531:    267 / 282 loss=3.001, nll_loss=1.081, glat_accu=0.606, glat_context_p=0.4, word_ins=2.88, length=2.861, ppl=8.01, wps=132736, ups=2.19, wpb=60725.2, bsz=2150.7, num_updates=149600, lr=8.17587e-05, gnorm=0.642, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:43:49 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:43:52 | INFO | valid | epoch 531 | valid on 'valid' subset | loss 12.476 | nll_loss 11.297 | word_ins 12.223 | length 5.066 | ppl 5697.36 | bleu 31.84 | wps 87584.2 | wpb 21176.3 | bsz 666.3 | num_updates 149615 | best_bleu 32.38
2023-06-14 09:43:52 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:44:01 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint531.pt (epoch 531 @ 149615 updates, score 31.84) (writing took 9.071624282747507 seconds)
2023-06-14 09:44:01 | INFO | fairseq_cli.train | end of epoch 531 (average epoch stats below)
2023-06-14 09:44:01 | INFO | train | epoch 531 | loss 2.999 | nll_loss 1.079 | glat_accu 0.607 | glat_context_p 0.4 | word_ins 2.878 | length 2.852 | ppl 7.99 | wps 115462 | ups 1.91 | wpb 60417.1 | bsz 2158.1 | num_updates 149615 | lr 8.17546e-05 | gnorm 0.636 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:44:01 | INFO | fairseq.trainer | begin training epoch 532
2023-06-14 09:44:46 | INFO | train_inner | epoch 532:     85 / 282 loss=2.995, nll_loss=1.075, glat_accu=0.599, glat_context_p=0.4, word_ins=2.875, length=2.853, ppl=7.97, wps=94470.9, ups=1.57, wpb=60230.7, bsz=2130.2, num_updates=149700, lr=8.17314e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:45:31 | INFO | train_inner | epoch 532:    185 / 282 loss=2.995, nll_loss=1.074, glat_accu=0.612, glat_context_p=0.4, word_ins=2.874, length=2.848, ppl=7.97, wps=133259, ups=2.2, wpb=60543.1, bsz=2179.2, num_updates=149800, lr=8.17041e-05, gnorm=0.632, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:46:16 | INFO | fairseq_cli.train | begin validation on "valid" subset
2023-06-14 09:46:19 | INFO | valid | epoch 532 | valid on 'valid' subset | loss 12.389 | nll_loss 11.215 | word_ins 12.147 | length 4.855 | ppl 5364.92 | bleu 31.89 | wps 88023.2 | wpb 21176.3 | bsz 666.3 | num_updates 149897 | best_bleu 32.38
2023-06-14 09:46:19 | INFO | fairseq_cli.train | begin save checkpoint
2023-06-14 09:46:31 | INFO | fairseq.checkpoint_utils | saved checkpoint /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat59/checkpoint532.pt (epoch 532 @ 149897 updates, score 31.89) (writing took 12.516877137124538 seconds)
2023-06-14 09:46:31 | INFO | fairseq_cli.train | end of epoch 532 (average epoch stats below)
2023-06-14 09:46:31 | INFO | train | epoch 532 | loss 2.997 | nll_loss 1.076 | glat_accu 0.606 | glat_context_p 0.4 | word_ins 2.876 | length 2.856 | ppl 7.98 | wps 113379 | ups 1.88 | wpb 60413.8 | bsz 2157.2 | num_updates 149897 | lr 8.16777e-05 | gnorm 0.633 | clip 0 | loss_scale 32768 | train_wall 128 | wall 0
2023-06-14 09:46:31 | INFO | fairseq.trainer | begin training epoch 533
2023-06-14 09:46:39 | INFO | train_inner | epoch 533:      3 / 282 loss=3, nll_loss=1.08, glat_accu=0.606, glat_context_p=0.4, word_ins=2.879, length=2.862, ppl=8, wps=88618.7, ups=1.48, wpb=59998.1, bsz=2142.9, num_updates=149900, lr=8.16769e-05, gnorm=0.639, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:47:25 | INFO | train_inner | epoch 533:    103 / 282 loss=2.995, nll_loss=1.075, glat_accu=0.613, glat_context_p=0.4, word_ins=2.874, length=2.835, ppl=7.97, wps=130525, ups=2.15, wpb=60663.6, bsz=2192.2, num_updates=150000, lr=8.16497e-05, gnorm=0.635, clip=0, loss_scale=32768, train_wall=46, wall=0
2023-06-14 09:48:11 | INFO | train_inner | epoch 533:    203 / 282 loss=3.01, nll_loss=1.09, glat_accu=0.604, glat_context_p=0.4, word_ins=2.888, length=2.856, ppl=8.05, wps=133282, ups=2.2, wpb=60537.7, bsz=2182.3, num_updates=150100, lr=8.16225e-05, gnorm=0.668, clip=0, loss_scale=32768, train_wall=45, wall=0
2023-06-14 09:48:47 | INFO | fairseq_cli.train | begin validation on "valid" subset
