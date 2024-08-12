source activate syngec
export CUDA_VISIBLE_DEVICES=0  # 只有一个可见
save_path="/opt/data/private/friends/tzc/checkpoint/checkpoint_glat/syngec/14"
# data_dir="/opt/data/private/friends/tzc/data/iwslt_de/iwslt_de/bin"  # raw data
data_dir="/opt/data/private/friends/tzc/data/iwslt_distill/fairseq_iwslt14.tokenized.distil.de-en/iwslt_de_en_distill/bin"  # distill data
user_dir="/opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model"
touch /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model/$(date -d "today" +"%Y.%m.%d_glat_glat_nonautoregressive_de_en.14").log
nohup fairseq-train  \
    ${data_dir} \
    -s de \
    -t en \
    --arch glat --noise full_mask --share-all-embeddings --ddp-backend=no_c10d \
    --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --update-freq 1 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1024 --max-target-positions 1024 --max-update 500000 --seed 0 --clip-norm 5\
    --save-dir ${save_path} --src-embedding-copy --length-loss-factor 0.05 --log-interval 1000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir ${user_dir} \
    > /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model/$(date -d "today" +"%Y.%m.%d_glat_glat_nonautoregressive_de_en.14").log 2>&1 &
tail -f /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model/$(date -d "today" +"%Y.%m.%d_glat_glat_nonautoregressive_de_en.14").log


# --max-tokens 8192
# --max-tokens 2048 --update-freq 4 
