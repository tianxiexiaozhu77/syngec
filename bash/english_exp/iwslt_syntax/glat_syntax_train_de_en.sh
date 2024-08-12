source activate syndecoder
export CUDA_VISIBLE_DEVICES=1  # 只有一个可见
save_path="/opt/data/private/zjx/ckpt/iwslt_distill_de_en_syntax/4"
# data_dir="/opt/data/private/friends/tzc/data/iwslt_de/iwslt_de/bin"  # raw
# data_dir="/opt/data/private/friends/tzc/data/iwslt_distill/iwslt_de/bin"  # 字典10104 distill
data_dir="/opt/data/private/zjx/data/fairseq_iwslt14.tokenized.distil.de-en/bin"  # 字典10152 distill
user_dir="/opt/data/private/zjx/syngec/src/src_syngec/syngec_model"
touch /opt/data/private/zjx/syngec/bash/english_exp/iwslt_syntax/log/$(date -d "today" +"%Y.%m.%d_syntax_glat_iwslt_distill_de_en").log
nohup fairseq-train  \
    ${data_dir} \
    -s de \
    -t en \
    --save-dir ${save_path} \
    --user-dir ${user_dir} \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --ddp-backend=no_c10d \
    --task syntax-glat-task \
    --criterion syntax_glat_loss \
    --arch syntax_glat_iwslt \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam  \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --apply-bert-init \
    --src-embedding-copy \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 2048 \
    --update-freq 4 \
    --fp16 \
    --max-update 500000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --activation-fn gelu \
    --length-loss-factor 0.05 \
    --clip-norm 5 \
    --warmup-updates 4000 \
    --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
    --no-scale-embedding \
    --seed 0 \
    > /opt/data/private/zjx/syngec/bash/english_exp/iwslt_syntax/log/$(date -d "today" +"%Y.%m.%d_syntax_glat_iwslt_distill_de_en").log 2>&1 &
# tail -f /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model/$(date -d "today" +"%Y.%m.%d_syntax_glat_de_en_46").log




# 需要修改
# 1. 日志的路径
# 2. checkpoint的路径
# 3.
# --task syntax-glat-task \
# --criterion syntax_glat_loss \
# --arch syntax_glat_iwslt \
# 4. source activate syngec