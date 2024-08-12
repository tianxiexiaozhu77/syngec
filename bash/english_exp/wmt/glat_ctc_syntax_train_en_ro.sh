source activate syngec
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 只有一个可见
save_path="/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/8"
data_dir="/opt/data/private/zjx/data/wmt/enro/bin"
user_dir="/opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model"
touch /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model/log/$(date -d "today" +"%Y.%m.%d_syntax_glat_ctc_wmt_en_ro_1").log
nohup fairseq-train \
    ${data_dir} --arch syntax_glat_ctc_wmt_en_de \
    -s en \
    -t ro \
    --noise full_mask --share-all-embeddings \
    --criterion syntax_ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task syntax-glat-task --max-tokens 1024 --update-freq 8 --weight-decay 0.01 --dropout 0.3 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1024 --max-target-positions 1024 --max-update 1100000 --seed 0 --clip-norm 2 \
    --save-dir ${save_path} --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir ${user_dir} \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --ddp-backend=no_c10d \
    > /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model/log/$(date -d "today" +"%Y.%m.%d_syntax_glat_ctc_wmt_en_ro_1").log 2>&1 &



    # --ddp-backend=no_c10d \
    # --use-syntax \
    # --only-gnn \
    # --syntax-encoder GCN \
    # --ddp-backend=no_c10d \


# 需要修改
# 1. 日志的路径
# 2. checkpoint的路径
# 3.
# --task syntax-glat-task \
# --criterion syntax_glat_loss \
# --arch syntax_glat_iwslt \
# 4. source activate syngec