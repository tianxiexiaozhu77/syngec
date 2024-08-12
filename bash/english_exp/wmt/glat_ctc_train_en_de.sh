source activate syngec
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 只有一个可见
save_path="/opt/data/private/friends/tzc/checkpoint/checkpoint_glat_ctc/1"
data_dir="/opt/data/private/zjx/data/wmt/ende/bin"
user_dir="/opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model"
touch /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model/log/$(date -d "today" +"%Y.%m.%d_glat_ctc_wmt_en_de_1").log
nohup fairseq-train \
    ${data_dir} --arch glat_ctc_wmt_en_de --noise full_mask --share-all-embeddings \
    --criterion ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1024 --max-target-positions 1024 --max-update 300000 --seed 0 --clip-norm 2 \
    --save-dir ${save_path} --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir ${user_dir} \
    > /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model/log/$(date -d "today" +"%Y.%m.%d_glat_ctc_wmt_en_de_1").log 2>&1 &
# tail -f /opt/data/private/friends/wj/GLAT/bash/$(date -d "today" +"%Y.%m.%d_glat_ctc_train_log_2").log
