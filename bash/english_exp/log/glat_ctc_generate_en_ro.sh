source activate syngec
export CUDA_LAUNCH_BLOCKING=1
data_dir=/opt/data/private/zjx/data/wmt/enro/bin
checkpoint_path=/opt/data/private/friends/tzc/checkpoint/checkpoint_glat_ctc/7/checkpoint_ave.pt
src=en
tgt=ro
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > glat_ctc_enro.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task translation_lev_modified --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 1 \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model
bash /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh glat_ctc_enro.log
# 记得修改这里面的log路径：/opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/score.py
# logging.basicConfig(filename='/opt/data/private/zjx/SynGEC-main/glat_ctc_enro_log.log',

    # --use-syntax \
    # --syntax-encoder GCN \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5. 迭代强化生成
# ###