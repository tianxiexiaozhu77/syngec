source activate syngec
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
data_dir=/opt/data/private/zjx/data/wmt/roen/bin
checkpoint_path=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat/wmt/roen/nat60/checkpoint_ave.pt
src=ro
tgt=en
echo "--beam1-↓---"
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-glat-task --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 1 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe \
    --quiet \


echo "--beam5-↓---"
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-glat-task --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 5 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe \
    --quiet \
    # > res_roen60.log

# bash /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res_roen60.log


# echo "--beam5-↓---"
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 5 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     > res.log

# bash /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res.log


# echo "--beam10-↓---"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 10 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     > res.log

# bash /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res.log
    # --use-syntax \
    # --syntax-encoder GCN \
        # --quiet \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5. 迭代强化生成
# ###