source activate syngec
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
# data_dir=/opt/data/private/friends/tzc/data/iwslt_de/iwslt_de/bin
# data_dir=/opt/data/private/friends/tzc/data/iwslt_distill/iwslt_de/bin  # 字典10104 distill data
data_dir=/opt/data/private/zjx/data/iwslt_de/iwslt_de/bin  # 字典10152 distill data
checkpoint_path=/opt/data/private/friends/tzc/checkpoint/iwslt/raw/deen/glat_syntax/nat33/checkpoint_average.pt
src=de
tgt=en

CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > glat_syntax_raw_deen.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-glat-task --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 10 --iter-decode-with-beam 5 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model
bash /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh glat_syntax_raw_deen.log



# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 10 --iter-decode-with-beam 5 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \


    # --use-syntax \
    # --syntax-encoder GCN \
# --gat-encoder-layers 5 \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5. 迭代强化生成
# ###