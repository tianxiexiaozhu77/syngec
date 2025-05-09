source activate syngec
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
# data_dir=/opt/data/private/friends/tzc/data/iwslt/bin  # raw data
# data_dir=/opt/data/private/friends/tzc/data/iwslt_distill/iwslt_de/bin  #
data_dir=/opt/data/private/zjx/data/iwslt_de_en_distill/bin  # 
checkpoint_path=/opt/data/private/friends/tzc/checkpoint/iwslt/distill/deen/glat/14/checkpoint_average.pt
src=de
tgt=en
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path} --iter-decode-force-max-iter \
#     --task translation_lev_modified --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 10 --iter-decode-with-beam 3 \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

    
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > glat_distll_iwslt_deen.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task translation_lev_modified --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 10 --iter-decode-with-beam 5 \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model
bash /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh glat_distll_iwslt_deen.log

    # --use-syntax \
    # --syntax-encoder GCN \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 5
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5.迭代强化生成