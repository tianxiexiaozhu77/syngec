source activate syngec
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
# data_dir=/opt/data/private/friends/tzc/data/iwslt/bin  # raw data
# data_dir=/opt/data/private/friends/tzc/data/iwslt_distill/iwslt_de/bin  # distill data
data_dir=/opt/data/private/friends/tzc/data/wmt_gp/process.disco/wmt16.en-de/deal/all/bin  # distill data
checkpoint_path=/opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat49/checkpoint_ave.pt
src=en
tgt=de
echo "--beam1-↓---"
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > res_edde.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-enhanced-nat --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 1 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe
bash /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res_edde.log

echo "--beam5-↓---"
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > res_edde.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-enhanced-nat --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 5 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe
bash /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res_edde.log

echo "--beam10-↓---"
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > res_edde.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-enhanced-nat --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 10 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe
bash /opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh res_edde.log

    # --use-syntax \
    # --syntax-encoder GCN \
        # --quiet \
    # --eval_tokenized_bleu \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5.迭代强化生成