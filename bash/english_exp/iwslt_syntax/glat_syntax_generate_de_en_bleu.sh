source activate syndecoder
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1
# data_dir=/opt/data/private/friends/tzc/data/iwslt_de/iwslt_de/bin
# data_dir=/opt/data/private/friends/tzc/data/iwslt_distill/iwslt_de/bin  # 
data_dir=/opt/data/private/zjx/data/fairseq_iwslt14.tokenized.distil.de-en/bin  # 
checkpoint_path=/opt/data/private/zjx/ckpt/iwslt_distill_de_en_syntax/4/checkpoint_ave.pt
src=de
tgt=en
python /opt/data/private/zjx/syngec/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py > res_33.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task syntax-glat-task --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 5 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/zjx/syngec/src/src_syngec/syngec_model 

bash src/src_syngec/fairseq-0.10.2/scripts/compound_split_bleu.sh /opt/data/private/zjx/syngec/res_33.log > res_33_bleu.txt

    # --remove-bpe

    # --use-syntax \
    # --syntax-encoder GCN \


# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5. 迭代强化生成
# ###