source activate syngec
export CUDA_LAUNCH_BLOCKING=1
data_dir=/opt/data/private/zjx/data/wmt/roen/bin
checkpoint_path_1=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave1.pt
checkpoint_path_2=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave2.pt
checkpoint_path_3=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave3.pt
checkpoint_path_4=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave4.pt
checkpoint_path_5=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave5.pt
checkpoint_path_6=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave6.pt
checkpoint_path_7=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave7.pt
checkpoint_path_8=/opt/data/private/friends/tzc/checkpoint/checkpoint_syntax_glat_ctc/9/checkpoint_ave8.pt
src=ro
tgt=en
# # 1
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_1} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

# # 2
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_2} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

# # 3
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_3} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \
# # 4
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_4} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

# # 5
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_5} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

# 6
CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path_6} --iter-decode-force-max-iter \
    --task syntax-glat-task --gen-subset test \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --batch-size 1 --iter-decode-with-beam 1 \
    --use-syntax \
    --syntax-encoder GCN \
    --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
    --remove-bpe \
    --quiet \

# # 7
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_7} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \

# # 8
# CUDA_VISIBLE_DEVICES=0 python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/fairseq_cli/generate.py \
#     ${data_dir} -s ${src} -t ${tgt} \
#     --path ${checkpoint_path_8} --iter-decode-force-max-iter \
#     --task syntax-glat-task --gen-subset test \
#     --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
#     --batch-size 1 --iter-decode-with-beam 1 \
#     --use-syntax \
#     --syntax-encoder GCN \
#     --user-dir /opt/data/private/zjx/SynGEC-main/src/src_syngec/syngec_model \
#     --remove-bpe \
#     --quiet \
    # --use-syntax \
    # --syntax-encoder GCN \

# 需要改4个地方
# 1. checkpoint_path指定的路径
# 2. --iter-decode-with-beam 3
# 3. --task translation_lev_modified
# 4. 激活哪个环境source activate syngec
# 5. 迭代强化生成
# ###