import shlex
'''
shlex.split 是 Python 标准库中的一个模块，主要用于将字符串按照 shell 的语法规则进行分割，
将其分割成一个个单独的词（token），返回一个列表。
这个函数能够处理带有空格、引号、转义符等特殊字符的字符串，它会忽略在引号中的空格，
同时也能够识别并正确解析转义符号。常见的用途是将字符串参数转换成程序所需的参数列表。
'''
command = "{data_dir} --arch syntax_glat_ctc_wmt_en_de \
    -s ro \
    -t en \
    --noise full_mask --share-all-embeddings \
    --criterion syntax_ctc_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas (0.9, 0.999) \
    --adam-eps 1e-6 --task syntax-glat-task --max-tokens 1024 --update-freq 8 --weight-decay 0.01 --dropout 0.3 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
    --max-source-positions 1024 --max-target-positions 1024 --max-update 1100000 --seed 0 --clip-norm 2 \
    --save-dir ${save_path} --length-loss-factor 0 --log-interval 100 \
    --eval-bleu --eval-bleu-args '{iter_decode_max_iter: 0, iter_decode_with_beam: 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu --user-dir ${user_dir} \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --ddp-backend=no_c10d "
# 使用 shlex.split() 将字符串转化为参数列表
args = shlex.split(command)
args_str = ", ".join([f'"{arg}"' for arg in args])
aaa = args_str.replace("\"--", "\n\"--")
print(aaa)