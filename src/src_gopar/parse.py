from supar import Parser
import sys
import pickle
import torch
import os
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
# CUDA_LAUNCH_BLOCKING=1

# CoNLL_SUFFIX=conll_predict_gopar
# IN_FILE=/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt
# OUT_FILE=/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar
# model=/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/emnlp2022_syngec_biaffine-dep-electra-en-gopar
# CUDA_VISIBLE_DEVICES=5 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &
#                              python ../../src/src_gopar/parse.py $tgt_file $tgt_file.conll_predict $vanilla_parser_path

# src_dir = sys.argv[1]
# save_dir = sys.argv[2]
# model_dir = sys.argv[3]

src_dir = "/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt"
save_dir = "/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar"
model_dir = "/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/emnlp2022_syngec_biaffine-dep-electra-en-gopar"

def load(filename):
    sents = []
    with open(filename, 'r') as f:
        for line in f:
            res = line.rstrip().split()
            if res:
                sents.append(res)
    return sents

dep = Parser.load(model_dir)
input_sentences = load(src_dir)
res = dep.predict(input_sentences, verbose=False, buckets=32, batch_size=3000, prob=True)
probs = []

with open(save_dir, 'w') as f:
    for r, t in zip(res, res.probs):
        f.write(str(r) + "\n")
        t1, t2 = t.split([1, len(t[0])-1], dim=-1)
        t = torch.cat((t2, t1), dim=-1)
        t = torch.cat((t, t.new_zeros((1, len(t[0])))))
        t.masked_fill_(torch.eye(len(t[0])) == 1.0, 1.0)
        t_list = t.numpy()
        probs.append(t_list)

with open(save_dir + ".probs", "wb") as o:
    pickle.dump(probs, o)

# with open(save_dir + ".probs_raw", "w") as o:
#     o.write(probs)

print("结束")