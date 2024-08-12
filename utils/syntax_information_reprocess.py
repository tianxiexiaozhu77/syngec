from multiprocessing import Pool
import numpy as np
import pickle
import sys
import gc
from tqdm import tqdm
from fairseq.data import LabelDictionary


num_workers = 64  # 64
src_file = sys.argv[1]
conll_suffix = sys.argv[2]
mode = sys.argv[3]
structure = sys.argv[4]

# src_file = "/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt"
# conll_suffix = "conll_predict_gopar"
# mode = "conll"  # "conll" "probs"
# structure = "transformer"


input_prefix = f"{src_file}.{conll_suffix}"

if structure == "transformer":
    output_prefix = input_prefix + "_np"  # 注意
else:
    output_prefix = input_prefix + "_bart_np"  # 注意

if mode in ["dpd", "probs"]:  # 需要subword对齐的 dpd没有在这里实现
    input_file = input_prefix + f".{mode}"
    output_file = output_prefix + f".{mode}"
else:
    input_file = input_prefix  # '/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar'
    output_file = output_prefix  # '/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar_np'

swm_list = []
if structure == "transformer":
    swm_file = src_file + ".swm"  # 注意 '/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.swm'
else:
   swm_file = src_file + ".bart_swm"  # 注意

swm_list = [[int(i) for i in line.rstrip("\n").split()] for line in open(swm_file, "r").readlines()]
label_file = "/opt/data/private/friends/tzc/SynGEC-main/data/dicts/syntax_label_gec.dict"   # 注意
label_dict = LabelDictionary.load(label_file)

def create_sentence_syntax_graph_matrix(chunk, append_eos=True):
    chunk = chunk.split("\n")
    seq_len = len(chunk)
    if append_eos:
        seq_len += 1
    incoming_matrix = np.ones((seq_len, seq_len))
    incoming_matrix *= label_dict.index("<nadj>")  # outcoming矩阵可以通过转置得到
    for l in chunk:
        infos = l.rstrip().split()  # infos:['1', 'What', '_', '_', '_', '_', '0', 'root', '_', '_']
        child, father = int(infos[0]) - 1, int(infos[6]) - 1  # 该弧的孩子和父亲 为什么减1，因为张宇的代码中根节点是0，这里根节点是-1（eos）
        if father == -1:
            father = len(chunk) # EOS代替Root
        rel = infos[7]  # 该弧的关系标签
        incoming_matrix[child,father] = label_dict.index(rel)
    return incoming_matrix
'''
得到邻接矩阵conll: 储存文件src.txt.conll_predict_gopar_np
incoming_matrix，2是<nadj>，是没有弧的意思
array([[ 2.,  2.,  2.,  2.,  2., 48.],
       [16.,  2.,  2.,  2.,  2.,  2.],
       [ 2.,  2.,  2.,  9.,  2.,  2.],
       [31.,  2.,  2.,  2.,  2.,  2.],
       [45.,  2.,  2.,  2.,  2.,  2.],
       [ 2.,  2.,  2.,  2.,  2.,  2.]])   # 这个是eos，代替root
'''

def use_swm_to_adjust_matrix(matrix, swm, append_eos=True):
    if append_eos:
        swm.append(matrix.shape[0]-1)  # 在最后添加上eos当做root
    new_matrix = np.zeros((len(swm), len(swm)))
    for i in range(len(swm)):
        for j in range(len(swm)):
            new_matrix[i,j] = matrix[swm[i],swm[j]]
    return new_matrix
'''
得到概率矩阵probs:储存文件src.txt.conll_predict_gopar_np.probs
new_matrix
array([[1.00000000e+00, 2.35405478e-05, 1.00099192e-06, 2.70750094e-02, 1.08126665e-06, 9.71481740e-01],  # 9.71481740e-01概率最大，所以有弧
       [9.46719289e-01, 1.00000000e+00, 2.53373200e-05, 5.27837090e-02, 1.13499414e-06, 4.66937257e-04],  # 9.46719289e-01概率最大，所以有弧
       [3.70726833e-04, 3.08131457e-05, 1.00000000e+00, 9.99597967e-01, 3.91068440e-08, 2.25652954e-07],  # 9.99597967e-01概率最大，所以有弧
       [9.91876006e-01, 7.49446277e-04, 3.96957694e-06, 1.00000000e+00, 1.95019084e-06, 7.28643499e-03],  # 9.91876006e-01概率最大，所以有弧
       [9.89951193e-01, 2.37164815e-04, 1.66921029e-06, 9.71460342e-03, 1.00000000e+00, 9.48902161e-05],  # 9.89951193e-01概率最大，所以有弧
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])   # 这个是eos，代替root
'''
def convert_list_to_nparray(matrix):
    return np.array(matrix)

def convert_probs_to_nparray(t):
    matrix, swm = t
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return use_swm_to_adjust_matrix(matrix, swm)


def convert_conll_to_nparray(t):
    conll_chunk, swm = t
    incoming_matrix = create_sentence_syntax_graph_matrix(conll_chunk)
    incoming_matrix = use_swm_to_adjust_matrix(incoming_matrix, swm)
    return incoming_matrix


def data_format_convert():  # mode:probs:'/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar_np.probs'
    with open(output_file, 'wb') as f_out:  # mode:conll :output_file:'/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar_np'
        res = []
        with Pool(num_workers) as pool:
            if mode == "dpd":
                with open(input_file, 'rb') as f_in:
                    gc.disable()
                    # arr_list = pickle.load(f_in)
                    gc.enable()
                    assert len(swm_list) == len(arr_list), print(len(swm_list), len(arr_list))
                    for mat in pool.imap(convert_list_to_nparray, tqdm(arr_list), chunksize=256):
                        res.append(mat)
            elif mode == "probs":
                with open(input_file, 'rb') as f_in:  # '/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar.probs'
                    gc.disable()
                    # arr_list = pickle.load(f_in)
                    gc.enable()
                    assert len(swm_list) == len(arr_list), print(len(swm_list), len(arr_list))
                    for mat in pool.imap(convert_probs_to_nparray, tqdm(zip(arr_list, swm_list)), chunksize=256):    
                        res.append(mat)
            elif mode == "conll":
                with open(input_file, 'r') as f_in:  #'/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar'
                    conll_chunks = [conll_chunk for conll_chunk in f_in.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
                    for mat in pool.imap(convert_conll_to_nparray, tqdm(zip(conll_chunks, swm_list)), chunksize=256):
                        res.append(mat)
        pickle.dump(res, f_out)  # 先不保存


if __name__ == "__main__":
    data_format_convert()
