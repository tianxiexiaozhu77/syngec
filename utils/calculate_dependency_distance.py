from tqdm import tqdm
import sys
from collections import defaultdict
import pickle
from itertools import zip_longest
from multiprocessing import Pool
import numpy as np
import re
# $TEST_SRC_FILE".${CoNLL_SUFFIX}":../../data/conll14_test/src.txt.conll_predict_gopar
# $PROCESSED_DIR/test.swm.src:../../preprocess/english_conll14_with_syntax_transformer/test.swm.src
# $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd":../../data/conll14_test/src.txt.conll_predict_gopar_np.dpb
conll_file = sys.argv[1]
swm_file = sys.argv[2]
output_file = sys.argv[3]
reference_file_gopar = sys.argv[4]
reference_file_swm = sys.argv[5]

# conll_file = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/parse_iwslt_00/train.en.conll_predict_gopar"
# swm_file = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/parse_iwslt_00/train.en.swm"
# output_file = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/parse_iwslt_00/train.en.conll_predict_gopar_np.dpd"
# conll_file = "/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.txt.conll_predict_gopar"
# swm_file = "/opt/data/private/friends/tzc/SynGEC-main/data/preprocess/english_conll14_with_syntax_transformer/test.swm.src"
# output_file = "/opt/data/private/friends/tzc/SynGEC-main/data/conll14_test/src.tf_xt.conll_predict_gopar_np.dpb"  # 得到依存距离文件
# reference_file_gopar = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/synnat_data/train.en"
# reference_file_swm = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/behind_moses_before_bpe/train.en"
# print("____%____%"*10)
# print(conll_file,swm_file,output_file,reference_file_gopar,reference_file_swm)
# print("____*____*"*10)


num_workers = 64  # 64
fg = open(reference_file_gopar, "r")
f_gopar = fg.readlines()
fs = open(reference_file_swm, "r")
f_swm = fs.readlines()

def use_swm_to_adjust_path(path, swm_list, num, chunk):
    new_path = []
    try:
        for i in swm_list:
            new_path.append(path[int(i)])
    except:
        print("_"*100)
        print("第",num+1,"个样本:") # ,swm_list ,"chunk:",chunk
        print("在behind_moses_before_bpe中：\n",f_swm[num])
        print("在synnat_data中：\n",f_gopar[num])
        aa = ""
        bb = ""
        for ii in chunk:
            jj = ii.split("\t")
            aa = aa + " " + jj[1]
            bb = bb + " " + jj[0]
        print("在本脚本的conll_predict_gopar文件中\n",aa[1:])
        print("\n",bb[1:])
    return new_path


def has_root(tree):
    for node in tree:
        if node.split("\t")[6] == "0":
            return True
    return False

def construct_tree(chunk, swm_list, num, chunk_alter=None,):
    tree = defaultdict(set)
    chunk_orig = chunk
    if not has_root(chunk):
        print(chunk)
    for line in chunk:
        li = line.split("\t")
        tree[li[6]].add(li[0])  # tree:{'0': {'1'}, '3': {'2', '4'}, '1': {'3'}, '6': {'5'}, '4': {'6'}})
    paths = get_path(tree)  
    path_list = []
    try:
        for i in range(1, len(paths)):
            path_list.append(paths[str(i)])  # 根据规则构造的Golden树可能出现环路
        new_path = use_swm_to_adjust_path(path_list, swm_list, num=num,chunk=chunk)  # [['0', '1'], ['0', '1', '3', '2'], ['0', '1', '3'], ['0', '1', '3', '4'], ['0', '1', '3', '4', '6', '5'], ['0', '1', '3', '4', '6']]
    except:
        tree = defaultdict(set)
        for line in chunk_alter:
            li = line.split("\t")
            tree[li[6]].add(li[0])
        paths = get_path(tree)
        path_list = []
        for i in range(1, len(paths)):
            path_list.append(paths[str(i)])
        new_path = use_swm_to_adjust_path(path_list, swm_list, num=num, chunk=chunk)
    new_path.append(paths["0"])  # [['0', '1'], ['0', '1', '3', '2'], ['0', '1', '3'], ['0', '1', '3', '4'], ['0', '1', '3', '4', '6', '5'], ['0', '1', '3', '4', '6'], ['0']]
    # print(tree, new_path)
    return new_path


def get_path(tree):
    paths = {}
    def dfs(node, path):
        paths[node] = path
        if node in tree.keys():
            for next_node in tree[node]:
                dfs(next_node, path + [next_node])
    dfs("0", ["0"])
    return paths
'''
# paths
# {'0': ['0'], '1': ['0', '1'], '3': ['0', '1', '3'], 
# '2': ['0', '1', '3', '2'], '4': ['0', '1', '3', '4'],
#  '6': ['0', '1', '3', '4', '6'], 
#  '5': ['0', '1', '3', '4', '6', '5']}
'''

def get_nearest_ancestor(path_a, path_b):
    idx = -1
    for i in range(min(len(path_a), len(path_b))):
        if path_a[i] == path_b[i]:
            idx += 1
        else:
            break
    return path_a[idx]

def calculate_dpd(path, pruned_nodes):
    # dpd_matrix = []
    dpd_matrix = np.zeros((len(path), len(path)))
    max_dist = 0
    for i in range(len(path)):
        for j in range(len(path)):
            dpd_matrix[i][j] = len(set(path[i]) ^ set(path[j]))
    
    max_dist = np.amax(dpd_matrix)
    if pruned_nodes:    # 人为调整依存距离
        # print("Before Pruning:")
        # print(dpd_matrix)
        for i in range(len(path)):
            for j in range(len(path)):
                nodes = set(path[i]) ^ set(path[j])
                if len(nodes) == 0:
                    continue
                nodes.add(get_nearest_ancestor(path[i], path[j]))
                # print(i, path[i], j, path[j],nodes, pruned_nodes)
                if len(nodes & pruned_nodes) > 0:
                    dpd_matrix[i][j] = max_dist
        # print("After Pruning:")
    return dpd_matrix


def get_node_with_special_tag(chunk, tags):
    nodes = set()
    for line in chunk:
        li = line.split("\t")
        if li[7] in tags:
            nodes.add(li[0])
    return nodes

def solve(t):
    # chunk, swm_list, chunk_alter = t
    i, (chunk, swm_list) = t
    chunk = chunk.split("\n")
    assert chunk is not None
    # if chunk_alter:
    #     chunk_alter = chunk_alter.split("\n")
    swm_list = swm_list.strip().split()
    pruned_nodes = get_node_with_special_tag(chunk, "R")
    # pruned_nodes = None
    # return calculate_dpd(construct_tree(chunk, swm_list, chunk_alter), pruned_nodes)
    return calculate_dpd(construct_tree(chunk, swm_list, num=i), pruned_nodes)

res = []
with open(conll_file, "r") as f1:
    with open(swm_file, "r") as f2:
        chunks = [c for c in f1.read().split("\n\n") if c]
        swm_lists = f2.readlines()
        assert len(chunks) == len(swm_lists), print(len(chunks), len(swm_lists))
        with Pool(num_workers) as pool:
            for dpd in pool.imap(solve, enumerate(tqdm(zip_longest(chunks, swm_lists),position=0,leave=True,total=len(swm_lists))), chunksize=64):
                if dpd is not None:
                    res.append(dpd)

'''
# dpd
# 得到距离矩阵dpd:储存文件conll14_test/src.txt.conll_predict_gopar_np.dpb
# array([[0., 2., 1., 2., 4., 3., 1.],
#        [2., 0., 1., 2., 4., 3., 3.],
#        [1., 1., 0., 1., 3., 2., 2.],
#        [2., 2., 1., 0., 2., 1., 3.],
#        [4., 4., 3., 2., 0., 1., 5.],
#        [3., 3., 2., 1., 1., 0., 4.],
#        [1., 3., 2., 3., 5., 4., 0.]])
'''

with open(output_file, "wb") as o:
    pickle.dump(res, o)

fg.close()
fs.close()


