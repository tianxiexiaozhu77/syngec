'''
只有主谓或者谓语
'''
from multiprocessing import Pool
import numpy as np
import pickle
import sys
import gc
from tqdm import tqdm
from fairseq.data import LabelDictionary
import pickle

path = "/opt/data/private/friends/tzc/SynGEC-main/data/iwslt/parse_iwslt_00/train.en.conll_predict_gopar"
label_file = "/opt/data/private/friends/tzc/SynGEC-main/data/dicts/syntax_label_gec.dict"   # 注意
out_path = "/opt/data/private/friends/tzc/SynGEC-main/output.txt"
out_pkl_path = "/opt/data/private/friends/tzc/SynGEC-main/output.pkl"
label_dict = LabelDictionary.load(label_file)


with open(path,encoding="utf-8",mode="r" ) as f_in:
    conll_chunks = [conll_chunk for conll_chunk in f_in.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
    print(len(conll_chunks))
    n = 0
    nn = 0
    nnn = 0
    error_ls = []
    append_eos = True
    all_nsubj_predicate = []

    for num_i, chunk in enumerate(conll_chunks):
        nnn = nnn + 1
        chunk = chunk.split("\n")

        #######################################
        
        for num, l in enumerate(chunk):  # 遍历chunk中的每一行
            infos = l.rstrip().split()  # infos:['1', 'What', '_', '_', '_', '_', '0', 'root', '_', '_']
            flag = 0
            if int(infos[6]) == 0:  # 找到根节点对应的行
                flag = 1
                root_v = int(infos[0])  # 找到根节点所支配的谓词的num
                assert root_v == (num + 1)
                a_nsubj_predicate = {}
                for snum, sl in enumerate(chunk):  # 重新遍历chunk,找谓词所支配的节点
                    sinfos = sl.rstrip().split() 
                    if int(sinfos[6]) == root_v and sinfos[7] == "nsubj":  # 主动
                        a_nsubj_predicate = {"nsubj":int(sinfos[0])-1,"root_v":root_v-1}
                    elif int(sinfos[6]) == root_v and sinfos[7] == "nsubjpass":  # 被动主语
                        a_nsubj_predicate = {"nsubj":int(sinfos[0])-1,"root_v":root_v-1} 
                    elif int(sinfos[6]) == root_v and sinfos[7] == "csubj":  # 主从
                        a_nsubj_predicate = {"nsubj":int(sinfos[0])-1,"root_v":root_v-1}
                    elif int(sinfos[6]) == root_v and sinfos[7] == "csubjpass":  # 被主从
                        a_nsubj_predicate = {"nsubj":int(sinfos[0])-1,"root_v":root_v-1}
                    else:  # 只有谓词，没有主语、被动主语、主语从句、被动主语从句
                        continue
                
                if len(a_nsubj_predicate) == 0:    
                    # print(ss) 
                    # s = ""          
                    # for ss in chunk:
                    #     print(ss)
                    #     s = s + " " + ss.split()[1]
                    # print(s)
                    # print("______________________________________")
                    a_nsubj_predicate = {"root_v":root_v-1}
                    n = n + 1
                    error_ls.append(chunk)
                all_nsubj_predicate.append(a_nsubj_predicate)
            else:
                continue
    print(len(all_nsubj_predicate))
    n_0 = 0
    n_1 = 0
    n_2h = 0
    for i_a,aa_nsubj_predicate in enumerate(all_nsubj_predicate):
        if len(aa_nsubj_predicate) == 0:
            print(0,":",aa_nsubj_predicate)
            n_0 = n_0 + 1
        elif len(aa_nsubj_predicate) == 1:
            n_1 = n_1 + 1
        else:
            n_2h = n_2h + 1
        
    print(n_0,"\n",n_1,"\n",n_2h)
    print(n_0 + n_1 + n_2h)

    # with open(out_path, encoding="utf-8", mode="a") as f_out:
    #     for ii in error_ls:
    #         for jj in ii:
    #             f_out.write(jj)
    #             f_out.write("\n")
    #         f_out.write("\n")
    #     f_out.close()
    with open(out_pkl_path, 'wb') as f1:
        pickle.dump(all_nsubj_predicate, f1)
    with open(out_pkl_path, 'rb') as f2:
        data_loaded = pickle.load(f2)

    print(data_loaded)




