import argparse

def main(log_path,match_s1,match_s2,match_s3,num,num_update):
    list_all_best = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if match_s1 in line and match_s2 in line and match_s3 in line:
                dict_item = dict()
                list_str = line[line.index("epoch"):].strip().split(" | ")
                if num_update != None:
                    if int(list_str[11].split()[1]) > int(num_update):
                        break
                for i in list_str:
                    k_v = i.split()
                    dict_item[k_v[0]] = k_v[1]
                dict_item.pop('valid')
                # if dict_item["bleu"] == dict_item["best_bleu"]:  # 加上这个不对
                list_all_best.append(dict_item)
    sort_bleu = sorted(list_all_best, key=lambda x: float(x["bleu"]),reverse=True)
    # print("排序结果：",sort_bleu[:5])
    
    print("\n%-10s\t %-10s\t %-10s\t %-10s"%("best_order","epoch","bleu","best_bleu"))
    print("—"*54)
    for i in range(num):  # 输出前20个bleu和对应的ckpt
        print("%-10s\t %-10s\t %-10s\t %-10s"%(i+1,sort_bleu[i]["epoch"],sort_bleu[i]["bleu"],sort_bleu[i]["best_bleu"]))
        if (i+1) % 5 == 0:
            print("—"*54)
    
    
    print("checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt"\
            %(sort_bleu[0]["epoch"].lstrip("0"),sort_bleu[1]["epoch"].lstrip("0"),sort_bleu[2]["epoch"].lstrip("0"),sort_bleu[3]["epoch"].lstrip("0"),sort_bleu[4]["epoch"].lstrip("0")))
    print("1-5: ","—"*48)
    print("checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt"\
            %(sort_bleu[5]["epoch"].lstrip("0"),sort_bleu[6]["epoch"].lstrip("0"),sort_bleu[7]["epoch"].lstrip("0"),sort_bleu[8]["epoch"].lstrip("0"),sort_bleu[9]["epoch"].lstrip("0")))
    print("6—10: ","—"*47)
    print("checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt"\
            %(sort_bleu[10]["epoch"].lstrip("0"),sort_bleu[11]["epoch"].lstrip("0"),sort_bleu[12]["epoch"].lstrip("0"),sort_bleu[13]["epoch"].lstrip("0"),sort_bleu[14]["epoch"].lstrip("0")))
    print("11—15: ","—"*46)
    print("checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt checkpoint%s.pt"\
            %(sort_bleu[15]["epoch"].lstrip("0"),sort_bleu[16]["epoch"].lstrip("0"),sort_bleu[17]["epoch"].lstrip("0"),sort_bleu[18]["epoch"].lstrip("0"),sort_bleu[19]["epoch"].lstrip("0")))
    print("16—20: ","—"*46)

if __name__ == "__main__":
    '''
    根据bleu值输出排序好的epoch checkpoint
    '''
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--log_path',  
                        help='Input checkpoint file paths.',
                        default="/opt/data/private/friends/tzc/SynGEC-main/src/src_syngec/syngec_model/2023.06.24_syntax_glat_en_de_17_19_24.log") # nargs='+',
    parser.add_argument('--num_update',  
                        help='The largest update step you want to compare',
                        default=None) # nargs='+',

    args = parser.parse_args()
    # log_path = "/opt/data/private/friends/wj/GLAT/bash/2023.05.04_glat_nat_train_log_2.log"
    match_s1 = "valid on \'valid\' subset"
    match_s2 = "bleu"
    match_s3 = "best_bleu"
    num = 20  # 输出前num个最好的bleu和对应的ckpt
    main(args.log_path,match_s1,match_s2,match_s3,num,args.num_update)
    
# best_order       epoch           bleu            best_bleu 
# ——————————————————————————————————————————————————————
# 1                222             16.55           16.55     
# 2                334             16.52           16.55     
# 3                261             16.5            16.55     
# 4                237             16.48           16.55     
# 5                199             16.46           16.46 
