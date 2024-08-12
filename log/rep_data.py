def remove_repeat_n_gram(input_file):
    with open(input_file, 'r', encoding='utf-8') as fin:
        cnt = 0
        tot = 0
        pos = 0
        for line in fin:
            line = line.strip()
            if line[0] == 'D':
                lst = line.split(' ')
                lst[0] = lst[0].split('\t')[-1]
                for i in range(1,len(lst)):
                    if lst[i] == lst[i-1] :
                        print("line: {}".format(line))
                        cnt += 1
                        # if len(lst) > 10 and len(lst) < 20:
                        #     print("pos: {}".format(line))
                        #     print("----------------------------------------------------------------")
                tot += len(lst)
            pos += 1
        print((cnt/tot)*100,"%")

def get_generate_sentence(path):
    # import pdb
    # , open("generate_sentences_renew.ref", 'w', encoding='utf-8') as fin_t, open("generate_sentences_renew.hyp", 'w', encoding='utf-8') as fin_h
    with open(path, 'r', encoding='utf-8') as fin, \
         open("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/de_syntax.src", 'w', encoding='utf-8') as fin_s, \
         open("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en_syntax.ref", 'w', encoding='utf-8') as fin_t, \
         open("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en_syntax.hyp", 'w', encoding='utf-8') as fin_h:
        for line in fin:
            if line[0] == 'S':
                # pdb.set_trace()
                sentence = line.split('\t')[-1]
                fin_s.write(sentence)
            elif line[0] == 'T':
                # pdb.set_trace()
                sentence = line.split('\t')[-1]
                fin_t.write(sentence)
            elif line[0] == 'D':
                # pdb.set_trace()
                sentence = line.split('\t')[-1]
                fin_h.write(sentence)

def reorder_res():
    key = []
    # 获取 key
    # with open('/opt/data/private/gp/fairseq-0.10.2/tacl-res/cmlm/res.log', 'r', encoding='utf-8') as fin, open('/opt/data/private/gp/fairseq-0.10.2/tacl-res/cmlm/cmlm-iter9.ref', 'w', encoding='utf-8') as fout:
    #     for line in fin:
    #         if line.startswith('T-'):
    #             fout.write(line.split('\t')[-1])
    with open('/opt/data/private/gp/fairseq-0.10.2/tacl-res/cmlm/res.log', 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.startswith('H-'):
                key.append(line.split('\t')[0])
    search_path = ['/opt/data/private/gp/fairseq-0.10.2/tacl-res/cmlm/12-3/cmlm-12-3-iter9.log']
    # search_path = ['/opt/data/private/gp/fairseq-0.10.2/tacl-res/amom/amom/amom-iter9.log']
    for path in search_path:
        output_path = path.split('/')
        output_path[-1] = output_path[-1].replace('log', 'sys')
        output_path = '/'.join(output_path)
        with open(path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
            k_v = {}
            for line in fin:
                if line.startswith('H-'):
                    k, v = line.split('\t')[0], line.split('\t')[-1]
                    k_v[k] = v
            for k in key:
                try:
                    fout.write(k_v[k])
                except:
                    print(path, k)

if __name__ == '__main__':
    # remove_repeat_n_gram("/opt/data/private/gp/fairseq-0.10.2/inference/cmlm_en_de_raw.log")
    # get_generate_sentence("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/glat_syntax_raw_deen.log")
    # remove_repeat_n_gram("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/glat_raw_deen.log")
    # remove_repeat_n_gram("/opt/data/private/zjx/SynGEC-main/log/iwslt_distill/glat_distll_iwslt_deen.log")
    # remove_repeat_n_gram("/opt/data/private/zjx/SynGEC-main/log/iwslt_distill/glat_syntax_distll_iwslt_deen.log")


    # remove_repeat_n_gram("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/glat_raw_deen.log")
    remove_repeat_n_gram("/opt/data/private/zjx/SynGEC-main/log/iwslt_raw/glat_syntax_raw_deen.log")
    # reorder_res()

    # /opt/data/private/zjx/SynGEC-main/log/enro/glat_ctc_syntax_enro.log