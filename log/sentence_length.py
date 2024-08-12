# 读取包含BPE后文件内容的文件 "ref" 通常用于指代人工或事先确定的参考翻译
import re

# 从文件中提取句子长度
def extract_sentence_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentence_lengths = [len(line.split()) for line in lines]
    return sentence_lengths

# 从文件中提取BLEU值
def extract_bleu_values(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    bleu_values = [float(re.search(r'BLEU4 = ([\d.]+)', line).group(1)) for line in lines]
    return bleu_values

# 文件路径
lengths_file_path = '/opt/data/private/zjx/SynGEC-main/log/iwslt_distill/glat_distll_iwslt_deen.log.ref'
bleu_values_file_path = '/opt/data/private/zjx/SynGEC-main/log/iwslt_distill/glat_distll_iwslt_deen_log.log'

# 提取句子长度和BLEU值
sentence_lengths = extract_sentence_lengths(lengths_file_path)
bleu_values = extract_bleu_values(bleu_values_file_path)
if len(sentence_lengths) != len(bleu_values):
    print("长度出错!")
    exit()

# 将句子长度分组
length_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, float('inf'))]

# 创建字典以存储每个长度范围内的BLEU值和计数
average_bleu_by_range = {range_: {'total_bleu': 0, 'count': 0} for range_ in length_ranges}

# 将BLEU值添加到对应的长度范围
for length, bleu in zip(sentence_lengths, bleu_values):
    for range_ in length_ranges:
        if range_[0] <= length < range_[1]:
            average_bleu_by_range[range_]['total_bleu'] += bleu
            average_bleu_by_range[range_]['count'] += 1
            break

# 计算每个长度范围的平均BLEU值
for range_, data in average_bleu_by_range.items():
    average_bleu = data['total_bleu'] / data['count'] if data['count'] > 0 else 0
    print(f"Length Range {range_}: Average BLEU = {average_bleu}")




####################计算句子长度####################
# with open('/opt/data/private/zjx/SynGEC-main/log/glat_ctc_enro.log.ref', 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # 计算每行句子的长度并输出
# for i, line in enumerate(lines, 1):
#     sentence_length = len(line.split())
#     print(f"Line {i}: Sentence length = {sentence_length}")




####################得到bleu####################
# import re

# # 读取包含日志内容的文件
# with open('/opt/data/private/zjx/SynGEC-main/log/glat_ctc_enro_log.log', 'r', encoding='utf-8') as file:
#     log_content = file.read()

# # 使用正则表达式提取BLEU4值
# matches = re.findall(r'BLEU4 = ([\d.]+)', log_content)
# sum_bleu = 0
# for x in matches:
#     if float(x)>100:
#         print(x)
#     sum_bleu = sum_bleu + float(x)

# print(sum_bleu)
# print(sum_bleu/1999)
# 打印提取的结果
# for i, match in enumerate(matches):
#     print(f"Line {i+1}: BLEU4 = {match}")

