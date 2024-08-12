source activate syndecoder
python /opt/data/private/zjx/syngec/src/src_syngec/fairseq-0.10.2/scripts/bleu_order_ckpt.py \
--log_path /opt/data/private/zjx/syngec/bash/english_exp/iwslt_syntax/log/2024.08.09_syntax_glat_iwslt_distill_de_en.log \

# best_order       epoch           bleu            best_bleu 
# ——————————————————————————————————————————————————————
# 1                458             30.98           30.98     
# 2                346             30.82           30.82     
# 3                459             30.82           30.98     
# 4                652             30.82           30.98     
# 5                343             30.81           30.81     
# ——————————————————————————————————————————————————————
# 6                348             30.76           30.82     
# 7                338             30.75           30.75     
# 8                463             30.73           30.98     
# 9                396             30.71           30.82     
# 10               412             30.71           30.82     
# ——————————————————————————————————————————————————————
# 11               653             30.71           30.98     
# 12               460             30.7            30.98     
# 13               456             30.69           30.82     
# 14               654             30.69           30.98     
# 15               655             30.68           30.98     
# ——————————————————————————————————————————————————————
# 16               725             30.68           30.98     
# 17               623             30.67           30.98     
# 18               717             30.67           30.98     
# 19               454             30.66           30.82     
# 20               276             30.65           30.65     
# ——————————————————————————————————————————————————————
# checkpoint458.pt checkpoint346.pt checkpoint459.pt checkpoint652.pt checkpoint343.pt
# 1-5:  ————————————————————————————————————————————————
# checkpoint348.pt checkpoint338.pt checkpoint463.pt checkpoint396.pt checkpoint412.pt
# 6—10:  ———————————————————————————————————————————————
# checkpoint653.pt checkpoint460.pt checkpoint456.pt checkpoint654.pt checkpoint655.pt
# 11—15:  ——————————————————————————————————————————————
# checkpoint725.pt checkpoint623.pt checkpoint717.pt checkpoint454.pt checkpoint276.pt
# 16—20:  ——————————————————————————————————————————————