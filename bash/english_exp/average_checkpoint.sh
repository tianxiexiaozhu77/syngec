# cd /opt/data/private/friends/tzc/checkpoint/checkpoints_syntax_nat/nat34
# cd /opt/data/private/friends/tzc/checkpoint/checkpoint_glat_ctc/5
# cd /opt/data/private/zjx/ckpt/iwslt_distill_de_en_syntax/6
# cd /opt/data/private/zjx/ckpt/iwslt_distill_de_en_syntax/20
# cd /opt/data/private/zjx/ckpt/iwslt_raw_de_en_syntax/6
cd /opt/data/private/zjx/ckpt/iwslt_raw_de_en/1
# cd /opt/data/private/friends/tzc/checkpoint/checkpoint_glat/8
# cd /opt/data/private/friends/tzc/checkpoint/checkpoint_glat/syngec/14
source activate syngec
export CUDA_VISIBLE_DEVICES=0
# 1
python /opt/data/private/zjx/data/syngec/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
--input checkpoint98.pt checkpoint96.pt checkpoint142.pt checkpoint94.pt checkpoint150.pt \
--output checkpoint_ave.pt

# 1.checkpoint475.pt checkpoint474.pt checkpoint463.pt checkpoint477.pt checkpoint466.pt
# 2.checkpoint498.pt checkpoint493.pt checkpoint464.pt checkpoint462.pt checkpoint494.pt
# 3.checkpoint463.pt checkpoint473.pt checkpoint459.pt checkpoint461.pt checkpoint434.pt
# 4.checkpoint499.pt checkpoint504.pt checkpoint491.pt checkpoint490.pt checkpoint498.pt
# 5.checkpoint288.pt checkpoint220.pt checkpoint302.pt checkpoint294.pt checkpoint300.pt # 有问题没跑完
# 6. checkpoint492.pt checkpoint448.pt checkpoint475.pt checkpoint487.pt checkpoint449.pt
# 2025.02.25_glat_raw_de_en_save1 checkpoint98.pt checkpoint96.pt checkpoint142.pt checkpoint94.pt checkpoint150.pt
# 2
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint614.pt checkpoint656.pt checkpoint711.pt checkpoint717.pt checkpoint599.pt \
# --output checkpoint_ave2.pt
# # 3
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint652.pt checkpoint654.pt checkpoint712.pt checkpoint609.pt checkpoint626.pt \
# --output checkpoint_ave3.pt
# # 4
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint695.pt checkpoint701.pt checkpoint720.pt checkpoint719.pt checkpoint721.pt \
# --output checkpoint_ave4.pt
# # 5
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint723.pt checkpoint524.pt checkpoint683.pt checkpoint705.pt checkpoint727.pt \
# --output checkpoint_ave5.pt
# # 6
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint531.pt checkpoint664.pt checkpoint399.pt checkpoint632.pt checkpoint633.pt \
# --output checkpoint_ave6.pt
# # 7
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint697.pt checkpoint710.pt checkpoint617.pt checkpoint630.pt checkpoint631.pt \
# --output checkpoint_ave7.pt
# # 8
# python /opt/data/private/zjx/SynGEC-main/src/src_syngec/fairseq-0.10.2/scripts/average_checkpoints.py \
# --input checkpoint634.pt checkpoint666.pt checkpoint675.pt checkpoint648.pt checkpoint706.pt \
# --output checkpoint_ave8.pt



# checkpoint531.pt checkpoint441.pt checkpoint449.pt checkpoint506.pt checkpoint527.pt  BLEU4 = 33.34
# checkpoint728.pt checkpoint582.pt checkpoint653.pt checkpoint748.pt checkpoint804.pt  BLEU4 = 33.49
# checkpoint658.pt checkpoint661.pt checkpoint797.pt checkpoint752.pt checkpoint767.pt  BLEU4 = 33.58
# checkpoint648.pt checkpoint724.pt checkpoint729.pt checkpoint745.pt checkpoint639.pt  BLEU4 = 33.55
# checkpoint944.pt checkpoint589.pt checkpoint648.pt checkpoint724.pt checkpoint729.pt  BLEU4 = 33.62
# Generate test with beam=5: BLEU4 = 32.52, 63.4/39.1/26.2/18.1 checkpoint193.pt checkpoint137.pt checkpoint192.pt checkpoint188.pt checkpoint172.pt 
# checkpoint1189.pt checkpoint1234.pt checkpoint1239.pt checkpoint652.pt checkpoint1255.pt 32.73
# checkpoint758.pt checkpoint888.pt checkpoint652.pt checkpoint889.pt checkpoint797.pt
# Generate test with beam=5: BLEU4 = 32.72, 64.3/39.8/26.7/18.5 (BP=0.976, ratio=0.976, syslen=49413, reflen=50606)
# best_order       epoch           bleu            best_bleu 
# ——————————————————————————————————————————————————————
# 1                196             33.17           33.17     
# 2                222             33.17           33.17     
# 3                276             33.16           33.17     
# 4                284             33.16           33.17     
# 5                286             33.15           33.17     
# ——————————————————————————————————————————————————————
# 6                288             33.15           33.17     
# 7                297             33.15           33.17     
# 8                178             33.14           33.14     
# 9                270             33.12           33.17     
# 10               189             33.1            33.14     
# ——————————————————————————————————————————————————————
# 11               387             33.1            33.17     
# 12               185             33.08           33.14     
# 13               209             33.08           33.17     
# 14               251             33.07           33.17     
# 15               206             33.06           33.17     
# ——————————————————————————————————————————————————————
# 16               207             33.06           33.17     
# 17               250             33.06           33.17     
# 18               320             33.06           33.17     
# 19               180             33.05           33.14     
# 20               230             33.05           33.17     
# ——————————————————————————————————————————————————————
# checkpoint196.pt checkpoint222.pt checkpoint276.pt checkpoint284.pt checkpoint286.pt
# 1-5:  ————————————————————————————————————————————————
# checkpoint288.pt checkpoint297.pt checkpoint178.pt checkpoint270.pt checkpoint189.pt
# 6—10:  ———————————————————————————————————————————————
# checkpoint387.pt checkpoint185.pt checkpoint209.pt checkpoint251.pt checkpoint206.pt
# 11—15:  ——————————————————————————————————————————————
# checkpoint207.pt checkpoint250.pt checkpoint320.pt checkpoint180.pt checkpoint230.pt
# 16—20:  ——————————————————————————————————————————————


# checkpoint848.pt checkpoint892.pt checkpoint766.pt checkpoint779.pt checkpoint876.pt 31.29 33.35
# checkpoint265.pt checkpoint264.pt checkpoint278.pt checkpoint250.pt checkpoint267.pt  # 25.24 26.55
# checkpoint265.pt checkpoint264.pt checkpoint250.pt checkpoint267.pt checkpoint249.pt
# checkpoint137.pt checkpoint138.pt checkpoint141.pt checkpoint132.pt checkpoint125.pt
# checkpoint528.pt checkpoint474.pt checkpoint526.pt checkpoint434.pt checkpoint514.pt
# checkpoint547.pt checkpoint528.pt checkpoint549.pt checkpoint474.pt checkpoint526.pt
# checkpoint547.pt checkpoint666.pt checkpoint528.pt checkpoint549.pt checkpoint685.pt
# checkpoint729.pt checkpoint696.pt checkpoint474.pt checkpoint741.pt checkpoint682.pt
# 生成在/opt/data/private/friends/tzc/checkpoint/checkpoint_glat/syngec/2/checkpoint_average.pt
# /opt/data/private/friends/tzc/checkpoint/checkpoint_glat/5/checkpoint_average.pt

# 需要该两个地方
# 1. cd后面的路径
# 2. --input后面的5个checkpoint