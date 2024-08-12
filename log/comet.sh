# source activate p38
# comet-score -s /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/de.src \
# -t /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en.hyp \
# -r /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en.ref


source activate p38
comet-score -s /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/de_syntax.src \
-t /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en_syntax.hyp \
-r /opt/data/private/zjx/SynGEC-main/log/iwslt_raw/en_syntax.ref