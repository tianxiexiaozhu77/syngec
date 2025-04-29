"""
syntax_eval.py
Compute Acceptability Score × Grammar‑Error# × Parse Validity
and ΔScore against a reference sentence.

"""

import subprocess, sys, importlib, pathlib, random

import spacy
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
    nlp = spacy.load("en_core_web_trf")

import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import language_tool_python as lt

tool = lt.LanguageTool("en-US")                               # 语法检查器
clf_tok = AutoTokenizer.from_pretrained("/opt/data/private/zjx/data/syngec/ckpt/textattack/bert-base-uncased-CoLA")
clf     = AutoModelForSequenceClassification.from_pretrained(
            "/opt/data/private/zjx/data/syngec/ckpt/textattack/bert-base-uncased-CoLA")

BAD_DEPS = {"dep", "orphan", "root"}                          # UD 回退标签


# ---------- 单句评测函数 ----------
def assess(sent: str, verbose: bool = False) -> dict:
    """
    Return dict(valid, acc, err, score)
    """
    # 依存解析合法性
    doc   = nlp(sent)
    valid = int(
        sum(tok.head == tok for tok in doc) == 1 and
        not any(tok.dep_ in BAD_DEPS for tok in doc)
    )

    # Grammar Error
    err_cnt = len(tool.check(sent))

    # Acceptability score (CoLA)
    with torch.no_grad():
        logits = clf(**clf_tok(sent, return_tensors="pt")).logits
        p_acc  = F.softmax(logits, dim=-1)[0, 1].item()       # idx 1 = "acceptable"

    # score = p_acc * valid * 1 / (1 + err_cnt)

    if verbose:
        print(f"  ParseValidity : {valid}")
        print(f"  Acceptability : {p_acc:.4f}")
        print(f"  GrammarErr #  : {err_cnt}")
        # print(f"  CompositeScore: {score:.4f}")
    return dict(valid=valid, acc=p_acc, err=err_cnt, score=score)



if __name__ == "__main__":
    # ref   = "the only country in the world ."
    # cand1 = "the only country country ."
    # cand2 = "the only country in the world ."

    ref   = "she did it for three years ."
    cand1 = "she did for three years ."
    cand2 = "she did it for three years ."

    for label, sent in [("REF", ref), ("CAND‑1", cand1), ("CAND‑2", cand2)]:
        print(f"\n--- {label} ---")
        res = assess(sent, verbose=True)
        globals()[f"score_{label.lower()}"] = res["score"]





# matches = tool.check(sent)
# err_cnt = len(matches)

# # 查看每一个错误的具体位置和提示
# for m in matches:
#     print(f"- Rule: {m.ruleId}")
#     print(f"  Message: {m.message}")
#     print(f"  Offset: {m.offset} to {m.offset + m.errorLength}")
#     print(f"  Error Text: '{sent[m.offset:m.offset + m.errorLength]}'")
#     print(f"  Suggestions: {m.replacements}")
#     print()


# import subprocess, sys
# import pandas as pd
# import spacy
# import torch, torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import language_tool_python as lt
# from tqdm import tqdm

# # 加载 spaCy 模型
# try:
#     nlp = spacy.load("en_core_web_trf")
# except OSError:
#     subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
#     nlp = spacy.load("en_core_web_trf")

# # 语法检查工具
# tool = lt.LanguageTool("en-US")

# # CoLA 接受度分类模型
# clf_tok = AutoTokenizer.from_pretrained("/opt/data/private/zjx/data/syngec/ckpt/textattack/bert-base-uncased-CoLA")
# clf     = AutoModelForSequenceClassification.from_pretrained(
#             "/opt/data/private/zjx/data/syngec/ckpt/textattack/bert-base-uncased-CoLA")

# BAD_DEPS = {"dep", "orphan", "root"}

# def assess(sent: str) -> dict:
#     """
#     对句子进行评估，返回 dict(valid, acc, err, score)
#     """
#     doc = nlp(sent)
#     valid = int(
#         sum(tok.head == tok for tok in doc) == 1 and
#         not any(tok.dep_ in BAD_DEPS for tok in doc)
#     )
#     err_cnt = len(tool.check(sent))
#     with torch.no_grad():
#         logits = clf(**clf_tok(sent, return_tensors="pt")).logits
#         p_acc = F.softmax(logits, dim=-1)[0, 1].item()
#     score = p_acc * valid * 1 / (1 + err_cnt)
#     return dict(valid=valid, acc=p_acc, err=err_cnt, score=score)


# def main():
#     file_path = "/opt/data/private/zjx/data/syngec/output3155_33.xlsx"
#     df = pd.read_excel(file_path)

#     cols = {
#         "T_text": "ref",
#         "H_text_1": "cand1",
#         "H_text_2": "cand2"
#     }

#     results = {name: {"valid": [], "acc": [], "err": [], "score": []} for name in cols.values()}

#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         for col, name in cols.items():
#             sent = str(row[col])
#             res = assess(sent)
#             for key in res:
#                 results[name][key].append(res[key])

#         if idx % 10 == 0:
#             print(f"Processed {idx}/{len(df)} rows...")

#     print("\n======= 平均评估结果 =======")
#     for name in cols.values():
#         print(f"\n--- {name.upper()} ---")
#         for metric in ["valid", "acc", "err", "score"]:
#             avg = sum(results[name][metric]) / len(results[name][metric])
#             print(f"{metric.capitalize():<15}: {avg:.4f}")


# if __name__ == "__main__":
#     main()

# - Rule: UPPERCASE_SENTENCE_START 
# Message: This sentence does not start with an uppercase letter. 
# Offset: 0 to 3 
# Error Text: 'the' 
# Suggestions: ['The'] 
# - Rule: ENGLISH_WORD_REPEAT_RULE 
# Message: Possible typo: you repeated a word. 
# Offset: 9 to 24 
# Error Text: 'country country' 
# Suggestions: ['country'] 
# - Rule: COMMA_PARENTHESIS_WHITESPACE 
# Message: Don’t put a space before the full stop. 
# Offset: 24 to 26 
# Error Text: ' .' 
# Suggestions: ['.']