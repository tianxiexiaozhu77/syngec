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



