from supar import Parser
import torch
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
# parser = Parser.load('biaffine-dep-en') 
# parser = Parser.load('/opt/data/private/biaffine-dep-en')

# parser = Parser.load('biaffine-dep-roberta-en')
parser_1 = Parser.load('/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/ptb.biaffine.dep.roberta')
# parser_1 = Parser.load('/opt/data/private/friends/tzc/data/iwslt_de/ud.biaffine.dep.xlmr')
# parser_2 = Parser.load('/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/emnlp2022_syngec_biaffine-dep-electra-en-gopar')

# dataset_1 = parser_1.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
# print("-"*20)
# print(dataset_1[0])
# dataset_2 = parser_2.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
# print("-"*20)
# print(dataset_2[0])


# print("_"*20)
# print(f"arcs:  {dataset_1.arcs[0]}\n"
#           f"rels:  {dataset_1.rels[0]}\n"
#           f"probs: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
# print("_"*20)
# print(f"arcs:  {dataset_2.arcs[0]}\n"
#           f"rels:  {dataset_2.rels[0]}\n"
#           f"probs: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict([["The", "40", "year", "@-@", "old", "hopes", "to", "have", "his", "best", "striker", "(", "with", "six", "goals", ")", "in", "the", "team", ".", "?"]],lang='en', prob=True, verbose=False)  #  lang='en',
print("-"*20)
print(dataset_1[0])
# dataset_2 = parser_2.predict("he also need to tell relatives like his grandmother 's sister 's son 's daughter 's nephew ?", lang='en', prob=True, verbose=False)
# print("-"*20)
# print(dataset_2[0])


print("_"*20)
print(f"arcs1:  {dataset_1.arcs[0]}\n"
          f"rels1:  {dataset_1.rels[0]}\n"
          f"probs1: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
'''
____________________
arcs1:  [3, 3, 0, 5, 3, 5, 6, 9, 11, 9, 13, 9, 17, 13, 17, 15, 7, 3]
rels1:  ['nsubj', 'advmod', 'root', 'aux', 'xcomp', 'dobj', 'prep', 'poss', 'poss', 
'possessive', 'poss', 'possessive', 'poss', 'possessive', 'poss', 'possessive', 'pobj', 'punct']
probs1: tensor([1.0000, 0.9997, 0.9999, 1.0000, 1.0000, 1.0000, 0.9995, 0.9458, 0.9724,
        0.9856, 0.0459, 0.6241, 0.8318, 0.6752, 0.9125, 0.8118, 0.9981, 0.9998])

dataset_1.probs[0].shape
torch.Size([18, 19])  # 这18个词与另外19个的关系
____________________
'''
# print(f"arcs2:  {dataset_2.arcs[0]}\n"
#           f"rels2:  {dataset_2.rels[0]}\n"
#           f"probs2: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict("i &apos; m going to tell you about some people who didn &apos; t move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
# dataset_2 = parser_2.predict("i &apos;m going to tell you about some people who didn &apos;t move out of their neighborhoods .", lang='en', prob=True, verbose=False)
# print("-"*20)
# print(dataset_2[0])

print("_"*20)
print(f"arcs1:  {dataset_1.arcs[0]}\n"
          f"rels1:  {dataset_1.rels[0]}\n"
          f"probs1: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
# print(f"arcs2:  {dataset_2.arcs[0]}\n"
#           f"rels2:  {dataset_2.rels[0]}\n"
#           f"probs2: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict("i 'm going to tell you about some people who didn 't move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
# dataset_2 = parser_2.predict("i 'm going to tell you about some people who didn 't move out of their neighborhoods .", lang='en', prob=True, verbose=False)
# print("-"*20)
# print(dataset_2[0])

print("_"*20)
print(f"arcs1:  {dataset_1.arcs[0]}\n"
          f"rels1:  {dataset_1.rels[0]}\n"
          f"probs1: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
# print(f"arcs2:  {dataset_2.arcs[0]}\n"
#           f"rels2:  {dataset_2.rels[0]}\n"
#           f"probs2: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
'''
--------------------
1       i       _       _       _       _       4       nsubj   _       _
2       &apos   _       _       _       _       4       dep     _       _
3       ;m      _       _       _       _       4       aux     _       _
4       going   _       _       _       _       0       root    _       _
5       to      _       _       _       _       6       aux     _       _
6       tell    _       _       _       _       4       xcomp   _       _
7       you     _       _       _       _       6       dobj    _       _
8       about   _       _       _       _       6       prep    _       _
9       some    _       _       _       _       10      det     _       _
10      people  _       _       _       _       8       pobj    _       _
11      who     _       _       _       _       14      nsubj   _       _
12      didn    _       _       _       _       14      aux     _       _
13      &apos;t _       _       _       _       14      neg     _       _
14      move    _       _       _       _       10      rcmod   _       _
15      out     _       _       _       _       14      prep    _       _
16      of      _       _       _       _       15      pcomp   _       _
17      their   _       _       _       _       18      poss    _       _
18      neighborhoods   _       _       _       _       16      pobj    _       _
19      .       _       _       _       _       4       punct   _       _

____________________
arcs1:  [4, 4, 4, 0, 6, 4, 6, 6, 10, 8, 14, 14, 14, 10, 14, 15, 18, 16, 4]
rels1:  ['nsubj', 'dep', 'aux', 'root', 'aux', 'xcomp', 'dobj', 'prep', 'det', 'pobj', 'nsubj', 'aux', 'neg', 'rcmod', 'prep', 'pcomp', 'poss', 'pobj', 'punct']
probs1: tensor([1.0000, 0.9981, 0.9974, 1.0000, 1.0000, 1.0000, 1.0000, 0.9998, 1.0000,
        1.0000, 0.9996, 0.9998, 0.9869, 1.0000, 0.9929, 0.9997, 1.0000, 1.0000,
        0.9999])
____________________
'''

'''
--------------------
1       he      _       _       _       _       3       nsubj   _       _
2       also    _       _       _       _       3       advmod  _       _
3       need    _       _       _       _       0       root    _       _
4       to      _       _       _       _       5       aux     _       _
5       tell    _       _       _       _       3       xcomp   _       _
6       relatives       _       _       _       _       5       dobj    _       _
7       like    _       _       _       _       6       prep    _       _
8       his     _       _       _       _       9       poss    _       _
9       grandmother     _       _       _       _       11      poss    _       _
10      's      _       _       _       _       9       possessive      _       _
11      sister  _       _       _       _       13      poss    _       _
12      's      _       _       _       _       9       possessive      _       _
13      son     _       _       _       _       17      poss    _       _
14      's      _       _       _       _       13      possessive      _       _
15      daughter        _       _       _       _       17      poss    _       _
16      's      _       _       _       _       15      possessive      _       _
17      nephew  _       _       _       _       7       pobj    _       _
18      ?       _       _       _       _       3       punct   _       _
'''