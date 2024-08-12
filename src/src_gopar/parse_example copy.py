from supar import Parser
import torch
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
# parser = Parser.load('biaffine-dep-en') 
# parser = Parser.load('/opt/data/private/biaffine-dep-en')

parser = Parser.load('biaffine-dep-roberta-en')
# parser_1 = Parser.load('/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/ptb.biaffine.dep.roberta')
# parser_2 = Parser.load('/opt/data/private/friends/tzc/SynGEC-main/src/src_gopar/emnlp2022_syngec_biaffine-dep-electra-en-gopar')

dataset_1 = parser_1.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
dataset_2 = parser_2.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_2[0])


print("_"*20)
print(f"arcs:  {dataset_1.arcs[0]}\n"
          f"rels:  {dataset_1.rels[0]}\n"
          f"probs: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
print(f"arcs:  {dataset_2.arcs[0]}\n"
          f"rels:  {dataset_2.rels[0]}\n"
          f"probs: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict("he also need to tell relatives like his grandmother 's sister 's son 's daughter 's nephew ?", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
dataset_2 = parser_2.predict("he also need to tell relatives like his grandmother 's sister 's son 's daughter 's nephew ?", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_2[0])

print("_"*20)
print(f"arcs:  {dataset_1.arcs[0]}\n"
          f"rels:  {dataset_1.rels[0]}\n"
          f"probs: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
print(f"arcs:  {dataset_2.arcs[0]}\n"
          f"rels:  {dataset_2.rels[0]}\n"
          f"probs: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict("i &apos;m going to tell you about some people who didn &apos;t move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
dataset_2 = parser_2.predict("i &apos;m going to tell you about some people who didn &apos;t move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_2[0])

print("_"*20)
print(f"arcs:  {dataset_1.arcs[0]}\n"
          f"rels:  {dataset_1.rels[0]}\n"
          f"probs: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
print(f"arcs:  {dataset_2.arcs[0]}\n"
          f"rels:  {dataset_2.rels[0]}\n"
          f"probs: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")


dataset_1 = parser_1.predict("i 'm going to tell you about some people who didn 't move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_1[0])
dataset_2 = parser_2.predict("i 'm going to tell you about some people who didn 't move out of their neighborhoods .", lang='en', prob=True, verbose=False)
print("-"*20)
print(dataset_2[0])

print("_"*20)
print(f"arcs:  {dataset_1.arcs[0]}\n"
          f"rels:  {dataset_1.rels[0]}\n"
          f"probs: {dataset_1.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")
print("_"*20)
print(f"arcs:  {dataset_2.arcs[0]}\n"
          f"rels:  {dataset_2.rels[0]}\n"
          f"probs: {dataset_2.probs[0].gather(1,torch.tensor(dataset_1.arcs[0]).unsqueeze(1)).squeeze(-1)}")