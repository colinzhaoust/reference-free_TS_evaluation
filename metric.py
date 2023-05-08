import ujson as json
import os
import random
import logging
import argparse
import math

import numpy as np
from nltk import word_tokenize
from tqdm import trange, tqdm
import torch
import transformers
from transformers import *

from word_complexity import ComplexityRanker, BertEncoder

from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, accuracy_score

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


def sent_encode(tokenizer, sent, device):
    # covert sentence to ids
    toks = [tokenizer.cls_token]
    word_index = [-1] # word index for toks
    
    ori_toks = word_tokenize(sent)
    
    for i, raw_tok in enumerate(ori_toks):
        new_tok = tokenizer.tokenize(raw_tok)
        toks += new_tok
        word_index.append([i for j in range(len(new_tok))])
    
    toks += [tokenizer.sep_token]
    word_index += [-1]
    
    input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(toks)).unsqueeze(0).to(device)
    
    return input_ids, word_index


def r_meaning_score(input_sent, output_sent, tokenizer, model, device):
    
    input_ids, input_index = sent_encode(tokenizer, input_sent, device)
    output_ids, output_index = sent_encode(tokenizer, output_sent, device)

    input_rep = model(input_ids)
    input_rep = input_rep[0].cpu().detach().numpy()
    output_rep = model(output_ids)
    output_rep = output_rep[0].cpu().detach().numpy()
    
    r_sim = []
    
    for _, vec in enumerate(input_rep[0]):
        o_sim = []
        for i, cand in enumerate(output_rep[0]):
            sim = np.dot(cand, vec)/(np.linalg.norm(cand)*np.linalg.norm(vec))
            o_sim.append(sim)
            
        r_sim.append(max(o_sim))
        
    return(sum(r_sim)/len(r_sim))


def get_rpredictions(wikidata, system_output, tokenizer, model, device):
    
    output_results = dict()
    metrics = ["rmean"]
    
    scope = ["Reference", "Dress-Ls", "Hybrid", "PBMT-R","UNTS","RM+EX+LS+RO", "BTRLTS", "BTTS10"]
    
    for met in metrics:
        output_results[met] = dict()
        for sys in scope:
            output_results[met][sys] = []
            
    for i, temp in tqdm(enumerate(wikidata)):
        input_sent = temp["input"]
        mul_refs = temp["reference"]
        sing_ref = system_output["Reference"][i]
        
        refs_sents = []
        for ref in mul_refs:
            refs_sents.append([ref])
        
        for sys in scope[1:]:
            output_sent = system_output[sys][i]
            rm = r_meaning_score(input_sent, output_sent, tokenizer, model, device)
            
            temp_collection = [rm]
            
            for j in range(len(metrics)):
                output_results[metrics[j]][sys].append(temp_collection[j])
                
    return output_results


def p_simp_score(input_sent, output_sent, tokenizer, lm_model, ranker, device):

    simp_score = -1
    input_sent = input_sent.lower()
    output_sent = output_sent.lower()

    input_words = word_tokenize(input_sent)
    output_words = word_tokenize(output_sent)

    unique_words = []
    changed_words = []

    for wd in output_words:
        if wd not in input_words:
            unique_words.append(wd)

    for wd in input_words:
        if wd not in output_words:
            changed_words.append(wd)

    p_simp = []

    uid_collection = []
    cid_collection = []

    sim_scores = []
    indexer = []

    for uwd in unique_words:
        temp_sim_scores = []

        uids = tokenizer.encode(uwd, add_special_tokens=True, return_tensors='pt').to(device)
        uid_collection.append(uids)
        u_rep = lm_model(uids)
        u_rep = u_rep[0][0].cpu().detach().numpy()

        for i, cwd in enumerate(changed_words):
            
            cids = tokenizer.encode(cwd, add_special_tokens=True, return_tensors='pt').to(device)
            cid_collection.append(cids)


            c_rep = lm_model(cids)
            c_rep = c_rep[0][0].cpu().detach().numpy()

            tok_level_sim = []
            for vec in u_rep:
                for cand in c_rep:
                    sim = np.dot(cand, vec)/(np.linalg.norm(cand)*np.linalg.norm(vec))
                    tok_level_sim.append(sim)
            
            # greedy
            temp_sim_scores.append(max(tok_level_sim))

        np_temp = np.array(temp_sim_scores)
        np_rank = np.argsort(np_temp)
        indexer.append(np_rank[-1])




    for i, uids in enumerate(uid_collection):
        # select the most similar
        cids = cid_collection[indexer[i]]

        prediction = ranker(cids, uids)
        p_simp.extend(prediction.cpu().detach().tolist())

    simp_score =  max(p_simp)

    return simp_score 


def get_ppredictions(wikidata, system_output, tokenizer, lm_model, ranker, device):
    
    output_results = dict()
    metrics = ["psimp"]
    
    scope = ["Reference", "Dress-Ls", "Hybrid", "PBMT-R","UNTS","RM+EX+LS+RO", "BTRLTS", "BTTS10"]
    
    for met in metrics:
        output_results[met] = dict()
        for sys in scope:
            output_results[met][sys] = []
            
    for i, temp in tqdm(enumerate(wikidata)):
        input_sent = temp["input"]
        mul_refs = temp["reference"]
        sing_ref = system_output["Reference"][i]
        
        refs_sents = []
        for ref in mul_refs:
            refs_sents.append([ref])
        
        for sys in scope[1:]:
            output_sent = system_output[sys][i]
            psimp = p_simp_score(input_sent, output_sent, tokenizer, lm_model, ranker, device)
            
            temp_collection = [psimp]
            
            for j in range(len(metrics)):
                output_results[metrics[j]][sys].append(temp_collection[j])
                
    return output_results
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ## parameters
    parser.add_argument("--gpu", default='0', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--model", default='bert-base-uncased', type=str, required=False,
                        help="choose the model to test")
    parser.add_argument("--lr", default=1e-5, type=float, required=False,
                        help="initial learning rate")
    parser.add_argument("--lrdecay", default=0.0, type=float, required=False,
                        help="learning rate decay every 5 epochs")
    parser.add_argument("--do_clean", default=True, type=bool, required=False,
                        help="if we do cleaning on the simpleppdb")
    parser.add_argument("--train_bert", default=False, type=bool, required=False,
                        help="if enables the training of bert")
    parser.add_argument("--max_len", default=15, type=int, required=False,
                        help="number of words")
    parser.add_argument("--epochs", default=3, type=int, required=False,
                        help="number of epochs")
    parser.add_argument("--covered_rules", default=12000, type=int, required=False,
                        help="number of covered rules")
    parser.add_argument("--mode", default="classification", type=str, required=False,
                        help="classification/regression as labels")


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = args.model

    tokenizer = BertTokenizer.from_pretrained(model)
    lm_model =  BertModel.from_pretrained(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_model.to(device)


    # ranker = ComplexityRanker(args)
    ranker = torch.load("./bert_base_all.ckpt")
    
    ranker.eval()
    ranker.to(device)

    with open("./wikidata/final_output.json","r") as f:
        sys_output = json.load(f)
        
    with open("./wikidata/wikilarge.json","r") as f:
        wikidata = json.load(f)
        
    p_simp = get_ppredictions(wikidata[:70], sys_output["wikitest"], tokenizer, lm_model, ranker, device)
    r_mean = get_rpredictions(wikidata[:70], sys_output["wikitest"], tokenizer, lm_model, device)

    output_results = {}
    output_results["p_simp"] = p_simp
    output_results["r_mean"] = r_mean

    with open("./scores/simp_metric_scores.json","w") as f:
        json.dump(output_results, f)