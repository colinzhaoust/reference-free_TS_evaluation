import re
import json
from functools import lru_cache
from typing import List
from collections import Counter

from tqdm.notebook import trange, tqdm

import numpy as np

import sacrebleu
import sacremoses
import nltk
import math

NGRAM_ORDER = 4 # for SARI

def to_words(text):
    return text.split()


def count_words(text):
    return len(to_words(text))


def to_sentences(text, language='english'):
    try:
        tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    return tokenizer.tokenize(text)


def count_sentences(text, language='english'):
    return len(to_sentences(text, language))


@lru_cache(maxsize=100000)
def count_syllables_in_word(word):
    # The syllables counting logic is adapted from the following scripts:
    # https://github.com/XingxingZhang/dress/blob/master/dress/scripts/readability/syllables_en.py
    # https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/readability/syllables_en.py
    special_words = {
        'the': 1,
        'tottered': 2,
        'chummed': 1,
        'peeped': 1,
        'moustaches': 2,
        'shamefully': 3,
        'messieurs': 2,
        'satiated': 4,
        'sailmaker': 4,
        'sheered': 1,
        'disinterred': 3,
        'propitiatory': 6,
        'bepatched': 2,
        'particularized': 5,
        'caressed': 2,
        'trespassed': 2,
        'sepulchre': 3,
        'flapped': 1,
        'hemispheres': 3,
        'pencilled': 2,
        'motioned': 2,
        'poleman': 2,
        'slandered': 2,
        'sombre': 2,
        'etc': 4,
        'sidespring': 2,
        'mimes': 1,
        'effaces': 2,
        'mr': 2,
        'mrs': 2,
        'ms': 1,
        'dr': 2,
        'st': 1,
        'sr': 2,
        'jr': 2,
        'truckle': 2,
        'foamed': 1,
        'fringed': 2,
        'clattered': 2,
        'capered': 2,
        'mangroves': 2,
        'suavely': 2,
        'reclined': 2,
        'brutes': 1,
        'effaced': 2,
        'quivered': 2,
        "h'm": 1,
        'veriest': 3,
        'sententiously': 4,
        'deafened': 2,
        'manoeuvred': 3,
        'unstained': 2,
        'gaped': 1,
        'stammered': 2,
        'shivered': 2,
        'discoloured': 3,
        'gravesend': 2,
        '60': 2,
        'lb': 1,
        'unexpressed': 3,
        'greyish': 2,
        'unostentatious': 5
    }
    special_syllables_substract = [
        'cial', 'tia', 'cius', 'cious', 'gui', 'ion', 'iou', 'sia$', '.ely$'
    ]
    special_syllables_add = [
        'ia', 'riet', 'dien', 'iu', 'io', 'ii', '[aeiouy]bl$', 'mbl$',
        '[aeiou]{3}', '^mc', 'ism$', '(.)(?!\\1)([aeiouy])\\2l$', '[^l]llien',
        '^coad.', '^coag.', '^coal.', '^coax.',
        '(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]', 'dnt$'
    ]
    word = word.lower().strip()
    if word in special_words:
        return special_words[word]
    # Remove final silent 'e'
    word = word.rstrip('e')
    # Count vowel groups
    count = 0
    prev_was_vowel = 0
    for c in word:
        is_vowel = c in ('a', 'e', 'i', 'o', 'u', 'y')
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Add & subtract syllables
    for r in special_syllables_add:
        if re.search(r, word):
            count += 1
    for r in special_syllables_substract:
        if re.search(r, word):
            count -= 1
    return count


def count_syllables_in_sentence(sentence):
    return sum([count_syllables_in_word(word) for word in to_words(sentence)])


def normalize(sentence, lowercase: bool = True, tokenizer: str = '13a', return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ['13a', 'intl']:
        normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == 'moses':
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == 'penn':
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent
    

def compute_precision_recall(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    return precision, recall


def compute_f1(precision, recall):
    f1 = 0.0
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_micro_sari(
    add_hyp_correct,
    add_hyp_total,
    add_ref_total,
    keep_hyp_correct,
    keep_hyp_total,
    keep_ref_total,
    del_hyp_correct,
    del_hyp_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version described in the original paper. Follows the equations.
    """
    add_precision = [0] * NGRAM_ORDER
    add_recall = [0] * NGRAM_ORDER
    keep_precision = [0] * NGRAM_ORDER
    keep_recall = [0] * NGRAM_ORDER
    del_precision = [0] * NGRAM_ORDER
    del_recall = [0] * NGRAM_ORDER

    for n in range(NGRAM_ORDER):
        add_precision[n], add_recall[n] = compute_precision_recall(
            add_hyp_correct[n], add_hyp_total[n], add_ref_total[n]
        )
        keep_precision[n], keep_recall[n] = compute_precision_recall(
            keep_hyp_correct[n], keep_hyp_total[n], keep_ref_total[n]
        )
        del_precision[n], del_recall[n] = compute_precision_recall(
            del_hyp_correct[n], del_hyp_total[n], del_ref_total[n]
        )

    avg_add_precision = sum(add_precision) / NGRAM_ORDER
    avg_add_recall = sum(add_recall) / NGRAM_ORDER
    avg_keep_precision = sum(keep_precision) / NGRAM_ORDER
    avg_keep_recall = sum(keep_recall) / NGRAM_ORDER
    avg_del_precision = sum(del_precision) / NGRAM_ORDER
    avg_del_recall = sum(del_recall) / NGRAM_ORDER

    add_f1 = compute_f1(avg_add_precision, avg_add_recall)

    keep_f1 = compute_f1(avg_keep_precision, avg_keep_recall)

    if use_f1_for_deletion:
        del_score = compute_f1(avg_del_precision, avg_del_recall)
    else:
        del_score = avg_del_precision

    return add_f1, keep_f1, del_score


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> List[Counter]:
    ngrams_per_order = []
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        ngrams = Counter()
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1
        ngrams_per_order.append(ngrams)

    return ngrams_per_order


def multiply_counter(c, v):
    c_aux = Counter()
    for k in c.keys():
        c_aux[k] = c[k] * v

    return c_aux


def compute_ngram_stats(
    orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]]):
    
    add_sys_correct = [0] * NGRAM_ORDER
    add_sys_total = [0] * NGRAM_ORDER
    add_ref_total = [0] * NGRAM_ORDER

    keep_sys_correct = [0] * NGRAM_ORDER
    keep_sys_total = [0] * NGRAM_ORDER
    keep_ref_total = [0] * NGRAM_ORDER

    del_sys_correct = [0] * NGRAM_ORDER
    del_sys_total = [0] * NGRAM_ORDER
    del_ref_total = [0] * NGRAM_ORDER

    for orig_sent, sys_sent, *ref_sents in zip(
        orig_sents, sys_sents, *refs_sents
    ):
        orig_ngrams = extract_ngrams(orig_sent)
        sys_ngrams = extract_ngrams(sys_sent)

        refs_ngrams = [Counter() for _ in range(NGRAM_ORDER)]
        for ref_sent in ref_sents:
            ref_ngrams = extract_ngrams(ref_sent)
            for n in range(NGRAM_ORDER):
                refs_ngrams[n] += ref_ngrams[n]

        num_refs = len(ref_sents)
        for n in range(NGRAM_ORDER):
            # ADD
            # added by the hypothesis (binary)
            sys_and_not_orig = set(sys_ngrams[n]) - set(orig_ngrams[n])
            add_sys_total[n] += len(sys_and_not_orig)
            # added by the references (binary)
            ref_and_not_orig = set(refs_ngrams[n]) - set(orig_ngrams[n])
            add_ref_total[n] += len(ref_and_not_orig)
            # added correctly (binary)
            add_sys_correct[n] += len(sys_and_not_orig & set(refs_ngrams[n]))

            # KEEP
            # kept by the hypothesis (weighted)
            orig_and_sys = multiply_counter(
                orig_ngrams[n], num_refs
            ) & multiply_counter(sys_ngrams[n], num_refs)
            keep_sys_total[n] += sum(orig_and_sys.values())
            # kept by the references (weighted)
            orig_and_ref = (
                multiply_counter(orig_ngrams[n], num_refs) & refs_ngrams[n]
            )
            keep_ref_total[n] += sum(orig_and_ref.values())
            # kept correctly?
            keep_sys_correct[n] += sum((orig_and_sys & orig_and_ref).values())

            # DELETE
            # deleted by the hypothesis (weighted)
            orig_and_not_sys = multiply_counter(
                orig_ngrams[n], num_refs
            ) - multiply_counter(sys_ngrams[n], num_refs)
            del_sys_total[n] += sum(orig_and_not_sys.values())
            # deleted by the references (weighted)
            orig_and_not_ref = (
                multiply_counter(orig_ngrams[n], num_refs) - refs_ngrams[n]
            )
            del_ref_total[n] += sum(orig_and_not_ref.values())
            # deleted correctly
            del_sys_correct[n] += sum(
                (orig_and_not_sys & orig_and_not_ref).values()
            )

    return (
        add_sys_correct,
        add_sys_total,
        add_ref_total,
        keep_sys_correct,
        keep_sys_total,
        keep_ref_total,
        del_sys_correct,
        del_sys_total,
        del_ref_total,
    )


def compute_precision_recall_f1(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    f1 = 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_macro_sari(
    add_sys_correct,
    add_sys_total,
    add_ref_total,
    keep_sys_correct,
    keep_sys_total,
    keep_ref_total,
    del_sys_correct,
    del_sys_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version released as a JAVA implementation and which was used in their experiments,
    as stated by the authors: https://github.com/cocoxu/simplification/issues/8
    """
    add_f1 = 0.0
    keep_f1 = 0.0
    del_f1 = 0.0
    for n in range(NGRAM_ORDER):
        _, _, add_f1_ngram = compute_precision_recall_f1(
            add_sys_correct[n], add_sys_total[n], add_ref_total[n]
        )
        _, _, keep_f1_ngram = compute_precision_recall_f1(
            keep_sys_correct[n], keep_sys_total[n], keep_ref_total[n]
        )
        if use_f1_for_deletion:
            _, _, del_score_ngram = compute_precision_recall_f1(
                del_sys_correct[n], del_sys_total[n], del_ref_total[n]
            )
        else:
            del_score_ngram, _, _ = compute_precision_recall_f1(del_sys_correct[n], del_sys_total[n], del_ref_total[n])
        add_f1 += add_f1_ngram / NGRAM_ORDER
        keep_f1 += keep_f1_ngram / NGRAM_ORDER
        del_f1 += del_score_ngram / NGRAM_ORDER
        
    return add_f1, keep_f1, del_f1


def get_corpus_sari_operation_scores(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                                     lowercase: bool = True, tokenizer: str = '13a',
                                     legacy=False, use_f1_for_deletion=True, use_paper_version=False):
    """The `legacy` parameter allows reproducing scores reported in previous work.
    It replicates a bug in the original JAVA implementation where only the system outputs and the reference sentences
    are further tokenized. 
    In addition, it assumes that all sentences are already lowercased. """
    if legacy:
        lowercase = False
    else:
        orig_sents = [
            normalize(sent, lowercase, tokenizer)
            for sent in orig_sents
        ]

    sys_sents = [
        normalize(sent, lowercase, tokenizer) for sent in sys_sents
    ]
    refs_sents = [
        [normalize(sent, lowercase, tokenizer) for sent in ref_sents]
        for ref_sents in refs_sents
    ]

    stats = compute_ngram_stats(orig_sents, sys_sents, refs_sents)

    if not use_paper_version:
        add_score, keep_score, del_score = compute_macro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    else:
        add_score, keep_score, del_score = compute_micro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    return 100. * add_score, 100. * keep_score, 100. * del_score


def corpus_sari(*args, **kwargs):
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(*args, **kwargs)
    return (add_score + keep_score + del_score) / 3, (add_score,keep_score,del_score)


# FK, BLEU, iBLEU, FKBLEU, SARI
class FKGLScorer:
    "https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests"

    def __init__(self):
        self.nb_words = 0
        self.nb_syllables = 0
        self.nb_sentences = 0

    def add(self, text):
        for sentence in to_sentences(text):
            self.nb_words += count_words(sentence)
            self.nb_syllables += count_syllables_in_sentence(sentence)
            self.nb_sentences += 1

    def score(self):
        # Flesch-Kincaid grade level
        if self.nb_sentences == 0 or self.nb_words == 0:
            return 0
        return max(
            0,
            0.39 * (self.nb_words / self.nb_sentences)
            + 11.8 * (self.nb_syllables / self.nb_words)
            - 15.59,)

    
def corpus_bleu(sys_sents: List[str],
                refs_sents: List[List[str]],
                smooth_method: str = 'exp',
                smooth_value: float = None,
                force: bool = True,
                lowercase: bool = False,
                tokenizer: str = '13a',
                use_effective_order: bool = False):

    sys_sents = [normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents]
                  for ref_sents in refs_sents]

    return sacrebleu.corpus_bleu(sys_sents, refs_sents, smooth_method, smooth_value, force,
                                 lowercase=False, tokenize='none', use_effective_order=use_effective_order).score
    
    
def corpus_ibleu(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                smooth_method: str = 'exp',
                smooth_value: float = None,
                force: bool = True,
                lowercase: bool = False,
                tokenizer: str = '13a',
                use_effective_order: bool = False):
    
    orig_sents = [normalize(sent, lowercase, tokenizer) for sent in orig_sents]
    sys_sents = [normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents]
                  for ref_sents in refs_sents]
    
    or_score = sacrebleu.corpus_bleu(sys_sents, refs_sents, smooth_method, smooth_value, force,
                                 lowercase=False, tokenize='none', use_effective_order=use_effective_order).score
    
    oi_score = sacrebleu.corpus_bleu(sys_sents, [orig_sents], smooth_method, smooth_value, force,
                             lowercase=False, tokenize='none', use_effective_order=use_effective_order).score
    
    return 0.9*or_score-0.1*oi_score


def corpus_fkbleu(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                smooth_method: str = 'exp',
                smooth_value: float = None,
                force: bool = True,
                lowercase: bool = False,
                tokenizer: str = '13a',
                use_effective_order: bool = False):
    scorer = FKGLScorer()
    scorer.add(sys_sents[0])
    fks = scorer.score()
    scorer = FKGLScorer()
    scorer.add(orig_sents[0])
    fki = scorer.score()
    
    orig_sents = [normalize(sent, lowercase, tokenizer) for sent in orig_sents]
    sys_sents = [normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents]
                  for ref_sents in refs_sents]

    or_score = sacrebleu.corpus_bleu(sys_sents, refs_sents, smooth_method, smooth_value, force,
                                 lowercase=False, tokenize='none', use_effective_order=use_effective_order).score
    
    oi_score = sacrebleu.corpus_bleu(sys_sents, [orig_sents], smooth_method, smooth_value, force,
                             lowercase=False, tokenize='none', use_effective_order=use_effective_order).score
    
    return (0.9*or_score-0.1*oi_score)*1/(1 + math.exp(-(fks-fki)))


def batch_metric_generation(orig_sents, sys_sents, refs_sents):
    
    scorer = FKGLScorer()
    scorer.add(sys_sents[0])
    fks = scorer.score()
    scorer = FKGLScorer()
    scorer.add(orig_sents[0])
    fki = scorer.score()
    
    sari, partial = corpus_sari(orig_sents, sys_sents, refs_sents) # add, keep, del

    b = corpus_bleu(sys_sents, refs_sents)

    ib = corpus_ibleu(orig_sents, sys_sents, refs_sents)

    fkb = corpus_fkbleu(orig_sents, sys_sents, refs_sents)
    
    return -1*fks, sari, partial, b, ib, fkb


def get_predictions(wikidata, system_output):
    
    output_results = dict() # -> metrics, system, sentences
    
    metrics = ["-FK", "BLEU_s", "BLEU_m", "iBLEU_s", "iBLEU_m", "FKBLEU", "SARI", "SARI_s", "SARI_a", "SARI_k", "SARI_d"]
    scope = ["Reference", "Dress-Ls", "Hybrid", "PBMT-R","UNTS","RM+EX+LS+RO", "BTRLTS", "BTTS10"]
    
    for met in metrics:
        output_results[met] = dict()
        for sys in scope:
            output_results[met][sys] = []
    
    for i, temp in tqdm(enumerate(wikidata)):
        orig_sent = temp["input"]
        mul_refs = temp["reference"]
        sing_ref = system_output["Reference"][i]
        
        refs_sents = []
        for ref in mul_refs:
            refs_sents.append([ref])
        
        for sys in scope[1:]:
            sys_sent = system_output[sys][i]
            fk, sari, partial, b, ib, fkb = batch_metric_generation([orig_sent], [sys_sent], refs_sents)
            _,sari_s,_,bs,ibs,_ = batch_metric_generation([orig_sent], [sys_sent], [[sing_ref]])
            
            temp_collection = [-1* fk, bs, b, ibs, ib, fkb, sari, sari_s, partial[0], partial[1], partial[2]]
            
            for j in range(len(metrics)):
                output_results[metrics[j]][sys].append(temp_collection[j])
                
    return output_results
                

if __name__ == "__main__":
    # examples    
    sari, partial = corpus_sari(orig_sents=["About 95 species are currently accepted.", "The cat perched on the mat."],  
                sys_sents=["About 95 you now get in.", "Cat on mat."], 
                refs_sents=[["About 95 species are currently known.", "The cat sat on the mat."],
                            ["About 95 species are now accepted.", "The cat is on the mat."],  
                            ["95 species are now accepted.", "The cat sat."]])

    b = corpus_bleu(sys_sents=["About 95 you now get in.", "Cat on mat."], 
                refs_sents=[["About 95 species are currently known.", "The cat sat on the mat."],
                            ["About 95 species are now accepted.", "The cat is on the mat."],  
                            ["95 species are now accepted.", "The cat sat."]])

    ib = corpus_ibleu(orig_sents=["About 95 species are currently accepted."],  
                sys_sents=["About 95 you now get in."], 
                refs_sents=[["About 95 species are currently known."],
                            ["About 95 species are now accepted."],  
                            ["95 species are now accepted."]])

    fk = corpus_fkbleu(orig_sents=["About 95 species are currently accepted."],  
                sys_sents=["About 95 you now get in."], 
                refs_sents=[["About 95 species are currently known."],
                            ["About 95 species are now accepted."],  
                            ["95 species are now accepted."]])

    print(sari,b,ib,fk)
