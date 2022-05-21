from .base import AbstractDataloader
from .utils import Prompter

import torch
import random
import numpy as np
import torch.utils.data as data_utils

import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# the following prompting is based on alpaca
def generate_and_tokenize_eval(args, data_point, tokenizer, prompter):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])
    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)
    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
    return tokenized_full_prompt


def generate_and_tokenize_train(args, data_point, tokenizer, prompter):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.llm_max_text_len,
                           padding=False,
                           return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = prompter.generate_prompt(data_point["system"],
                                           data_point["input"],
                                           data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
    if not args.llm_train_on_inputs:
        tokenized_full_prompt["labels"][:-2] = [-100] * len(tokenized_full_prompt["labels"][:-2])
    
    return tokenized_full_prompt


def seq_to_token_ids(args, seq, candidates, label, text_dict, tokenizer, prompter, eval=False):
    def truncate_title(title):
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title

    seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
                       for idx, item in enumerate(seq)])
    can_t = ' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
                       for idx, item in enumerate(candidates)])
    output = chr(ord('A') + candidates.index(label))  # ranking only
    
    data_point = {}
    data_point['system'] = args.llm_system_template if args.llm_system_template is not None else DEFAULT_SYSTEM_PROMPT
    data_point['input'] = args.llm_input_template.format(seq_t, can_t)
    data_point['output'] = output
    
    if eval:
        return generate_and_tokenize_eval(args, data_point, tokenizer, prompter)
    else:
        return generate_and_tokenize_train(args, data_point, tokenizer, prompter)


class LLMDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        seq_dataset =