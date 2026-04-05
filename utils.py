import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
#os.environ["TRANSFORMERS_OFFLINE"] = "1" # remove this line when downloading fresh
import numpy as np
import pandas as pd
import random

def load_model_tokenizer(model_name, device, dtype=torch.float32):
    # REMOVE local_files_only=True for the tokenizer to ensure it gets the Llama 3 version
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        # local_files_only=True,  <-- REMOVE THIS
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        output_attentions=True,
        dtype=dtype,  
        local_files_only=True, # Keep this True if your model weights are already downloaded
    )
    model.to(device)
    model.eval()
    
    # Quick sanity check for the tokenizer bug
    test_len = len(tokenizer("This is a test to see if lengths work", add_special_tokens=False).input_ids)
    if test_len <= 1:
        print("CRITICAL WARNING: Tokenizer is STILL mapping entire sentences to a single token.")
    else:
        print(f"SUCCESS: Tokenizer is working! Test sentence length: {test_len} tokens.")
        
    return tokenizer, model

class PromptUtils:
    def __init__(self, tokenizer, doc_ids, dict_all_docs):
        self.dict_doc_name_id = {key:idx for idx, key in enumerate(doc_ids)}
        self.tokenizer = tokenizer
        self.prompt_seperator = " \n\n"
        user_header = '<|start_header_id|>user<|end_header_id|>'
        asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self.item_instruction = f" Here are all the available tools:"
        self.prompt_prefix = user_header + self.item_instruction
        self.prompt_suffix = asst_header
        self.prompt_prefix_length = len(tokenizer(self.prompt_prefix, add_special_tokens=False).input_ids)
        self.prompt_suffix_length = len(tokenizer(self.prompt_suffix, add_special_tokens=False).input_ids)
        
        self.doc_text = lambda idx, doc_name, doc_info: f"tool_id: {doc_name}\ntool description: {doc_info}"
        self.add_text1 = f"Now, please output ONLY the correct tool_id for the query below."

        (
            self.all_docs_info_string, 
            self.doc_names_str, 
            self.doc_lengths,
            self.doc_spans
        ) = self.create_doc_pool_string(doc_ids, dict_all_docs)
        self.add_text1_length = len(tokenizer(self.add_text1, add_special_tokens=False).input_ids)

    def create_prompt(self, query):
        query_prompt = f"Query: {query}"+ "\nCorrect tool_id:"
        prompt = self.prompt_prefix + \
                self.all_docs_info_string + \
                self.prompt_seperator + \
                self.add_text1 + \
                self.prompt_seperator + \
                query_prompt + \
                self.prompt_suffix
        return prompt

    def create_doc_pool_string(self, shuffled_keys, all_docs):
        doc_lengths = []
        doc_list_str = []
        all_schemas = ""
        doc_spans = []
        doc_st_index = self.prompt_prefix_length + 1 # includes " \n\n"
        
        for ix, key in enumerate(shuffled_keys):
            value = all_docs[key]
            doc_list_str.append(key)
            text = self.prompt_seperator
            doc_text = self.doc_text(idx=self.dict_doc_name_id[key] + 1, doc_name=key, doc_info=value).strip()
            doc_text_len = len(self.tokenizer(doc_text, add_special_tokens=False).input_ids)
            text += doc_text
            doc_spans.append((doc_st_index, doc_st_index + doc_text_len))
            doc_st_index = doc_st_index + 1 + doc_text_len
            doc_lengths.append(doc_text_len)
            all_schemas += text
            
        doc_list_str = ", ".join(doc_list_str)    
        return all_schemas, doc_list_str, doc_lengths, doc_spans

def get_queries_and_items():
    with open("data/test_queries.json", "r") as f: test_queries = json.load(f)
    with open("data/train_queries.json", "r") as f: train_queries  = json.load(f)
    with open("data/tools.json", "r") as f: tools = json.load(f)
    return train_queries, test_queries, tools