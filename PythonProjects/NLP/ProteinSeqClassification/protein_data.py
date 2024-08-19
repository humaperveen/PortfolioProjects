#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, sequences, targets, tokenizer, max_len):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, item):
        # sequence = str(self.sequences[item])
        sequence = self.sequences[item]
        target = self.targets[item]
        
        encoding = self.tokenizer.encode_plus(
          sequence,
          add_special_tokens = True,
          max_length = self.max_len,
          return_token_type_ids = False,
          padding = 'max_length',
          truncation = 'longest_first',
          return_attention_mask = True,
          return_tensors = 'pt',
        )
        
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          #'token_type_ids': encoding['token_type_ids'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

class ProtDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sequence = str(self.data.sequences[index])
        # title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=False,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.classes[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len