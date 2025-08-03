# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# contriever.py
import os
import torch
import transformers
from transformers import BertModel, XLMRobertaModel

from src import utils, inbatch


# Add this to your contriever.py file

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QwenRetriever(nn.Module):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__()
        self.config = config
        self.config.pooling = pooling
        
        # Load the base Qwen model
        self.qwen_model = AutoModel.from_pretrained(config.name_or_path, trust_remote_code=True)
        
        # Get hidden size from config
        self.hidden_size = self.qwen_model.config.hidden_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):
        # Forward pass through Qwen model
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden states
        last_hidden = outputs.last_hidden_state
        
        # Apply attention mask
        if attention_mask is not None:
            last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        # Apply pooling strategy
        if self.config.pooling == "average":
            # Average pooling over sequence length
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            # Use first token (similar to CLS in BERT)
            emb = last_hidden[:, 0]
        elif self.config.pooling == "last":
            # Use last non-padded token
            batch_size = last_hidden.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            emb = last_hidden[torch.arange(batch_size), sequence_lengths]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
            
        return emb


def load_retriever(model_path, pooling="average", random_init=False):
    # Check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            retriever_model_id = "Qwen/Qwen2-0.5B-Instruct"
            
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id, trust_remote_code=True)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id, trust_remote_code=True)
        
        # Add model path to config for QwenRetriever
        cfg.name_or_path = retriever_model_id

        if hasattr(opt, "run_name"):
            if hasattr(opt, "training_mode"):
                print(opt.training_mode)
                if opt.training_mode in ['standard']:
                    model_class = inbatch.InBatch
                elif opt.training_mode == 'gru':
                    model_class = inbatch.InBatchGRU
                elif opt.training_mode == 'linear_projection':
                    model_class = inbatch.InBatchLinearProjection
                else:
                    model_class = inbatch.InBatch
            else:
                raise NotImplementedError
        else:
            model_class = QwenRetriever
            
        print(f"Using model class: {model_class}, retriever_model_id: {retriever_model_id}")
        pretrained_dict = pretrained_dict["model"]
        
        if model_class in [QwenRetriever]:
            retriever = model_class(cfg)
            retriever.load_state_dict(pretrained_dict, strict=True)
        else:
            if opt.training_mode in ['standard']:
                model = inbatch.InBatch(opt, None, None)
                print("Using model = InBatch", flush=True)
            elif opt.training_mode == 'gru':
                model = inbatch.InBatchGRU(opt, None, None)
                print("Using model = InBatchGRU", flush=True)
            elif opt.training_mode == 'linear_projection':
                model = inbatch.InBatchLinearProjection(opt, None, None)
                print("Using model = InBatchLinearProjection", flush=True)
            else:
                model = inbatch.InBatch(opt, None, None)

            model.load_state_dict(pretrained_dict, strict=True)
            model.eval()
            print('finish getting model')
            if opt.training_mode in ['standard']:
                retriever = model.encoder
            else:
                retriever = model
        
    else:
        # Loading base Qwen model from HuggingFace
        retriever_model_id = model_path
        model_class = QwenRetriever
        
        # Load config and tokenizer with trust_remote_code=True for Qwen
        cfg = utils.load_hf(transformers.AutoConfig, model_path, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path, trust_remote_code=True)
        
        # Add model path to config
        cfg.name_or_path = model_path
        cfg.pooling = pooling
        
        # Initialize QwenRetriever
        retriever = model_class(cfg, pooling=pooling)

    return retriever, tokenizer, retriever_model_id