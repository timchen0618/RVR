# INF-Retriever model wrapper
# Based on infly/inf-retriever-v1-1.5b

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Tuple

from src import utils, inbatch


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool embeddings by taking the last token's representation
    Handles both left and right padding
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class INFRetriever(nn.Module):
    
    def __init__(self, base_model, pooling="last_token", **kwargs):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.config = base_model.config
        if not hasattr(self.config, "pooling"):
            self.config.pooling = pooling
    
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
        normalize=True,
    ):
        # Get model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        last_hidden = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling == "last_token":
            emb = last_token_pool(last_hidden, attention_mask)
        elif self.pooling == "average":
            last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Normalize if requested (INF-Retriever typically uses normalized embeddings)
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
            
        return emb


def load_retriever(model_path, pooling="last_token", random_init=False):

    # Check if loading from local checkpoint
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        # Loading from local checkpoint (following Contriever format)
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        
        # For INF-Retriever, default to the base model
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            retriever_model_id = "infly/inf-retriever-v1-1.5b"
        
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(
            retriever_model_id,
            trust_remote_code=True,
        )
        cfg = AutoConfig.from_pretrained(retriever_model_id, trust_remote_code=True)
        
        # Determine model class based on training mode
        if hasattr(opt, "training_mode"):
            print(f"Loading INF-Retriever with training mode: {opt.training_mode}")
            if opt.training_mode in ['standard', 'base']:
                model_class = inbatch.InBatchINF
            elif opt.training_mode == 'gru':
                model_class = inbatch.InBatchINFGRU
            elif opt.training_mode == 'linear_projection':
                model_class = inbatch.InBatchINFLinearProjection
            else:
                model_class = inbatch.InBatchINF
        else:
            # Load base INF-Retriever
            base_model = AutoModel.from_pretrained(retriever_model_id, trust_remote_code=True)
            retriever = INFRetriever(base_model, pooling=pooling)
            retriever.load_state_dict(pretrained_dict["model"], strict=False)
            return retriever, tokenizer, retriever_model_id
        
        # Load the model with specific training mode
        pretrained_dict = pretrained_dict["model"]
        model = model_class(opt, None, None)
        model.load_state_dict(pretrained_dict, strict=True)
        model.eval()
        
        if opt.training_mode in ['standard', 'base']:
            retriever = model.encoder
        else:
            retriever = model
            
    else:
        retriever_model_id = model_path
        print(f"Loading INF-Retriever from HuggingFace: {retriever_model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            retriever_model_id,
            trust_remote_code=True
        )
        
        base_model = AutoModel.from_pretrained(
            retriever_model_id,
            trust_remote_code=True
        )
        
        retriever = INFRetriever(base_model, pooling=pooling)
    
    return retriever, tokenizer, retriever_model_id