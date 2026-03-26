import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import XLMRobertaModel, AutoTokenizer
from xai_roberta import XLMRobertaEncoderXAI  # Using your provided file
from test_bilrp_func import get_explanation

class E5ModelXAI(nn.Module):
    def __init__(self, model_name_or_path, device="cuda"):
        super().__init__()
        # 1. Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        self.temp_model = XLMRobertaModel.from_pretrained(model_name_or_path)
        self.config = self.temp_model.config
        
        # 3. Extract components
        self.embeddings = self.temp_model.embeddings
        
        # Replace the standard encoder with your custom XAI encoder
        self.encoder = XLMRobertaEncoderXAI(self.config, lrp=True)
        self.encoder.load_state_dict(self.temp_model.encoder.state_dict())
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _pool_encoding(self, last_hidden_state, attention_mask):
        """Standard E5 Mean Pooling."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward_explain(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 1. Forward Pass using the custom encoder
        emb = self.embeddings(input_ids)
        ext_mask = self.temp_model.get_extended_attention_mask(attention_mask, input_ids.size(), self.device)
        
        # This returns the activations in dict 'A'
        
        
        Rsens = []
        # 2. Manual Layer-by-Layer LRP
        for j in range(self.config.hidden_size):
            # Re-run forward pass as per the authors' implementation
            enc_outs = self.encoder.forward_explain(emb, ext_mask)
            sentence_embeddings = self._pool_encoding(enc_outs['last_hidden_state'], attention_mask)
            # 1. Start at the top: Pooling layer
            A = enc_outs['A']
            # Detach and set as leaf node
            sentence_embeddings_data = sentence_embeddings.data.squeeze().requires_grad_(True)
            
            # 2. Backward for dimension j
            sen_feature = sentence_embeddings_data[j]
            sen_feature.backward() 
            
            # 3. Initial relevance (R0)
            Rsen_ = (sentence_embeddings_data.grad * sentence_embeddings)
            # 4. The Layer-wise handoff loop
            for i, _ in list(enumerate(self.encoder.layer))[::-1]:
                Rsen_.sum().backward()
                
                # Calculate local relevance: Relevance = Gradient * Activation
                Rsen_next = (A[f'attn_input_{i}_data'].grad * A[f'attn_input_{i}'])
                A[f'attn_input_{i}_data'].grad = None
                Rsen_ = Rsen_next
            # Rsen_ now contains the token-wise relevance for dimension j
            Rsens.append(Rsen_.sum(2).detach().cpu().numpy())

        return {
            "sentence_embeddings": [sentence_embeddings],
            "Rsen": [np.array(Rsens)[np.newaxis, :, :]],
            "last_hidden_state": enc_outs['last_hidden_state']
        }