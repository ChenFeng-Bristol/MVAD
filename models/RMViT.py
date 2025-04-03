# Copied over from https://github.com/jasminepp/RMTBVQA

import torch.nn as nn
from torch.nn import functional as F

import math
import torch
from torch.nn import CrossEntropyLoss

from .vit_for_small_dataset import ViT
class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, emb_dim):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        self.create_memory(num_mem_tokens,emb_dim)

    def create_memory(self, num_mem_tokens,emb_dim):
        memory_tensor = torch.randn(1, num_mem_tokens, emb_dim)
        memory_weights = torch.nn.init.kaiming_normal_(torch.empty(1, num_mem_tokens, emb_dim), mode='fan_in', nonlinearity='relu')
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
        self.read_memory_position = range(num_mem_tokens)


    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        inputs_embeds = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(inputs_embeds)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state
    
    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        seg_kwargs = dict(**kwargs)
        inputs_embeds = torch.cat([memory_state,  input_ids], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        return inputs_embeds
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            memory_state = model_outputs[:, :self.num_mem_tokens]
            out = model_outputs[:, self.num_mem_tokens:]
        else:
            memory_state = None
            out = model_outputs
            
        return out, memory_state 

import random
class RecurrentWrapperWithViT(nn.Module):
    def __init__(self, num_mem_token, emb_dim, **rmt_kwargs):
        super(RecurrentWrapperWithViT, self).__init__()
        self.rmt_config = rmt_kwargs
        self.num_mem_tokens = num_mem_token
        self.base_model = ViT(
        num_mem_token=num_mem_token, 
        segment_size=self.rmt_config.get('segment_size'),
        dim =emb_dim, 
        depth = 8,
        heads = 64,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1) 
        self.memory_cell = MemoryCell(self.base_model, num_mem_token, emb_dim =emb_dim) 

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        first_segment = False
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            if first_segment:

                initial_memory_state =  torch.cat([segment['input_ids'], segment['input_ids']], dim=1)
                cell_out, memory_state = self.memory_cell(**segment, memory_state=initial_memory_state, output_hidden_states=True)
                first_segment = False
            else:
                cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)

            self.manage_gradients(memory_state, seg_num)
        
        cell_outputs_tensor = torch.stack(cell_outputs, dim=0)
        out = torch.mean(cell_outputs_tensor, dim=0)
        out = torch.cat([memory_state, out], dim=1)
        return out
    
    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:]) if end - start == segment_size]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:]) if end - start == segment_size]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = [chunk for chunk in torch.chunk(tensor, n_seg, dim=1) if chunk.shape[1] == segment_size]
        else:
            raise NotImplementedError
        return segments

  
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False
