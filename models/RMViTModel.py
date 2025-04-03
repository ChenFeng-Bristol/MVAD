# Copied over from https://github.com/jasminepp/RMTBVQA

import torch.nn as nn
import torch
from .RMViT import RecurrentWrapperWithViT 
class RMViTModel(nn.Module):
    """
    c_out inception_time output
    n_out model output
    """
    def __init__(self, num_mem_token, emb_dim, segment_size, projection_dim, normalize):
        # super().__init__(**kwargs)
        super().__init__()
        self.encoder = RecurrentWrapperWithViT(num_mem_token=num_mem_token, emb_dim =emb_dim, segment_size=segment_size, segment_alignment='left', k2 =3, max_n_segments=14)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        self.normalize = normalize
        self.c_out = emb_dim

    def forward(self, x_i, x_j):
        # global features
        h_i, h_i_T= self.encoder(x_i)
        h_j,_= self.encoder(x_j)
        h_i_original = self.avgpool_feature(h_i)
        h_j_original =self.avgpool_feature(h_j)

        # local features
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)
        h_i_patch = h_i_patch.reshape(-1,self.n_features,\
                                    self.patch_dim[0]*self.patch_dim[1])

        h_j_patch = h_j_patch.reshape(-1,self.n_features,\
                                    self.patch_dim[0]*self.patch_dim[1])
 
        h_i_patch = torch.transpose(h_i_patch,2,1)
        h_i_patch = h_i_patch.reshape(-1, self.n_features)
        h_j_patch = torch.transpose(h_j_patch,2,1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)

        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)
        h_i_T = self.avgpool(h_i_T)

        h_i = h_i.view(-1, self.n_features)
        h_j = h_j.view(-1, self.n_features) 
        h_i_T = h_i_T.view(-1, 192)
        h_i_predict = self.prediction_head(h_i_T)
        
        
        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)
            h_i_T = nn.functional.normalize(h_i_T, dim=1)
            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)
        
        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        z_i_T = self.projector2(h_i_T)
        z_i_predict = self.projector2 (h_i_predict)
        
    
        # local projections
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)

        
        return z_i, z_j, z_i_patch, z_j_patch, z_i_predict, z_i_T, h_i_original, h_i
