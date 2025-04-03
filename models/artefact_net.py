import os
import math
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional.classification import binary_accuracy
from contextlib import contextmanager
import einops

from models.swin_backbone import SwinTransformer3D
from utils import instantiate_from_config, LitEma



class VQAHead(nn.Module):
    """MLP Regression Head for VQA. Copied from https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x)))).mean((-3, -2, -1)).squeeze()
        return qlt_score


class Projector(nn.Module):
    """MLP Projecting to obtain latent z for contrastive learning.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
    """

    def __init__(
        self, in_channels, hidden_channels, z_dim
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.z_dim = z_dim

        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1)) # (B,hid_ch,D,H,W)
        self.fc_last = nn.Conv3d(self.hidden_channels, self.z_dim, (1, 1, 1)) # (B, z_dim, D, H, W)
        self.gelu = nn.GELU()


    def forward(self, x):
        z = self.fc_last(self.gelu(self.fc_hid(x))).contiguous().view(x.shape[0], -1) # (B, n)
        return z


class ArtefactNet(nn.Module):
    def __init__(self, artefacts, feat_dim=768, head_dim=64, head_dropout=0.5, pretrained_backbone_path=None):
        super().__init__()

        self.feat_extractor = SwinTransformer3D()
        # prediction heads for 10 artefacts
        self.artefacts = artefacts
        for artf in artefacts:
            setattr(self, 'head_'+artf, VQAHead(feat_dim, head_dim, head_dropout))

        if pretrained_backbone_path is not None:
            bb_dict = torch.load(pretrained_backbone_path)
            if 'state_dict' in bb_dict.keys():
                bb_dict = bb_dict['state_dict']
            missing, unexpected = self.feat_extractor.load_state_dict(bb_dict, strict=False)
            print(f"Restored from {pretrained_backbone_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")


    def forward(self, vclips, return_feat=False):
        '''
        Forward pass of the model. Taking fragments as input.

        Returns:
            res_dict -- {artf: (B,) for artf in artefacts}, logits for each artefact
            [Optional] feat -- (B,C,D,H,W) video features extracted from SWIN backbone
        '''
        feat = self.feat_extractor(vclips, multi=False, layer=-1, adaptive_window_size=True)
        res_dict = {}
        for artf in self.artefacts:
            res_dict[artf] = getattr(self, 'head_'+artf)(feat.contiguous())
        
        if return_feat:
            return res_dict, feat
        return res_dict, None


class ArtefactDetector(pl.LightningModule):
    def __init__(self,
                 model_config,
                 optimizer_config,
                 scheduler_config=None,
                 input_key='fragments', 
                 contrastive_loss_config=None,
                 use_ema=True,
                 monitor='val/bce_loss'):
        super().__init__()
        self.input_key = input_key
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.model = instantiate_from_config(model_config)

        if contrastive_loss_config is not None:
            assert contrastive_loss_config['projector_params']['in_channels'] == model_config['params']['feat_dim'], \
                'Feature dimension not consistent between projector and cls head!'
            for artf in self.model.artefacts:
                setattr(self, 'proj_'+artf, Projector(**contrastive_loss_config['projector_params']))
            self.contrastive_loss = instantiate_from_config(contrastive_loss_config) # TODO: need to determine weight of feat loss

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        
        if monitor is not None:
            self.monitor = monitor


    def shared_step(self, batch):
        '''
        Shared step for training and validation, i.e. forward pass and compute losses
        '''
        # get data
        vclips = batch[self.input_key] # tensor of shape (B,C,T,H,W), i.e. fragments of input
        y_dict = batch['labels'] # dict {artefact: binary label (B,)}
        batch_size = vclips.shape[0]

        if len(vclips.shape) == 6: # (B,n,C,T,H,W), i.e. num_clips:=n > 1
            # combine clips as batches and replicate labels
            num_clips = vclips.shape[1]
            vclips = einops.rearrange(vclips, 'b n c t h w -> (n b) c t h w').contiguous()
            y_dict = {artf: y_dict[artf].repeat(num_clips) for artf in self.model.artefacts}
            batch_size *= num_clips

        # infer: extract feats + compute logits
        y_pred_dict, feat = self.model(vclips, return_feat=hasattr(self, "contrastive_loss"))

        # compute loss
        prefix = 'train' if self.training else 'val'
        log_dict = {}

        total_loss = 0.
        # BCE
        bce_loss = 0.
        for artf in self.model.artefacts:
            bce_loss += F.binary_cross_entropy_with_logits(y_pred_dict[artf], y_dict[artf].float())
        log_dict[f'{prefix}/bce_loss'] = bce_loss
        total_loss += bce_loss

        # Contrastive loss
        if feat is not None:
            z_dict = {}
            for artf in self.model.artefacts:
                # get projection for each artefact
                z_dict[f'{artf}'] = getattr(self, 'proj_'+artf)(feat.contiguous()) # (B,N)
            contrastive_loss = self.contrastive_loss(y_dict, z_dict)
            log_dict[f'{prefix}/contrastive_loss'] = contrastive_loss
            total_loss += contrastive_loss

        log_dict[f'{prefix}/total_loss'] = total_loss

        # accuracy
        for artf in self.model.artefacts:
            log_dict[f'{prefix}/{artf}_acc'] = binary_accuracy(torch.sigmoid(y_pred_dict[artf]), y_dict[artf])            

        return total_loss, log_dict, batch_size


    def training_step(self, batch, batch_idx):
        loss, log_dict, batch_size = self.shared_step(batch)

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('global_step', float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                _, loss_dict, batch_size = self.shared_step(batch)
                log_dict = {key + '_ema': loss_dict[key] for key in loss_dict}
        else:
            _, log_dict, batch_size = self.shared_step(batch)
        
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)


    def on_test_start(self):
        print('=================== Testing begins')
        # variables to help calculate epoch level test stats
        self.test_step_num_correct_dict = {artf: 0 for artf in self.model.artefacts}
        self.test_step_num_total_dict = {artf: 0 for artf in self.model.artefacts}
        self.pred_dict = {}
        # set up test dir
        self.test_prefix = f'test_{self.test_name}'
        self.test_dir = os.path.join(self.logger.save_dir, self.test_prefix)
        os.makedirs(self.test_dir, exist_ok=True)
        print(f'=================== Saving predictions to {self.test_dir}')


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        '''
        Test batch size is hard set to 1 for now to make it easy to infer on a video by sampling multiple fragments.
        '''
        # get data
        vclips = batch[self.input_key] # tensor of shape (B,C,T,H,W), i.e. fragments of input
        y_dict = batch['labels'] # dict {artefact: binary label (B,)}
        subset_flags_dict = batch['subset_flags'] # dict {artefact: binary (B,)} indicating whether the samples belong to subset of artefacts    
        vnames = batch['name'] # list of length B
        vname = vnames[0] # bs=1
        self.pred_dict[vname] = {}

        if len(vclips.shape) == 6: # (B,n,C,T,H,W), i.e. num_clips:=n > 1
            # combine clips as batches
            vclips = einops.rearrange(vclips, 'b n c t h w -> (n b) c t h w').contiguous()

        # infer on test batch
        if self.use_ema:
            with self.ema_scope():
                y_pred_dict, _ = self.model(vclips)
        else:
            y_pred_dict, _ = self.model(vclips)

        # record number of correct predictions and total number of samples in the batch for each artefact
        for artf in self.model.artefacts:
            # convert logits to binary predictions (threshold 0.5 for sigmoid), avg muliple clips
            y_pred_binary = (torch.sigmoid(y_pred_dict[artf]).amax(dim=0,keepdim=True) >= 0.5)
            # record predictions
            self.pred_dict[vname][artf] = int(y_pred_binary.item())

            correct = (y_pred_binary == y_dict[artf]).float()
            # mask out samples not in current sub testset
            if artf in subset_flags_dict.keys():
                correct *= subset_flags_dict[artf].float()
            num_correct = torch.sum(correct)
            num_total = torch.sum(subset_flags_dict[artf]) if artf in subset_flags_dict.keys() else correct.shape[0]
            self.test_step_num_correct_dict[artf] += num_correct
            self.test_step_num_total_dict[artf] += num_total
        

    def on_test_epoch_end(self):
        log_dict = {}
        # compute and log overall test accuracies
        for artf in self.model.artefacts:
            log_dict[f'{self.test_prefix}/{artf}_acc'] = self.test_step_num_correct_dict[artf] / self.test_step_num_total_dict[artf]
            # reset for next epoch
            self.test_step_num_correct_dict[artf] = 0
            self.test_step_num_total_dict[artf] = 0

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # save pred_dict
        import json
        with open(os.path.join(self.test_dir, 'pred_dict.json'), 'w') as f:
            json.dump(self.pred_dict, f, indent=4)


    def configure_optimizers(self):
        param_groups = [
            {'params': self.model.feat_extractor.parameters(), 'lr': self.optimizer_config['learning_rate']*self.optimizer_config['lr_factor_bb']},
        ]
        for artf in self.model.artefacts:
            param_groups.append({'params': getattr(self.model, 'head_'+artf).parameters(), 'lr': self.optimizer_config['learning_rate']})

        if hasattr(self, "contrastive_loss"):
            for artf in self.model.artefacts:
                param_groups.append({'params': getattr(self, 'proj_'+artf).parameters(), 'lr': self.optimizer_config['learning_rate']})
        optimizer = optim.AdamW(params=param_groups, 
                                lr=self.optimizer_config['learning_rate'],
                                weight_decay=self.optimizer_config['weight_decay'])
    
        if self.scheduler_config is not None:
            # use the same schedule as FASTVQA
            warmup_iter = self.scheduler_config['warmup_iters']

            max_iter = self.scheduler_config['max_iters'] # num_epochs * len(train_loader)
            lr_lambda = (
                lambda cur_iter: cur_iter / warmup_iter
                if cur_iter <= warmup_iter
                else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
            )
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                'interval': 'step',
                'frequency': 1
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return optimizer


    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
