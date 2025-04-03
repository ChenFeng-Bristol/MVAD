import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, weight, temperature, world_size=1):
        '''
        Contrastive learning for artefact-related features, reducing 
        similarity between features of videos containing the same artefacts
        Adapted from https://github.com/pavancm/CONTRIQUE?tab=readme-ov-file
        '''
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.world_size = world_size


    def forward(self, label_dict, z_dict):
        '''
        Args:
            label_dict - {artf: (B,) for artf in artefacts}, binary labels for each artefact
            z_dict - {artf: (B,N) for artf in artefacts} feature projections for each artefact
        '''

        loss = 0.
        for artf in label_dict.keys():
            z = z_dict[artf] # (B,N)
            labels = label_dict[artf] # (B,)
            N = labels.shape[0] * self.world_size 

            if self.world_size > 1:
                z = torch.cat(GatherLayer.apply(z), dim=0)
                labels = torch.cat(GatherLayer.apply(labels), dim=0)

            # normalise z
            z = F.normalize(z, p=2, dim=1)
            sim = torch.mm(z, z.T) / self.temperature # (B,B)

            # need to turn labels into one-hot vectors otherwise 0-label videos never get positive sample
            labels = F.one_hot(labels).float()
            positive_mask = torch.mm(labels, labels.T) # (B,B)
            positive_mask = positive_mask.fill_diagonal_(0).to(sim.device) # exclude sample itself from its positive samples
            zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)
            
            # calculate normalized cross entropy value
            positive_sum = torch.sum(positive_mask, dim=1) # number of positive samples for each datapoint
            denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
            loss += torch.mean(torch.log(denominator) - \
                            (torch.sum(sim * positive_mask, dim=1)/positive_sum))
            
        return self.weight * loss
