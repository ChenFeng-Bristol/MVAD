import torch
import glob
import os
import json

from utils import instantiate_from_config
from data.sampling import get_spatial_and_temporal_samples




artefacts = ['motion_blur', 'dark_scenes', 'graininess', 'aliasing', 'banding', 'blockiness', 'spatial_blur', 'frame_drop', 'transmission_error', 'black_screen']
class BviArtefact(torch.utils.data.Dataset):
    def __init__(self, data_dir, sampler_config, sampling_config, phase='train', augment=False):
        '''
        Args
            sampler_config -- args to sampler class, e.g. temporal frag num&size&frame interval
            sampling_config -- args to spatial frag sampling function, e.g. spatial frag num&size etc.
        '''
        super().__init__()

        all_videos = glob.glob(os.path.join(data_dir, 'avi', '*.avi'))
        with open(os.path.join(data_dir, 'val_split.json'), 'r') as f:
            val_split = set(json.load(f)['val'])
            assert len(val_split) == 4800

        if phase == 'train':
            self.video_list = [avi for avi in all_videos if os.path.split(avi)[1] not in val_split]
        elif phase == 'val':
            self.video_list = [avi for avi in all_videos if os.path.split(avi)[1] in val_split]
        else:
            raise ValueError("split must be either train or val")

        with open(os.path.join(data_dir, 'labels.json'), 'r') as f:
            self.label_dict = json.load(f)

        self.sampler = instantiate_from_config(sampler_config)
        self.sampling_config = sampling_config
        self.phase = phase
        self.augment = augment

    def __getitem__(self, index):
        video_path = self.video_list[index]

        # get fragments, 'data': {'fragments': (C,T,H,W) tensor}
        data, frame_inds = get_spatial_and_temporal_samples(video_path, self.sampler, self.sampling_config, 
                                                            self.phase == "train", 
                                                            self.augment and (self.phase == "train"))
        # decide whether to augment
        data[self.sampler.type] = data[self.sampler.type] / 127.5 - 1.0 # scale to [-1,1]

        if self.sampler.num_clips > 1:
            # split into two fragments
            data[self.sampler.type] = data[self.sampler.type].reshape(
                data[self.sampler.type].shape[0], 
                self.sampler.num_clips, 
                -1,
                *data[self.sampler.type].shape[2:]
            ).transpose(0,1) # (C,nT,H,W) -> (C,n,T,H,W) -> (n,C,T,H,W)
            
            frame_inds[self.sampler.type] = frame_inds[self.sampler.type].reshape(self.sampler.num_clips, -1)

        # get label for each artefact
        label = {}
        fname = os.path.split(video_path)[1]
        fname_yuv = fname.replace('.avi', '.yuv')
        for artf in artefacts:
            label[artf] = self.label_dict[fname_yuv]['noise_config'][artf]['active']

        data["labels"] = label
        data["name"] = fname
        data["frame_inds"] = frame_inds

        return data


    def __len__(self):
        return len(self.video_list)

    

class BviArtefactTestset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sampler_config, sampling_config):
        '''
        Args
            sampler_config -- args to sampler class, e.g. temporal frag num&size&frame interval
            sampling_config -- args to spatial frag sampling function, e.g. spatial frag num&size etc.
        '''
        super().__init__()

        # update these if necessary
        self.video_list = sorted(glob.glob(os.path.join(data_dir, 'avi', '*.avi')))
        with open(os.path.join(data_dir, 'labels.json')) as f:
            self.label_dict = json.load(f)
        
        with open(os.path.join(data_dir, 'subsets.json')) as f:
            self.subset_dict = json.load(f)
        self.subset_dict = {k: set(v) for k, v in self.subset_dict.items()}
        
        self.sampler = instantiate_from_config(sampler_config)
        self.sampling_config = sampling_config

    def __getitem__(self, index):
        video_path = self.video_list[index]

        # get fragments, `data`: {'fragments`: (C,D,H,W) tensor}
        data, frame_inds = get_spatial_and_temporal_samples(video_path, self.sampler, self.sampling_config, is_train=False, augment=False)

        for k, v in data.items():
            # v is (C,D,H,W)
            data[k] = v / 127.5 - 1.0 # scale to [-1,1]
        
        if self.sampler.num_clips > 1:
            # split into two fragments
            data[self.sampler.type] = data[self.sampler.type].reshape(
                data[self.sampler.type].shape[0], 
                self.sampler.num_clips, 
                -1,
                *data[self.sampler.type].shape[2:]
            ).transpose(0,1) # (C,nT,H,W) -> (C,n,T,H,W) -> (n,C,T,H,W)
            
            frame_inds[self.sampler.type] = frame_inds[self.sampler.type].reshape(self.sampler.num_clips, -1)

        # get label for each artefact
        label = {}
        fname = os.path.split(video_path)[1]
        fname_yuv = fname.replace('.avi', '.yuv')
        label['motion_blur'] = int('Motion' in self.label_dict[fname_yuv]['output'])
        label['dark_scenes'] = int('Dark' in self.label_dict[fname_yuv]['output'])
        label['graininess'] = int('graininess' in self.label_dict[fname_yuv]['output'])
        label['aliasing'] = int('aliasing' in self.label_dict[fname_yuv]['output'])
        label['banding'] = int('banding' in self.label_dict[fname_yuv]['output'])
        label['blockiness'] = int('QP47' in self.label_dict[fname_yuv]['output'])
        label['spatial_blur'] = int(self.label_dict[fname_yuv]['noise_config']['spatial_blur']['active'])
        label['frame_drop'] = int(self.label_dict[fname_yuv]['noise_config']['frame_drop']['active'])
        label['transmission_error'] = int(self.label_dict[fname_yuv]['noise_config']['transmission_error']['active'])
        label['black_screen'] = int(self.label_dict[fname_yuv]['noise_config']['black_screen']['active'])

        # also indicate whether this video is in the test subset of particular artefacts
        subset_flag = {artf: int(fname_yuv in self.subset_dict[artf]) for artf in ['motion_blur', 'dark_scenes', 'graininess', 'aliasing', 'banding']}

        data['subset_flags'] = subset_flag
        data["labels"] = label
        data["name"] = fname
        data["frame_inds"] = frame_inds

        return data
    

    def __len__(self):
        return len(self.video_list)