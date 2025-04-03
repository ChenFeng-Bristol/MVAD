import numpy as np
import random
import torch
import decord
from decord import VideoReader


decord.bridge.set_bridge("torch")

def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)



    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


def get_cropped_video(
    video,
    size_h=224,
    size_w=224,
    **kwargs,
):
    kwargs["fragments_h"], kwargs["fragments_w"] = 1, 1
    kwargs["fsize_h"], kwargs["fsize_w"] = size_h, size_w
    return get_spatial_fragments(video, **kwargs)


def get_single_sample(
    video,
    sample_type="resize",
    **kwargs,
):
    if sample_type.startswith("fragments"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type.startswith("crop"):
        video = get_cropped_video(video, **kwargs)
    elif sample_type == "original":
        return video
        
    return video


def get_spatial_and_temporal_samples(
    video_path,
    sampler,
    sampling_config,
    is_train=False,
    augment=False,
):
    '''
    Sample fragments. Copied from https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
    Args:
        sampler -- sampler object, e.g. FragmentSampleFrames
    '''
    video = {}
    assert not video_path.endswith(".yuv"), "Reading large YUVs is not supported yet!"

    vreader = VideoReader(video_path)
    ### Avoid duplicated video decoding!!! Important!!!!
    stype = sampler.type
    all_frame_inds = []
    frame_inds = {}
    frame_inds[stype] = sampler(len(vreader), is_train)
    all_frame_inds.append(frame_inds[stype])
        
    ### Each frame is only decoded one time!!!
    all_frame_inds = np.concatenate(all_frame_inds,0)
    frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}
    
    imgs = [frame_dict[idx] for idx in frame_inds[stype]]
    video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    sampled_video[stype] = get_single_sample(video[stype], stype, **sampling_config)
    return sampled_video, frame_inds


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0.):
        self.type = 'fragments'
        self.fragments_t = fragments_t # 8
        self.fsize_t = fsize_t # 4
        self.size_t = fragments_t * fsize_t # 32
        self.frame_interval = frame_interval # 2
        self.num_clips = num_clips # 1
        self.drop_rate = drop_rate # 0

        print(f'num_clips={num_clips}, batch size will be multiplied by this factor!')

    def get_frame_indices(self, num_frames, train=False):
        '''
        Divide N frames into `fragments_t` segments, then from each segment randomly
        sample `fsize_t` continuous frames (with interval `frame_interval`).
        '''

        # get index of starting frame in each segment
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        # get length of each segment
        tlength = num_frames // self.fragments_t # 75 for 600

        # get relative indices of the frames to sample in each segment
        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)
        
        # get absolute indices of the frames to sample in each segment
        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval # [0,2,4,6]
            + rnd_t[:, None]
            + tgrids[:, None]
        ) # 8 x 4 continuous (with interval of 2 frames) 
        
        
        drop = random.sample(list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate))
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]
            
        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)