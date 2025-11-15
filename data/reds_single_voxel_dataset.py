import numpy as np
import os.path as osp
import random

import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, paired_event_random_crop, augment_with_event
from basicsr.utils import FileClient, get_root_logger, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class REDSVoxelDataset(data.Dataset):
    """REDS voxel dataset for pretrain unet

    idx -> {voxel: [B, Bins, H, W], I1: [B, C, H, W], I2: [B, C, H, W]}

    

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.lq_root = opt[dataroot]
        self.keys = []
        self.name = opt[name]
        self.is_pretrain = opt[is_pretrain]

        with open(opt[meta_info_file], r) as fin:
            for line in fin:
                name, _ = line.split( )
                self.keys.extend([f{name}/{i:06d} for i in range(int(99))])
        fin.close()


        ## So here, we can get self.keys(list)
        ## Example:
        ## self.keys[0] = 00001_0001.h5

        # file client(io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.is_hdf5 = False
        if self.io_backend_opt[type] == hdf5:
            self.is_hdf5 = True
            self.io_backend_opt[h5_paths] = [self.lq_root]
            self.io_backend_opt[client_keys] = [LR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_pretrain] = self.is_pretrain
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]
        clip_name, frame_name = osp.dirname(key), osp.basename(key)
        self.io_backend_opt[h5_clip] = clip_name
        # if self.file_client is None:
        self.file_client = FileClient(self.io_backend_opt[type], **self.io_backend_opt)

        start_frame_idx = int(frame_name)
        end_frame_idx = start_frame_idx + 2
        self.neighbor_list = list(range(start_frame_idx, end_frame_idx, 1))


        # print(neighbor_list: , self.neighbor_list, \n)
        I1, I2, voxel = self.file_client.get(self.neighbor_list)
        I1 = img2tensor(I1)
        I2 = img2tensor(I2)
        voxel = torch.from_numpy(voxel)


        return {I1: I1, I2: I2, voxel: voxel}