import numpy as np
import random
import torch
from pathlib import Path
import os.path as osp
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CEDRecurrentDataset(data.Dataset):
    """CED dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_CED_train.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    simple/simple_wires_1/images 370 (260,346,3)
    simple/simple_color_keyboard_1/images 263 (260,346,3)
    people/people_static_air_guitar/images 718 (260,346,3)
    ...

    Key examples: simple/simple_wires_1/images/000000
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. REDS4 or
                official.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt[dataroot_gt]), Path(opt[dataroot_lq])
        self.num_frame = opt[num_frame]

        self.keys = []
        self.frame_num_dict = {}
        with open(opt[meta_info_file], r) as fin:
            for line in fin:
                folder, frame_num, _ = line.split( )
                self.keys.extend([f{folder}/{i:06d} for i in range(int(frame_num))])
                self.frame_num_dict[folder] = frame_num # beacuse of class imbalance, so trace each folders frame_num

        fin.close()

        # remove the video clips used in validation
        # val_partition = [
        #     people/people_dynamic_wave/images,
        #     indoors/indoors_foosball_2/images,
        #     simple/simple_wires_2/images,
        #     people/people_dynamic_dancing/images,
        #     people/people_dynamic_jumping/images,
        #     simple/simple_fruit_fast/images,
        #     additional_IR_filter/outdoor_jumping_infrared_2/images,
        #     simple/simple_carpet_fast/images,
        #     people/people_dynamic_armroll/images,
        #     indoors/indoors_kitchen_2/images,
        #     people/people_dynamic_sitting/images
        # ]

        # if opt[test_mode]:
        #     self.keys = [v for v in self.keys if osp.dirname(v) in val_partition]
        # else:
        #     self.keys = [v for v in self.keys if osp.dirname(v) not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.is_lmdb = False
        if self.io_backend_opt[type] == lmdb:
            self.is_lmdb = True
            if hasattr(self, flow_root) and self.flow_root is not None:
                self.io_backend_opt[db_paths] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt[client_keys] = [lq, gt, flow]
            else:
                self.io_backend_opt[db_paths] = [self.lq_root, self.gt_root]
                self.io_backend_opt[client_keys] = [lq, gt]

        # temporal augmentation configs
        self.interval_list = opt.get(interval_list, [1])
        self.random_reverse = opt.get(random_reverse, False)
        interval_str = ,.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(fTemporal augmentation interval list: [{interval_str}]; 
                    frandom reverse is {self.random_reverse}.)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop(type), **self.io_backend_opt)

        scale = self.opt[scale]
        gt_size = self.opt[gt_size]
        key = self.keys[index]
        clip_name, frame_name = osp.dirname(key), osp.basename(key)  # key example: simple/simple_wires_1/images/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        frame_num = int(self.frame_num_dict[clip_name])
        if start_frame_idx > frame_num - self.num_frame * interval:
            start_frame_idx = random.randint(0, frame_num - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f{clip_name}/{neighbor:06d}
                img_gt_path = f{clip_name}/{neighbor:06d}
            else:
                img_lq_path = self.lq_root / clip_name / f{neighbor:06d}.png
                img_gt_path = self.gt_root / clip_name / f{neighbor:06d}.png

            # get LQ
            #print(img_lq_path: , img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, lq)
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, gt)
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt[use_hflip], self.opt[use_rot])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {lq: img_lqs, gt: img_gts, key: key}

    def __len__(self):
        return len(self.keys)
