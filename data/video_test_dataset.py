import glob
from matplotlib.pyplot import axis
import torch
import numpy as np
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir, FileClient, list_of_groups
from basicsr.utils.img_util import img2tensor
from basicsr.data.transforms import mod_crop
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VideoWithEventsTestDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'folder': []}
        self.scale = opt['scale']

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.name = opt['name']
        if self.io_backend_opt['type'] == 'hdf5':
            self.io_backend_opt['h5_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['LR', 'HR']
        else:
            raise ValueError(f"We don't realize {self.io_backend_opt['type']} backend")

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoWithEventsTestDataset - {opt["name"]}')

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split(' ')[0])
                    clips_num.append(line.split(' ')[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip(clips, clips_num):
            self.io_backend_opt['h5_clip'] = clip
            self.file_client = FileClient(self.io_backend_opt['type'], **self.io_backend_opt)

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num))))

            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.join('vid4', osp.splitext(clip)[0]))
            self.data_info['folder'].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        voxel_f = event_lq[:len(event_lq) // 2]
        voxel_b = event_lq[len(event_lq) // 2:]
        return {
            'lq': img_lq,
            'gt': img_gt,
            'voxels_f': voxel_f,
            'voxels_b': voxel_b,
            'folder': folder,
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt[cache_data]
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {lq_path: [], gt_path: [], folder: [], idx: [], border: []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        assert self.io_backend_opt[type] != lmdb, No need to use lmdb during validation/test.

        logger = get_root_logger()
        logger.info(fGenerate data info for VideoTestDataset - {opt[name]})
        self.imgs_lq, self.imgs_gt = {}, {}
        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                subfolders = [line.split( )[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, *)))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, *)))

        if opt[name].lower() in [vid4, reds4, redsofficial]:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (fDifferent number of images in lq ({max_idx})
                                                      f and gt folders ({len(img_paths_gt)}))

                self.data_info[lq_path].extend(img_paths_lq)
                self.data_info[gt_path].extend(img_paths_gt)
                self.data_info[folder].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info[idx].append(f{i}/{max_idx})
                border_l = [0] * max_idx
                for i in range(self.opt[num_frame] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info[border].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(fCache {subfolder_name} for VideoTestDataset...)
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(fNon-supported video test dataset: {type(opt[name])})

    def __getitem__(self, index):
        folder = self.data_info[folder][index]
        idx, max_idx = self.data_info[idx][index].split(/)
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info[border][index]
        lq_path = self.data_info[lq_path][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt[num_frame], padding=self.opt[padding])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            lq: imgs_lq,  # (t, c, h, w)
            gt: img_gt,  # (c, h, w)
            folder: folder,  # folder name
            idx: self.data_info[idx][index],  # e.g., 0/99
            border: border,  # 1 for border, 0 for non-border
            lq_path: lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info[gt_path])


#########################################################
@DATASET_REGISTRY.register()
class Vid4WithEventsEDVRTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.is_bidirectional = opt[is_bidirectional]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for Vid4WithEventsEDVRTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.join(vid4, osp.splitext(clip)[0]))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)




@DATASET_REGISTRY.register()
class VideoTestVimeo90KDataset(data.Dataset):
    Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt[cache_data]
        if self.cache_data:
            raise NotImplementedError(cache_data in Vimeo90K-Test dataset is not implemented.)
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {lq_path: [], gt_path: [], folder: [], idx: [], border: []}
        neighbor_list = [i + (9 - opt[num_frame]) // 2 for i in range(opt[num_frame])]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        assert self.io_backend_opt[type] != lmdb, No need to use lmdb during validation/test.

        logger = get_root_logger()
        logger.info(fGenerate data info for VideoTestDataset - {opt[name]})
        with open(opt[meta_info_file], r) as fin:
            subfolders = [line.split( )[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, im4.png)
            self.data_info[gt_path].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, fim{i}.png) for i in neighbor_list]
            self.data_info[lq_path].append(lq_paths)
            self.data_info[folder].append(vimeo90k)
            self.data_info[idx].append(f{idx}/{len(subfolders)})
            self.data_info[border].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info[lq_path][index]
        gt_path = self.data_info[gt_path][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)

        return {
            lq: imgs_lq,  # (t, c, h, w)
            gt: img_gt,  # (c, h, w)
            folder: self.data_info[folder][index],  # folder name
            idx: self.data_info[idx][index],  # e.g., 0/843
            border: self.data_info[border][index],  # 0 for non-border
            lq_path: lq_path[self.opt[num_frame] // 2]  # center frame
        }

    def __len__(self):
        return len(self.data_info[gt_path])


@DATASET_REGISTRY.register()
class VideoTestDUFDataset(VideoTestDataset):
     Video test dataset for DUF dataset.

    Args:
        opt (dict): Config for train dataset. Most of keys are the same as VideoTestDataset.
            It has the following extra keys:
        use_duf_downsampling (bool): Whether to use duf downsampling to generate low-resolution frames.
        scale (bool): Scale, which will be added automatically.
    

    def __getitem__(self, index):
        folder = self.data_info[folder][index]
        idx, max_idx = self.data_info[idx][index].split(/)
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info[border][index]
        lq_path = self.data_info[lq_path][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt[num_frame], padding=self.opt[padding])

        if self.cache_data:
            if self.opt[use_duf_downsampling]:
                # read imgs_gt to generate low-resolution frames
                imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt[scale])
            else:
                imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            if self.opt[use_duf_downsampling]:
                img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
                # read imgs_gt to generate low-resolution frames
                imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt[scale])
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt[scale])
            else:
                img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt[scale])
            img_gt.squeeze_(0)

        return {
            lq: imgs_lq,  # (t, c, h, w)
            gt: img_gt,  # (c, h, w)
            folder: folder,  # folder name
            idx: self.data_info[idx][index],  # e.g., 0/99
            border: border,  # 1 for border, 0 for non-border
            lq_path: lq_path  # center frame
        }


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        opt (dict): Same as VideoTestDataset. Unused opt:
        padding (str): Padding mode.

    

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info[folder])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError(Without cache_data is not implemented.)

        return {
            lq: imgs_lq,
            gt: imgs_gt,
            folder: folder,
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class CEDOnlyFramesTestDataset(data.Dataset):
    def __init__(self, opt):
        super(CEDOnlyFramesTestDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]

        logger = get_root_logger()
        logger.info(fGenerate data info for CEDOnlyFramesTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            # split long clip
            split_num = list_of_groups(list(range(int(num))), 100)
            for idx in range(len(split_num)):
                img_lqs, img_gts = self.file_client.get(split_num[idx])
                # mod_crop gt image for scale
                img_gts = [mod_crop(img, self.scale) for img in img_gts]
                self.imgs_lq[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_lqs), dim=0)
                self.imgs_gt[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_gts), dim=0)
                self.folders.append(osp.join(clip, f{idx:06d}))
                self.lq_paths.append(osp.join(clip.split(_)[0], osp.splitext(clip)[0]))
                self.data_info[folder].extend([osp.join(clip, f{idx:06d})] * len(split_num[idx]))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]

        return {
            lq: img_lq,
            gt: img_gt,
            folder: folder,
            lq_path: lq_path
        }

    def __len__(self):
        return len(self.folders)

#########################################################
@DATASET_REGISTRY.register()
class CEDWithEventsTestDataset(data.Dataset):
    def __init__(self, opt):
        super(CEDWithEventsTestDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.is_bidirectional = opt[is_bidirectional]

        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for CEDWithEventsTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            # split long clip
            split_num = list_of_groups(list(range(int(num))), 80)
            for idx in range(len(split_num)):
                img_lqs, img_gts, event_lqs = self.file_client.get(split_num[idx])
                # mod_crop gt image for scale
                img_gts = [mod_crop(img, self.scale) for img in img_gts]
                self.imgs_lq[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_lqs), dim=0)
                self.imgs_gt[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_gts), dim=0)
                self.event_lqs[osp.join(clip, f{idx:06d})] = torch.from_numpy(np.stack(event_lqs, axis=0))
                self.folders.append(osp.join(clip, f{idx:06d}))
                self.lq_paths.append(osp.join(clip.split(_)[0], osp.splitext(clip)[0]))
                self.data_info[folder].extend([osp.join(clip, f{idx:06d})] * len(split_num[idx]))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]
        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class Vid4onlyFramesTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]

        logger = get_root_logger()
        logger.info(fGenerate data info for Vid4onlyFramesTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.folders.append(clip)
            self.lq_paths.append(osp.join(vid4, osp.splitext(clip)[0]))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]

        return {
            lq: img_lq,
            gt: img_gt,
            folder: folder,
            lq_path: lq_path
        }

    def __len__(self):
        return len(self.folders)

#########################################################
@DATASET_REGISTRY.register()
class Vid4WithEventsTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.is_bidirectional = opt[is_bidirectional]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for Vid4WithEventsTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.join(vid4, osp.splitext(clip)[0]))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)



@DATASET_REGISTRY.register()
class Vimeo90kOnlyCenterFrameTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        self.center_frame_only = opt[center_frame_only]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[center_frame_only] = self.center_frame_only

        logger = get_root_logger()
        logger.info(fGenerate data info for Vimeo90kOnlyCenterFrameTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(7)
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.folders.append(clip)
            self.lq_paths.append(osp.splitext(clip)[0].replace(_, /))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder] # (t, c, h, w)
        img_gt = self.imgs_gt[folder] # (c, scale * h, scale * w)
        img_gt.squeeze_(0)

        return {
            lq: img_lq,
            gt: img_gt,
            folder: folder,
            lq_path: lq_path
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class Vimeo90kOnlyCenterFrameWithEventTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.flip_seq = opt[flip_seq]
        self.is_bidirectional = opt[is_bidirectional]
        self.center_frame_only = opt[center_frame_only]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[center_frame_only] = self.center_frame_only
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for Vimeo90kOnlyCenterFrameWithEventTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    if self.flip_seq:
                        clips_num.append(13)
                    else:
                        clips_num.append(7)
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.splitext(clip)[0].replace(_, /))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder] # (t, c, h, w)
        img_gt = self.imgs_gt[folder] # (c, scale * h, scale * w)
        img_gt.squeeze_(0)
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class REDSOnlyFramesTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]

        logger = get_root_logger()
        logger.info(fGenerate data info for REDSOnlyFramesTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts = self.file_client.get(list(range(int(num))))
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.folders.append(clip)
            self.lq_paths.append(osp.join(reds, osp.splitext(clip)[0]))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]

        return {
            lq: img_lq,
            gt: img_gt,
            folder: folder,
            lq_path: lq_path
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class REDSWithEventsTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.is_bidirectional = opt[is_bidirectional]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for REDSWithEventsTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            img_lqs, img_gts, event_lqs = self.file_client.get(list(range(int(num)))) #这里就是搞到了img_lqs, img_gts, event_lqs
            # mod_crop gt image for scale
            img_gts = [mod_crop(img, self.scale) for img in img_gts]
            self.imgs_lq[clip] = torch.stack(img2tensor(img_lqs), dim=0)
            self.imgs_gt[clip] = torch.stack(img2tensor(img_gts), dim=0)
            self.event_lqs[clip] = torch.from_numpy(np.stack(event_lqs, axis=0))
            self.folders.append(clip)
            self.lq_paths.append(osp.join(vid4, osp.splitext(clip)[0]))
            self.data_info[folder].extend([clip] * int(num))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.folders[index]


        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class REDSOnlyFramesSplitTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]

        logger = get_root_logger()
        logger.info(fGenerate data info for REDSOnlyFramesSplitTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            # split reds4 clip to 4 segments to sava gpu memory when inference
            split_num = list_of_groups(list(range(int(num))), 100)
            for idx in range(len(split_num)):
                img_lqs, img_gts = self.file_client.get(split_num[idx])
                # mod_crop gt image for scale
                img_gts = [mod_crop(img, self.scale) for img in img_gts]
                self.imgs_lq[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_lqs), dim=0)
                self.imgs_gt[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_gts), dim=0)
                self.folders.append(osp.join(clip, f{idx:06d}))
                self.lq_paths.append(osp.join(reds, osp.splitext(clip)[0]))
                self.data_info[folder].extend([osp.join(clip, f{idx:06d})] * len(split_num[idx]))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]

        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]

        return {
            lq: img_lq,
            gt: img_gt,
            folder: folder,
            lq_path: lq_path
        }

    def __len__(self):
        return len(self.folders)

@DATASET_REGISTRY.register()
class REDSWithEventsSplitTestDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt[dataroot_gt], opt[dataroot_lq]
        self.data_info = {folder: []}
        self.scale = opt[scale]
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt[io_backend]
        self.name = opt[name]
        self.is_event = opt[is_event]
        self.is_bidirectional = opt[is_bidirectional]
        if self.io_backend_opt[type] == hdf5:
            self.io_backend_opt[h5_paths] = [self.lq_root, self.gt_root]
            self.io_backend_opt[client_keys] = [LR, HR]
            self.io_backend_opt[name] = self.name
            self.io_backend_opt[is_event] = self.is_event
            self.io_backend_opt[is_bidirectional] = self.is_bidirectional
        else:
            raise ValueError(fWe dont realize {self.io_backend_opt[type]} backend)

        logger = get_root_logger()
        logger.info(fGenerate data info for REDSWithEventsSplitTestDataset - {opt[name]})

        if meta_info_file in opt:
            with open(opt[meta_info_file], r) as fin:
                clips = []
                clips_num = []
                for line in fin:
                    clips.append(line.split( )[0])
                    clips_num.append(line.split( )[1])
        else:
            raise NotImplementedError

        self.imgs_lq, self.imgs_gt, self.event_lqs = {}, {}, {}
        self.folders = []
        self.lq_paths = []
        for clip, num in zip (clips, clips_num):
            self.io_backend_opt[h5_clip] = clip
            self.file_client = FileClient(
                self.io_backend_opt[type],
                **self.io_backend_opt
                )

            # split reds4 clip to 4 segments to sava gpu memory when inference
            split_num = list_of_groups(list(range(int(num))), 100)
            for idx in range(len(split_num)):
                img_lqs, img_gts, event_lqs = self.file_client.get(split_num[idx])
                # mod_crop gt image for scale
                img_gts = [mod_crop(img, self.scale) for img in img_gts]
                self.imgs_lq[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_lqs), dim=0)
                self.imgs_gt[osp.join(clip, f{idx:06d})] = torch.stack(img2tensor(img_gts), dim=0)
                self.event_lqs[osp.join(clip, f{idx:06d})] = torch.from_numpy(np.stack(event_lqs, axis=0))
                self.folders.append(osp.join(clip, f{idx:06d}))
                self.lq_paths.append(osp.join(reds, osp.splitext(clip)[0]))
                self.data_info[folder].extend([osp.join(clip, f{idx:06d})] * len(split_num[idx]))

    def __getitem__(self, index):
        folder = self.folders[index]
        lq_path = self.lq_paths[index]


        img_lq = self.imgs_lq[folder]
        img_gt = self.imgs_gt[folder]
        event_lq = self.event_lqs[folder]

        if self.is_bidirectional:
            voxel_f = event_lq[:len(event_lq) // 2]
            voxel_b = event_lq[len(event_lq) // 2:]
            return {
                lq: img_lq,
                gt: img_gt,
                voxels_f: voxel_f,
                voxels_b: voxel_b,
                folder: folder,
                lq_path: lq_path
            }
        else:
            return {
                lq: img_lq,
                gt: img_gt,
                event_lq: event_lq,
                folder: folder,
                lq_path: lq_path
            }

    def __len__(self):
        return len(self.folders)
    
