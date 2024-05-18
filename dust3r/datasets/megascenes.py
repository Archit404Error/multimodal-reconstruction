import os.path as osp
import os, sys
from os.path import join
import pickle
import json
import itertools
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
#from dust3r.utils.image import imread_cv2
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping
import PIL 
from PIL import Image # I forget which one is the correct one
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
Image.LOAD_TRUNCATED_IMAGES = True

#from torch.utils.data import Dataset

from .datautils.data_helpers import *
from .datautils.depth_helpers import *


def subsample_data(save_dir, pairs, dicts):
    required_paths = set()
    new_pairs = []
    for i in tqdm(range(1, len(pairs), 10_000)):
        cur_pair = pairs[i]
        new_pairs.append(cur_pair)
        required_paths.add(cur_pair[0])
        required_paths.add(cur_pair[1])

    new_dicts = {img_path: dicts[img_path] for img_path in dicts if img_path in required_paths}
    assert len(new_dicts) == len(required_paths)

    with open(f"{save_dir}/subsampled_pairs.pkl", "wb") as f:
        pickle.dump(new_pairs, f)
    with open(f"{save_dir}/subsampled_dicts.pkl", "wb") as f:
        pickle.dump(new_dicts, f)

    return new_pairs, new_dicts


class MegaScenes(BaseStereoViewDataset):
    def __init__(self, data_root, save_dir, subsampled=True, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # # load pairs 
        # with open(join(data_root, f'large_{self.split}_pairs.pkl'), 'rb') as f:
        #     pairs = pickle.load(f)

        # # load img info
        # # img path as keys; value is dict with extrinsics, intrinsics, 3d point cloud as keys
        # with open(join(data_root, f'large_{self.split}_images.pkl'), 'rb') as f:
        #     imgdicts = pickle.load(f)

        if subsampled and osp.exists(f"{save_dir}/subsampled_pairs.pkl"):
            pairs = pickle.load(open(f"{save_dir}/subsampled_pairs.pkl", "rb"))
            imgdicts = pickle.load(open(f"{save_dir}/subsampled_dicts.pkl", "rb"))
        else:
            pairs = []
            imgdicts = {}
            if self.split == 'train':
                indices = ['10']
                for i in tqdm(indices):
                    with open(join(data_root, f'grouped_pairs/large_pairs_{i}.pkl'), 'rb') as f:
                        pairs.extend(pickle.load(f))
                    with open(join(data_root, f'grouped_imgdicts/imgdict_{i}.pkl'), 'rb') as f:
                        imgdicts.update(pickle.load(f))

            if self.split == 'test':
                with open(join(data_root, f'grouped_pairs/large_pairs_38.pkl'), 'rb') as f:
                    pairs = pickle.load(f)
                with open(join(data_root, f'grouped_imgdicts/imgdict_38.pkl'), 'rb') as f:
                    imgdicts = pickle.load(f)

            if subsampled:
                if not osp.exists(f"{save_dir}/subsampled_pairs.pkl"):
                    pairs, imgdicts = subsample_data(save_dir, pairs, imgdicts)

        metadata_files = {}
        image_exifs = {}
        for img_path in tqdm(imgdicts):
            metadata_file_path = osp.abspath(osp.join(img_path, "../../../raw_metadata.json"))
            if not osp.exists(metadata_file_path):
                continue
            if metadata_file_path not in metadata_files:
                metadata = json.load(open(metadata_file_path))["elements"]
                title_exif_map = {}
                for page_dict in metadata:
                    for page_id, page_data in page_dict.items():
                        try:
                            metadata_string = "; ".join(list(map(lambda item_dict: f"{item_dict['name']}: {item_dict['value']}", page_data["imageinfo"][0]["commonmetadata"])))
                            title_exif_map[page_data["title"].split(":", 1)[1]] = metadata_string
                        except KeyError:
                            continue
                metadata_files[metadata_file_path] = title_exif_map

            if metadata_file_path in metadata_files:
                title_exif_map = metadata_files[metadata_file_path]
                try:
                    image_exifs[img_path] = title_exif_map[img_path.rsplit("/", 1)[1]]
                except KeyError:
                    image_exifs[img_path] = ""
            else:
                image_exifs[img_path] = ""

        self.data_root = data_root
        self.pairs = pairs
        self.imgdicts = imgdicts
        self.image_exifs = image_exifs


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        # step 0: setup resolution, rng
        if isinstance(idx, tuple):
            # ar_idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)

        # step 1: get the pair from idx
        pair = self.pairs[idx]
        img1path, img2path = pair
        img1 = Image.open(img1path)
        img2 = Image.open(img2path)
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        # step 2: get the image info
        img1_info = self.imgdicts[img1path]
        img2_info = self.imgdicts[img2path]
        extrinsics1 = img1_info['extrinsics'] # w2c, invert to c2w for view1, view2
        extrinsics2 = img2_info['extrinsics']
        intrinsics1 = img1_info['intrinsics']
        intrinsics2 = img2_info['intrinsics']

        pts3d1 = img1_info['pointxyz']
        pts3d2 = img2_info['pointxyz']

        sparsedepth1 = create_sparse_depth_map(pts3d1, intrinsics1, extrinsics1, (img1.size[1], img1.size[0]))
        sparsedepth2 = create_sparse_depth_map(pts3d2, intrinsics2, extrinsics2, (img2.size[1], img2.size[0]))

        # step 3: crop based on resolution and add captions
        img1, sparsedepth1, intrinsics1 = self._crop_resize_if_necessary(img1, sparsedepth1, intrinsics1, resolution, self._rng)
        img2, sparsedepth2, intrinsics2 = self._crop_resize_if_necessary(img2, sparsedepth2, intrinsics2, resolution, self._rng)

        caption1 = self.image_exifs[img1path] if img1path in self.image_exifs else ""
        caption2 = self.image_exifs[img2path] if img2path in self.image_exifs else ""

        # step 4: format based on requirements 
        
        view1 = dict(img=img1, depthmap=sparsedepth1.astype(np.float32), camera_intrinsics=intrinsics1.astype(np.float32), camera_pose=np.linalg.inv(extrinsics1).astype(np.float32), instance=img1path, caption=caption1)
        view2 = dict(img=img2, depthmap=sparsedepth2.astype(np.float32), camera_intrinsics=intrinsics2.astype(np.float32), camera_pose=np.linalg.inv(extrinsics2).astype(np.float32), instance=img2path, caption=caption2)
        views = [view1, view2]

        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {idx}"
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {idx}'
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {idx}'
            #view['idx'] = (idx, ar_idx, v)

            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            #import ipdb; ipdb.set_trace()
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {idx}"
            K = view['camera_intrinsics']

        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        return views


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None

def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]
