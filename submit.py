# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math
from copy import deepcopy
import json

import os
import argparse

import numpy as np
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser
from PIL import Image

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
import datasets.transforms as T
from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)

class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        self.normalize = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.scales = [640]
        self.img_size = [640, 640]
        self.transform = T.Compose([
            T.RandomResize(self.scales, max_size=640),
            T.HSV(),
            self.normalize,
        ])

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.mean = [0.5, 0.5, 0.5]
        # self.std = [0.5, 0.5, 0.5]


    def load_img_from_filecdz(self, f_path):
        im = Image.open(os.path.join(self.mot_path, f_path))
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []

        # for line in self.det_db[f_path[:-4] + '.txt']:
        #     l, t, w, h, s = list(map(float, line.split(',')))
        #     proposals.append([(l + w / 2) / im_w,
        #                       (t + h / 2) / im_h,
        #                       w / im_w,
        #                       h / im_h,
        #                       s])
        return im, torch.as_tensor(proposals).reshape(-1, 5), cur_img

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_h=math.ceil(target_h/32)*32
        target_w = int(self.seq_w * scale)
        target_w=math.ceil(target_w/32)*32
        img = cv2.resize(img, (target_w, target_h))
        #img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img=F.to_tensor(img)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def init_imgcdz(self, img, proposals):
        ori_img=img.copy()
        im=letterbox(img,self.img_size,stride=32,auto=True)[0]
        im=im.transpose((2,0,1))
        im=np.ascontiguousarray(im)
        im = torch.from_numpy(im)
        im = im.float() # uint8 to fp16/32
        im /= 255
        im=im[None]
        #images, targets = self.transform(img, target)
        return im, ori_img,proposals

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # img, proposals, cur_img = self.load_img_from_filecdz(self.img_list[index])
        # ori_img = cur_img.copy()
        # img, _ = self.init_imgcdz(img)
        # img=img.unsqueeze(0)
        #
        # return img, ori_img, proposals

        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' or 'png' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = {"small": None, "medium": None, "large": None}
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances['small'] is not None:
                for k, v in track_instances.items():
                    track_instances[k].remove('boxes')
                    track_instances[k].remove('labels')

            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_imagecdz(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances = track_instances
            dt_instances = Instances.cat([dt_instances['small'], dt_instances['medium'], dt_instances['large']])
            # dt_instances = dt_instances['small']

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()
            # if i==0:
            #     last_id=identities[0]

            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # '''for MOT17 submit''' 
    sub_dir = 'swjtu/val'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for vid in vids:
        detr, _, _ = build_model(args)
        detr.track_embed_large.score_thr = args.update_score_threshold
        detr.track_embed_medium.score_thr = args.update_score_threshold
        detr.track_embed_small.score_thr = args.update_score_threshold

        detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
        checkpoint = torch.load(args.resume, map_location='cpu')
        detr = load_model(detr, args.resume)
        detr.eval()
        detr = detr.cuda()
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold)
