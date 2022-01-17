from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import os
import torch

import numpy as np
from TransPose.lib.core.inference import get_final_preds
from TransPose.lib.utils import transforms, vis
import cv2

from TransPose.visualize import inspect_atten_map_by_locations
from TransPose.visualize import inspect_atten_map_by_locations_hoda

from TransPose.visualize import update_config, add_path

lib_path = osp.join('lib')
add_path(lib_path)

# import dataset as dataset
import TransPose.lib.dataset as dataset
# import TransPose.lib.models as models
from TransPose.lib.models import transpose_h
from TransPose.lib.models import transpose_r
# from config import cfg
# import models
from TransPose.lib.config import cfg
import os
import torchvision.transforms as T

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# file_name = r'F:\Projects\Transpose\TransPose\experiments\stanford\transpose_h\TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml'
file_name = r'F:\Projects\Transpose\TransPose\experiments\coco\transpose_h\TP_H_w48_256x192_stage3_1_4_d96_h192_relu_enc6_mh1.yaml' # choose a yaml file
f = open(file_name, 'r')
update_config(cfg, file_name)

model_name = 'T-H-A6'
assert model_name in ['T-R', 'T-H','T-H-L','T-R-A4', 'T-H-A6', 'T-H-A5', 'T-H-A4' ,'T-R-A4-DirectAttention']

normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
if 'coco' in cfg.DATASET.DATASET:
    dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            T.Compose([
                T.ToTensor(),
                normalize,
            ])
        )
else:
    dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg,
            T.Compose([
                T.ToTensor(),
                normalize,
            ])
        )

device = torch.device('cuda')
# model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
#     cfg, is_train=True
# )
if cfg.MODEL.NAME == 'transpose_h':
    model = transpose_h.get_pose_net(cfg, is_train=False)
else:
    model = transpose_r.get_pose_net(cfg, is_train=False)

if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
else:
    raise ValueError("please choose one ckpt in cfg.TEST.MODEL_FILE")

model.to(device)
print("model params:{:.3f}M".format(sum([p.numel() for p in model.parameters()])/1000**2))

with torch.no_grad():
    model.eval()
    tmp = []
    tmp2 = []
    idx = 22
    img = dataset[idx][0]

    inputs = torch.cat([img.to(device)]).unsqueeze(0)
    outputs, _ = model(inputs)
    if isinstance(outputs, list):
        output = outputs[-1]
    else:
        output = outputs

    if cfg.TEST.FLIP_TEST:
        input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda()
        outputs_flipped, _ = model(input_flipped)

        if isinstance(outputs_flipped, list):
            output_flipped = outputs_flipped[-1]
        else:
            output_flipped = outputs_flipped

        output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
                                              dataset.flip_pairs)
        output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

        output = (output + output_flipped) * 0.5

    preds, maxvals = get_final_preds(
        cfg, output.clone().cpu().numpy(), None, None, transform_back=False)

# from heatmap_coord to original_image_coord
query_locations = np.array([p * 4 + 0.5 for p in preds[0]]) # for 64 * 48 => 256 * 192
print(query_locations)

inspect_atten_map_by_locations_hoda(img, model, query_locations,
                               model_name="transpose_h",
                               mode='dependency',
                               save_img=True,
                               threshold=0.0,
                               idx=idx)