# ----------------------------------------------------------------------------#

import os
from mmcv import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import torch

# ----------------------------------------------------------------------------#

config_file = r"C:\Users\anton\Desktop\パイソン\checkpoints\mask-rcnn_r50_fpn_1x_coco.py"
checkpoint_file = r"C:\Users\anton\Desktop\パイソン\checkpoints\mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"

cfg = Config.fromfile(config_file)

if hasattr(cfg.model, 'data_preprocessor'):
    delattr(cfg.model, 'data_preprocessor')

cfg.classes = ('character',)

# Image normalization.
cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Training pipeline.
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# Test/Validation pipeline.
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Data configuration (CPU compatible).
cfg.data = dict(
    samples_per_gpu=1,  # Batch size per GPU.
    workers_per_gpu=0,  # For CPU training, keep it as 0.
    train=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\anton\Desktop\writers_100_126\train_annotations.json",
        img_prefix=r"C:\Users\anton\Desktop\writers_100_126\WordImages",
        classes=cfg.classes,
        pipeline=cfg.train_pipeline
    ),
    val=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\anton\Desktop\writers_100_126\val_annotations.json",
        img_prefix=r"C:\Users\anton\Desktop\writers_100_126\WordImages",
        classes=cfg.classes,
        pipeline=cfg.test_pipeline
    ),
    test=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\anton\Desktop\writers_100_126\val_annotations.json",
        img_prefix=r"C:\Users\anton\Desktop\writers_100_126\WordImages",
        classes=cfg.classes,
        pipeline=cfg.test_pipeline
    )
)

# Model: single-class Mask R-CNN.
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Work directory.
cfg.work_dir = './work_dirs/character_segmentation'

# CPU settings.
cfg.gpu_ids = [0]  # For CPU training, keep it as [0].
cfg.device = 'cpu'

cfg.resume_from = None
cfg.auto_resume = False

cfg.workflow = [('train', 1)]

# Optimizer.
cfg.optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
cfg.optimizer_config = dict(grad_clip=None)

# Learning rate schedule.
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1]
)

cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)
cfg.log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
cfg.evaluation = dict(interval=1, metric=['bbox', 'segm'])
cfg.checkpoint_config = dict(interval=1)

cfg.seed = 0
set_random_seed(0, deterministic=False)

# Building the dataset & model.
datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.init_weights()

# Loading pretrained weights.
if checkpoint_file and os.path.exists(checkpoint_file):
    model.CLASSES = cfg.classes
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} parameters from pretrained weights")
    else:
        print("No state_dict found in checkpoint")
else:
    print("No checkpoint file found, training from scratch")

# ----------------------------------------------------------------------------#

# Training!
train_detector(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=True,
    meta=dict()
)
