# ----------------------------------------------------------------------------#

import gc
import os
from mmengine import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import torch
import traceback
import os

import faulthandler
import sys
logfile = open(r"C:\Users\bgat\TranscriptMapping\util\faulthandler.log", "w")
faulthandler.enable(file=logfile)

try:
    import signal
    faulthandler.register(signal.SIGSEGV, file=logfile, all_threads=True)
    faulthandler.register(signal.SIGABRT, file=logfile, all_threads=True)
except Exception:
    pass

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# ----------------------------------------------------------------------------#

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

gc.collect()
torch.cuda.empty_cache()

config_file = r"C:\Users\bgat\TranscriptMapping\checkpoints_rcnn\mask_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = r"C:\Users\bgat\TranscriptMapping\checkpoints_rcnn\mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"

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
    dict(type='Resize', img_scale=(900, 900), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.01),
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
            dict(type='RandomFlip', flip_ratio=0.01),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Data configuration (CPU compatible).
cfg.data = dict(
    samples_per_gpu=1,  # Batch size per GPU.
    workers_per_gpu=0,  # Keeping this as 0 for both GPU and CPU training.
    train=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\bgat\TranscriptMapping\util\train_annotations.json",
        img_prefix=r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages",
        classes=cfg.classes,
        pipeline=cfg.train_pipeline
    ),
    val=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\bgat\TranscriptMapping\util\val_annotations.json",
        img_prefix=r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages",
        classes=cfg.classes,
        pipeline=cfg.test_pipeline
    ),
    test=dict(
        type='CocoDataset',
        ann_file=r"C:\Users\bgat\TranscriptMapping\util\val_annotations.json",
        img_prefix=r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages",
        classes=cfg.classes,
        pipeline=cfg.test_pipeline
    )
)

# Model: single-class Mask R-CNN.
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Work directory.
cfg.work_dir = './work_dirs/character_segmentation_rcnn'

# CPU settings.
cfg.gpu_ids = [0]  # For CPU and GPU training, keep it as [0].
cfg.device = 'cuda'

cfg.resume_from = None
cfg.auto_resume = False

# Optimizer.
cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
cfg.optimizer_config = dict(grad_clip=None)

# Learning rate schedule.
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 32, 48]
)

cfg.runner = dict(type='EpochBasedRunner', max_epochs=75)
cfg.log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
cfg.evaluation = dict(interval=1, metric=['bbox', 'segm'])
cfg.checkpoint_config = dict(interval=1, by_epoch=True, save_optimizer=True)

cfg.seed = 1
set_random_seed(1, deterministic=False)

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

try:
    print("Training starts...")
    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())
    print("Training finished!")
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"CUDA Error: {e}")
        print("This is likely a memory issue or kernel error")
    print("Training crashed!")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    traceback.print_exc()
except Exception as e:
    print("Training crashed!")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    traceback.print_exc()
