# ----------------------------------------------------------------------------#

from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import mmcv
import os
import torch

# ----------------------------------------------------------------------------#

# Load and modify the config
config_file = r"C:\Users\bgat\TranscriptMapping\checkpoints\mask_rcnn\mask_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

# Applying class modifications to match the training script.
cfg.classes = ('character',)
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Saving the modified config to a temporary file.
modified_config_file = r"C:\Users\bgat\TranscriptMapping\checkpoints\mask_rcnn\mask_rcnn_r50_fpn_1x_coco_modified.py"
cfg.dump(modified_config_file)

# Paths!
checkpoint_file = r"C:\Users\bgat\TranscriptMapping\src\work_dirs\character_segmentation_rcnn\epoch_40.pth"

# Model loading (with the modified file).
model = init_detector(modified_config_file, checkpoint_file, device="cuda:0")

# Running on a sample image.
img = r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages\035_4_L_01_W_06.tif"
result = inference_detector(model, img)

# Result print.
model.show_result(img, result, out_file="segmentation_output5.png")

# Temporary config file cleanup.
# os.remove(modified_config_file)
