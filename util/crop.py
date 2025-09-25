from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import mmcv
import os
import numpy as np
import cv2

# ----------------------------------------------------------------------------#

# Loading and modifying the config.
config_file = r"C:\Users\bgat\TranscriptMapping\checkpoints\mask_rcnn\mask_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

cfg.classes = ('character',)
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

modified_config_file = r"C:\Users\bgat\TranscriptMapping\checkpoints\mask_rcnn\mask_rcnn_r50_fpn_1x_coco_modified.py"
cfg.dump(modified_config_file)

checkpoint_file = r"C:\Users\bgat\TranscriptMapping\src\work_dirs\character_segmentation_rcnn\epoch_40.pth"

model = init_detector(modified_config_file, checkpoint_file, device="cuda:0")

# Input image.
img_path = r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages\035_4_L_01_W_07.tif"
result = inference_detector(model, img_path)
img = mmcv.imread(img_path)

# --------------------------------------------------

bbox_results, segm_results = result

# We use single class training.
bboxes = bbox_results[0]
masks = segm_results[0]

# Filtering by confidence score.
score_thr = 0.5
indices = np.where(bboxes[:, -1] > score_thr)[0]

char_crops = []
for i in indices:
    bbox = bboxes[i]
    mask = masks[i].astype(np.uint8)  # Binary mask.

    # Tight bounding box from mask.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        continue
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Image cropping.
    cropped = img[y_min:y_max+1, x_min:x_max+1].copy()

    # Applying the mask to the crop.
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
    cropped[mask_crop == 0] = 255  # White background.

    char_crops.append((x_min, cropped))

# Sorting left-to-right.
char_crops.sort(key=lambda x: x[0])

# Saving the results!
save_dir = "segmented_characters"
os.makedirs(save_dir, exist_ok=True)

for idx, (_, crop) in enumerate(char_crops):
    save_path = os.path.join(save_dir, f"char_{idx+1}.png")
    cv2.imwrite(save_path, crop)

print(f"Saved {len(char_crops)} characters to {save_dir}")
