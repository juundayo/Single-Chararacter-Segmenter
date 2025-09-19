# ----------------------------------------------------------------------------#

from mmdet.apis import init_detector, inference_detector
import mmcv

# ----------------------------------------------------------------------------#

# Paths!
config_file = "configs/???.py"
checkpoint_file = "C:\Users\bgat\TranscriptMapping\src\work_dirs\character_segmentation\latest.pth"

# Model loading.
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# Running on a sample image.
img = "C:\Users\bgat\TranscriptMapping\Dataset\WordImages\027_3_L_01_W_01.tif"
result = inference_detector(model, img)

# Result print.
model.show_result(img, result, out_file="result.png")
