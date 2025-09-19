# Single Character Segmentation Model
A deep learning model that produces single-character-level bounding boxes given an input image of a handwritten Greek word. 

# Data
Manually extracted single-characters from each word of the ICDAR 2012 Writer Identification Contest. The data for each character was stored in a txt with polygons formatted as follows:
`word.tif` -> `x1 y1 x2 y2 x3 y3 x4 y4`
The full dataset is public at:
https://www.dropbox.com/scl/fi/g6y9ptqvmtzl96ieykey8/Dataset.zip?rlkey=1fhc2xb4kpyfdndfffgl28zdo&st=0lmmxt3u&dl=0

# Training
5 different models were fine tuned on our own data:
- Mask R-CNN
- Cascade Mask R-CNN
- SOLO
- SOLO V2
- Point Rend
