# Handwritten Single Character Segmentation Model
A deep learning model that produces single-character-level bounding boxes given an input image of a handwritten Greek word. 

<div align="center">
  <img width="500" height="298" alt="image" src="https://github.com/user-attachments/assets/36a3e21e-14b6-4537-9203-b92cb412819a" />
  <img width="500" height="298" alt="image" src="https://github.com/user-attachments/assets/d9921d1d-0970-4665-ad9f-69a711cf8f4a" />
</div>
<div align="center">
  <img width="150" height="100" alt="image" src="https://github.com/user-attachments/assets/df82ecb2-e6bc-4cc8-8a23-8041774058d2" />
  <img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/3b93efd1-6689-465d-8d7d-8ef4e1d1658b" />
  <img width="140" height="100" alt="image" src="https://github.com/user-attachments/assets/7c7f9284-8852-4ae2-8bd1-8bc4188f5154" />
</div>

# Dataset
Manually extracted single-characters from each word of the ICDAR 2012 Writer Identification Contest. The data for each character was stored in a txt with polygons formatted as follows:
`word.tif` -> `x1 y1 x2 y2 ... xn yn`

The full dataset will become public after the ICDAR 2026 publication.

# Training
Three different models were fine tuned on our data for 40 epochs; Mask R-CNN and Cascade Mask R-CNN, and HTC.

<!-- Epoch 2 Table -->
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Mask R-CNN</th>
      <th>Cascade Mask R-CNN</th>
      <th>HTC</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>bbox_mAP</strong></td><td>0.7407</td><td><strong>0.7674</strong></td><td>0.7629</td></tr>
    <tr><td>bbox_mAP_50</td><td>0.9839</td><td><strong>0.9844</strong></td><td>0.9835</td></tr>
    <tr><td>bbox_mAP_75</td><td>0.8471</td><td><strong>0.8569</strong></td><td>0.8524</td></tr>
    <tr><td>bbox_mAP_s</td><td>0.7401</td><td>0.7728</td><td><strong>0.7770</strong></td></tr>
    <tr><td>bbox_mAP_m</td><td>0.7446</td><td><strong>0.7678</strong></td><td>0.7606</td></tr>
    <tr><td>bbox_mAP_l</td><td><strong>0.8010</strong></td><td>0.6020</td><td>0.6525</td></tr>
    <tr><td><strong>segm_mAP</strong></td><td>0.7773</td><td>0.7789</td><td><strong>0.7895</strong></td></tr>
    <tr><td>segm_mAP_50</td><td>0.9845</td><td><strong>0.9850</strong></td><td>0.9830</td></tr>
    <tr><td>segm_mAP_75</td><td>0.8618</td><td>0.8611</td><td><strong>0.8704</strong></td></tr>
    <tr><td>segm_mAP_s</td><td>0.7718</td><td>0.7764</td><td><strong>0.7862</strong></td></tr>
    <tr><td>segm_mAP_m</td><td>0.7799</td><td>0.7798</td><td><strong>0.7905</strong></td></tr>
    <tr><td>segm_mAP_l</td><td>0.5515</td><td>0.3535</td><td><strong>0.6010</strong></td></tr>
  </tbody>
</table>




