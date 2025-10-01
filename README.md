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

  <!-- Epoch 40 Table -->
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
      <tr><td><strong>bbox_mAP</strong></td><td>0.6545</td><td><strong>0.678</strong></td><td>0.6676</td></tr>
      <tr><td>bbox_mAP_50</td><td>0.9491</td><td>0.951</td><td><strong>0.9586</strong></td></tr>
      <tr><td>bbox_mAP_75</td><td>0.6816</td><td><strong>0.712</strong></td><td>0.6986</td></tr>
      <tr><td>bbox_mAP_s</td><td>0.6759</td><td><strong>0.699</strong></td><td>0.689</td></tr>
      <tr><td>bbox_mAP_m</td><td>0.6481</td><td><strong>0.672</strong></td><td>0.6598</td></tr>
      <tr><td>bbox_mAP_l</td><td>0.6974</td><td>0.783</td><td><strong>0.7801</strong></td></tr>
      <tr><td><strong>segm_mAP</strong></td><td>0.6588</td><td><strong>0.675</strong></td><td>0.6709</td></tr>
      <tr><td>segm_mAP_50</td><td>0.9621</td><td>0.963</td><td><strong>0.9622</strong></td></tr>
      <tr><td>segm_mAP_75</td><td>0.6907</td><td><strong>0.726</strong></td><td>0.7133</td></tr>
      <tr><td>segm_mAP_s</td><td>0.6929</td><td><strong>0.711</strong></td><td>0.7039</td></tr>
      <tr><td>segm_mAP_m</td><td>0.6467</td><td><strong>0.661</strong></td><td>0.6571</td></tr>
      <tr><td>segm_mAP_l</td><td>0.6146</td><td>0.644</td><td><strong>0.6661</strong></td></tr>
    </tbody>
  </table>
</div>

