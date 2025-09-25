# Single Character Segmentation Model
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
`word.tif` -> `x1 y1 x2 y2 x3 y3 x4 y4`
The full dataset is public at:
[https://www.dropbox.com/scl/fi/g6y9ptqvmtzl96ieykey8/Dataset.zip?rlkey=1fhc2xb4kpyfdndfffgl28zdo&st=0lmmxt3u&dl=0](https://www.dropbox.com/scl/fi/lscf8p00bmnbqhvmun2sd/Dataset.zip?rlkey=m6uj2d2fi8dte6h6lc5v600cj&st=v4s0bbyx&dl=0)

# Training
Two different models were fine tuned on our data for 40 epochs; Mask R-CNN and Cascade R-CNN.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>bbox<br>mAP</th>
      <th>bbox<br>mAP@50</th>
      <th>bbox<br>mAP@75</th>
      <th>bbox<br>mAP_s</th>
      <th>bbox<br>mAP_m</th>
      <th>bbox<br>mAP_l</th>
      <th>segm<br>mAP</th>
      <th>segm<br>mAP@50</th>
      <th>segm<br>mAP@75</th>
      <th>segm<br>mAP_s</th>
      <th>segm<br>mAP_m</th>
      <th>segm<br>mAP_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Mask R-CNN</strong></td>
      <td>0.6545</td>
      <td>0.9491</td>
      <td>0.6816</td>
      <td>0.6759</td>
      <td>0.6481</td>
      <td>0.6974</td>
      <td>0.6588</td>
      <td>0.9621</td>
      <td>0.6907</td>
      <td>0.6929</td>
      <td>0.6467</td>
      <td>0.6146</td>
    </tr>
    <tr>
      <td><strong>Cascade Mask R-CNN</strong></td>
      <td>0.6529</td>
      <td>0.9400</td>
      <td>0.6646</td>
      <td>0.6663</td>
      <td>0.6494</td>
      <td>0.7408</td>
      <td>0.6583</td>
      <td>0.9410</td>
      <td>0.6879</td>
      <td>0.6881</td>
      <td>0.6491</td>
      <td>0.6410</td>
    </tr>
  </tbody>
</table>

