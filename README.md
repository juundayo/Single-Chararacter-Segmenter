# Single Character Segmentation Model
A deep learning model that produces single-character-level bounding boxes given an input image of a handwritten Greek word. 

# Data
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
      <th>segm<br>mAP</th>
      <th>segm<br>mAP@50</th>
      <th>segm<br>mAP@75</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Mask R-CNN</strong></td>
      <td>0.6545</td>
      <td>0.9491</td>
      <td>0.6816</td>
      <td>0.6588</td>
      <td>0.9621</td>
      <td>0.6907</td>
    </tr>
    <tr>
      <td><strong>Cascade Mask R-CNN</strong></td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
      <td>X</td>
    </tr>
  </tbody>
</table>
