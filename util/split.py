# ----------------------------------------------------------------------------#

import json
import random
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------#

# Creating the train/val splits.
def create_split(original_data, split_ids, output_path):
    split_images = [img for img in original_data['images'] if img['id'] in split_ids]
    split_annotations = [ann for ann in original_data['annotations'] if ann['image_id'] in split_ids]

    split_data = {
        'info': original_data['info'],
        'licenses': original_data['licenses'],
        'categories': original_data['categories'],
        'images': split_images,
        'annotations': split_annotations
    }

    with open(output_path, 'w') as f:
        json.dump(split_data, f)

# ----------------------------------------------------------------------------#

if __name__ == '__main__':
    # Loading the annotations.
    with open(r'C:\Users\anton\Desktop\writers_100_126\annotations_chars_coco.json',
            'r') as f:
        data = json.load(f)

    # Splitting the image IDs.
    image_ids = [img['id'] for img in data['images']]
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=1)

    create_split(data, train_ids, r'C:\Users\anton\Desktop\writers_100_126\train_annotations.json')
    create_split(data, val_ids, r'C:\Users\anton\Desktop\writers_100_126\val_annotations.json')
