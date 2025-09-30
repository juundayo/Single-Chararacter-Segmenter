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
    with open(r'C:\Users\bgat\TranscriptMapping\util\annotations_chars_coco.json',
            'r') as f:
        data = json.load(f)

    # Creating a mapping from writer number to image IDs.
    writer_to_images = {}
    for img in data['images']:
        # Extracting the writer number from the filename.
        filename = img['file_name']
        
        if '_' in filename:
            # Pattern: "027_something".
            parts = filename.split('_')
            for part in parts:
                if part.isdigit() and len(part) == 3:  # Always a 3-digit number.
                    writer_num = int(part)
                    if writer_num not in writer_to_images:
                        writer_to_images[writer_num] = []
                    writer_to_images[writer_num].append(img['id'])
                    break

    print(f"\nFound writers: {sorted(writer_to_images.keys())}")
    print(f"Number of writers: {len(writer_to_images)}")

    # Getting all writers in range 27-126.
    all_writers = [w for w in range(27, 127) if w in writer_to_images]
    print(f"Writers in range 27-126: {sorted(all_writers)}")
    print(f"Number of writers in range: {len(all_writers)}")

    if len(all_writers) == 0:
        print("ERROR: No writers found in the range 27-126!")
        print("Available writers:", sorted(writer_to_images.keys()))
        exit(1)

    # Training: first 80% of writers (27-106).
    train_writers = [w for w in all_writers if 27 <= w <= 106]
    
    # Remaining writers for val/test.
    remaining_writers = [w for w in all_writers if w > 106]
    
    # Splitting remaining writers 50/50 for val and test.
    if len(remaining_writers) > 1:
        val_writers, test_writers = train_test_split(remaining_writers, test_size=0.5, random_state=1)
    else:
        # If only one writer remains, put it in val (odd number count).
        val_writers = remaining_writers
        test_writers = []

    print(f"Train writers: {sorted(train_writers)} ({len(train_writers)} writers)")
    print(f"Val writers: {sorted(val_writers)} ({len(val_writers)} writers)") 
    print(f"Test writers: {sorted(test_writers)} ({len(test_writers)} writers)")

    # Getting image IDs for each split.
    train_ids = []
    for writer in train_writers:
        train_ids.extend(writer_to_images[writer])
    
    val_ids = []
    for writer in val_writers:
        val_ids.extend(writer_to_images[writer])
    
    test_ids = []
    for writer in test_writers:
        test_ids.extend(writer_to_images[writer])

    print(f"\nTrain images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")

    if len(train_ids) == 0 or len(val_ids) == 0:
        print("ERROR: One of the splits is empty!")
        exit(1)

    create_split(data, train_ids, r'C:\Users\bgat\TranscriptMapping\util\train_annotations.json')
    create_split(data, val_ids, r'C:\Users\bgat\TranscriptMapping\util\val_annotations.json')
    if test_ids:
        create_split(data, test_ids, r'C:\Users\bgat\TranscriptMapping\util\test_annotations.json')
