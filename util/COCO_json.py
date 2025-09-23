# ----------------------------------------------------------------------------#

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
import glob

# ----------------------------------------------------------------------------#

def parse_annotation_file(annotation_path: str) -> List[List[float]]:
    """
    Parses the annotation file with polygon coordinates.
    Each line contains a variable number of coordinates representing one polygon.
    Format: x1 y1 x2 y2 x3 y3 ... xn yn (any even number of coordinates)
    """
    polygons = []
    
    if not os.path.exists(annotation_path):
        return polygons
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parsing all coordinates from the line.
            coords = [int(round(c)) for c in map(float, line.split())]
            
            # Each polygon must have an even number of coordinates (pairs of x,y).
            if len(coords) % 2 != 0:
                print(f"Warning: Line {line_num} has odd number of coordinates ({len(coords)}): {line}")
                continue
            
            if len(coords) < 6:  # Need at least 3 points (6 coordinates) for a polygon.
                print(f"Warning: Line {line_num} has too few coordinates ({len(coords)}): {line}")
                continue
            
            polygons.append(coords)
            print(f"Line {line_num}: Found polygon with {len(coords)//2} points")
    
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {e}")
    
    return polygons

# ----------------------------------------------------------------------------#

def polygon_to_bbox(polygon: List[float]) -> Tuple[float, float, float, float]:
    """
    Converting polygon coordinates to 
    COCO bbox format [x, y, width, height]
    """
    # Getting x and y coordinates from the polygon.
    xs = polygon[0::2]  # All even indices: x1, x2, x3, ...
    ys = polygon[1::2]  # All odd indices: y1, y2, y3, ...
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
        
    return x_min, y_min, width, height

# ----------------------------------------------------------------------------#

def polygon_to_segmentation(polygon: List[float]) -> List[List[float]]:
    """
    Converts polygon coordinates to COCO segmentation format.
    COCO expects a list of points flattened: [[x1, y1, x2, y2, ..., xn, yn]]
    """
    # For polygons with many points, COCO expects the same flattened format.
    return [polygon]

# ----------------------------------------------------------------------------#

def get_image_info(image_path: str, image_id: int) -> Dict[str, Any]:
    """
    Gets image information for COCO format.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        
        return {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path),
            "license": 1,
            "date_captured": ""
        }
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

# ----------------------------------------------------------------------------#

def create_coco_annotations(words_dir: str, chars_dir: str, output_json: str):
    """
    Create COCO-style annotations from word images and character annotations
    """
    # Supported image extensions!
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    
    # Finding all image files.
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(words_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(words_dir, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    print(f"Found {len(image_files)} image files")
    
    # COCO dataset structure.
    coco_data = {
        "info": {
            "description": "Character Segmentation Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "url": "",
            "date_created": ""
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "character",
                "supercategory": "text"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    image_id = 1
    
    for image_path in image_files:
        image_filename = os.path.basename(image_path)
        
        # Getting the corresponding annotation file.
        annotation_filename = image_filename + ".txt"
        annotation_path = os.path.join(chars_dir, annotation_filename)
        
        # Getting image info.
        image_info = get_image_info(image_path, image_id)
        if image_info is None:
            continue
        
        # Parsing annotations.
        polygons = parse_annotation_file(annotation_path)
        
        if not polygons:
            print(f"No annotations found for {image_filename}")
            continue
        
        # Adding the image to COCO data.
        coco_data["images"].append(image_info)
        
        # Processing each polygon (character).
        for polygon in polygons:
            # Converting to COCO bbox format.
            x, y, width, height = polygon_to_bbox(polygon)
            
            if width < 2 or height < 2:
                continue

            # Converting to COCO segmentation format.
            segmentation = polygon_to_segmentation(polygon)
            
            # Calculating the area.
            area = width * height
            
            # Creating the annotation.
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # All are characters.
                "segmentation": segmentation,
                "area": area,
                "bbox": [x, y, width, height],
                "iscrowd": 0
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
        
        # Progress update.
        if image_id % 100 == 0:
            print(f"Processed {image_id} images...")
    
    # Saving to JSON!
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"COCO annotations saved to {output_json}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")

# ----------------------------------------------------------------------------#

def validate_annotations(output_json: str):
    """
    Validates the generated COCO annotations.
    """
    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        print()
        print("Validation Results:")
        print(f"Images: {len(coco_data['images'])}")
        print(f"Annotations: {len(coco_data['annotations'])}")
        print(f"Categories: {len(coco_data['categories'])}")
        
        # Checking a few annotations.
        if coco_data['annotations']:
            sample_ann = coco_data['annotations'][0]
            print(f"\nSample annotation:")
            print(f"  Image ID: {sample_ann['image_id']}")
            print(f"  BBox: {sample_ann['bbox']}")
            print(f"  Area: {sample_ann['area']}")
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False

# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    words_dir = r"C:\Users\bgat\TranscriptMapping\Dataset\WordImages"
    chars_dir = r"C:\Users\bgat\TranscriptMapping\Dataset\Chars"
    output_json = r"C:\Users\bgat\TranscriptMapping\util\annotations_chars_coco.json"
    
    create_coco_annotations(words_dir, chars_dir, output_json)
    validate_annotations(output_json)
