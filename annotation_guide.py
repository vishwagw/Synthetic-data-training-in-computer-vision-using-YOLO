"""
YOLO Annotation Guide & Tools
Complete toolkit for creating YOLO format annotations for apple detection
"""
import cv2
import numpy as np
from pathlib import Path
import json

class YOLOAnnotator:
    """
    Interactive tool for creating YOLO format annotations
    """
    def __init__(self, image_dir, output_dir="annotations"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        self.current_idx = 0
        
        # Annotation state
        self.boxes = []
        self.drawing = False
        self.start_point = None
        
        print(f"✓ Loaded {len(self.images)} images from {image_dir}")
        print(f"✓ Annotations will be saved to {output_dir}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            
            # Add box (start_x, start_y, end_x, end_y)
            self.boxes.append([
                min(self.start_point[0], end_point[0]),
                min(self.start_point[1], end_point[1]),
                max(self.start_point[0], end_point[0]),
                max(self.start_point[1], end_point[1])
            ])
            print(f"  Added box {len(self.boxes)}")
    
    def convert_to_yolo(self, box, img_width, img_height):
        """
        Convert bounding box to YOLO format
        
        Args:
            box: [x1, y1, x2, y2] in pixel coordinates
            img_width: Image width
            img_height: Image height
        
        Returns:
            [class, x_center, y_center, width, height] normalized to 0-1
        """
        x1, y1, x2, y2 = box
        
        # Calculate center, width, height
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize to 0-1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Class 0 for apple
        return [0, x_center, y_center, width, height]
    
    def save_annotations(self, image_path):
        """Save annotations in YOLO format"""
        if len(self.boxes) == 0:
            print("  No boxes to save, skipping...")
            return
        
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Convert all boxes to YOLO format
        yolo_boxes = [self.convert_to_yolo(box, w, h) for box in self.boxes]
        
        # Save to text file
        output_file = self.output_dir / f"{image_path.stem}.txt"
        with open(output_file, 'w') as f:
            for box in yolo_boxes:
                # Format: class x_center y_center width height
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        
        print(f"✓ Saved {len(self.boxes)} annotations to {output_file}")
    
    def annotate_interactive(self):
        """
        Interactive annotation tool
        
        Controls:
        - Draw boxes with mouse (click and drag)
        - 'n': Next image (saves current annotations)
        - 'p': Previous image
        - 'u': Undo last box
        - 'c': Clear all boxes
        - 's': Save and continue
        - 'q': Quit
        """
        if len(self.images) == 0:
            print("No images to annotate!")
            return
        
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        
        print("\n" + "="*60)
        print("YOLO Annotation Tool")
        print("="*60)
        print("\nControls:")
        print("  Mouse: Click and drag to draw bounding boxes around apples")
        print("  'n': Next image (saves annotations)")
        print("  'p': Previous image")
        print("  'u': Undo last box")
        print("  'c': Clear all boxes")
        print("  's': Save annotations")
        print("  'q': Quit")
        print("="*60 + "\n")
        
        while True:
            if self.current_idx >= len(self.images):
                print("\n✓ All images annotated!")
                break
            
            image_path = self.images[self.current_idx]
            img = cv2.imread(str(image_path))
            
            if img is None:
                print(f"Error loading {image_path}")
                self.current_idx += 1
                continue
            
            # Display current progress
            print(f"\nImage {self.current_idx + 1}/{len(self.images)}: {image_path.name}")
            
            while True:
                display = img.copy()
                
                # Draw existing boxes
                for box in self.boxes:
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(display, 'apple', (box[0], box[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw temporary box while dragging
                if self.drawing and hasattr(self, 'temp_end'):
                    cv2.rectangle(display, self.start_point, self.temp_end, (255, 0, 0), 2)
                
                # Add info overlay
                info_text = f"Boxes: {len(self.boxes)} | Image: {self.current_idx + 1}/{len(self.images)}"
                cv2.putText(display, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Annotator', display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('n'):  # Next
                    self.save_annotations(image_path)
                    self.boxes = []
                    self.current_idx += 1
                    break
                
                elif key == ord('p'):  # Previous
                    if self.current_idx > 0:
                        self.boxes = []
                        self.current_idx -= 1
                        break
                
                elif key == ord('u'):  # Undo
                    if self.boxes:
                        self.boxes.pop()
                        print("  Undid last box")
                
                elif key == ord('c'):  # Clear
                    self.boxes = []
                    print("  Cleared all boxes")
                
                elif key == ord('s'):  # Save
                    self.save_annotations(image_path)
                
                elif key == ord('q'):  # Quit
                    print("\nSaving and exiting...")
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    return
        
        cv2.destroyAllWindows()

  class YOLOAnnotationConverter:
    """
    Utilities for converting between annotation formats
    """
    
    @staticmethod
    def yolo_to_corners(yolo_box, img_width, img_height):
        """
        Convert YOLO format to corner coordinates
        
        Args:
            yolo_box: [class, x_center, y_center, width, height] (normalized)
            img_width, img_height: Image dimensions
        
        Returns:
            [x1, y1, x2, y2] in pixels
        """
        cls, x_center, y_center, width, height = yolo_box
        
        # Denormalize
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Calculate corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return [x1, y1, x2, y2]
    
    @staticmethod
    def visualize_yolo_annotations(image_path, label_path):
        """Visualize YOLO annotations on an image"""
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    yolo_box = [int(parts[0])] + [float(x) for x in parts[1:]]
                    x1, y1, x2, y2 = YOLOAnnotationConverter.yolo_to_corners(yolo_box, w, h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, 'apple', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img
    
    @staticmethod
    def validate_annotations(label_dir, image_dir):
        """Validate YOLO annotations"""
        label_dir = Path(label_dir)
        image_dir = Path(image_dir)
        
        labels = list(label_dir.glob("*.txt"))
        issues = []
        
        print(f"\nValidating {len(labels)} annotation files...")
        
        for label_path in labels:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        issues.append(f"{label_path.name}:{line_num} - Wrong number of values (expected 5, got {len(parts)})")
                        continue
                    
                    try:
                        cls = int(parts[0])
                        x, y, w, h = [float(v) for v in parts[1:]]
                        
                        # Check if values are normalized (0-1)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            issues.append(f"{label_path.name}:{line_num} - Values not normalized (must be 0-1)")
                        
                        # Check if box is valid
                        if w <= 0 or h <= 0:
                            issues.append(f"{label_path.name}:{line_num} - Invalid box size (width/height must be > 0)")
                    
                    except ValueError as e:
                        issues.append(f"{label_path.name}:{line_num} - Invalid format: {e}")
        
        if issues:
            print(f"\n⚠ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ All annotations valid!")
        
        return issues


# Example usage and documentation
if __name__ == "__main__":
    print("""
================================================================================
YOLO Annotation Format Guide
================================================================================

YOLO Format Structure:
---------------------
Each image has a corresponding .txt file with the same name.
Each line in the file represents one object:

    class x_center y_center width height

Where:
  - class: Object class ID (0 for apple in our case)
  - x_center: Center X coordinate (normalized 0-1)
  - y_center: Center Y coordinate (normalized 0-1)
  - width: Box width (normalized 0-1)
  - height: Box height (normalized 0-1)

Example:
--------
For an image "apple001.jpg" (640x480 pixels) with an apple at:
  - Top-left corner: (100, 50)
  - Bottom-right corner: (300, 250)

Calculations:
  x_center = ((100 + 300) / 2) / 640 = 0.3125
  y_center = ((50 + 250) / 2) / 480 = 0.3125
  width = (300 - 100) / 640 = 0.3125
  height = (250 - 50) / 480 = 0.4167

The annotation file "apple001.txt" would contain:
  0 0.3125 0.3125 0.3125 0.4167

================================================================================

METHOD 1: Interactive Annotation Tool (Recommended)
================================================================================

from yolo_annotator import YOLOAnnotator

# Create annotator
annotator = YOLOAnnotator(
    image_dir='path/to/images',
    output_dir='annotations'
)

# Start interactive annotation
annotator.annotate_interactive()

# The tool will:
# 1. Display each image
# 2. Let you draw boxes with mouse
# 3. Save annotations in YOLO format
# 4. Move to next image automatically

================================================================================

METHOD 2: Use Professional Tools
================================================================================

1. LabelImg (Free, Easy to Use)
   - Download: https://github.com/HumanSignal/labelImg
   - Install: pip install labelImg
   - Run: labelImg
   - Set format to YOLO in the sidebar
   - Draw boxes and save

2. Roboflow (Web-based, Free Tier)
   - Go to: https://roboflow.com
   - Upload images
   - Annotate online
   - Export in YOLO format

3. CVAT (Professional, Free & Open Source)
   - Website: https://www.cvat.ai
   - Supports team collaboration
   - Export in YOLO format

================================================================================

METHOD 3: Convert from Other Formats
================================================================================

If you have annotations in other formats (like JSON, XML), you can convert them:

from yolo_annotator import YOLOAnnotationConverter

# Example: Visualize existing annotations
img = YOLOAnnotationConverter.visualize_yolo_annotations(
    'path/to/image.jpg',
    'path/to/label.txt'
)
cv2.imshow('Annotated', img)
cv2.waitKey(0)

# Validate all annotations
YOLOAnnotationConverter.validate_annotations(
    label_dir='annotations',
    image_dir='images'
)

================================================================================

Quick Start Example:
================================================================================

# 1. Install requirements
# pip install opencv-python numpy

# 2. Annotate your synthetic images
annotator = YOLOAnnotator('synthetic_apples', 'annotations')
annotator.annotate_interactive()

# 3. Validate annotations
from yolo_annotator import YOLOAnnotationConverter
YOLOAnnotationConverter.validate_annotations('annotations', 'synthetic_apples')

# 4. Use with your detector
from apple_detector import AppleDetector
detector = AppleDetector()
detector.prepare_synthetic_data('synthetic_apples', 'annotations')

================================================================================

Tips for Good Annotations:
================================================================================

✓ Draw tight boxes around apples (no extra space)
✓ Include partially visible apples
✓ Label all apples in the image
✓ Be consistent with box boundaries
✓ Include variety (different angles, sizes, lighting)
✗ Don't make boxes too large
✗ Don't skip small or partially hidden apples
✗ Don't annotate reflections or drawings of apples

================================================================================
    """)
