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

  
