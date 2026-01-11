# # using interactive tools :

from yolo_annotator import YOLOAnnotator

# Start annotating
annotator = YOLOAnnotator(
    image_dir='synthetic_apples',
    output_dir='annotations'
)
annotator.annotate_interactive()
