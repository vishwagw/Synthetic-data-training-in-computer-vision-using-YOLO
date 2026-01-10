# 

from apple_detector import AppleDetector

detector = AppleDetector()

# Load your synthetic data
detector.prepare_synthetic_data(
    image_dir='path/to/synthetic/images',
    annotation_dir='path/to/annotations'
)

# Train the model
model, model_path = detector.train_model(
    epochs=100,
    batch_size=16,
    device='cuda'  # or 'cpu' if no GPU
)
