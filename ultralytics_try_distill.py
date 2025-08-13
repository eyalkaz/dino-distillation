from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Load a pretrained YOLO11n model
model = YOLO("distilized_model.pt")
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in ultralytics_distill: {total_params}")
model.info()

# Perform object detection on an image
path = "test"
image ="wild.jpg"

IMAGE_PATH = os.path.join(path, image)
results = model(IMAGE_PATH)  # Predict on an image
# Process results list
# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.65

# Collect detections above threshold
filtered_detections = []

for result in results:
    boxes = result.boxes
    if boxes is not None:
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i].item())
                filtered_detections.append((x1, y1, x2, y2, conf, class_id))

# Display function using your code
def display_detections(image_path, detections, save_path=None):
    """Display image with bounding boxes."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 10,
            f'Conf: {conf:.2f}',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=10, color='black'
        )

    ax.set_title(f'Detections: {len(detections)} objects with conf ≥ {CONFIDENCE_THRESHOLD}')
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Result saved to {save_path}")

    plt.show()

# Call the display function
display_detections(IMAGE_PATH, filtered_detections)