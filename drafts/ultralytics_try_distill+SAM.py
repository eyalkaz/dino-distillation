from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from TinySAM.tinysam import sam_model_registry, SamPredictor
import torch
from PIL import Image
import json
sam_model_type = "vit_t"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Load a pretrained YOLO11n model
model = YOLO("distilized_model.pt")
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in ultralytics_distill: {total_params}")
model.info()

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_t"](checkpoint="./TinySAM/weights/tinysam.pth").to(device)
sam_predictor = SamPredictor(sam)

# Perform object detection on an image
path = "drafts"
#image ="bus.jpg"
image ="wild.jpg"
#image ="tomatos.png"
IMAGE_PATH = os.path.join(path, image)
results = model(IMAGE_PATH)  # Predict on an image
# Process results list
# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.65


#sam setup
show_image = Image.open(IMAGE_PATH).convert("RGB")
sam_image = cv2.imread(IMAGE_PATH)
image_height, image_width, _ = sam_image.shape

sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
#sam setup
sam = sam_model_registry[sam_model_type](checkpoint="./TinySAM/weights/tinysam.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

predictor.set_image(sam_image)     # tinysam predictor


def get_boxes_from_yolo(result):
    yolo_boxes = []
    data = json.loads(result.to_json())
    for obj in data:
        box = obj['box']
        yolo_boxes.append([box["x1"], box["y1"],box["x2"],box["y2"]])

    return yolo_boxes
filtered_detections = []
# Collect detections above threshold
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i].item())
                filtered_detections.append([x1, y1, x2, y2])

masks = []
scores = []
for b in filtered_detections:
    box = np.array(b)
    mask, score, logits = predictor.predict(
        box=box
    )

    masks.append(mask)
    scores.append(score)


#show result
plt.figure(figsize=(10,10))
plt.imshow(show_image)
for i in range(len(masks)):
    show_mask(masks[i][scores[i].argmax(),:,:], plt.gca())
plt.axis('off')
plt.show()