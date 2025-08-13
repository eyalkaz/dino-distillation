from ultralytics import YOLO
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import sys
import os
import time
import random
import shutil
import pathlib
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from TinySAM.tinysam import sam_model_registry, SamPredictor

# debug = True mean use existing dataset foder and dont create it and call dino
debug = True
objects = []
images_for_dino = "images_for_dino"
results_dir = "results"
constants_path = "CONSTANTS/"
root_dir = "datasets/"
labels_dir = root_dir + "labels"
images_dir = root_dir + "images"
test_dir = "test"
test_image = "tomatos.png"
sam_model_type = "vit_t"
dino_model_id = "IDEA-Research/grounding-dino-tiny"
train_val_probability = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epoches = 100

#dino setup
processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_boxes_from_dino(result):
    dino_boxes = []
    dino_scores = []
    for box, score in zip(result["boxes"], result["scores"]):
        box = [round(x, 2) for x in box.tolist()]
        dino_boxes.append(box)
        dino_scores.append(score.item()) #convert scalar tensor to float
    return [dino_boxes, dino_scores]

def get_boxes_from_yolo(result):
    yolo_boxes = []
    data = json.loads(result.to_json())
    for obj in data:
        box = obj['box']
        yolo_boxes.append([box["x1"], box["y1"],box["x2"],box["y2"]])

    return yolo_boxes

def detect_and_segment_image(filename, directory, segment = True, model = dino_model, dino = True):
    image_path = os.path.join(directory, filename)

    sam_image = cv2.imread(image_path)
    image_height, image_width, _ = sam_image.shape

    sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
    dino_image = Image.open(image_path).convert("RGB")

    start_detection_time = time.time()
    if(dino):
        #use dino for detection
        inputs = processor(images=dino_image, text=objects, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[dino_image.size[::-1]]
        )
        result = results[0]
        boxes, dino_scores = get_boxes_from_dino(result)
        if not segment:
            return [boxes, dino_scores, image_width, image_height]

    else:
        # use distilized yolo for detection
        results = model(image_path)
        result = results[0]
        boxes = get_boxes_from_yolo(result)
        if not segment:
            return boxes

    #sam setup
    sam = sam_model_registry[sam_model_type](checkpoint="./TinySAM/weights/tinysam.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    start_sam_time = time.time()
    print(f"finished detection in {start_sam_time - start_detection_time} seconds")

    predictor.set_image(sam_image)     # tinysam predictor
    # use tiny sam to get mask segment of each box
    masks = []
    scores = []
    for b in boxes:
        box = np.array(b)
        mask, score, logits = predictor.predict(
            box=box
        )

        masks.append(mask)
        scores.append(score)

    finish_sam_time = time.time()
    print(f"finished sam in {finish_sam_time - start_sam_time} seconds")
    print(f"total amount of time {finish_sam_time - start_detection_time} seconds")

    #show result
    plt.figure(figsize=(10,10))
    plt.imshow(dino_image)
    for i in range(len(masks)):
        show_mask(masks[i][scores[i].argmax(),:,:], plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, filename))

    return boxes, masks, scores

def write_label(filename, dino_boxes, dino_scores, image_width, image_height, train = True):
    # get dino boxes and convert them to yolo boxes to write as labels
    filename_txt = os.path.splitext(filename)[0] +".txt"
    if train:
        image_type = "train"
    else:
        image_type = "val"

    labels_folder = os.path.join(labels_dir, image_type)

    with open(os.path.join(labels_folder, filename_txt), "w") as f:
        for box, score in zip(dino_boxes,dino_scores):
            f.write("0 ")
            x_min  = box[0]
            x_max = box[2]
            y_min = box[1]
            y_max = box[3]

            x_center = ((x_min + x_max) /2.0) / image_width
            y_center = ((y_min + y_max) /2.0) / image_height
            width = (x_max - x_min) / image_width
            height  = (y_max - y_min) / image_height

            yolo_box = (x_center, y_center, width, height)
            for b in yolo_box:
                f.write(str(b) + " ")
            f.write("\n")

def copy_constants():
    for filename in os.listdir(constants_path):
        image_path = os.path.join(constants_path, filename)
        image_dest = os.path.join(images_for_dino, filename)
        shutil.copy(image_path, image_dest)

def distill_dino():
    # first apply dino on the dataset to get labels
    print("using dino to create labels")
    if(not debug):
        #copy constants into images_for_dino
        copy_constants()
        for filename in os.listdir(images_for_dino):
            dino_boxes, dino_scores, image_width, image_height = detect_and_segment_image(filename, images_for_dino,  segment= False, dino = True)
            image_path = os.path.join(images_for_dino, filename)
            if(random.random() <= train_val_probability):
                train = True # image for train
                image_dest = os.path.join(images_dir, "train")
            else:
                train = False # image for val not for train
                image_dest = os.path.join(images_dir, "val")

            shutil.copy(image_path, image_dest)
            write_label(filename, dino_boxes,dino_scores, image_width, image_height, train=train)

    print("starting to train distilized_model")
    # Load a pretrained YOLO model
    distilized_model = YOLO("parent_model_yolo11n.pt")
    # Train the model on the dino output 100 epochs
    train_results = distilized_model.train(
        data="distilled.yaml",  # Path to dataset configuration file
        epochs=num_epoches,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=device,  # Device to run on
        save = False,
        plots=False,
        name=None,
        project = None
    )

    return distilized_model



def test_distill_yolo(model, image, test_dir):
    # Perform object detection on an image
    path = os.path.join(test_dir, image)
    results = model(path)  # Predict on an image
    # Process results list
    for result in results:
        result.show()  # display to screen


def setup_folders():
    shutil.rmtree(root_dir, ignore_errors=True)
    pathlib.Path(os.path.join(images_dir, "train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(images_dir, "val")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(labels_dir, "train")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(labels_dir, "val")).mkdir(parents=True, exist_ok=True)

def main():
    global images_for_dino

    if len(sys.argv) > 1:
        #objects.extend(sys.argv[2:]) #support list of objects will need to add support for diffrent classes
        objects.append(sys.argv[1]) #support only 1 object
    else:
        print("No objects to detect, please run with name of object")

    if len(sys.argv) > 2:
        images_for_dino = sys.argv[2]
    else:
        print(f"no image folders given, using images_for_dino")
    if(not debug):
        setup_folders()
    # create the distilled model from grounding dino and the data
    distilized_model = distill_dino()
    #distilized_model = YOLO("tomato_distill.pt")
    total_params = sum(p.numel() for p in dino_model.parameters())
    print(f"Number of parameters in grounding dino: {total_params}")
    total_params = sum(p.numel() for p in distilized_model.parameters())
    print(f"Number of parameters in the distilized model: {total_params}")
    distilized_model.save("distilized_model.pt")

    #test_distill_yolo(distilized_model, test_image, test_dir)
    image ="1598 (1).jpg"
    #detect_and_segment_image(image, images_for_dino, model = distilized_model, dino = False)


    # # this code also segment and save the results
    # for filename in os.listdir(test_dir):
    #     dino_boxes, masks, scores = detect_and_segment_image(filename, test_dir, model = distilized_model, dino = False)


if __name__ == "__main__":
    main()