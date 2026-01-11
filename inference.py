import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import argparse
import random

# class config
TARGET_CLASSES = ['__background__', 'aeroplane', 'bicycle', 'car', 'person', 'dog']
OUTPUT_DIR = "task1_results"

# distinct colors for each class (BGR format for OpenCV)
CLASS_COLORS = {
    'aeroplane': (255, 0, 0),    # Blue
    'bicycle':   (0, 255, 0),    # Green
    'car':       (0, 0, 255),    # Red
    'person':    (255, 255, 0),  # Cyan
    'dog':       (255, 0, 255)   # Magenta
}

# dataset class
class CustomVOCDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.target_classes = TARGET_CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.target_classes)}
        
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self.valid_files = [] 

        if not os.path.exists(self.ann_dir):
            print(f" Error: Annotations dir not found: {self.ann_dir}")
            return

        # only look for images that contain our gven classes(for demonstration purposes)
        print("Scanning dataset for valid images...")
        all_xmls = [f for f in os.listdir(self.ann_dir) if f.endswith(".xml")]
        
        for xml_file in all_xmls:
            tree = ET.parse(os.path.join(self.ann_dir, xml_file))
            root = tree.getroot()
            
            # Check content
            classes_in_img = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in self.class_to_idx and name != '__background__':
                    classes_in_img.append(name)
            
            # Only add if relevant content exists
            if classes_in_img:
                img_name = root.find("filename").text
                if not img_name.endswith('.jpg'): img_name += '.jpg'
                if os.path.exists(os.path.join(self.img_dir, img_name)):
                    self.valid_files.append((xml_file, img_name, list(set(classes_in_img))))

    def __getitem__(self, idx):
        xml_file, img_name, classes_present = self.valid_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        root = ET.parse(os.path.join(self.ann_dir, xml_file)).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in self.class_to_idx:
                bndbox = obj.find("bndbox")
                boxes.append([float(bndbox.find("xmin").text), float(bndbox.find("ymin").text),
                              float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)])
                labels.append(self.class_to_idx[name])
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        return img, target, img_name, classes_present

    def __len__(self): return len(self.valid_files)

def get_model(num_classes, width_mult=1.0):
    backbone = torchvision.models.mobilenet_v2(weights=None, width_mult=width_mult).features
    with torch.no_grad(): backbone.out_channels = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)),
                       box_roi_pool=torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2))
    return model

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea / float((boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1) + (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1) - interArea)

# visualize
def plot_prediction(img, target, prediction, threshold=0.5, metrics=None, save_path="result.png"):
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    
    def draw_box(image, box, label_text, color_rgb):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, 3)
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color_rgb, -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # test image
    axs[0].imshow(img)
    axs[0].set_title("1. Input Image", fontsize=16)
    axs[0].axis('off')

    # gt
    img_gt = img.copy()
    for box, label in zip(target['boxes'], target['labels']):
        name = TARGET_CLASSES[label]
        color = CLASS_COLORS.get(name, (128, 128, 128))
        draw_box(img_gt, box.tolist(), f"{name} (GT)", color)
    axs[1].imshow(img_gt)
    axs[1].set_title(f"2. Ground Truth", fontsize=16)
    axs[1].axis('off')

    # predicted
    img_pred = img.copy()
    pred_boxes = prediction['boxes'].cpu().detach().numpy()
    pred_labels = prediction['labels'].cpu().detach().numpy()
    pred_scores = prediction['scores'].cpu().detach().numpy()
    
    count = 0
    for i, box in enumerate(pred_boxes):
        if pred_scores[i] >= threshold:
            count += 1
            name = TARGET_CLASSES[pred_labels[i]]
            color = CLASS_COLORS.get(name, (128, 128, 128))
            label_txt = f"{name}: {pred_scores[i]:.2f}"
            draw_box(img_pred, box, label_txt, color)
            
    axs[2].imshow(img_pred)
    axs[2].set_title(f"3. Prediction (Thresh={threshold})", fontsize=16)
    axs[2].axis('off')

    if metrics:
        info_text = (f"Sample Stats:\nMatches: {metrics['matches']}/{metrics['total_gt']}\nAvg IoU: {metrics['avg_iou']:.2f}")
        plt.figtext(0.5, 0.05, info_text, ha="center", fontsize=16, bbox={"facecolor":"#f0f0f0", "alpha":1.0, "pad":10})

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n SUCCESS: Result saved to: {save_path}")

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default='~/VOC2012_test/VOC2012_test')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--width', type=float, default=1.0)
    args = parser.parse_args()

    args.test_dir = os.path.expanduser(args.test_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading valid test images...")
    dataset = CustomVOCDataset(args.test_dir)
    if len(dataset) == 0:
        print(" Error: No valid images found.")
        exit()

    # Pick a Random Valid Image
    print(" Picking a random image containing one of the 5 classes...")
    idx = random.randint(0, len(dataset)-1)
    img_np, target, img_name, classes_in_img = dataset[idx]
    
    print(f" Selected Image: {img_name}")
    print(f"  Contains Classes: {classes_in_img}")

    # Load Model
    print(f"Loading model (Width {args.width}x)...")
    model = get_model(len(TARGET_CLASSES), args.width)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f" Error: Weights file '{args.model_path}' not found.")
        exit()
    model.to(device)
    model.eval()

    # Inference
    img_tensor = T.ToTensor()(img_np).to(device)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Metrics
    gt_boxes = target['boxes'].numpy()
    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()
    
    matches, total_iou = 0, 0
    used_gt = [False] * len(gt_boxes)
    for i, p_box in enumerate(pred_boxes):
        if pred_scores[i] < args.threshold: continue
        best_iou, best_gt_idx = 0, -1
        for j, g_box in enumerate(gt_boxes):
            if not used_gt[j]:
                iou = box_iou(p_box, g_box)
                if iou > best_iou: best_iou, best_gt_idx = iou, j
        if best_iou > 0.5:
            matches += 1
            total_iou += best_iou
            used_gt[best_gt_idx] = True

    metric_stats = {'matches': matches, 'total_gt': len(gt_boxes), 'avg_iou': (total_iou / matches) if matches > 0 else 0.0}

    # Save
    clean_name = os.path.splitext(img_name)[0]
    
    class_str = "_".join(classes_in_img)
    save_path = os.path.join(OUTPUT_DIR, f"{clean_name}_{class_str}_prediction.png")
    
    plot_prediction(img_np, target, prediction, threshold=args.threshold, metrics=metric_stats, save_path=save_path)
