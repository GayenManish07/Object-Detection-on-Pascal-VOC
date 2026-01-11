import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import argparse
import numpy as np
import os
import time
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# classes to use
TARGET_CLASSES = ['aeroplane', 'bicycle', 'car', 'person', 'dog']

# augmentations
def get_transform(train, strategy='none'):
    transforms = []
    transforms.append(T.ToTensor())
    if train and strategy == 'basic':
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    return T.Compose(transforms)

def apply_mixup(images, targets, alpha=1.0):
  
    batch_size = len(images)
    if batch_size < 2: return images, targets 
    
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size)
    mixed_images = []
    mixed_targets = []

    for i in range(batch_size):
        img1 = images[i]       # Shape: [C, H1, W1]
        t1 = targets[i]
        
        idx2 = index[i]
        img2 = images[idx2]    # Shape: [C, H2, W2]
        t2 = targets[idx2]
        
        # resize images for mixup
        
        if img1.shape[-2:] != img2.shape[-2:]:
            h1, w1 = img1.shape[1], img1.shape[2]
            h2, w2 = img2.shape[1], img2.shape[2]
            
            # Resize Image 2
            # interpolate expects [N, C, H, W], so we unsqueeze/squeeze
            img2 = torch.nn.functional.interpolate(
                img2.unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # Resize Boxes 2
            # Scale factors
            sf_x = w1 / w2
            sf_y = h1 / h2
            
            b2 = t2['boxes'].clone()
            if b2.numel() > 0:
                b2[:, 0] *= sf_x  # xmin
                b2[:, 1] *= sf_y  # ymin
                b2[:, 2] *= sf_x  # xmax
                b2[:, 3] *= sf_y  # ymax
            l2 = t2['labels']
        else:
            b2 = t2['boxes']
            l2 = t2['labels']
        

        # mixup
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_images.append(mixed_img)

        b1, l1 = t1['boxes'], t1['labels']
        
        # merging for mixup
        if b1.numel() == 0: new_b, new_l = b2, l2
        elif b2.numel() == 0: new_b, new_l = b1, l1
        else:
            new_b = torch.cat((b1, b2), dim=0)
            new_l = torch.cat((l1, l2), dim=0)

        target = {
            "boxes": new_b, "labels": new_l, "image_id": t1["image_id"],
            "area": (new_b[:, 3] - new_b[:, 1]) * (new_b[:, 2] - new_b[:, 0]) if new_b.numel() > 0 else torch.tensor([0.]),
            "iscrowd": torch.zeros((len(new_l),), dtype=torch.int64, device=l1.device)
        }
        mixed_targets.append(target)
        
    return mixed_images, mixed_targets

class CustomVOCDataset(Dataset):
    def __init__(self, root_dir, transform, target_classes):
        self.root = root_dir
        self.transform = transform
        self.target_classes = ['__background__'] + target_classes
        self.class_to_idx = {name: i for i, name in enumerate(self.target_classes)}
        
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self.valid_files = []

        if not os.path.exists(self.ann_dir): raise FileNotFoundError(f"Annotations not found at {self.ann_dir}")

        all_xmls = [f for f in os.listdir(self.ann_dir) if f.endswith(".xml")]
        for xml_file in all_xmls:
            tree = ET.parse(os.path.join(self.ann_dir, xml_file))
            root = tree.getroot()
            if any(obj.find("name").text in self.class_to_idx for obj in root.findall("object")):
                img_name = root.find("filename").text
                if not img_name.endswith('.jpg'): img_name += '.jpg'
                if os.path.exists(os.path.join(self.img_dir, img_name)):
                    self.valid_files.append((xml_file, img_name))

    def __getitem__(self, idx):
        xml_file, img_name = self.valid_files[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
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
        
        if self.transform: img = self.transform(img)
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)}

    def __len__(self): return len(self.valid_files)

# model and training
def get_model(num_classes, width_mult=1.0):
    backbone = torchvision.models.mobilenet_v2(weights=None, width_mult=width_mult).features
    with torch.no_grad(): backbone.out_channels = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
    
    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)),
                       box_roi_pool=torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2))
    return model

def train_one_epoch(model, optimizer, loader, device, aug_strategy):
    model.train()
    total_loss = 0
    for images, targets in loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if aug_strategy == 'mixup': images, targets = apply_mixup(images, targets)
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    start = time.time()
    for images, targets in loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        metric.update(model(images), targets)
    fps = len(loader.dataset) / (time.time() - start)
    m = metric.compute()
    return m['map_50'].item(), m['map'].item(), fps

def get_model_size_mb(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=float, required=True)
    parser.add_argument('--aug', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--fraction', type=float, default=0.3, help="Fraction of dataset to use (0.0 - 1.0)")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--train_dir', type=str, default='~/VOC2012_train_val/VOC2012_train_val')
    parser.add_argument('--test_dir', type=str, default='~/VOC2012_test/VOC2012_test')
    args = parser.parse_args()

    args.train_dir = os.path.expanduser(args.train_dir)
    args.test_dir = os.path.expanduser(args.test_dir)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f"[{args.aug} | {args.width}x] Starting on {device}...")

    # load full dataset
    full_train_ds = CustomVOCDataset(args.train_dir, get_transform(True, args.aug), TARGET_CLASSES)
    
    # take subset of dataset
    if args.fraction < 1.0:
        torch.manual_seed(7)
        np.random.seed(7)
        
        subset_size = int(len(full_train_ds) * args.fraction)
        indices = torch.randperm(len(full_train_ds))[:subset_size].tolist()
        train_ds = Subset(full_train_ds, indices)
        print(f"[{args.aug} | {args.width}x] Using Fixed Subset (Seed=7): {subset_size}/{len(full_train_ds)} images")
    else:
        train_ds = full_train_ds
        print(f"[{args.aug} | {args.width}x] Using Full Dataset: {len(full_train_ds)} images")

    # test subset
    test_ds = CustomVOCDataset(args.test_dir, get_transform(False, 'none'), TARGET_CLASSES)
    if args.fraction < 1.0:
        torch.manual_seed(7) # Ensure test subset is also consistent
        test_indices = torch.randperm(len(test_ds))[:int(len(test_ds)*args.fraction)].tolist()
        test_ds = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=2)

    # Model
    model = get_model(len(TARGET_CLASSES)+1, width_mult=args.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train
    train_start = time.time()
    for ep in range(args.epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, args.aug)
        if (ep+1) % 10 == 0:
            print(f"[{args.aug} | {args.width}x] Ep {ep+1} Loss: {loss:.4f}")

    # Eval
    map50, map_all, fps = evaluate(model, test_loader, device)
    size_mb = get_model_size_mb(model)
    
    # Save Results
    config_name = f"Width_{args.width}_Aug_{args.aug}"
    torch.save(model.state_dict(), f"weights_{config_name}.pth")
    
    result_data = {
        "Model Width": [args.width],
        "Augmentation": [args.aug],
        "mAP_50": [round(map50, 4)],
        "mAP_0.5:0.95": [round(map_all, 4)],
        "FPS": [round(fps, 2)],
        "Size (MB)": [round(size_mb, 2)],
        "Train Time (m)": [round((time.time() - train_start)/60, 1)]
    }
    
    # Save results CSV
    pd.DataFrame(result_data).to_csv(f"result_{config_name}.csv", index=False)
    print(f"[{args.aug} | {args.width}x] Finished. Saved result_{config_name}.csv")
