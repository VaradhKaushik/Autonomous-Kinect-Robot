from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
import os
import cv2
import numpy as np
from safetensors.torch import save_model, load_model
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from engine import train_one_epoch, evaluate

def generate_bounding_box(mask):
    # Convert mask to binary image if necessary
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store min bounding box coordinates
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    # Find minimum bounding box for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Update minimum bounding box coordinates
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    bounding_box = np.array([min_x, min_y, max_x, max_y])

    return bounding_box

class ImageDataset(Dataset):
    def __init__(self, image_folder='./data', subset=-1):
        self.image_folder = image_folder
        self.images = [x for x in os.listdir(image_folder) if 'depth' not in x and 'mask' not in x and 'loc' not in x]
        if subset > 0:
            idxs = np.random.choice(len(self.images), subset)
            self.images = [self.images[i] for i in idxs]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        image = read_image(image_path).to(dtype=torch.float32)
        # mask = read_image(image_path.replace('.png', '_mask.png'), mode=ImageReadMode.GRAY)
        # masks = (mask == 1).to(dtype=torch.float32)
        # boxes = masks_to_boxes(masks)
        mask = cv2.imread(image_path.replace('.png', '_mask.png'), cv2.IMREAD_GRAYSCALE)
        masks = torch.tensor(mask > 127).unsqueeze(0).to(dtype=torch.uint8)
        boxes = torch.from_numpy(generate_bounding_box(mask)).unsqueeze(0)
        label = [1 if 'box' in image_path else 2 if 'can' in image_path else 3]
        if 3 in label:
            raise ValueError('Label not found', image_path)
        labels = torch.Tensor(label).to(dtype=torch.int64)
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        image = tv_tensors.Image(image)
        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        return image, target
    
def create_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def custom_collate_fn(batch):
    images = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    return images, targets

def train():
    print('Cuda: ', torch.cuda.is_available())
    print('Cuda device count: ', torch.cuda.device_count())
    print('Cuda device id', torch.cuda.current_device())
    print('Cuda device name: ', torch.cuda.get_device_name())
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_dataset = ImageDataset()
    train_set, val_set = random_split(image_dataset, [0.8, 0.2])
    train_set_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    val_set_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    model = create_model(num_classes=3).to(device) # 4 classes: background, box, can
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_set_loader, device, epoch, print_freq=10)
        save_model(model, f'chkpt/model{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}_epoch{epoch}.safetensors')
        lr_scheduler.step()
        evaluate(model, val_set_loader, device=device)
    save_model(model, f'model{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.safetensors')
    test(model, image_dataset, device)

def test(model, image_dataset, device):
    model.eval()
    with torch.no_grad():
        image = image_dataset[np.random.randint(len(image_dataset))][0].to(device)
        prediction = model([image])[0]
        # print(prediction)
        labels = prediction['labels'].cpu()
        labels = ['box' if label == 1 else 'can' if label == 2 else 'background' for label in labels]
        boxes = prediction['boxes'].cpu().to(dtype=torch.uint8)
        print(boxes)
        xmin = min(boxes[0][0], boxes[0][2])
        ymin = min(boxes[0][1], boxes[0][3])
        xmax = max(boxes[0][0], boxes[0][2])
        ymax = max(boxes[0][1], boxes[0][3])
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]])
        
        masks = prediction['masks'].cpu().squeeze(0)
        masks = (masks > 0.5).to(dtype=torch.bool)
        my_boxes = masks_to_boxes(masks)
        print(my_boxes)
        output_image = draw_bounding_boxes(image.to(dtype=torch.uint8), 
                                           my_boxes, 
                                           labels,
                                           colors='red')
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors='blue')
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0).cpu().numpy())
        plt.waitforbuttonpress()
            

def test_display_image_w_box():
    ds = ImageDataset()
    oimg = draw_bounding_boxes(ds[2][0].to(dtype=torch.uint8), ds[2][1]['boxes'], colors='red').to(dtype=torch.uint8)
    print(oimg.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(oimg.cpu().permute(1, 2, 0).numpy())
    plt.waitforbuttonpress()
    
if __name__ == '__main__':
    # train()
    os.chdir('CV Model')
    model = create_model(3)
    load_model(model, './chkpt/model13-04-2024_03-09-00_epoch9.safetensors')
    ds = ImageDataset()
    device = torch.device('cpu')
    test(model, ds, device)
    # test_display_image_w_box()
    