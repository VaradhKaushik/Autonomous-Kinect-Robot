import ctypes
import numpy as np

import cv2
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


# COCO class labels for torchvision models
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



# Load the pre-trained model and set to evaluation mode
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Assuming you're using a CPU for this example
device = torch.device('cpu')
# device = torch.device('cuda')
model.to(device)

# Initialize the Kinect runtime for depth frames
# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

transform = transforms.Compose([
    transforms.ToTensor(),
])

skip_frames = 5  # Number of frames to skip before processing the next one
frame_count = 0

while True:
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        color_frame = kinect.get_last_color_frame()
        color_image = color_frame.reshape((1080, 1920, 4))  # RGBA format
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
        
        # Resize color image
        resized_color_image = cv2.resize(color_image, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Convert resized color image for model input
        pil_image = Image.fromarray(resized_color_image)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip this frame
        # Run object detection
        with torch.no_grad():
            prediction = model(image_tensor)
        
        # Draw the detected bounding boxes and labels with scores
        for i, box in enumerate(prediction[0]['boxes']):
            score = prediction[0]['scores'][i].cpu().item()
            label_index = prediction[0]['labels'][i].cpu().item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_index]
            
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                cv2.rectangle(resized_color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_color_image, f"{label}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        depth_frame = kinect.get_last_depth_frame()
        depth_array = np.array(depth_frame).astype(np.uint16)
        depth_image = depth_array.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        
        cv2.imshow('Depth Image', depth_colormap)
        cv2.imshow('Color Image', resized_color_image)

        L = depth_frame.size
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * L
        csps = TYPE_CameraSpacePoint_Array()
        ptr_depth = np.ctypeslib.as_ctypes(depth_frame.flatten())
        error_state = kinect._mapper.MapDepthFrameToCameraSpace(L, ptr_depth,L, csps)
        if error_state:
            print('Error in mapping depth frame to camera space')
        else:
            print('Depth frame mapped to camera space')
        
        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        data = np.ctypeslib.as_array(pf_csps, shape=(L, 3)) # coordinates of each pixel in depth image in 3D space

        # get coordinates for detected objects
        point_cloud_objects = []
        for i, box in enumerate(prediction[0]['boxes']):
            score = prediction[0]['scores'][i].cpu().item()
            label_index = prediction[0]['labels'][i].cpu().item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_index]
            
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                coords = data[y1:y2, x1:x2]
                point_cloud_objects.append((coords, label))
        
        if cv2.waitKey(1) == 27:
            break

kinect.close()
cv2.destroyAllWindows()