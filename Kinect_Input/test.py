import numpy as np
import cv2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

# Initialize the Kinect runtime for depth frames
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

while True:
    # Check if a new frame is available
    if kinect.has_new_depth_frame():
        # Access the depth frame data
        depth_frame = kinect.get_last_depth_frame()
        
        # Convert the depth frame to a numpy array and reshape it to the correct dimensions (424, 512 for Kinect v2)
        depth_array = np.array(depth_frame).astype(np.uint16)
        depth_image = depth_array.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        
        # Optional: Map the depth values to a grayscale image for visualization
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)  # Scale factor for visualization
        
        # Show the depth image using OpenCV
        cv2.imshow('Depth Image', depth_colormap)
        
        # Break loop on ESC key press
        if cv2.waitKey(1) == 27:
            break

# Release resources
kinect.close()
cv2.destroyAllWindows()
