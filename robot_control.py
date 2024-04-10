import ctypes
import numpy as np

import cv2
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# from pykinect2 import PyKinectV2
# from pykinect2.PyKinectV2 import *
# from pykinect2 import PyKinectRuntime

from shapely.geometry import Point, Polygon, LineString

from typing import List, Tuple

OBJECT_CLASSES = ['CAN', 'BOX']

def get_point_cloud_and_predictions(model, kinect) -> Tuple[np.array, List[Tuple[np.array, str]]]:
    pass

class States():
    SEARCH_FOR_OBJECT = 0
    APPROACH_OBJECT = 1 
    PUSH_OBJECT = 2
    RETURN_TO_START = 3
    STOPPED = 4
    FINISHED = 5


class RobotController():
    def __init__(self):
        self.state_machine = States.SEARCH_FOR_OBJECT
        # self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
        self.objects = []
        self.position = (0, 0)
        self.angle = 0

    def drive_to(self, x, y):
        # based off current location and target location, determine the angle to turn to
        relative_x = x - self.position[0]
        relative_y = y - self.position[1]
    
    def run(self):
        while self.state_machine != States.FINISHED:
            if self.state_machine == States.SEARCH_FOR_OBJECT:
                self.search_for_object()
            elif self.state_machine == States.APPROACH_OBJECT:
                self.approach_object()
            elif self.state_machine == States.PUSH_OBJECT:
                self.push_object()
            elif self.state_machine == States.RETURN_TO_START:
                self.return_to_start()
            elif self.state_machine == States.STOPPED:
                self.stop()
            else:
                self.finish()
    
    def search_for_object(self):
        point_cloud, predictions = get_point_cloud_and_predictions(self.model, self.kinect)
        
        # for each prediction, get a 2D top-down view of the object
        for prediction in predictions:
            coords, label = prediction
            coords_list = coords.reshape(coords.shape[0]*coords.shape[1], 3)
            # get x and z coordinates of each point in a list
            coords_2d = coords_list[:, [0, 2]]
            # transform from local coordinate frame to global coordinate frame
            coords_2d = np.dot(coords_2d, np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])) + self.position
            # if the label is can, fit a circle to the points. Else, fit a rectangle
            if label == 'CAN':
                center, radius = cv2.minEnclosingCircle(coords_2d)
                circle_polygon = Point(center[0], center[1]).buffer(radius)
                self.objects.append((circle_polygon, label))
            else:
                rect = cv2.minAreaRect(coords_2d)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                polygon = Polygon(box)
                self.objects.append((polygon, label))
        
        # if there are objects, select the closest one
        if self.objects:
            distances = [obj.distance(self.position) for obj, _ in self.objects]
            closest_object_index = distances.index(min(distances))
            closest_object = self.objects[closest_object_index]
            self.target_object = closest_object
            self.state_machine = States.APPROACH_OBJECT
        else:
            # move to a new position
            self.drive_to(np.random.randint(-10, 10) + self.position[0], np.random.randint(-10, 10) + self.position[1])
    
    def _check_edge(self, node1, node2, object_ignore=[]):
        """
        Check if the line between node1 and node2 does not intersect with any object.
        If no collision, return True. Otherwise, return False.
        """
        line = LineString([node1, node2])
        for obj, _ in self.objects:
            if obj not in object_ignore and obj.intersects(line):
                return False
        return True
    
    def _rrt(self, q_start, q_goal):
        # RRT to find a path to the goal location
        max_iterations = 1000
        q_start = tuple(q_start)
        q_goal = tuple(q_goal)

        # Initialize the RRT tree
        tree = {q_start: None}

        for _ in range(max_iterations):
            # print(tree)
            # Sample a point from a bimodal normal distribution
            if np.random.rand() < 0.7:
                tree_points = list(tree.keys())
                random_tree_point_idx = np.random.randint(0, len(tree_points))
                random_tree_point = tree_points[random_tree_point_idx]
                sample = tuple(np.random.normal(loc=random_tree_point, scale=3.0, size=2))
            else:
                sample = tuple(np.random.normal(loc=q_goal, scale=3.0, size=2))

            # Find the nearest node in the tree
            min_dist = float('inf')
            nearest_node = None
            for node in tree.keys():
                dist = np.linalg.norm(np.array(node) - np.array(sample))
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            
            if sample != nearest_node and self._check_edge(nearest_node, sample):
                tree[sample] = nearest_node
            # Check if the new node is close enough to the goal location
            if sample != nearest_node and np.linalg.norm(np.array(nearest_node) - np.array(q_goal)) < 5:
                # Path found, update the state machine and exit the loop
                tree[q_goal] = nearest_node
                break
                    
                

        # Get the path from the tree
        path = [q_goal]
        while path[-1] != q_start and len(path) < 10:
            print(path)
            path.append(tree[path[-1]])
        
        return path[::-1]

    def approach_object(self):
        assert(self.target_object is not None)
        goal_location = self.target_object.centroid.coords[0]
        if np.linalg.norm(np.array(goal_location) - np.array(self.position)) < 0.1:
            self.state_machine = States.PUSH_OBJECT
        else:
            if self._check_edge(self.position, goal_location):
                self.drive_to(goal_location[0], goal_location[1])
            else:
                path = self._rrt(self.position, goal_location)
                for node in path:
                    self.drive_to(node[0], node[1])
            self.state_machine = States.PUSH_OBJECT


if __name__ == '__main__':
    robot = RobotController()
    robot.objects = [(Polygon([(0, 10), (0, 20), (10, 20), (10, 10)]), 'BOX')]
    print(robot._rrt((0, 0), (20, 20)))