# Autonomous Object Manipulating Robot

This project details the development of an autonomous robot capable of identifying and manipulating specific objects within a given space using advanced computer vision and path planning techniques. Designed as part of the CS5335/4610 course, the robot showcases integration of mechanical design, software programming, sensor technologies, and real-time operational functionalities.

## Project Overview

The primary goal of this project is to develop a robotic system that can autonomously identify objects from a predefined list and manipulate them by moving to a designated area. This involves both hardware setup and software development to ensure seamless operation and interaction within dynamic environments.

## Technical Specifications

### Hardware Components

- **Base Vehicle**: 2-wheel drive robot controlled by an Arduino Uno.
- **Sensing**: Xbox Kinect camera for RGB and depth image capture.
- **Manipulation Mechanism**: Features barriers to stabilize and manipulate the objects during movement.

### Software and Libraries

- **Main Control Script**: Python script for high-level operations including object detection, path planning, and robot control.
- **Object Detection and Vision Processing**:
  - **Vision System**: Utilizes RGB and depth images from Kinect.
  - **Model**: Mask R-CNN with a ResNet-50-FPN backbone, fine-tuned on a RGB-D dataset.
  - **Libraries**: PyKinect2 for interfacing with Kinect, PyTorch and TorchVision for model operations, and OpenCV for image processing.

### Path Planning

- **Algorithm**: Rapidly-exploring Random Tree (RRT) algorithm for path planning.
- **Execution**: Real-time path calculation to maneuver towards and manipulate designated objects.

## Setup and Installation

### Requirements

- Python 3.x
- Arduino IDE
- Libraries: PyKinect2, PyTorch, TorchVision, OpenCV, PyQtGraph

### Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/VaradhKaushik/CS5335-4610-Project.git
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup Arduino environment according to the schematics provided in the `Arduino_Omega` folder.

## Usage

1. Connect the Kinect and Arduino to your system.
2. Run the main script to start the robot:
   ```bash
   python main.py
   ```

## Demonstration

The robot has been demonstrated to identify objects placed randomly within an environment and interact with them effectively. The demonstration includes object detection, navigation, and object manipulation sequences.

[![Watch the video](https://img.youtube.com/vi/Y-eC66pAPso/maxresdefault.jpg)](https://youtu.be/Y-eC66pAPso)

## Challenges and Solutions

- **Hardware Weight Management**: Adjustments made by adding supplementary wheels.
- **Software Compatibility**: Transitioned from Conda to Python virtual environments due to package conflicts.

## Future Work

Enhancements such as real-time video processing, consistent locomotion improvement, and sophisticated object manipulation mechanisms are considered for future iterations of the project.

## Contributors

- Varadh Kaushik
- Shrey Patel
- Michael Ambrozie

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
