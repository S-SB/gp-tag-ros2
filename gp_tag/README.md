# GP-Tag (Global Positioning Tag)

A ROS2 package for detecting and processing GP-Tags, a novel fiducial marker system that encodes global position information (latitude, longitude, altitude) along with additional metadata.

## Overview

GP-Tag is a computer vision-based positioning system that uses specialized 2D tags containing encoded global position data. When detected, these tags provide:
- Absolute global position (lat/long/altitude)
- 6-DOF relative pose estimation
- Tag metadata (ID, version, scale)

## Package Components

### Nodes
- `gp_tag_realsense.py`: RealSense camera interface node
- `gp_tag_camera.py`: Generic camera interface node
- `gp_tag_node.py`: Main tag detection and processing node
- `gp_tag_viewer.py`: Visualization interface
- `gp_tag_slam_node.py`: SLAM integration node (In Development)

### Core Modules
- `sift_detector.py`: SIFT-based tag detection
- `data_decoder.py`: Tag data decoding
- `finder_decoder.py`: Tag finder pattern analysis
- `annuli_decoder.py`: Tag orientation detection
- `spike_detector.py`: Tag corner refinement

### Configuration
- `config/camera_calibration.yaml`: Camera intrinsics and extrinsics
- `config/realsense_params.yaml`: RealSense-specific parameters
- `config/tag_detector_params.yaml`: Detection algorithm parameters

## Dependencies

### ROS2 Dependencies
- ROS2 (Tested on Humble and Jazzy)
- rclpy
- sensor_msgs
- cv_bridge
- geometry_msgs
- visualization_msgs
- tf2_ros
- realsense2_camera

### Python Dependencies
- OpenCV (cv2)
- NumPy
- reedsolo (Reed-Solomon error correction)

## Installation

1. Create a ROS2 workspace (if you haven't already):
```bash
mkdir -p ~/dev_ws/src
cd ~/dev_ws/src
```

2. Clone this repository:
```bash
git clone https://github.com/S-SB/gp_tag.git
```

3. Install dependencies:
```bash
cd ~/dev_ws
rosdep install --from-paths src --ignore-src -r -y
```

4. Build the package:
```bash
colcon build --packages-select gp_tag
```

5. Source the setup file:
```bash
source install/setup.bash
```

## Usage

### Basic Usage with RealSense Camera
```bash
# Launch RealSense node and tag detector
ros2 launch gp_tag gp_tag.launch.py camera_type:=realsense mode:=visual

# Example with a Logitech 910 Webcam with device_id 4
ros2 launch gp_tag gp_tag.launch.py camera_type:=generic mode:=visual device_id:=4
```

## Performance Metrics

### Tested Hardware Configurations

#### Raspberry Pi 5
- Ubuntu 24.04 with ROS2 Jazzy
- RealSense D435i & Logitech C910
- Performance: ~2-3 seconds per detection

#### MSI Vector GP68HX 12V
- Specifications:
  - CPU: Intel i9-12900HX
  - GPU: NVIDIA RTX 4080 12GB
  - RAM: 32GB DDR4
  - Storage: 6TB NVMe SSD
- Ubuntu 22.04 with ROS2 Humble
- Performance:
  - RealSense D435i (1280x720 + 640x480 depth + IMU): 200-250ms per detection
  - 1920x1080p (RealSense/Logitech): ~350ms per detection

## Topics

### Subscribed Topics
- `/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/color/camera_info` (sensor_msgs/CameraInfo)

### Published Topics
- `/gp_tag/detections` (gp_tag_msgs/TagDetection)
- `/gp_tag/debug_image` (sensor_msgs/Image)
- `/tf` (tf2_msgs/TFMessage)

## Parameters

### Camera Parameters
- `camera_id` (int, default: 0): Camera device ID for generic cameras
- `image_width` (int, default: 1920): Image width in pixels
- `image_height` (int, default: 1080): Image height in pixels

### Detection Parameters
- `detection_rate` (float, default: 1.0): Detection attempt rate in Hz
- `debug_visualization` (bool, default: false): Enable debug visualization


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024 S. E. Sundén Byléhn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

S. E. Sundén Byléhn ([GitHub](https://github.com/S-SB))

## Acknowledgments

- RealSense ROS2 team for camera drivers
- OpenCV team for computer vision tools

