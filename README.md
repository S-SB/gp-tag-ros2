# GP-Tag ROS2

ROS2 implementation of the GP-Tag fiducial marker system. This package provides ROS2 nodes and interfaces for detecting and processing GP-Tags, which encode global position information (latitude, longitude, altitude) along with additional metadata.

## Related Repositories

- [GP-Tag](https://github.com/S-SB/gp-tag) - Core GP-Tag implementation and documentation
- [GP-Tag Mobile](https://github.com/S-SB/gp-tag-mobile) - Mobile applications for GP-Tag
- [GP-Tag ROS2](https://github.com/S-SB/gp-tag-ros2) - This repository

## Package Structure

This repository contains two main packages:
- `gp_tag`: Core ROS2 implementation of GP-Tag detection and processing
- `gp_tag_msgs`: Custom message definitions for GP-Tag detections

For detailed documentation about the GP-Tag ROS2 implementation, see the [package README](gp_tag/README.md).

## Dependencies

## ROS2 Dependencies
```bash
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-tf2-geometry-msgs
sudo apt install ros-humble-nav-msgs
sudo apt install ros-humble-visualization-msgs
```

## RealSense Dependencies
Install the RealSense SDK and ROS2 wrapper:
```bash
# Install RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev

# Install ROS2 RealSense wrapper
sudo apt install ros-humble-realsense2-camera
```

## Python Dependencies
```bash
pip install numpy opencv-python reedsolo
```
## Quick Start

1. Create a ROS2 workspace:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone this repository:
```bash
git clone https://github.com/S-SB/gp-tag-ros2.git
```

3. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

4. Build the packages:
```bash
colcon build
```

5. Source the workspace:
```bash
source install/setup.bash
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

S. E. Sundén Byléhn ([GitHub](https://github.com/S-SB))