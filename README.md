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