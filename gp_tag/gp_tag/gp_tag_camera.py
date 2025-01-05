#!/usr/bin/env python3
"""
GP-Tag Camera Node - Generic camera interface for GP-Tag detection system.

This module provides a ROS2 node for interfacing with generic cameras (webcams, USB cameras)
using OpenCV and GStreamer. It handles camera initialization, frame capture, and publishing
of images and camera calibration information.

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
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
import cv2
import numpy as np
from cv_bridge import CvBridge
import time

class GPTagCameraNode(Node):
    """
    ROS2 node for capturing and publishing camera frames with calibration information.
    
    This node handles:
    - Camera initialization using either GStreamer pipeline or direct OpenCV capture
    - Frame capture and publishing as ROS Image messages
    - Camera calibration information publishing
    - Automatic reconnection on camera disconnection
    - Camera parameter configuration (exposure, white balance, etc.)
    
    Publishers:
        - /camera/image_raw (sensor_msgs/Image): Raw camera frames
        - /camera/camera_info (sensor_msgs/CameraInfo): Camera calibration information
    
    Parameters:
        - device_id (int, default: 6): Camera device ID
        - width (int, default: 1920): Image width in pixels
        - height (int, default: 1080): Image height in pixels
        - fps (int, default: 30): Frame rate
        - camera_matrix.fx (float): Focal length x
        - camera_matrix.fy (float): Focal length y
        - camera_matrix.cx (float): Principal point x
        - camera_matrix.cy (float): Principal point y
        - distortion_coeffs (list): Distortion coefficients [k1, k2, p1, p2, k3]
        - frame_id (str): TF frame ID for the camera
        - auto_exposure (bool): Enable auto exposure
        - auto_white_balance (bool): Enable auto white balance
        - brightness (int): Camera brightness setting
        - contrast (int): Camera contrast setting
        - saturation (int): Camera saturation setting
        - sharpness (int): Camera sharpness setting
        - retry_delay (float): Delay between reconnection attempts
    """
    
    def __init__(self):
        """Initialize the camera node with parameters and publishers."""
        super().__init__('gp_tag_camera')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device_id', 6),
                ('width', 1920),
                ('height', 1080),
                ('fps', 30),
                ('camera_matrix.fx', 1344.0),
                ('camera_matrix.fy', 1344.0),
                ('camera_matrix.cx', 960.0),
                ('camera_matrix.cy', 540.0),
                ('distortion_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0]),
                ('frame_id', 'camera_optical_frame'),
                ('auto_exposure', True),
                ('auto_white_balance', True),
                ('brightness', 128),
                ('contrast', 128),
                ('saturation', 128),
                ('sharpness', 128),
                ('retry_delay', 1.0)
            ]
        )
        
        # Get parameters
        self.device_id = self.get_parameter('device_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        self.retry_delay = self.get_parameter('retry_delay').value
        
        # Create reliable QoS profile for camera info
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            'camera/image_raw',
            10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            'camera/camera_info',
            reliable_qos)
            
        # Initialize OpenCV capture
        self.initialize_camera()
        
        # Create camera info message
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = self.frame_id
        self.camera_info_msg.height = self.height
        self.camera_info_msg.width = self.width
        
        # Set camera matrix
        self.camera_info_msg.k = [
            self.get_parameter('camera_matrix.fx').value, 0.0, self.get_parameter('camera_matrix.cx').value,
            0.0, self.get_parameter('camera_matrix.fy').value, self.get_parameter('camera_matrix.cy').value,
            0.0, 0.0, 1.0
        ]
        
        # Set distortion coefficients
        self.camera_info_msg.d = self.get_parameter('distortion_coeffs').value
        self.camera_info_msg.distortion_model = 'plumb_bob'
        
        # Initialize bridge
        self.bridge = CvBridge()
        
        # Create timer for image capture
        self.create_timer(1.0/self.fps, self.timer_callback)
        
        self.get_logger().info('Camera node initialized')
        
    def initialize_camera(self):
        """
        Initialize camera connection with automatic retries.
        
        Attempts to initialize the camera using a GStreamer pipeline first,
        falling back to direct OpenCV capture if GStreamer fails. Will continuously
        retry on failure with a configured delay between attempts.
        
        The GStreamer pipeline is optimized for MJPEG cameras with memory mapping
        for efficient frame capture.
        
        Raises:
            RuntimeError: If camera initialization fails (caught and retried)
        """
        while rclpy.ok():
            try:
                # Create GStreamer pipeline
                gst_pipeline = (
                    f"v4l2src device=/dev/video{self.device_id} "
                    "io-mode=2 " # Use memory mapping
                    f"! image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 "
                    "! jpegdec "
                    "! videoconvert "
                    "! video/x-raw,format=BGR "
                    "! appsink drop=1"
                )
                
                self.get_logger().info(f'Opening camera with pipeline: {gst_pipeline}')
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if not self.cap.isOpened():
                    # Fallback to direct capture if GStreamer fails
                    self.get_logger().warn('GStreamer pipeline failed, trying direct capture...')
                    self.cap = cv2.VideoCapture(self.device_id)
                    if not self.cap.isOpened():
                        raise RuntimeError("Failed to open camera")
                    
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                
                # Set additional camera parameters (if direct capture)
                if self.get_parameter('auto_exposure').value:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                if self.get_parameter('auto_white_balance').value:
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                    
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.get_parameter('brightness').value)
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.get_parameter('contrast').value)
                self.cap.set(cv2.CAP_PROP_SATURATION, self.get_parameter('saturation').value)
                self.cap.set(cv2.CAP_PROP_SHARPNESS, self.get_parameter('sharpness').value)
                
                # Verify settings
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                self.get_logger().info(
                    f'Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps'
                )
                return
                
            except Exception as e:
                self.get_logger().error(f'Failed to initialize camera: {str(e)}')
                self.get_logger().info(f'Retrying in {self.retry_delay} seconds...')
                time.sleep(self.retry_delay)
    
    def timer_callback(self):
        """
        Timer callback for capturing and publishing camera frames.
        
        Captures a frame from the camera and publishes it along with the camera
        calibration information. Handles camera disconnections by attempting
        to reinitialize the connection.
        
        The callback maintains synchronized timestamps between the image
        and camera_info messages for proper temporal alignment.
        """
        if not self.cap.isOpened():
            self.get_logger().warn('Camera disconnected, attempting to reconnect...')
            self.initialize_camera()
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to capture frame')
                return
                
            # Create messages
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            
            # Set timestamps
            now = self.get_clock().now()
            stamp = now.to_msg()
            img_msg.header.stamp = stamp
            self.camera_info_msg.header.stamp = stamp
            
            # Set frame IDs
            img_msg.header.frame_id = self.frame_id
            
            # Publish messages
            self.image_pub.publish(img_msg)
            self.camera_info_pub.publish(self.camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error capturing frame: {str(e)}')
    
    def destroy_node(self):
        """
        Clean up node resources.
        
        Ensures proper release of camera resources when the node is shutdown.
        """
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    """
    Main entry point for the GP-Tag camera node.
    
    Args:
        args: Command line arguments passed to rclpy.init
    """
    rclpy.init(args=args)
    node = GPTagCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()