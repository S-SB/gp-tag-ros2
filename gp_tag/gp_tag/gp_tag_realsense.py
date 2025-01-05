#!/usr/bin/env python3
"""
GP-Tag RealSense Node - RealSense D400 series camera interface for GP-Tag detection.

This module provides a ROS2 node for interfacing with Intel RealSense D400 series
cameras, specifically optimized for GP-Tag detection. It handles color and depth
streams, IMU data when available, and provides orientation estimation through
complementary filtering.

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
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Vector3
import cv2
import numpy as np
from cv_bridge import CvBridge
import pyrealsense2 as rs
from math import atan2, sqrt, pi
import time

class ComplementaryFilter:
    """
    Implements a complementary filter for IMU sensor fusion.
    
    This filter combines accelerometer and gyroscope data to produce stable
    orientation estimates. It uses a weighted combination of integrated gyroscope
    data and accelerometer-based orientation estimates.
    
    Attributes:
        alpha (float): Weight factor for gyroscope data (0-1)
        roll (float): Current roll angle in radians
        pitch (float): Current pitch angle in radians
        yaw (float): Current yaw angle in radians
        last_time (float): Timestamp of last update
    """
    
    def __init__(self, alpha=0.96):
        """
        Initialize the complementary filter.
        
        Args:
            alpha (float, optional): Weight for gyro vs accel data. Defaults to 0.96.
        """
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = time.time()
        
    def update(self, accel_data, gyro_data):
        """
        Update orientation estimates using new sensor data.
        
        Combines accelerometer and gyroscope data using complementary filtering
        to produce stable orientation estimates.
        
        Args:
            accel_data (Vector3): Acceleration in m/s^2
            gyro_data (Vector3): Angular velocity in rad/s
            
        Returns:
            tuple: (roll, pitch, yaw) angles in degrees
        """
        # Get time delta
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Calculate roll and pitch from accelerometer
        accel_roll = atan2(accel_data.y, sqrt(accel_data.x**2 + accel_data.z**2))
        accel_pitch = atan2(-accel_data.x, sqrt(accel_data.y**2 + accel_data.z**2))
        
        # Integrate gyroscope data
        gyro_roll = self.roll + gyro_data.x * dt
        gyro_pitch = self.pitch + gyro_data.y * dt
        gyro_yaw = self.yaw + gyro_data.z * dt
        
        # Complementary filter
        self.roll = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        self.yaw = gyro_yaw  # Yaw cannot be corrected with accelerometer
        
        # Convert to degrees
        roll_deg = self.roll * 180.0 / pi
        pitch_deg = self.pitch * 180.0 / pi
        yaw_deg = self.yaw * 180.0 / pi
        
        return roll_deg, pitch_deg, yaw_deg

class GPTagRealsenseNode(Node):
    """
    ROS2 node for RealSense camera integration with GP-Tag detection.
    
    This node manages the RealSense camera pipeline, handling:
    - Color and depth stream configuration
    - IMU data processing when available
    - Orientation estimation via complementary filtering
    - Camera calibration information
    - Frame alignment between color and depth
    
    Publishers:
        - camera/image_raw (sensor_msgs/Image): Color image frames
        - camera/camera_info (sensor_msgs/CameraInfo): Camera calibration data
        - camera/distance (std_msgs/Float32): Depth at frame center
        - camera/has_imu (std_msgs/Bool): IMU availability flag
        - camera/orientation (geometry_msgs/Vector3): Filtered orientation
        - camera/accel (geometry_msgs/Vector3): Raw accelerometer data
        - camera/gyro (geometry_msgs/Vector3): Raw gyroscope data
        
    Configuration:
        - Color stream: 1280x720 @ 30fps
        - Depth stream: 640x480 @ 30fps
        - IMU streams: Configured if available
            - Accelerometer: 250Hz
            - Gyroscope: 200Hz
    """
    
    def __init__(self):
        """Initialize the RealSense camera node and configure streams."""
        super().__init__('gp_tag_realsense')
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Try to get a list of connected devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            self.get_logger().error('No RealSense devices found')
            raise RuntimeError('No RealSense devices found')
            
        # Get the first device's serial number
        device = devices[0]
        serial_number = device.get_info(rs.camera_info.serial_number)
        self.get_logger().info(f'Using RealSense device: {serial_number}')
        
        # Configure base streams
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Initialize variables
        self.has_imu = False
        self.align = None
        self.imu_filter = ComplementaryFilter()
        
        # Configure publishers
        self._setup_publishers()
        
        # Start streaming with initial config
        self._start_pipeline()
            
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create timer for image capture
        self.create_timer(1.0/30.0, self.timer_callback)
        
        self.get_logger().info('RealSense node initialized')
        
    def _setup_publishers(self):
        """Configure all ROS2 publishers with appropriate QoS settings."""
        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            'camera/image_raw',
            10)
            
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            'camera/camera_info',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            ))
            
        self.distance_pub = self.create_publisher(
            Float32,
            'camera/distance',
            10)
            
        self.has_imu_pub = self.create_publisher(
            Bool,
            'camera/has_imu',
            10)
            
        self.orientation_pub = self.create_publisher(
            Vector3,
            'camera/orientation',
            10)
            
        self.accel_pub = self.create_publisher(
            Vector3,
            'camera/accel',
            10)
            
        self.gyro_pub = self.create_publisher(
            Vector3,
            'camera/gyro',
            10)
            
    def _start_pipeline(self):
        """
        Initialize and start the RealSense pipeline with all configured streams.
        
        Attempts to configure IMU streams if available, sets up frame alignment,
        and initializes camera calibration information.
        
        Raises:
            RuntimeError: If pipeline initialization fails
        """
        try:
            # Try to enable IMU if available
            try:
                # Add IMU streams to config
                self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                self.has_imu = True
                self.get_logger().info('IMU streams configured')
            except Exception as e:
                self.get_logger().warn(f'Could not configure IMU streams: {str(e)}')

            # Start pipeline with all configured streams
            pipeline_profile = self.pipeline.start(self.config)
            self.get_logger().info('RealSense pipeline started successfully')
            
            # Create align object to align depth frames to color frames
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Setup camera info message
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.header.frame_id = 'camera_optical_frame'
            self.camera_info_msg.width = color_intrinsics.width
            self.camera_info_msg.height = color_intrinsics.height
            self.camera_info_msg.k = [
                float(color_intrinsics.fx), 0.0, float(color_intrinsics.ppx),
                0.0, float(color_intrinsics.fy), float(color_intrinsics.ppy),
                0.0, 0.0, 1.0
            ]
            self.camera_info_msg.d = [float(x) for x in color_intrinsics.coeffs]
            self.camera_info_msg.distortion_model = 'plumb_bob'
            
            # Publish IMU availability
            imu_msg = Bool()
            imu_msg.data = self.has_imu
            self.has_imu_pub.publish(imu_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to start pipeline: {str(e)}')
            raise
        
    def timer_callback(self):
        """
        Process and publish camera and IMU data at regular intervals.
        
        This callback:
        - Captures aligned color and depth frames
        - Processes IMU data if available
        - Updates orientation estimates
        - Publishes all data on appropriate topics
        
        The callback handles:
        - Color images
        - Depth measurements
        - IMU data processing
        - Orientation estimation
        - Timestamp synchronization
        """
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
                
            # Get depth at center of frame
            center_x = int(color_frame.get_width() / 2)
            center_y = int(color_frame.get_height() / 2)
            distance = depth_frame.get_distance(center_x, center_y)
            
            # Publish distance
            dist_msg = Float32()
            dist_msg.data = float(distance)
            self.distance_pub.publish(dist_msg)
            
            # Get IMU data if available
            if self.has_imu:
                # Get IMU frames
                accel = frames.first_or_default(rs.stream.accel)
                gyro = frames.first_or_default(rs.stream.gyro)
                
                if accel and gyro:
                    # Get IMU data
                    accel_data = accel.as_motion_frame().get_motion_data()
                    gyro_data = gyro.as_motion_frame().get_motion_data()
                    
                    # Create Vector3 messages
                    accel_msg = Vector3()
                    accel_msg.x = float(accel_data.x)
                    accel_msg.y = float(accel_data.y)
                    accel_msg.z = float(accel_data.z)
                    
                    gyro_msg = Vector3()
                    gyro_msg.x = float(gyro_data.x)
                    gyro_msg.y = float(gyro_data.y)
                    gyro_msg.z = float(gyro_data.z)
                    
                    # Calculate orientation
                    roll, pitch, yaw = self.imu_filter.update(accel_msg, gyro_msg)
                    
                    # Create and publish orientation message
                    orientation_msg = Vector3()
                    orientation_msg.x = float(roll)   # Roll
                    orientation_msg.y = float(pitch)  # Pitch
                    orientation_msg.z = float(yaw)    # Yaw
                    self.orientation_pub.publish(orientation_msg)
                    
                    # Publish raw IMU data
                    self.accel_pub.publish(accel_msg)
                    self.gyro_pub.publish(gyro_msg)
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create messages
            img_msg = self.bridge.cv2_to_imgmsg(color_image, 'bgr8')
            
            # Set timestamps
            now = self.get_clock().now()
            stamp = now.to_msg()
            img_msg.header.stamp = stamp
            self.camera_info_msg.header.stamp = stamp
            
            # Set frame IDs
            img_msg.header.frame_id = 'camera_optical_frame'
            
            # Publish messages
            self.image_pub.publish(img_msg)
            self.camera_info_pub.publish(self.camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error capturing frame: {str(e)}')
            
    def destroy_node(self):
        """
        Clean up RealSense pipeline resources.
        
        Ensures proper shutdown of the RealSense pipeline and releases
        all allocated resources.
        """
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    """
    Main entry point for the GP-Tag RealSense node.
    
    Args:
        args: Command line arguments passed to rclpy.init
    """
    rclpy.init(args=args)
    node = GPTagRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()