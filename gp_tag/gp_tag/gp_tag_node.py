#!/usr/bin/env python3
"""
GP-Tag Node - Main detection and processing node for the GP-Tag system.

This module implements the core GP-Tag detection node that processes camera frames
to detect and decode GP-Tags, extract their embedded global position data, and
estimate their 6-DOF pose relative to the camera. The node includes automatic
performance monitoring and throttling to maintain system stability.

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
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion
from gp_tag_msgs.msg import TagDetection
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gp_tag.sift_detector import SIFTDetector6DoF

class GPTagNode(Node):
    """
    ROS2 node for GP-Tag detection and processing.
    
    This node subscribes to camera images and calibration information, processes
    frames to detect GP-Tags, and publishes detection results including pose
    estimation and decoded tag data.
    
    Features:
    - Adaptive processing rate based on system performance
    - Automatic throttling when processing time exceeds limits
    - Visual debug output showing detected tags
    - Full 6-DOF pose estimation of detected tags
    - Decoding of embedded tag data (position, metadata)
    
    Publishers:
        - gp_tag/camera_pose (geometry_msgs/Pose): Estimated camera pose
        - gp_tag/detections (gp_tag_msgs/TagDetection): Full detection results
        - gp_tag/debug_image (sensor_msgs/Image): Visualization of detections
        
    Subscribers:
        - /camera/image_raw (sensor_msgs/Image): Input camera frames
        - /camera/camera_info (sensor_msgs/CameraInfo): Camera calibration data
        
    Parameters:
        - detection_rate (float, default: 2.0): Target detection rate in Hz
        - min_detection_interval (float, default: 0.5): Minimum time between detections
        - max_processing_time (float, default: 2.0): Maximum allowed processing time
    """
    
    def __init__(self):
        """Initialize the GP-Tag detection node."""
        super().__init__('gp_tag_detector')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detection_rate', 2.0),  # Try to detect twice per second
                ('min_detection_interval', 0.5),  # Minimum time between detections
                ('max_processing_time', 2.0)  # Maximum allowed processing time before throttling
            ]
        )
        
        # Initialize timing variables
        self.last_detection_time = self.get_clock().now()
        self.last_processing_time = 0.0
        self.is_throttled = False
        self.process_interval = 1.0 / self.get_parameter('detection_rate').value
        self.min_interval = self.get_parameter('min_detection_interval').value
        self.max_processing = self.get_parameter('max_processing_time').value
        
        # Initialize detector
        self.detector = SIFTDetector6DoF()
        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Create reliable QoS profile
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(
            Pose, 
            'gp_tag/camera_pose', 
            10)
            
        self.detection_pub = self.create_publisher(
            TagDetection,
            'gp_tag/detections',
            reliable_qos)
            
        self.debug_image_pub = self.create_publisher(
            Image,
            'gp_tag/debug_image',
            10)
        
        # Subscribers
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
            
        self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            reliable_qos)
        
        self.get_logger().info('GP-Tag detector node initialized')
    
    def camera_info_callback(self, msg):
        """
        Process incoming camera calibration information.
        
        Extracts and stores camera matrix and distortion coefficients for use
        in pose estimation calculations.
        
        Args:
            msg (CameraInfo): ROS2 camera calibration message
        """
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Received camera calibration')
    
    def image_callback(self, msg):
        """
        Process incoming camera frames and detect GP-Tags.
        
        This callback handles:
        - Adaptive rate control based on processing performance
        - GP-Tag detection and pose estimation
        - Debug visualization generation
        - Publishing of detection results and poses
        
        The callback includes automatic throttling if processing time exceeds
        the configured maximum, helping maintain system stability under load.
        
        Args:
            msg (Image): ROS2 image message containing the camera frame
        """
        current_time = self.get_clock().now()
        
        # Calculate time since last detection
        time_since_last = (current_time - self.last_detection_time).nanoseconds / 1e9
        
        # Check if we need to throttle based on processing time
        if self.last_processing_time > self.max_processing:
            if not self.is_throttled:
                self.get_logger().warn(
                    f'Processing time ({self.last_processing_time:.1f}s) exceeded limit. '
                    'Reducing detection rate.'
                )
                self.is_throttled = True
                self.process_interval = max(self.process_interval * 2, self.min_interval)
        else:
            if self.is_throttled:
                self.get_logger().info('Processing time normal, resuming standard rate.')
                self.is_throttled = False
                self.process_interval = 1.0 / self.get_parameter('detection_rate').value
        
        # Check if enough time has passed
        if time_since_last < self.process_interval:
            return
            
        if self.camera_matrix is None:
            self.get_logger().warn('No camera calibration received yet')
            return
            
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Record start time
            start_time = self.get_clock().now()
            
            # Run detection
            detection = self.detector.detect(
                image=cv_image,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )
            
            # Calculate processing time
            self.last_processing_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            
            if detection and detection.get('success', False):
                # Get core data
                tag_data = detection.get('tag_data', {})
                if not tag_data:
                    return
                    
                # Ensure tag_id is an integer and exists
                tag_id = tag_data.get('tag_id')
                if tag_id is None:
                    return
                
                self.get_logger().info(
                    f"Tag detected - ID {tag_id} "
                    f"Time {detection['detection_time_ms']:.1f}ms "
                    f"Lat {tag_data.get('latitude', 0):.6f}° "
                    f"Long {tag_data.get('longitude', 0):.6f}°"
                )
                
                # Create and publish pose message
                pose_msg = Pose()
                
                # Set position
                pose_msg.position = Point(
                    x=float(detection['position'][0]),
                    y=float(detection['position'][1]),
                    z=float(detection['position'][2])
                )
                
                # Set orientation
                q = detection['rotation']
                pose_msg.orientation = Quaternion(
                    x=float(q[0]),
                    y=float(q[1]),
                    z=float(q[2]),
                    w=float(q[3])
                )
                
                # Create detection message
                detect_msg = TagDetection()
                detect_msg.header.stamp = self.get_clock().now().to_msg()
                detect_msg.header.frame_id = "camera_optical_frame"
                
                detect_msg.tag_id = int(tag_id)
                detect_msg.version_id = int(tag_data.get('version_id', 0))
                detect_msg.latitude = float(tag_data.get('latitude', 0.0))
                detect_msg.longitude = float(tag_data.get('longitude', 0.0))
                detect_msg.altitude = float(tag_data.get('altitude', 0.0))
                detect_msg.pose = pose_msg
                detect_msg.scale = float(tag_data.get('scale', 0.0))
                detect_msg.accuracy = int(tag_data.get('accuracy', 0))
                detect_msg.quaternion = [float(x) for x in detection['rotation']]
                detect_msg.detection_confidence = float(detection.get('confidence', 0.0))
                detect_msg.num_features = int(detection.get('num_features', 0))
                detect_msg.num_inliers = int(detection.get('num_inliers', 0))
                detect_msg.processing_time = float(detection.get('detection_time_ms', 0.0))
                detect_msg.success = True
                detect_msg.id_decode_success = bool(tag_data.get('id_decode_success', False))
                
                # Add corner points to message
                corners = np.array(detection['corners'], dtype=np.float64)
                corners_flat = []
                for corner in corners:
                    corners_flat.extend([float(corner[0]), float(corner[1])])
                detect_msg.corners = corners_flat
                
                # Create debug image
                debug_image = cv_image.copy()
                corners = np.array(detection['corners'], dtype=np.int32)
                
                # Draw detection boundary
                for i in range(4):
                    cv2.line(debug_image, 
                            tuple(corners[i]), 
                            tuple(corners[(i+1)%4]), 
                            (0, 255, 0), 2)
                
                # Create ROS Image message and publish
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.header.frame_id = "camera_optical_frame"
                
                # Publish all messages
                self.pose_pub.publish(pose_msg)
                self.detection_pub.publish(detect_msg)
                self.debug_image_pub.publish(debug_msg)
                
                # Update timing
                self.last_detection_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    """
    Main entry point for the GP-Tag detector node.
    
    Args:
        args: Command line arguments passed to rclpy.init
    """
    rclpy.init(args=args)
    node = GPTagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()