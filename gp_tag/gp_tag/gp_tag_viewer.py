#!/usr/bin/env python3
"""
GP-Tag Viewer Node - Real-time visualization interface for GP-Tag detections.

This module provides a real-time visualization interface for GP-Tag detections,
displaying camera feeds, detection overlays, pose information, and sensor data
from RealSense cameras when available. It handles multiple data streams and
provides an interactive display window with configurable overlays.

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
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Vector3
from gp_tag_msgs.msg import TagDetection
from std_msgs.msg import Float32, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import signal

def quaternion_to_euler_NegY(q):
    """
    Convert quaternion to Euler angles with negated pitch for camera conventions.
    
    This function converts a quaternion to Euler angles (roll, pitch, yaw) while
    negating the pitch angle to match camera coordinate conventions.
    
    Args:
        q: Quaternion object with w, x, y, z components
        
    Returns:
        numpy.ndarray: Array of [roll, pitch, yaw] in degrees
    """
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([np.degrees(roll),
                    np.degrees(-pitch),
                    np.degrees(yaw)])

class GPTagViewer(Node):
    """
    ROS2 node for real-time visualization of GP-Tag detections and sensor data.
    
    This node provides a graphical interface for:
    - Displaying camera feeds
    - Visualizing tag detections with overlays
    - Showing pose and position information
    - Displaying RealSense-specific data (depth, IMU)
    - Interactive window with configurable display options
    
    The viewer runs in a separate thread to maintain UI responsiveness and
    handles multiple data streams with appropriate synchronization.
    
    Subscribers:
        - /camera/image_raw (sensor_msgs/Image): Raw camera feed
        - gp_tag/debug_image (sensor_msgs/Image): Detection visualization
        - gp_tag/camera_pose (geometry_msgs/Pose): Camera pose
        - gp_tag/detections (gp_tag_msgs/TagDetection): Tag detection data
        - camera/distance (std_msgs/Float32): RealSense depth data
        - camera/has_imu (std_msgs/Bool): IMU availability
        - camera/orientation (geometry_msgs/Vector3): IMU orientation
        - camera/accel (geometry_msgs/Vector3): IMU acceleration
        - camera/gyro (geometry_msgs/Vector3): IMU angular velocity
        
    Parameters:
        - window_width (int): Display window width
        - window_height (int): Display window height
        - show_coordinates (bool): Show coordinate overlays
        - show_pose (bool): Show pose information
        - font_scale (float): Size of overlay text
        - overlay_opacity (float): Transparency of information overlays
    """
    
    def __init__(self):
        """Initialize the viewer node with display thread and subscribers."""
        super().__init__('gp_tag_viewer')
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.get_logger().info('Starting viewer initialization...')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('window_width', 1280),
                ('window_height', 720),
                ('show_coordinates', True),
                ('show_pose', True),
                ('font_scale', 1.2),
                ('overlay_opacity', 0.7)
            ]
        )
        
        # Initialize state
        self._initialize_state()
        
        # Initialize display thread
        self.display_thread = threading.Thread(target=self._display_thread_function)
        self.display_thread.daemon = True
        
        # Create reliable QoS profile
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Setup subscribers
        self._setup_subscribers(reliable_qos)
        
        # Start display thread
        self.display_thread.start()
        self.get_logger().info('Display thread started')
        
        self.get_logger().info('GP-Tag viewer initialization complete')
    
    def _initialize_state(self):
        """Initialize internal state variables and buffers."""
        self.bridge = CvBridge()
        self.current_image = None
        self.current_pose = None
        self.current_corners = None
        self.tag_data = None
        self.last_detection_time = self.get_clock().now()
        self.detection_timeout = 1.0  # seconds
        self.running = True
        self.window_name = 'GP-Tag Viewer'
        
        # RealSense specific state
        self.distance = None
        self.has_imu = False
        self.accel_data = None
        self.gyro_data = None
        self.orientation_data = None
        self.last_imu_time = self.get_clock().now()
        self.imu_timeout = 0.5  # seconds
        self.is_realsense = False
    
    def _setup_subscribers(self, reliable_qos):
        """
        Setup all ROS2 subscribers with appropriate QoS profiles.
        
        Args:
            reliable_qos: QoS profile for reliable communication
        """
        # Core subscriptions
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.raw_image_callback,
            10)
            
        self.create_subscription(
            Image,
            'gp_tag/debug_image',
            self.debug_image_callback,
            reliable_qos)
            
        self.create_subscription(
            Pose,
            'gp_tag/camera_pose',
            self.pose_callback,
            reliable_qos)
            
        self.create_subscription(
            TagDetection,
            'gp_tag/detections',
            self.tag_callback,
            reliable_qos)

        # RealSense specific subscriptions
        self.create_subscription(
            Float32,
            'camera/distance',
            self.distance_callback,
            10)
            
        self.create_subscription(
            Bool,
            'camera/has_imu',
            self.imu_status_callback,
            10)
            
        self.create_subscription(
            Vector3,
            'camera/orientation',
            self.orientation_callback,
            10)
            
        self.create_subscription(
            Vector3,
            'camera/accel',
            self.accel_callback,
            10)
            
        self.create_subscription(
            Vector3,
            'camera/gyro',
            self.gyro_callback,
            10)
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.
        
        Ensures clean shutdown of display window and node resources.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.get_logger().info('Shutdown signal received')
        self.running = False
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
        try:
            self.destroy_node()
        except:
            pass
    
    def distance_callback(self, msg):
        """Handle incoming depth measurements."""
        self.distance = msg.data
        self.is_realsense = True
        
    def imu_status_callback(self, msg):
        """Handle IMU availability status updates."""
        self.has_imu = msg.data
        
    def orientation_callback(self, msg):
        """Handle IMU orientation updates."""
        self.orientation_data = msg
        self.last_imu_time = self.get_clock().now()
        
    def accel_callback(self, msg):
        """Handle IMU acceleration updates."""
        self.accel_data = msg
        self.last_imu_time = self.get_clock().now()
        
    def gyro_callback(self, msg):
        """Handle IMU angular velocity updates."""
        self.gyro_data = msg
        self.last_imu_time = self.get_clock().now()
    
    def raw_image_callback(self, msg):
        """
        Store latest camera image.
        
        Args:
            msg (Image): ROS2 image message from camera
        """
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting raw image: {str(e)}')
    
    def debug_image_callback(self, msg):
        """
        Handle debug image with detection visualization.
        
        Updates the display image only when active detections exist.
        
        Args:
            msg (Image): ROS2 image message with detection overlays
        """
        try:
            if self.tag_data is not None:
                self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting debug image: {str(e)}')
    
    def pose_callback(self, msg):
        """
        Handle incoming pose data.
        
        Args:
            msg (Pose): ROS2 pose message with position and orientation
        """
        self.current_pose = msg
        self.last_detection_time = self.get_clock().now()
    
    def tag_callback(self, msg):
        """
        Handle incoming tag detection data.
        
        Processes detection data and extracts corner points for visualization.
        
        Args:
            msg (TagDetection): Custom message with tag detection information
        """
        self.tag_data = msg
        if len(msg.corners) == 8:  # Make sure we have all 4 corners (8 coordinates)
            self.current_corners = np.array([[msg.corners[i], msg.corners[i+1]] 
                                        for i in range(0, 8, 2)])
        self.last_detection_time = self.get_clock().now()
    
    def _display_thread_function(self):
        """
        Main display thread function.
        
        Handles:
        - Window creation and sizing
        - Display initialization
        - Main display loop
        - User input processing
        - Graceful shutdown
        """
        try:
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.waitKey(1)  
            
            width = self.get_parameter('window_width').value
            height = self.get_parameter('window_height').value
            cv2.resizeWindow(self.window_name, width, height)
            cv2.waitKey(1)

            dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(dummy_image, "Waiting for camera feed...", 
                       (int(width/4), int(height/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(self.window_name, dummy_image)
            cv2.waitKey(1)
            
            self.get_logger().info(f'Display window created: {width}x{height}')
            
            while self.running and rclpy.ok():
                try:
                    if self.current_image is not None:
                        display_img = self.current_image.copy()
                        display_img = self.draw_overlay(display_img)
                        cv2.imshow(self.window_name, display_img)
                    
                    key = cv2.waitKey(16)  # ~60 FPS
                    if key == 27:  # ESC key
                        self.get_logger().info('ESC pressed, shutting down...')
                        self.running = False
                        break
                except Exception as e:
                    self.get_logger().error(f'Display error: {str(e)}')
                    continue
                    
        except Exception as e:
            self.get_logger().error(f'Failed to create display window: {str(e)}')
        finally:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except:
                pass
    
    def draw_overlay(self, image):
        """
        Draw information overlays on the display image.
        
        Adds visual elements including:
        - Tag detection boundaries
        - Position and orientation information
        - RealSense sensor data when available
        - Performance metrics
        
        Args:
            image: OpenCV image to draw on
            
        Returns:
            OpenCV image with overlays added
        """
        if self.tag_data is None or self.current_pose is None:
            return image

        try:
            time_since_detection = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
            if time_since_detection > self.detection_timeout:
                self.current_pose = None
                self.tag_data = None
                return image

            overlay = image.copy()

            if self.current_corners is not None:
                corners = self.current_corners.astype(np.int32)
                for i in range(4):
                    cv2.line(overlay, 
                            tuple(corners[i]), 
                            tuple(corners[(i+1)%4]), 
                            (0, 255, 0), 2)

            pos = self.current_pose.position
            euler_angles = quaternion_to_euler_NegY(self.current_pose.orientation)
            roll, pitch, yaw = euler_angles

            lat = str(round(float(self.tag_data.latitude), 6))
            lon = str(round(float(self.tag_data.longitude), 6))
            alt = str(round(float(self.tag_data.altitude), 1))
            roll_val = str(round(float(roll), 1))
            pitch_val = str(round(float(pitch), 1))
            yaw_val = str(round(float(yaw), 1))

            tag_info_lines = [
                f"Tag ID {str(self.tag_data.tag_id)}",
                f"Version {str(self.tag_data.version_id)}",
                "",
                "Global Position",
                f"  Lat {lat} deg",
                f"  Long {lon} deg",
                f"  Alt {alt} m",
                "",
                "Camera Pose",
                f"  Roll {roll_val} deg",
                f"  Pitch {pitch_val} deg",
                f"  Yaw {yaw_val} deg",
                "",
                "Position (m)",
                f"  X {str(round(float(pos.x), 3))}",
                f"  Y {str(round(float(pos.y), 3))}",
                f"  Z {str(round(float(pos.z), 3))}"
            ]

            margin = 20
            line_height = 30
            font_scale = self.get_parameter('font_scale').value
            font = cv2.FONT_HERSHEY_SIMPLEX
            overlay_opacity = self.get_parameter('overlay_opacity').value

            rect_height = len(tag_info_lines) * line_height + 2 * margin
            rect_width = 300
            cv2.rectangle(overlay, 
                        (margin, margin), 
                        (margin + rect_width, margin + rect_height),
                        (0, 0, 0), 
                        -1)

            y = margin + line_height
            for line in tag_info_lines:
                if line:
                    cv2.putText(overlay, line,
                            (margin + 10, y),
                            font, font_scale * 0.8,
                            (0, 255, 0),
                            2, cv2.LINE_AA)
                y += line_height

            if self.is_realsense:
                realsense_info_lines = ["RealSense Data"]
                
                if self.distance is not None:
                    realsense_info_lines.extend([
                        "",
                        "Distance Information",
                        f"  Distance: {str(round(float(self.distance), 3))} m"
                    ])

                time_since_imu = (self.get_clock().now() - self.last_imu_time).nanoseconds / 1e9
                if self.has_imu and time_since_imu < self.imu_timeout:
                    if self.orientation_data:
                        realsense_info_lines.extend([
                            "",
                            "Orientation",
                            f"  Roll: {str(round(float(self.orientation_data.x), 2))} deg",
                            f"  Pitch: {str(round(float(self.orientation_data.y), 2))} deg",
                            f"  Yaw: {str(round(float(self.orientation_data.z), 2))} deg"
                        ])
                        
                    if self.accel_data:
                        realsense_info_lines.extend([
                            "",
                            "Accelerometer (m/s²)",
                            f"  X: {str(round(float(self.accel_data.x), 2))}",
                            f"  Y: {str(round(float(self.accel_data.y), 2))}",
                            f"  Z: {str(round(float(self.accel_data.z), 2))}"
                        ])
                        
                    if self.gyro_data:
                        realsense_info_lines.extend([
                            "",
                            "Gyroscope (deg/s)",
                            f"  X: {str(round(float(self.gyro_data.x * 180/np.pi), 2))} deg",
                            f"  Y: {str(round(float(self.gyro_data.y * 180/np.pi), 2))} deg",
                            f"  Z: {str(round(float(self.gyro_data.z * 180/np.pi), 2))} deg"
                        ])

                right_rect_height = len(realsense_info_lines) * line_height + 2 * margin
                right_rect_width = 300
                right_x = image.shape[1] - margin - right_rect_width
                
                cv2.rectangle(overlay, 
                            (right_x, margin),
                            (right_x + right_rect_width, margin + right_rect_height),
                            (0, 0, 0),
                            -1)

                # Draw RealSense info
                y = margin + line_height
                for line in realsense_info_lines:
                    if line:
                        cv2.putText(overlay, line,
                                (right_x + 10, y),
                                font, font_scale * 0.8,
                                (0, 255, 0),
                                2, cv2.LINE_AA)
                    y += line_height

            # Blend overlay with original image
            cv2.addWeighted(overlay, overlay_opacity,
                        image, 1 - overlay_opacity,
                        0, image)
            
            return image
            
        except Exception as e:
            self.get_logger().error(f'Error drawing overlay: {str(e)}')
            return image
    
    def destroy_node(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        super().destroy_node()

def main(args=None):
    """
    Main entry point for the GP-Tag RealSense node.
    
    Args:
        args: Command line arguments passed to rclpy.init
    """
    rclpy.init(args=args)
    viewer = GPTagViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()