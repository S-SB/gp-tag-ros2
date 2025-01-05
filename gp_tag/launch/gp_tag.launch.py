#!/usr/bin/env python3
"""
GP-Tag Launch File - Main launch configuration for the GP-Tag system.

This launch file configures and starts the GP-Tag detection system components:
- Camera interface (generic or RealSense)
- Tag detector node
- Visualization interface
- SLAM integration (when enabled)

The launch system supports different operating modes and camera types through
command-line arguments.

MIT License

Copyright (c) 2025 S. E. Sundén Byléhn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """
    Generate launch description for the GP-Tag system.
    
    This function configures and returns a launch description that includes:
    - Camera interface node (generic or RealSense)
    - GP-Tag detector node
    - Visualization node (optional)
    
    Launch Arguments:
        camera_type (str): Type of camera to use ('realsense' or 'generic')
        mode (str): Operating mode ('basic', 'visual', or 'slam')
        device_id (str): Camera device ID for generic cameras
        detection_rate (float): Tag detection rate in Hz
        min_detection_interval (float): Minimum time between detections
        max_processing_time (float): Maximum processing time before throttling
    
    Returns:
        LaunchDescription: Complete launch configuration for the system
    """
    pkg_share = FindPackageShare(package='gp_tag').find('gp_tag')
    config_path = os.path.join(pkg_share, 'config', 'camera')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'camera_type',
            default_value='generic',
            description='Type of camera (realsense or generic)'
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='basic',
            description='Operating mode (basic, visual, or slam)'
        ),
        DeclareLaunchArgument(
            'device_id',
            default_value='6',
            description='Camera device ID'
        ),
        DeclareLaunchArgument(
            'detection_rate',
            default_value='2.0',
            description='Tag detection rate in Hz'
        ),
        DeclareLaunchArgument(
            'min_detection_interval',
            default_value='0.5',
            description='Minimum time between detections in seconds'
        ),
        DeclareLaunchArgument(
            'max_processing_time',
            default_value='2.0',
            description='Maximum allowed processing time before throttling'
        ),
        
        # Generic camera node
        Node(
            package='gp_tag',
            executable='gp_tag_camera',
            name='gp_tag_camera',
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration('camera_type'), "' == 'generic'"
                ])
            ),
            parameters=[
                os.path.join(config_path, 'generic.yaml'),
                {'device_id': LaunchConfiguration('device_id')}
            ],
            output='screen'
        ),
        
        # RealSense camera node
        Node(
            package='gp_tag',
            executable='gp_tag_realsense',
            name='gp_tag_camera',
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration('camera_type'), "' == 'realsense'"
                ])
            ),
            parameters=[os.path.join(config_path, 'realsense.yaml')],
            output='screen'
        ),
        
        # GP-Tag detector node
        Node(
            package='gp_tag',
            executable='gp_tag_node',
            name='gp_tag_detector',
            parameters=[{
                'detection_rate': LaunchConfiguration('detection_rate'),
                'min_detection_interval': LaunchConfiguration('min_detection_interval'),
                'max_processing_time': LaunchConfiguration('max_processing_time')
            }],
            output='screen'
        ),
        
        # Viewer node
        Node(
            package='gp_tag',
            executable='gp_tag_viewer',
            name='gp_tag_viewer',
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration('mode'), "' in ['visual', 'slam']"
                ])
            ),
            parameters=[{
                'window_width': 1280,
                'window_height': 720,
                'show_coordinates': True,
                'show_pose': True,
                'font_scale': 1.2,
                'overlay_opacity': 0.7
            }],
            output='screen'
        )
    ])