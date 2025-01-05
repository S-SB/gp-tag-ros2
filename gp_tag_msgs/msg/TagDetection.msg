# GP-Tag detection message definition
#
# MIT License
# Copyright (c) 2024 S. E. Sundén Byléhn
#
# Carries detection information including:
# - Tag identification and version
# - Global position (lat/long/alt)
# - Local pose and corner points
# - Detection quality metrics

# Header for timestamp and frame information
std_msgs/Header header

# Tag identification
uint32 tag_id
uint8 version_id

# Global position data
float64 latitude
float64 longitude
float64 altitude

# Local pose information
geometry_msgs/Pose pose
float64[36] pose_covariance  # Row-major 6x6 covariance matrix

# Corner points in image coordinates (x,y pairs)
float64[8] corners  # [x1,y1,x2,y2,x3,y3,x4,y4]

# Tag metrics
float64 scale  # Physical size scale factor
uint8 accuracy  # Positioning accuracy level
float64[4] quaternion  # Tag orientation quaternion

# Detection quality metrics
float64 detection_confidence
uint32 num_features
uint32 num_inliers
float64 processing_time

# Status flags
bool success  # Overall detection success
bool id_decode_success  # ID decoding success