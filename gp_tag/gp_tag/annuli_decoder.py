"""
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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Configuration
OFFSET = 0.5  # Angle offset from boundaries in degrees
REFINEMENT_SAMPLES = 5  # Number of refinement attempts before stopping
MIN_ADJUSTMENT = 0.001  # Minimum angle adjustment to continue refinement (degrees)

# Template Points Configuration in Y-down frame (clockwise positive angles)
# Format: (radius, angle, middle_value, inner_value)
TEMPLATE_POINTS = [
    # Q4Q1 boundary (0°)
    (16.5, -OFFSET, 0, 0),  # Q4: Middle/Inner = white/white
    (16.5, +OFFSET, 1, 1),  # Q1: Middle/Inner = black/black
    
    # Q1Q2 boundary (90°)
    (16.5, 90-OFFSET, 1, 1),  # Q1: Middle/Inner = black/black
    (16.5, 90+OFFSET, 1, 0),  # Q2: Middle/Inner = black/white
    
    # Q2Q3 boundary (180°)
    (16.5, 180-OFFSET, 1, 0),  # Q2: Middle/Inner = black/white
    (16.5, 180+OFFSET, 0, 1),  # Q3: Middle/Inner = white/black
    
    # Q3Q4 boundary (270°)
    (16.5, 270-OFFSET, 0, 1),  # Q3: Middle/Inner = white/black
    (16.5, 270+OFFSET, 0, 0),  # Q4: Middle/Inner = white/white
]

def calculate_marker_radii(corners: np.ndarray, center: np.ndarray) -> Dict[str, float]:
    """
    Calculate concentric marker radii from the tag size.
    
    Uses the tag corners to determine the base unit scale (U) and calculates
    the radii of the concentric annuli rings:
    - Inner radius (grid boundary): 15U
    - Inner ring outer: 16U
    - Middle ring outer: 17U
    - Outer ring: 18U
    
    Args:
        corners: Array of 4 corner points [[x,y], ...]
        center: Center point [x,y]
        
    Returns:
        Dictionary containing radii measurements:
            'inner_inner': Grid boundary radius (15U)
            'inner_outer': Inner ring outer radius (16U)
            'middle_outer': Middle ring outer radius (17U)
            'outer': Outer ring radius (18U)
            'unit': Base unit scale (U)
    """
    # Convert inputs to numpy arrays
    corners = np.array(corners)
    center = np.array(center)
    
    grid_width = np.linalg.norm(corners[1] - corners[0])
    U = float(grid_width / 36)  # Base unit scale
    
    R_i = float(15 * U)  # Inner radius (to grid boundary)
    
    radii = {
        'inner_inner': float(R_i),        # 15U
        'inner_outer': float(R_i + U),    # 16U
        'middle_outer': float(R_i + 2*U), # 17U
        'outer': float(R_i + 3*U),        # 18U
        'unit': float(U)
    }
    
    return radii

def rotate_point(r: float, theta_base: float, theta_rotation: float, 
                center: np.ndarray, U: float) -> Tuple[int, int]:
    """
    Transform a template point to image coordinates using Y-down rotation.
    
    Converts polar coordinates (r, θ) to image coordinates (x,y) in the Y-down
    frame where clockwise rotations are positive.
    
    Args:
        r: Radius in unit scales
        theta_base: Base angle in degrees
        theta_rotation: Additional rotation in degrees
        center: Center point [x,y]
        U: Unit scale factor
        
    Returns:
        Tuple (x,y) of integer image coordinates
    """
    r = float(r)
    theta_base = float(theta_base)
    theta_rotation = float(theta_rotation)
    U = float(U)
    center = np.array(center)
    
    theta = float((theta_base + theta_rotation) % 360)
    theta_rad = float(np.radians(theta))
    
    # Y-down frame (clockwise positive angles)
    x = float(center[0] + r * U * np.cos(theta_rad))  # +x is right
    y = float(center[1] + r * U * np.sin(theta_rad))  # +y is down
    
    return int(x), int(y)

def check_point(image: np.ndarray, x: int, y: int) -> Optional[int]:
    """
    Sample an image point and classify it as black or white.
    
    Uses a threshold of 128 to convert grayscale values to binary.
    Points outside the image bounds return None.
    
    Args:
        image: Input grayscale image
        x, y: Point coordinates to sample
    
    Returns:
        1 for black pixels (value < 128)
        0 for white pixels (value >= 128)
        None if coordinates are outside image bounds
    """
    x, y = int(x), int(y)
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        return 1 if image[y, x] < 128 else 0
    return None

def find_orientation(image: np.ndarray, center: np.ndarray, U: float, step: float = 0.25) -> Tuple[Optional[float], int, List]:
    """
    Find initial tag orientation using template pattern matching.
    
    Scans through possible orientations testing against the template pattern
    defined in TEMPLATE_POINTS. For each angle, samples points in the inner
    and middle rings to count matching black/white transitions.
    
    Args:
        image: Input grayscale image
        center: Tag center coordinates [x,y]
        U: Unit scale factor from tag dimensions
        step: Angular step size for orientation search (degrees)
        
    Returns:
        Tuple containing:
        - Best matching angle in degrees (None if no good match)
        - Number of matching points at best angle
        - List of sampled points [(x,y,value), ...] for visualization
    
    The search covers 0-360° looking for regions where both inner and middle
    ring patterns match the template. A perfect match has 8 matching points.
    """

    center = np.array(center)
    U = float(U)
    step = float(step)
    
    best_match = 0
    best_angle = None
    best_points = []
    
    # Search through all possible rotations (clockwise in Y-down)
    for theta in np.arange(0, 360, step):
        matches = 0
        current_points = []
        
        # Check all template points at this rotation
        for r, theta_base, middle_val, inner_val in TEMPLATE_POINTS:
            # Sample middle ring point
            x_mid, y_mid = rotate_point(r, theta_base, theta, center, U)
            mid_val = check_point(image, x_mid, y_mid)
            current_points.append((x_mid, y_mid, mid_val))
            
            # Sample inner ring point
            x_in, y_in = rotate_point(r-1, theta_base, theta, center, U)
            in_val = check_point(image, x_in, y_in)
            current_points.append((x_in, y_in, in_val))
            
            # Count as match only if BOTH middle and inner match expected values
            if mid_val is not None and in_val is not None:
                if mid_val == middle_val and in_val == inner_val:
                    matches += 1
        
        # Update best match if this angle gives us more matching points
        if matches > best_match:
            best_match = matches
            best_angle = float(theta)
            best_points = current_points
            
    return best_angle, best_match, best_points

def find_transition_pairs() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Identify template point pairs that represent black/white transitions.
    
    Analyzes TEMPLATE_POINTS to find adjacent points where values change:
    - Middle ring transitions (changes in middle_value)
    - Inner ring transitions (changes in inner_value)
    
    Returns:
        Two lists of point pair indices:
        - Middle ring transition pairs [(i,j), ...]
        - Inner ring transition pairs [(i,j), ...]
        Where i,j are indices into TEMPLATE_POINTS
    """
    middle_pairs = []
    inner_pairs = []
    for i in range(0, len(TEMPLATE_POINTS), 2):
        p1 = TEMPLATE_POINTS[i]
        p2 = TEMPLATE_POINTS[i+1]
        if p1[2] != p2[2]:  # Middle ring transitions
            middle_pairs.append((i, i+1))
        if p1[3] != p2[3]:  # Inner ring transitions
            inner_pairs.append((i, i+1))
    return middle_pairs, inner_pairs

def calculate_mean_distances(image: np.ndarray, center: np.ndarray, U: float, angle: float,
                           middle_pairs: List[Tuple[int, int]], 
                           inner_pairs: List[Tuple[int, int]]) -> Tuple[Optional[float], Optional[float], List[Tuple[int, int]]]:
    """
    Calculate mean distances to color transitions in both ring patterns.
    
    For each ring, samples points between transition pair locations to find
    actual black/white boundaries. Measures distances in both clockwise and
    counterclockwise directions from expected transition points.
    
    Args:
        image: Input grayscale image
        center: Tag center coordinates [x,y]
        U: Unit scale factor
        angle: Current orientation angle estimate
        middle_pairs: Middle ring transition point indices
        inner_pairs: Inner ring transition point indices
        
    Returns:
        Tuple containing:
        - Mean clockwise distance to transitions (None if not found)
        - Mean counterclockwise distance to transitions (None if not found)
        - List of detected transition points [(x,y), ...]
        
    Distance means combine measurements from both rings for better accuracy.
    All angles follow the Y-down convention (clockwise positive).
    """
    # Ensure proper types
    center = np.array(center)
    U = float(U)
    angle = float(angle)
    
    # Check middle ring transitions
    middle_cw = []
    middle_ccw = []
    middle_points = []
    
    for p1_idx, p2_idx in middle_pairs:
        p1 = TEMPLATE_POINTS[p1_idx]
        p2 = TEMPLATE_POINTS[p2_idx]
        radius = float(16.5 * U)  # Middle ring radius
        theta1 = float((p1[1] + angle) % 360)
        theta2 = float((p2[1] + angle) % 360)
        
        # Sample points to find middle ring transition
        prev_val = None
        for t in np.linspace(theta1, theta2, num=100):
            x = int(center[0] + radius * np.cos(np.radians(float(t))))
            y = int(center[1] + radius * np.sin(np.radians(float(t))))
            val = check_point(image, x, y)
            
            if prev_val is not None and val != prev_val:
                transition_theta = float(t)
                d1 = float((transition_theta - theta1) % 360)
                d2 = float((theta2 - transition_theta) % 360)
                
                x_trans = int(center[0] + radius * np.cos(np.radians(transition_theta)))
                y_trans = int(center[1] + radius * np.sin(np.radians(transition_theta)))
                middle_points.append((x_trans, y_trans))
                
                if p1[2] == 0:  # 0->1 is clockwise in Y-down
                    middle_cw.append(float(d1))
                    middle_ccw.append(float(d2))
                else:  # 1->0 is counterclockwise
                    middle_ccw.append(float(d1))
                    middle_cw.append(float(d2))
                break
            prev_val = val
            
    # Check inner ring transitions
    inner_cw = []
    inner_ccw = []
    inner_points = []
    
    for p1_idx, p2_idx in inner_pairs:
        p1 = TEMPLATE_POINTS[p1_idx]
        p2 = TEMPLATE_POINTS[p2_idx]
        radius = float(15.5 * U)  # Inner ring radius
        theta1 = float((p1[1] + angle) % 360)
        theta2 = float((p2[1] + angle) % 360)
        
        # Sample points to find inner ring transition
        prev_val = None
        for t in np.linspace(theta1, theta2, num=100):
            x = int(center[0] + radius * np.cos(np.radians(float(t))))
            y = int(center[1] + radius * np.sin(np.radians(float(t))))
            val = check_point(image, x, y)
            
            if prev_val is not None and val != prev_val:
                transition_theta = float(t)
                d1 = float((transition_theta - theta1) % 360)
                d2 = float((theta2 - transition_theta) % 360)
                
                x_trans = int(center[0] + radius * np.cos(np.radians(transition_theta)))
                y_trans = int(center[1] + radius * np.sin(np.radians(transition_theta)))
                inner_points.append((x_trans, y_trans))
                
                if p1[3] == 0:  # 0->1 is clockwise in Y-down
                    inner_cw.append(float(d1))
                    inner_ccw.append(float(d2))
                else:  # 1->0 is counterclockwise
                    inner_ccw.append(float(d1))
                    inner_cw.append(float(d2))
                break
            prev_val = val

    # Average both sets of measurements
    if middle_cw and inner_cw:
        cw_mean = float(np.mean([float(x) for x in (middle_cw + inner_cw)]))
    else:
        cw_mean = None
        
    if middle_ccw and inner_ccw:
        ccw_mean = float(np.mean([float(x) for x in (middle_ccw + inner_ccw)]))
    else:
        ccw_mean = None
        
    transition_points = middle_points + inner_points
    
    return cw_mean, ccw_mean, transition_points

def draw_visualization(debug_image: np.ndarray, center: np.ndarray, radii: Dict[str, float],
                      angle: float, points: List[Tuple[int, int, Optional[int]]], 
                      stage_text: str) -> np.ndarray:
    """
    Create debug visualization of the detection process.
    
    Draws detection state including:
    - Concentric rings at detected radii
    - Sampled points color-coded by value (red=white, blue=black)
    - Coordinate axes showing current orientation
    - Status text showing detection stage
    
    Args:
        debug_image: RGB image to draw on
        center: Tag center coordinates [x,y]
        radii: Dictionary of ring radii measurements
        angle: Current orientation angle
        points: List of (x,y,value) sampled points
        stage_text: Text describing current detection stage
        
    Returns:
        Debug image with visualizations added
        
    Colors:
        - Ring circles: Gray (128,128,128)
        - White points: Red (0,0,255)
        - Black points: Blue (255,0,0)
        - X-axis: Red with arrow
        - Y-axis: Green with arrow
    """
    # Convert center to numpy array
    center = np.array(center)
    
    # Draw marker circles
    for radius in [radii['inner_inner'], radii['inner_outer'],
                  radii['middle_outer'], radii['outer']]:
        cv2.circle(debug_image, 
                  (int(center[0]), int(center[1])), 
                  int(radius), 
                  (128, 128, 128), 1)
    
    # Draw sampled points
    for point in points:
        x, y, val = point
        if val is not None:
            color = (0, 0, 255) if val else (255, 0, 0)  # Red=white, Blue=black
            cv2.circle(debug_image, (int(x), int(y)), 3, color, -1)
    
    # Draw axes (Y-down frame)
    axis_length = int(8 * radii['unit'])
    angle_rad = np.radians(float(angle))
    
    # X axis - red
    x_end = (
        int(center[0] + axis_length * np.cos(angle_rad)),
        int(center[1] + axis_length * np.sin(angle_rad))
    )
    cv2.arrowedLine(debug_image, 
                  (int(center[0]), int(center[1])), 
                  x_end, (0, 0, 255), 3)
    cv2.putText(debug_image, "X", x_end, 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Y axis - green
    y_end = (
        int(center[0] + axis_length * np.cos(angle_rad + np.pi/2)),
        int(center[1] + axis_length * np.sin(angle_rad + np.pi/2))
    )
    cv2.arrowedLine(debug_image, 
                  (int(center[0]), int(center[1])), 
                  y_end, (0, 255, 0), 3)
    cv2.putText(debug_image, "Y", y_end, 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
               
    # Add stage text
    cv2.putText(debug_image, stage_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return debug_image

def refine_angle(image: np.ndarray, center: np.ndarray, U: float, 
                initial_angle: float, radii: Dict[str, float]) -> float:
    """
    Refine orientation angle by balancing transition distances.
    
    Uses iterative adjustment to find angle where transitions in both rings
    are equidistant from expected locations. Stops when improvement is below
    threshold or maximum iterations reached.
    
    Args:
        image: Input grayscale image
        center: Tag center coordinates [x,y]
        U: Unit scale factor
        initial_angle: Starting angle estimate
        radii: Dictionary of ring radii measurements
        
    Returns:
        Refined orientation angle in degrees
        
    Process:
        1. Measure CW/CCW distances to transitions
        2. Adjust angle to balance distances
        3. Reduce adjustment size if no improvement 
        4. Stop when adjustment < MIN_ADJUSTMENT
    """
    # Convert center to numpy array if it's a tuple or list
    center = np.array(center)
    
    # Ensure initial_angle is a float
    initial_angle = float(initial_angle)
    angle = initial_angle
    middle_pairs, inner_pairs = find_transition_pairs()
    
    adjustment = 0.1  # Start with 0.1 degree adjustments
    no_improvement = 0
    best_diff = float('inf')
    best_angle = angle
    
    while no_improvement < REFINEMENT_SAMPLES and adjustment > MIN_ADJUSTMENT:
        # Get current distances
        cw_mean, ccw_mean, _ = calculate_mean_distances(
            image, center, float(U), angle, middle_pairs, inner_pairs)
        
        if cw_mean is None or ccw_mean is None:
            return angle
            
        # Calculate imbalance
        current_diff = abs(float(cw_mean) - float(ccw_mean))
        
        if current_diff < MIN_ADJUSTMENT:
            return angle
            
        # Determine adjustment direction (Y-down clockwise positive)
        if float(cw_mean) > float(ccw_mean):
            test_angle = float((angle + adjustment) % 360)
        else:
            test_angle = float((angle - adjustment) % 360)
            
        # Test new angle
        test_cw, test_ccw, _ = calculate_mean_distances(
            image, center, float(U), test_angle, middle_pairs, inner_pairs)
        
        if test_cw is None or test_ccw is None:
            return angle
            
        test_diff = abs(float(test_cw) - float(test_ccw))
        
        if test_diff < current_diff:
            angle = test_angle
            best_diff = test_diff
            best_angle = angle
            no_improvement = 0
        else:
            adjustment /= 2
            no_improvement += 1
            
    return best_angle

def find_tag_corners(image: np.ndarray, center: np.ndarray, U: float,
                    refined_angle: float, radii: Dict[str, float],
                    debug_info: bool = False) -> np.ndarray:
    """
    Calculate tag corner positions from orientation.
    
    Uses tag geometry to find corners, starting from top-left (-135° from
    orientation) and proceeding clockwise. Corner distance is outer radius * √2
    since corners are 45° from coordinate axes.
    
    Args:
        image: Input image
        center: Tag center coordinates [x,y]
        U: Unit scale factor
        refined_angle: Final orientation angle
        radii: Dictionary of ring radii measurements
        debug_info: If True, saves corner visualization
        
    Returns:
        4x2 array of corner coordinates [[x,y], ...] in clockwise order
        starting from top-left
        
    Debug output (if enabled):
        Saves 'corner_detection.png' showing numbered corners
    """
    center = np.array(center)
    outer_radius = float(radii['outer'])
    
    corners = []
    # Start at top-left (-135° from tag angle) and go clockwise
    for i in range(4):
        corner_angle = refined_angle + (i * 90) - 135  # -135° to start at top-left
        corner_angle_rad = np.radians(corner_angle)
        
        # Corner distance is radius * sqrt(2) since it's 45° from axes
        corner_radius = outer_radius * np.sqrt(2)
        
        x = center[0] + corner_radius * np.cos(corner_angle_rad)
        y = center[1] + corner_radius * np.sin(corner_angle_rad)
        corners.append([x, y])

    corners = np.array(corners)
        
    if debug_info:
        debug = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Draw detected corners
        for i, corner in enumerate(corners):
            cv2.circle(debug, tuple(map(int, corner)), 5, (0,255,0), -1)
            cv2.putText(debug, f"{i}", tuple(map(int, corner)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imwrite('corner_detection.png', debug)
    
    return corners

def decode_annuli_rotation(image: np.ndarray, 
                         corners: np.ndarray, 
                         center: np.ndarray,
                         debug_info: bool = False) -> Tuple[Optional[float], 
                                                          Optional[np.ndarray], 
                                                          np.ndarray]:
    """
    Decode marker rotation based on concentric ring patterns.
    
    Analyzes the black/white patterns in concentric rings to determine tag
    orientation. Uses a template matching approach followed by refinement
    based on edge transition locations.
    
    The process works in several stages:
    1. Template matching finds approximate orientation
    2. Edge transitions in the rings are located
    3. Refinement balances transition distances
    4. Corner positions are computed from final angle
    
    Note:
        Uses Y-down coordinate frame where clockwise rotations are positive.
        Pattern detection is based on sampling points in the inner and middle rings,
        looking for specific black/white transitions that indicate orientation.
    
    Args:
        image: Input image (grayscale or BGR)
        corners: Array of 4 corner points [[x,y], ...]
        center: Center point [x,y]
        debug_info: Whether to generate debug visualizations
        
    Returns:
        Tuple containing:
            - Decoded angle in degrees (None if detection fails)
            - Array of ordered corner points (None if detection fails)
            - Debug visualization image
    """
    # Convert image to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_GRAY2GRAY)
    else:
        gray = image.copy()

    # Convert inputs to numpy arrays if needed
    center = np.array(center)

    # Calculate radii and unit scale
    radii = calculate_marker_radii(corners, center)
    U = float(radii['unit'])
    
    # Find orientation using template matching
    angle, matches, points = find_orientation(gray, center, U)
    
    # Create debug image
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    if angle is not None and matches >= 7:  # Allow 1 mismatch
        # Draw initial template matching visualization
        debug_img1 = draw_visualization(debug_img.copy(), center, radii, angle, points,
                            f"Template Match: {angle:.1f}° ({matches}/8)")
        
        # Run refinement
        refined_angle = refine_angle(gray, center, U, angle, radii)
        
        # Get transition points for visualization
        _, _, transition_points = calculate_mean_distances(
            gray, center, U, refined_angle, *find_transition_pairs())
        
        # Draw refined visualization
        debug_img2 = draw_visualization(debug_img.copy(), center, radii, refined_angle,
                            [(x, y, None) for x, y in transition_points],
                            f"Refined Angle: {refined_angle:.1f}°")
        
        # Find corners independently using annuli measurements
        ordered_corners = find_tag_corners(gray, center, U, refined_angle, radii, debug_info)
        
        debug_combined = np.hstack((debug_img1, debug_img2))
        
        return refined_angle, ordered_corners, debug_combined
    
    return None, None, debug_img
       
