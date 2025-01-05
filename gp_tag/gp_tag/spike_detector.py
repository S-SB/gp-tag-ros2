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
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class SpikeDetectorResults:
    """Class to hold spike detection results and metrics for image corner and perspective correction.
    
    Encapsulates the results from the spike detection process including corrected
    image, corner positions, transformation data, and quality metrics.
    
    Attributes:
        corrected_image: The perspective-corrected output image
        source_corners: Original detected corner positions in [x,y] format
        target_corners: Destination corner positions after correction
        transform_matrix: 3x3 perspective transform matrix used for correction
        average_corner_displacement: Mean distance corners moved during correction
        max_corner_displacement: Maximum distance any corner moved
        skew_angles: Horizontal and vertical skew angles in degrees (0° = no skew)
    """
    corrected_image: np.ndarray
    source_corners: np.ndarray
    target_corners: np.ndarray
    transform_matrix: np.ndarray
    average_corner_displacement: float
    max_corner_displacement: float
    skew_angles: Tuple[float, float]

class SpikeDetector:
    """
    Detector for finding and correcting corner spikes in images.
    
    Uses edge detection and line intersection analysis to detect corner spikes,
    then applies perspective correction to normalize the image.
    """
    
    def __init__(self, target_size: int = 360, corner_roi_size: int = 75):
        """
        Initialize the spike detector with configurable parameters.
        
        Args:
            target_size: Size of the output square image in pixels.
                Must be positive.
            corner_roi_size: Size of region to analyze for corners in pixels.
                Should be smaller than target_size/2.
                
        Raises:
            ValueError: If target_size <= 0 or corner_roi_size >= target_size/2
        """
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        if corner_roi_size >= target_size/2:
            raise ValueError("corner_roi_size must be less than half of target_size")
            
        self.target_size = target_size
        self.corner_size = corner_roi_size

    def _find_corner_lines(self, roi: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find potential corner-forming lines in the region of interest.
        
        Uses Hough transform to detect strong lines that might form corners.
        
        Args:
            roi: Binary edge image region to analyze
            
        Returns:
            List of line segments, each defined by two (x,y) point tuples.
            Returns empty list if no lines are found.
        """
        lines = cv2.HoughLinesP(roi,
                               rho=1,
                               theta=np.pi/180,
                               threshold=20,
                               minLineLength=20,
                               maxLineGap=5)
        
        if lines is None:
            return []
            
        return [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines[:, 0]]

    def _find_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[int, int]]:
        """
        Calculate the intersection point of two lines.
        
        Args:
            line1: First line as ((x1,y1), (x2,y2))
            line2: Second line as ((x1,y1), (x2,y2))
            
        Returns:
            (x,y) intersection point if lines intersect within bounds,
            None if lines are parallel or intersection is out of bounds
        """
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        return x, y

    def _find_corner_point(self, lines: List, corner_type: str) -> Optional[Tuple[int, int]]:
        """
        Find the most likely corner point from a set of detected lines.
        
        Uses voting accumulator to find point where most lines intersect.
        
        Args:
            lines: List of detected line segments
            corner_type: Location identifier ('top_left', 'top_right', etc.)
            
        Returns:
            (x,y) coordinates of detected corner, None if no clear corner found
        """
        if len(lines) < 2:
            return None
            
        accumulator = np.zeros((self.corner_size, self.corner_size))
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                intersection = self._find_intersection(lines[i], lines[j])
                if intersection:
                    x, y = intersection
                    if 0 <= x < self.corner_size and 0 <= y < self.corner_size:
                        y_indices, x_indices = np.ogrid[-y:self.corner_size-y, -x:self.corner_size-x]
                        mask = x_indices*x_indices + y_indices*y_indices <= 25
                        accumulator[mask] += 1

        if np.max(accumulator) > 0:
            y, x = np.unravel_index(accumulator.argmax(), accumulator.shape)
            return (x, y)
            
        return None

    def _detect_corners(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect corners in the image using edge detection and line intersection.
        
        Args:
            img: Input grayscale image of size target_size x target_size
            
        Returns:
            List of (x,y) corner coordinates in clockwise order from top-left.
            Returns empty list if corners cannot be detected.
        """
        # Edge detection
        edges = cv2.Canny(img, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        corner_points = []
        corners = [
            ('top_left', (0, 0)),
            ('top_right', (self.target_size - self.corner_size, 0)),
            ('bottom_left', (0, self.target_size - self.corner_size)),
            ('bottom_right', (self.target_size - self.corner_size, self.target_size - self.corner_size))
        ]
        
        for corner_name, (x, y) in corners:
            roi = dilated[y:y + self.corner_size, x:x + self.corner_size]
            lines = self._find_corner_lines(roi)
            intersection = self._find_corner_point(lines, corner_name)
            
            if intersection:
                global_x = x + intersection[0]
                global_y = y + intersection[1]
                corner_points.append((global_x, global_y))
        
        return corner_points

    def _order_points(self, pts: List[Tuple[int, int]]) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left.
        
        Args:
            pts: List of (x,y) corner coordinates
            
        Returns:
            numpy array of ordered points as float32 for perspective transform
        """
        pts = np.array(pts)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect

    def _calculate_skew_angles(self, corners: np.ndarray) -> Tuple[float, float]:
        """
        Calculate horizontal and vertical skew angles from corner positions.
        
        Args:
            corners: Ordered corner points array
            
        Returns:
            Tuple of (horizontal_angle, vertical_angle) in degrees
        """
        # Calculate horizontal skew (top edge)
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        horizontal_angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate vertical skew (left edge)
        dx = corners[3][0] - corners[0][0]
        dy = corners[3][1] - corners[0][1]
        vertical_angle = np.degrees(np.arctan2(dx, dy))
        
        return horizontal_angle, vertical_angle

    def _calculate_corner_displacements(self, src_corners: np.ndarray, dst_corners: np.ndarray) -> Tuple[float, float]:
        """
        Calculate average and maximum corner displacements.
        
        Args:
            src_corners: Original corner positions
            dst_corners: Target corner positions
            
        Returns:
            Tuple of (average_displacement, max_displacement) in pixels
        """
        distances = np.sqrt(np.sum((src_corners - dst_corners) ** 2, axis=1))
        return float(np.mean(distances)), float(np.max(distances))

    def detect(self, image: np.ndarray) -> Optional[SpikeDetectorResults]:
        """
        Detect corner spikes and correct image perspective distortion.
        
        Args:
            image: Grayscale input image matching target_size dimensions
                
        Returns:
            SpikeDetectorResults object containing corrected image and metrics
            if successful, None if corner detection fails
                
        Raises:
            ValueError: If input image dimensions don't match target_size
            
        Note:
            Input image should be pre-processed (grayscale, normalized contrast)
        """
        if image.shape != (self.target_size, self.target_size):
            raise ValueError(f"Input image must be {self.target_size}x{self.target_size}")
        
        # Detect corners
        corner_points = self._detect_corners(image)
        if len(corner_points) != 4:
            return None
        
        # Order corners
        src_corners = self._order_points(corner_points)
        
        # Define target corner positions
        dst_corners = np.array([
            [0, 0],
            [self.target_size-1, 0],
            [self.target_size-1, self.target_size-1],
            [0, self.target_size-1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Apply transform
        corrected = cv2.warpPerspective(image, transform_matrix, 
                                     (self.target_size, self.target_size))
        
        # Calculate metrics
        avg_displacement, max_displacement = self._calculate_corner_displacements(
            src_corners, dst_corners)
        skew_angles = self._calculate_skew_angles(src_corners)
        
        return SpikeDetectorResults(
            corrected_image=corrected,
            source_corners=src_corners,
            target_corners=dst_corners,
            transform_matrix=transform_matrix,
            average_corner_displacement=avg_displacement,
            max_corner_displacement=max_displacement,
            skew_angles=skew_angles
        )
