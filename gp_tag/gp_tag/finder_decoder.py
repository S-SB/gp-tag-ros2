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
from typing import List, Tuple, Dict

class FinderDecoder:
    """
    A decoder for finding and analyzing finder patterns in images.
    
    Processes square images to locate and evaluate finder patterns in the corners,
    providing confidence scores and visualization. The patterns are used to validate
    tag orientation and positioning.
    
    The finder patterns follow a 5x5 grid pattern:
    - Outer border: all black (1)
    - Inner pattern: specific arrangement of black (1) and white (0)
    
    Pattern structure:
        1 1 1 1 1
        1 0 0 0 1
        1 0 1 0 1
        1 0 0 0 1
        1 1 1 1 1
    
    Attributes:
        img_size (int): Size of the square input image (360px)
        grid_size (int): Size of the main grid area (210px)
        grid_offset (int): Calculated offset from image edge to grid start
        cell_size (int): Size of each cell in the finder pattern (10px)
    """
    
    def __init__(self):
        """
        Initialize the FinderDecoder with standard dimensions.
        
        Sets up the decoder with these dimensions (in pixels):
        - 360x360 image size
        - 210x210 grid size
        - 10x10 cell size
        
        The grid_offset is calculated to center the grid in the image.
        """
        self.img_size = 360
        self.grid_size = 210
        self.grid_offset = (self.img_size - self.grid_size) // 2
        self.cell_size = 10
        
    def check_finder_grid(self, img: np.ndarray, x: int, y: int) -> Tuple[float, np.ndarray]:
        """
        Check a single finder pattern at the specified coordinates.
        
        Analyzes a 5x5 cell region for the presence of a valid finder pattern,
        computing a confidence score and generating a debug visualization.
        
        Args:
            img: Grayscale input image
            x: X-coordinate of top-left corner of finder pattern
            y: Y-coordinate of top-left corner of finder pattern
            
        Returns:
            Tuple containing:
                - Confidence score (0-100%)
                - Debug visualization image with pattern analysis
                
        Note:
            - Uses 3x3 sampling window at center of each cell
            - Color coding in debug image:
                - Green: Correctly matched cells
                - Red: Incorrectly matched cells
                - Blue dots: Sampling points
        """
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        correct_cells = 0
        
        # Expected finder pattern (1=black, 0=white)
        expected = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        
        for i in range(5):
            for j in range(5):
                center_x = x + (j * self.cell_size) + (self.cell_size // 2)
                center_y = y + (i * self.cell_size) + (self.cell_size // 2)
                
                roi = img[center_y-1:center_y+2, center_x-1:center_x+2]
                value = np.mean(roi) < 127
                
                if value == expected[i][j]:
                    correct_cells += 1
                    cell_color = (0, 255, 0)  # Green for correct
                else:
                    cell_color = (0, 0, 255)  # Red for incorrect
                
                # Draw cell rectangle
                cv2.rectangle(debug_img, 
                            (x + j * self.cell_size, y + i * self.cell_size),
                            (x + (j+1) * self.cell_size, y + (i+1) * self.cell_size),
                            cell_color, 1)
                
                # Draw sample point
                cv2.circle(debug_img, (center_x, center_y), 1, (255, 0, 0), -1)
                    
        confidence = (correct_cells / 25) * 100
        
        # Draw overall pattern box with confidence score
        cv2.rectangle(debug_img, 
                     (x, y), 
                     (x + 5 * self.cell_size, y + 5 * self.cell_size),
                     (0, 255, 0) if confidence > 80 else (0, 0, 255), 2)
        
        # Add confidence text with background
        text = f"{confidence:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(debug_img, 
                     (x, y-15), 
                     (x + text_w + 4, y-2), 
                     (0, 0, 0), -1)
        cv2.putText(debug_img, text, 
                   (x + 2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (0, 255, 0) if confidence > 80 else (0, 0, 255), 1)
                     
        return confidence, debug_img

    def analyze_finder_patterns(self, img: np.ndarray) -> Tuple[List[Tuple[str, float]], float, np.ndarray]:
        """
        Analyze all four finder patterns in the image.
        
        Processes each corner of the grid to detect and evaluate finder patterns,
        providing individual and overall confidence scores with visualization.
        
        Args:
            img: Grayscale input image of size img_size x img_size
            
        Returns:
            Tuple containing:
                - List of (corner_name, confidence_score) pairs
                - Average confidence across all patterns (0-100%)
                - Debug visualization showing all pattern analyses
                
        Note:
            - Debug visualization uses color coding:
                - Green: High confidence (>80%)
                - Red: Low confidence (≤80%)
            - Corner names are: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
            
        Example:
            ```python
            decoder = FinderDecoder()
            scores, avg_confidence, debug_img = decoder.analyze_finder_patterns(image)
            if avg_confidence > 80:
                print("Valid finder patterns detected")
            cv2.imshow("Debug", debug_img)
            ```
        """
        debug_combined = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        corners = [
            ('top_left', (self.grid_offset, self.grid_offset)),
            ('top_right', (self.grid_offset + self.grid_size - 5*self.cell_size, self.grid_offset)),
            ('bottom_left', (self.grid_offset, self.grid_offset + self.grid_size - 5*self.cell_size)),
            ('bottom_right', (self.grid_offset + self.grid_size - 5*self.cell_size, 
                            self.grid_offset + self.grid_size - 5*self.cell_size))
        ]
        
        pattern_scores = []
        total_confidence = 0
        
        for corner_name, (x, y) in corners:
            confidence, pattern_debug = self.check_finder_grid(img, x, y)
            pattern_scores.append((corner_name, confidence))
            total_confidence += confidence
            
            # Copy the pattern debug into the combined image
            pattern_region = pattern_debug[y:y + 5*self.cell_size, x:x + 5*self.cell_size]
            debug_combined[y:y + 5*self.cell_size, x:x + 5*self.cell_size] = pattern_region
        
        # Calculate average confidence
        avg_confidence = total_confidence / 4
        
        # Add overall confidence score to image
        text = f"Overall: {avg_confidence:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_combined, 
                     (5, 5), 
                     (5 + text_w + 4, 20), 
                     (0, 0, 0), -1)
        cv2.putText(debug_combined, text, 
                   (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if avg_confidence > 80 else (0, 0, 255), 1)
        
        return pattern_scores, avg_confidence, debug_combined