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
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from annuli_decoder import decode_annuli_rotation
from decoder.data_decoder import decode_data
from spike_detector import SpikeDetector
from finder_decoder import FinderDecoder

class SIFTDetector6DoF:
    """
    A tag detector that estimates 6-DoF pose using SIFT features and homography matching.
    
    The detector processes images to find tags and determine their position and orientation
    in 3D space. It uses a multi-stage process to improve accuracy:
    1. SIFT features are matched between the input image and a template
    2. A homography is computed to correct perspective distortion
    3. The image is rectified and checked with finder patterns
    4. If needed, spike detection corrects corner distortions
    5. Tag orientation is determined using annuli patterns
    6. The data grid is decoded to extract tag information
    
    Attributes:
        sift: OpenCV SIFT feature detector instance
        matcher: FLANN-based feature matcher configured with:
            - 5 randomized kd-trees for feature organization
            - 50 checks per search for match quality
    """

    def __init__(self):
        """
        Initializes SIFT detector and FLANN matcher.
        
        Uses FLANN matcher with randomized kd-trees for efficient feature matching:
        - 5 trees for feature organization
        - 50 checks per search balances speed and match quality
        """
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def detect(self, image: np.ndarray, 
               camera_matrix: np.ndarray, 
               dist_coeffs: np.ndarray, 
               debug_info: bool = False,
               save_imgs: bool = False,
               debug_directory: Optional[str] = None) -> Optional[Dict]:
        """
        Detects a tag in an image and estimates its 6-DoF pose.
        
        Args:
            image: Input image (grayscale or BGR)
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            debug_info: Whether to include extra debug data in results
            save_imgs: Whether to save debug visualizations
            debug_directory: Output directory for debug images
            
        Returns:
            Results dictionary or None if detection fails:
            {
                'success': bool indicating detection success
                'position': [x, y, z] translation vector in meters
                'rotation': [x, y, z, w] rotation as quaternion
                'corners': [[x, y], ...] detected corner positions
                'num_features': Number of SIFT matches used
                'num_inliers': Number of homography inliers
                'detection_time_ms': Total processing time
                'timing_stats': Timing breakdown for each step
                'rectified_image': Tag image after perspective correction
                'annuli_angle': Detected tag rotation
                'tag_data': Decoded information from tag
                'spike_detector_results': Corner correction results
                'finder_stats': Finder pattern analysis data
            }
            
        Raises:
            ValueError: If the template image tag3_blank_360.png isn't found
            
        The detector will attempt several refinement steps if initial detection
        isn't sufficiently confident. The spike detector and additional SIFT
        matching helps handle cases with perspective distortion.
        """        
        total_start_time = time.time()
        timing_stats = {}
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Load template
        template = cv2.imread("tag3_blank_360.png", cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError("Could not load template image")

        # Initial SIFT detection
        sift_start = time.time()
        kp1, des1 = self.sift.detectAndCompute(template, None)
        kp2, des2 = self.sift.detectAndCompute(gray, None)
        
        if save_imgs and debug_directory:
            self._save_keypoints(template, kp1, "Template Keypoints",
                            os.path.join(debug_directory, '1_template_keypoints.png'))
            self._save_keypoints(gray, kp2, "Image Keypoints",
                            os.path.join(debug_directory, '2_image_keypoints.png'))
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
            
        # Match features
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if save_imgs and debug_directory:
            initial_mask = np.ones(len(good_matches), dtype=bool)
            self._save_matches(template, kp1, gray, kp2,
                            good_matches, initial_mask,
                            "Initial SIFT Matches",
                            os.path.join(debug_directory, '3_initial_matches.png'))
                
        if len(good_matches) < 4:
            return None
            
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate initial homography
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None

        timing_stats['initial_sift'] = (time.time() - sift_start) * 1000

        try:
            # Initial pose estimation
            pose_start = time.time()
            
            # Decompose homography matrix
            _, Rs, ts, ns = cv2.decomposeHomographyMat(H, camera_matrix)
            scores = [np.abs(np.dot(n.flatten(), np.array([0, 0, 1]))) for n in ns]
            best_idx = np.argmax(scores)
            
            R_initial = Rs[best_idx]
            t_initial = ts[best_idx]
            
            # First get real corners
            h, w = template.shape
            template_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
            corners_2d = cv2.perspectiveTransform(template_corners, H).reshape(-1, 2)
            
            # Add padding to the corners in image space
            padding = 0.235  # 50% padding
            center = np.mean(corners_2d, axis=0)
            padded_corners = []
            for corner in corners_2d:
                vector = corner - center
                padded_corner = center + vector * (1 + padding)
                padded_corners.append(padded_corner)
            padded_corners = np.array(padded_corners)

            # Create larger rectified image
            target_size = (int(360 * (1 + 2*padding)), int(360 * (1 + 2*padding)))
            dst_corners = np.array([
                [0, 0],
                [target_size[0]-1, 0],
                [target_size[0]-1, target_size[1]-1],
                [0, target_size[1]-1]
            ], dtype=np.float32)
            
            H_rect = cv2.getPerspectiveTransform(padded_corners, dst_corners)
            rectified_wide = cv2.warpPerspective(gray, H_rect, target_size)
            
            timing_stats['initial_pose'] = (time.time() - pose_start) * 1000

            if save_imgs and debug_directory:
                cv2.imwrite(os.path.join(debug_directory, '4_wide_rectified.png'), rectified_wide)
                debug_corners = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Draw original corners in blue
                for i, corner in enumerate(corners_2d):
                    cv2.circle(debug_corners, tuple(corner.astype(int)), 5, (255,0,0), -1)
                    cv2.putText(debug_corners, f"O{i}", tuple(corner.astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # Draw padded corners in red
                for i, corner in enumerate(padded_corners):
                    cv2.circle(debug_corners, tuple(corner.astype(int)), 5, (0,0,255), -1)
                    cv2.putText(debug_corners, f"P{i}", tuple(corner.astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imwrite(os.path.join(debug_directory, '5_detected_corners.png'), debug_corners)

            # Run refinement SIFT on the wider rectified image
            refine_start = time.time()
            kp_template_refine, des_template_refine = self.sift.detectAndCompute(template, None)
            kp_wide_refine, des_wide_refine = self.sift.detectAndCompute(rectified_wide, None)

            if save_imgs and debug_directory:
                self._save_keypoints(template, kp_template_refine, "Template Keypoints (Refinement)",
                                os.path.join(debug_directory, '6_template_keypoints_refine.png'))
                self._save_keypoints(rectified_wide, kp_wide_refine, "Wide Image Keypoints (Refinement)",
                                os.path.join(debug_directory, '7_wide_keypoints_refine.png'))

            if des_template_refine is not None and des_wide_refine is not None:
                raw_matches_refine = self.matcher.knnMatch(des_template_refine, des_wide_refine, k=2)
                good_matches_refine = []
                for m, n in raw_matches_refine:
                    if m.distance < 0.7 * n.distance:
                        good_matches_refine.append(m)

                if save_imgs and debug_directory:
                    refine_mask = np.ones(len(good_matches_refine), dtype=bool)
                    self._save_matches(template, kp_template_refine, rectified_wide, kp_wide_refine,
                                    good_matches_refine, refine_mask,
                                    "Refinement SIFT Matches",
                                    os.path.join(debug_directory, '8_refinement_matches.png'))

                if len(good_matches_refine) >= 4:
                    src_pts_refine = np.float32([kp_template_refine[m.queryIdx].pt for m in good_matches_refine]).reshape(-1, 1, 2)
                    dst_pts_refine = np.float32([kp_wide_refine[m.trainIdx].pt for m in good_matches_refine]).reshape(-1, 1, 2)
                    H_refine, mask_refine = cv2.findHomography(src_pts_refine, dst_pts_refine, cv2.RANSAC, 3.0)

                    if H_refine is not None:
                        if save_imgs and debug_directory:
                            # Save visualization of the refinement homography inliers
                            inlier_matches = [match for match, inlier in zip(good_matches_refine, mask_refine.ravel()) if inlier]
                            inlier_mask = np.ones(len(inlier_matches), dtype=bool)
                            self._save_matches(template, kp_template_refine, rectified_wide, kp_wide_refine,
                                            inlier_matches, inlier_mask,
                                            "Refinement SIFT Inliers",
                                            os.path.join(debug_directory, '9_refinement_inliers.png'))

                        # Use template corners for source points
                        h, w = template.shape
                        template_corners = np.float32([
                            [0, 0],
                            [w-1, 0],
                            [w-1, h-1],
                            [0, h-1]
                        ]).reshape(-1, 1, 2)
                    
                        # Transform corners with H_refine
                        dst_corners = cv2.perspectiveTransform(template_corners, H_refine)
                        
                        # Create homography for final 360x360 image
                        final_corners = np.array([[0, 0], [359, 0], [359, 359], [0, 359]], dtype=np.float32)
                        H_final = cv2.getPerspectiveTransform(dst_corners.reshape(-1, 2), final_corners)
                        rectified_test = cv2.warpPerspective(rectified_wide, H_final, (360, 360))
                        
                        if save_imgs and debug_directory:
                            cv2.imwrite(os.path.join(debug_directory, '10_refined_rectified.png'), rectified_test)
                    else:
                        pad_pixels = int(360 * padding)
                        rectified_test = rectified_wide[pad_pixels:pad_pixels+360, pad_pixels:pad_pixels+360]
                        if save_imgs and debug_directory:
                            cv2.imwrite(os.path.join(debug_directory, '10_refined_rectified.png'), rectified_test)
                else:
                    pad_pixels = int(360 * padding)
                    rectified_test = rectified_wide[pad_pixels:pad_pixels+360, pad_pixels:pad_pixels+360]
                    if save_imgs and debug_directory:
                        cv2.imwrite(os.path.join(debug_directory, '10_refined_rectified.png'), rectified_test)
            else:
                pad_pixels = int(360 * padding)
                rectified_test = rectified_wide[pad_pixels:pad_pixels+360, pad_pixels:pad_pixels+360]
                if save_imgs and debug_directory:
                    cv2.imwrite(os.path.join(debug_directory, '10_refined_rectified.png'), rectified_test)

            timing_stats['refinement_sift'] = (time.time() - refine_start) * 1000

            # Check finder patterns using the refined image
            finder_start = time.time()
            finder_decoder = FinderDecoder()
            initial_scores, initial_confidence, initial_debug = finder_decoder.analyze_finder_patterns(rectified_test)
            timing_stats['finder_patterns'] = (time.time() - finder_start) * 1000
            
            finder_stats = {
                'initial_confidence': initial_confidence,
                'initial_pattern_scores': initial_scores,
                'needed_correction': False,
                'corrected_confidence': None,
                'corrected_pattern_scores': None
            }
            
            if save_imgs and debug_directory:
                cv2.imwrite(os.path.join(debug_directory, '11_finder_patterns_initial.png'), initial_debug)

            spike_results = None
            # Only use spike detector if finder patterns don't pass
            if initial_confidence < 90:
                spike_start = time.time()
                finder_stats['needed_correction'] = True
                spike_detector = SpikeDetector()
                spike_results = spike_detector.detect(rectified_test)
                timing_stats['spike_detector'] = (time.time() - spike_start) * 1000
                
                if spike_results:
                    rectified_test = spike_results.corrected_image
                    if save_imgs and debug_directory:
                        cv2.imwrite(os.path.join(debug_directory, '12_spike_detector_debug.png'), spike_results.debug_image)
                        cv2.imwrite(os.path.join(debug_directory, '13_spike_detector_rectified.png'), rectified_test)
                    
                    # Recheck finder patterns after spike correction
                    finder2_start = time.time()
                    corrected_scores, corrected_confidence, corrected_debug = finder_decoder.analyze_finder_patterns(rectified_test)
                    timing_stats['finder_patterns_2'] = (time.time() - finder2_start) * 1000
                    
                    finder_stats['corrected_confidence'] = corrected_confidence
                    finder_stats['corrected_pattern_scores'] = corrected_scores
                    
                    if save_imgs and debug_directory:
                        cv2.imwrite(os.path.join(debug_directory, '14_finder_patterns_after_spike.png'), corrected_debug)

            # Check orientation with annuli decoder
            rectified_corners = np.array([[0, 0], [359, 0], [359, 359], [0, 359]])
            center_point = np.mean(rectified_corners, axis=0)
            
            annuli_start = time.time()
            annuli_angle, _, annuli_debug = decode_annuli_rotation(
                rectified_test,
                rectified_corners,
                center_point,
                debug_info=debug_info
            )
            timing_stats['annuli_decoder'] = (time.time() - annuli_start) * 1000

            if save_imgs and debug_directory:
                cv2.imwrite(os.path.join(debug_directory, '15_annuli_detection.png'), annuli_debug)

            final_pose_start = time.time()
            if annuli_angle is not None:
                correction_angle = round(annuli_angle / 90) * 90 % 360
                correction_steps = (correction_angle // 90) % 4
                final_idx = (best_idx + correction_steps) % 4
                R = Rs[final_idx]
                t = ts[final_idx]
            else:
                R = R_initial
                t = t_initial


            # Convert from tag->camera to camera->tag coordinates
            t = -t
            R = R.T
            q = self._rotation_matrix_to_quaternion(R)
            
            timing_stats['final_pose'] = (time.time() - final_pose_start) * 1000

            # Decode tag data
            data_start = time.time()
            tag_data = decode_data(
                rectified_test,
                rectified_corners,
                center_point,
                annuli_angle if annuli_angle is not None else 0.0,
                debug_info=debug_info
            )
            timing_stats['data_decoder'] = (time.time() - data_start) * 1000

            if save_imgs and debug_directory and tag_data and 'debug_image' in tag_data:
                grid_vis = tag_data['debug_image']
                if grid_vis is not None:
                    cv2.imwrite(os.path.join(debug_directory, '16_grid_debug.png'), grid_vis)

            if tag_data:
                physical_size_mm = 36.0 / tag_data['scale']
                tag_size_meters = physical_size_mm / 1000.0
                t = t * tag_size_meters

            if save_imgs and debug_directory:
                final_vis = rectified_test.copy() if len(rectified_test.shape) == 2 else cv2.cvtColor(rectified_test, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(debug_directory, '17_final_rectified.png'), final_vis)

            # Calculate total time
            timing_stats['total'] = (time.time() - total_start_time) * 1000

            return {
                'success': True,
                'position': t.ravel().tolist(),
                'rotation': q.tolist(),
                'corners': corners_2d.tolist(),
                'num_features': len(good_matches),
                'num_inliers': int(np.sum(inlier_mask)),
                'detection_time_ms': timing_stats['total'],
                'timing_stats': timing_stats,
                'rectified_image': rectified_test,
                'annuli_angle': float(annuli_angle) if annuli_angle is not None else None,
                'tag_data': tag_data if tag_data else None,
                'spike_detector_results': spike_results if spike_results else None,
                'finder_stats': finder_stats
            }

        except Exception as e:
            return None

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Converts a 3x3 rotation matrix to quaternion form.
        
        Uses a numerically stable method that handles edge cases and gimbal lock.
        The returned quaternion uses [x, y, z, w] convention.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            4D array containing quaternion as [x, y, z, w]
        """
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2,1] - R[1,2]) / S
            y = (R[0,2] - R[2,0]) / S
            z = (R[1,0] - R[0,1]) / S
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                w = (R[2,1] - R[1,2]) / S
                x = 0.25 * S
                y = (R[0,1] + R[1,0]) / S
                z = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                w = (R[0,2] - R[2,0]) / S
                x = (R[0,1] + R[1,0]) / S
                y = 0.25 * S
                z = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                w = (R[1,0] - R[0,1]) / S
                x = (R[0,2] + R[2,0]) / S
                y = (R[1,2] + R[2,1]) / S
                z = 0.25 * S
        return np.array([x, y, z, w])

    def _save_keypoints(self, image: np.ndarray,
                       keypoints: List[cv2.KeyPoint],
                       title: str,
                       save_path: str) -> None:
        """
        Creates a debug visualization of detected SIFT keypoints.
        
        The visualization shows each keypoint's position, scale, and orientation.
        Keypoints are drawn as circles with radius showing scale and lines
        indicating orientation.
        
        Args:
            image: Source image
            keypoints: List of detected keypoints to visualize
            title: Text to add at top of visualization
            save_path: Where to save the output image
        """
        keypoints_image = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.putText(keypoints_image, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imwrite(save_path, keypoints_image)

    def _save_matches(self, img1: np.ndarray,
                     kp1: List[cv2.KeyPoint],
                     img2: np.ndarray,
                     kp2: List[cv2.KeyPoint],
                     matches: List[cv2.DMatch],
                     mask: np.ndarray,
                     title: str,
                     save_path: str) -> None:
        """
        Creates a visualization of SIFT feature matches between two images.
        
        The visualization shows both images stacked vertically with lines
        connecting matched features. Valid matches (according to mask) are
        shown in green. Images are automatically resized to the same width
        and keypoints are scaled accordingly.
        
        Args:
            img1: First input image
            kp1: Keypoints from first image
            img2: Second input image  
            kp2: Keypoints from second image
            matches: List of feature matches
            mask: Boolean array marking valid matches
            title: Text to add at top of visualization
            save_path: Where to save the output image
        """
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()

        if len(img2.shape) == 2:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()

        # Ensure consistent sizes
        max_width = max(img1_color.shape[1], img2_color.shape[1])
        img1_resized = cv2.resize(img1_color, (max_width, 
                    int(img1_color.shape[0] * max_width / img1_color.shape[1])))
        img2_resized = cv2.resize(img2_color, (max_width, 
                    int(img2_color.shape[0] * max_width / img2_color.shape[1])))

        # Stack images vertically
        vis_img = np.vstack([img1_resized, img2_resized])

        # Calculate scaling factors
        scale_x1 = max_width / img1_color.shape[1]
        scale_y1 = img1_resized.shape[0] / img1_color.shape[0]
        scale_x2 = max_width / img2_color.shape[1]
        scale_y2 = img2_resized.shape[0] / img2_color.shape[0]

        # Draw matches
        for match, use_match in zip(matches, mask):
            if use_match:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx

                x1, y1 = kp1[img1_idx].pt
                x1 = int(x1 * scale_x1)
                y1 = int(y1 * scale_y1)

                x2, y2 = kp2[img2_idx].pt
                x2 = int(x2 * scale_x2)
                y2 = int(y2 * scale_y2) + img1_resized.shape[0]

                cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(vis_img, (x1, y1), 4, (255, 0, 0), -1)
                cv2.circle(vis_img, (x2, y2), 4, (255, 0, 0), -1)

        # Add title
        cv2.putText(vis_img, f"{title} - {np.sum(mask)} matches",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(save_path, vis_img)