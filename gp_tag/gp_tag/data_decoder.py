"""
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
import cv2
import numpy as np
import reedsolo
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

def normalize_tag_image(image: np.ndarray, corners: np.ndarray, angle: float, debug_info=False) -> np.ndarray:
    """
    Transform detected tag to normalized 360x360 image with standard orientation.
    
    Args:
        image: Input image
        corners: Four corner points
        angle: Tag rotation angle
        debug_info: Whether to enable debug visualization
        
    Returns:
        360x360 normalized grayscale image
    """
    if len(image.shape) > 2:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()
    
    dst_size = 360
    dst_corners = np.array([
        [0, 0],               # Top-left
        [dst_size-1, 0],      # Top-right
        [dst_size-1, dst_size-1],  # Bottom-right
        [0, dst_size-1]       # Bottom-left
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
    warped = cv2.warpPerspective(img, M, (dst_size, dst_size))
    
    return warped

def read_cell_with_debug(img: np.ndarray, x: int, y: int, start_x: int, start_y: int, 
                        U: int, debug_img: Optional[np.ndarray] = None) -> int:
    """
    Read a single cell value with optional debugging visualization.
    
    Args:
        img: Input grayscale image
        x, y: Cell coordinates in grid
        start_x, start_y: Grid origin coordinates
        U: Cell size in pixels
        debug_img: Optional debug image to draw on
        
    Returns:
        1 for black cells, 0 for white cells
    """
    cell_x = int(start_x + x * U)
    cell_y = int(start_y + y * U)
    
    center_x = cell_x + U//2
    center_y = cell_y + U//2
    
    value = 1 if img[center_y, center_x] < 128 else 0
    
    if debug_img is not None:
        cv2.rectangle(debug_img, 
                     (cell_x, cell_y),
                     (cell_x + U, cell_y + U),
                     (0, 255, 0), 1)
        cv2.circle(debug_img, (center_x, center_y), 1, (255, 255, 0), -1)
    
    return value

def decode_data(image: np.ndarray, corners: np.ndarray, center: np.ndarray, angle: float, 
                debug_info: bool = False) -> Optional[Dict[str, Union[int, float, list]]]:
    """
    Decode tag data including position, orientation and metadata.
    
    Data format:
    - Main grid (178 bits):
        - Latitude (35 bits): -90° to +90°
        - Longitude (36 bits): -180° to +180°
        - Altitude (25 bits): -10km to +10km
        - Quaternion (4x16 bits): -1 to +1 each
        - Accuracy (2 bits)
        - Scale (16 bits): 0 to 3.6m
    - Reserved data:
        - Tag ID (12 bits)
        - Version ID (4 bits)
    
    Args:
        image: Input image (grayscale or BGR)
        corners: Four corner points
        center: Center point coordinates
        angle: Tag rotation angle
        debug_info: Whether to enable debug visualization
        
    Returns:
        Dictionary containing decoded data or None if decoding fails
    """
    normalized = normalize_tag_image(image, corners, angle, debug_info)
    debug_vis = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    image_size = 360
    U = 10  # Unit size in normalized image
    origin_x = image_size // 2
    origin_y = image_size // 2
    grid_size_main = 21
    grid_size_full = 36

    grid_start_x = origin_x - (grid_size_main * U) // 2
    grid_start_y = origin_y - (grid_size_main * U) // 2
    full_grid_start_x = origin_x - (grid_size_full * U) // 2 - U/2
    full_grid_start_y = origin_y - (grid_size_full * U) // 2 - U/2

    # Draw grids
    for i in range(grid_size_main + 1):
        x = grid_start_x + i * U
        y = grid_start_y + i * U
        cv2.line(debug_vis, (int(grid_start_x), int(y)), 
                (int(grid_start_x + grid_size_main * U), int(y)), (0,255,0), 1)
        cv2.line(debug_vis, (int(x), int(grid_start_y)), 
                (int(x), int(grid_start_y + grid_size_main * U)), (0,255,0), 1)

    for i in range(grid_size_full + 1):
        x = full_grid_start_x + i * U
        y = full_grid_start_y + i * U
        cv2.line(debug_vis, (int(full_grid_start_x), int(y)), 
                (int(full_grid_start_x + grid_size_full * U), int(y)), (128,128,128), 1)
        cv2.line(debug_vis, (int(x), int(full_grid_start_y)), 
                (int(x), int(full_grid_start_y + grid_size_full * U)), (128,128,128), 1)

    # Add legend
    cv2.putText(debug_vis, "Grid Sample Points", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(debug_vis, "Yellow dot = Sample point", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # Read main grid first
    reserved_modules = set()
    for y in range(grid_size_main):
        for x in range(grid_size_main):
            if (x < 5 and y < 5) or \
               (x < 5 and y > grid_size_main-6) or \
               (x > grid_size_main-6 and y < 5) or \
               (x > grid_size_main-6 and y > grid_size_main-6) or \
               x == 5 or y == 5:
                reserved_modules.add((x, y))

    main_bits = []
    total_bits = 300  # 178 data bits + ECC
    bit_index = 0
    
    for x in range(grid_size_main - 1, -1, -1):
        if x == 6:  # Skip timing column
            continue
        for y in range(grid_size_main - 1, -1, -1):
            if (x, y) not in reserved_modules:
                if bit_index < total_bits:
                    bit = read_cell_with_debug(normalized, x, y, grid_start_x, grid_start_y, U, 
                                             debug_vis)
                    main_bits.append(bit)
                    bit_index += 1

    # Convert main bits to bytes
    all_bits_str = ''.join(str(bit) for bit in main_bits)
    all_bytes = int(all_bits_str, 2).to_bytes(len(main_bits)//8, 'big')

    data_bytes = int(np.ceil(178 / 8))  # 23 bytes
    ecc_bytes = int(np.ceil(data_bytes * 0.5))  # 12 bytes
    rs_main = reedsolo.RSCodec(ecc_bytes)

    try:
        # Try to decode main data first
        decoded_main = rs_main.decode(all_bytes)[0]
        bit_stream = ''.join(format(b, '08b') for b in decoded_main)
        bit_stream = bit_stream[-178:]  # Extract original data bits
        
        # Parse position data
        latitude = int(bit_stream[:35], 2) * (180 / (2**35 - 1)) - 90
        longitude = int(bit_stream[35:71], 2) * (360 / (2**36 - 1)) - 180
        altitude = int(bit_stream[71:96], 2) * (20000 / (2**25 - 1)) - 10000

        # Parse quaternion
        quaternion = []
        q_offset = 96
        for i in range(4):
            q_val = int(bit_stream[q_offset + i*16:q_offset + (i+1)*16], 2)
            q_val = (q_val * 2 / (2**16 - 1)) - 1
            quaternion.append(q_val)

        # Parse metadata
        accuracy = int(bit_stream[160:162], 2)
        scale = int(bit_stream[162:178], 2) * (3.6 / (2**16 - 1))

        # Main data decoded successfully, now try reserved data
        RESERVED_CELL_PAIRS = [
            ((15,32), (21,4)), ((16,32), (20,4)), ((17,32), (19,4)),
            ((18,32), (18,4)), ((19,32), (17,4)), ((20,32), (16,4)),
            ((21,32), (15,4)), ((14,31), (22,5)), ((15,31), (21,5)),
            ((16,31), (20,5)), ((17,31), (19,5)), ((18,31), (18,5)),
            ((19,31), (17,5)), ((20,31), (16,5)), ((21,31), (15,5)),
            ((22,31), (14,5)), ((17,30), (19,6)), ((18,30), (18,6)),
            ((19,30), (17,6)), ((4,15), (32,21)), ((4,16), (32,20)),
            ((4,17), (32,19)), ((4,18), (32,18)), ((4,19), (32,17)),
            ((4,20), (32,16)), ((4,21), (32,15)), ((5,14), (31,22)),
            ((5,15), (31,21)), ((5,16), (31,20)), ((5,17), (31,19)),
            ((5,18), (31,18)), ((5,19), (31,17)), ((5,20), (31,16)),
            ((5,21), (31,15)), ((5,22), (31,14)), ((6,17), (30,19)),
            ((6,18), (30,18)), ((6,19), (30,17))
        ]

        reserved_bits = []
        for pair in RESERVED_CELL_PAIRS:
            primary = pair[0]
            bit = read_cell_with_debug(normalized, primary[0], primary[1], 
                                     full_grid_start_x, full_grid_start_y, U, 
                                     debug_vis)
            reserved_bits.append(bit)

        reserved_bytes = bytearray()
        for i in range(0, len(reserved_bits), 8):
            byte = 0
            bits = reserved_bits[i:i+8]
            for bit in bits:
                byte = (byte << 1) | bit
            reserved_bytes.append(byte)

        rs_reserved = reedsolo.RSCodec(1)
        id_decode_success = False
        tag_id = None
        version_id = None

        try:
            decoded_reserved = rs_reserved.decode(reserved_bytes)[0]
            tag_id = (decoded_reserved[0] << 4) | (decoded_reserved[1] >> 4)
            version_id = decoded_reserved[1] & 0x0F
            id_decode_success = True
            
            cv2.putText(debug_vis, f"Tag ID: {tag_id}, Version: {version_id}", (10, 340), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(debug_vis, "ID Decode Successful", (10, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        except reedsolo.ReedSolomonError:
            cv2.putText(debug_vis, "ID Decode Failed", (10, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(debug_vis, f"Lat: {latitude:.6f}, Lon: {longitude:.6f}", (10, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return {
            'success': True,  # Main data decoded successfully
            'id_decode_success': id_decode_success,
            'tag_id': tag_id,
            'version_id': version_id,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'quaternion': quaternion,
            'accuracy': accuracy,
            'scale': scale,
            'debug_image': debug_vis
        }
        
    except reedsolo.ReedSolomonError:
        cv2.putText(debug_vis, "Main Data Decode Failed", (10, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return {
            'success': False,
            'id_decode_success': False,
            'debug_image': debug_vis,
            'failure': 'main_decode'
        }