import cv2
import numpy as np

class ARVisualizer:
    """
    Handles rendering of AR elements onto the image.
    """
    
    def __init__(self):
        pass
        
    def draw_ar_path(self, image, points, color=(0, 255, 255), thickness=3):
        """
        Draw the projected path on the image.
        
        Args:
            image: Input image (BGR)
            points: List of (u, v) tuples
            color: BGR color tuple (default Yellow)
            thickness: Line thickness
            
        Returns:
            image: Image with AR overlay
        """
        if not points or len(points) < 2:
            return image
            
        # Convert points to integer numpy array
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Create an overlay for transparency
        overlay = image.copy()
        
        # Draw smooth curve
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # Draw arrow head at the end
        end_pt = tuple(points[-1])
        prev_pt = tuple(points[-2])
        
        # Calculate direction
        dx = end_pt[0] - prev_pt[0]
        dy = end_pt[1] - prev_pt[1]
        angle = np.arctan2(dy, dx)
        
        # Arrow size
        arrow_len = 20
        angle_offset = np.pi / 6 # 30 degrees
        
        p1 = (int(end_pt[0] - arrow_len * np.cos(angle - angle_offset)),
              int(end_pt[1] - arrow_len * np.sin(angle - angle_offset)))
        p2 = (int(end_pt[0] - arrow_len * np.cos(angle + angle_offset)),
              int(end_pt[1] - arrow_len * np.sin(angle + angle_offset)))
              
        cv2.line(overlay, tuple(map(int, end_pt)), p1, color, thickness, cv2.LINE_AA)
        cv2.line(overlay, tuple(map(int, end_pt)), p2, color, thickness, cv2.LINE_AA)
        
        # Blend overlay
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image

    def draw_info(self, image, text):
        """Draw status text"""
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image
