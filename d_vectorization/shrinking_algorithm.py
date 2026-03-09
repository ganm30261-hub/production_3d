"""
Shrinking Algorithm for Wall Rectangle Fitting

Implements the shrinking rectangle fitting algorithm from the Fraunhofer HHI paper:
"Automatic Reconstruction of Semantic 3D Models from 2D Floor Plans" (MVA 2023)

The algorithm iteratively shrinks an initial bounding box to fit wall contours,
producing axis-aligned rectangles that represent wall segments.

Key Features:
- Iterative shrinking from bounding box to tight fit
- IoU-based quality metric
- Support for rotated rectangles
- Multi-scale fitting for complex shapes
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FitMethod(Enum):
    """Rectangle fitting methods."""
    SHRINKING = "shrinking"      # Paper's shrinking algorithm
    MIN_AREA_RECT = "min_area"   # OpenCV minAreaRect
    HYBRID = "hybrid"            # Best of both


@dataclass
class ShrinkingConfig:
    """Configuration for Shrinking Algorithm."""
    
    # Shrinking parameters
    shrink_step: float = 1.0  # Pixels to shrink per iteration
    max_iterations: int = 1000  # Maximum shrinking iterations
    convergence_threshold: float = 0.001  # Stop when change < threshold
    
    # Quality thresholds
    min_iou: float = 0.5  # Minimum IoU to accept rectangle
    min_aspect_ratio: float = 0.05  # Minimum width/height ratio
    max_aspect_ratio: float = 20.0  # Maximum width/height ratio
    
    # Rectangle constraints
    min_width: float = 3.0  # Minimum rectangle width
    min_height: float = 3.0  # Minimum rectangle height
    min_area: float = 20.0  # Minimum rectangle area
    
    # Fitting options
    method: FitMethod = FitMethod.SHRINKING
    allow_rotation: bool = True  # Allow rotated rectangles
    angle_step: float = 1.0  # Angle increment for rotation search
    
    # Multi-scale options
    use_multiscale: bool = False
    scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])


@dataclass
class FittedRectangle:
    """Represents a fitted rectangle."""
    
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float  # Rotation angle in degrees
    
    iou: float = 0.0  # Intersection over Union with original contour
    coverage: float = 0.0  # Percentage of contour covered
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point."""
        return (self.center_x, self.center_y)
    
    @property
    def size(self) -> Tuple[float, float]:
        """Return (width, height)."""
        return (self.width, self.height)
    
    @property
    def area(self) -> float:
        """Return rectangle area."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Return aspect ratio (smaller/larger)."""
        if self.width == 0 or self.height == 0:
            return 0.0
        return min(self.width, self.height) / max(self.width, self.height)
    
    @property
    def corners(self) -> np.ndarray:
        """Return 4 corner points as numpy array."""
        # Create rectangle centered at origin
        w, h = self.width / 2, self.height / 2
        corners = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ], dtype=np.float32)
        
        # Rotate
        angle_rad = np.radians(self.angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)
        
        corners = corners @ rotation.T
        
        # Translate to center
        corners[:, 0] += self.center_x
        corners[:, 1] += self.center_y
        
        return corners
    
    @property
    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return wall segment endpoints (midpoints of shorter edges).
        
        For wall rectangles, returns the centerline.
        """
        corners = self.corners
        
        # Determine which edges are shorter (wall ends)
        edge1_len = np.linalg.norm(corners[1] - corners[0])
        edge2_len = np.linalg.norm(corners[2] - corners[1])
        
        if edge1_len < edge2_len:
            # Edges 0-1 and 2-3 are shorter
            p1 = (corners[0] + corners[1]) / 2
            p2 = (corners[2] + corners[3]) / 2
        else:
            # Edges 1-2 and 3-0 are shorter
            p1 = (corners[1] + corners[2]) / 2
            p2 = (corners[3] + corners[0]) / 2
        
        return (tuple(p1), tuple(p2))
    
    @property
    def thickness(self) -> float:
        """Return wall thickness (shorter dimension)."""
        return min(self.width, self.height)
    
    @property
    def length(self) -> float:
        """Return wall length (longer dimension)."""
        return max(self.width, self.height)
    
    def to_cv_rect(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Convert to OpenCV RotatedRect format."""
        return ((self.center_x, self.center_y), (self.width, self.height), self.angle)
    
    @classmethod
    def from_cv_rect(
        cls,
        cv_rect: Tuple[Tuple[float, float], Tuple[float, float], float],
        iou: float = 0.0
    ) -> 'FittedRectangle':
        """Create from OpenCV RotatedRect."""
        center, size, angle = cv_rect
        return cls(
            center_x=center[0],
            center_y=center[1],
            width=size[0],
            height=size[1],
            angle=angle,
            iou=iou
        )


class ShrinkingAlgorithm:
    """
    Shrinking rectangle fitting algorithm from Fraunhofer HHI paper.
    
    The algorithm works by:
    1. Starting with the bounding box of a contour
    2. Iteratively shrinking each edge inward
    3. Measuring IoU at each step
    4. Stopping when IoU stops improving or constraints are violated
    
    Example:
        algorithm = ShrinkingAlgorithm()
        rect, iou = algorithm.fit_rectangle(contour)
        if iou > 0.7:
            print(f"Good fit: {rect.width}x{rect.height} at {rect.angle}°")
    """
    
    def __init__(self, config: Optional[ShrinkingConfig] = None):
        """
        Initialize the shrinking algorithm.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or ShrinkingConfig()
    
    def fit_rectangle(
        self,
        contour: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dominant_angle: Optional[float] = None
    ) -> Tuple[FittedRectangle, float]:
        """
        Fit a rectangle to a contour using the shrinking algorithm.
        
        Args:
            contour: Input contour (Nx1x2 or Nx2 array)
            mask: Optional binary mask for more accurate IoU
            dominant_angle: If provided, constrain to this angle
            
        Returns:
            Tuple of (FittedRectangle, IoU score)
        """
        # Ensure proper contour shape
        contour = self._normalize_contour(contour)
        
        if len(contour) < 4:
            logger.warning("Contour has fewer than 4 points")
            return self._fallback_rect(contour), 0.0
        
        # Choose fitting method
        if self.config.method == FitMethod.MIN_AREA_RECT:
            return self._fit_min_area_rect(contour, mask, dominant_angle)
        elif self.config.method == FitMethod.HYBRID:
            return self._fit_hybrid(contour, mask, dominant_angle)
        else:
            return self._fit_shrinking(contour, mask, dominant_angle)
    
    def fit_rectangles(
        self,
        contours: List[np.ndarray],
        mask: Optional[np.ndarray] = None,
        dominant_angle: Optional[float] = None,
        min_iou: Optional[float] = None
    ) -> List[Tuple[FittedRectangle, float]]:
        """
        Fit rectangles to multiple contours.
        
        Args:
            contours: List of contours
            mask: Optional binary mask
            dominant_angle: Constrain angles to this direction
            min_iou: Minimum IoU threshold (uses config default if None)
            
        Returns:
            List of (FittedRectangle, IoU) tuples for accepted rectangles
        """
        min_iou = min_iou if min_iou is not None else self.config.min_iou
        results = []
        
        for contour in contours:
            try:
                rect, iou = self.fit_rectangle(contour, mask, dominant_angle)
                
                # Filter by quality
                if iou >= min_iou and self._is_valid_rect(rect):
                    results.append((rect, iou))
            except Exception as e:
                logger.warning(f"Failed to fit rectangle: {e}")
                continue
        
        return results
    
    def _fit_shrinking(
        self,
        contour: np.ndarray,
        mask: Optional[np.ndarray],
        dominant_angle: Optional[float]
    ) -> Tuple[FittedRectangle, float]:
        """
        Core shrinking algorithm implementation.
        
        Paper method: iteratively shrink bounding box to maximize IoU.
        """
        # Get initial bounding box
        if self.config.allow_rotation:
            initial_rect = cv2.minAreaRect(contour)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            initial_rect = ((x + w/2, y + h/2), (w, h), 0)
        
        # Override angle if dominant angle provided
        if dominant_angle is not None:
            center, size, _ = initial_rect
            initial_rect = (center, size, dominant_angle)
        
        # Create contour mask if not provided
        if mask is None:
            mask = self._create_contour_mask(contour)
        
        # Start shrinking
        best_rect = initial_rect
        best_iou = self._compute_iou(initial_rect, contour, mask)
        
        center, (w, h), angle = initial_rect
        
        for iteration in range(self.config.max_iterations):
            improved = False
            
            # Try shrinking each dimension
            for dim in [0, 1]:  # Width, Height
                for direction in [-1, 1]:  # Shrink/grow
                    test_size = [w, h]
                    test_size[dim] += direction * self.config.shrink_step
                    
                    if test_size[dim] < (self.config.min_width if dim == 0 else self.config.min_height):
                        continue
                    
                    test_rect = (center, tuple(test_size), angle)
                    test_iou = self._compute_iou(test_rect, contour, mask)
                    
                    if test_iou > best_iou + self.config.convergence_threshold:
                        best_iou = test_iou
                        best_rect = test_rect
                        w, h = test_size
                        improved = True
                        break
            
            # Try shifting center
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_center = (center[0] + dx * self.config.shrink_step,
                              center[1] + dy * self.config.shrink_step)
                test_rect = (test_center, (w, h), angle)
                test_iou = self._compute_iou(test_rect, contour, mask)
                
                if test_iou > best_iou + self.config.convergence_threshold:
                    best_iou = test_iou
                    best_rect = test_rect
                    center = test_center
                    improved = True
            
            if not improved:
                break
        
        # Create result
        result = FittedRectangle.from_cv_rect(best_rect, best_iou)
        result.coverage = self._compute_coverage(best_rect, contour)
        
        return result, best_iou
    
    def _fit_min_area_rect(
        self,
        contour: np.ndarray,
        mask: Optional[np.ndarray],
        dominant_angle: Optional[float]
    ) -> Tuple[FittedRectangle, float]:
        """Fit using OpenCV's minAreaRect."""
        rect = cv2.minAreaRect(contour)
        
        # Override angle if specified
        if dominant_angle is not None:
            center, size, _ = rect
            rect = (center, size, dominant_angle)
        
        # Create mask if needed
        if mask is None:
            mask = self._create_contour_mask(contour)
        
        iou = self._compute_iou(rect, contour, mask)
        
        result = FittedRectangle.from_cv_rect(rect, iou)
        result.coverage = self._compute_coverage(rect, contour)
        
        return result, iou
    
    def _fit_hybrid(
        self,
        contour: np.ndarray,
        mask: Optional[np.ndarray],
        dominant_angle: Optional[float]
    ) -> Tuple[FittedRectangle, float]:
        """
        Hybrid method: try both and return best.
        """
        # Try shrinking
        rect1, iou1 = self._fit_shrinking(contour, mask, dominant_angle)
        
        # Try minAreaRect
        rect2, iou2 = self._fit_min_area_rect(contour, mask, dominant_angle)
        
        # Return best
        if iou1 >= iou2:
            return rect1, iou1
        else:
            return rect2, iou2
    
    def _compute_iou(
        self,
        rect: Tuple,
        contour: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute Intersection over Union between rectangle and contour.
        """
        # Get rectangle mask
        rect_mask = np.zeros_like(mask)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(rect_mask, [box], 255)
        
        # Get contour mask
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour.astype(np.int32)], -1, 255, -1)
        
        # Compute IoU
        intersection = np.logical_and(rect_mask > 0, contour_mask > 0).sum()
        union = np.logical_or(rect_mask > 0, contour_mask > 0).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_coverage(
        self,
        rect: Tuple,
        contour: np.ndarray
    ) -> float:
        """Compute what percentage of contour is covered by rectangle."""
        # Get bounding dimensions
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create masks
        size = (h + 10, w + 10)
        offset = (-x + 5, -y + 5)
        
        rect_mask = np.zeros(size, dtype=np.uint8)
        contour_mask = np.zeros(size, dtype=np.uint8)
        
        # Shift rectangle
        shifted_rect = (
            (rect[0][0] + offset[0], rect[0][1] + offset[1]),
            rect[1],
            rect[2]
        )
        
        box = cv2.boxPoints(shifted_rect)
        box = np.int32(box)
        cv2.fillPoly(rect_mask, [box], 255)
        
        # Shift contour
        shifted_contour = contour.copy()
        shifted_contour[:, 0] += offset[0]
        shifted_contour[:, 1] += offset[1]
        cv2.drawContours(contour_mask, [shifted_contour.astype(np.int32)], -1, 255, -1)
        
        # Compute coverage
        contour_area = (contour_mask > 0).sum()
        if contour_area == 0:
            return 0.0
        
        covered = np.logical_and(rect_mask > 0, contour_mask > 0).sum()
        return covered / contour_area
    
    def _create_contour_mask(self, contour: np.ndarray) -> np.ndarray:
        """Create a binary mask from contour."""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        pad = 10
        mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        
        # Shift contour
        shifted = contour.copy()
        shifted[:, 0] -= x - pad
        shifted[:, 1] -= y - pad
        
        cv2.drawContours(mask, [shifted.astype(np.int32)], -1, 255, -1)
        
        return mask
    
    def _normalize_contour(self, contour: np.ndarray) -> np.ndarray:
        """Ensure contour has shape (N, 2)."""
        contour = np.asarray(contour)
        
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        
        return contour.astype(np.float32)
    
    def _fallback_rect(self, contour: np.ndarray) -> FittedRectangle:
        """Create fallback rectangle for invalid contours."""
        if len(contour) == 0:
            return FittedRectangle(0, 0, 1, 1, 0)
        
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        return FittedRectangle(
            center_x=x + w / 2,
            center_y=y + h / 2,
            width=max(w, 1),
            height=max(h, 1),
            angle=0
        )
    
    def _is_valid_rect(self, rect: FittedRectangle) -> bool:
        """Check if rectangle meets validity criteria."""
        # Check dimensions
        if rect.width < self.config.min_width:
            return False
        if rect.height < self.config.min_height:
            return False
        if rect.area < self.config.min_area:
            return False
        
        # Check aspect ratio
        ar = rect.aspect_ratio
        if ar < self.config.min_aspect_ratio:
            return False
        if ar > self.config.max_aspect_ratio:
            return False
        
        return True
    
    def visualize(
        self,
        image: np.ndarray,
        rectangles: List[FittedRectangle],
        contours: Optional[List[np.ndarray]] = None,
        show_endpoints: bool = True
    ) -> np.ndarray:
        """
        Visualize fitted rectangles on image.
        
        Args:
            image: Background image
            rectangles: List of fitted rectangles
            contours: Original contours (optional)
            show_endpoints: Draw endpoint markers
            
        Returns:
            Visualization image
        """
        # Prepare output
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw contours if provided
        if contours is not None:
            for contour in contours:
                cv2.drawContours(
                    vis, [contour.astype(np.int32)],
                    -1, (100, 100, 100), 1
                )
        
        # Draw rectangles
        for i, rect in enumerate(rectangles):
            # Get color based on IoU
            if rect.iou > 0.8:
                color = (0, 255, 0)  # Green - good fit
            elif rect.iou > 0.6:
                color = (0, 255, 255)  # Yellow - okay fit
            else:
                color = (0, 0, 255)  # Red - poor fit
            
            # Draw rectangle
            box = rect.corners.astype(np.int32)
            cv2.polylines(vis, [box], True, color, 2)
            
            # Draw endpoints
            if show_endpoints:
                p1, p2 = rect.endpoints
                cv2.circle(vis, (int(p1[0]), int(p1[1])), 4, (255, 0, 0), -1)
                cv2.circle(vis, (int(p2[0]), int(p2[1])), 4, (255, 0, 0), -1)
            
            # Add IoU label
            cv2.putText(
                vis, f"{rect.iou:.2f}",
                (int(rect.center_x), int(rect.center_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        return vis


def fit_wall_rectangles(
    contours: List[np.ndarray],
    dominant_angle: Optional[float] = None,
    min_iou: float = 0.5,
    **kwargs
) -> List[FittedRectangle]:
    """
    Convenience function to fit rectangles to wall contours.
    
    Args:
        contours: List of wall contours
        dominant_angle: Constrain to this angle
        min_iou: Minimum IoU threshold
        **kwargs: Additional config parameters
        
    Returns:
        List of fitted rectangles
    """
    config = ShrinkingConfig(**kwargs)
    config.min_iou = min_iou
    
    algorithm = ShrinkingAlgorithm(config)
    results = algorithm.fit_rectangles(contours, dominant_angle=dominant_angle)
    
    return [rect for rect, _ in results]


if __name__ == "__main__":
    # Test with synthetic wall
    import matplotlib.pyplot as plt
    
    # Create test image with a wall-like shape
    img = np.zeros((200, 400), dtype=np.uint8)
    
    # Draw a rectangle (wall)
    pts = np.array([
        [50, 80], [350, 75], [355, 95], [55, 100]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    
    # Find contour
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fit rectangle
    algorithm = ShrinkingAlgorithm()
    rect, iou = algorithm.fit_rectangle(contours[0])
    
    print(f"Fitted rectangle:")
    print(f"  Center: ({rect.center_x:.1f}, {rect.center_y:.1f})")
    print(f"  Size: {rect.width:.1f} x {rect.height:.1f}")
    print(f"  Angle: {rect.angle:.1f}°")
    print(f"  IoU: {iou:.3f}")
    print(f"  Thickness: {rect.thickness:.1f}")
    print(f"  Length: {rect.length:.1f}")
    
    # Visualize
    vis = algorithm.visualize(img, [rect], contours)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Input")
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Fitted Rectangle (IoU={iou:.2f})")
    plt.tight_layout()
    plt.savefig("shrinking_test.png")
    print("Saved visualization to shrinking_test.png")
