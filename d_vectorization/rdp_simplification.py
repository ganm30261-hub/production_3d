"""
Ramer-Douglas-Peucker (RDP) Polygon Simplification Module

Implements polygon simplification for floor plan contours.
Part of the original system's vectorization pipeline.

Features:
- Standard RDP algorithm with configurable epsilon
- Adaptive epsilon based on contour size
- Area-preserving simplification variant
- Corner detection and preservation
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimplificationMode(Enum):
    """Simplification mode selection."""
    STANDARD = "standard"        # Standard RDP
    ADAPTIVE = "adaptive"        # Adaptive epsilon based on size
    AREA_PRESERVING = "area"     # Minimize area change
    CORNER_PRESERVING = "corner" # Preserve detected corners


@dataclass
class RDPConfig:
    """Configuration for RDP simplification."""
    
    # Basic RDP parameters
    epsilon: float = 2.0  # Distance threshold for point removal
    closed: bool = True   # Whether contour is closed
    
    # Adaptive mode parameters
    adaptive_factor: float = 0.01  # epsilon = perimeter * factor
    min_epsilon: float = 1.0
    max_epsilon: float = 10.0
    
    # Simplification constraints
    min_points: int = 4      # Minimum points in result
    max_points: int = 100    # Maximum points (0 = unlimited)
    min_area_ratio: float = 0.9  # Minimum area preservation ratio
    
    # Corner preservation
    corner_angle_threshold: float = 30.0  # Degrees - angles sharper than this are corners
    corner_protection_radius: int = 3     # Points around corner to protect
    
    # Mode
    mode: SimplificationMode = SimplificationMode.ADAPTIVE


@dataclass
class SimplifiedPolygon:
    """Represents a simplified polygon."""
    
    points: np.ndarray  # Simplified points (N, 2)
    original_points: int  # Number of original points
    epsilon_used: float   # Actual epsilon value used
    
    area_original: float = 0.0
    area_simplified: float = 0.0
    perimeter_original: float = 0.0
    perimeter_simplified: float = 0.0
    
    @property
    def num_points(self) -> int:
        """Number of points in simplified polygon."""
        return len(self.points)
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of original to simplified points."""
        if self.num_points == 0:
            return 0.0
        return self.original_points / self.num_points
    
    @property
    def area_preservation(self) -> float:
        """Ratio of simplified area to original."""
        if self.area_original == 0:
            return 0.0
        return self.area_simplified / self.area_original
    
    @property
    def is_valid(self) -> bool:
        """Check if polygon is valid (at least 3 points)."""
        return self.num_points >= 3
    
    def to_contour(self) -> np.ndarray:
        """Convert to OpenCV contour format (N, 1, 2)."""
        return self.points.reshape(-1, 1, 2).astype(np.int32)
    
    @property
    def corners(self) -> List[Tuple[float, float]]:
        """Return corner points as list of tuples."""
        return [tuple(p) for p in self.points]


class RDPSimplifier:
    """
    Ramer-Douglas-Peucker polygon simplification.
    
    Reduces the number of points in a polygon while preserving its shape.
    Multiple modes available for different use cases:
    
    - STANDARD: Fixed epsilon value
    - ADAPTIVE: Epsilon scales with polygon size
    - AREA_PRESERVING: Iteratively adjust epsilon to preserve area
    - CORNER_PRESERVING: Detect and protect corner points
    
    Example:
        simplifier = RDPSimplifier()
        result = simplifier.simplify(contour)
        print(f"Reduced {result.original_points} -> {result.num_points} points")
    """
    
    def __init__(self, config: Optional[RDPConfig] = None):
        """
        Initialize the RDP simplifier.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or RDPConfig()
    
    def simplify(
        self,
        contour: np.ndarray,
        epsilon: Optional[float] = None
    ) -> SimplifiedPolygon:
        """
        Simplify a polygon contour.
        
        Args:
            contour: Input contour (N, 2) or (N, 1, 2)
            epsilon: Override epsilon value (uses config if None)
            
        Returns:
            SimplifiedPolygon object
        """
        # Normalize contour
        points = self._normalize_contour(contour)
        
        if len(points) < 3:
            return SimplifiedPolygon(
                points=points,
                original_points=len(points),
                epsilon_used=0
            )
        
        # Calculate original metrics
        area_original = cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32))
        perimeter_original = cv2.arcLength(
            points.reshape(-1, 1, 2).astype(np.float32),
            self.config.closed
        )
        
        # Choose simplification method
        if self.config.mode == SimplificationMode.STANDARD:
            simplified, eps = self._simplify_standard(points, epsilon)
        elif self.config.mode == SimplificationMode.ADAPTIVE:
            simplified, eps = self._simplify_adaptive(points, perimeter_original)
        elif self.config.mode == SimplificationMode.AREA_PRESERVING:
            simplified, eps = self._simplify_area_preserving(points, area_original)
        elif self.config.mode == SimplificationMode.CORNER_PRESERVING:
            simplified, eps = self._simplify_corner_preserving(points, epsilon)
        else:
            simplified, eps = self._simplify_standard(points, epsilon)
        
        # Calculate simplified metrics
        area_simplified = cv2.contourArea(
            simplified.reshape(-1, 1, 2).astype(np.float32)
        )
        perimeter_simplified = cv2.arcLength(
            simplified.reshape(-1, 1, 2).astype(np.float32),
            self.config.closed
        )
        
        return SimplifiedPolygon(
            points=simplified,
            original_points=len(points),
            epsilon_used=eps,
            area_original=area_original,
            area_simplified=area_simplified,
            perimeter_original=perimeter_original,
            perimeter_simplified=perimeter_simplified
        )
    
    def simplify_multiple(
        self,
        contours: List[np.ndarray],
        epsilon: Optional[float] = None
    ) -> List[SimplifiedPolygon]:
        """
        Simplify multiple contours.
        
        Args:
            contours: List of contours
            epsilon: Override epsilon value
            
        Returns:
            List of SimplifiedPolygon objects
        """
        results = []
        for contour in contours:
            try:
                result = self.simplify(contour, epsilon)
                if result.is_valid:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to simplify contour: {e}")
                continue
        
        return results
    
    def _simplify_standard(
        self,
        points: np.ndarray,
        epsilon: Optional[float]
    ) -> Tuple[np.ndarray, float]:
        """Standard RDP with fixed epsilon."""
        eps = epsilon if epsilon is not None else self.config.epsilon
        
        simplified = cv2.approxPolyDP(
            points.reshape(-1, 1, 2).astype(np.float32),
            eps,
            self.config.closed
        )
        
        return simplified.reshape(-1, 2), eps
    
    def _simplify_adaptive(
        self,
        points: np.ndarray,
        perimeter: float
    ) -> Tuple[np.ndarray, float]:
        """Adaptive epsilon based on perimeter."""
        # Calculate adaptive epsilon
        eps = perimeter * self.config.adaptive_factor
        eps = np.clip(eps, self.config.min_epsilon, self.config.max_epsilon)
        
        simplified = cv2.approxPolyDP(
            points.reshape(-1, 1, 2).astype(np.float32),
            eps,
            self.config.closed
        )
        
        result = simplified.reshape(-1, 2)
        
        # Ensure minimum points
        if len(result) < self.config.min_points:
            # Try with smaller epsilon
            eps = eps / 2
            simplified = cv2.approxPolyDP(
                points.reshape(-1, 1, 2).astype(np.float32),
                eps,
                self.config.closed
            )
            result = simplified.reshape(-1, 2)
        
        return result, eps
    
    def _simplify_area_preserving(
        self,
        points: np.ndarray,
        original_area: float
    ) -> Tuple[np.ndarray, float]:
        """Binary search for epsilon that preserves area."""
        if original_area == 0:
            return self._simplify_standard(points, None)
        
        low_eps = self.config.min_epsilon
        high_eps = self.config.max_epsilon
        best_result = points
        best_eps = 0
        
        for _ in range(10):  # Binary search iterations
            mid_eps = (low_eps + high_eps) / 2
            
            simplified = cv2.approxPolyDP(
                points.reshape(-1, 1, 2).astype(np.float32),
                mid_eps,
                self.config.closed
            )
            
            result = simplified.reshape(-1, 2)
            
            if len(result) < self.config.min_points:
                high_eps = mid_eps
                continue
            
            area = cv2.contourArea(simplified)
            ratio = area / original_area
            
            if ratio >= self.config.min_area_ratio:
                best_result = result
                best_eps = mid_eps
                low_eps = mid_eps  # Try larger epsilon
            else:
                high_eps = mid_eps  # Need smaller epsilon
        
        return best_result, best_eps
    
    def _simplify_corner_preserving(
        self,
        points: np.ndarray,
        epsilon: Optional[float]
    ) -> Tuple[np.ndarray, float]:
        """Simplify while preserving detected corners."""
        eps = epsilon if epsilon is not None else self.config.epsilon
        
        # Detect corners
        corners = self._detect_corners(points)
        
        if len(corners) == 0:
            return self._simplify_standard(points, eps)
        
        # Create protected indices
        protected = set()
        for corner_idx in corners:
            for offset in range(-self.config.corner_protection_radius,
                               self.config.corner_protection_radius + 1):
                protected.add((corner_idx + offset) % len(points))
        
        # Custom RDP that protects corners
        result = self._rdp_with_protection(points, eps, protected)
        
        return np.array(result), eps
    
    def _detect_corners(self, points: np.ndarray) -> List[int]:
        """Detect corner points based on angle threshold."""
        if len(points) < 3:
            return []
        
        corners = []
        n = len(points)
        
        for i in range(n):
            # Get neighboring points
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]
            
            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Check if it's a corner
            if angle < 180 - self.config.corner_angle_threshold:
                corners.append(i)
        
        return corners
    
    def _rdp_with_protection(
        self,
        points: np.ndarray,
        epsilon: float,
        protected: set
    ) -> List[np.ndarray]:
        """
        RDP implementation that preserves protected points.
        
        Custom implementation since OpenCV doesn't support protection.
        """
        def rdp_recursive(start: int, end: int) -> List[int]:
            if end - start < 2:
                return []
            
            # Find point with maximum distance
            max_dist = 0
            max_idx = start
            
            p1 = points[start]
            p2 = points[end]
            
            for i in range(start + 1, end):
                # Skip protected points
                if i in protected:
                    continue
                
                dist = self._point_line_distance(points[i], p1, p2)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # Check for protected points
            has_protected = any(i in protected for i in range(start + 1, end))
            
            if max_dist > epsilon or has_protected:
                # Recursively simplify
                left = rdp_recursive(start, max_idx)
                right = rdp_recursive(max_idx, end)
                return left + [max_idx] + right
            else:
                return []
        
        # Run RDP
        n = len(points)
        
        if self.config.closed:
            # For closed contours, find the best starting point
            indices = [0] + rdp_recursive(0, n - 1) + [n - 1]
        else:
            indices = [0] + rdp_recursive(0, n - 1) + [n - 1]
        
        # Add protected points
        indices = sorted(set(indices) | protected)
        
        return [points[i] for i in indices]
    
    def _point_line_distance(
        self,
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-8:
            return np.linalg.norm(point - line_start)
        
        # Normalized line vector
        line_unit = line_vec / line_len
        
        # Vector from line start to point
        point_vec = point - line_start
        
        # Project point onto line
        proj_length = np.dot(point_vec, line_unit)
        proj_length = np.clip(proj_length, 0, line_len)
        
        # Calculate distance
        proj_point = line_start + line_unit * proj_length
        return np.linalg.norm(point - proj_point)
    
    def _normalize_contour(self, contour: np.ndarray) -> np.ndarray:
        """Ensure contour has shape (N, 2)."""
        contour = np.asarray(contour)
        
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        
        return contour.astype(np.float32)
    
    def visualize(
        self,
        image: np.ndarray,
        original: np.ndarray,
        simplified: SimplifiedPolygon
    ) -> np.ndarray:
        """
        Visualize original vs simplified polygon.
        
        Args:
            image: Background image
            original: Original contour
            simplified: Simplified polygon
            
        Returns:
            Visualization image
        """
        # Prepare output
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw original (gray, thin)
        original = self._normalize_contour(original)
        cv2.polylines(
            vis,
            [original.astype(np.int32)],
            self.config.closed,
            (150, 150, 150),
            1
        )
        
        # Draw simplified (green, thick)
        cv2.polylines(
            vis,
            [simplified.points.astype(np.int32)],
            self.config.closed,
            (0, 255, 0),
            2
        )
        
        # Draw vertices
        for point in simplified.points:
            cv2.circle(
                vis,
                (int(point[0]), int(point[1])),
                3,
                (0, 0, 255),
                -1
            )
        
        # Add info
        info = (
            f"Points: {simplified.original_points} -> {simplified.num_points} "
            f"(eps={simplified.epsilon_used:.1f})"
        )
        cv2.putText(
            vis, info, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        return vis


def simplify_contour(
    contour: np.ndarray,
    epsilon: Optional[float] = None,
    mode: str = "adaptive",
    **kwargs
) -> np.ndarray:
    """
    Convenience function to simplify a contour.
    
    Args:
        contour: Input contour
        epsilon: Distance threshold
        mode: Simplification mode ("standard", "adaptive", "area", "corner")
        **kwargs: Additional config parameters
        
    Returns:
        Simplified contour points (N, 2)
    """
    mode_map = {
        "standard": SimplificationMode.STANDARD,
        "adaptive": SimplificationMode.ADAPTIVE,
        "area": SimplificationMode.AREA_PRESERVING,
        "corner": SimplificationMode.CORNER_PRESERVING
    }
    
    config = RDPConfig(**kwargs)
    config.mode = mode_map.get(mode, SimplificationMode.ADAPTIVE)
    
    if epsilon is not None:
        config.epsilon = epsilon
    
    simplifier = RDPSimplifier(config)
    result = simplifier.simplify(contour)
    
    return result.points


def simplify_to_rectangle(
    contour: np.ndarray,
    tolerance: float = 5.0
) -> Optional[np.ndarray]:
    """
    Attempt to simplify contour to a 4-point rectangle.
    
    Args:
        contour: Input contour
        tolerance: Maximum deviation from rectangle
        
    Returns:
        4-point rectangle if successful, None otherwise
    """
    config = RDPConfig(
        mode=SimplificationMode.CORNER_PRESERVING,
        min_points=4,
        max_points=6,
        corner_angle_threshold=tolerance
    )
    
    simplifier = RDPSimplifier(config)
    
    # Try increasing epsilon until we get 4 points
    for eps in [2.0, 5.0, 10.0, 15.0, 20.0]:
        result = simplifier.simplify(contour, eps)
        
        if result.num_points == 4:
            # Verify it's approximately rectangular
            if _is_approximate_rectangle(result.points, tolerance):
                return result.points
        elif result.num_points < 4:
            break
    
    return None


def _is_approximate_rectangle(
    points: np.ndarray,
    tolerance: float
) -> bool:
    """Check if 4 points form an approximate rectangle."""
    if len(points) != 4:
        return False
    
    # Check angles at each corner
    for i in range(4):
        p_prev = points[(i - 1) % 4]
        p_curr = points[i]
        p_next = points[(i + 1) % 4]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Should be close to 90°
        if abs(angle - 90) > tolerance:
            return False
    
    return True


if __name__ == "__main__":
    # Test with synthetic contour
    import matplotlib.pyplot as plt
    
    # Create a complex contour
    t = np.linspace(0, 2 * np.pi, 100)
    r = 100 + 20 * np.sin(5 * t) + 10 * np.sin(10 * t)
    x = 200 + r * np.cos(t)
    y = 200 + r * np.sin(t)
    
    contour = np.column_stack([x, y]).astype(np.float32)
    
    # Test different modes
    modes = ["standard", "adaptive", "area", "corner"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for ax, mode in zip(axes, modes):
        config = RDPConfig()
        config.mode = SimplificationMode[mode.upper()]
        
        simplifier = RDPSimplifier(config)
        result = simplifier.simplify(contour)
        
        # Plot
        ax.plot(contour[:, 0], contour[:, 1], 'b-', alpha=0.5, label='Original')
        ax.plot(result.points[:, 0], result.points[:, 1], 'r-', linewidth=2, label='Simplified')
        ax.scatter(result.points[:, 0], result.points[:, 1], c='r', s=30)
        
        ax.set_title(f"{mode.upper()}: {result.original_points} -> {result.num_points} points")
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("rdp_test.png")
    print("Saved visualization to rdp_test.png")
