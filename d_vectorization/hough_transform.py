"""
Hough Transform Angle Detection Module

Detects dominant angles in floor plan wall masks using Hough Transform.
Shared component used by both Shrinking Algorithm (paper) and Manhattan Constraints (original).

Reference:
- Fraunhofer HHI Paper: "Automatic Reconstruction of Semantic 3D Models from 2D Floor Plans"
- Manhattan World Assumption: Most buildings have walls aligned to 2-3 dominant directions
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import logging

logger = logging.getLogger(__name__)


@dataclass
class HoughConfig:
    """Configuration for Hough Transform angle detection."""
    
    # Hough Transform parameters
    rho: float = 1.0  # Distance resolution in pixels
    theta: float = np.pi / 180  # Angle resolution in radians (1 degree)
    threshold: int = 50  # Accumulator threshold
    min_line_length: int = 30  # Minimum line length
    max_line_gap: int = 10  # Maximum gap between line segments
    
    # Angle clustering parameters
    angle_tolerance: float = 5.0  # Degrees tolerance for clustering
    min_cluster_size: int = 3  # Minimum lines per cluster
    max_clusters: int = 4  # Maximum number of dominant angles
    
    # Preprocessing
    use_canny: bool = True
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 3
    
    # Morphological operations
    use_morphology: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 1


@dataclass
class AngleCluster:
    """Represents a cluster of detected angles."""
    
    angle: float  # Mean angle in degrees [0, 180)
    weight: float  # Sum of line lengths in cluster
    count: int  # Number of lines
    std: float  # Standard deviation
    lines: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    @property
    def angle_rad(self) -> float:
        """Return angle in radians."""
        return np.radians(self.angle)
    
    @property
    def perpendicular(self) -> float:
        """Return perpendicular angle in degrees."""
        return (self.angle + 90) % 180
    
    def is_orthogonal_to(self, other: 'AngleCluster', tolerance: float = 5.0) -> bool:
        """Check if this cluster is orthogonal to another."""
        diff = abs(self.angle - other.angle)
        diff = min(diff, 180 - diff)
        return abs(diff - 90) < tolerance


class HoughAngleDetector:
    """
    Detects dominant angles in floor plan images using Hough Transform.
    
    Implements a multi-step process:
    1. Preprocessing (blur, Canny edge detection)
    2. Probabilistic Hough Transform for line detection
    3. Angle extraction and normalization
    4. Hierarchical clustering of angles
    5. Dominant angle selection based on weight/count
    
    Example:
        detector = HoughAngleDetector()
        angles = detector.detect_angles(wall_mask)
        print(f"Dominant angles: {[a.angle for a in angles]}")
    """
    
    def __init__(self, config: Optional[HoughConfig] = None):
        """
        Initialize the angle detector.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or HoughConfig()
    
    def detect_angles(
        self,
        image: np.ndarray,
        return_lines: bool = False
    ) -> List[AngleCluster]:
        """
        Detect dominant angles in the input image.
        
        Args:
            image: Input image (grayscale or binary wall mask)
            return_lines: If True, include detected lines in clusters
            
        Returns:
            List of AngleCluster objects sorted by weight (strongest first)
        """
        # Preprocess image
        edges = self._preprocess(image)
        
        # Detect lines using Hough Transform
        lines = self._detect_lines(edges)
        
        if lines is None or len(lines) == 0:
            logger.warning("No lines detected in image")
            return []
        
        # Extract angles from lines
        angles, lengths, line_list = self._extract_angles(lines)
        
        if len(angles) == 0:
            return []
        
        # Cluster angles
        clusters = self._cluster_angles(
            angles, lengths, line_list, return_lines
        )
        
        # Sort by weight (strongest first)
        clusters.sort(key=lambda c: c.weight, reverse=True)
        
        # Limit to max clusters
        clusters = clusters[:self.config.max_clusters]
        
        logger.info(f"Detected {len(clusters)} dominant angles: "
                   f"{[f'{c.angle:.1f}°' for c in clusters]}")
        
        return clusters
    
    def detect_manhattan_axes(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect the two Manhattan World axes (orthogonal directions).
        
        Args:
            image: Input wall mask
            
        Returns:
            Tuple of (primary_angle, secondary_angle) in degrees,
            where secondary = primary + 90. Returns (None, None) if not found.
        """
        clusters = self.detect_angles(image)
        
        if len(clusters) == 0:
            return None, None
        
        # Find the strongest cluster
        primary = clusters[0]
        
        # Look for orthogonal cluster
        for cluster in clusters[1:]:
            if primary.is_orthogonal_to(cluster):
                # Return normalized pair
                angle1 = primary.angle % 90
                angle2 = (angle1 + 90) % 180
                return angle1, angle2
        
        # No orthogonal found, assume standard axes
        angle1 = primary.angle % 90
        angle2 = (angle1 + 90) % 180
        
        return angle1, angle2
    
    def get_rotation_angle(self, image: np.ndarray) -> float:
        """
        Get the rotation angle needed to align walls to horizontal/vertical.
        
        Args:
            image: Input wall mask
            
        Returns:
            Rotation angle in degrees to apply to image
        """
        primary, _ = self.detect_manhattan_axes(image)
        
        if primary is None:
            return 0.0
        
        # Find smallest rotation to align to 0° or 90°
        if primary <= 45:
            return -primary
        else:
            return 90 - primary
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for line detection."""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize if needed
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Apply blur
        if self.config.blur_kernel > 0:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.blur_kernel, self.config.blur_kernel),
                0
            )
        
        # Morphological operations for cleaner edges
        if self.config.use_morphology:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.config.morph_kernel_size, self.config.morph_kernel_size)
            )
            gray = cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, kernel,
                iterations=self.config.morph_iterations
            )
        
        # Edge detection
        if self.config.use_canny:
            edges = cv2.Canny(
                gray,
                self.config.canny_low,
                self.config.canny_high
            )
        else:
            # Use threshold for binary masks
            _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Find edges from binary
            edges = cv2.Laplacian(edges, cv2.CV_8U)
        
        return edges
    
    def _detect_lines(
        self,
        edges: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect lines using Probabilistic Hough Transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=self.config.rho,
            theta=self.config.theta,
            threshold=self.config.threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap
        )
        
        return lines
    
    def _extract_angles(
        self,
        lines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Extract angles and lengths from detected lines.
        
        Normalizes angles to [0, 180) range.
        """
        angles = []
        lengths = []
        line_list = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle in degrees
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize to [0, 180)
            angle = angle % 180
            
            # Calculate length
            length = np.sqrt(dx**2 + dy**2)
            
            angles.append(angle)
            lengths.append(length)
            line_list.append((x1, y1, x2, y2))
        
        return np.array(angles), np.array(lengths), line_list
    
    def _cluster_angles(
        self,
        angles: np.ndarray,
        lengths: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        return_lines: bool
    ) -> List[AngleCluster]:
        """
        Cluster angles using hierarchical clustering.
        
        Handles wraparound at 0°/180° boundary.
        """
        if len(angles) < 2:
            if len(angles) == 1:
                return [AngleCluster(
                    angle=float(angles[0]),
                    weight=float(lengths[0]),
                    count=1,
                    std=0.0,
                    lines=lines if return_lines else []
                )]
            return []
        
        # Convert angles to unit vectors for proper distance calculation
        # This handles the wraparound at 180°
        angles_rad = np.radians(angles * 2)  # Double to handle 180° wraparound
        vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        # Hierarchical clustering
        try:
            # Calculate pairwise distances
            distances = pdist(vectors, metric='euclidean')
            
            # Linkage
            Z = linkage(distances, method='average')
            
            # Form clusters with angle tolerance
            # Convert tolerance to distance threshold
            tol_rad = np.radians(self.config.angle_tolerance * 2)
            threshold = 2 * np.sin(tol_rad / 2)  # Chord length
            
            cluster_labels = fcluster(Z, t=threshold, criterion='distance')
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using simple binning")
            return self._simple_bin_angles(angles, lengths, lines, return_lines)
        
        # Build clusters
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_angles = angles[mask]
            cluster_lengths = lengths[mask]
            cluster_lines = [lines[i] for i in np.where(mask)[0]]
            
            # Skip small clusters
            if len(cluster_angles) < self.config.min_cluster_size:
                continue
            
            # Calculate weighted mean angle (handling wraparound)
            mean_angle = self._circular_mean(cluster_angles, cluster_lengths)
            
            cluster = AngleCluster(
                angle=mean_angle,
                weight=float(np.sum(cluster_lengths)),
                count=len(cluster_angles),
                std=float(np.std(cluster_angles)),
                lines=cluster_lines if return_lines else []
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _simple_bin_angles(
        self,
        angles: np.ndarray,
        lengths: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        return_lines: bool
    ) -> List[AngleCluster]:
        """Fallback: simple binning for angle clustering."""
        bin_width = self.config.angle_tolerance
        bins = np.arange(0, 180 + bin_width, bin_width)
        
        # Assign angles to bins
        bin_indices = np.digitize(angles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        clusters = []
        for bin_idx in np.unique(bin_indices):
            mask = bin_indices == bin_idx
            
            if np.sum(mask) < self.config.min_cluster_size:
                continue
            
            cluster_angles = angles[mask]
            cluster_lengths = lengths[mask]
            cluster_lines = [lines[i] for i in np.where(mask)[0]]
            
            mean_angle = self._circular_mean(cluster_angles, cluster_lengths)
            
            cluster = AngleCluster(
                angle=mean_angle,
                weight=float(np.sum(cluster_lengths)),
                count=len(cluster_angles),
                std=float(np.std(cluster_angles)),
                lines=cluster_lines if return_lines else []
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _circular_mean(
        self,
        angles: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate weighted circular mean of angles.
        
        Handles wraparound at 0°/180° boundary.
        """
        if weights is None:
            weights = np.ones_like(angles)
        
        # Convert to radians (double angle for 180° periodicity)
        angles_rad = np.radians(angles * 2)
        
        # Weighted mean of unit vectors
        x = np.average(np.cos(angles_rad), weights=weights)
        y = np.average(np.sin(angles_rad), weights=weights)
        
        # Convert back to angle
        mean_rad = np.arctan2(y, x) / 2
        mean_deg = np.degrees(mean_rad)
        
        # Normalize to [0, 180)
        return mean_deg % 180
    
    def visualize(
        self,
        image: np.ndarray,
        clusters: Optional[List[AngleCluster]] = None,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize detected lines colored by angle cluster.
        
        Args:
            image: Original image
            clusters: Angle clusters (will detect if None)
            line_thickness: Line drawing thickness
            
        Returns:
            Visualization image with colored lines
        """
        # Detect if needed
        if clusters is None:
            clusters = self.detect_angles(image, return_lines=True)
        
        # Prepare output image
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Colors for clusters
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw lines
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            for x1, y1, x2, y2 in cluster.lines:
                cv2.line(vis, (x1, y1), (x2, y2), color, line_thickness)
        
        # Add legend
        y_offset = 30
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            text = f"Angle: {cluster.angle:.1f}° (n={cluster.count})"
            cv2.putText(
                vis, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            y_offset += 25
        
        return vis


def detect_dominant_angles(
    image: np.ndarray,
    max_angles: int = 2,
    **kwargs
) -> List[float]:
    """
    Convenience function to detect dominant angles.
    
    Args:
        image: Input wall mask
        max_angles: Maximum number of angles to return
        **kwargs: Additional config parameters
        
    Returns:
        List of dominant angles in degrees
    """
    config = HoughConfig(**kwargs)
    config.max_clusters = max_angles
    
    detector = HoughAngleDetector(config)
    clusters = detector.detect_angles(image)
    
    return [c.angle for c in clusters]


if __name__ == "__main__":
    # Test with synthetic image
    import matplotlib.pyplot as plt
    
    # Create test image with lines at known angles
    img = np.zeros((500, 500), dtype=np.uint8)
    
    # Draw lines at 0°, 45°, 90°
    cv2.line(img, (50, 250), (450, 250), 255, 3)   # 0°
    cv2.line(img, (50, 200), (450, 200), 255, 3)   # 0°
    cv2.line(img, (250, 50), (250, 450), 255, 3)   # 90°
    cv2.line(img, (200, 50), (200, 450), 255, 3)   # 90°
    cv2.line(img, (100, 100), (400, 400), 255, 3)  # 45°
    
    # Detect angles
    detector = HoughAngleDetector()
    clusters = detector.detect_angles(img, return_lines=True)
    
    print("Detected angle clusters:")
    for c in clusters:
        print(f"  {c.angle:.1f}° (weight={c.weight:.0f}, n={c.count})")
    
    # Visualize
    vis = detector.visualize(img, clusters)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Input")
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Detected Angles")
    plt.tight_layout()
    plt.savefig("hough_test.png")
    print("Saved visualization to hough_test.png")
