"""
Wall Extractor Module - Unified Vectorization Pipeline

Combines multiple vectorization methods for robust wall segment extraction:
- Hough Transform: Dominant angle detection (shared)
- Shrinking Algorithm: Rectangle fitting (paper method)
- RDP Simplification: Polygon simplification (original)
- Manhattan Constraints: Angle alignment (original)

This module fuses the Fraunhofer HHI paper's approach with the original
system's methods for optimal wall extraction quality.

Reference:
- Fraunhofer HHI Paper: "Automatic Reconstruction of Semantic 3D Models
  from 2D Floor Plans" (MVA 2023)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import logging

from .hough_transform import HoughAngleDetector, HoughConfig, AngleCluster
from .shrinking_algorithm import ShrinkingAlgorithm, ShrinkingConfig, FittedRectangle
from .rdp_simplification import RDPSimplifier, RDPConfig, SimplifiedPolygon
from .manhattan_constraints import ManhattanConstraints, ManhattanConfig, AlignedSegment

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Wall extraction method selection."""
    SHRINKING = "shrinking"      # Paper: Shrinking rectangles
    CONTOUR = "contour"          # Original: Contour + RDP
    HYBRID = "hybrid"            # Best of both
    AUTO = "auto"                # Automatic selection


@dataclass
class WallExtractorConfig:
    """Configuration for wall extraction pipeline."""
    
    # Method selection
    method: ExtractionMethod = ExtractionMethod.HYBRID
    
    # Preprocessing
    use_morphology: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 2
    
    # Contour extraction
    contour_mode: int = cv2.RETR_EXTERNAL
    contour_method: int = cv2.CHAIN_APPROX_SIMPLE
    min_contour_area: float = 100.0
    min_contour_length: float = 20.0
    
    # Component configs
    hough_config: Optional[HoughConfig] = None
    shrinking_config: Optional[ShrinkingConfig] = None
    rdp_config: Optional[RDPConfig] = None
    manhattan_config: Optional[ManhattanConfig] = None
    
    # Quality thresholds
    min_wall_length: float = 10.0
    max_wall_length: float = 5000.0
    min_wall_thickness: float = 2.0
    max_wall_thickness: float = 50.0
    
    # Post-processing
    apply_manhattan: bool = True
    merge_nearby: bool = True
    merge_distance: float = 5.0
    
    # Output
    return_debug_info: bool = False


@dataclass
class WallSegment:
    """Represents an extracted wall segment."""
    
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float
    angle: float
    confidence: float = 1.0
    method: str = ""
    contour_idx: int = -1
    
    @property
    def p1(self) -> Tuple[float, float]:
        return (self.x1, self.y1)
    
    @property
    def p2(self) -> Tuple[float, float]:
        return (self.x2, self.y2)
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.midpoint
    
    @property
    def direction(self) -> np.ndarray:
        d = np.array([self.x2 - self.x1, self.y2 - self.y1])
        norm = np.linalg.norm(d)
        return d / norm if norm > 0 else np.array([1.0, 0.0])
    
    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])
    
    @property
    def corners(self) -> np.ndarray:
        mid = np.array(self.midpoint)
        direction = self.direction
        normal = self.normal
        half_len = self.length / 2
        half_thick = self.thickness / 2
        return np.array([
            mid - direction * half_len - normal * half_thick,
            mid + direction * half_len - normal * half_thick,
            mid + direction * half_len + normal * half_thick,
            mid - direction * half_len + normal * half_thick
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'thickness': self.thickness,
            'angle': self.angle,
            'confidence': self.confidence,
            'length': self.length,
            'method': self.method
        }
    
    def to_aligned_segment(self) -> AlignedSegment:
        return AlignedSegment(
            x1=self.x1, y1=self.y1,
            x2=self.x2, y2=self.y2,
            angle=self.angle,
            original_angle=self.angle,
            thickness=self.thickness,
            confidence=self.confidence
        )
    
    @classmethod
    def from_rectangle(cls, rect: FittedRectangle, method: str = "shrinking") -> 'WallSegment':
        p1, p2 = rect.endpoints
        return cls(
            x1=p1[0], y1=p1[1],
            x2=p2[0], y2=p2[1],
            thickness=rect.thickness,
            angle=rect.angle,
            confidence=rect.iou,
            method=method
        )
    
    @classmethod
    def from_aligned_segment(cls, seg: AlignedSegment, method: str = "manhattan") -> 'WallSegment':
        return cls(
            x1=seg.x1, y1=seg.y1,
            x2=seg.x2, y2=seg.y2,
            thickness=seg.thickness,
            angle=seg.angle,
            confidence=seg.confidence,
            method=method
        )


@dataclass
class ExtractionResult:
    """Result of wall extraction."""
    
    walls: List[WallSegment]
    dominant_angles: List[float]
    contours: Optional[List[np.ndarray]] = None
    rectangles: Optional[List[FittedRectangle]] = None
    angle_clusters: Optional[List[AngleCluster]] = None
    preprocessing_mask: Optional[np.ndarray] = None
    
    @property
    def num_walls(self) -> int:
        return len(self.walls)
    
    @property
    def total_length(self) -> float:
        return sum(w.length for w in self.walls)
    
    @property
    def average_thickness(self) -> float:
        if not self.walls:
            return 0.0
        return np.mean([w.thickness for w in self.walls])
    
    def filter_by_length(self, min_length: float = 0, max_length: float = float('inf')) -> 'ExtractionResult':
        filtered = [w for w in self.walls if min_length <= w.length <= max_length]
        return ExtractionResult(
            walls=filtered,
            dominant_angles=self.dominant_angles,
            contours=self.contours,
            rectangles=self.rectangles,
            angle_clusters=self.angle_clusters,
            preprocessing_mask=self.preprocessing_mask
        )
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'ExtractionResult':
        filtered = [w for w in self.walls if w.confidence >= min_confidence]
        return ExtractionResult(
            walls=filtered,
            dominant_angles=self.dominant_angles,
            contours=self.contours,
            rectangles=self.rectangles,
            angle_clusters=self.angle_clusters,
            preprocessing_mask=self.preprocessing_mask
        )


class WallExtractor:
    """
    Unified wall extraction pipeline.
    
    Combines multiple vectorization methods:
    1. Preprocessing: Morphological operations to clean wall mask
    2. Angle Detection: Hough Transform finds dominant directions
    3. Contour Extraction: Find wall region boundaries
    4. Rectangle Fitting: Shrinking Algorithm or RDP Simplification
    5. Manhattan Alignment: Snap to dominant angles
    6. Post-processing: Merge nearby segments
    
    Example:
        extractor = WallExtractor()
        result = extractor.extract(wall_mask)
        
        for wall in result.walls:
            print(f"Wall: {wall.length:.1f}px at {wall.angle:.1f}°")
    """
    
    def __init__(self, config: Optional[WallExtractorConfig] = None):
        self.config = config or WallExtractorConfig()
        
        self.hough_detector = HoughAngleDetector(
            self.config.hough_config or HoughConfig()
        )
        self.shrinking_algo = ShrinkingAlgorithm(
            self.config.shrinking_config or ShrinkingConfig()
        )
        self.rdp_simplifier = RDPSimplifier(
            self.config.rdp_config or RDPConfig()
        )
        self.manhattan = ManhattanConstraints(
            self.config.manhattan_config or ManhattanConfig()
        )
    
    def extract(
        self,
        wall_mask: np.ndarray,
        dominant_angles: Optional[List[float]] = None
    ) -> ExtractionResult:
        """
        Extract wall segments from binary mask.
        
        Args:
            wall_mask: Binary wall mask (0=background, 255=wall)
            dominant_angles: Override dominant angles (auto-detect if None)
            
        Returns:
            ExtractionResult with walls and debug info
        """
        # Preprocess
        processed = self._preprocess(wall_mask)
        
        # Detect dominant angles
        if dominant_angles is None:
            angle_clusters = self.hough_detector.detect_angles(
                processed, return_lines=self.config.return_debug_info
            )
            dominant_angles = [c.angle for c in angle_clusters]
        else:
            angle_clusters = None
        
        if not dominant_angles:
            dominant_angles = [0.0, 90.0]
            logger.warning("No dominant angles detected, using standard axes")
        
        # Extract contours
        contours = self._find_contours(processed)
        
        if len(contours) == 0:
            logger.warning("No contours found in wall mask")
            return ExtractionResult(
                walls=[],
                dominant_angles=dominant_angles,
                preprocessing_mask=processed if self.config.return_debug_info else None
            )
        
        # Extract walls using selected method
        if self.config.method == ExtractionMethod.SHRINKING:
            walls, rectangles = self._extract_shrinking(contours, processed, dominant_angles)
        elif self.config.method == ExtractionMethod.CONTOUR:
            walls = self._extract_contour(contours, dominant_angles)
            rectangles = None
        elif self.config.method == ExtractionMethod.HYBRID:
            walls, rectangles = self._extract_hybrid(contours, processed, dominant_angles)
        else:
            walls, rectangles = self._extract_auto(contours, processed, dominant_angles)
        
        # Apply Manhattan constraints
        if self.config.apply_manhattan and walls:
            walls = self._apply_manhattan(walls, dominant_angles)
        
        # Post-process
        if self.config.merge_nearby and walls:
            walls = self._merge_nearby_walls(walls)
        
        # Filter by constraints
        walls = self._filter_walls(walls)
        
        return ExtractionResult(
            walls=walls,
            dominant_angles=dominant_angles,
            contours=contours if self.config.return_debug_info else None,
            rectangles=rectangles if self.config.return_debug_info else None,
            angle_clusters=angle_clusters if self.config.return_debug_info else None,
            preprocessing_mask=processed if self.config.return_debug_info else None
        )
    
    def _preprocess(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess wall mask."""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if not self.config.use_morphology:
            return binary
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel,
            iterations=self.config.morph_iterations
        )
        
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find and filter contours."""
        contours, _ = cv2.findContours(
            mask,
            self.config.contour_mode,
            self.config.contour_method
        )
        
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            
            if area >= self.config.min_contour_area and \
               length >= self.config.min_contour_length:
                filtered.append(contour)
        
        return filtered
    
    def _extract_shrinking(
        self,
        contours: List[np.ndarray],
        mask: np.ndarray,
        dominant_angles: List[float]
    ) -> Tuple[List[WallSegment], List[FittedRectangle]]:
        """Extract walls using shrinking algorithm."""
        walls = []
        rectangles = []
        
        primary_angle = dominant_angles[0] if dominant_angles else None
        
        for i, contour in enumerate(contours):
            try:
                rect, iou = self.shrinking_algo.fit_rectangle(
                    contour, mask, dominant_angle=primary_angle
                )
                
                rectangles.append(rect)
                
                if iou >= 0.3:
                    wall = WallSegment.from_rectangle(rect, method="shrinking")
                    wall.contour_idx = i
                    walls.append(wall)
                    
            except Exception as e:
                logger.warning(f"Shrinking failed for contour {i}: {e}")
                continue
        
        return walls, rectangles
    
    def _extract_contour(
        self,
        contours: List[np.ndarray],
        dominant_angles: List[float]
    ) -> List[WallSegment]:
        """Extract walls using contour simplification."""
        walls = []
        
        for i, contour in enumerate(contours):
            try:
                simplified = self.rdp_simplifier.simplify(contour)
                
                if not simplified.is_valid:
                    continue
                
                points = simplified.points
                n = len(points)
                
                for j in range(n):
                    p1 = points[j]
                    p2 = points[(j + 1) % n]
                    
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length < self.config.min_wall_length:
                        continue
                    
                    angle = np.degrees(np.arctan2(dy, dx)) % 180
                    
                    wall = WallSegment(
                        x1=p1[0], y1=p1[1],
                        x2=p2[0], y2=p2[1],
                        thickness=3.0,
                        angle=angle,
                        confidence=1.0,
                        method="contour",
                        contour_idx=i
                    )
                    walls.append(wall)
                    
            except Exception as e:
                logger.warning(f"Contour extraction failed for contour {i}: {e}")
                continue
        
        return walls
    
    def _extract_hybrid(
        self,
        contours: List[np.ndarray],
        mask: np.ndarray,
        dominant_angles: List[float]
    ) -> Tuple[List[WallSegment], List[FittedRectangle]]:
        """Hybrid extraction: use shrinking for good fits, contour for others."""
        walls = []
        rectangles = []
        
        primary_angle = dominant_angles[0] if dominant_angles else None
        
        for i, contour in enumerate(contours):
            try:
                # Try shrinking first
                rect, iou = self.shrinking_algo.fit_rectangle(
                    contour, mask, dominant_angle=primary_angle
                )
                
                rectangles.append(rect)
                
                if iou >= 0.6:  # Good rectangle fit
                    wall = WallSegment.from_rectangle(rect, method="shrinking")
                    wall.contour_idx = i
                    walls.append(wall)
                else:
                    # Fall back to contour method
                    simplified = self.rdp_simplifier.simplify(contour)
                    
                    if simplified.is_valid:
                        points = simplified.points
                        n = len(points)
                        
                        for j in range(n):
                            p1 = points[j]
                            p2 = points[(j + 1) % n]
                            
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            length = np.sqrt(dx**2 + dy**2)
                            
                            if length < self.config.min_wall_length:
                                continue
                            
                            angle = np.degrees(np.arctan2(dy, dx)) % 180
                            
                            wall = WallSegment(
                                x1=p1[0], y1=p1[1],
                                x2=p2[0], y2=p2[1],
                                thickness=3.0,
                                angle=angle,
                                confidence=0.8,
                                method="contour",
                                contour_idx=i
                            )
                            walls.append(wall)
                            
            except Exception as e:
                logger.warning(f"Hybrid extraction failed for contour {i}: {e}")
                continue
        
        return walls, rectangles
    
    def _extract_auto(
        self,
        contours: List[np.ndarray],
        mask: np.ndarray,
        dominant_angles: List[float]
    ) -> Tuple[List[WallSegment], List[FittedRectangle]]:
        """Auto method selection based on contour characteristics."""
        walls = []
        rectangles = []
        
        primary_angle = dominant_angles[0] if dominant_angles else None
        
        for i, contour in enumerate(contours):
            try:
                # Analyze contour shape
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Circularity: 1 for circle, lower for elongated shapes
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Bounding box aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                
                # Walls are typically elongated (low circularity, low aspect)
                is_wall_like = circularity < 0.3 and aspect < 0.3
                
                if is_wall_like:
                    # Use shrinking for wall-like shapes
                    rect, iou = self.shrinking_algo.fit_rectangle(
                        contour, mask, dominant_angle=primary_angle
                    )
                    rectangles.append(rect)
                    
                    if iou >= 0.4:
                        wall = WallSegment.from_rectangle(rect, method="auto-shrinking")
                        wall.contour_idx = i
                        walls.append(wall)
                else:
                    # Use contour for complex shapes
                    simplified = self.rdp_simplifier.simplify(contour)
                    
                    if simplified.is_valid:
                        points = simplified.points
                        n = len(points)
                        
                        for j in range(n):
                            p1 = points[j]
                            p2 = points[(j + 1) % n]
                            
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            length = np.sqrt(dx**2 + dy**2)
                            
                            if length < self.config.min_wall_length:
                                continue
                            
                            angle = np.degrees(np.arctan2(dy, dx)) % 180
                            
                            wall = WallSegment(
                                x1=p1[0], y1=p1[1],
                                x2=p2[0], y2=p2[1],
                                thickness=3.0,
                                angle=angle,
                                confidence=0.7,
                                method="auto-contour",
                                contour_idx=i
                            )
                            walls.append(wall)
                            
            except Exception as e:
                logger.warning(f"Auto extraction failed for contour {i}: {e}")
                continue
        
        return walls, rectangles
    
    def _apply_manhattan(
        self,
        walls: List[WallSegment],
        dominant_angles: List[float]
    ) -> List[WallSegment]:
        """Apply Manhattan constraints to walls."""
        # Convert to AlignedSegment
        segments = [w.to_aligned_segment() for w in walls]
        
        # Apply constraints
        aligned = self.manhattan.process(segments, dominant_angles)
        
        # Convert back
        result = []
        for seg in aligned:
            wall = WallSegment.from_aligned_segment(seg, method="manhattan")
            result.append(wall)
        
        return result
    
    def _merge_nearby_walls(self, walls: List[WallSegment]) -> List[WallSegment]:
        """Merge nearby parallel walls."""
        if len(walls) < 2:
            return walls
        
        merged = []
        used = set()
        
        for i, wall_i in enumerate(walls):
            if i in used:
                continue
            
            current = wall_i
            
            for j, wall_j in enumerate(walls[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if parallel (similar angle)
                angle_diff = abs(current.angle - wall_j.angle)
                angle_diff = min(angle_diff, 180 - angle_diff)
                
                if angle_diff > 10:
                    continue
                
                # Check if nearby endpoints
                distances = [
                    np.sqrt((current.x1 - wall_j.x1)**2 + (current.y1 - wall_j.y1)**2),
                    np.sqrt((current.x1 - wall_j.x2)**2 + (current.y1 - wall_j.y2)**2),
                    np.sqrt((current.x2 - wall_j.x1)**2 + (current.y2 - wall_j.y1)**2),
                    np.sqrt((current.x2 - wall_j.x2)**2 + (current.y2 - wall_j.y2)**2),
                ]
                
                min_dist = min(distances)
                
                if min_dist < self.config.merge_distance:
                    # Merge
                    current = self._merge_two_walls(current, wall_j)
                    used.add(j)
            
            merged.append(current)
        
        return merged
    
    def _merge_two_walls(self, wall1: WallSegment, wall2: WallSegment) -> WallSegment:
        """Merge two walls into one."""
        points = [
            np.array([wall1.x1, wall1.y1]),
            np.array([wall1.x2, wall1.y2]),
            np.array([wall2.x1, wall2.y1]),
            np.array([wall2.x2, wall2.y2])
        ]
        
        # Find most distant pair
        max_dist = 0
        best_pair = (0, 1)
        
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)
        
        p1 = points[best_pair[0]]
        p2 = points[best_pair[1]]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        
        return WallSegment(
            x1=p1[0], y1=p1[1],
            x2=p2[0], y2=p2[1],
            thickness=max(wall1.thickness, wall2.thickness),
            angle=angle,
            confidence=(wall1.confidence + wall2.confidence) / 2,
            method="merged"
        )
    
    def _filter_walls(self, walls: List[WallSegment]) -> List[WallSegment]:
        """Filter walls by constraints."""
        result = []
        
        for wall in walls:
            # Length constraints
            if wall.length < self.config.min_wall_length:
                continue
            if wall.length > self.config.max_wall_length:
                continue
            
            # Thickness constraints
            if wall.thickness < self.config.min_wall_thickness:
                continue
            if wall.thickness > self.config.max_wall_thickness:
                continue
            
            result.append(wall)
        
        return result
    
    def visualize(
        self,
        image: np.ndarray,
        result: ExtractionResult,
        show_angles: bool = True,
        show_thickness: bool = True
    ) -> np.ndarray:
        """
        Visualize extraction result.
        
        Args:
            image: Background image
            result: Extraction result
            show_angles: Show angle labels
            show_thickness: Draw walls with thickness
            
        Returns:
            Visualization image
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw walls
        for wall in result.walls:
            # Color by method
            color_map = {
                "shrinking": (0, 255, 0),
                "contour": (255, 0, 0),
                "manhattan": (0, 255, 255),
                "merged": (255, 0, 255),
                "auto-shrinking": (0, 200, 0),
                "auto-contour": (200, 0, 0),
            }
            color = color_map.get(wall.method, (255, 255, 255))
            
            if show_thickness:
                # Draw as rectangle
                corners = wall.corners.astype(np.int32)
                cv2.fillPoly(vis, [corners], color)
                cv2.polylines(vis, [corners], True, (0, 0, 0), 1)
            else:
                # Draw as line
                pt1 = (int(wall.x1), int(wall.y1))
                pt2 = (int(wall.x2), int(wall.y2))
                cv2.line(vis, pt1, pt2, color, 2)
            
            # Draw endpoints
            cv2.circle(vis, (int(wall.x1), int(wall.y1)), 3, (255, 255, 255), -1)
            cv2.circle(vis, (int(wall.x2), int(wall.y2)), 3, (255, 255, 255), -1)
            
            if show_angles:
                mid = wall.midpoint
                label = f"{wall.angle:.0f}°"
                cv2.putText(
                    vis, label,
                    (int(mid[0]), int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
                )
        
        # Add legend
        y_offset = 20
        for method, color in [
            ("shrinking", (0, 255, 0)),
            ("contour", (255, 0, 0)),
            ("manhattan", (0, 255, 255)),
            ("merged", (255, 0, 255)),
        ]:
            cv2.rectangle(vis, (10, y_offset - 10), (25, y_offset), color, -1)
            cv2.putText(
                vis, method, (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            y_offset += 20
        
        return vis


def extract_walls(
    wall_mask: np.ndarray,
    method: str = "hybrid",
    apply_manhattan: bool = True,
    **kwargs
) -> List[WallSegment]:
    """
    Convenience function to extract walls from mask.
    
    Args:
        wall_mask: Binary wall mask
        method: Extraction method ("shrinking", "contour", "hybrid", "auto")
        apply_manhattan: Apply Manhattan constraints
        **kwargs: Additional config parameters
        
    Returns:
        List of WallSegment objects
    """
    method_map = {
        "shrinking": ExtractionMethod.SHRINKING,
        "contour": ExtractionMethod.CONTOUR,
        "hybrid": ExtractionMethod.HYBRID,
        "auto": ExtractionMethod.AUTO,
    }
    
    config = WallExtractorConfig(**kwargs)
    config.method = method_map.get(method, ExtractionMethod.HYBRID)
    config.apply_manhattan = apply_manhattan
    
    extractor = WallExtractor(config)
    result = extractor.extract(wall_mask)
    
    return result.walls


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create test image with walls
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Draw some walls
    cv2.rectangle(img, (50, 50), (350, 60), 255, -1)   # Top horizontal
    cv2.rectangle(img, (50, 340), (350, 350), 255, -1) # Bottom horizontal
    cv2.rectangle(img, (50, 50), (60, 350), 255, -1)   # Left vertical
    cv2.rectangle(img, (340, 50), (350, 350), 255, -1) # Right vertical
    cv2.rectangle(img, (150, 150), (160, 250), 255, -1) # Interior wall
    
    # Extract walls
    config = WallExtractorConfig(return_debug_info=True)
    extractor = WallExtractor(config)
    result = extractor.extract(img)
    
    print(f"Extracted {result.num_walls} walls:")
    for wall in result.walls:
        print(f"  {wall.method}: length={wall.length:.1f}, "
              f"angle={wall.angle:.1f}°, thickness={wall.thickness:.1f}")
    
    print(f"\nDominant angles: {result.dominant_angles}")
    
    # Visualize
    vis = extractor.visualize(img, result)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Input Wall Mask")
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Extracted Walls ({result.num_walls})")
    plt.tight_layout()
    plt.savefig("wall_extractor_test.png")
    print("\nSaved visualization to wall_extractor_test.png")
