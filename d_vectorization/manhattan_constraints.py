"""
Manhattan World Constraints Module

Implements Manhattan World assumption for floor plan vectorization.
Aligns detected wall segments to dominant orthogonal directions.

Reference:
- Manhattan World assumption: Most indoor environments have walls
  aligned to 2-3 orthogonal directions
- Used in conjunction with Hough angle detection

Features:
- Segment angle alignment
- Endpoint snapping
- Collinear segment merging
- T-junction detection and correction
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManhattanConfig:
    """Configuration for Manhattan constraints."""
    
    # Angle alignment
    angle_tolerance: float = 5.0  # Degrees tolerance for alignment
    dominant_angles: List[float] = field(default_factory=lambda: [0.0, 90.0])
    allow_diagonal: bool = True  # Allow 45° angles
    
    # Endpoint snapping
    snap_distance: float = 10.0  # Max distance for endpoint snapping
    snap_enabled: bool = True
    
    # Segment merging
    merge_enabled: bool = True
    merge_collinear_distance: float = 5.0  # Max perpendicular distance
    merge_gap_threshold: float = 15.0  # Max gap between segments to merge
    
    # T-junction handling
    tjunction_enabled: bool = True
    tjunction_threshold: float = 10.0  # Distance to detect T-junctions
    
    # Length constraints
    min_segment_length: float = 10.0
    max_segment_length: float = 5000.0


@dataclass
class AlignedSegment:
    """Represents an aligned wall segment."""
    
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float  # Aligned angle in degrees
    original_angle: float  # Original angle before alignment
    thickness: float = 1.0
    confidence: float = 1.0
    
    @property
    def p1(self) -> Tuple[float, float]:
        """Start point."""
        return (self.x1, self.y1)
    
    @property
    def p2(self) -> Tuple[float, float]:
        """End point."""
        return (self.x2, self.y2)
    
    @property
    def length(self) -> float:
        """Segment length."""
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Segment midpoint."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector."""
        d = np.array([self.x2 - self.x1, self.y2 - self.y1])
        norm = np.linalg.norm(d)
        return d / norm if norm > 0 else np.array([1.0, 0.0])
    
    @property
    def is_horizontal(self) -> bool:
        """Check if segment is approximately horizontal."""
        return abs(self.angle % 180) < 10 or abs(self.angle % 180 - 180) < 10
    
    @property
    def is_vertical(self) -> bool:
        """Check if segment is approximately vertical."""
        return abs(self.angle % 180 - 90) < 10
    
    def angle_deviation(self) -> float:
        """Angular deviation from aligned to original."""
        diff = abs(self.angle - self.original_angle)
        return min(diff, 180 - diff)
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Perpendicular distance from point to segment line."""
        px, py = point
        
        # Line equation: ax + by + c = 0
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        
        # Perpendicular distance
        num = abs(dy * px - dx * py + self.x2 * self.y1 - self.y2 * self.x1)
        den = np.sqrt(dx**2 + dy**2)
        
        return num / den if den > 0 else float('inf')
    
    def project_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Project point onto segment line."""
        px, py = point
        
        v = np.array([self.x2 - self.x1, self.y2 - self.y1])
        w = np.array([px - self.x1, py - self.y1])
        
        v_norm_sq = np.dot(v, v)
        if v_norm_sq == 0:
            return (self.x1, self.y1)
        
        t = np.dot(w, v) / v_norm_sq
        t = np.clip(t, 0, 1)
        
        return (self.x1 + t * v[0], self.y1 + t * v[1])
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_points(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        thickness: float = 1.0
    ) -> 'AlignedSegment':
        """Create segment from two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        
        return cls(
            x1=p1[0], y1=p1[1],
            x2=p2[0], y2=p2[1],
            angle=angle,
            original_angle=angle,
            thickness=thickness
        )


class ManhattanConstraints:
    """
    Apply Manhattan World constraints to wall segments.
    
    The Manhattan World assumption states that most architectural
    environments have walls aligned to 2-3 dominant orthogonal directions.
    This class aligns detected segments to these directions and performs
    cleanup operations like snapping and merging.
    
    Example:
        constraints = ManhattanConstraints()
        aligned = constraints.align(segments, dominant_angles=[0, 90])
        cleaned = constraints.clean(aligned)
    """
    
    def __init__(self, config: Optional[ManhattanConfig] = None):
        """
        Initialize Manhattan constraints.
        
        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or ManhattanConfig()
    
    def align(
        self,
        segments: List[AlignedSegment],
        dominant_angles: Optional[List[float]] = None
    ) -> List[AlignedSegment]:
        """
        Align segments to dominant Manhattan directions.
        
        Args:
            segments: Input wall segments
            dominant_angles: Override dominant angles (uses config if None)
            
        Returns:
            List of aligned segments
        """
        angles = dominant_angles or self.config.dominant_angles
        
        # Add diagonal if allowed
        if self.config.allow_diagonal:
            # Add 45° offsets from each dominant angle
            diagonals = []
            for a in angles:
                diagonals.extend([(a + 45) % 180, (a - 45) % 180])
            angles = list(set(angles + diagonals))
        
        aligned = []
        for segment in segments:
            aligned_seg = self._align_segment(segment, angles)
            if aligned_seg is not None:
                aligned.append(aligned_seg)
        
        return aligned
    
    def clean(self, segments: List[AlignedSegment]) -> List[AlignedSegment]:
        """
        Clean up aligned segments (snap, merge, fix junctions).
        
        Args:
            segments: Aligned segments
            
        Returns:
            Cleaned segments
        """
        result = segments
        
        if self.config.snap_enabled:
            result = self._snap_endpoints(result)
        
        if self.config.merge_enabled:
            result = self._merge_collinear(result)
        
        if self.config.tjunction_enabled:
            result = self._fix_tjunctions(result)
        
        # Filter by length
        result = [s for s in result
                 if self.config.min_segment_length <= s.length <= self.config.max_segment_length]
        
        return result
    
    def process(
        self,
        segments: List[AlignedSegment],
        dominant_angles: Optional[List[float]] = None
    ) -> List[AlignedSegment]:
        """
        Full processing: align and clean.
        
        Args:
            segments: Input segments
            dominant_angles: Override angles
            
        Returns:
            Processed segments
        """
        aligned = self.align(segments, dominant_angles)
        cleaned = self.clean(aligned)
        return cleaned
    
    def _align_segment(
        self,
        segment: AlignedSegment,
        angles: List[float]
    ) -> Optional[AlignedSegment]:
        """Align a single segment to nearest allowed angle."""
        # Find nearest allowed angle
        current_angle = segment.angle
        
        min_diff = float('inf')
        best_angle = current_angle
        
        for angle in angles:
            # Handle wraparound
            diff = abs(current_angle - angle)
            diff = min(diff, 180 - diff)
            
            if diff < min_diff:
                min_diff = diff
                best_angle = angle
        
        # Check if within tolerance
        if min_diff > self.config.angle_tolerance:
            # Too far from any allowed angle
            logger.debug(f"Segment at {current_angle:.1f}° not aligned "
                        f"(min diff: {min_diff:.1f}°)")
            return segment  # Return unchanged
        
        # Rotate segment to aligned angle
        return self._rotate_segment(segment, best_angle)
    
    def _rotate_segment(
        self,
        segment: AlignedSegment,
        target_angle: float
    ) -> AlignedSegment:
        """Rotate segment to target angle, keeping midpoint fixed."""
        # Get segment properties
        mid = segment.midpoint
        length = segment.length
        
        # Calculate new endpoints
        angle_rad = np.radians(target_angle)
        dx = (length / 2) * np.cos(angle_rad)
        dy = (length / 2) * np.sin(angle_rad)
        
        return AlignedSegment(
            x1=mid[0] - dx,
            y1=mid[1] - dy,
            x2=mid[0] + dx,
            y2=mid[1] + dy,
            angle=target_angle,
            original_angle=segment.original_angle,
            thickness=segment.thickness,
            confidence=segment.confidence
        )
    
    def _snap_endpoints(
        self,
        segments: List[AlignedSegment]
    ) -> List[AlignedSegment]:
        """Snap nearby endpoints together."""
        if len(segments) < 2:
            return segments
        
        # Collect all endpoints
        endpoints = []
        for i, seg in enumerate(segments):
            endpoints.append((i, 0, seg.p1))  # (segment_idx, endpoint_idx, point)
            endpoints.append((i, 1, seg.p2))
        
        # Find snap groups
        snap_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        group_id = 0
        assigned = set()
        
        for i, (seg_i, ep_i, p_i) in enumerate(endpoints):
            if i in assigned:
                continue
            
            # Start new group
            snap_groups[group_id].append((seg_i, ep_i))
            assigned.add(i)
            
            # Find nearby points
            for j, (seg_j, ep_j, p_j) in enumerate(endpoints[i+1:], i+1):
                if j in assigned:
                    continue
                
                dist = np.sqrt((p_i[0] - p_j[0])**2 + (p_i[1] - p_j[1])**2)
                
                if dist < self.config.snap_distance:
                    snap_groups[group_id].append((seg_j, ep_j))
                    assigned.add(j)
            
            group_id += 1
        
        # Apply snapping
        result = [AlignedSegment(
            x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
            angle=s.angle, original_angle=s.original_angle,
            thickness=s.thickness, confidence=s.confidence
        ) for s in segments]
        
        for members in snap_groups.values():
            if len(members) < 2:
                continue
            
            # Calculate average position
            points = []
            for seg_idx, ep_idx in members:
                seg = segments[seg_idx]
                points.append(seg.p1 if ep_idx == 0 else seg.p2)
            
            avg_x = np.mean([p[0] for p in points])
            avg_y = np.mean([p[1] for p in points])
            
            # Update endpoints
            for seg_idx, ep_idx in members:
                if ep_idx == 0:
                    result[seg_idx].x1 = avg_x
                    result[seg_idx].y1 = avg_y
                else:
                    result[seg_idx].x2 = avg_x
                    result[seg_idx].y2 = avg_y
        
        return result
    
    def _merge_collinear(
        self,
        segments: List[AlignedSegment]
    ) -> List[AlignedSegment]:
        """Merge collinear segments."""
        if len(segments) < 2:
            return segments
        
        # Group by angle
        angle_groups: Dict[float, List[AlignedSegment]] = defaultdict(list)
        for seg in segments:
            # Round angle to group
            rounded = round(seg.angle / 5) * 5
            angle_groups[rounded].append(seg)
        
        result = []
        
        for angle, group in angle_groups.items():
            merged = self._merge_group(group)
            result.extend(merged)
        
        return result
    
    def _merge_group(
        self,
        segments: List[AlignedSegment]
    ) -> List[AlignedSegment]:
        """Merge collinear segments within a group."""
        if len(segments) < 2:
            return segments
        
        merged = []
        used = set()
        
        for i, seg_i in enumerate(segments):
            if i in used:
                continue
            
            current = seg_i
            
            for j, seg_j in enumerate(segments[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if collinear
                if not self._are_collinear(current, seg_j):
                    continue
                
                # Check if they can be merged (close enough)
                gap = self._segment_gap(current, seg_j)
                
                if gap < self.config.merge_gap_threshold:
                    # Merge segments
                    current = self._merge_two_segments(current, seg_j)
                    used.add(j)
            
            merged.append(current)
        
        return merged
    
    def _are_collinear(
        self,
        seg1: AlignedSegment,
        seg2: AlignedSegment
    ) -> bool:
        """Check if two segments are collinear."""
        # Check angle difference
        angle_diff = abs(seg1.angle - seg2.angle)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        if angle_diff > self.config.angle_tolerance:
            return False
        
        # Check perpendicular distance
        dist1 = seg1.distance_to_point(seg2.midpoint)
        dist2 = seg2.distance_to_point(seg1.midpoint)
        
        max_dist = max(dist1, dist2)
        
        return max_dist < self.config.merge_collinear_distance
    
    def _segment_gap(
        self,
        seg1: AlignedSegment,
        seg2: AlignedSegment
    ) -> float:
        """Calculate gap between two segments."""
        # Find closest endpoints
        distances = [
            np.sqrt((seg1.x1 - seg2.x1)**2 + (seg1.y1 - seg2.y1)**2),
            np.sqrt((seg1.x1 - seg2.x2)**2 + (seg1.y1 - seg2.y2)**2),
            np.sqrt((seg1.x2 - seg2.x1)**2 + (seg1.y2 - seg2.y1)**2),
            np.sqrt((seg1.x2 - seg2.x2)**2 + (seg1.y2 - seg2.y2)**2),
        ]
        
        # If segments overlap, gap is 0
        min_dist = min(distances)
        
        # Check for overlap
        total_length = seg1.length + seg2.length
        endpoint_dist = max(distances)
        
        if endpoint_dist < total_length:
            return 0
        
        return min_dist
    
    def _merge_two_segments(
        self,
        seg1: AlignedSegment,
        seg2: AlignedSegment
    ) -> AlignedSegment:
        """Merge two collinear segments into one."""
        # Collect all endpoints
        points = [
            np.array([seg1.x1, seg1.y1]),
            np.array([seg1.x2, seg1.y2]),
            np.array([seg2.x1, seg2.y1]),
            np.array([seg2.x2, seg2.y2])
        ]
        
        # Find the two most distant points
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
        
        return AlignedSegment(
            x1=p1[0], y1=p1[1],
            x2=p2[0], y2=p2[1],
            angle=seg1.angle,
            original_angle=seg1.original_angle,
            thickness=max(seg1.thickness, seg2.thickness),
            confidence=(seg1.confidence + seg2.confidence) / 2
        )
    
    def _fix_tjunctions(
        self,
        segments: List[AlignedSegment]
    ) -> List[AlignedSegment]:
        """Fix T-junctions by extending segments to meet."""
        result = []
        
        for i, seg_i in enumerate(segments):
            modified = AlignedSegment(
                x1=seg_i.x1, y1=seg_i.y1,
                x2=seg_i.x2, y2=seg_i.y2,
                angle=seg_i.angle,
                original_angle=seg_i.original_angle,
                thickness=seg_i.thickness,
                confidence=seg_i.confidence
            )
            
            for j, seg_j in enumerate(segments):
                if i == j:
                    continue
                
                # Check if seg_i's endpoints are near seg_j's line
                for endpoint_idx, endpoint in [(0, seg_i.p1), (1, seg_i.p2)]:
                    dist = seg_j.distance_to_point(endpoint)
                    
                    if dist < self.config.tjunction_threshold:
                        # Project endpoint onto seg_j
                        proj = seg_j.project_point(endpoint)
                        
                        # Check if projection is on segment
                        on_segment = self._point_on_segment(proj, seg_j)
                        
                        if on_segment:
                            # Snap endpoint to projection
                            if endpoint_idx == 0:
                                modified.x1, modified.y1 = proj
                            else:
                                modified.x2, modified.y2 = proj
            
            result.append(modified)
        
        return result
    
    def _point_on_segment(
        self,
        point: Tuple[float, float],
        segment: AlignedSegment
    ) -> bool:
        """Check if point lies on segment (between endpoints)."""
        px, py = point
        
        # Check if point is between endpoints
        min_x = min(segment.x1, segment.x2) - 1
        max_x = max(segment.x1, segment.x2) + 1
        min_y = min(segment.y1, segment.y2) - 1
        max_y = max(segment.y1, segment.y2) + 1
        
        return min_x <= px <= max_x and min_y <= py <= max_y
    
    def visualize(
        self,
        image: np.ndarray,
        segments: List[AlignedSegment],
        show_angles: bool = True
    ) -> np.ndarray:
        """
        Visualize aligned segments.
        
        Args:
            image: Background image
            segments: Aligned segments
            show_angles: Show angle labels
            
        Returns:
            Visualization image
        """
        # Prepare output
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Color by angle
        def angle_to_color(angle: float) -> Tuple[int, int, int]:
            # Normalize angle to hue
            hue = int((angle / 180) * 179)
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return tuple(int(c) for c in rgb[0, 0])
        
        for seg in segments:
            color = angle_to_color(seg.angle)
            
            # Draw segment
            pt1 = (int(seg.x1), int(seg.y1))
            pt2 = (int(seg.x2), int(seg.y2))
            
            thickness = max(2, int(seg.thickness / 2))
            cv2.line(vis, pt1, pt2, color, thickness)
            
            # Draw endpoints
            cv2.circle(vis, pt1, 3, (255, 255, 255), -1)
            cv2.circle(vis, pt2, 3, (255, 255, 255), -1)
            
            # Add angle label
            if show_angles:
                mid = seg.midpoint
                label = f"{seg.angle:.0f}°"
                cv2.putText(
                    vis, label,
                    (int(mid[0]), int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
                )
        
        return vis


def apply_manhattan_constraints(
    segments: List[Tuple[float, float, float, float]],
    dominant_angles: Optional[List[float]] = None,
    **kwargs
) -> List[Tuple[float, float, float, float]]:
    """
    Convenience function to apply Manhattan constraints.
    
    Args:
        segments: List of (x1, y1, x2, y2) tuples
        dominant_angles: Dominant directions
        **kwargs: Additional config parameters
        
    Returns:
        List of aligned (x1, y1, x2, y2) tuples
    """
    config = ManhattanConfig(**kwargs)
    
    if dominant_angles is not None:
        config.dominant_angles = dominant_angles
    
    # Convert to AlignedSegment
    aligned_segments = []
    for x1, y1, x2, y2 in segments:
        seg = AlignedSegment.from_points((x1, y1), (x2, y2))
        aligned_segments.append(seg)
    
    # Apply constraints
    constraints = ManhattanConstraints(config)
    result = constraints.process(aligned_segments, dominant_angles)
    
    # Convert back
    return [(s.x1, s.y1, s.x2, s.y2) for s in result]


if __name__ == "__main__":
    # Test with synthetic segments
    import matplotlib.pyplot as plt
    
    # Create test segments with some noise
    np.random.seed(42)
    
    segments = [
        # Horizontal walls (should align to 0°)
        AlignedSegment.from_points((50, 100), (200, 103)),
        AlignedSegment.from_points((50, 200), (200, 197)),
        # Vertical walls (should align to 90°)
        AlignedSegment.from_points((50, 100), (52, 200)),
        AlignedSegment.from_points((200, 100), (198, 200)),
        # Diagonal (should align to 45° if allowed)
        AlignedSegment.from_points((100, 100), (180, 175)),
    ]
    
    # Apply constraints
    config = ManhattanConfig(allow_diagonal=True)
    constraints = ManhattanConstraints(config)
    
    aligned = constraints.process(segments, [0, 90])
    
    # Visualize
    img = np.zeros((300, 300), dtype=np.uint8)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    vis1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        pt1 = (int(seg.x1), int(seg.y1))
        pt2 = (int(seg.x2), int(seg.y2))
        cv2.line(vis1, pt1, pt2, (0, 255, 0), 2)
    
    axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Segments")
    
    # Aligned
    vis2 = constraints.visualize(img, aligned)
    axes[1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Manhattan Aligned")
    
    plt.tight_layout()
    plt.savefig("manhattan_test.png")
    print("Saved visualization to manhattan_test.png")
    
    # Print results
    print("\nAlignment results:")
    for orig, aligned_seg in zip(segments, aligned):
        print(f"  {orig.angle:.1f}° -> {aligned_seg.angle:.1f}°")
