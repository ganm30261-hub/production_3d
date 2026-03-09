"""
FloorPlan3D Vectorization Module

Combines paper's Shrinking Algorithm with original Manhattan World constraints
for robust wall segment extraction from floor plan images.

Key Components:
- HoughAngleDetector: Dominant angle detection using Hough Transform
- ShrinkingAlgorithm: Paper method for rectangle fitting (MVA 2023)
- RDPSimplifier: Douglas-Peucker polygon simplification
- ManhattanConstraints: Angle alignment to dominant directions
- WallExtractor: Unified extraction pipeline
"""

from .hough_transform import (
    HoughAngleDetector,
    HoughConfig,
    AngleCluster,
)

from .shrinking_algorithm import (
    ShrinkingAlgorithm,
    ShrinkingConfig,
    FittedRectangle,
)

from .rdp_simplification import (
    RDPSimplifier,
    RDPConfig,
    SimplifiedPolygon,
)

from .manhattan_constraints import (
    ManhattanConstraints,
    ManhattanConfig,
    AlignedSegment,
)

from .wall_extractor import (
    WallExtractor,
    WallExtractorConfig,
    WallSegment,
    ExtractionResult,
)

__all__ = [
    # Hough Transform
    "HoughAngleDetector",
    "HoughConfig",
    "AngleCluster",
    # Shrinking Algorithm (Paper)
    "ShrinkingAlgorithm",
    "ShrinkingConfig",
    "FittedRectangle",
    # RDP Simplification (Original)
    "RDPSimplifier",
    "RDPConfig",
    "SimplifiedPolygon",
    # Manhattan Constraints (Original)
    "ManhattanConstraints",
    "ManhattanConfig",
    "AlignedSegment",
    # Wall Extractor (Fusion)
    "WallExtractor",
    "WallExtractorConfig",
    "WallSegment",
    "ExtractionResult",
]
