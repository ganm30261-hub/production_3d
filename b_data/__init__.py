"""
data — CubiCasa5k 户型图数据处理包

模块:
    exceptions      自定义异常层级
    schemas         数据结构定义与校验
    cubicasa_parser SVG / House 解析器
    coco_converter  COCO 格式转换
    yolo_converter  YOLO 格式转换
    preprocessing   裁剪、滑动窗口、数据增强
    pipeline        一站式流水线编排
"""

from .exceptions import (
    FloorplanError,
    SVGParseError,
    ImageLoadError,
    InvalidAnnotationError,
    SplitFileError,
    ConversionError,
    PreprocessingError,
    PipelineError,
)
from .schemas import (
    RoomAnnotation,
    WallAnnotation,
    OpeningAnnotation,
    StructuralAnnotation,
    SampleData,
    ConversionStats,
)
from .cubicasa_parser import CubiCasaParser, ParserConfig, SVGParser
from .coco_converter import COCOConverter, COCOConverterConfig
from .yolo_converter import YOLOConverter, YOLOConverterConfig
from .preprocessing import (
    Preprocessor,
    PreprocessConfig,
    WallCropper,
    SlidingWindowProcessor,
    DataAugmentor,
)
from .pipeline import FloorplanPipeline, PipelineConfig

__all__ = [
    # Exceptions
    "FloorplanError", "SVGParseError", "ImageLoadError",
    "InvalidAnnotationError", "SplitFileError", "ConversionError",
    "PreprocessingError", "PipelineError",
    # Schemas
    "RoomAnnotation", "WallAnnotation", "OpeningAnnotation",
    "StructuralAnnotation", "SampleData", "ConversionStats",
    # Parser
    "CubiCasaParser", "ParserConfig", "SVGParser",
    # Converters
    "COCOConverter", "COCOConverterConfig",
    "YOLOConverter", "YOLOConverterConfig",
    # Preprocessing
    "Preprocessor", "PreprocessConfig",
    "WallCropper", "SlidingWindowProcessor", "DataAugmentor",
    # Pipeline
    "FloorplanPipeline", "PipelineConfig",
]
