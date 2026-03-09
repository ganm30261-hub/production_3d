"""
自定义异常层级

为整个 floorplan 数据处理流水线定义统一的异常体系，
便于调用方精确捕获和处理不同类型的错误。
"""


class FloorplanError(Exception):
    """所有 floorplan 处理异常的基类"""
    pass


class SVGParseError(FloorplanError):
    """SVG 文件解析失败"""
    pass


class ImageLoadError(FloorplanError):
    """图像加载失败"""
    pass


class InvalidAnnotationError(FloorplanError):
    """标注数据无效（polygon 点数不足、NaN 坐标等）"""
    pass


class SplitFileError(FloorplanError):
    """划分文件缺失或格式错误"""
    pass


class ConversionError(FloorplanError):
    """格式转换过程中的错误"""
    pass


class PreprocessingError(FloorplanError):
    """预处理过程中的错误"""
    pass


class PipelineError(FloorplanError):
    """流水线编排错误"""
    pass
