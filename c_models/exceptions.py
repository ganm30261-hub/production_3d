"""
模型模块自定义异常
"""


class ModelError(Exception):
    """模型相关错误基类"""
    pass


class BackboneError(ModelError):
    """Backbone 初始化或前向传播错误"""
    pass


class LossError(ModelError):
    """损失计算错误"""
    pass


class TrainerError(ModelError):
    """训练器配置或运行错误"""
    pass
