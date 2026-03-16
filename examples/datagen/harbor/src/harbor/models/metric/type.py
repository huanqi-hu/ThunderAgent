from enum import Enum


class MetricType(str, Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    PASS_AT_K = "pass@k"
    UV_SCRIPT = "uv-script"
