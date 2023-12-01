"""
실/열화상 정합

metrics: 두 영상 간 유사도 평가
evaluation, feature_match, sitk: 실/열화상 정합 방법/패러미터 테스트를 위한 코드
"""

from ._evaluation import BaseEvaluation
from .metrics import (
  MutualInformation,
  calculate_all_metrics,
  compute_ncc,
  compute_rmse,
  compute_sse,
  image_entropy,
)
