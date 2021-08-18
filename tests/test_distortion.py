import numpy as np
import pytest

from pano.distortion import radial


class TestRadialDistortionModel:
  x0 = 0.0  # 왜곡 중심 좌표
  y0 = 0.0
  k1 = -1e-5  # 왜곡 계수

  # 왜곡 전 직선 ax+by+c=0의 계수들
  abc = np.array([
      [1, 1, 1],
      [1, 0, 1],
      [-1, 1, 1],
      [0, 1, -1],
  ])

  # 왜곡된 원호 (x - xc)^2 + (y - yc)^2 = r^2의 계수들
  xc = -abc[:, 0] / (2 * abc[:, 2] * k1) + x0  # x_ci = x0 - a_i/(2*c_i*k1)
  yc = -abc[:, 1] / (2 * abc[:, 2] * k1) + y0  # y_ci = y0 - b_i/(2*c_i*k1)
  r = np.sqrt(
      (np.square(abc[:, 0]) + np.square(abc[:, 1])) / np.square(
          (2 * abc[:, 2] * k1)) -
      1 / k1)  # r_i = (a_i / (2*c_i*k1))^2 + (b_i / (2*c_i*k1))^2 - 1 / k1

  xcycr = np.vstack([xc, yc, r]).T

  def test_static_estimation(self):
    x0y0k1 = radial.RadialDistortionModel.static_estimate(self.xcycr)

    assert x0y0k1[0] == pytest.approx(self.x0, rel=1e-4, abs=1e-8)
    assert x0y0k1[1] == pytest.approx(self.y0, rel=1e-4, abs=1e-8)
    assert x0y0k1[2] == pytest.approx(self.k1, rel=1e-4, abs=1e-8)


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_distortion'])
