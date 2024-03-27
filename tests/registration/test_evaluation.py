# ruff: noqa: PLR6301 PLR2004 N806
import numpy as np
import pytest

from pano import registration

metrics = registration.evaluation.metrics


class TestEval:
  img1 = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
  ])

  img2 = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
  ])

  img3 = np.array([
    [2.0, 3.0],
    [4.0, 5.0],
  ])

  def test_sum_of_squared_difference(self):
    assert metrics.compute_sse(self.img1, self.img2, norm=False) == 22

  def test_root_mean_square_error(self):
    expected = np.sqrt(22 / 4.0)
    assert metrics.compute_rmse(self.img1, self.img2, norm=False) == pytest.approx(
      expected
    )

  def test_normalized_cross_correlation(self):
    assert metrics.compute_ncc(self.img1, self.img2) == 0.0
    assert metrics.compute_ncc(self.img1, self.img3) == pytest.approx(1.0)


class TestMI:
  img1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
  ])

  img2 = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
  ])

  bins = 2
  base = 2
  mi = metrics.MutualInformation(img1, img2, bins, base)

  expected_entropy = 1.0  # -2 * 0.5 * np.log2(0.5)

  def test_np_histogram(self):
    hist, _edges = np.histogram(self.img1.ravel(), bins=self.bins)

    assert hist == pytest.approx(np.array([2, 2]))

  def test_np_histogram2d(self):
    H, _xedges, _yedges = np.histogram2d(
      x=self.img1.ravel(), y=self.img2.ravel(), bins=2
    )
    expected = np.array([[0, 2], [2, 0]])
    assert pytest.approx(expected) == H

  def test_np_mmi(self):
    pxy = np.array([[1, 2], [3, 4]])
    px = np.sum(pxy, axis=0)
    py = np.sum(pxy, axis=1).reshape([-1, 1])

    assert pxy / px == pytest.approx(
      np.array([
        [1 / 4, 2 / 6],
        [3 / 4, 4 / 6],
      ])
    )
    assert pxy / py == pytest.approx(
      np.array([
        [1 / 3, 2 / 3],
        [3 / 7, 4 / 7],
      ])
    )

    pxy_pxpy = pxy / (px * py)
    assert pxy_pxpy == pytest.approx(
      np.array([
        [1 / 12, 2 / 18],
        [3 / 28, 4 / 42],
      ])
    )

  def test_image_entropy(self):
    entropy = metrics.image_entropy(self.img1, bins=self.bins)
    assert entropy == pytest.approx(pytest.approx(self.expected_entropy))

  def test_mi_image_entropy(self):
    assert self.mi.image1_entropy == pytest.approx(self.expected_entropy)
    assert self.mi.image2_entropy == pytest.approx(self.expected_entropy)

  def test_mi_joint_hist(self):
    expected = np.array([[0, 2], [2, 0]])
    assert self.mi.joint_hist == pytest.approx(expected)

  def test_mi_joint_entropy(self):
    assert self.mi.joint_entropy == pytest.approx(pytest.approx(self.expected_entropy))

  def test_mi_mutual_information(self):
    # ` expected = 2 * self.expected_entropy - self.expected_entropy
    # ` E(img1) == E(img2) == E(img1, img2)

    assert self.mi.mutual_information == pytest.approx(self.expected_entropy)
