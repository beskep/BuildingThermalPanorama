import dataclasses as dc
from typing import Generator

from .analysis import AnalysisPlotController
from .output import OutputPlotController
from .panorama import PanoramaPlotController
from .panorama import save_manual_correction
from .plot_controller import PanoPlotController
from .plot_controller import WorkingDirNotSet
from .registration import RegistrationPlotController
from .segmentation import SegmentationPlotController


@dc.dataclass
class PlotControllers:
  registration: RegistrationPlotController
  segmentation: SegmentationPlotController
  panorama: PanoramaPlotController
  analysis: AnalysisPlotController
  output: OutputPlotController

  @classmethod
  def classes(cls):
    for field in dc.fields(cls):
      yield field.name, field.type

  def controllers(self) -> Generator[PanoPlotController, None, None]:
    for field in dc.fields(self):
      yield getattr(self, field.name)
