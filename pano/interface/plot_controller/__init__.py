import dataclasses as dc
from collections.abc import Iterable

from .analysis import AnalysisPlotController
from .output import OutputPlotController
from .panorama import PanoramaPlotController, save_manual_correction
from .plot_controller import PanoPlotController
from .registration import RegistrationPlotController
from .segmentation import SegmentationPlotController
from .wwr import WWRPlotController


@dc.dataclass
class PlotControllers:
  registration: RegistrationPlotController
  segmentation: SegmentationPlotController
  panorama: PanoramaPlotController
  analysis: AnalysisPlotController
  output: OutputPlotController
  wwr: WWRPlotController

  @classmethod
  def classes(cls) -> Iterable[tuple[str, type[PanoPlotController]]]:
    for field in dc.fields(cls):
      yield field.name, field.type

  def controllers(self) -> dict[str, PanoPlotController]:
    return {f.name: getattr(self, f.name) for f in dc.fields(self)}
