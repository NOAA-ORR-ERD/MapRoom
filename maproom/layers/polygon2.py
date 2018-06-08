import numpy as np

# Enthought library imports.
from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..library.scipy_ckdtree import cKDTree
from ..library.Boundary import Boundaries
from ..library.shapely_utils import shapely_to_polygon
from ..renderer import data_types
from ..command import UndoInfo
from ..mouse_commands import DeleteLinesCommand, MergePointsCommand

from . import LineLayer, Folder, state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PolygonBoundaryLayer(LineLayer):
    """Layer for points/lines/polygons.

    """
    name = "Polygon Boundary"

    type = "polygon_boundary"

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]


class HoleLayer(LineLayer):
    """Layer for points/lines/polygons.

    """
    name = "Polygon Hole"

    type = "polygon_hole"

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]


class PolygonParentLayer(Folder, LineLayer):
    """Parent folder for group of polygons. Direct children will be
    PolygonBoundaryLayer objects (with grandchildren will be HoleLayers) or
    PointLayer objects.

    """
    name = "Polygon"

    type = "polygon_folder"

    visibility_items = ["points", "lines", "labels"]

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    def has_groupable_objects(self):
        return True

    def get_info_panel_text(self, prop):
        if prop == "Point count":
            total = 0
            for child in self.get_child_layers():
                try:
                    total += len(self.points)
                except TypeError:
                    pass
            return str(total)
        elif prop == "Line segment count":
            total = 0
            for child in self.get_child_layers():
                try:
                    total += len(self.line_segment_indexes)
                except TypeError:
                    pass
            return str(total)
        return None
