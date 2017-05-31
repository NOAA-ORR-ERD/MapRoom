import numpy as np

# Enthought library imports.
from traits.api import Str
from traits.api import Unicode

# MapRoom imports
from base import Layer
from line import LineLayer
from ..library import rect
from ..renderer import data_types

# local package imports
import state

import logging
log = logging.getLogger(__name__)


class Folder(Layer):
    """Layer that contains other layers.
    """
    name = Unicode("Folder")

    type = Str("folder")

    def is_folder(self):
        return True

    @property
    def is_renderable(self):
        return False

    def set_visibility_when_checked(self, checked, project_layer_visibility):
        # Folders will automatically set their children's visiblity state to
        # the parent state
        project_layer_visibility[self]["layer"] = checked
        children = self.manager.get_layer_children(self)
        for layer in children:
            layer.set_visibility_when_checked(checked, project_layer_visibility)


class RootLayer(Folder):
    """Root layer
    
    Only one root layer per project.
    """
    name = Unicode("Root Layer")

    type = Str("root")

    skip_on_insert = True

    def is_root(self):
        return True

    def serialize_json(self, index):
        # Root layer is never serialized
        pass


class BoundedFolder(LineLayer, Folder):
    """Layer that contains other layers.
    """
    name = Unicode("BoundedFolder")

    type = Str("boundedfolder")

    @property
    def is_renderable(self):
        return np.alen(self.points) > 0

    def set_data_from_bounds(self, bounds):
        ((l, b), (r, t)) = bounds
        if l is None:
            points = []
        else:
            points = [(l, b), (r, b), (r, t), (l, t)]
        f_points = np.asarray(points, dtype=np.float32)
        n = np.alen(f_points)
        self.points = self.make_points(n)
        log.debug("SET_DATA_FROM_BOUNDS:start %s" % self)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n] = f_points
            self.points.z = 0.0
            self.points.color = self.style.line_color
            self.points.state = 0

            lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
            n = len(lines)
            self.line_segment_indexes = self.make_line_segment_indexes(n)
            self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[0: n] = lines
            self.line_segment_indexes.color = self.style.line_color
            self.line_segment_indexes.state = 0
        log.debug("SET_DATA_FROM_BOUNDS:end %s" % self)

    def compute_bounding_rect(self, mark_type=state.CLEAR):
        bounds = rect.NONE_RECT
        log.debug("COMPUTE_BOUNDING_RECT:before %s %s" % (self, self.bounds))

        children = self.manager.get_layer_children(self)
        for layer in children:
            log.debug("  CHILD %s %s" % (layer, layer.bounds))
            bounds = rect.accumulate_rect(bounds, layer.bounds)

        log.debug("COMPUTE_BOUNDING_RECT:after %s %s" % (self, bounds))
        self.set_data_from_bounds(bounds)
        return bounds
