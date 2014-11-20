"""
Layer type for particles

created by removing stuff from PointLayer
"""
import os
import os.path
import time
import sys
import numpy as np

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..renderer import color_to_int, data_types

from base import Layer, ProjectedLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


## FIXME: Needs to be subclassed from PointsLayer or create a new common base
## class for all this duplicated stuff.
class ParticleLayer(ProjectedLayer):
    """Layer for points/lines/polygons.
    
    """
    name = Unicode("Particle Layer")
    
    type = Str("particle")
    
    points = Any
    
    def __str__(self):
        try:
            points = len(self.points)
        except TypeError:
            points = 0
        return "%s layer '%s': %d points" % (self.type, self.name, points)

    def new(self):
        Layer.new(self)
        self.new_points()
    
    def new_points(self, num=0):
        #fixme: this shuld be done differently...
        self.determine_layer_color()
        self.points = self.make_points(num)

    def empty(self):##fixme: make a property?
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)

        return no_points

    ##fixme: can we remove all the visibility stuff???    
    def get_visibility_dict(self):
        d = ProjectedLayer.get_visibility_dict(self)
        d["labels"] = False
        return d

    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["points", "labels"]
    
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_points):
        n = np.alen(f_points)
        self.determine_layer_color()
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_SIMPLE_DTYPE).xy[0:n] = f_points
            self.points.color = self.color
            self.points.state = 0

        self.update_bounds()
    
    def can_save(self):
        ## fixme -- should be abeout save out as a nc+particles file -- maybe....
        return False
        
    def update_bounds(self):
        self.bounds = self.compute_bounding_rect()

    def make_points(self, count):
        return np.repeat(
            np.array([(np.nan, np.nan, 0, 0)], dtype=data_types.POINT_SIMPLE_DTYPE),
            count,
        ).view(np.recarray)

    def determine_layer_color(self):
        ## fixme -- this should work differently
        if not self.color:
            self.color = DEFAULT_COLORS[
                Layer.next_default_color_index
            ]

            Layer.next_default_color_index = (
                Layer.next_default_color_index + 1
            ) % len(DEFAULT_COLORS)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if (self.points is not None and len(self.points) > 0):
            if (mark_type == STATE_NONE):
                points = self.points
            else:
                points = self.points[self.get_selected_point_indexes(mark_type)]
            # fixme -- could be more eficient numpy-wise
            l = points.x.min()
            r = points.x.max()
            b = points.y.min()
            t = points.y.max()
            bounds = rect.accumulate_rect(bounds, ((l, b), (r, t)))

        return bounds
    
    def get_state(self, index):
        ##fixme -- is this needed?
        return self.points.state[index]
    
    def create_renderer(self, renderer):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        if self.points != None and renderer.point_renderer is None:
            renderer.rebuild_point_renderer(self, create=True)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        log.log(5, "Rendering ParticleLayer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
        if (not layer_visibility["layer"]):
            return

        # the points
        if (renderer.point_renderer != None):
            renderer.point_renderer.render(layer_index_base + renderer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET,
                                           pick_mode,
                                           self.point_size,
                                           layer_visibility["points"],
                                           )

