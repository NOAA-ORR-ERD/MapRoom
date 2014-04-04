import os
import sys
import numpy as np
from ..library import rect

from constants import *

class Layer(object):
    """Base Layer class with some abstract methods.
    """
    default_name = "Empty Layer"
    
    default_type = "" # "folder", "root", or the extension of the file type

    next_default_color_index = 0
    
    new_layer_index = 0

    def __init__(self, manager):
        self.lm = manager
        self.name = self.default_name
        self.type = self.default_type
        self.is_expanded = True

        self.file_path = ""
        self.depth_unit = "unknown"
        self.default_depth = DEFAULT_DEPTH
        self.color = 0
        self.point_size = 4.0
        self.selected_point_size = 15.0
        self.line_width = 2.0
        self.selected_line_width = 10.0
        self.triangle_line_width = 1.0
        self.bounds = rect.NONE_RECT
        self.load_error_string = ""

        # this is any change that might affect the properties panel (e.g., number of points selected)
        self.change_count = 0

        self.is_dirty = False

    def __repr__(self):
        return self.name

    def new(self):
        self.name = "New %s" % self.default_name
        Layer.new_layer_index += 1
        if Layer.new_layer_index > 1:
            self.name += " %d" % Layer.new_layer_index

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return True

    def get_visibility_dict(self):
        d = dict()
        d["layer"] = True
        d["images"] = True
        d["polygons"] = True
        d["points"] = True
        d["lines"] = True
        d["triangles"] = True
        d["labels"] = True
        if self.type == "bna":
            # FIXME: this was the default visibility from maproom 2; is this correct?
            d["labels"] = False
            d["points"] = False
        return d
    
    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return []
    
    def visibility_item_exists(self, label):
        """Returns True if the visibility item should be shown in the UI
        """
        return False

    def update_bounds(self):
        self.bounds = self.compute_bounding_rect()

    def determine_layer_color(self):
        if not self.color:
            print "setting layer color?"
            self.color = DEFAULT_COLORS[
                Layer.next_default_color_index
            ]

            Layer.next_default_color_index = (
                Layer.next_default_color_index + 1
            ) % len(DEFAULT_COLORS)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT
        return bounds

    def clear_all_selections(self, mark_type=STATE_SELECTED):
        self.clear_all_point_selections(mark_type)
        self.clear_all_line_segment_selections(mark_type)
        self.clear_all_polygon_selections(mark_type)
        self.increment_change_count()

    def clear_all_point_selections(self, mark_type=STATE_SELECTED):
        pass

    def clear_all_line_segment_selections(self, mark_type=STATE_SELECTED):
        pass

    def clear_all_polygon_selections(self, mark_type=STATE_SELECTED):
        pass

    def has_points(self):
        return False

    def merge_from_source_layers(self, layer_a, layer_b):
        raise NotImplementedError

    def increment_change_count(self):
        self.change_count += 1
        if (self.change_count == sys.maxint):
            self.change_count = 0
    
    def create_renderer(self, storage):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        pass

    def render_projected(self, storage, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode=False):
        print "Layer %s doesn't have projected objects to render" % self.name

    def render_screen(self, storage, world_rect, projected_rect, screen_rect, layer_visibility):
        print "Layer %s doesn't have screen objects to render" % self.name
    
    def destroy(self):
        self.lm.delete_undo_operations_for_layer(self)
        self.lm = None
