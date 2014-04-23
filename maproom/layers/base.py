import os
import sys
import numpy as np

# Enthought library imports.
from traits.api import HasTraits, Any, Int, Float, List, Set, Bool, Str, Unicode, Event

# MapRoom imports
from ..library import rect

# local package imports
from constants import *

class Layer(HasTraits):
    """Base Layer class with some abstract methods.
    """
    next_default_color_index = 0
    
    new_layer_index = 0
    
    # Traits
    
    name = Unicode("Empty Layer")
    
    type = Str("")
    
    file_path = Unicode
    
    default_depth = Float(1.0)
    
    color = Int(0)
    
    point_size = Float(4.0)
    
    selected_point_size = Float(15.0)
    
    line_width = Float(2.0)
    
    selected_line_width = Float(10.0)
    
    triangle_line_width = Float(1.0)
    
    bounds = Any(rect.NONE_RECT)
    
    mouse_selection_mode = Str("BaseLayer")
    
    # this is any change that might affect the properties panel (e.g., number
    # of points selected)
    change_count = Int(0)
    
    # FIXME: is really a view parameter so instead of affecting all views,
    # should be stored in the view somehow
    is_expanded = Bool(True)
    
    load_error_string = Str
    
    manager = Any

    def __repr__(self):
        return self.name

    def new(self):
        self.name = "New %s" % self.name
        Layer.new_layer_index += 1
        if Layer.new_layer_index > 1:
            self.name += " %d" % Layer.new_layer_index

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return True
    
    def is_root(self):
        return False

    def get_visibility_dict(self):
        d = dict()
        d["layer"] = True
        d["images"] = True
        d["polygons"] = True
        d["points"] = True
        d["lines"] = True
        d["triangles"] = True
        d["labels"] = True
        if self.type == ".bna":
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
    
    def is_zoomable(self):
        print "bounds! %s" % str(self.bounds)
        return self.bounds != rect.NONE_RECT

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

    def render(self, opengl_renderer, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode=False):
        if hasattr(self, "render_projected"):
            opengl_renderer.prepare_to_render_projected_objects()
            self.render_projected(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode)
        if hasattr(self, "render_screen"):
            opengl_renderer.prepare_to_render_screen_objects()
            self.render_screen(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode)


class Folder(Layer):
    """Layer that contains other layers.
    """
    name = Unicode("Folder")
    
    type = Str("folder")


class RootLayer(Folder):
    """Root layer
    
    Only one root layer per project.
    """
    name = Unicode("Root Layer")
    
    type = Str("root")
    
    def is_root(self):
        return True


class ProjectedLayer(Layer):
    def render_projected(self, storage, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode=False):
        print "Layer %s doesn't have projected objects to render" % self.name


class ScreenLayer(Layer):
    def render_screen(self, storage, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, pick_mode=False):
        print "Layer %s doesn't have screen objects to render" % self.name
