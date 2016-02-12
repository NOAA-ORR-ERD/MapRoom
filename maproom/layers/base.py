import os
import sys
import numpy as np

# Enthought library imports.
from traits.api import HasTraits, Any, Int, Float, List, Set, Bool, Str, Unicode, Event

from omnivore.utils.runtime import get_all_subclasses

# MapRoom imports
from ..library import rect

# local package imports
from constants import *
from style import LayerStyle

import logging
log = logging.getLogger(__name__)

class Layer(HasTraits):
    """Base Layer class with some abstract methods.
    """
    next_default_color_index = 0
    
    new_layer_index = 0
    
    # Traits
    
    name = Unicode("Empty Layer")
    
    # invariant is sort of a serial number of the layer in a LayerManager: an
    # id that doesn't change when the layer is renamed or reordered.  It is
    # unique within a particular instance of a LayerManager, and gets created
    # when the layer is added to a LayerManager.  Initial value of -999 is a
    # flag to indicate that the invariant hasn't been initialized.
    invariant = Int(-999)
    
    # the invariant of the parent layer (used in triangulation so that a
    # retriangulation will replace the older triangulation.
    dependent_of = Int(-1)
    
    # type is a string identifier that uniquely refers to a layer class.
    # Base classes should use an empty string to show that they won't be
    # serializable.
    type = Str("")
    
    mime = Str("")
    
    skip_on_insert = Bool(False)
    
    file_path = Unicode
    
    style = Any
    
    point_size = Float(4.0)
    
    selected_point_size = Float(15.0)
    
    selected_line_width = Float(10.0)
    
    bounds = Any(rect.NONE_RECT)
    
    mouse_mode_toolbar = Str("BaseLayerToolBar")
    
    # this is any change that might affect the properties panel (e.g., number
    # of points selected)
    change_count = Int(0)
    
    load_error_string = Str
    
    manager = Any

    pickable = False # is this a layer that support picking?

    visibility_items = []
    
    layer_info_panel = ["Layer name"]
    
    selection_info_panel = []
    
    def _style_default(self):
        return LayerStyle()

    def __repr__(self):
        return self.name
    
    def get_info_panel_text(self, prop):
        # Subclasses should define this to return values for any properties in
        # layer_info_panel that are read-only and can be represented as strings
        pass

    def new(self):
        ## fixme -- shouldn't layer indexes, etc be controled by the layer_manager?
        ## and maybe this should be using the python copy() mechanism anyway
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
    
    def highlight_exception(self, e):
        """Highlight items flagged in the exception"""
        pass
    
    def clear_flagged(self, refresh=False):
        """Clear any items previously flagged with highlight_exception"""
        pass
    
    def is_folder(self):
        return False
    
    def is_root(self):
        return False
    
    def can_copy(self):
        return False
    
    def can_save(self):
        """Can the layer be saved using the current filename?"""
        return False
    
    def can_save_as(self):
        """Can the layer be saved if given a new filename?"""
        return False
    
    def save_to_file(self, file_path):
        raise NotImplementedError
    
    def serialize_json(self, index):
        """Create json representation that can restore layer.
        
        Layers don't know their own indexes, so the manager must pass the index
        here so it can be serialized with the rest of the data.
        """
        json = {
            'index': index,
            'invariant': self.invariant,
            'type': self.type,
            'version': 1,
            'has encoded data': False,
            'name': self.name,
            'style': str(self.style),
            }
        if self.file_path:
            json['url'] = self.file_path
            json['mime'] = self.mime
        
        update = {}
        for attr, to_json in self.get_attrs_with_json():
            update[attr] = to_json()
        if update:
            json['has encoded data'] = True
            json.update(update)
        return json
    
    def get_attrs_with_json(self):
        return [(m[0:-8], getattr(self, m)) for m in dir(self) if m.endswith("_to_json")]
    
    def unserialize_json(self, json_data, batch_flags):
        """Restore layer from json representation.
        
        The json data passed to this function will be the subset of the json
        applicable to this layer only, so it doesn't have to deal with parsing
        anything other than what's necessary to restore itself.
        
        This is the driver routine and will call the version-specific routine
        based on the 'version' keyword in json_data.  The method should be
        named 'unserialize_json_versionX' where X is the version number.
        """
        name = "unserialize_json_version" + str(json_data['version'])
        try:
            method = getattr(self, name)
        except AttributeEror:
            batch_flags.errors.append("Unsupported MapRoom save file version %s" % str(json_data['version']))
            raise
        log.debug("Restoring JSON data using %s" % name)
        method(json_data, batch_flags)
        self.update_bounds()
    
    def unserialize_json_version1(self, json_data, batch_flags):
        """Restore layer from json representation.
        
        The json data passed to this function will be the subset of the json
        applicable to this layer only, so it doesn't have to deal with parsing
        anything other than what's necessary to restore itself.
        """
        self.name = json_data['name']
        if 'invariant' in json_data:
            # handle legacy case where invariant wasn't saved.  Restore of any
            # linked control points will be broken in cases like this
            self.invariant = json_data['invariant']
        self.style = LayerStyle()
        self.style.parse(json_data['style'])
        if 'url' in json_data:
            self.file_path = json_data['url']
            self.mime = json_data['mime']
        for attr, to_json in self.get_attrs_with_json():
            try:
                from_json = getattr(self, attr + "_from_json")
                from_json(json_data)
            except KeyError:
                message = "%s not present in layer %s; attempting to continue" % (attr, self.name)
                log.warning(message)
                batch_flags.messages.append("WARNING: %s" % message)
    
    type_to_class_defs = {}
    
    @classmethod
    def get_subclasses(cls):
        if not cls.type_to_class_defs:
            subclasses = get_all_subclasses(Layer)
            for kls in subclasses:
                layer = kls()
                if layer.type:
                    cls.type_to_class_defs[layer.type] = kls
        return cls.type_to_class_defs
            
    @classmethod
    def type_to_class(cls, type_string):
        cls.get_subclasses()
        return cls.type_to_class_defs[type_string]
    
    @classmethod
    def load_from_json(cls, json_data, manager, batch_flags=None):
        t = json_data['type']
        kls = cls.type_to_class(t)
        log.debug("load_from_json: found type %s, class=%s" % (t, kls))
        if 'url' in json_data and not json_data['has encoded data']:
            from maproom.layers import loaders
            
            log.debug("Loading layers from url %s" % json_data['url'])
            loader, layers = loaders.load_layers_from_url(json_data['url'], json_data['mime'], manager)
        else:
            log.debug("Loading layers from json encoded data")
            layer = kls(manager=manager)
            layer.unserialize_json(json_data, batch_flags)
            layers = [layer]
        log.debug("returning layers: %s" % str(layers))
        return layers
    
    def set_dependent_of(self, layer):
        self.dependent_of = layer.invariant
    
    def check_for_problems(self, window):
        pass
    
    def check_projection(self, window):
        pass

    def get_visibility_dict(self):
        ##fixme: you'be GOT to be kidding me!
        ## shouldn't visibility be governed by the layer manager?
        ## or each layer has its own sub-layer visibility
        d = dict()
        d["layer"] = True
        d["images"] = True
        d["polygons"] = True
        d["points"] = True
        d["lines"] = True
        d["triangles"] = True
        d["labels"] = True
        ## why is this not handled in the subclass????
        if self.type == "polygon":
            d["labels"] = False
            d["points"] = False
        return d
        
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in self.visibility_items:
            return self.points is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def update_bounds(self, parents=False):
        if parents:
            parent = self
            while not parent.is_root():
                parent.update_bounds()
                parent = self.manager.get_folder_of_layer(parent)
        else:
            self.bounds = self.compute_bounding_rect()
    
    def is_zoomable(self):
        return self.bounds != rect.NONE_RECT

    def set_layer_style_defaults(self):
        self.style.use_next_default_color()
    
    def set_style(self, style):
        # Hook for subclasses to change colors and styles
        self.style.copy_from(style)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT
        return bounds

    def clear_all_selections(self, mark_type=STATE_SELECTED):
        self.clear_all_point_selections(mark_type)
        self.clear_all_line_segment_selections(mark_type)
        self.clear_all_polygon_selections(mark_type)
        self.increment_change_count()

    def delete_all_selected_objects(self):
        pass

    def clear_all_point_selections(self, mark_type=STATE_SELECTED):
        pass

    def clear_all_line_segment_selections(self, mark_type=STATE_SELECTED):
        pass

    def clear_all_polygon_selections(self, mark_type=STATE_SELECTED):
        pass
    
    def set_visibility_when_selected(self, layer_visibility):
        """Called when layer is selected to provide a hook if the layer has
        elements that should be visibile only when it it selected.
        
        """
        pass

    def clear_visibility_when_deselected(self, layer_visibility):
        """Called when layer is deselected to provide a hook if the layer has
        elements that should be visibile only when it it selected.
        
        """
        pass
    
    def set_visibility_when_checked(self, checked, project_layer_visibility):
        """Called when layer visibility changes to provide a hook if the layer
        has elements that should be visibile only when it it selected.
        
        """
        pass

    def has_points(self):
        return False

    def has_selection(self):
        return False

    def has_flagged(self):
        return False
    
    def has_boundaries(self):
        return False
    
    def display_properties(self):
        return []
    
    def get_display_property(self, prop):
        return ""

    def merge_layer_into_new(self, other_layer):
        targets = []
        if self.is_mergeable_with(other_layer):
            targets.append(self.find_merge_layer_class(other_layer))
        if other_layer.is_mergeable_with(other_layer):
            targets.append(other_layer.find_merge_layer_class(self))
        if not targets:
            return
        
        # Use the most specific layer as the merge layer target
        if len(targets) > 1:
            a, b = targets[0], targets[1]
            if issubclass(a, b):
                new_layer_cls = a
            else:
                new_layer_cls = b
        else:
            new_layer_cls = targets[0]
        
        layer = new_layer_cls(manager=self.manager)
        layer.name = "Merged"
        layer.merge_from_source_layers(self, other_layer)
        return layer
    
    def is_mergeable_with(self, other_layer):
        return False
    
    def find_merge_layer_class(self, other_layer):
        return None

    def merge_from_source_layers(self, layer_a, layer_b):
        raise NotImplementedError

    def increment_change_count(self):
        self.change_count += 1
        if (self.change_count == sys.maxint):
            self.change_count = 0
    
    def can_crop(self):
        return False
    
    def crop_rectangle(self, world_rect):
        raise NotImplementedError
    
    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer
        
        """
        pass

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
        """Set up or rebuild any rendering elements prior to rendering.
        
        The rendering loop in BaseCanvas.render loops through all layers and
        calls this method on each of them before any layers are rendered.
        This can be used to update the world coordinates that other layers
        may depend on, used e.g. with linked control points.
        """
        pass

    def render(self, renderer,
               world_rect,
               projected_rect,
               screen_rect,
               layer_visibility,
               layer_index_base,
               picker,
               control_points_only=False):
        if control_points_only:
            renderer.prepare_to_render_projected_objects()
            self.render_control_points_only(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker)
        else:
            if hasattr(self, "render_projected"):
                renderer.prepare_to_render_projected_objects()
                self.render_projected(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker)
            if hasattr(self, "render_screen"):
                renderer.prepare_to_render_screen_objects()
                self.render_screen(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker)
            if picker.is_active:
                # Control points should always be clickable, so render them
                # on top of everything else in this layer when creating the
                # picker framebuffer
                renderer.prepare_to_render_projected_objects()
                self.render_control_points_only(renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker)
    
    def render_control_points_only(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        pass


class EmptyLayer(Layer):
    """Emply layer used when a folder has no other children.
    """
    name = Unicode("<empty folder>")
    
    type = Str("empty")


class Folder(Layer):
    """Layer that contains other layers.
    """
    name = Unicode("Folder")
    
    type = Str("folder")
    
    def is_folder(self):
        return True
    
    def set_visibility_when_checked(self, checked, project_layer_visibility):
        # Folders will automatically set their children's visiblity state to
        # the parent state
        children = self.manager.get_layer_children(self)
        for layer in children:
            project_layer_visibility[layer]["layer"] = checked

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        children = self.manager.get_layer_children(self)
        for layer in children:
            bounds = rect.accumulate_rect(bounds, layer.bounds)

        return bounds


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


class AnnotationLayer(Folder):
    """Layer for vector annotation image
    
    """
    name = Unicode("Annotation Layer")

    type = Str("annotation")
    
    mouse_mode_toolbar = Str("AnnotationLayerToolBar")


class ProjectedLayer(Layer):
    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        print "Layer %s doesn't have projected objects to render" % self.name


class ScreenLayer(Layer):
    def render_screen(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        print "Layer %s doesn't have screen objects to render" % self.name
