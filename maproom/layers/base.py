import sys
import time
import datetime
import calendar

from fs.errors import ResourceNotFoundError

# Enthought library imports.
from traits.api import Any
from traits.api import Bool
from traits.api import Float
from traits.api import HasTraits
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from omnivore.utils.runtime import get_all_subclasses

# MapRoom imports
from ..library import rect

# local package imports
import state
from style import LayerStyle

import logging
log = logging.getLogger(__name__)


class Layer(HasTraits):
    """Base Layer class with some abstract methods.
    """
    # Class attributes

    use_color_cycling = False

    new_layer_index = 0

    restore_from_url = False

    grouped_indicator_prefix = u"\u271a"  # a bold plus

    # False means it is a layer without bounds like grid, scale, tile, etc.
    # These types of layers can not be added to any group and will always be
    # added to the root level of the layer tree, below any groupable layers but
    # above any ungroupable layers.
    bounded = True

    # background only applies to ungroupable layers; insert at the bottom of
    # the layer tree, below other ungroupable layers.
    background = False  # True implies ungroupable

    # True if an image or other layer where it occludes stuff behind it
    opaque = False

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

    bounds = Any(rect.NONE_RECT)

    grouped = Bool

    mouse_mode_toolbar = Str("BaseLayerToolBar")

    # this is any change that might affect the properties panel (e.g., number
    # of points selected)
    change_count = Int(0)

    load_error_string = Str

    load_warning_string = Str

    load_warning_details = Str

    manager = Any

    start_time = Float(0.0)

    end_time = Float(0.0)

    pickable = False  # is this a layer that support picking?

    transient_edit_layer = False

    visibility_items = []

    layer_info_panel = []

    selection_info_panel = []

    def _style_default(self):
        style = self.manager.get_default_style_for(self)
        if self.use_color_cycling:
            style.use_next_default_color()
        log.debug("_style_default for %s: %s" % (self.type, str(style)))
        return style

    def __repr__(self):
        return "%s (%x)" % (self.name, id(self))

    def __str__(self):
        return "%s layer '%s' (%s) %s" % (self.type, unicode(self.name).encode("utf-8"), "grouped" if self.grouped else "ungrouped", self.pretty_time_range())

    @property
    def pretty_name(self):
        if self.grouped:
            prefix = self.grouped_indicator_prefix
        else:
            prefix = ""
        return prefix + self.name

    def debug_info(self, indent=""):
        lines = []
        lines.append("%s (%x) invariant=%s dependent_of=%s grouped=%s" % (self.name, id(self), self.invariant, self.dependent_of, self.grouped))
        return ("\n%s" % indent).join(lines)

    def test_contents_equal(self, other):
        """Test routine to compare layers"""
        return self.type == other.type

    def clickable_object_info(self, picker, object_type, object_index):
        """Return info about the object of given type and index.

        Typically used to show information about the object under the cursor
        """
        if picker.is_ugrid_point_type(object_type):
            info = self.point_object_info(object_index)
            long_info = self.point_object_long_info(object_index)
        elif picker.is_ugrid_line_type(object_type):
            info = self.line_object_info(object_index)
            long_info = self.line_object_long_info(object_index)
        elif picker.is_interior_type(object_type):
            info = self.interior_object_info(object_index)
            long_info = self.interior_object_long_info(object_index)
        else:
            info = ""
            long_info = ""
        return info, long_info

    def point_object_info(self, object_index):
        return "Point %s on %s" % (object_index + 1, self.name)

    def line_object_info(self, object_index):
        return "Line %s on %s" % (object_index + 1, self.name)

    def interior_object_info(self, object_index):
        return "Polygon %s on %s" % (object_index + 1, self.name)

    def point_object_long_info(self, object_index):
        return ""

    def line_object_long_info(self, object_index):
        return ""

    def interior_object_long_info(self, object_index):
        return ""

    def show_unselected_layer_info_for(self, layer):
        """Whether or not the selected layer allows for other layers
        that are being moused-over to show info about the object the mouse
        is over.
        """
        return False

    def get_info_panel_text(self, prop):
        # Subclasses should define this to return values for any properties in
        # layer_info_panel that are read-only and can be represented as strings
        pass

    def new(self):
        # fixme -- shouldn't layer indexes, etc be controled by the layer_manager?
        # and maybe this should be using the python copy() mechanism anyway
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

    def clear_flagged(self, refresh=False):
        """Clear any items previously flagged with highlight_exception"""

    def layer_selected_hook(self):
        """Hook to allow layer to display some feedback when the layer
        selection changes to this layer"""
        pass

    def is_folder(self):
        return False

    def is_root(self):
        return False

    @property
    def is_renderable(self):
        return True

    @property
    def is_overlay(self):
        # flag if overlay layer, that is: layer that has some items in pixel
        # coordinates, not word coordinates.
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

    def children_affected_by_move(self):
        """ Returns a list of layers that will be affected by moving a control
        point.  This is used for layer groups; moving a control point of a
        group will affect all the layers in the group.
        """
        return []

    def parents_affected_by_move(self):
        """ Returns a list of layers that might need to have boundaries
        recalculated after moving this layer.
        """
        return []

    def get_undo_info(self):
        """ Return a copy of any data needed to restore the state of the layer

        It must be a copy, not a referece, so that it can be stored unchanged
        even if the layer has further changes by commands in the future.
        """
        return []

    def restore_undo_info(self, info):
        """ Restore the state of the layer given the data previously generated
        by get_undo_info
        """

    def extra_files_to_serialize(self):
        """Pathnames on the local filesystem to any files that need to be
        included in the maproom project file that can't be recreated with JSON.

        If this list is used, the first path in list must correspond to the
        path named with the file_path trait.
        """
        return []

    def serialize_json(self, index, children=False):
        """Create json representation that can restore layer.

        Layers don't know their own indexes, so the manager must pass the index
        here so it can be serialized with the rest of the data.
        """
        json = {
            'index': index,
            'invariant': self.invariant,
            'type': self.type,
            'grouped': self.grouped,
            'version': 1,
            'has encoded data': False,
            'name': self.name,
            'style': str(self.style),
            'children': [],

            # control point links only used for copy/paste of layers, not for
            # restoring from project files
            'control_point_links': self.manager.get_control_point_links(self),
        }
        if self.file_path:
            json['url'] = self.file_path
            json['mime'] = self.mime

        update = {}
        for attr, to_json in self.get_to_json_attrs():
            update[attr] = to_json()
        if update:
            json['has encoded data'] = True
            json.update(update)
        if children:
            print "CHILDREN"
            for c in self.manager.get_layer_children(self):
                json['children'].append(c.serialize_json(-999, True))
        return json

    def get_to_json_attrs(self):
        return [(m[0:-8], getattr(self, m)) for m in dir(self) if m.endswith("_to_json")]

    def get_from_json_attrs(self):
        return [(m[0:-10], getattr(self, m)) for m in dir(self) if m.endswith("_from_json")]

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
        except AttributeError:
            batch_flags.errors.append("Unsupported MapRoom save file version %s" % str(json_data['version']))
            raise
        log.debug("Restoring JSON data using %s" % name)
        method(json_data, batch_flags)
        self.from_json_sanity_check_after_load(json_data)
        self.update_bounds()
        log.debug("Restored JSON data: %s" % self.debug_info())

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
        self.grouped = json_data.get('grouped', False)
        self.style = LayerStyle()
        self.style.parse(json_data['style'])
        if 'url' in json_data:
            self.file_path = json_data['url']
            self.mime = json_data['mime']
        for attr, from_json in self.get_from_json_attrs():
            try:
                from_json(json_data)
            except KeyError:
                message = "%s not present in layer %s; attempting to continue" % (attr, self.name)
                log.warning(message)
                #batch_flags.messages.append("WARNING: %s" % message)
            except TypeError:
                log.warning("Skipping from_json function %s", from_json)

    def from_json_sanity_check_after_load(self, json_data):
        """Fix up any errors or missing data after json unserialization

        Subclasses can create missing metadata here if, say, loading an old
        version of a save file which doesn't have all the data of a newer
        version.
        """
        pass

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
        if 'url' in json_data and kls.restore_from_url:
            from maproom.layers import loaders

            log.debug("Loading layers from url %s" % json_data['url'])
            try:
                loader, layers = loaders.load_layers_from_url(json_data['url'], json_data['mime'], manager)
            except ResourceNotFoundError:
                raise RuntimeError("Failed loading from %s" % json_data['url'])

            # need to restore other metadata that isn't part of the URL load
            layers[0].unserialize_json(json_data, batch_flags)
        else:
            log.debug("Loading layers from json encoded data")
            layer = kls(manager=manager)
            layer.unserialize_json(json_data, batch_flags)
            layers = [layer]
        log.debug("returning layers: %s" % str(layers))
        return layers

    ##### time support

    def start_time_to_json(self):
        return self.start_time

    def start_time_from_json(self, json_data):
        if 'start_time' not in json_data:
            self.start_time = self.start_time_from_json_guess(json_data)
        else:
            jd = json_data['start_time']
            # check for float or datetime?
            self.start_time = jd

    def start_time_from_json_guess(self, json_data):
        # default is to set it to n/a
        return 0.0

    def end_time_to_json(self):
        return self.end_time

    def end_time_from_json(self, json_data):
        if 'end_time' not in json_data:
            self.end_time = self.end_time_from_json_guess(json_data)
        else:
            jd = json_data['end_time']
            # check for float or datetime?
            self.end_time = jd

    def end_time_from_json_guess(self, json_data):
        # default is to set it to n/a
        return 0.0

    def parse_time_to_float(self, t):
        if t is None:
            t = 0.0
        elif isinstance(t, datetime.datetime):
            t = calendar.timegm(t.timetuple())
        else:
            t = float(t)
        return t

    def set_datetime(self, start, end=None):
        self.start_time = self.parse_time_to_float(start)
        self.end_time = self.parse_time_to_float(end)

    def pretty_time(self, t):
        d = datetime.datetime.utcfromtimestamp(t)
        if t > 0:
            return d.isoformat()
        return "inf"

    def pretty_time_range(self):
        s = self.pretty_time(self.start_time)
        e = self.pretty_time(self.end_time)
        if s == "inf" and e == "inf":
            text = "(always shown)"
        else:
            if s == "inf":
                text = "(inf->"
            else:
                text = "[%s->" % s
            if e == "inf":
                text += "inf)"
            else:
                text += "%s)" % e
        return text

    #####

    def set_dependent_of(self, layer):
        self.dependent_of = layer.invariant

    def check_for_problems(self, window):
        pass

    def check_projection(self, task):
        pass

    def get_visibility_dict(self):
        # fixme: you'be GOT to be kidding me!
        # shouldn't visibility be governed by the layer manager?
        # or each layer has its own sub-layer visibility
        d = dict()
        d["layer"] = True
        d["images"] = True
        d["polygons"] = True
        d["points"] = True
        d["lines"] = True
        d["triangles"] = True
        d["labels"] = True
        # why is this not handled in the subclass????
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
                log.debug("updating bounds for %s" % parent)
                parent.update_bounds()
                parent = self.manager.get_folder_of_layer(parent)
        else:
            self.bounds = self.compute_bounding_rect()

    def update_overlay_bounds(self):
        pass

    def is_zoomable(self):
        return self.bounds != rect.NONE_RECT

    @property
    def style_name(self):
        if hasattr(self, "style_as"):
            type_name = self.style_as
        elif hasattr(self, "type"):
            type_name = self.type
        else:
            type_name = self
        return type_name

    def set_style(self, style):
        # Hook for subclasses to change colors and styles
        log.debug(" befory style copy for %s: %s" % (self.type, self.style))
        if style is None:
            style = self.manager.get_default_style_for(self)
            log.debug("style not specified, using default for %s: %s" % (self.type, style))
        else:
            log.debug("using style for %s: %s" % (self.type, style))
        self.style.copy_from(style)
        log.debug(" after style copy for %s: %s" % (self.type, self.style))

    def compute_bounding_rect(self, mark_type=state.CLEAR):
        bounds = rect.NONE_RECT
        return bounds

    def clear_all_selections(self, mark_type=state.SELECTED):
        self.clear_all_point_selections(mark_type)
        self.clear_all_line_segment_selections(mark_type)
        self.clear_all_ring_selections(mark_type)
        self.increment_change_count()

    def delete_all_selected_objects(self):
        pass

    def clear_all_point_selections(self, mark_type=state.SELECTED):
        pass

    def clear_all_line_segment_selections(self, mark_type=state.SELECTED):
        pass

    def clear_all_ring_selections(self, mark_type=state.SELECTED):
        pass

    def set_visibility_when_selected(self, layer_visibility):
        """Called when layer is selected to provide a hook if the layer has
        elements that should be visibile only when it it selected.

        """

    def clear_visibility_when_deselected(self, layer_visibility):
        """Called when layer is deselected to provide a hook if the layer has
        elements that should be visibile only when it it selected.

        """

    def set_visibility_when_checked(self, checked, project_layer_visibility):
        """Called when layer visibility changes to provide a hook if the layer
        has elements that should be visibile only when it it selected.

        """
        project_layer_visibility[self]["layer"] = checked

    def has_points(self):
        return False

    def has_selection(self):
        return False

    def has_flagged(self):
        return False

    def has_boundaries(self):
        return False

    def has_groupable_objects(self):
        return False

    def display_properties(self):
        return []

    def get_display_property(self, prop):
        return ""

    def normalize_longitude(self):
        pass

    def merge_layer_into_new(self, other_layer, depth_unit=""):
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
        layer.merge_from_source_layers(self, other_layer, depth_unit)
        return layer

    def is_mergeable_with(self, other_layer):
        return False

    def find_merge_layer_class(self, other_layer):
        return None

    def merge_from_source_layers(self, layer_a, layer_b, depth_unit=""):
        raise NotImplementedError

    def increment_change_count(self):
        self.change_count += 1
        if (self.change_count == sys.maxint):
            self.change_count = 0

    def can_crop(self):
        return False

    def crop_rectangle(self, world_rect):
        raise NotImplementedError

    def can_highlight_clickable_object(self, canvas, object_type, object_index):
        return False

    def get_highlight_lines(self, object_type, object_index):
        return []

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
        """Set up or rebuild any rendering elements prior to rendering.

        The rendering loop in BaseCanvas.render loops through all layers and
        calls this method on each of them before any layers are rendered.
        This can be used to update the world coordinates that other layers
        may depend on, used e.g. with linked control points.
        """

    def render(self, renderer,
               world_rect,
               projected_rect,
               screen_rect,
               layer_visibility,
               picker,
               control_points_only=False):
        if control_points_only:
            renderer.prepare_to_render_projected_objects()
            self.render_control_points_only(renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker)
        else:
            if hasattr(self, "render_projected"):
                renderer.prepare_to_render_projected_objects()
                self.render_projected(renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker)
            if hasattr(self, "render_screen"):
                renderer.prepare_to_render_screen_objects()
                self.render_screen(renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker)
            if picker.is_active:
                # Control points should always be clickable, so render them
                # on top of everything else in this layer when creating the
                # picker framebuffer
                renderer.prepare_to_render_projected_objects()
                self.render_control_points_only(renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker)

    def render_control_points_only(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        pass


class EmptyLayer(Layer):
    """Emply layer used when a folder has no other children.
    """
    name = Unicode("<empty folder>")

    type = Str("empty")


class ProjectedLayer(Layer):
    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        print "Layer %s doesn't have projected objects to render" % self.name


class ScreenLayer(Layer):
    def render_screen(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        print "Layer %s doesn't have screen objects to render" % self.name
