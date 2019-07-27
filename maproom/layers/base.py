import sys
import time
import datetime
import calendar

from sawx.utils.runtime import get_all_subclasses
from sawx.loader import identify_file

# MapRoom imports
from ..library import rect
from ..styles import LayerStyle

# local package imports
from . import state

import logging
log = logging.getLogger(__name__)


class Layer:
    """Base Layer class with some abstract methods.
    """
    # Class attributes

    use_color_cycling = False

    new_layer_index = 0

    restore_from_url = False

    grouped_indicator_prefix = "\u271a"  # a bold plus

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

    name = "Empty"

    # type is a string identifier that uniquely refers to a layer class.
    # Base classes should use an empty string to show that they won't be
    # serializable.
    type = ""

    mouse_mode_toolbar = "BaseLayerToolBar"

    skip_on_insert = False

    has_control_points = False

    pickable = False  # is this a layer that support picking?

    transient_edit_layer = False

    draw_on_top_when_selected = False

    visibility_items = []

    layer_info_panel = []

    selection_info_panel = []

    def __init__(self, manager):
        self.manager = manager

        # invariant is sort of a serial number of the layer in a LayerManager:
        # an id that doesn't change when the layer is renamed or reordered.  It
        # is unique within a particular instance of a LayerManager, and gets
        # created when the layer is added to a LayerManager.  Initial value of
        # -999 is a flag to indicate that the invariant hasn't been
        # initialized.
        self.invariant = -999

        # the invariant of the parent layer (used in triangulation so that a
        # retriangulation will replace the older triangulation.
        self.dependent_of = -1

        self.mime = ""
        self.file_path = ""

        self.style = self.calc_initial_style()
        self.bounds = rect.NONE_RECT
        self.grouped = False

        # this is any change that might affect the properties panel (e.g.,
        # number of points selected)
        self.change_count = 0

        self.load_error_string = ""
        self.load_warning_string = ""
        self.load_warning_details = ""

        self.start_time = 0.0
        self.end_time = 0.0

        self.rebuild_needed = False

    def calc_initial_style(self):
        style = self.manager.get_default_style_for(self)
        if self.use_color_cycling:
            style.use_next_default_color()
        log.debug("_style_default for %s: %s" % (self.type, str(style)))
        return style

    def __repr__(self):
        return "%s (%x)" % (self.name, id(self))

    def __str__(self):
        return "%s layer '%s' (%s) %s" % (self.type, self.name, "grouped" if self.grouped else "ungrouped", self.pretty_time_range())

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

    def clickable_object_info(self, picker, picker_type, object_index):
        """Return info about the object of given type and index.

        Typically used to show information about the object under the cursor
        """
        if picker.is_ugrid_point_type(picker_type):
            info = self.point_object_info(object_index)
            long_info = self.point_object_long_info(object_index)
        elif picker.is_ugrid_line_type(picker_type):
            info = self.line_object_info(object_index)
            long_info = self.line_object_long_info(object_index)
        elif picker.is_interior_type(picker_type):
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

    def layer_deselected_hook(self):
        """Hook to allow layer to display some feedback when the layer
        selection is removed from this layer"""
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

    @property
    def contains_overlays(self):
        return False

    @property
    def can_rotate(self):
        return False

    def can_copy(self):
        return False

    def can_save(self):
        """Can the layer be saved, assuming it is given a valid filename?"""
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

    def can_reparent_to(self, potential_parent_layer):
        """ Check to see if the current layer can be reparented to the specified
        layer. Used by the layer_tree_control to potentially abort the reordering
        of the layer hierarchy if e.g. a annotation object is moved outside an
        annotation layer.
        """
        return True

    @property
    def can_contain_annotations(self):
        """ Can this layer contain annotation layer objects?"""
        return False

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
            for c in self.manager.get_layer_children(self):
                json['children'].append(c.serialize_json(-999, True))
        return json

    def get_to_json_attrs(self):
        return [(m[0:-8], getattr(self, m)) for m in dir(self) if m.endswith("_to_json")]

    def get_from_json_attrs(self):
        return [(m[0:-10], getattr(self, m)) for m in dir(self) if m.endswith("_from_json") and m != "load_from_json"]

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
            log.debug(f"{self.name}: restoring json using {from_json}")
            try:
                from_json(json_data)
            except KeyError as e:
                message = f"{attr} not present in layer {self.name}; attempting to continue ({str(e)})"
                log.warning(message)
                #batch_flags.messages.append("WARNING: %s" % message)
            except TypeError as e:
                log.warning(f"Skipping from_json function {from_json} ({str(e)})")

    def from_json_sanity_check_after_load(self, json_data):
        """Fix up any errors or missing data after json unserialization

        Subclasses can create missing metadata here if, say, loading an old
        version of a save file which doesn't have all the data of a newer
        version.
        """
        pass

    def restore_layer_relationships_after_load(self):
        """Restore any layer relationships after all layers have been loaded

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
                if kls.type:
                    cls.type_to_class_defs[kls.type] = kls
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
            url = json_data['url']
            file_metadata = identify_file(url)
            # print(f"file metadata: {file_metadata}")
            loader = file_metadata["loader"]

            log.debug(f"Loading layers from {url} using {loader}")
            try:
                undo_info = loader.load_layers_from_uri(url, manager)
            except OSError:
                raise RuntimeError(f"Failed loading from {url}")

            # need to restore other metadata that isn't part of the URL load
            layers = undo_info.data[0]
            layers[0].unserialize_json(json_data, batch_flags)
        else:
            log.debug("Loading layers from json encoded data")
            layer = kls(manager=manager)
            layer.unserialize_json(json_data, batch_flags)
            layers = [layer]
        log.debug("returning layers: %s" % str(layers))
        return layers

    def dependent_of_to_json(self):
        return self.dependent_of

    def dependent_of_from_json(self, json_data):
        self.dependent_of = json_data.get('dependent_of', -1)

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

    def is_visible_at_time(self, time):
        return time is None or self.start_time == 0 or (self.start_time <= time and (self.end_time == 0.0 or time < self.end_time))

    def is_visible_in_time_range(self, begin, end):
        return begin is None or (self.start_time >= begin and self.start_time < end) or (self.end_time >= begin and self.end_time < end) or (self.start_time < begin and (self.end_time == 0.0 or self.end_time >= end))

    #####

    def set_dependent_of(self, layer):
        self.dependent_of = layer.invariant

    def check_for_problems(self):
        pass

    def check_projection(self):
        pass

    def get_visibility_dict(self, project):
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
        if other_layer.is_mergeable_with(self):
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
        if (self.change_count == sys.maxsize):
            self.change_count = 0

    def can_crop(self):
        return False

    def crop_rectangle(self, world_rect):
        raise NotImplementedError

    def can_highlight_clickable_object(self, canvas, picker_type, object_index):
        return False

    def get_highlight_lines(self, picker_type, object_index):
        return []

    def subset_using_logical_operation(self, operation):
        pass

    #### output

    def can_output_feature_list(self):
        return False

    def calc_output_feature_list(self):
        pass

    #### rendering

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

    ##### User interface

    def calc_context_menu_desc(self, picker_type, object_index, world_point):
        """Return actions that are appropriate when the right mouse button
        context menu is displayed over a particular object within the layer.
        """
        log.warning(f"no popup actions for {self}")
        return []


class EmptyLayer(Layer):
    """Emply layer used when a folder has no other children.
    """
    name = "<empty folder>"

    type = "empty"


class ProjectedLayer(Layer):
    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        log.debug("Layer %s doesn't have projected objects to render" % self.name)


class ScreenLayer(Layer):
    def render_screen(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        log.debug("Layer %s doesn't have screen objects to render" % self.name)


class StickyLayer(ScreenLayer):
    layer_info_panel = ["X location", "Y location"]

    mouse_mode_toolbar = "StickyLayerToolBar"

    skip_on_insert = True

    bounded = False

    background = True

    pickable = True

    x_offset = 10
    y_offset = 10

    def __init__(self, manager, x_percentage=None, y_percentage=None):
        super().__init__(manager)
        self.x_percentage = 0.0 if x_percentage is None else x_percentage
        self.y_percentage = 0.0 if y_percentage is None else y_percentage
        self.usable_screen_size = (100, 100)

    def x_percentage_to_json(self):
        return self.x_percentage

    def x_percentage_from_json(self, json_data):
        self.x_percentage = json_data['x_percentage']

    def y_percentage_to_json(self):
        return self.y_percentage

    def y_percentage_from_json(self, json_data):
        self.y_percentage = json_data['y_percentage']

    def get_undo_info(self):
        return (self.x_percentage, self.y_percentage)

    def restore_undo_info(self, info):
        self.x_percentage, self.y_percentage = info

    def move_layer(self, orig_x_percentage, orig_y_percentage, dx, dy):
        log.debug(f"move_layer: {orig_x_percentage}, {orig_y_percentage}, {dx}, {dy}")
        sw, sh = self.usable_screen_size
        px = (orig_x_percentage * sw + dx) / sw
        py = (orig_y_percentage * sh + dy) / sh
        # leave unchanged if off screen
        if 0 < px < 1.0:
            self.x_percentage = px
        if 0 < py < 1.0:
            self.y_percentage = py

    def dragging_layer(self, dx, dy):
        from ..screen_object_commands import MoveStickyLayerCommand
        cmd = MoveStickyLayerCommand(self, dx, dy)
        return cmd

    def rotating_layer(self, dx, dy):
        return None

    def calc_bounding_box(self, s_r, pixel_width, pixel_height):
        """Calculate the OpenGL-referenced bounding box: origin at the lower
        left. The Y-coords must be subtracted from the screen height if drawing
        in wx pixel coords.
        """
        w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - pixel_width
        h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset - pixel_height

        x = s_r[0][0] + (w * self.x_percentage) + self.x_offset
        y = s_r[0][1] + (h * self.y_percentage) + self.y_offset

        self.usable_screen_size = (w, h)
        bounding_box = ((x, y), (x + pixel_width, y + pixel_height))
        return bounding_box


class StickyResizableLayer(StickyLayer):
    layer_info_panel = ["X location", "Y location", "Magnification"]

    def __init__(self, manager, x_percentage=None, y_percentage=None, magnification=None):
        super().__init__(manager, x_percentage, y_percentage)
        self.magnification = 0.2 if magnification is None else magnification

    def magnification_to_json(self):
        return self.magnification

    def magnification_from_json(self, json_data):
        self.magnification = json_data['magnification']
