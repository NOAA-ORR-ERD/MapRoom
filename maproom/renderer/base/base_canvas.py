import time

import math
import numpy as np

from .renderer import BaseRenderer
from .picker import NullPicker
import maproom.library.rect as rect
from maproom.library.projection import Projection
import maproom.preferences

from sawx.ui import progress_dialog

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class BaseCanvas(object):
    """Abstract class defining the rendering interface

    """

    def __init__(self, project):
        self.project = project

        self.layer_renderers = {}

        self.init_overlay()

        self.picker = self.new_picker()
        self.hide_picker_layer = None

        self.screen_rect = rect.EMPTY_RECT

        # limiting value to prevent screen cluttering: if more characters would
        # be visible in all labels on screen, disable the display of the labels
        self.max_label_characters = 1000

        self.debug_show_picker_framebuffer = False

        self.projection = Projection(maproom.preferences.DEFAULT_PROJECTION_STRING)

        # two variables keep track of what's visible on the screen:
        # (1) the projected point at the center of the screen (Seattle Area)
        # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)
        self.set_viewport((-13664393.732, 6048089.93218), 1000)

        # mouse handler events
        self.mouse_handler = None  # defined in subclass

    def get_native_control(self):
        raise NotImplementedError

    def debug_structure(self, indent=""):
        lines = ["layer_canvas summary:"]
        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)
        lines.append("screen rect: %s" % (str(s_r)))
        lines.append("projection: %s" % (str(self.projection)))
        lines.append("projected rect: %s" % (str(p_r)))
        lines.append("projected center: %s" % (str(self.projected_point_center)))
        lines.append("projected units per pixel: %s" % (str(self.projected_units_per_pixel)))
        lines.append("zoom_level: %s" % (str(self.zoom_level)))
        lines.append("world rect: %s" % (str(w_r)))
        return ("\n" + indent).join(lines)

    def init_overlay(self):
        pass

    def new_picker(self):
        return NullPicker()

    def new_renderer(self, layer):
        return BaseRenderer(self, layer)

    def get_renderer(self, layer):
        return self.layer_renderers[layer]

    def update_renderer(self, layer):
        if layer.is_renderable and layer not in self.layer_renderers:
            log.debug("update_renderers: rebuilding layer %s" % layer)
            r = self.new_renderer(layer)
            layer.rebuild_renderer(r)
            self.layer_renderers[layer] = r

    def update_renderers(self):
        for layer in self.project.layer_manager.flatten():
            self.update_renderer(layer)

    def remove_renderer_for_layer(self, layer):
        if layer in self.layer_renderers:
            del self.layer_renderers[layer]

    def rebuild_renderers(self):
        layers = list(self.layer_renderers)  # shallow copy; we'll be deleting from the list
        for layer in layers:
            self.remove_renderer_for_layer(layer)
        self.update_renderers()

    def rebuild_renderer_for_layer(self, layer, in_place=False):
        log.debug(f"rebuild_renderer_for_layer: {layer}")
        if layer in self.layer_renderers:
            r = self.layer_renderers[layer]
            layer.rebuild_renderer(r, in_place)
            log.debug(f"renderer rebuilt for {layer}")
        else:
            log.warning("layer %s isn't in layer_renderers!" % layer)
            for layer in list(self.layer_renderers.keys()):
                log.warning("  layer: %s" % layer)

    def begin_rendering_screen(self, projected_rect, screen_rect):
        self.screen_rect = screen_rect
        self.s_w = rect.width(screen_rect)
        self.s_h = rect.height(screen_rect)
        self.projected_rect = projected_rect
        p_w = rect.width(projected_rect)
        p_h = rect.height(projected_rect)

        if (self.s_w <= 0 or self.s_h <= 0 or p_w <= 0 or p_h <= 0):
            log.error(f"begin_rendering_screen: nonsensical projection! {self.projected_rect}")
            return False

        self.prepare_screen_viewport()
        return True

    def prepare_screen_viewport(self):
        pass

    def finalize_rendering_screen(self):
        pass

    def set_screen_rendering_attributes(self):
        pass

    def is_canvas_pickable(self):
        return True

    def begin_rendering_picker(self, screen_rect):
        self.picker.prepare_to_render(screen_rect)
        self.set_picker_rendering_attributes()
        self.prepare_picker_viewport()

    def prepare_picker_viewport(self):
        pass

    def set_picker_rendering_attributes(self):
        pass

    def done_rendering_picker(self):
        self.picker.done_rendering()
        self.set_screen_rendering_attributes()

    def get_object_at_mouse_position(self, screen_point):
        if rect.contains_point(self.screen_rect, screen_point):
            return self.picker.get_object_at_mouse_position(screen_point)
        return None

    def get_edit_layer(self):
        # Subclasses should return the selected layer, to be used to render the
        # selected layer's control points above all others, regardless of the
        # stacking order of the layers
        return None

    def hide_from_picker(self, layer):
        self.hide_picker_layer = layer

    def is_screen_ready(self):
        return True

    def render(self, event=None):
        if not self.is_screen_ready():
            return

        if self.project.in_batch_processing:
            log.debug("render: Skipping render in batch processing mode")
            return

        # Get interactive console here:
#        import traceback
#        traceback.print_stack();
#        import code; code.interact( local = locals() )
        t0 = time.perf_counter()
        log.debug("render: RENDERING at %f" % t0)
        self.update_renderers()

        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)
        log.debug(f"render: screen: {s_r}\nprojected: {p_r}\nworld: {w_r}")

        if not self.begin_rendering_screen(p_r, s_r):
            return

        selected = self.get_edit_layer()
        all_layers = list(enumerate(self.project.layer_manager.flatten()))
        all_layers.reverse()

        # Update any linked control points by first looping through all layers
        # to update the world position, then updating the links.
        layer_draw_order = []
        for i, layer in all_layers:
            vis = self.project.layer_visibility[layer]
            if layer.is_renderable:
                if layer not in self.layer_renderers:
                    # renderer may not yet exist on BoundedFolder layers the
                    # first time through because dependent layers just fed
                    # their info up to the parent in update_renderers above,
                    # but at the time the BoundedFolder went through the list
                    # it wasn't renderable.
                    self.update_renderer(layer)
                renderer = self.layer_renderers[layer]
                layer.pre_render(renderer, w_r, p_r, s_r, vis)
                layer_draw_order.append((i, layer))
        affected_layers = self.project.layer_manager.update_linked_control_points()
        for layer in affected_layers:
            log.debug(f"render: rebuilding layer {layer} to update control points")
            renderer = self.layer_renderers[layer]
            layer.rebuild_renderer(renderer, True)

        elapsed = time.perf_counter() - t0
        if progress_dialog.is_active():
            log.debug(f"render: ABORTING RENDERING because progress dialog is active; total time = {elapsed}")
            return

        log.debug(f"render: FINISHED REBUILDING LAYERS; total time = {elapsed}")

        null_picker = NullPicker()

        def render_layers(layer_order, picker=null_picker):
            log.debug("render: rendering at time %s, range %s" % (self.project.timeline.current_time, self.project.timeline.selected_time_range))
            delayed_pick_layer = None
            control_points_layer = None
            for i, layer in layer_order:
                renderer = self.layer_renderers[layer]
                vis = self.project.layer_visibility[layer]
                if not vis["layer"]:
                    # short circuit displaying anything if entire layer is hidden
                    continue
                log.debug("render: valid times: %s - %s; layer=%s" % (layer.start_time, layer.end_time, layer))
                if not self.project.is_layer_visible_at_current_time(layer):
                    log.debug("render: skipping layer %s; not in currently displayed time")
                    continue
                if picker.is_active:
                    if layer.pickable:
                        if layer == self.hide_picker_layer:
                            log.debug("render: Hiding picker layer %s from picking itself" % layer)
                            continue
                        elif layer == selected:
                            delayed_pick_layer = (layer, vis)
                        else:
                            layer.render(renderer, w_r, p_r, s_r, vis, picker)
                else:  # not in pick-mode
                    if layer == selected:
                        control_points_layer = (layer, vis)
                    if not layer.draw_on_top_when_selected:
                        layer.render(renderer, w_r, p_r, s_r, vis, picker)
            if delayed_pick_layer is not None:
                layer, vis = delayed_pick_layer
                renderer = self.layer_renderers[layer]
                layer.render(renderer, w_r, p_r, s_r, vis, picker)
            if control_points_layer is not None:
                layer, vis = control_points_layer
                renderer = self.layer_renderers[layer]
                if layer.draw_on_top_when_selected:
                    layer.render(renderer, w_r, p_r, s_r, vis, picker)
                else:
                    layer.render(renderer, w_r, p_r, s_r, vis, picker, control_points_only=True)

        render_layers(layer_draw_order)

        self.render_overlay()

        if self.is_canvas_pickable():
            log.debug("render: rendering picker")
            self.begin_rendering_picker(s_r)
            render_layers(layer_draw_order, picker=self.picker)
            self.done_rendering_picker()
            if self.debug_show_picker_framebuffer:
                self.picker.render_picker_to_screen()

        elapsed = time.perf_counter() - t0
        self.post_render_update_ui_hook(elapsed, event)

        self.finalize_rendering_screen()
        log.debug(f"render: FINISHED RENDERING; total time = {elapsed}")

    def render_overlay(self):
        pass

    def post_render_update_ui_hook(self, elapsed, event):
        pass

    # functions related to world coordinates, projected coordinates, and screen coordinates

    def get_screen_size(self):
        raise NotImplementedError

    def set_screen_size(self, size):
        # provided for non-screen based canvas like PDF that copy their
        # settings from a screen canvas
        pass

    def get_screen_rect(self):
        size = self.get_screen_size()
        #
        return ((0, 0), (size[0], size[1]))

    def get_projected_point_from_screen_point(self, screen_point):
        c = rect.center(self.get_screen_rect())
        d = (screen_point[0] - c[0], screen_point[1] - c[1])
        d_p = (d[0] * self.projected_units_per_pixel, d[1] * self.projected_units_per_pixel)
        #
        return (self.projected_point_center[0] + d_p[0],
                self.projected_point_center[1] - d_p[1])

    def get_projected_rect_from_screen_rect(self, screen_rect):
        left_bottom = (screen_rect[0][0], screen_rect[1][1])
        right_top = (screen_rect[1][0], screen_rect[0][1])
        #
        return (self.get_projected_point_from_screen_point(left_bottom),
                self.get_projected_point_from_screen_point(right_top))

    def get_screen_point_from_projected_point(self, projected_point):
        d_p = (projected_point[0] - self.projected_point_center[0],
               projected_point[1] - self.projected_point_center[1])
        d = (d_p[0] / self.projected_units_per_pixel, d_p[1] / self.projected_units_per_pixel)
        r = self.get_screen_rect()
        c = rect.center(r)
        #
        return (c[0] + d[0], c[1] - d[1])

    def get_screen_rect_from_projected_rect(self, projected_rect):
        left_top = (projected_rect[0][0], projected_rect[1][1])
        right_bottom = (projected_rect[1][0], projected_rect[0][1])
        #
        return (self.get_screen_point_from_projected_point(left_top),
                self.get_screen_point_from_projected_point(right_bottom))

    def get_world_point_from_projected_point(self, projected_point):
        return self.projection(projected_point[0], projected_point[1], inverse=True)

    def get_world_rect_from_projected_rect(self, projected_rect):
        return (self.get_world_point_from_projected_point(projected_rect[0]),
                self.get_world_point_from_projected_point(projected_rect[1]))

    def get_projected_point_from_world_point(self, world_point):
        return self.projection(world_point[0], world_point[1])

    def get_projected_rect_from_world_rect(self, world_rect):
        return (self.get_projected_point_from_world_point(world_rect[0]),
                self.get_projected_point_from_world_point(world_rect[1]))

    def get_world_point_from_screen_point(self, screen_point):
        return self.get_world_point_from_projected_point(self.get_projected_point_from_screen_point(screen_point))

    def get_numpy_world_point_from_screen_point(self, screen_point):
        world_point = self.get_world_point_from_projected_point(self.get_projected_point_from_screen_point(screen_point))
        w = np.empty_like(world_point)
        w[:] = world_point
        return w

    def get_world_rect_from_screen_rect(self, screen_rect):
        return self.get_world_rect_from_projected_rect(self.get_projected_rect_from_screen_rect(screen_rect))

    def get_screen_point_from_world_point(self, world_point):
        screen_point = self.get_screen_point_from_projected_point(self.get_projected_point_from_world_point(world_point))
        # screen points are pixels, which should be int values
        return (round(screen_point[0]), round(screen_point[1]))

    def get_numpy_screen_point_from_world_point(self, world_point):
        screen_point = self.get_screen_point_from_projected_point(self.get_projected_point_from_world_point(world_point))
        s = np.empty_like(world_point)
        s[:] = screen_point
        return s

    def get_screen_rect_from_world_rect(self, world_rect):
        rect = self.get_screen_rect_from_projected_rect(self.get_projected_rect_from_world_rect(world_rect))
        return ((int(round(rect[0][0])), int(round(rect[0][1]))), (int(round(rect[1][0])), int(round(rect[1][1]))))

    def zoom(self, amount=1, ratio=2.0, focus_point_screen=None):
        zoom_level = self.get_zoom_level(self.projected_units_per_pixel)
        zoom_level = self.get_zoom_level(self.projected_units_per_pixel, round=.25)
        if amount > 0:
            zoom_level += ratio - 1.0
        else:
            zoom_level -= ratio - 1.0
        units_per_pixel = self.get_units_per_pixel_from_zoom(zoom_level)
        return self.constrain_zoom(units_per_pixel)

    def zoom_in(self):
        zoom_level = self.get_zoom_level(self.projected_units_per_pixel, round=.25)
        zoom_level += .5
        units_per_pixel = self.get_units_per_pixel_from_zoom(zoom_level)
        return self.constrain_zoom(units_per_pixel)

    def zoom_out(self):
        zoom_level = self.get_zoom_level(self.projected_units_per_pixel, round=.25)
        zoom_level -= .5
        units_per_pixel = self.get_units_per_pixel_from_zoom(zoom_level)
        return self.constrain_zoom(units_per_pixel)

    def zoom_to_fit(self):
        center, units_per_pixel = self.calc_zoom_to_fit()
        self.set_viewport(center, units_per_pixel)

    def calc_zoom_to_fit(self):
        layers = self.project.layer_manager.get_visible_layers(self.project.layer_visibility)
        return self.calc_zoom_to_layers(layers)

    def zoom_to_layers(self, layers):
        center, units_per_pixel = self.calc_zoom_to_layers(layers)
        self.set_viewport(center, units_per_pixel)

    def calc_zoom_to_layers(self, layers):
        w_r = self.project.layer_manager.accumulate_layer_bounds(layers)
        return self.calc_zoom_to_world_rect(w_r)

    def zoom_to_world_rect(self, w_r, border=True):
        center, units_per_pixel = self.calc_zoom_to_world_rect(w_r, border)
        self.set_viewport(center, units_per_pixel)

    def calc_zoom_to_world_rect(self, w_r, border=True):
        if (w_r == rect.NONE_RECT):
            return self.projected_point_center, self.projected_units_per_pixel

        p_r = self.get_projected_rect_from_world_rect(w_r)
        size = self.get_screen_size()
        if border:
            # so that when we zoom, the points don't hit the very edge of the window
            EDGE_PADDING = 20
        else:
            EDGE_PADDING = 0
        size.x -= EDGE_PADDING * 2
        size.y -= EDGE_PADDING * 2
        pixels_h = rect.width(p_r) / self.projected_units_per_pixel
        pixels_v = rect.height(p_r) / self.projected_units_per_pixel
        ratio_h = float(pixels_h) / float(size[0])
        ratio_v = float(pixels_v) / float(size[1])
        ratio = max(ratio_h, ratio_v)

        units_per_pixel = self.constrain_zoom(self.projected_units_per_pixel * ratio)
        center = rect.center(p_r)
        return center, units_per_pixel

    def get_zoom_rect(self):
        return self.get_world_rect_from_screen_rect(self.get_screen_rect())

    def zoom_to_include_world_rect(self, w_r):
        view_w_r = self.get_zoom_rect()
        if (not rect.contains_rect(view_w_r, w_r)):
            # first try just panning
            p_r = self.get_projected_rect_from_world_rect(w_r)
            self.projected_point_center = rect.center(p_r)
            view_w_r = self.get_world_rect_from_screen_rect(self.get_screen_rect())
            if (not rect.contains_rect(view_w_r, w_r)):
                # otherwise we have to zoom (i.e., zoom out because panning didn't work)
                self.zoom_to_world_rect(w_r)

    def constrain_zoom(self, units_per_pixel):
        # Using open street map style zoom factors to limit zoom
        max_zoom_in = self.get_units_per_pixel_from_zoom(22)
        max_zoom_out = self.get_units_per_pixel_from_zoom(1.0)
        units_per_pixel = max(units_per_pixel, max_zoom_in)
        units_per_pixel = min(units_per_pixel, max_zoom_out)
        return units_per_pixel

    def get_zoom_level(self, units_per_pixel, round=0.0):
        # Calculate tile size in standard wmts/google maps units.  At each zoom
        # level n, there are 2**n tiles across the 360 degrees of longitude,
        # where each tile is 256 pixels wide.
        delta_lon_per_tile = self.get_delta_lon_per_pixel(256, units_per_pixel)
        num_tiles = 360.0 / delta_lon_per_tile
        zoom_level = math.log(num_tiles, 2)
        if round > 0.0:
            zoom_level = int(zoom_level / round) * round
        log.debug("get_zoom_level: units_per_pixel %s, num_tiles %s, zoom level %s" % (units_per_pixel, num_tiles, zoom_level))
        return zoom_level

    def get_delta_lon_per_pixel(self, num_pixels, units_per_pixel=None):
        if units_per_pixel is None:
            units_per_pixel = self.projected_units_per_pixel
        return self.get_world_point_from_projected_point((num_pixels * units_per_pixel, 0.0))[0]

    def get_units_per_pixel_from_zoom(self, zoom_level):
        num_tiles = math.pow(2.0, zoom_level)
        delta_lon_per_tile = 360.0 / num_tiles
        units_per_pixel = self.get_projected_point_from_world_point((delta_lon_per_tile / 256.0, 0.0))[0]
        log.debug("get_units_per_pixel_from_zoom: units_per_pixel %s, num_tiles %s, zoom level %s" % (units_per_pixel, num_tiles, zoom_level))
        return units_per_pixel

    @property
    def world_center(self):
        return self.get_world_point_from_projected_point(self.projected_point_center)

    def set_viewport(self, center, units_per_pixel):
        self.projected_point_center = center
        self.projected_units_per_pixel = units_per_pixel
        self.zoom_level = self.get_zoom_level(units_per_pixel)

    def set_center(self, center):
        self.projected_point_center = center

    def set_units_per_pixel(self, units_per_pixel):
        self.projected_units_per_pixel = units_per_pixel
        self.zoom_level = self.get_zoom_level(units_per_pixel)

    def copy_viewport_from(self, other):
        self.projected_units_per_pixel = other.projected_units_per_pixel
        self.projected_point_center = tuple(other.projected_point_center)
        self.projection = other.projection
        self.set_screen_size(other.get_screen_size())

    def get_surrounding_screen_rects(self, r):
        # return four disjoint rects surround r on the screen
        sr = self.get_screen_rect()

        if (r[0][1] <= sr[0][1]):
            above = rect.EMPTY_RECT
        else:
            above = (sr[0], (sr[1][0], r[0][1]))

        if (r[1][1] >= sr[1][1]):
            below = rect.EMPTY_RECT
        else:
            below = ((sr[0][0], r[1][1]), sr[1])

        if (r[0][0] <= sr[0][0]):
            left = rect.EMPTY_RECT
        else:
            left = ((sr[0][0], r[0][1]), (r[0][0], r[1][1]))

        if (r[1][0] >= sr[1][0]):
            right = rect.EMPTY_RECT
        else:
            right = ((r[1][0], r[0][1]), (sr[1][0], r[1][1]))

        return [above, below, left, right]

    def get_visible_labels(self, values, projected_points, projected_rect):
        r1 = projected_points[:,0] >= projected_rect[0][0]
        r2 = projected_points[:,0] <= projected_rect[1][0]
        r3 = projected_points[:,1] >= projected_rect[0][1]
        r4 = projected_points[:,1] <= projected_rect[1][1]
        mask = np.logical_and(np.logical_and(r1, r2), np.logical_and(r3, r4))
        relevant_indexes = np.where(mask)[0]
        relevant_points = projected_points[relevant_indexes]

        relevant_values = values[relevant_indexes]
        labels = list(map(str, relevant_values))
        n = sum(map(len, labels))

        if (n == 0 or n > self.max_label_characters):
            return 0, 0, 0
        return n, labels, relevant_points

    def get_canvas_as_image(self):
        raise NotImplementedError
