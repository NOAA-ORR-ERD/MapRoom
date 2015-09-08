import os
import time

import math
import numpy as np

from picker import NullPicker
import maproom.library.rect as rect
from maproom.library.projection import Projection, NullProjection
import maproom.preferences

import logging
log = logging.getLogger(__name__)


class BaseCanvas(object):
    """Abstract class defining the rendering interface

    """

    def __init__(self, layer_manager, project):
        self.layer_manager = layer_manager
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
        self.projected_point_center = (-13664393.732, 6048089.93218)
        # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)        
        ## does this get re-set anyway? pretty arbitrary.
        self.projected_units_per_pixel = 1000
        
        # mouse handler events
        self.mouse_handler = None  # defined in subclass
    
    def init_overlay(self):
        pass
    
    def new_picker(self):
        return NullPicker()
    
    def new_renderer(self, layer):
        return NullRenderer(self, layer)

    def change_view(self, layer_manager):
        self.layer_manager = layer_manager

    def get_renderer(self, layer):
        return self.layer_renderers[layer]

    def update_renderers(self):
        for layer in self.layer_manager.flatten():
            if not layer in self.layer_renderers:
                r = self.new_renderer(layer)
                layer.rebuild_renderer(r)
                self.layer_renderers[layer] = r
    
    def remove_renderer_for_layer(self, layer):
        if layer in self.layer_renderers:
            del self.layer_renderers[layer]
        pass

    def rebuild_renderers(self):
        for layer in self.layer_manager.flatten():
            self.remove_renderer_for_layer(layer)
        self.update_renderers()

    def rebuild_renderer_for_layer(self, layer, in_place=False):
        if layer in self.layer_renderers:
            r = self.layer_renderers[layer]
            layer.rebuild_renderer(r, in_place)
            log.debug("renderer rebuilt")
        else:
            log.warning("layer %s isn't in layer_renderers!" % layer)
            for layer in self.layer_renderers.keys():
                log.warning("  layer: %s" % layer)
    
    def begin_rendering_screen(self, projected_rect, screen_rect):
        self.screen_rect = screen_rect
        self.s_w = rect.width(screen_rect)
        self.s_h = rect.height(screen_rect)
        self.projected_rect = projected_rect
        p_w = rect.width(projected_rect)
        p_h = rect.height(projected_rect)

        if (self.s_w <= 0 or self.s_h <= 0 or p_w <= 0 or p_h <= 0):
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

    def get_selected_layer(self):
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
        # Get interactive console here:
#        import traceback
#        traceback.print_stack();
#        import code; code.interact( local = locals() )
        t0 = time.clock()
        self.update_renderers()

        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)

        if not self.begin_rendering_screen(p_r, s_r):
            return

        selected = self.get_selected_layer()
        layer_draw_order = list(enumerate(self.layer_manager.flatten()))
        layer_draw_order.reverse()

        # Update any linked control points by first looping through all layers
        # to update the world position, then updating the links.
        for i, layer in layer_draw_order:
            renderer = self.layer_renderers[layer]
            layer.pre_render(renderer, w_r, p_r, s_r)
        affected_layers = self.layer_manager.update_linked_control_points()
        for layer in affected_layers:
            renderer = self.layer_renderers[layer]
            layer.rebuild_renderer(renderer, True)

        null_picker = NullPicker()
        def render_layers(layer_order, picker=null_picker):
            self.layer_manager.pick_layer_index_map = {} # make sure it's cleared
            pick_layer_index = -1
            delayed_pick_layer = None
            control_points_layer = None
            for i, layer in layer_order:
                renderer = self.layer_renderers[layer]
                vis = self.project.layer_visibility[layer]
                if not vis["layer"]:
                    # short circuit displaying anything if entire layer is hidden
                    continue
                if picker.is_active:
                    if layer.pickable:
                        pick_layer_index += 1
                        self.layer_manager.pick_layer_index_map[pick_layer_index] = i
                        layer_index_base = picker.get_picker_index_base(pick_layer_index)
                        if layer == self.hide_picker_layer:
                            log.debug("Hiding picker layer %s from picking itself" % pick_layer_index)
                            continue
                        elif layer == selected:
                            delayed_pick_layer = (layer, layer_index_base, vis)
                        else:
                            layer.render(renderer, w_r, p_r, s_r, vis, layer_index_base, picker)
                else: # not in pick-mode
                    if layer == selected:
                        control_points_layer = (layer, vis)
                    layer.render(renderer, w_r, p_r, s_r, vis, -1, picker)
            if delayed_pick_layer is not None:
                layer, layer_index_base, vis = delayed_pick_layer
                renderer = self.layer_renderers[layer]
                layer.render(renderer, w_r, p_r, s_r, vis, layer_index_base, picker)
            if control_points_layer is not None:
                layer, vis = control_points_layer
                renderer = self.layer_renderers[layer]
                layer.render(renderer, w_r, p_r, s_r, vis, -1, picker, control_points_only=True)

        render_layers(layer_draw_order)

        self.render_overlay()

        if self.is_canvas_pickable():
            self.begin_rendering_picker(s_r)
            render_layers(layer_draw_order, picker=self.picker)
            self.done_rendering_picker()
            if self.debug_show_picker_framebuffer:
                self.picker.render_picker_to_screen()

        elapsed = time.clock() - t0
        self.post_render_update_ui_hook(elapsed, event)
        
        self.finalize_rendering_screen()
    
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

    def zoom(self, steps=1, ratio=2.0, focus_point_screen=None):
        if ratio > 0:
            units_per_pixel = self.projected_units_per_pixel / ratio
        else:
            units_per_pixel = self.projected_units_per_pixel * abs(ratio)
        return self.constrain_zoom(units_per_pixel)

    def zoom_in(self):
        return self.zoom(ratio=2.0)

    def zoom_out(self):
        return self.zoom(ratio=-2.0)

    def zoom_to_fit(self):
        self.projected_point_center, self.projected_units_per_pixel = self.calc_zoom_to_fit()

    def calc_zoom_to_fit(self):
        layers = self.layer_manager.get_visible_layers(self.project.layer_visibility)
        return self.calc_zoom_to_layers(layers)

    def zoom_to_layers(self, layers):
        self.projected_point_center, self.projected_units_per_pixel = self.calc_zoom_to_layers(layers)

    def calc_zoom_to_layers(self, layers):
        w_r = self.layer_manager.accumulate_layer_bounds(layers)
        if (w_r == rect.NONE_RECT):
            return self.projected_point_center, self.projected_units_per_pixel
        return self.calc_zoom_to_world_rect(w_r)

    def zoom_to_world_rect(self, w_r, border=True):
        self.projected_point_center, self.projected_units_per_pixel = self.calc_zoom_to_world_rect(w_r, border)

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

        center = rect.center(p_r)
        units_per_pixel = self.constrain_zoom(self.projected_units_per_pixel * ratio)
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
        ## fixme: this  should not be hard coded -- could scale to projection(90,90, inverse=True or ??)
        ## Also should be in some kind of app preferences location...
        min_val = .02
        max_val = 80000
        units_per_pixel = max(units_per_pixel, min_val)
        units_per_pixel = min(units_per_pixel, max_val)
        return units_per_pixel
    
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
        r1 = projected_points[:, 0] >= projected_rect[0][0]
        r2 = projected_points[:, 0] <= projected_rect[1][0]
        r3 = projected_points[:, 1] >= projected_rect[0][1]
        r4 = projected_points[:, 1] <= projected_rect[1][1]
        mask = np.logical_and(np.logical_and(r1, r2), np.logical_and(r3, r4))
        relevant_indexes = np.where(mask)[0]
        relevant_points = projected_points[relevant_indexes]

        relevant_values = values[relevant_indexes]
        labels = map(str, relevant_values)
        n = sum(map(len, labels))

        if (n == 0 or n > self.max_label_characters):
            return 0, 0, 0
        return n, labels, relevant_points

    def get_canvas_as_image(self):
        raise NotImplementedError
