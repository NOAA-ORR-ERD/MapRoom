import os
import time

import math
import numpy as np

from picker import NullPicker
import maproom.library.rect as rect

import logging
log = logging.getLogger(__name__)


class BaseCanvas(object):
    """Abstract class defining the rendering interface

    """

    def __init__(self, layer_manager, project):
        self.layer_manager = layer_manager
        self.project = project
        
        self.layer_renderers = {}
        
        self.overlay = self.get_overlay_renderer()
        self.picker = self.get_picker()
        self.hide_picker_layer = None

        self.screen_rect = rect.EMPTY_RECT
        
        self.debug_show_bounding_boxes = False
        self.debug_show_picker_framebuffer = False

        # two variables keep track of what's visible on the screen:
        # (1) the projected point at the center of the screen
        self.projected_point_center = (0, 0)
        # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)        
        ## does this get re-set anyway? pretty arbitrary.
        self.projected_units_per_pixel = 10000
        
        # mouse handler events
        self.mouse_handler = None  # defined in subclass
    
    def get_picker(self):
        return NullPicker()
    
    def get_overlay_renderer(self):
        return NullRenderer(self, None)
    
    def get_renderer(self, layer):
        return NullRenderer(self, layer)

    def change_view(self, layer_manager):
        self.layer_manager = layer_manager

    def update_renderers(self):
        for layer in self.layer_manager.flatten():
            if not layer in self.layer_renderers:
                r = layer.create_renderer(self)
                self.layer_renderers[layer] = r
        pass
    
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
            layer.rebuild_renderer(in_place)
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

    def set_screen_rendering_attributes(self):
        pass

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
            layer.pre_render()
        affected_layers = self.layer_manager.update_linked_control_points()
        for layer in affected_layers:
            layer.rebuild_renderer(True)

        null_picker = NullPicker()
        def render_layers(layer_order, picker=null_picker):
            self.layer_manager.pick_layer_index_map = {} # make sure it's cleared
            pick_layer_index = -1
            delayed_pick_layer = None
            control_points_layer = None
            for i, layer in layer_order:
                vis = self.project.layer_visibility[layer]
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
                            layer.render(self, w_r, p_r, s_r, vis, layer_index_base, picker)
                else: # not in pick-mode
                    if layer == selected:
                        control_points_layer = (layer, vis)
                    layer.render(self, w_r, p_r, s_r, vis, -1, picker)
            if delayed_pick_layer is not None:
                layer, layer_index_base, vis = delayed_pick_layer
                layer.render(self, w_r, p_r, s_r, vis, layer_index_base, picker)
            if control_points_layer is not None:
                layer, vis = control_points_layer
                layer.render(self, w_r, p_r, s_r, vis, -1, picker, control_points_only=True)

        render_layers(layer_draw_order)

        self.overlay.prepare_to_render_screen_objects()
        if self.debug_show_bounding_boxes:
            self.draw_bounding_boxes()
        
        self.mouse_handler.render_overlay(self.overlay)

        self.begin_rendering_picker(s_r)
        render_layers(layer_draw_order, picker=self.picker)
        self.done_rendering_picker()
        if self.debug_show_picker_framebuffer:
            self.picker.render_picker_to_screen()

        elapsed = time.clock() - t0
        self.post_render_update_ui_hook(elapsed, event)

    def post_render_update_ui_hook(self, elapsed, event):
        pass

    # functions related to world coordinates, projected coordinates, and screen coordinates

    def get_screen_size(self):
        raise NotImplementedError

    def get_canvas_as_image(self):
        raise NotImplementedError
