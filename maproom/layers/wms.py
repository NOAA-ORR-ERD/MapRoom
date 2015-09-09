import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str, Int, Bool, Any

from ..library import rect, coordinates
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_html_color_string, alpha_from_int, ImageData

from base import ProjectedLayer

from maproom.library import numpy_images

import logging
log = logging.getLogger(__name__)

class WMSLayer(ProjectedLayer):
    """Web Map Service
    
    """
    name = Unicode("WMS")
    
    type = Str("wms")
    
    layer_info_panel = ["Map server", "Map layer"]
    
    map_server_id = Int(0)
    
    map_layer = Str("")
    
    image_data = Any
    
    current_size = Any(None)  # holds tuple of screen size
    
    current_world = Any(None)  # holds rect of world coords
    
    rebuild_needed = Bool(True)
    
    threaded_request_ready = Any(None)
    
    def get_image_array(self):
        if self.threaded_request_ready is None:
            downloader = self.manager.project.task.get_threaded_wms_by_id(self.map_server_id)
            layers = [self.map_layer]
            downloader.request_map(self.current_world, self.current_size, layers, self.manager, self)
            return self.current_world, numpy_images.get_checkerboard(*self.current_size)
        else:
            wms_result = self.threaded_request_ready
            self.threaded_request_ready = None
            return wms_result.world_rect, wms_result.get_image_array()

    def rebuild_renderer(self, renderer, in_place=False):
        projection = self.manager.project.layer_canvas.projection
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None and self.current_size is not None:
            world_rect, raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[0], raw.shape[1])
            self.image_data.load_numpy_array(None, raw)
            # OpenGL y coords are backwards, so simply flip world y coords and
            # OpenGL handles it correctly.
            flipped = ((world_rect[0][0], world_rect[1][1]),
                       (world_rect[1][0], world_rect[0][1]))
            self.image_data.set_rect(flipped, None)
            print "setting image data from wms connection:", world_rect
        if self.image_data is not None:
            renderer.set_image_projection(self.image_data, projection)
            self.rebuild_needed = False

    def resize(self, renderer, world_rect, screen_rect):
        print "world_rect = %r" % (world_rect,)
        print "screen_rect = %r" % (screen_rect,)
        s = rect.size(screen_rect)
        w = ((world_rect[0][0], world_rect[0][1]), (world_rect[1][0], world_rect[1][1]))
        if self.current_size is not None:
            if s != self.current_size or w != self.current_world:
                renderer.canvas.set_minimum_delay_callback(self.wms_rebuild, 1000)
        self.current_size = s
        self.current_world = w
        if self.image_data is None:
            # first time, set up immediate callback
            self.rebuild_needed = True
    
    def wms_rebuild(self, canvas):
        self.rebuild_needed = True
        canvas.render()

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect):
        self.resize(renderer, world_rect, screen_rect)
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)

    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering wms!!! pick=%s" % (picker))
        if self.image_data is not None:
            renderer.draw_image(layer_index_base, picker, 1.0)
