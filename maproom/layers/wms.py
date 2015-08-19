import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str, Int, Bool, Any

from ..library import rect, coordinates
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_html_color_string, alpha_from_int, ImageData

from base import ProjectedLayer

import logging
log = logging.getLogger(__name__)

class WMSLayer(ProjectedLayer):
    """Web Map Service
    
    """
    name = Unicode("WMS")
    
    type = Str("wms")
    
    image_data = Any
    
    current_size = Any((50, 200))
    
    current_world = Any(((0, 0), (10, 10)))
    
    timer = Int(999)
    
    rebuild_needed = Bool(True)
    
    def get_image_array(self):
        from maproom.library.numpy_images import get_rect
        return get_rect(*self.current_size)

    def rebuild_renderer(self, renderer, in_place=False):
        projection = self.manager.project.layer_canvas.projection
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[0], raw.shape[1])
            self.image_data.load_numpy_array(None, raw)
            self.image_data.set_rect(self.current_world, None)
        renderer.set_image_projection(self.image_data, projection)
        self.rebuild_needed = False

    def resize(self, renderer, world_rect, screen_rect):
        print "world_rect = %r" % (world_rect,)
        print "screen_rect = %r" % (screen_rect,)
        s = rect.size(screen_rect)
        if (s[0] != self.current_size[0] or s[1] != self.current_size[1]) or True:
            self.current_size = s
            self.timer += 1
            if self.timer > 100:
                self.rebuild_needed = True
                self.timer = 0
        self.current_world = ((world_rect[0][0], world_rect[0][1]), (world_rect[1][0], world_rect[1][1]))

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect):
        self.resize(renderer, world_rect, screen_rect)
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)

    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering wms!!! pick=%s" % (picker))
        renderer.draw_image(layer_index_base, picker, 1.0)
