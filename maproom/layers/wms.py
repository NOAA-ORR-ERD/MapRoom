import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str, Bool, Any

from ..library import rect, coordinates
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_html_color_string, alpha_from_int, ImageData

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)

class WMSLayer(ScreenLayer):
    """Web Map Service
    
    """
    name = Unicode("WMS")
    
    type = Str("wms")
    
    image_data = Any
    
    current_size = Any((50, 200))
    
    rebuild_needed = Bool(True)
    
    def get_image_array(self):
        from maproom.library.numpy_images import get_rect
        return get_rect(*self.current_size)

    def rebuild_renderer(self, renderer, in_place=False):
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None:
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[0], raw.shape[1])
            self.image_data.load_numpy_array(None, raw)
        renderer.set_image_screen(self.image_data)
        self.rebuild_needed = False

    def resize(self, renderer, world_rect, screen_rect):
        print "world_rect = %r" % (world_rect,)
        print "screen_rect = %r" % (screen_rect,)
        s = rect.size(screen_rect)
        if s != self.current_size:
            self.current_size = s
            self.rebuild_needed = True

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect):
        self.resize(renderer, world_rect, screen_rect)
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)

    # fixme == this should be able to get the various rects from the render_window object...
    def render_screen(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering wms!!! pick=%s" % (picker))
        render_window = renderer.canvas
        renderer.set_image_to_screen_rect(self.image_data, screen_rect)
        renderer.draw_image(layer_index_base, picker, 1.0)
