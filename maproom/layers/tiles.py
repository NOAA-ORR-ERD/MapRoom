import numpy as np
import Queue

# Enthought library imports.
from traits.api import Unicode, Str, Int, Bool, Any, Set

from ..library import rect, coordinates
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_html_color_string, alpha_from_int, TileImageData

from base import ProjectedLayer

from maproom.library import numpy_images

import logging
log = logging.getLogger(__name__)

class TileLayer(ProjectedLayer):
    """Web Tile Service
    
    """
    name = Unicode("Tiles")
    
    type = Str("tiles")
    
    layer_info_panel = ["Transparency"]
    
    map_server_id = Int(0)
    
    image_data = Any(None)
    
    current_size = Any(None)  # holds tuple of screen size
    
    current_proj = Any(None)  # holds rect of projected coords
    
    current_world = Any(None)  # holds rect of world coords
    
    current_zoom = Int(-1)  # holds map zoom level
    
    rebuild_needed = Bool(True)
    
    threaded_request_results = Any
    
    download_status_text = Any(None)
    
    checkerboard_when_loading = False
    
    def _threaded_request_results_default(self):
        return Queue.Queue()
    
    def is_valid_threaded_result(self, map_server_id, tile_request):
        if map_server_id == self.map_server_id:
            self.threaded_request_results.put_nowait(tile_request)
            return True
        return False
    
    def rebuild_renderer(self, renderer, in_place=False):
        # Called only when tile server changed: throws away current tiles and
        # starts fresh
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
            self.rebuild_needed = False
        if self.image_data is None:
            projection = self.manager.project.layer_canvas.projection
            downloader = self.manager.project.task.get_threaded_tile_server_by_id(self.map_server_id)
            self.image_data = TileImageData(projection, downloader, renderer)
        if self.image_data is not None:
            self.image_data.add_tiles(self.threaded_request_results, renderer.image_textures)
            self.change_count += 1  # Force info panel update
            self.manager.project.layer_canvas.project.update_info_panels(self, True)

    def resize(self, renderer, world_rect, proj_rect, screen_rect):
        zoom_level = renderer.canvas.zoom_level
        print "RESIZE: zoom=", zoom_level
        self.current_proj = ((proj_rect[0][0], proj_rect[0][1]), (proj_rect[1][0], proj_rect[1][1]))
        self.current_world = ((world_rect[0][0], world_rect[0][1]), (world_rect[1][0], world_rect[1][1]))
        if zoom_level < 0:
            self.rebuild_renderer(renderer)
        elif zoom_level != self.image_data.zoom_level:
            renderer.canvas.set_minimum_delay_callback(self.zoom_changed, 1000)
            return
        # first time, load map immediately
        self.image_data.update_tiles(zoom_level, self.current_world, self.manager, (self, self.map_server_id))
    
    def zoom_changed(self, canvas):
        print "ZOOM CHANGED:", canvas.zoom_level
        self.image_data.update_tiles(canvas.zoom_level, self.current_world, self.manager, (self, self.map_server_id))
        self.change_count += 1  # Force info panel update
        canvas.project.update_info_panels(self, True)
    
    def change_server_id(self, id, canvas):
        if id != self.map_server_id:
            self.map_server_id = id
            self.map_layers = None
            self.rebuild_needed = True
            canvas.render()
            self.change_count += 1  # Force info panel update
            canvas.project.update_info_panels(self, True)

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
        if not layer_visibility["layer"]:
            return
        self.rebuild_renderer(renderer)
        self.resize(renderer, world_rect, projected_rect, screen_rect)

    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering tiles!!! pick=%s" % (picker))
        if self.image_data is not None:
            alpha = alpha_from_int(self.style.line_color)
            print "DRAW TILES HERE!!!"
            #renderer.draw_image(layer_index_base, picker, alpha)
