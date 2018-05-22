import queue

# Enthought library imports.
from traits.api import Any
from traits.api import Bool
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..renderer import TileImageData
from ..renderer import alpha_from_int

from .base import ProjectedLayer


import logging
log = logging.getLogger(__name__)


class TileLayer(ProjectedLayer):
    """Web Tile Service
    
    """
    name = "Tiles"

    type = "tiles"

    layer_info_panel = ["Layer Name", "Transparency", "Server status", "Server reload"]

    selection_info_panel = ["Tile server"]

    map_server_id = Int

    image_data = Any(None)

    current_size = Any(None)  # holds tuple of screen size

    current_proj = Any(None)  # holds rect of projected coords

    current_world = Any(None)  # holds rect of world coords

    current_zoom = Int(-1)  # holds map zoom level

    rebuild_needed = Bool(True)

    threaded_request_results = Any

    download_status_text = Any(None)

    checkerboard_when_loading = False

    # class attributes

    bounded = False

    background = True

    opaque = True

    ##### Traits

    def _map_server_id_default(self):
        return self.manager.project.task.get_default_tile_server_id()

    ##### Serialization

    def map_server_id_to_json(self):
        # get a representative URL to use as the reference in the project file
        # so we can restore the correct tile server
        tile_host = self.manager.project.task.get_tile_server_by_id(self.map_server_id)
        url = tile_host.get_next_url()
        return url

    def map_server_id_from_json(self, json_data):
        url = json_data['map_server_id']
        index = self.manager.project.task.get_tile_server_id_from_url(url)
        if index is not None:
            self.map_server_id = index

    def _threaded_request_results_default(self):
        return queue.Queue()

    def is_valid_threaded_result(self, map_server_id, tile_request):
        if map_server_id == self.map_server_id:
            self.threaded_request_results.put_nowait(tile_request)
            return True
        return False

    def rebuild_renderer(self, renderer, in_place=False):
        # Called only when tile server changed: throws away current tiles and
        # starts fresh
        if self.rebuild_needed:
            renderer.release_tiles()
            self.image_data = None
            self.rebuild_needed = False
        if self.image_data is None:
            projection = self.manager.project.layer_canvas.projection
            downloader = self.get_downloader(self.map_server_id)
            self.image_data = TileImageData(projection, downloader, renderer)
            self.name = downloader.host.name
            self.manager.project.layer_metadata_changed(self)
        if self.image_data is not None:
            renderer.set_tiles(self.image_data)
            self.image_data.add_tiles(self.threaded_request_results, renderer.image_tiles)
            renderer.image_tiles.reorder_tiles(self.image_data)
            self.change_count += 1  # Force info panel update
            self.manager.project.layer_canvas.project.update_info_panels(self, True)

    def resize(self, renderer, world_rect, proj_rect, screen_rect):
        zoom_level = renderer.canvas.zoom_level
        log.debug("RESIZE: zoom=%d image data zoom=%d", (zoom_level, self.image_data.zoom_level))
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
        log.debug("ZOOM CHANGED: %d" % canvas.zoom_level)
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

    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering tiles!!! pick=%s" % (picker))
        if self.image_data is not None:
            alpha = alpha_from_int(self.style.line_color)
            log.debug("calling renderer.draw_tiles")
            renderer.draw_tiles(self, picker, alpha)

    # Utility routines used by info_panels to abstract the server info

    def get_downloader(self, server_id):
        return self.manager.project.task.get_tile_downloader_by_id(server_id)

    def get_server_names(self):
        return self.manager.project.task.get_known_tile_server_names()
