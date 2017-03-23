
# Enthought library imports.
from traits.api import Any
from traits.api import Bool
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..library import rect
from ..renderer import ImageData
from ..renderer import alpha_from_int

from base import ProjectedLayer

from maproom.library import numpy_images

import logging
log = logging.getLogger(__name__)


class WMSLayer(ProjectedLayer):
    """Web Map Service
    
    """
    name = Unicode("WMS")

    type = Str("wms")

    layer_info_panel = ["Layer name", "Transparency", "Server status", "Server reload", "Map status"]

    selection_info_panel = ["Map server", "Map layer"]

    map_server_id = Int(0)

    map_layers = Any(None)  # holds a set of names representing shown overlay layers

    image_data = Any

    current_size = Any(None)  # holds tuple of screen size

    current_proj = Any(None)  # holds rect of projected coords

    current_world = Any(None)  # holds rect of world coords

    rebuild_needed = Bool(True)

    threaded_request_ready = Any(None)

    download_status_text = Any(None)

    checkerboard_when_loading = False

    def map_server_id_to_json(self):
        # get a representative URL to use as the reference in the project file
        # so we can restore the correct tile server
        wms_host = self.manager.project.task.get_wms_server_by_id(self.map_server_id)
        return wms_host.url

    def map_server_id_from_json(self, json_data):
        url = json_data['map_server_id']
        index = self.manager.project.task.get_wms_server_id_from_url(url)
        if index is not None:
            self.map_server_id = index

    def map_layers_to_json(self):
        # get a representative URL to use as the reference in the project file
        # so we can restore the correct tile server
        if self.map_layers is not None:
            return list(self.map_layers)

    def map_layers_from_json(self, json_data):
        layers = json_data['map_layers']
        if layers is not None:
            layers = set(layers)
        self.map_layers = layers

    def is_valid_threaded_result(self, map_server_id, wms_request):
        if map_server_id == self.map_server_id:
            self.rebuild_needed = True
            self.threaded_request_ready = wms_request
            return True
        return False

    def get_image_array(self):
        if self.threaded_request_ready is None:
            return self.current_world, numpy_images.get_checkerboard(*self.current_size), None
        else:
            wms_result = self.threaded_request_ready
            self.threaded_request_ready = None
            log.debug("threaded_request_ready = %s" % wms_result)
            return wms_result.world_rect, wms_result.get_image_array(), wms_result.error

    def rebuild_renderer(self, renderer, in_place=False):
        projection = self.manager.project.layer_canvas.projection
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None and self.current_size is not None:
            world_rect, raw, error = self.get_image_array()

            # Need to use an image even on an error condition, otherwise when
            # the screen is redrawn from some external event (window unhidden,
            # switching layer, etc.) the checkerboard image will be loaded
            # but because rebuild_needed wasn't set, no corresponding threaded
            # call will be issued to load a new image

            self.image_data = ImageData(raw.shape[1], raw.shape[0])
            self.image_data.load_numpy_array(None, raw)
            # OpenGL y coords are backwards, so simply flip world y coords and
            # OpenGL handles it correctly.
            flipped = ((world_rect[0][0], world_rect[1][1]),
                       (world_rect[1][0], world_rect[0][1]))
            self.image_data.set_rect(flipped, None)

            if error is None and self.image_data.is_blank:
                error = "Blank Image"
            self.download_status_text = ("error", error)

            self.change_count += 1  # Force info panel update
            self.manager.project.layer_canvas.project.update_info_panels(self, True)
        if self.image_data is not None:
            renderer.set_image_projection(self.image_data, projection)
            self.rebuild_needed = False

    def resize(self, renderer, world_rect, proj_rect, screen_rect):
        old_size = self.current_size
        old_world = self.current_world
        self.current_size = rect.size(screen_rect)
        self.current_proj = ((proj_rect[0][0], proj_rect[0][1]), (proj_rect[1][0], proj_rect[1][1]))
        self.current_world = ((world_rect[0][0], world_rect[0][1]), (world_rect[1][0], world_rect[1][1]))
        if old_size is not None:
            if old_size != self.current_size or old_world != self.current_world:
                renderer.canvas.set_minimum_delay_callback(self.wms_rebuild, 1000)
        else:
            # first time, load map immediately
            self.wms_rebuild(renderer.canvas)
            self.rebuild_needed = True

    def wms_rebuild(self, canvas):
        downloader = self.manager.project.task.get_threaded_wms_by_id(self.map_server_id)
        if downloader.is_valid():
            if self.map_layers is None:
                self.map_layers = set(downloader.server.get_default_layers())
            layers = list(self.map_layers)
            self.download_status_text = (None, "Downloading...")
            downloader.request_map(self.current_world, self.current_proj, self.current_size, layers, self.manager, (self, self.map_server_id))
            if self.checkerboard_when_loading:
                self.rebuild_needed = True
                canvas.render()
            self.name = downloader.host.name
            canvas.project.layer_metadata_changed(self)
        else:
            self.download_status_text = None
            # Try again, waiting till we get a successful contact
            if not downloader.server.has_error():
                log.debug("WMS not initialized yet, waiting...")
                canvas.set_minimum_delay_callback(self.wms_rebuild, 200)
            else:
                log.debug("WMS error, not attempting to contact again")
        self.change_count += 1  # Force info panel update
        canvas.project.update_info_panels(self, True)

    def change_server_id(self, id, canvas):
        if id != self.map_server_id:
            self.map_server_id = id
            self.map_layers = None
            self.wms_rebuild(canvas)
            self.change_count += 1  # Force info panel update
            canvas.project.update_info_panels(self, True)

    def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
        if not layer_visibility["layer"]:
            return
        self.resize(renderer, world_rect, projected_rect, screen_rect)
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)

    def render_projected(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering wms!!! pick=%s" % (picker))
        if self.image_data is not None:
            alpha = alpha_from_int(self.style.line_color)
            renderer.draw_image(self, picker, alpha)

    # Utility routines used by info_panels to abstract the server info

    def get_downloader(self, server_id):
        return self.manager.project.task.get_threaded_wms_by_id(server_id)

    def get_server_names(self):
        return self.manager.project.task.get_known_wms_names()
