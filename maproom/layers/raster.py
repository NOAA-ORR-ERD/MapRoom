from ..library import rect

from ..renderer import NullPicker
from ..renderer import alpha_from_int

from .base import ProjectedLayer
from . import state

import logging
log = logging.getLogger(__name__)


class RasterLayer(ProjectedLayer):
    """Layer for raster images
    
    """
    # class attributes

    restore_from_url = True

    background = True

    opaque = True

    name = "Raster"

    type = "image"

    layer_info_panel = ["Transparency", "Raster size", "Memory used"]

    selection_info_panel = []

    def __init__(self, manager):
        super().__init__(manager)
        self.image_data = None

    def test_contents_equal(self, other):
        """Test routine to compare layers"""
        if self.image_data is not None and other.image_data is not None:
            return self.image_data.x == other.image_data.x and self.image_data.y == other.image_data.y and ProjectedLayer.test_contents_equal(self, other)
        return ProjectedLayer.test_contents_equal(self, other)

    def get_info_panel_text(self, prop):
        if prop == "Raster size":
            return "%dx%d" % (self.image_data.x, self.image_data.y)
        elif prop == "Memory used":
            return "%sM" % (self.image_data.x * self.image_data.y * 4 / 1024 / 1024)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return self.image_data is None

    def extra_files_to_serialize(self):
        """Pathnames to any files that need to be included in the maproom
        project file that can't be recreated with JSON
        """
        return [self.file_path]

    def get_allowable_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["images"]

    def visibility_item_exists(self, label):
        if label == "images":
            return self.image_data is not None

    def check_projection(self):
        # change the app projection to latlong if this image is latlong projection
        # and we don't currently have a mercator image loaded;
        # alternatively, if we are in latlong and we don't currently have
        # a latlong image loaded, and this image is mercator, change to mercator

        # TODO: handle other projections besides +proj=merc and +proj=longlat
        raster_layers = self.manager.count_raster_layers()
        vector_layers = self.manager.count_vector_layers()

        if raster_layers == 0:
            self.manager.projection_changed_event(self)
            return
        e = self.manager.project
        currently_merc = e.layer_canvas.projection.srs.find("+proj=merc") != -1
        currently_longlat = e.layer_canvas.projection.srs.find("+proj=longlat") != -1
        incoming_merc = self.image_data.projection.srs.find("+proj=merc") != -1
        incoming_longlat = self.image_data.projection.srs.find("+proj=longlat") != -1

        disagreement = (currently_merc != incoming_merc) or (currently_longlat != incoming_longlat)
        if (disagreement):
            if (incoming_merc):
                type = "Mercator"
                srs = "+proj=merc +units=m +over"
            else:
                type = "Longitude/Latitude"
                srs = "+proj=longlat +over"
            message = None
            if (raster_layers > 0):
                message = "The file you are loading is in " + type + " projection, but one or more other raster files already loaded have a different projection. Do you want to load this file anyway, with distortion?"
            elif (vector_layers > 0):
                message = "The file you are loading is in " + type + " projection. Would you like to convert the loaded vector data to this projection?"

            if message is not None:
                if not e.frame.confirm(message):
                    self.load_error_string = "Projection conflict"
                    return

                self.manager.projection_changed_event(self)

    def compute_bounding_rect(self, mark_type=state.CLEAR):
        bounds = rect.NONE_RECT

        if self.image_data is not None:
            bounds = self.image_data.get_bounds()

        return bounds

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer
        
        """
        if not self.image_data:
            return

        projection = renderer.canvas.projection
        renderer.set_image_projection(self.image_data, projection)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering line layer!!! pick=%s" % (picker))
        if picker.is_active:
            return

        # the image
        null_picker = NullPicker()
        if (layer_visibility["images"]):
            alpha = alpha_from_int(self.style.line_color)
            renderer.draw_image(self, null_picker, alpha)
