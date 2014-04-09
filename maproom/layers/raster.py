import os
import os.path
import time
import sys
import numpy as np
import wx
from pytriangle import triangulate_simple

# Enthought library imports.
from traits.api import Unicode, Str, Any

from ..library import File_loader, rect
from ..library.scipy_ckdtree import cKDTree
from ..library.formats import verdat
from ..library.accumulator import flatten
from ..library.Projection import Projection
from ..library.Boundary import find_boundaries, generate_inside_hole_point, generate_outside_hole_point
from ..renderer import color_to_int, data_types

from base import Layer, ProjectedLayer
from constants import *

class RasterLayer(ProjectedLayer):
    """Layer for raster images
    
    """
    name = Unicode("Raster Layer")

    images = Any
    
    image_sizes = Any
    
    image_world_rects = Any

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_images = (self.images is None or len(self.images) == 0)

        return no_images
    
    def get_allowable_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["images"]
    
    def visibility_item_exists(self, label):
        if label == "images":
            return self.images is not None

    def read_from_file(self, file_path):
        self.file_path = file_path
        self.name = os.path.split(file_path)[1]
        (base, ext) = os.path.splitext(file_path)
        ext = ext.lower()

        (self.load_error_string,
         self.images,
         self.image_sizes,
         self.image_world_rects,
         projection) = File_loader.load_image_file(file_path)

        if (self.load_error_string == ""):
            self.type = ext

            # change the app projection to latlong if this image is latlong projection
            # and we don't currently have a mercator image loaded;
            # alternatively, if we are in latlong and we don't currently have
            # a latlong image loaded, and this image is mercator, change to mercator

            # TODO: handle other projections besides +proj=merc and +proj=longlat
            raster_layers = self.manager.count_raster_layers()
            vector_layers = self.manager.count_vector_layers()
            
            # FIXME: what was this pubsub message used for?
#            if raster_layers == 0 and vector_layers == 0:
#                pub.sendMessage(('layer', 'proejction', 'changed'), layer=self, projection=projection.srs)
            currently_merc = self.manager.project.control.projection.srs.find("+proj=merc") != -1
            currently_longlat = self.manager.project.control.projection.srs.find("+proj=longlat") != -1
            incoming_merc = projection.srs.find("+proj=merc") != -1
            incoming_longlat = projection.srs.find("+proj=longlat") != -1

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
                    print message
                    tlw = wx.GetApp().GetTopWindow()
                    dialog = wx.MessageDialog(
                        tlw,
                        message=message,
                        caption="Projection Conflict",
                        style=wx.OK | wx.CANCEL | wx.ICON_QUESTION,
                    )

                    if (dialog.ShowModal() != wx.ID_OK):
                        self.load_error_string = "Projection conflict"
                        #
                        return

                    # FIXME: what was this pubsub message used for?
#                    pub.sendMessage(('layer', 'proejction', 'changed'), layer=self, projection=srs)

        if (self.load_error_string == ""):
            self.update_bounds()
            print "FIXME: old call to pubsub!"
            #pub.sendMessage(('layer', 'loaded'), layer=self)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if (self.image_world_rects != None):
            world_rect_flat_list = flatten(self.image_world_rects)
            b = world_rect_flat_list[0]
            for r in world_rect_flat_list[1:]:
                b = rect.accumulate_rect(b, r)
            bounds = rect.accumulate_rect(bounds, b)

        return bounds
    
    def create_renderer(self, renderer):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        if self.images and not renderer.image_set_renderer:
            renderer.rebuild_image_set_renderer(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        print "Rendering raster!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode)
        if (not layer_visibility["layer"]):
            return

        if (renderer.image_set_renderer != None):
            renderer.image_set_renderer.render(-1, pick_mode)
