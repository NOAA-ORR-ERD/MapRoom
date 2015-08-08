"""
Layer type for particles

not much here now -- but there will be...
"""
import numpy as np

from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..renderer import color_floats_to_int

from base import Folder
from point_base import PointBaseLayer

# import logging
# log = logging.getLogger(__name__)
# progress_log = logging.getLogger("progress")


class ParticleFolder(Folder):
    """Layer for vector annotation image
    
    """
    name = Unicode("Particles")

    type = Str("particles")
    
    start_index = Int(-1)
    
    end_index = Int(-1)
    
    layer_info_panel = ["Start Time", "End Time"]
    
    def get_particle_layers(self):
        timesteps = []
        children = self.manager.get_layer_children(self)
        for layer in children:
            if layer.type == "particle":
                timesteps.append(layer)
        return timesteps
    
    def clamp_index(self, index):
        if index < 0:
            index = 0
        else:
            steps = self.get_particle_layers()
            if index >= len(steps):
                index = len(steps) - 1
        return index
    
    def set_start_index(self, index):
        index = self.clamp_index(index)
        self.start_index = index
        if self.end_index < index:
            self.end_index = index
    
    def set_end_index(self, index):
        index = self.clamp_index(index)
        self.end_index = index
        if self.start_index > index:
            self.start_index = index
    
    def update_timestep_visibility(self,  project):
        # Folders will automatically set their children's visiblity state to
        # the parent state
        steps = self.get_particle_layers()
        for i, layer in enumerate(steps):
            checked = (self.start_index <= i <= self.end_index)
            project.layer_visibility[layer]["layer"] = checked
        project.layer_metadata_changed(self)
        project.refresh()

class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = Unicode("Particle Layer")
    
    type = Str("particle")
    
    # FIXME: Arbitrary colors for now till we decide on values
    status_code_to_color = np.array([color_floats_to_int(0, 0, 0, 1.0),
                                     color_floats_to_int(1.0, 0, 0, 1.0),
                                     color_floats_to_int(0, 1.0, 0, 1.0),
                                     color_floats_to_int(0, 0, 1.0, 1.0),
                                     color_floats_to_int(0, 1.0, 1.0, 1.0),
                                     ], dtype=np.uint32)
    
    # def set_layer_style_defaults(self):
    #     ## this should do something different for particles
    
    def set_data(self, f_points, status_codes):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        status_codes = np.clip(status_codes, 0, np.alen(self.status_code_to_color))
        colors = self.status_code_to_color[status_codes]
        self.points.color = colors
