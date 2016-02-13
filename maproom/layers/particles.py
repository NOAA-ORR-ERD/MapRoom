"""
Layer type for particles

Particle folders now manage particle layers within the folders, but there's
nothing explicit about keeping particle layers in there.  They can be moved
around but only the ones within the folder will be managed by the folder.
Also there's nothing currently preventing other layer types being moved into
a particle folder.
"""
import sys

import numpy as np

from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..renderer import color_floats_to_int

from folder import Folder
from point_base import PointBaseLayer

# import logging
# log = logging.getLogger(__name__)
# progress_log = logging.getLogger("progress")


class ParticleFolder(Folder):
    """Layer for vector annotation image
    
    """
    name = Unicode("Particles")

    type = Str("particles")
    
    start_index = Int(0)
    
    end_index = Int(sys.maxint)
    
    layer_info_panel = ["Start time", "End time", "Particle Color"]
    
    def start_index_to_json(self):
        return self.start_index
    
    def start_index_from_json(self, json_data):
        self.start_index = json_data['start_index']
    
    def end_index_to_json(self):
        return self.end_index
    
    def end_index_from_json(self, json_data):
        self.end_index = json_data['end_index']
    
    def get_particle_layers(self):
        timesteps = []
        children = self.manager.get_layer_children(self)
        for layer in children:
            if layer.type == "particle":
                timesteps.append(layer)
        return timesteps
    
    def get_selected_particle_layers(self, project):
        steps = self.get_particle_layers()
        layers = []
        for layer in steps:
            if project.layer_visibility[layer]["layer"]:
                layers.append(layer)
        return layers
    
    def clamp_index(self, index):
        if index < 0:
            index = 0
        else:
            last_index = len(self.get_particle_layers()) - 1
            if index > last_index:
                index = last_index
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
    
    def get_particle_color(self, project):
        for layer in self.get_selected_particle_layers(project):
            color = layer.points[0].color
            return color
        return color_floats_to_int(0, 0, 0, 1.0)
    
    def update_particle_color(self, project, int_color):
        for layer in self.get_selected_particle_layers(project):
            layer.points.color = int_color
            layer.change_count += 1
        self.change_count += 1

class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = Unicode("Particle Layer")
    
    type = Str("particle")
    
    layer_info_panel = ["Particle Color"]

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
        
    def get_selected_particle_layers(self, project):
        return [self]
    
    def get_particle_color(self, project):
        if np.alen(self.points) > 0:
            return self.points[0].color
        return color_floats_to_int(0, 0, 0, 1.0)
    
    def update_particle_color(self, project, int_color):
        self.points.color = int_color
        self.change_count += 1
