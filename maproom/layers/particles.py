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
    
    layer_info_panel = ["Particle Color", "Status Code Color"]

    status_codes = Any  # numpy list matching array size of points

    status_code_names = Any

    status_code_colors = Any

    # FIXME: Arbitrary colors for now till we decide on values
    status_code_to_color = np.array([color_floats_to_int(0, 0, 0, 1.0),
                                     color_floats_to_int(1.0, 0, 0, 1.0),
                                     color_floats_to_int(0, 1.0, 0, 1.0),
                                     color_floats_to_int(0, 0, 1.0, 1.0),
                                     color_floats_to_int(0, 1.0, 1.0, 1.0),
                                     ], dtype=np.uint32)

    status_code_color_map = {
        7: color_floats_to_int(1.0, 0, 0, 1.0),
        12: color_floats_to_int(1.0, 1.0, 1.0, 1.0),
        0: color_floats_to_int(0, 0, 0, 1.0),
        10: color_floats_to_int(1.0, 1.0, 0, 1.0),
        2: color_floats_to_int(0, 1.0, 0, 1.0),
        3: color_floats_to_int(0, 0, 1.0, 1.0),
    }
    
    ##### JSON Serialization
    
    def status_codes_to_json(self):
        if self.status_codes is not None:
            return self.status_codes.tolist()
    
    def status_codes_from_json(self, json_data):
        jd = json_data['status_codes']
        if jd is not None:
            self.status_codes = np.asarray(jd, np.uint32)
        else:
            p = json_data['points']
            if p is not None:
                self.status_codes = np.zeros(len(p), np.uint32)
            else:
                self.status_codes = np.zeros(0, np.uint32)
    
    def status_code_names_to_json(self):
        if self.status_code_names is not None:
            return [(str(k), v) for k,v in self.status_code_names.iteritems()]
    
    def status_code_names_from_json(self, json_data):
        jd = json_data['status_code_names']
        if jd is not None:
            self.status_code_names = dict((int(k), v) for k,v in jd)
        else:
            self.status_code_names = None
    
    # def set_layer_style_defaults(self):
    #     ## this should do something different for particles
    
    def set_data(self, f_points, status_codes, status_code_names):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        self.status_codes = status_codes
        self.status_code_names = status_code_names
        self.status_code_colors = {}
        for k, v in status_code_names.iteritems():
            self.status_code_colors[k] = self.status_code_color_map.get(k, color_floats_to_int(1.0, 0, 0, 1.0))
        s = np.clip(status_codes, 0, np.alen(self.status_code_to_color) - 1)
        colors = self.status_code_to_color[s]
        self.points.color = colors
        print "status_code_names", self.status_code_names
        print "status_code_colors", self.status_code_colors
        
    def get_selected_particle_layers(self, project):
        return [self]
    
    def get_particle_color(self, project):
        if np.alen(self.points) > 0:
            return self.points[0].color
        return color_floats_to_int(0, 0, 0, 1.0)
    
    def update_particle_color(self, project, int_color):
        self.points.color = int_color
        self.change_count += 1


    def set_status_code_color(self, code, color):
        print self.status_codes
        print code
        index = np.where(self.status_codes == code)
        print index
        self.points.color[index] = color
        self.status_code_colors[code] = color
