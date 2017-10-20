"""
Layer type for particles

Particle folders now manage particle layers within the folders, but there's
nothing explicit about keeping particle layers in there.  They can be moved
around but only the ones within the folder will be managed by the folder.
Also there's nothing currently preventing other layer types being moved into
a particle folder.
"""
import sys
from dateutil import parser as date_parser
import calendar

import numpy as np

from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..renderer import color_floats_to_int

from folder import Folder
from point_base import PointBaseLayer

import logging
log = logging.getLogger(__name__)
# progress_log = logging.getLogger("progress")


class ParticleFolder(Folder):
    """Layer for vector annotation image

    """
    name = Unicode("Particles")

    type = Str("particles")

    style_as = "particle"

    start_index = Int(0)

    end_index = Int(sys.maxint)

    layer_info_panel = ["Start time", "End time", "Status Code Color", "Outline color", "Outline transparency"]

    @property
    def status_code_names(self):
        children = self.get_particle_layers()
        if children:
            names = children[0].status_code_names
        else:
            names = dict()
        summary_names = dict()
        for k, name in names.iteritems():
            if " (" in name and name .endswith(")"):
                name, _ = name.rsplit(" (", 1)
            summary_names[k] = name
        return summary_names

    @property
    def status_code_colors(self):
        status_code_names = self.status_code_names
        status_code_colors = ParticleLayer.create_status_code_color_map(status_code_names)
        return status_code_colors

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

    def update_timestep_visibility(self, project):
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

    layer_info_panel = ["Status Code Color", "Outline color", "Outline transparency"]

    pickable = True  # this is a layer that supports picking

    status_codes = Any  # numpy list matching array size of points

    status_code_names = Any

    status_code_colors = Any

    # FIXME: Arbitrary colors for now till we decide on values
    status_code_color_map = {
        7: color_floats_to_int(1.0, 0, 1.0, 1.0),  # off maps
        12: color_floats_to_int(0.5, 0.5, 0.5, 1.0),  # to be removed
        0: color_floats_to_int(0, 1.0, 0, 1.0),  # not released
        10: color_floats_to_int(1.0, 1.0, 0, 1.0),  # evaporated
        2: color_floats_to_int(0, 0, 0, 1.0),  # in water
        3: color_floats_to_int(1.0, 0, 0, 1.0),  # on land
    }

    @classmethod
    def create_status_code_color_map(cls, status_code_names):
        status_code_colors = {}
        for k, v in status_code_names.iteritems():
            status_code_colors[k] = cls.status_code_color_map.get(k, color_floats_to_int(1.0, 0, 0, 1.0))
        return status_code_colors

    # JSON Serialization

    def start_time_from_json_guess(self, json_data):
        # start time is missing because this is an old version of the project
        # file, but if it's in the standard format, we can guess from the title
        name = json_data['name']
        try:
            dt = date_parser.parse(name)
            log.info("Guessed date %s from %s for %s" % (str(dt), name, self))
        except ValueError:
            log.info("Can't parse %s to get default date for %s" % (name, self))
            raise KeyError
        t = calendar.timegm(dt.timetuple())
        return t

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
            return self.status_code_names.items()

    def status_code_names_from_json(self, json_data):
        jd = json_data['status_code_names']
        if jd is not None:
            self.status_code_names = dict(jd)
        else:
            self.status_code_names = None

    def status_code_colors_to_json(self):
        if self.status_code_colors is not None:
            # force numbers to be python ints, not numpy. JSON can't serialize numpy
            return [(k, int(v)) for k, v in self.status_code_colors.iteritems()]

    def status_code_colors_from_json(self, json_data):
        jd = json_data['status_code_colors']
        if jd is not None:
            self.status_code_colors = dict(jd)
        else:
            self.status_code_colors = None

    def point_object_info(self, object_index):
        sc = self.status_codes[object_index]
        try:
            name = self.status_code_names[sc]
            name, _ = name.rsplit(" ", 1)
            sctext = ", %s," % name
        except:
            sctext = ""
        return "Point %s%s %s" % (object_index + 1, sctext, self.name)

    def show_unselected_layer_info_for(self, layer):
        return layer.type == self.type

    def set_data(self, f_points, status_codes, status_code_names):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        self.status_codes = status_codes
        self.status_code_names = dict(status_code_names)
        self.status_code_colors = self.create_status_code_color_map(status_code_names)
        colors = np.zeros(np.alen(f_points), dtype=np.uint32)
        for code, color in self.status_code_colors.iteritems():
            index = np.where(self.status_codes == code)
            colors[index] = color
            num = len(index[0])
            self.status_code_names[code] += " (%d)" % num
        self.points.color = colors

    def get_selected_particle_layers(self, project):
        return [self]

    def set_status_code_color(self, code, color):
        index = np.where(self.status_codes == code)
        self.points.color[index] = color
        self.status_code_colors[code] = color
