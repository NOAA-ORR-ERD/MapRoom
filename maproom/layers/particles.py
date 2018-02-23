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
import wx

from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..renderer import color_floats_to_int, linear_contour
from ..library import colormap

from folder import Folder
from base import ProjectedLayer
from point_base import PointBaseLayer
import state

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

    layer_info_panel = ["Start time", "End time", "Scalar value", "Outline color", "Status Code Color"]

    @property
    def scalar_var_names(self):
        children = self.get_particle_layers()
        names = set()
        for c in children:
            names.update(c.scalar_var_names)
        return names

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
    def current_scalar_var(self):
        children = self.get_particle_layers()
        if children:
            current = children[0].current_scalar_var
            for c in children:
                if c.current_scalar_var != current:
                    current = None
                    break
        else:
            current = None
        return current

    @property
    def status_code_colors(self):
        log.debug("Creating status code colors for %s" % self)
        status_code_names = self.status_code_names
        status_code_colors = ParticleLayer.create_status_code_color_map(status_code_names)
        return status_code_colors

    @property
    def status_code_count(self):
        return {name:1 for name in self.status_code_names}

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

    def set_scalar_var(self, var):
        children = self.get_particle_layers()
        for c in children:
            c.set_scalar_var(var)


class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = Unicode("Particle Layer")

    type = Str("particle")

    layer_info_panel = ["Scalar value", "Outline color", "Status Code Color"]

    pickable = True  # this is a layer that supports picking

    status_codes = Any  # numpy list matching array size of points

    status_code_names = Any

    status_code_colors = Any

    status_code_count = Any({})

    scalar_vars = Any({})

    current_scalar_var = Any(None)

    colormap_name = Str

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

    @property
    def scalar_var_names(self):
        names = set(self.scalar_vars.keys())
        return names

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

    def status_code_count_to_json(self):
        if self.status_code_count is not None:
            return self.status_code_count.items()

    def status_code_count_from_json(self, json_data):
        jd = json_data['status_code_count']
        if jd is not None:
            self.status_code_count = dict(jd)
        else:
            self.status_code_count = {}

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

    def scalar_vars_to_json(self):
        d = []  # transform since numpy values can't be directly serialized
        for name, values in self.scalar_vars.iteritems():
            v = [name, str(values.dtype)]
            v.extend(values.tolist())
            d.append(v)
        return d

    def scalar_vars_from_json(self, json_data):
        jd = json_data['scalar_vars']
        if jd is not None:
            self.scalar_vars = {}
            for item in jd:
                name = item[0]
                dtype = item[1]
                values = item[2:]
                self.scalar_vars[name] = np.asarray(values, dtype=dtype)
        else:
            self.scalar_vars = None

    def current_scalar_var_to_json(self):
        return self.current_scalar_var

    def current_scalar_var_from_json(self, json_data):
        self.current_scalar_var = json_data['current_scalar_var']

    def colormap_name_to_json(self):
        return self.colormap_name

    def colormap_name_from_json(self, json_data):
        self.colormap_name = json_data['colormap_name']

    def from_json_sanity_check_after_load(self, json_data):
        if not self.status_code_count:
            self.status_code_count = {}
            for k, v in self.status_code_names.iteritems():
                count = 0
                if v.endswith(")") and "(" in v:
                    _, text = v[:-1].rsplit("(", 1)
                    count = int(text)
                self.status_code_count[k] = count

    #

    def layer_selected_hook(self):
        self.select_all_points(state.FLAGGED)
        wx.CallLater(400, self.clear_flagged, True)

    def point_object_info(self, object_index):
        sc = self.status_codes[object_index]
        try:
            name = self.status_code_names[sc]
            name, _ = name.rsplit(" ", 1)
            sctext = ", %s," % name
        except:
            sctext = ""
        return "Point %s%s %s" % (object_index + 1, sctext, self.name)

    def point_object_long_info(self, object_index):
        info = []
        for n in self.scalar_var_names:
            info.append("%s: %s" % (n, self.scalar_vars[n][object_index]))
        return "\n".join(info)

    def show_unselected_layer_info_for(self, layer):
        return layer.type == self.type

    def set_data(self, f_points, status_codes, status_code_names, scalar_vars):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        self.status_codes = status_codes
        self.status_code_names = dict(status_code_names)
        self.status_code_count = {}
        self.status_code_colors = self.create_status_code_color_map(status_code_names)
        self.scalar_vars = scalar_vars
        self.set_colors_from_status_codes()

    def set_colors_from_status_codes(self):
        log.debug("setting status code colors to: %s" % self.status_code_colors)
        colors = np.zeros(np.alen(self.points), dtype=np.uint32)
        for code, color in self.status_code_colors.iteritems():
            index = np.where(self.status_codes == code)
            colors[index] = color
            num = len(index[0])
            self.status_code_names[code] += " (%d)" % num
            self.status_code_count[code] = num
        self.points.color = colors

    def set_scalar_var(self, var):
        if var is None:
            self.set_colors_from_status_codes()
        else:
            self.set_colors_from_scalar(var)
        self.current_scalar_var = var

    def set_colors_from_scalar(self, var):
        if var not in self.scalar_vars:
            log.error("%s not in scalar data for layer %s" % (var, self))
            print self.scalar_vars
            self.set_colors_from_status_codes()
            return
        values = self.scalar_vars[var]
        print(var, values)
        try:
            colors = colormap.get_opengl_colors(self.colormap_name, values)
        except ValueError:
            prefs = self.manager.project.task.preferences
            self.colormap_name = prefs.colormap_name
            colors = colormap.get_opengl_colors(self.colormap_name, values)
        self.points.color = colors
        self.manager.layer_contents_changed = self
        self.manager.refresh_needed = None

    def get_selected_particle_layers(self, project):
        return [self]

    def set_status_code_color(self, code, color):
        log.debug("Setting status code %s color: %x" % (code, color))
        index = np.where(self.status_codes == code)
        self.points.color[index] = color
        self.status_code_colors[code] = color

    def set_style(self, style):
        # Style changes in particle layer only affect the outline color; point
        # colors are controlled by the status colors
        ProjectedLayer.set_style(self, style)
