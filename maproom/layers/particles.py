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
from traits.api import Int, Float
from traits.api import Str
from traits.api import Unicode

from ..renderer import color_floats_to_int, linear_contour
from ..library import colormap, math_utils

from folder import Folder
from base import ProjectedLayer, ScreenLayer
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

    layer_info_panel = ["Start time", "End time", "Scalar value", "Colormap", "Discrete colormap", "Status Code Color", "Outline color", "Point size"]

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
    def current_min_max(self):
        children = self.get_particle_layers()
        if children:
            current = children[0].current_min_max
        else:
            current = None
        return current

    @property
    def current_scalar_var(self):
        children = self.get_particle_layers()
        if children:
            current = children[0].current_scalar_var
        else:
            current = None
        return current

    @property
    def colormap(self):  # Raises IndexError if no children
        return self.get_particle_layers()[0].colormap

    @property
    def status_code_colors(self):
        log.debug("Creating status code colors for %s" % self)
        status_code_names = self.status_code_names
        status_code_colors = ParticleLayer.create_status_code_color_map(status_code_names)
        return status_code_colors

    @property
    def status_code_count(self):
        return {name:1 for name in self.status_code_names}

    @property
    def point_size(self):
        children = self.get_particle_layers()
        sizes = set()
        for c in children:
            sizes.add(c.point_size)
        return max(sizes)

    @point_size.setter
    def point_size(self, size):
        children = self.get_particle_layers()
        for c in children:
            c.point_size = size

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

    def set_colormap(self, name):
        children = self.get_particle_layers()
        for c in children:
            c.set_colormap(name)

    def is_using_colormap(self, var=None):
        if var is None:
            var = self.current_scalar_var
        return var != "status codes" and var in self.scalar_var_names

    def is_using_discrete_colormap(self, var=None):
        return self.is_using_colormap() and self.colormap.is_discrete


class ParticleLegend(ScreenLayer):
    """Layer for vector annotation image

    """
    name = Unicode("Legend")

    type = Str("legend")

    x_percentage = Float(1.0)

    y_percentage = Float(0.0)

    legend_pixel_width = Int(20)

    legend_pixel_height = Int(300)

    tick_pixel_width = Int(4)

    tick_label_pixel_spacing = Int(4)

    layer_info_panel = ["X location", "Y location"]

    source_particle_folder = Any(-1)

    x_offset = 20

    y_offset = 40

    ##### traits defaults

    # these are mutually exclusive, used at different times.
    # source_particle_folder_default is only used after a project load and
    # dependent_of_default is used when serializing
    def _source_particle_folder_default(self):
        return self.manager.get_layer_by_invariant(self.dependent_of)

    def _dependent_of_default(self):
        return self.source_particle_folder.invariant

    ##### serialization

    def x_percentage_to_json(self):
        return self.x_percentage

    def x_percentage_from_json(self, json_data):
        self.x_percentage = json_data['x_percentage']

    def y_percentage_to_json(self):
        return self.y_percentage

    def y_percentage_from_json(self, json_data):
        self.y_percentage = json_data['y_percentage']

    def legend_pixel_width_to_json(self):
        return self.legend_pixel_width

    def legend_pixel_width_from_json(self, json_data):
        self.legend_pixel_width = json_data['legend_pixel_width']

    def legend_pixel_height_to_json(self):
        return self.legend_pixel_height

    def legend_pixel_height_from_json(self, json_data):
        self.legend_pixel_height = json_data['legend_pixel_height']

    @property
    def is_renderable(self):
        return True

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering legend!!! pick=%s" % (picker))

        parent = self.source_particle_folder

        if parent.current_scalar_var is not None:
            if parent.current_min_max == None:
                labels1 = []
            else:
                labels1 = math_utils.calc_labels(*parent.current_min_max)

            label_width = 0
            labels2 = []
            for perc, text in labels1:
                w, h = renderer.get_drawn_string_dimensions(text)
                labels2.append((perc, text, w, h))
                label_width = max(label_width, w)
            label_width += self.tick_label_pixel_spacing

            w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - self.legend_pixel_width - label_width
            h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset - self.legend_pixel_height

            x = s_r[0][0] + (w * self.x_percentage) + self.x_offset
            y = s_r[1][1] - (h * self.y_percentage) - self.y_offset

            c = parent.colormap
            r = ((x,y), (x+self.legend_pixel_width,y-self.legend_pixel_height))
            colors = c.calc_rgba_texture()
            renderer.draw_screen_textured_rect(r, colors, labels2, label_width, self.x_offset, self.y_offset, self.tick_pixel_width, self.tick_label_pixel_spacing, up_color=c.over_rgba, down_color=c.under_rgba)


class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = Unicode("Particle Layer")

    type = Str("particle")

    layer_info_panel = ["Scalar value", "Point size", "Outline color", "Status Code Color"]

    pickable = True  # this is a layer that supports picking

    status_codes = Any  # numpy list matching array size of points

    status_code_names = Any

    status_code_colors = Any

    status_code_count = Any({})

    scalar_vars = Any({})

    scalar_min_max = Any({})

    current_min_max = Any(None)

    current_scalar_var = Any(None)

    colormap = Any

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

    def _colormap_default(self):
        return colormap.get_colormap("Dark2")

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

    def scalar_min_max_to_json(self):
        return self.scalar_min_max

    def scalar_min_max_from_json(self, json_data):
        self.scalar_min_max = json_data['scalar_min_max']

    def current_scalar_var_to_json(self):
        return self.current_scalar_var

    def current_scalar_var_from_json(self, json_data):
        self.current_scalar_var = json_data['current_scalar_var']

    def current_min_max_to_json(self):
        return self.current_min_max

    def current_min_max_from_json(self, json_data):
        self.current_min_max = json_data['current_min_max']

    def colormap_to_json(self):
        return self.colormap.name

    def colormap_from_json(self, json_data):
        self.colormap = colormap.get_colormap(json_data['colormap_name'])

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
        self.set_colors_from_status_codes(True)

    def set_colors_from_status_codes(self, update_count=False):
        log.debug("setting status code colors to: %s" % self.status_code_colors)
        colors = np.zeros(np.alen(self.points), dtype=np.uint32)
        for code, color in self.status_code_colors.iteritems():
            index = np.where(self.status_codes == code)
            colors[index] = color
            if update_count:
                num = len(index[0])
                self.status_code_names[code] += " (%d)" % num
                self.status_code_count[code] = num
        self.points.color = colors

    def set_scalar_var(self, var):
        var = self.set_colors_from_scalar(var)
        self.current_scalar_var = var

    def is_using_colormap(self, var=None):
        if var is None:
            var = self.current_scalar_var
        return var != "status codes" and var in self.scalar_var_names

    def set_colors_from_scalar(self, var):
        if self.is_using_colormap(var):
            values = self.scalar_vars[var]
            lo, hi = self.scalar_min_max[var]
            self.current_min_max = lo, hi
            self.colormap.set_bounds(lo, hi)
            colors = self.colormap.get_opengl_colors(values)
            self.points.color = colors
        else:
            log.error("%s not in scalar data for layer %s" % (var, self))
            self.set_colors_from_status_codes()
            var = None
        self.manager.layer_contents_changed = self
        self.manager.refresh_needed = None
        return var

    def set_colormap(self, name):
        try:
            self.colormap = colormap.get_colormap(name)
        except ValueError:
            prefs = self.manager.project.task.preferences
            self.colormap = colormap.get_colormap(prefs.colormap_name)
        self.set_colors_from_scalar(self.current_scalar_var)

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
