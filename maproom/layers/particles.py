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

from sawx.utils.parseutil import NumpyFloatExpression, ParseException

from ..renderer import color_floats_to_int, linear_contour
from ..library import colormap, math_utils

from .folder import Folder
from .base import ProjectedLayer, StickyLayer
from .point_base import PointBaseLayer
from . import state

import logging
log = logging.getLogger(__name__)
# progress_log = logging.getLogger("progress")


valid_legend_types = ["Text", "Scale"]


class ParticleFolder(Folder):
    """Layer for vector annotation image

    """
    name = "Particles"

    type = "particles"

    style_as = "particle"

    # FIXME: Arbitrary colors for now till we decide on values
    status_code_color_map = {
        7: color_floats_to_int(1.0, 0, 1.0, 1.0),  # off maps
        12: color_floats_to_int(0.5, 0.5, 0.5, 1.0),  # to be removed
        0: color_floats_to_int(0, 1.0, 0, 1.0),  # not released
        10: color_floats_to_int(1.0, 1.0, 0, 1.0),  # evaporated
        2: color_floats_to_int(0, 0, 0, 1.0),  # in water
        3: color_floats_to_int(1.0, 0, 0, 1.0),  # on land
    }

    layer_info_panel = ["Start time", "End time", "Scalar value", "Scalar value expression", "Legend type", "Colormap", "Discrete colormap", "Status Code Color", "Outline color", "Point size"]

    selection_info_panel = ["Scalar value ranges"]

    def __init__(self, manager):
        super().__init__(manager)

        self.start_index = 0
        self.end_index = sys.maxsize
        self.scalar_subset_expression = ""
        self.current_min_max = None
        self.current_scalar_var = None
        self.scalar_min_max = {}
        self.status_code_names = None
        self.status_code_colors = None
        self.colormap = colormap.get_colormap("gnome")

    # class attributes

    @classmethod
    def create_status_code_color_map(cls, status_code_names):
        status_code_colors = {}
        for k, v in status_code_names.items():
            status_code_colors[k] = cls.status_code_color_map.get(k, color_floats_to_int(1.0, 0, 0, 1.0))
        return status_code_colors

    # attribute properties

    @property
    def scalar_var_names(self):
        children = self.get_particle_layers()
        names = set()
        for c in children:
            names.update(c.scalar_var_names)
        return names

    def init_status_codes(self, status_code_names):
        self.status_code_names = status_code_names
        self.status_code_colors = self.create_status_code_color_map(status_code_names)
        print(f"status code names: {self.status_code_names}")

    def scalar_value_ranges(self):
        children = self.get_particle_layers()
        r = {}
        for c in children:
            for name, vals in c.scalar_value_ranges().items():
                if name not in r:
                    r[name] = ([],[])
                r[name][0].append(vals[0])
                r[name][1].append(vals[1])
        ranges = {}
        for name, vals in r.items():
            ranges[name] = (min(vals[0]), max(vals[1]))
        return ranges

    def calc_scalar_value_summary(self):
        ranges = self.scalar_value_ranges()
        lines = []
        for name in sorted(ranges.keys()):
            r = ranges[name]
            lines.append("%s\n    %f-%f" % (name, r[0], r[1]))
        return "\n".join(lines)

    @property
    def status_code_count(self):
        return {name:1 for name in self.status_code_names}

    def status_code_label(self, code):
        return f"{self.status_code_names[code]}"

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

    @property
    def legend_type(self):
        children = self.get_particle_layers()
        sizes = set()
        for c in self.manager.get_layer_children(self):
            if hasattr(c, 'legend_type'):
                return c.legend_type
        return valid_legend_types[0]

    @legend_type.setter
    def legend_type(self, legend_type):
        for c in self.manager.get_layer_children(self):
            if hasattr(c, 'legend_type'):
                c.legend_type = legend_type

    def start_index_to_json(self):
        return self.start_index

    def start_index_from_json(self, json_data):
        self.start_index = json_data['start_index']

    def end_index_to_json(self):
        return self.end_index

    def end_index_from_json(self, json_data):
        self.end_index = json_data['end_index']

    def scalar_subset_expression_to_json(self):
        return self.scalar_subset_expression

    def scalar_subset_expression_from_json(self, json_data):
        self.scalar_subset_expression = json_data['scalar_subset_expression']

    def current_scalar_var_to_json(self):
        return self.current_scalar_var

    def current_scalar_var_from_json(self, json_data):
        self.current_scalar_var = json_data['current_scalar_var']

    def current_min_max_to_json(self):
        return self.current_min_max

    def current_min_max_from_json(self, json_data):
        self.current_min_max = json_data['current_min_max']

    def scalar_min_max_to_json(self):
        return self.scalar_min_max

    def scalar_min_max_from_json(self, json_data):
        self.scalar_min_max = json_data['scalar_min_max']

    def status_code_names_to_json(self):
        if self.status_code_names is not None:
            return list(self.status_code_names.items())

    def status_code_names_from_json(self, json_data):
        jd = json_data['status_code_names']
        if jd is not None:
            self.status_code_names = dict(jd)
        else:
            self.status_code_names = None

    def status_code_colors_to_json(self):
        if self.status_code_colors is not None:
            # force numbers to be python ints, not numpy. JSON can't serialize numpy
            return [(k, int(v)) for k, v in self.status_code_colors.items()]

    def status_code_colors_from_json(self, json_data):
        jd = json_data['status_code_colors']
        if jd is not None:
            self.status_code_colors = dict(jd)
        else:
            self.status_code_colors = None

    def colormap_to_json(self):
        return self.colormap.to_json()

    def colormap_from_json(self, json_data):
        try:
            self.colormap = colormap.DiscreteColormap.from_json(json_data['colormap'])
            colormap.register_colormap(self.colormap)
        except (KeyError, TypeError):
            self.colormap = colormap.get_colormap(json_data['colormap'])

    def restore_layer_relationships_after_load(self):
        children = self.manager.get_layer_children(self)
        log.debug(f"restoring layer relationships: {self}")
        for first in children:
            try:
                b = first.backward_compatibility_json
            except AttributeError:
                pass
            else:
                log.warning(f"checking backward compatibility json data from {first}: {b}")
                if self.status_code_colors is None:
                    log.warning(f"using status_code_colors from: {b['status_code_colors']}")
                    self.status_code_colors = b.get('status_code_colors', None)
                if self.status_code_names is None:
                    log.warning(f"using status_code_names from: {b['status_code_names']}")
                    self.status_code_names = b.get('status_code_names', None)
                if not self.scalar_min_max:
                    log.warning(f"using scalar_min_max from: {b['scalar_min_max']}")
                    self.scalar_min_max = b.get('scalar_min_max', None)
                if self.current_scalar_var is None:
                    log.warning(f"using current_scalar_var from: {b['current_scalar_var']}")
                    self.current_scalar_var = b.get('current_scalar_var', None)
                if self.current_min_max is None:
                    log.warning(f"using current_min_max from: {b['current_min_max']}")
                    self.current_min_max = b.get('current_min_max', None)
                if self.scalar_subset_expression is None:
                    log.warning(f"using scalar_subset_expression from: {b['scalar_subset_expression']}")
                    self.scalar_subset_expression = b.get('scalar_subset_expression', None)
                if self.colormap is None:
                    log.warning(f"using colormap from: {b['colormap']}")
                    self.set_colormap(b.get('colormap', None))
                break

        found_legend = False
        for layer in children:
            log.debug(f"restoring layer relationships for child: {layer}")
            layer.source_particle_folder = self
            try:
                layer.update_status_code_count()
            except AttributeError:
                pass
            if hasattr(layer, "legend_type"):
                found_legend = True
        if not found_legend:
            layer = ParticleLegend(self.manager, self)
            self.manager.insert_loaded_layer(layer, first_child_of=self)
        self.set_scalar_var(self.current_scalar_var)
        self.subset_using_logical_operation(self.scalar_subset_expression)

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
        if var is not None and self.is_using_colormap(var):
            lo, hi = self.scalar_min_max[var]
            self.current_min_max = lo, hi
            self.colormap.adjust_bounds(lo, hi)
            self.set_colors_from_colormap(var, self.colormap)
        else:
            log.error("%s not in scalar data for layer %s" % (var, self))
            self.set_colors_from_status_codes()
            var = None
        self.current_scalar_var = var

    def set_colors_from_colormap(self, var, colormap):
        children = self.get_particle_layers()
        for c in children:
            if var in c.scalar_vars:
                c.recalc_colors_from_colormap(var, colormap)
                self.manager.layer_contents_changed_event(c)
        self.manager.refresh_needed_event(None)

    def set_colors_from_status_codes(self):
        children = self.get_particle_layers()
        for c in children:
            c.recalc_colors_from_status_codes()
            self.manager.layer_contents_changed_event(c)
        self.manager.refresh_needed_event(None)

    def set_colormap(self, new_colormap):
        self.colormap = new_colormap
        colormap.register_colormap(new_colormap)
        self.set_colors_from_colormap(self.current_scalar_var, self.colormap)

    def is_using_colormap(self, var=None):
        if var is None:
            var = self.current_scalar_var
        return var != "status codes" and var in self.scalar_var_names

    def is_using_discrete_colormap(self, var=None):
        return self.is_using_colormap() and self.colormap.is_discrete

    def subset_using_logical_operation(self, operation):
        print(("folder: op=%s" % operation))

        self.scalar_subset_expression = operation
        children = self.get_particle_layers()
        for c in children:
            c.subset_using_logical_operation(operation)
        return children

    def num_below_above(self):
        total_lo = 0
        total_hi = 0
        for c in self.get_particle_layers():
            lo, hi = c.num_below_above()
            total_lo += lo
            total_hi += hi
        return total_lo, total_hi


class ParticleLegend(StickyLayer):
    """Layer for vector annotation image

    """
    name = "Legend"

    type = "legend"

    layer_info_panel = ["X location", "Y location", "Legend type", "Legend labels"]

    x_offset = 20
    y_offset = 40

    def __init__(self, manager, source_particle_folder=None):
        super().__init__(manager, x_percentage=1.0, y_percentage=0.0)

        self.legend_pixel_width = 20
        self.legend_pixel_height = 300
        self.tick_pixel_width = 4
        self.tick_label_pixel_spacing = 4
        self.source_particle_folder = source_particle_folder
        self.legend_type = "Text"
        self.legend_labels = "Light\nMedium\nHeavy"
        self.legend_bucket_width = 20
        self.legend_bucket_height = 14

    # these are mutually exclusive, used at different times.
    # source_particle_folder_default is only used after a project load and
    # dependent_of_default is used when serializing
    def _source_particle_folder_default(self):
        return self.manager.get_layer_by_invariant(self.dependent_of)

    def _dependent_of_default(self):
        return self.source_particle_folder.invariant

    ##### serialization

    def legend_pixel_width_to_json(self):
        return self.legend_pixel_width

    def legend_pixel_width_from_json(self, json_data):
        self.legend_pixel_width = json_data['legend_pixel_width']

    def legend_pixel_height_to_json(self):
        return self.legend_pixel_height

    def legend_pixel_height_from_json(self, json_data):
        self.legend_pixel_height = json_data['legend_pixel_height']

    def legend_type_to_json(self):
        return self.legend_type

    def legend_type_from_json(self, json_data):
        self.legend_type = json_data['legend_type']

    def legend_labels_to_json(self):
        return self.legend_labels

    def legend_labels_from_json(self, json_data):
        self.legend_labels = json_data['legend_labels']

    @property
    def is_renderable(self):
        return True

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering legend!!! pick=%s" % (picker))

        parent = self.source_particle_folder

        if parent.current_scalar_var is not None:
            if self.legend_type == "Text":
                self.render_text_legend(renderer, w_r, p_r, s_r)
            else:
                self.render_scale_legend(renderer, w_r, p_r, s_r)

    def calc_legend_xy(self, interior_width, interior_height, s_r):
        w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - interior_width
        h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset - interior_height

        x = s_r[0][0] + (w * self.x_percentage) + self.x_offset
        y = s_r[1][1] - (h * self.y_percentage) - self.y_offset
        return x, y

    def render_scale_legend(self, renderer, w_r, p_r, s_r):
        log.log(5, "Rendering scale legend!!!")

        parent = self.source_particle_folder
        c = parent.colormap
        if parent.current_min_max == None:
            labels1 = []
        else:
            labels1 = c.calc_labels(*parent.current_min_max)
        log.debug("rendering legend: colormap: %s, min_max=%s" % (c, parent.current_min_max))

        label_width = 0
        labels2 = []
        for perc, text in labels1:
            w, h = renderer.get_drawn_string_dimensions(text)
            labels2.append((perc, text, w, h))
            label_width = max(label_width, w)
        label_width += self.tick_label_pixel_spacing

        x, y = self.calc_legend_xy(self.legend_pixel_width + label_width, self.legend_pixel_height, s_r)

        r = ((x,y), (x+self.legend_pixel_width,y-self.legend_pixel_height))
        colors = c.calc_rgba_texture()
        try:
            lo, hi = parent.num_below_above()
            up_color = c.over_rgba if hi > 0 else None
            down_color = c.under_rgba if lo > 0 else None
        except AttributeError:
            # continuous colormap doesn't have values outside of bounds
            up_color = down_color = None
        renderer.draw_screen_textured_rect(r, colors, labels2, label_width, self.x_offset, self.y_offset, self.tick_pixel_width, self.tick_label_pixel_spacing, up_color=up_color, down_color=down_color)

    def render_text_legend(self, renderer, w_r, p_r, s_r):
        log.log(5, "Rendering text legend!!!")

        parent = self.source_particle_folder
        c = parent.colormap

        interior_width = 0
        interior_height = self.tick_label_pixel_spacing
        labels = self.legend_labels.splitlines()
        row_height = []
        for text in labels:
            w, h = renderer.get_drawn_string_dimensions(text)
            interior_width = max(interior_width, w)
            row_height.append(h)
            h = max(h, self.legend_bucket_height)
            interior_height += h + self.tick_label_pixel_spacing
        interior_width += 2 * self.tick_label_pixel_spacing + self.legend_bucket_width + self.tick_label_pixel_spacing

        x, y = self.calc_legend_xy(interior_width, interior_height, s_r)
        r = ((x,y), (x+interior_width,y-interior_height))
        colors = c.calc_rgba_texture()
        try:
            lo, hi = parent.num_below_above()
            up_color = c.over_rgba if hi > 0 else None
            down_color = c.under_rgba if lo > 0 else None
        except AttributeError:
            # continuous colormap doesn't have values outside of bounds
            up_color = down_color = None
        renderer.draw_screen_rect(r, 1.0, 1.0, 1.0, 1.0, flip=False)
        renderer.draw_screen_box(r, flip=False)
        x += self.tick_label_pixel_spacing
        y -= self.tick_label_pixel_spacing
        colors = c.bin_colors
        color_index = 1
        for text, h in zip(labels, row_height):
            h1 = self.legend_bucket_height
            if h1 < h:
                y1 = y - (h - h1) / 2
                y2 = y
                row_h = h
            else:
                y1 = y
                y2 = y - (h1 - h) / 2
                row_h = h1
            r = ((x,y1), (x+self.legend_bucket_width, y1-h1))
            try:
                c = colors[color_index]
            except IndexError:
                c = [.5, .5, .5, 1.0]
            renderer.draw_screen_rect(r, c[0], c[1], c[2], c[3], flip=False)

            renderer.draw_screen_string((x + self.legend_bucket_width + self.tick_label_pixel_spacing, y2), text, False)

            y -= row_h + self.tick_label_pixel_spacing
            color_index += 1



class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = "Particle"

    type = "particle"

    layer_info_panel = ["Point size", "Outline color", "Status Code Color"]

    selection_info_panel = ["Scalar value ranges"]

    pickable = True  # this is a layer that supports picking

    def __init__(self, manager, source_particle_folder=None):
        super().__init__(manager)

        self.source_particle_folder = source_particle_folder
        self.status_codes = None  # numpy list matching array size of points
        self.status_code_count = {}
        self.scalar_vars = {}

    def init_from_parent(self):
        self.recalc_colors_from_status_codes(True)
        print(f"  {self}: status code count: {self.status_code_count}")

    @property
    def status_code_names(self):
        return self.source_particle_folder.status_code_names

    @property
    def status_code_colors(self):
        return self.source_particle_folder.status_code_colors

    def status_code_label(self, code):
        return f"{self.status_code_names[code]} ({self.status_code_count[code]})"

    @property
    def scalar_var_names(self):
        names = set(self.scalar_vars.keys())
        return names

    def scalar_value_ranges(self):
        ranges = {}
        for name, vals in self.scalar_vars.items():
            ranges[name] = (min(vals), max(vals))
        return ranges

    def calc_scalar_value_summary(self):
        ranges = self.scalar_value_ranges()
        lines = []
        for name in sorted(ranges.keys()):
            r = ranges[name]
            lines.append("%s: %f-%f" % (name, r[0], r[1]))
        return "\n".join(lines)

    def is_using_colormap(self, var=None):
        print(self)
        return self.source_particle_folder.is_using_colormap(var)

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

    def status_code_count_to_json(self):
        if self.status_code_count is not None:
            return list(self.status_code_count.items())

    def status_code_count_from_json(self, json_data):
        jd = json_data['status_code_count']
        if jd is not None:
            self.status_code_count = dict(jd)
        else:
            self.status_code_count = {}

    def scalar_vars_to_json(self):
        d = []  # transform since numpy values can't be directly serialized
        for name, values in self.scalar_vars.items():
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

    def update_status_code_count(self):
        if not self.status_code_count:
            self.status_code_count = {}
            for k, v in self.status_code_names.items():
                count = 0
                if v.endswith(")") and "(" in v:
                    _, text = v[:-1].rsplit("(", 1)
                    count = int(text)
                self.status_code_count[k] = count

    # Backward compatibility loaders
    def from_json_sanity_check_after_load(self, json_data):
        b = {}
        jd = json_data.get('status_code_names', None)
        b['status_code_names'] = dict(jd) if jd is not None else None
        jd = json_data.get('status_code_colors', None)
        b['status_code_colors'] = dict(jd) if jd is not None else None
        b['scalar_min_max'] = json_data.get('scalar_min_max', None)
        b['current_scalar_var'] = json_data.get('current_scalar_var', None)
        b['current_min_max'] = json_data.get('current_min_max', None)
        b['scalar_subset_expression'] = json_data.get('scalar_subset_expression', None)
        jd = json_data.get('colormap', None)
        try:
            b['colormap'] = colormap.get_colormap(jd) if jd is not None else None
        except KeyError:
            b['colormap'] = colormap.get_colormap('tab20')
        print(f"SANITY!!!!!!! {b}")
        self.backward_compatibility_json = b
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

    def set_data(self, f_points, status_codes, scalar_vars):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        self.status_codes = status_codes
        self.status_code_count = {}
        self.scalar_vars = scalar_vars
        self.hidden_points = None

    def recalc_colors_from_status_codes(self, update_count=False):
        log.debug("setting status code colors to: %s" % self.status_code_colors)
        colors = np.zeros(np.alen(self.points), dtype=np.uint32)
        for code, color in self.status_code_colors.items():
            index = np.where(self.status_codes == code)
            colors[index] = color
            if update_count:
                num = len(index[0])
                self.status_code_count[code] = num
        self.points.color = colors

    def recalc_colors_from_colormap(self, var, colormap):
        values = self.scalar_vars[var]
        colors = colormap.get_opengl_colors(values)
        self.points.color = colors

    def num_below_above(self):
        var = self.current_scalar_var
        if var is not None and self.is_using_colormap(var):
            values = self.scalar_vars[var]
            num_lo = len(np.where(values < self.colormap.under_value)[0])
            num_hi = len(np.where(values >= self.colormap.over_value)[0])
            return num_lo, num_hi
        return 0, 0

    def get_particle_layers(self):
        return self.source_particle_folder.get_particle_layers()

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

    def subset_using_logical_operation(self, operation):
        log.debug("using operation %s" % operation)
        try:
            matches = self.get_matches(operation)
            # print matches
        except ValueError:
            self.hidden_points = None
        else:
            self.hidden_points = np.where(matches == False)[0]
            # print self.hidden_points
        return [self]

    def get_matches(self, search_text):
        expression = NumpyFloatExpression(self.scalar_vars)
        try:
            result = expression.eval(search_text)
            return result
        except ParseException as e:
            raise ValueError(e)
