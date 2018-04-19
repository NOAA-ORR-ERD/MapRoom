import numpy as np

from .. import math_utils

from . import colors
from . import cm

import logging
log = logging.getLogger(__name__)


class ColormapMapping(object):
    def __init__(self, colormap_generator, lo=None, hi=None, extra_padding=0.1):
        if lo is None:
            lo = float(min(values))
        if hi is None:
            hi = float(max(values))
        delta = abs(hi - lo)
        self.lo -= extra_padding * delta
        self.hi += extra_padding * delta
        self.mapping = colormap_generator.create_colormap(self.lo, self.hi)


class ListedBoundedColormap(colors.ListedColormap):
    def __init__(self, bin_colors, name):
        colors.ListedColormap.__init__(self, bin_colors[1:-1], name)
        self.set_under(bin_colors[0])
        self.set_over(bin_colors[-1])


class ScaledColormap(object):
    is_discrete = False

    def __init__(self, name, cmap, lo=None, hi=None, extra_padding=0.0, values=None, autoscale=True):
        self.autoscale = autoscale
        self.name = name
        self.cmap = self.preprocess_colormap(cmap)
        self.mapping = None
        self.set_bounds(lo, hi, extra_padding, values)

    def __repr__(self):
        return "ScaledColormap %s, %s-%s" % (self.name, self.lo_val, self.hi_val)

    @property
    def under_rgba(self):
        return None

    @property
    def over_rgba(self):
        return None

    def preprocess_colormap(self, cmap):
        return cmap

    def set_bounds(self, lo=None, hi=None, extra_padding=0.0, values=None):
        self.extra_padding = extra_padding
        if values is not None:
            lo = min(values)
            hi = max(values)
            bounds = np.asarray(values, dtype=np.float64)
        if lo is None:
            lo = 0.0
        if hi is None:
            hi = 1.0
        if lo == hi:
            hi += 1.0
        self.lo_val = lo
        self.hi_val = hi
        delta = abs(hi - lo)
        self.lo_padded = lo - extra_padding * delta
        self.hi_padded = hi + extra_padding * delta
        if values is None:
            bounds = np.linspace(self.lo_padded, self.hi_padded, self.cmap.N + 1)
        self.mapping = self.create_colormap(bounds)

    def adjust_bounds(self, lo=None, hi=None, extra_padding=0.0, values=None):
        # always scale values if not a discrete colormap
        self.set_bounds(lo, hi, extra_padding, values)

    def set_values(self, values):
        self.set_bounds(values=values)

    def calc_labels(self, lo, hi):
        return math_utils.calc_labels(lo, hi)

    def create_colormap(self, bounds):
        norm  = colors.Normalize(vmin=self.lo_padded, vmax=self.hi_padded)
        mapping = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return mapping

    def get_rgb_colors(self, values):
        values = self.mapping.to_rgba(values, bytes=True)
        return values[:,0:3]

    def get_opengl_colors(self, values):
        byte_values = self.mapping.to_rgba(values, bytes=True)
        return np.array(byte_values).view(np.uint32)[:,0]

    def calc_rgba_texture(self, length=256, width=1, alpha=0.5):
        line = np.linspace(self.lo_padded, self.hi_padded, length)
        colors = self.get_rgb_colors(line)
        if alpha is not None:
            array = np.empty((width, length, 4), dtype='uint8')
            array[:,:,3] = alpha
        else:
            array = np.empty((width, length, 3), dtype='uint8')
        array[:,:,0:3] = colors
        return array

    def calc_rgb_texture(self, length=256, width=1):
        return self.calc_rgba_texture(length, width, None)


class DiscreteColormap(ScaledColormap):
    # Always have an under and over!
    is_discrete = True

    def __repr__(self):
        return "DiscreteColormap %s, %s-%s, bins=%s" % (self.name, self.lo_val, self.hi_val, self.bin_borders)

    def to_json(self):
        return {
            'bin_colors': [colors.to_hex(c, True) for c in self.bin_colors],
            'bin_borders': [float(v) for v in self.bin_borders],  # get rid of numpy objects
            'extra_padding': float(self.extra_padding),
            'autoscale': self.autoscale,
            'name': self.name,
        }

    @classmethod
    def from_json(self, e):
        bin_colors = e['bin_colors']
        c = ListedBoundedColormap(bin_colors, e['name'])
        return DiscreteColormap(e['name'], c, extra_padding=e['extra_padding'], values=e['bin_borders'], autoscale=e['autoscale'])

    def adjust_bounds(self, lo=None, hi=None, extra_padding=0.0):
        if self.autoscale:
            values = self.bin_borders
            v_lo = min(values)
            delta = max(values) - v_lo
            if delta == 0.0:
                delta = 1.0
            perc = (values - v_lo) / delta
            scaled_delta = hi - lo
            if scaled_delta == 0.0:
                scaled_delta = 1.0
            scaled = (perc * scaled_delta) + lo
            self.set_bounds(None, None, extra_padding, scaled)
        else:
            log.debug("colormap: %s already scaled, not scaling automatically")

    def calc_labels(self, lo, hi):
        # Return percent, text pairs for each bin boundary. Lo and hi values
        # are ignored in discrete colormaps -- they always use the colormap
        # bins to determine lo and hi values.
        bins = self.bin_borders
        log.debug("calc_labels: bins=%s" % str(bins))
        labels = []
        lo, hi, rounded_labels = math_utils.round_minimum_unique_digits(bins)
        for val, rounded in zip(bins, rounded_labels):
            perc = (val - lo) / (hi - lo)
            labels.append((perc, rounded))
        log.debug("calc_labels: generated labels: %s" % str(labels))
        return labels

    @property
    def under_rgba(self):
        return colors.colorConverter.to_rgba(self.cmap._rgba_under)

    @property
    def over_rgba(self):
        return colors.colorConverter.to_rgba(self.cmap._rgba_over)

    @property
    def under_value(self):
        return self.mapping.norm.boundaries[0]

    @property
    def over_value(self):
        return  self.mapping.norm.boundaries[-1]

    def preprocess_colormap(self, cmap):
        self.source_cmap = cmap
        return self.copy_source_colormap()

    def copy_source_colormap(self):
        cmap = self.source_cmap
        c = list(cmap.colors)
        if cmap._rgba_under is not None:
            c[0:0] = [cmap._rgba_under]
        if cmap._rgba_over is not None:
            c.append(cmap._rgba_over)
        return ListedBoundedColormap(c, cmap.name)

    def copy(self):
        return DiscreteColormap(self.name, self.source_cmap, self.lo_val, self.hi_val, self.extra_padding, self.bin_borders, self.autoscale)

    def create_colormap(self, bounds):
        log.debug("create_colormap discrete = %s values, bounds = %s" % (self.name, str(bounds)))
        norm = colors.BoundaryNorm(bounds, self.cmap.N)
        mapping = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return mapping

    @property
    def bin_borders(self):
        if self.mapping is not None:
            bins = list(self.mapping.norm.boundaries)
            log.debug("using bin_borders: %s" % str(bins))
        else:
            bins = []
        return bins

    @property
    def bin_colors(self):
        c = list(self.cmap.colors)
        c[0:0] = [self.cmap._rgba_under]
        c.append(self.cmap._rgba_over)
        return colors.colorConverter.to_rgba_array(c)

    def set_bins(self, borders, colors):
        pass

builtin_continuous_colormaps = {name:ScaledColormap(name, cm.get_cmap(name)) for name in cm.cmap_continuous}

builtin_discrete_colormaps = {name:DiscreteColormap(name, cm.get_cmap(name)) for name in cm.cmap_discrete}

sample_discrete = colors.ListedColormap(['#ff0000', '#004400', '#007700', '#00aa00', '#0000ff'])
builtin_discrete_colormaps['sample_lat'] = DiscreteColormap('sample_lat', sample_discrete)
user_defined_discrete_colormaps = {}

def get_colormap(name, discrete_only=False):
    if not discrete_only and name in builtin_continuous_colormaps:
        log.debug("found colormap %s in builtin_continuous_colormaps" % name)
        return builtin_continuous_colormaps[name]
    elif name in builtin_discrete_colormaps:
        log.debug("found colormap %s in builtin_discrete_colormaps" % name)
        return builtin_discrete_colormaps[name]
    elif name in user_defined_discrete_colormaps:
        log.debug("found colormap %s in user_defined_discrete_colormaps" % name)
        return user_defined_discrete_colormaps[name]
    raise KeyError("unknown colormap '%s'" % name)

def register_colormap(c):
    global user_defined_discrete_colormaps

    user_defined_discrete_colormaps[c.name] = c

def user_defined_colormaps_to_json():
    e = []
    for name, colormap in user_defined_discrete_colormaps.iteritems():
        j = colormap.to_json()
        e.append(colormap.to_json())
    log.debug("serialized colormaps: %s" % (e))
    return e

def user_defined_colormaps_from_json(e):
    try:
        for j in e:
            log.debug("restoring colormap item %s" % repr(j))
            try:
                c = DiscreteColormap.from_json(j)
            except Exception, e:
                log.error("%s: Failed parsing colormap for %s" % (e, repr(j)))
            else:
                builtin_discrete_colormaps[j['name']] = c
    except Exception, e:
        log.error("%s: Invalid colormap format in json", e)
        raise
