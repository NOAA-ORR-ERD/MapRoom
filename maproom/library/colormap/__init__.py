import numpy as np

from .. import math_utils

from . import colors
from . import cm


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


class ScaledColormap(object):
    is_discrete = False

    def __init__(self, name, cmap, lo=None, hi=None, extra_padding=0.0, values=None):
        self.name = name
        self.cmap = self.preprocess_colormap(cmap)
        self.mapping = None
        self.user_defined_bin_bounds = None
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
        if lo is None:
            lo = 0.0
        if hi is None:
            hi = 1.0
        if values is not None:
            lo = min(values)
            hi = max(values)
            self.user_defined_bin_bounds = np.asarray(values, dtype=np.float32)
        self.lo_val = lo
        self.hi_val = hi
        delta = abs(hi - lo)
        self.lo_padded = lo - extra_padding * delta
        self.hi_padded = hi + extra_padding * delta
        self.mapping = self.create_colormap()

    def set_values(self, values):
        self.set_bounds(values=values)

    def calc_labels(self, lo, hi):
        return math_utils.calc_labels(lo, hi)

    def create_colormap(self):
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
            'bin_borders': self.bin_borders,
            'extra_padding': self.extra_padding,
            'name': self.name,
        }

    @classmethod
    def from_json(self, e):
        bin_colors = e['bin_colors']
        c = colors.ListedColormap(bin_colors[1:-1], e['name'])
        c.set_under(bin_colors[0])
        c.set_over(bin_colors[-1])
        return DiscreteColormap(e['name'], c, extra_padding=e['extra_padding'], values=e['bin_borders'])

    def calc_labels(self, lo, hi):
        # Return percent, text pairs for each bin boundary
        bins = self.bin_borders
        print("BOESUBRCOEUHRCOEUHSOEUHSRCOEUH", bins)
        labels = []
        rounded_labels = math_utils.round_minimum_unique_digits(bins)
        for val, rounded in zip(bins, rounded_labels):
            perc = (val - lo) / (hi - lo)
            labels.append((perc, rounded))
        return labels

    @property
    def under_rgba(self):
        return colors.colorConverter.to_rgba(self.cmap._rgba_under)

    @property
    def over_rgba(self):
        return colors.colorConverter.to_rgba(self.cmap._rgba_over)

    def preprocess_colormap(self, cmap):
        self.source_cmap = cmap
        return self.copy_source_colormap()

    def copy_source_colormap(self):
        cmap = self.source_cmap
        c = colors.ListedColormap(cmap.colors[1:-1], cmap.name)
        c.set_under(cmap.colors[0])
        c.set_over(cmap.colors[-1])
        return c

    def copy(self):
        return DiscreteColormap(self.name, self.source_cmap, self.lo_val, self.hi_val, self.extra_padding, self.user_defined_bin_bounds)

    def create_colormap(self):
        print("CREATE DISCRETE COLORMAP %s values" % self.name, self.user_defined_bin_bounds)
        if self.user_defined_bin_bounds is not None:
            bins = self.user_defined_bin_bounds[:]
            nbins = (bins - min(bins)) / max(bins)
            bounds_range = self.hi_val - self.lo_val
            bounds = (nbins * bounds_range) + self.lo_val
            print("  NEW DISCRETE COLORMAP bounds", bounds)
        else:
            bounds = np.linspace(self.lo_padded, self.hi_padded, self.cmap.N + 1)
        norm = colors.BoundaryNorm(bounds, self.cmap.N)
        mapping = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return mapping

    @property
    def bin_borders(self):
        if self.mapping is not None:
            bins = list(self.mapping.norm.boundaries)
            bins[0] = self.lo_val
            bins[-1] = self.hi_val
            print("using bin_borders: %s" % str(bins))
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
        return builtin_continuous_colormaps[name]
    elif name in builtin_discrete_colormaps:
        return builtin_discrete_colormaps[name]
    elif name in user_defined_discrete_colormaps:
        return user_defined_discrete_colormaps[name]
    raise KeyError("unknown colormap '%s'" % name)

def register_colormap(c):
    global user_defined_discrete_colormaps

    user_defined_discrete_colormaps[c.name] = c

def user_defined_colormaps_to_json():
    e = {}
    for name, colormap in user_defined_discrete_colormaps.iteritems():
        e[name] = colormap.to_json()
