import numpy as np

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

    def __init__(self, name, cmap, lo=None, hi=None, extra_padding=0.0):
        self.name = name
        self.cmap = self.preprocess_colormap(cmap)
        self.set_bounds(lo, hi, extra_padding)

    def preprocess_colormap(self, cmap):
        return cmap

    def set_bounds(self, lo=None, hi=None, extra_padding=0.0):
        if lo is None:
            lo = 0.0
        if hi is None:
            hi = 1.0
        self.lo_val = lo
        self.hi_val = hi
        delta = abs(hi - lo)
        self.lo_padded = lo - extra_padding * delta
        self.hi_padded = hi + extra_padding * delta
        self.mapping = self.create_colormap()

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
        return DiscreteColormap(self.name, self.source_cmap, self.lo_val, self.hi_val)

    def create_colormap(self):
        #bounds = [13.442, 13.443, 13.445, 13.448]
        bounds = np.linspace(self.lo_padded, self.hi_padded, self.cmap.N + 1)
        norm = colors.BoundaryNorm(bounds, self.cmap.N)
        mapping = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return mapping

    @property
    def bin_borders(self):
        bins = list(self.mapping.norm.boundaries)
        bins[0] = self.lo_val
        bins[-1] = self.hi_val
        return bins

    @property
    def bin_colors(self):
        c = list(self.cmap.colors)
        c[0:0] = [self.cmap._rgba_under]
        c.append(self.cmap._rgba_over)
        return colors.colorConverter.to_rgba_array(c)

    def set_bins(self, borders, colors):
        pass

builtin_continuous_colormaps = {name:ScaledColormap(name, cm.get_cmap(name)) for name in sorted(cm.cmap_continuous)}

builtin_discrete_colormaps = {name:DiscreteColormap(name, cm.get_cmap(name)) for name in sorted(cm.cmap_discrete)}

sample_discrete = colors.ListedColormap(['#ff0000', '#004400', '#007700', '#00aa00', '#0000ff'])
builtin_discrete_colormaps['sample_lat'] = DiscreteColormap('sample_lat', sample_discrete)

def get_colormap(name):
    if name in builtin_continuous_colormaps:
        return builtin_continuous_colormaps[name]
    elif name in builtin_discrete_colormaps:
        return builtin_discrete_colormaps[name]
    raise RuntimeError("DiscreteColormap!")
