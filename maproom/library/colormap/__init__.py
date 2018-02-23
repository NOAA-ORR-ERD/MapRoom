import numpy as np

from . import colors
from . import cm

get_cmap = cm.get_cmap
normalize = colors.Normalize
scale = cm.ScalarMappable

def create_scaled_colormap(name, vmin, vmax):
    scm = get_cmap(name)
    norm  = normalize(vmin=vmin, vmax=vmax)
    mapping = scale(norm=norm, cmap=scm)
    return mapping

def get_rgb_colors(name, values, extra_padding=0.1):
    lo = float(min(values))
    hi = float(max(values))
    delta = abs(hi - lo)
    lo -= .1 * delta
    hi += .1 * delta
    mapping = create_scaled_colormap(name, lo, hi)
    values = mapping.to_rgba(values, bytes=True)
    return values[:,0:3]

def get_opengl_colors(name, values, extra_padding=0.1):
    lo = float(min(values))
    hi = float(max(values))
    delta = abs(hi - lo)
    lo -= .1 * delta
    hi += .1 * delta
    mapping = create_scaled_colormap(name, lo, hi)
    byte_values = mapping.to_rgba(values, bytes=True)
    return np.array(byte_values).view(np.uint32)[:,0]

def list_colormaps():
    return sorted(cm.cmap_d.keys())
