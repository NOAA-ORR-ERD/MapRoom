from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from maproom.library.colormap import colors as colors
from maproom.library.colormap import cm as cmx
import numpy as np

# define some random data that emulates your indeded code:
NCURVES = 10
np.random.seed(101)
curves = [np.random.random(20) for i in range(NCURVES)]
print(curves)
values = list(range(NCURVES))

jet = cm = cmx.get_cmap('gnuplot') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
print(scalarMap.get_clim())

lines = []
for idx in range(len(curves)):
    line = curves[idx]
    colorVal = scalarMap.to_rgba(line)
    print(line)
    print(colorVal)
