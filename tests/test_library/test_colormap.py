"""
informal tests for colormap code

if nothing else, it makes sure it imports and a couple things run
"""


from maproom.library.colormap import colors as colors
from maproom.library.colormap import cm as cmx
import numpy as np

# define some random data that emulates your indeded code:
NCURVES = 5
np.random.seed(101)
curves = [np.random.random(20) for i in range(NCURVES)]
values = list(range(NCURVES))

def test_ScalarMappable():
    """
    Tests the ScalarMappable
     -- why? I don't know. but at least some code will run
    """

    jet = cmx.get_cmap('gnuplot')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    limits = scalarMap.get_clim()

    assert limits == (0, NCURVES - 1)

    for line in curves:
        colorVal = scalarMap.to_rgba(line)
        print(f"{line=}")
        print(f"{colorVal=}")
