

import os

# Include maproom directory so that maproom modules can be imported normally
import sys
maproom_dir = os.path.realpath(os.path.abspath(".."))
if maproom_dir not in sys.path:
    sys.path.insert(0, maproom_dir)

import pytest
try:
    slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
        )
except AttributeError:
    # pytest doesn't load the config module when not run using py.test
    # skip this check when running a test_*.py from the command line
    import functools
    slow = lambda a: functools.partial(print, "skipping slow test %s" % repr(a))

# Turn logging on by default at the DEBUG level for tests
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import numpy as np
from numpy.testing import assert_almost_equal

from pyugrid.ugrid import UGrid

from maproom.mock import *

# Initialize default styles, but don't load user styles
from maproom.styles import replace_default_styles
replace_default_styles(None)
