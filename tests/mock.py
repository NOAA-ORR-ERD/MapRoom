import os

# Include maproom directory so that maproom modules can be imported normally
import sys
maproom_dir = os.path.realpath(os.path.abspath(".."))
if maproom_dir not in sys.path:
    sys.path.insert(0, maproom_dir)

from nose.tools import *

# Turn logging on by default at the DEBUG level for tests
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import numpy as np

from pyugrid.ugrid import UGrid

from maproom.mock import *
