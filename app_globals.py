# Hack to begin the transition away from global variables!
import os

# this is a shared object to avoid having to pass key singletons throughout the code
main_logger = None
opengl_logger = None
version = None

import sys
import peppy2
image_path = peppy2.get_image_path("maproom/ui/images", sys.modules[__name__])
