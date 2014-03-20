# Hack to begin the transition away from global variables!
import os

# this is a shared object to avoid having to pass key singletons throughout the code
main_logger = None
opengl_logger = None
version = None

import sys
import peppy2
image_path = peppy2.get_image_path("maproom/ui/images", sys.modules[__name__])

preferences = {
    "Coordinate Display Format": "degrees decimal minutes",
    "Scroll Zoom Speed": "Slow",
    
    # FIXME: preferences doesn't handle non-string items when saving/loading
    "Number of Recent Files": 20,
}

error_email_from = "maproombugreports@gmail.com"
error_email_passwd = "bushy206"
error_email_to = "rob.mcmullen@noaa.gov"
