# Hack to begin the transition away from global variables!
import os

# this is a shared object to avoid having to pass key singletons throughout the code
main_logger = None
opengl_logger = None
version = None

import sys
frozen = getattr(sys, 'frozen', False)
image_path = os.path.join(os.path.dirname(__file__), "maproom/ui/images")
if frozen and frozen in ('macosx_app'):
    print "FROZEN!!! %s" % frozen
    root = os.environ['RESOURCEPATH']
    zippath, image_path = image_path.split(".zip/")
    image_path = os.path.join(root, image_path)

preferences = {
    "Coordinate Display Format": "degrees decimal minutes",
    "Scroll Zoom Speed": "Slow",
    
    # FIXME: preferences doesn't handle non-string items when saving/loading
    "Number of Recent Files": 20,
}

error_email_from = "maproombugreports@gmail.com"
error_email_passwd = "bushy206"
error_email_to = "rob.mcmullen@noaa.gov"
