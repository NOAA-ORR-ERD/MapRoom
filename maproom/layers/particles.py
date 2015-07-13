"""
Layer type for particles

not much here now -- but there will be...
"""
import numpy as np

from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..renderer import color_floats_to_int

from point_base import PointBaseLayer

# import logging
# log = logging.getLogger(__name__)
# progress_log = logging.getLogger("progress")


class ParticleLayer(PointBaseLayer):
    """Layer for particle files from GNOME, etc.

       not much here, but this class exists because we're going to add stuff.
    """
    name = Unicode("Particle Layer")
    
    type = Str("particle")
    
    # FIXME: Arbitrary colors for now till we decide on values
    status_code_to_color = np.array([color_floats_to_int(0, 0, 0, 1.0),
                                     color_floats_to_int(1.0, 0, 0, 1.0),
                                     color_floats_to_int(0, 1.0, 0, 1.0),
                                     color_floats_to_int(0, 0, 1.0, 1.0),
                                     color_floats_to_int(0, 1.0, 1.0, 1.0),
                                     ], dtype=np.uint32)
    
    # def set_layer_style_defaults(self):
    #     ## this should do something different for particles
    
    def set_data(self, f_points, status_codes):
        PointBaseLayer.set_data(self, f_points)
        # force status codes to fall into range of valid colors
        status_codes = np.clip(status_codes, 0, np.alen(self.status_code_to_color))
        colors = self.status_code_to_color[status_codes]
        self.points.color = colors
