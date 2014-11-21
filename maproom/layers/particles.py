"""
Layer type for particles

not much here now -- but there will be...
"""

from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

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
    # def determine_layer_color(self):
    #     ## this should do something different for particles
    
