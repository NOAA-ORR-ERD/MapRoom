"""
loader for nc_particles files

i.e. GNOME output
"""

import os
import numpy as np
#import re

from common import BaseLayerLoader
from maproom.layers.particles import ParticleLayer, ParticleFolder

import logging
progress_log = logging.getLogger("progress")

from post_gnome import nc_particles

class nc_particles_file_loader():
    """
    iterator for loading all the timesteps in an nc_particles file

    note: this should probably be in the nc_particles lib...

    """
    def __init__(self, file_path):
        self.reader = nc_particles.Reader(file_path)
        self.current_timestep = 0 # fixme## hard coded limit!!!!!

    def __iter__(self):
        return self

    def next(self):
        if self.current_timestep >= len(self.reader.times):
            raise StopIteration
        data = self.reader.get_timestep(self.current_timestep, variables=['latitude', 'longitude'])
        time = self.reader.times[self.current_timestep]
        points = np.c_[data['longitude'], data['latitude']]
        if 'status_codes' in self.reader.variables:
            data = self.reader.get_timestep(self.current_timestep, variables=['status_codes'])
            status_codes = np.array(data['status_codes'], dtype=np.uint32)
        else:
            status_codes = np.zeros(np.alen(data['longitude']), dtype=np.uint32)

        self.current_timestep += 1

        return (points, status_codes, time) # error_string, points, time

class ParticleLoader(BaseLayerLoader):
    """
    loader for nc_particles file

    creates a bunch of particle layers, one for each tiem step in the file
    """

    mime = "application/x-nc_particles"
    layer_types = ["particle"]
    extensions = [".nc"]
    name = "nc_particles"
    
    def load_layers(self, metadata, manager):
        """
        load the nc_particles file

        :param metadata: the metadata object from the file opener guess object.

        :param manager: The layer manager

        """
        parent = ParticleFolder(manager=manager)
        parent.file_path = metadata.uri
        parent.mime = self.mime ## fixme: tricky here, as one file has multiple layers
        parent.name = os.path.split(parent.file_path)[1]
        layers = [parent]
        ## loop through all the time steps in the file.
        for (points, status_codes, time) in nc_particles_file_loader(metadata.uri):
            layer = ParticleLayer(manager=manager)
            layer.file_path = metadata.uri
            layer.mime = self.mime ## fixme: tricky here, as one file has multiple layers
            layer.name = time.isoformat().rsplit(':',1)[0]
            progress_log.info("Finished loading %s" % layer.name)
            layer.set_data(points, status_codes)
            layers.append(layer)
        progress_log.info("Finished loading %s" % metadata.uri)
        return layers
