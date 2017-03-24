"""
loader for nc_particles files

i.e. GNOME output
"""

import os
import numpy as np
#import re

from fs.opener import opener

from common import BaseLayerLoader
from maproom.layers.particles import ParticleLayer, ParticleFolder

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

from post_gnome import nc_particles


class nc_particles_file_loader():
    """
    iterator for loading all the timesteps in an nc_particles file

    note: this should probably be in the nc_particles lib...

    """

    def __init__(self, uri):
        fs, relpath = opener.parse(uri)
        if not fs.hassyspath(relpath):
            raise RuntimeError("Only file URIs are supported for NetCDF: %s" % uri)
        path = fs.getsyspath(relpath)
        self.reader = nc_particles.Reader(path)
        self.current_timestep = 0  # fixme## hard coded limit!!!!!
        try:
            attributes = self.reader.get_attributes("status_codes")
        except KeyError:
            # try "status" instead
            attributes = self.reader.get_attributes("status")
        meanings = attributes['flag_meanings']
        self.status_code_map = dict()
        if "," in meanings:
            splitchar = ","
        else:
            splitchar = " "
        for code in meanings.split(splitchar):
            if code and ":" in code:
                k, v = code.split(":", 1)
                self.status_code_map[int(k.strip())] = v.strip()

    def __iter__(self):
        return self

    def next(self):
        warning = None
        if self.current_timestep >= len(self.reader.times):
            log.debug("Finished loading")
            raise StopIteration
        # while self.current_timestep < 12:
        #     data = self.reader.get_timestep(self.current_timestep, variables=['latitude', 'longitude'])
        #     self.current_timestep += 1
        data = self.reader.get_timestep(self.current_timestep, variables=['latitude', 'longitude'])
        time = self.reader.times[self.current_timestep]
        points = np.c_[data['longitude'], data['latitude']]
        if 'status_codes' in self.reader.variables:
            data = self.reader.get_timestep(self.current_timestep, variables=['status_codes'])
            status_codes = np.array(data['status_codes'], dtype=np.uint32)
        else:
            status_codes = np.zeros(np.alen(data['longitude']), dtype=np.uint32)
        abslon = np.absolute(points[:,0])
        abslat = np.absolute(points[:,1])
        bogus = np.where((abslon > 1e3) | (abslat > 1e3))
        log.debug("Loaded timestep %s @ %s, %d points, %d bogus" % (self.current_timestep, time, len(points), len(bogus[0])))
        if len(bogus[0] > 0):
            log.debug("Bogus values: %s" % points[bogus[0]])
            short = "(Timestep %d) # points: %d" % (self.current_timestep + 1, len(bogus[0]))
            details = "%d spurious values in timestep %d\nindexes: %s" % (len(bogus[0]), self.current_timestep + 1, str(bogus[0]))
            warning = (short, details)
            print bogus[0], points[bogus[0]]

            points = np.delete(points, bogus[0], 0)
            status_codes = np.delete(status_codes, bogus[0], 0)

        # if self.current_timestep > 14:
        #     raise StopIteration

        self.current_timestep += 1

        return (points, status_codes, self.status_code_map, time, warning)


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
        parent.mime = self.mime  # fixme: tricky here, as one file has multiple layers
        parent.name = os.path.split(parent.file_path)[1]

        warnings = []
        layers = []
        # loop through all the time steps in the file.
        for (points, status_codes, code_map, time, warning) in nc_particles_file_loader(metadata.uri):
            layer = ParticleLayer(manager=manager)
            layer.file_path = metadata.uri
            layer.mime = self.mime  # fixme: tricky here, as one file has multiple layers
            layer.name = time.isoformat().rsplit(':', 1)[0]
            progress_log.info("Finished loading %s" % layer.name)
            layer.set_data(points, status_codes, code_map)
            layers.append(layer)
            if warning:
                layer.load_warning_details = warning[1]
                warnings.append("%s %s" % (layer.name, warning[0]))
        progress_log.info("Finished loading %s" % metadata.uri)
        layers.reverse()
        layers[0:0] = [parent]
        if warnings:
            warnings[0:0] = ["The following layers have spurious values. Those values have been removed.\n"]
        parent.load_warning_string = "\n  ".join(warnings)
        log.debug("Adding layers: %s" % ("\n".join([str(layer) for layer in layers])))
        return layers
