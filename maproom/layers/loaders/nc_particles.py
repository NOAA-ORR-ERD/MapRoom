"""
loader for nc_particles files

i.e. GNOME output
"""

import os
import numpy as np
#import re

from fs.opener import opener

from .common import BaseLayerLoader
from maproom.layers.particles import ParticleLayer, ParticleFolder, ParticleLegend

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
        self.status_id = "status_codes"
        try:
            attributes = self.reader.get_attributes(self.status_id)
        except KeyError:
            # try "status" instead
            self.status_id = "status"
            attributes = self.reader.get_attributes(self.status_id)
        log.debug("Using '%s' for status code identifier" % self.status_id)
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

    def __next__(self):
        warning = None
        if self.current_timestep >= len(self.reader.times):
            log.debug("Finished loading")
            raise StopIteration
        # while self.current_timestep < 12:
        #     data = self.reader.get_timestep(self.current_timestep, variables=['latitude', 'longitude'])
        #     self.current_timestep += 1
        data = self.reader.get_timestep(self.current_timestep, variables=['latitude', 'longitude'])
        timecode = self.reader.times[self.current_timestep]
        points = np.c_[data['longitude'], data['latitude']]
        if self.status_id in self.reader.variables:
            data = self.reader.get_timestep(self.current_timestep, variables=[self.status_id])
            status_codes = np.array(data[self.status_id], dtype=np.uint32)
        else:
            status_codes = np.zeros(np.alen(data['longitude']), dtype=np.uint32)
        abslon = np.absolute(points[:,0])
        abslat = np.absolute(points[:,1])
        bogus = np.where((abslon > 1e3) | (abslat > 1e3))
        log.debug("Loaded timestep %s @ %s, %d points, %d bogus" % (self.current_timestep, timecode, len(points), len(bogus[0])))
        if len(bogus[0] > 0):
            log.debug("Bogus values: %s" % points[bogus[0]])
            short = "(Timestep %d) # points: %d" % (self.current_timestep + 1, len(bogus[0]))
            details = "%d spurious values in timestep %d\nindexes: %s" % (len(bogus[0]), self.current_timestep + 1, str(bogus[0]))
            warning = (short, details)

            points = np.delete(points, bogus[0], 0)
            status_codes = np.delete(status_codes, bogus[0], 0)

        scalar_vars = {}
        scalar_min_max = {}
        data = self.reader.get_timestep(self.current_timestep, variables=self.reader.variables)
        for var in self.reader.variables:
            if var in data:
                d = data[var]
                log.debug("timestep %d: %s" % (self.current_timestep, (var, d.dtype, d.shape)))
                if len(d.shape) == 1:
                    d = np.delete(d, bogus[0], 0)
                    scalar_vars[var] = d
                    scalar_min_max[var] = (min(d), max(d))
            else:
                log.warning("%s not present in timestep %d" % (var, self.current_timestep))

        # if self.current_timestep > 14:
        #     raise StopIteration

        self.current_timestep += 1

        return (points, status_codes, self.status_code_map, timecode, warning, scalar_vars, scalar_min_max)


class ParticleLoader(BaseLayerLoader):
    """
    loader for nc_particles file

    creates a bunch of particle layers, one for each tiem step in the file
    """

    mime = "application/x-nc_particles"
    layer_types = ["particle"]
    extensions = [".nc"]
    name = "nc_particles"

    def load_layers(self, metadata, manager, **kwargs):
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
        folder_min_max = {}
        folder_status_code_names = {}
        # loop through all the time steps in the file.
        for (points, status_codes, code_map, timecode, warning, scalar_vars, scalar_min_max) in nc_particles_file_loader(metadata.uri):
            layer = ParticleLayer(manager=manager, source_particle_folder=parent)
            layer.file_path = metadata.uri
            layer.mime = self.mime  # fixme: tricky here, as one file has multiple layers
            layer.name = timecode.isoformat().rsplit(':', 1)[0]
            # print timecode, type(timecode), layer.name, timecode.tzinfo
            progress_log.info("Finished loading %s" % layer.name)
            layer.set_data(points, status_codes, scalar_vars)
            layer.set_datetime(timecode)
            layers.append(layer)
            if warning:
                layer.load_warning_details = warning[1]
                warnings.append("%s %s" % (layer.name, warning[0]))

            # compute scalar vars min and max as we go through the list of
            # layers
            for k, v in scalar_min_max.items():
                lo, hi = v
                if k in folder_min_max:
                    flo, fhi = folder_min_max[k]
                    folder_min_max[k] = (float(min(lo, flo)), float(max(hi, fhi)))
                else:
                    folder_min_max[k] = (float(lo), float(hi))
            folder_status_code_names.update(code_map)

        progress_log.info("Finished loading %s" % metadata.uri)
        layers.reverse()

        # now we can tell the folder what the overall min/max are because
        # we've seen all the layers.
        parent.scalar_min_max = folder_min_max
        parent.init_status_codes(folder_status_code_names)

        # The end time for each time step defaults to the start time of the
        # subsequent step
        end_time = layers[0].start_time
        for i, layer in enumerate(layers):
            layer.init_from_parent()
            if i > 0:
                layer.end_time = end_time
                end_time = layer.start_time

        legend = ParticleLegend(manager=manager, source_particle_folder=parent)
        layers[0:0] = [parent, legend]
        if warnings:
            warnings[0:0] = ["The following layers have spurious values. Those values have been removed.\n"]
        parent.load_warning_string = "\n  ".join(warnings)
        log.debug("Adding layers: %s" % ("\n".join([str(lr) for lr in layers])))
        return layers
