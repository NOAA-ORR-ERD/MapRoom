from traits.api import HasTraits, provides

from omnivore.file_type.i_file_recognizer import IFileRecognizer, RecognizerBase

# replaced with the UGRID_Recognizer
# @provides(IFileRecognizer)
# class HDF5Recognizer(RecognizerBase):
#     """Recognizer for HDF5

#     """
#     id = "application/x-hdf"

#     # GDAL recognizes HDF files, so this needs to be before the GDAL recognizer
#     before = "image/x-gdal"

#     def identify(self, guess):
#         byte_stream = guess.bytes
#         if byte_stream[0:8] == "\211HDF\r\n\032\n":
#             return self.id


@provides(IFileRecognizer)
class UGRID_Recognizer(RecognizerBase):
    """Recognizer for UGRID netcdf files

    These can be HDF (netcdf4) or CDF (netcdf3)

    but this looks for the file attributes that it should have, so should find either.
    """

    id = "application/x-nc_ugrid"

    # GDAL recognizes HDF files, so this needs to be before the GDAL recognizer
    before = "image/x-gdal"

    def identify(self, guess):
        byte_stream = guess.bytes
        # check if it is either HDF or CDF
        if byte_stream[:3] == "CDF" or byte_stream[0:8] == "\211HDF\r\n\032\n":
            if ('cf_role' in byte_stream) and ('mesh_topology' in byte_stream):
                return self.id


@provides(IFileRecognizer)
class NC_ParticleRecognizer(RecognizerBase):
    """Recognizer for nc_particles file.

    These can be HDF (netcdf4) or CDF (netcdf3)

    but this looks for the file attributes that it should have
    """

    id = "application/x-nc_particles"

    # GDAL recognizes HDF files, so this needs to be before the GDAL recognizer
    before = "image/x-gdal"

    def identify(self, guess):
        byte_stream = guess.bytes
        # check if is is HDF or CDF
        if byte_stream[:3] == "CDF" or byte_stream[0:8] == "\211HDF\r\n\032\n":
            if ('feature_type' in byte_stream) and ('particle_trajector' in byte_stream):
                return self.id
