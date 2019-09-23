import cython
import numpy as np
cimport numpy as np


@cython.boundscheck( False )
def scale_half( np.ndarray[ np.uint8_t, ndim = 3, mode = "c" ] a not None ):
    """
    Make a scaled-down version of the given image at half its original width
    and height.

    :param a: RGBA image to scale
    :type a: HxWx4 numpy array of type uint8
    :return: scaled half-size RGBA image
    :rtype: HxWx4 numpy array of type uint8
    """
    cdef unsigned int i, j, b, w, h, d
    h, w, d = a.shape[ 0 ], a.shape[ 1 ], a.shape[ 2 ]

    cdef np.ndarray[ np.uint8_t, ndim=3, mode="c" ] a2 = \
       np.ndarray( ( h / 2, w / 2, 4 ), np.uint8 )

    for j in range( h / 2 ):
        for i in range( w / 2 ):
            for b in range( d ): # color band
                a2[ j, i, b ] = (
                   <int>a[ 2*j, 2*i, b ] + a[ 2*j+1, 2*i, b ] +
                        a[ 2*j, 2*i+1, b ] + a[ 2*j+1, 2*i+1, b ]
                ) / 4

    return a2


@cython.boundscheck( False )
def expand(
    np.ndarray[ np.uint8_t, ndim = 3, mode = "c" ] image not None,
    np.uint32_t new_width,
    np.uint32_t new_height,
):
    """
    Make an expanded version of the given image at a larger size. The image
    is not scaled, but rather is pasted into the upper-left of a new
    transparent image.

    If the given image's dimensions are equal to the new width and height,
    then the image is simply returned as-is.

    :param image: RGBA image to expand
    :type image: HxWx4 numpy array of type uint8
    :param new_width: width of the new, larger image
    :type new_width: uint32
    :param new_height: height of the new, larger image
    :type new_height: uint32
    :return: new, larger image
    :rtype: HxWx4 numpy array of type uint8
    """
    if image.shape[ 0 ] == new_height and image.shape[ 1 ] == new_width:
        return image

    cdef np.ndarray[ np.uint8_t, ndim=3, mode="c" ] expanded = \
       np.ndarray( ( new_height, new_width, 4 ), np.uint8 )

    # Set the initial alpha channel of the expanded image to transparent.
    cdef unsigned int i, j, b

    for j in range( new_height ):
        for i in range( new_width ):
            expanded[ j, i, 0 ] = 255
            expanded[ j, i, 1 ] = 255
            expanded[ j, i, 2 ] = 255
            expanded[ j, i, 3 ] = 0

    # Copy the original image into the upper-left of the expanded image.
    for j in range( image.shape[ 0 ] ):
        for i in range( image.shape[ 1 ] ):
            for b in range( 4 ):
                expanded[ j, i, b ] = image[ j, i, b ]

    return expanded


@cython.boundscheck( False )
def paletted_to_rgba(
    np.ndarray[ np.uint8_t, ndim=2, mode="c" ] band not None,
    np.ndarray[ np.uint8_t, ndim=2, mode="c" ] palette not None,
    np.uint8_t alpha,
):
    """
    Convert a paletted 8-bit RGB image to an RGBA image.

    :param band: 8-bit RGB image to convert
    :type band: HxW numpy array of type uint8
    :param palette: RGB palette to use during the conversion
    :type palette: 256x3 numpy array of type uint8
    :param alpha: alpha color to apply to the entire image
    :type alpha: uint8
    :return: converted RGBA image
    :rtype: HxWx4 numpy array of type uint8
    """

    cdef unsigned int i, j, w, h, c

    if palette.shape[ 0 ] != 256 or palette.shape[ 1 ] != 3:
        raise ValueError( "palette must be an 256x3 array of uint8" )

    h, w = band.shape[ 0 ], band.shape[ 1 ]

    cdef np.ndarray[ np.uint8_t, ndim=3, mode="c" ] image = \
        np.ndarray( ( h, w, 4 ), np.uint8 )

    for i in range( h ):
        for j in range( w ):
            c = band[ i, j ]
            image[ i, j, 0 ] = palette[ c, 0 ]
            image[ i, j, 1 ] = palette[ c, 1 ]
            image[ i, j, 2 ] = palette[ c, 2 ]
            image[ i ,j, 3 ] = alpha

    return image
