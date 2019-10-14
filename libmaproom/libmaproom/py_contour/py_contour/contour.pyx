
# Cython code for calling contouring algorithm described here:

# http://paulbourke.net/papers/conrec/

# The C version derived from the fortran version by Paul Bourke
 

# d               ! matrix of data to contour
# ilb,iub,jlb,jub ! index bounds of data matrix
# x               ! data matrix column coordinates
# y               ! data matrix row coordinates
# nc              ! number of contour levels
# z               ! contour levels in increasing order

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cdef extern from "conrec.cxx":
    int conrec(double **d,
               int iub,
               int jub,
               double *x,
               double *y,
               int nc,
               double *z,
               void (*ConrecLine)(double, double, double, double, double)
               );


## Fixme: using a global to hold th countour segments -- NOT thread safe!
contour_segments = {}

cdef void conrecline(double x1, double y1, double x2, double y2, double z):
     contour_segments.setdefault(z, []).append( ((x1, y1), (x2, y2)) ) 

def contour( data,
             x,
             y,
             levels,
             ):
    """
    contour the data in the data array, with one contour per level

    :param data: the data to contour -- rectangular array of values

    :param x: the x-coord of the data

    :param y: the y-coord of the data

    :param levels: the levels you want contoured
    """
    cdef cnp.ndarray[double, ndim=2, mode='c'] data_arr = np.asarray(data, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] x_arr = np.asarray(x, dtype=np.float64).reshape((-1,))
    cdef cnp.ndarray[double, ndim=1] y_arr = np.asarray(y, dtype=np.float64).reshape((-1,))
    cdef cnp.ndarray[double, ndim=1] levels_arr = np.asarray(levels, dtype=np.float64)

    # check if array sizes match:
    if x_arr.shape[0] != data_arr.shape[0] or y_arr.shape[0] != data_arr.shape[1]:
        raise ValueError("x and y arrays much match the size of the data array")
    #fixme:  check that levels is monotonically increasing.

    # call the contouring routine...
    ## routine required pointer to a pointer...
    cdef double** pointer_arr = <double**>malloc(data.shape[0] * sizeof(double*))
    if not pointer_arr:
        raise MemoryError()
    cdef int i
    # fill in the pointer array
    for i in range(data_arr.shape[0]):
        pointer_arr[i] = &data_arr[i,0]
    # clear dict before calling:
    contour_segments.clear()
    conrec( pointer_arr,
            data_arr.shape[0]-1,
            data_arr.shape[1]-1,
            &x_arr[0],
            &y_arr[0],
            len(levels),
            &levels_arr[0],
            conrecline
          )
    free(pointer_arr)
    return contour_segments.copy()

def sort_segments(segments):
    """
    Sorts the segments to make polygons and polylines

    :param segments: list of pairs of coordinates to sort:
       [((0.0),(1,1)),
        ((2,3),(0,0)),
        ((1,1),(2,3)),
        ....
        ]

    :returns: list of polygons. each polygon is a list of coordinate pairs.

    """
    # must be a list (or, mutable sequence anyway...)
    if type(segments) is np.ndarray:
        segments = segments.tolist()
    # just in case it's a tuple or something
    segments = list(segments) if type(segments) is not list else segments

    polygons = []
    # Go backwards for efficieny
    while segments: # are there any left unaccounted for?
        # start a new polygon
        seg = segments.pop() # might as well take the last one
        polygons.append([seg[0], seg[1]])
        # look for segments to add to that polygon:
        while True:
            for seg in (segments):
                # see if it belongs to the last polygon:
                poly = polygons[-1]
                if seg[0] == poly[-1]:
                    poly.append(seg[1])
                    segments.remove(seg)
                    break #start again at the beginning
                elif seg[1] == poly[-1]:
                    poly.append(seg[0])
                    segments.remove(seg)
                    break #start again at the beginning
                # maybe it fits on the beginning:
                elif seg[0] == poly[0]:
                    poly.insert(0,seg[1])
                    segments.remove(seg)
                    break #start again at the beginning
                elif seg[1] == poly[0]:
                    poly.insert(0,seg[0])
                    segments.remove(seg)
                    break #start again at the beginning
                # segment doesn't match anything, moving to next segment
            else: # did not find a matching segment -- done with that polygon
                break # break out of while loop
                      # only if we get through the for loop without hitting a matching segment
    return polygons








