import multiprocessing
import queue
import cython
import numpy as np
cimport numpy as np

np.import_array()


cdef extern from "defines.h":
    pass


cdef extern from "triangle.h":
    cdef struct triangulateio:
        double* pointlist
        double* pointattributelist
        int* pointmarkerlist
        int numberofpoints
        int numberofpointattributes

        int* trianglelist
        double* triangleattributelist
        double* trianglearealist
        int* neighborlist
        int numberoftriangles
        int numberofcorners
        int numberoftriangleattributes

        int* segmentlist
        int* segmentmarkerlist
        int numberofsegments

        double* holelist
        int numberofholes

        double* regionlist
        int numberofregions

        int* edgelist
        int* edgemarkerlist
        double* normlist
        int numberofedges

    void triangulate(
        char* triswitches,
        triangulateio* in_data,
        triangulateio* out_data,
        triangulateio* vorout,
    )


cdef unsigned int PROCESSING_TIMEOUT = 5


@cython.boundscheck( False )
def triangulate_simple(
    param_text,
    np.ndarray[ double, ndim = 2 ] points_xy not None,
    np.ndarray[ float ] points_z not None,
    np.ndarray[ unsigned int, ndim = 2 ] lines not None,
    np.ndarray[ double, ndim = 2 ] hole_points_xy not None,
):
    """
    Triangulate the area bounded by the given lines and using the given
    points. This is accomplished with quality mesh generation by Delaunay
    refinement.

    The point depth values are interpolated to populate the depth values of
    the resulting output points.

    Because an unsuccessful triangulation might exit() or hang, this function
    spawns a separate process to actually perform the triangulation, thereby 
    preventing any negative impact from disturbing the caller.

    :param points_xy: geographic points to triangulate
    :type points_xy: Nx2 numpy array of type float, where each row is the 2D
                     coordinate of a point
    :param points_z: depth values of geographic points
    :type points_z: scalar numpy array of type float
    :param lines: line segments describing outer boundary and holes of area to
                  triangulate
    :type lines: Nx2 numpy array of type unsigned int, where each row is a pair of
                 segment endpoint indices in the points_xy array
    :param hole_points_xy: geographic points describing hole regions to
                           exclude from triangulation
    :type hole_points_xy: Nx2 numpy array of type float, where each row is the
                          2D coordinate of any point within a hole (one point
                          per hole region)
    :return: (
        triangle corner points xy,
        triangle corner points z,
        indices of each line segment point,
        indices of each triangle point,
    )
    :rtype: (
        Nx2 numpy array of type float, with a 2D point coordinate per row,
        scalar numpy array of type float, with a point depth per row,
        Nx2 numpy array of type int, with two point indices per row,
        Nx3 numpy array of type int, with three point indices per row,
    )
    :raises: RuntimeError in the case of a triangulation timeout
    :raises: EOFError in the case of an unsuccessful triangulation
    """
    
    # if not isinstance( param_text, unicode ):
    #     raise ValueError( "requires text input, got %s" % type( param_text ) )
    q = multiprocessing.Queue()
    child = multiprocessing.Process(
        target = triangulate_simple_child,
        args = (q, param_text, points_xy, points_z, lines, hole_points_xy),
    )
    child.daemon = True
    child.start()

    # For a large amount of data, have to pull from the queue before joining
    # the process. See https://bugs.python.org/issue8426
    try:
        output = q.get(True, PROCESSING_TIMEOUT)
    except queue.Empty:
        timeout = True
    else:
        timeout = False

    child.join(PROCESSING_TIMEOUT)
    if child.is_alive():
        # timeout reached! Kill the process
        child.terminate()
        timeout = True
    if timeout:
        raise RuntimeError("Triangulation timeout.")
    try:
        out_points_xy, out_points_z, out_lines, out_triangles = output
    except ValueError:
        raise RuntimeError("Triangulation error: no output")

    return out_points_xy, out_points_z, out_lines, out_triangles


@cython.boundscheck( False )
def triangulate_simple_child(
    output_queue,
    param_text,
    np.ndarray[ double, ndim = 2 ] points_xy not None,
    np.ndarray[ float ] points_z not None,
    np.ndarray[ unsigned int, ndim = 2 ] lines not None,
    np.ndarray[ double, ndim = 2 ] hole_points_xy not None,
):
    """
    Call Shewchuk's Triangle library to generate the Delauney triangulation.

    Must be called in its own subprocess because the library may terminate or
    even hang if the triangulation is not successful.
    """
    switchesP = b"pzB" + param_text.encode('UTF-8') # TODO: Add -a# -q# -Q
    cdef char* switches = switchesP
    cdef triangulateio in_data, out_data

    cdef np.ndarray[np.float64_t] points_z_64 = points_z.astype(np.float64)

    # Pass the data to Triangle and triangulate with it.
    in_data.pointlist = <double*>points_xy.data
    in_data.pointattributelist = <double*>points_z_64.data
    in_data.pointmarkerlist = NULL
    in_data.numberofpoints = <int>points_xy.shape[0]
    in_data.numberofpointattributes = 1
    in_data.trianglelist = NULL
    in_data.triangleattributelist = NULL
    in_data.trianglearealist = NULL
    in_data.numberoftriangles = 0
    in_data.numberofcorners = 0
    in_data.numberoftriangleattributes = 0
    in_data.segmentlist = <int*>lines.data # Note the conversion to signed!
    in_data.segmentmarkerlist = NULL
    in_data.numberofsegments = <int>lines.shape[0]
    in_data.holelist = <double*>hole_points_xy.data
    in_data.numberofholes = <int>hole_points_xy.shape[0]
    in_data.regionlist = NULL
    in_data.numberofregions = 0

    out_data.pointlist = NULL
    out_data.pointattributelist = NULL
    out_data.trianglelist = NULL
    out_data.segmentlist = NULL

    print(f"Calling triangulate")
    triangulate( switches, &in_data, &out_data, NULL )
    print(f"Returned from triangulate")

    out_points_xy = np.PyArray_SimpleNewFromData(2, [out_data.numberofpoints, 2], np.NPY_DOUBLE, out_data.pointlist)
    print(f"out_points_xy: {len(out_points_xy)}")

    out_points_z = np.PyArray_SimpleNewFromData(1, [out_data.numberofpoints], np.NPY_DOUBLE, out_data.pointattributelist).astype(np.float32)
    print(f"out_points_z: {len(out_points_z)}")

    out_lines = np.PyArray_SimpleNewFromData(2, [out_data.numberofsegments, 2], np.NPY_INT, out_data.segmentlist)
    print(f"out_lines: {len(out_lines)}")

    out_triangles = np.PyArray_SimpleNewFromData(2, [out_data.numberoftriangles, out_data.numberofcorners], np.NPY_INT, out_data.trianglelist)
    print(f"out_triangles: {len(out_triangles)}")

    output_queue.put((out_points_xy, out_points_z, out_lines, out_triangles))
