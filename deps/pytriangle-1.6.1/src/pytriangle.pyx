import multiprocessing
import cython
import numpy as np
cimport numpy as np
cimport libc.stdlib

np.import_array()

cdef extern from "numpy/arrayobject.h":
    # We only care about the flags member.
    ctypedef struct PyArrayObject:
        void* _ob_next
        void* _ob_prev
        Py_ssize_t ob_refcnt
        void* ob_type
        char* data
        int nd
        np.npy_intp* dimensions
        np.npy_intp* strides
        void* base
        void* descr
        int flags
        void* weakreflist


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


cdef unsigned int TIMEOUT = 1


@cython.boundscheck( False )
def triangulate_simple(
    param_text,
    np.ndarray[ float, ndim = 2 ] points_xy not None,
    np.ndarray[ float ] points_z not None,
    np.ndarray[ unsigned int, ndim = 2 ] lines not None,
    np.ndarray[ float, ndim = 2 ] hole_points_xy not None,
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
    utf8_param_text = param_text.encode( 'UTF-8' )
    
    ( connection, child_connection ) = \
        multiprocessing.Pipe( duplex = True )

    child = multiprocessing.Process(
        target = triangulate_simple_child,
        args = ( child_connection, utf8_param_text ),
    )
    child.daemon = True
    child.start()

    # Send data to be triangulated to child process.
    connection.send( points_xy.shape[ 0 ] )
    connection.send_bytes( points_xy )
    connection.send_bytes( points_z )

    connection.send( lines.shape[ 0 ] )
    connection.send_bytes( lines )

    connection.send( hole_points_xy.shape[ 0 ] )
    connection.send_bytes( hole_points_xy )

    # Receive data resulting from triangulation from child process. Allow
    # longer for the initial timeout since it includes triangulation time.
    if not connection.poll( TIMEOUT * 5 ):
        cleanup_child( connection, child )
        raise RuntimeError( "Triangulation timeout." )
    cdef unsigned int out_point_count = <unsigned int>connection.recv()

    cdef np.ndarray[ np.float32_t, ndim = 2 ] out_points_xy = \
       np.ndarray( ( out_point_count, 2 ), np.float32 )

    poll_with_timeout( connection, child )
    connection.recv_bytes_into( out_points_xy )

    cdef np.ndarray[ np.float32_t ] out_points_z = \
       np.ndarray( ( out_point_count, ), np.float32 )

    poll_with_timeout( connection, child )
    connection.recv_bytes_into( out_points_z )

    poll_with_timeout( connection, child )
    cdef unsigned int out_line_count = <unsigned int>connection.recv()

    cdef np.ndarray[ np.uint32_t, ndim = 2 ] out_lines = \
       np.ndarray( ( out_line_count, 2 ), np.uint32 )

    poll_with_timeout( connection, child )
    connection.recv_bytes_into( out_lines )

    poll_with_timeout( connection, child )
    cdef unsigned int out_triangle_count = <unsigned int>connection.recv()

    cdef np.ndarray[ np.uint32_t, ndim = 2 ] out_triangles = \
       np.ndarray( ( out_triangle_count, 3 ), np.uint32 )

    poll_with_timeout( connection, child )
    connection.recv_bytes_into( out_triangles )

    cleanup_child( connection, child )

    return ( out_points_xy, out_points_z, out_lines, out_triangles )


cdef poll_with_timeout( connection, child ):
    if not connection.poll( TIMEOUT ):
        cleanup_child( connection, child )
        raise RuntimeError( "Triangulation timeout." )


cdef cleanup_child( connection, child ):
    connection.close()
    child.join( TIMEOUT )

    if child.is_alive():
        child.terminate()


@cython.boundscheck( False )
def triangulate_simple_child( connection, utf8_param_text ):
    """
    Triangulate the area bounded by the lines and points received from the
    given pipe connections. This is accomplished with quality mesh generation
    by Delaunay refinement.

    The point depth values are interpolated to populate the depth values of
    the resulting output points.

    This function is intended to be invoked as a separate child process.

    :param connection: duplex pipe connection used to exchange data with
                       parent process
    :type connection: multiprocessing.Connection

    The following buffers are read in order from the connection:

    :param point_count: number of points to triangulate
    :type point_count: int
    :param points_xy: geographic points to triangulate
    :type points_xy: buffer of type float, where each pair of elements
                     represent the 2D coordinate of a point
    :param points_z: depth values of geographic points
    :type points_z: buffer of type float
    :param line_count: number of line segments to triangulate
    :type line_count: int
    :param lines: line segments describing outer boundary and holes of area to
                  triangulate
    :type lines: buffer of type int, where each pair of elements represent
                 segment endpoint indices in the points_xy array
    :param hole_point_count: number of hole points to exclude from
                             triangulation
    :type hole_point_count: int
    :param hole_points_xy: geographic points describing hole regions to
                           exclude from triangulation
    :type hole_points_xy: buffer of type float, where each pair of elements
                          represent the 2D coordinate of any point within a
                          hole (one point per hole region)

    If the triangulation is successful, the following data and buffers are
    written in order to the connection:

    :param out_point_count: number of points resulting from triangulation
    :type out_point_count: int
    :param out_points_xy: geographic points resulting from triangulation
    :type out_points_xy: buffer of type float, where each pair of elements
                         represent the 2D coordinate of a point
    :param out_points_z: depth values of geographic points resulting from
                         triangulation
    :type out_points_z: buffer of type float
    :param out_line_count: number of line segments resulting from triangulation
    :type out_line_count: int
    :param out_lines: line segments describing outer boundary and holes
                      resulting from triangulation
    :type out_lines: buffer of type int, where each pair of elements represent
                     segment endpoint indices in the out_points_xy array
    :param out_triangle_count: number of triangles resulting from
                               triangulation
    :type out_triangle_count: int
    :param out_triangles: triangles resulting from triangulation
    :type out_triangles: buffer of type int, where each triple of elements
                         represent triangle corner indices in the
                         out_points_xy array

    If the triangulation is not successful, then the child process may simply
    terminate or even hang.
    """
    switchesP = "pqzB" + utf8_param_text # TODO: Add -a# -q# -Q
    cdef char* switches = switchesP
    cdef triangulateio in_data, out_data

    # Receive incoming data from the parent process.
    if not connection.poll( TIMEOUT ): return
    cdef unsigned int point_count = <unsigned int>connection.recv()

    # Receive as 32-bit floats, then convert to 64-bit floats below, so the
    # Triangle library has more precision to work with.
    cdef np.ndarray[ np.float32_t, ndim = 2 ] points_xy = \
       np.ndarray( ( point_count, 2 ), np.float32 )

    if not connection.poll( TIMEOUT ): return
    connection.recv_bytes_into( points_xy )

    cdef np.ndarray[ np.float64_t, ndim = 2 ] points_xy_64 = \
       points_xy.astype( np.float64 )

    cdef np.ndarray[ np.float32_t ] points_z = \
       np.ndarray( ( point_count, ), np.float32 )

    if not connection.poll( TIMEOUT ): return
    connection.recv_bytes_into( points_z )

    cdef np.ndarray[ np.float64_t ] points_z_64 = \
       points_z.astype( np.float64 )

    if not connection.poll( TIMEOUT ): return
    cdef unsigned int line_count = <unsigned int>connection.recv()

    cdef np.ndarray[ np.uint32_t, ndim = 2 ] lines = \
       np.ndarray( ( line_count, 2 ), np.uint32 )

    if not connection.poll( TIMEOUT ): return
    connection.recv_bytes_into( lines )

    if not connection.poll( TIMEOUT ): return
    cdef unsigned int hole_point_count = <unsigned int>connection.recv()

    cdef np.ndarray[ np.float32_t, ndim = 2 ] hole_points_xy = \
       np.ndarray( ( hole_point_count, 2 ), np.float32 )

    if not connection.poll( TIMEOUT ): return
    connection.recv_bytes_into( hole_points_xy )

    cdef np.ndarray[ np.float64_t, ndim = 2 ] hole_points_xy_64 = \
       hole_points_xy.astype( np.float64 )

    # Pass the data to Triangle and triangulate with it.
    in_data.pointlist = <double*>points_xy_64.data
    in_data.pointattributelist = <double*>points_z_64.data
    in_data.pointmarkerlist = NULL
    in_data.numberofpoints = point_count
    in_data.numberofpointattributes = 1
    in_data.trianglelist = NULL
    in_data.triangleattributelist = NULL
    in_data.trianglearealist = NULL
    in_data.numberoftriangles = 0
    in_data.numberofcorners = 0
    in_data.numberoftriangleattributes = 0
    in_data.segmentlist = <int*>lines.data # Note the conversion to signed!
    in_data.segmentmarkerlist = NULL
    in_data.numberofsegments = lines.shape[ 0 ]
    in_data.holelist = <double*>hole_points_xy_64.data
    in_data.numberofholes = hole_points_xy.shape[ 0 ]
    in_data.regionlist = NULL
    in_data.numberofregions = 0

    out_data.pointlist = NULL
    out_data.pointattributelist = NULL
    out_data.trianglelist = NULL
    out_data.segmentlist = NULL

    triangulate( switches, &in_data, &out_data, NULL )

    # Create NumPy arrays from the output data, giving NumPy ownership of (and
    # responsibility for ultimately deallocating) that memory. Convert the
    # 64-bit floats to 32-bit floats before sending out.
    ## fixme: why run triangle in 64 bit mode?
    cdef np.npy_intp* out_points_xy_dims = [ out_data.numberofpoints, 2 ]
    cdef np.ndarray[ float, ndim = 2 ] out_points_xy = \
        np.PyArray_SimpleNewFromData(
            2, out_points_xy_dims, np.NPY_DOUBLE, out_data.pointlist,
        ).astype( np.float32 )
    ( <PyArrayObject*>out_points_xy )[ 0 ].flags |= np.NPY_OWNDATA

    cdef np.npy_intp* out_points_z_dims = [ out_data.numberofpoints ]
    cdef np.ndarray[ float ] out_points_z = \
        np.PyArray_SimpleNewFromData(
            1, out_points_z_dims, np.NPY_DOUBLE, out_data.pointattributelist,
        ).astype( np.float32 )
    ( <PyArrayObject*>out_points_z )[ 0 ].flags |= np.NPY_OWNDATA

    cdef np.npy_intp* out_lines_dims = [
        out_data.numberofsegments, 2,
    ]
    cdef np.ndarray[ int, ndim = 2 ] out_lines = \
        np.PyArray_SimpleNewFromData(
            2, out_lines_dims, np.NPY_INT, out_data.segmentlist,
        )
    ( <PyArrayObject*>out_lines )[ 0 ].flags |= np.NPY_OWNDATA

    cdef np.npy_intp* out_triangles_dims = [
        out_data.numberoftriangles, out_data.numberofcorners
    ]
    cdef np.ndarray[ int, ndim = 2 ] out_triangles = \
        np.PyArray_SimpleNewFromData(
            2, out_triangles_dims, np.NPY_INT, out_data.trianglelist,
        )
    ( <PyArrayObject*>out_triangles )[ 0 ].flags |= np.NPY_OWNDATA

    # Send outgoing data to the parent process.
    connection.send( out_points_xy.shape[ 0 ] )
    connection.send_bytes( out_points_xy )
    connection.send_bytes( out_points_z )

    connection.send( out_lines.shape[ 0 ] )
    connection.send_bytes( out_lines )

    connection.send( out_triangles.shape[ 0 ] )
    connection.send_bytes( out_triangles )

    connection.close()
