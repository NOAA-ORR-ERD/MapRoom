import cython
import numpy as np
cimport numpy as np
cimport opengl as gl
cimport libc.stdlib as stdlib
cimport libc.stdio as stdio

## this should have come from opengl.pxd
ctypedef   void ( __stdcall *_GLUfuncptr)()   nogil
    
ctypedef struct Point:
    np.float64_t x
    np.float64_t y

ctypedef struct Double_point:
    np.double_t x
    np.double_t y
    np.double_t z

ctypedef struct State:
    np.uint32_t current_type           # Type of primitive (strip/fan/tris).

    Point* current_points              # Points for current primitive.
    np.uint32_t current_point_count    # Number of points in this primitive.
    np.uint32_t current_point_capacity # Capacity of current_points array.

    Point* points                      # Points for current polygon.
    np.uint32_t point_count            # Number of points in this polygon.
    np.uint32_t point_capacity         # Capacity of points array.

    Point* scratch_points              # Points used by combine_data callback
    np.uint32_t scratch_point_count
    np.uint32_t scratch_point_capacity

ctypedef void ( __stdcall *glGenBuffers_pointer )( gl.GLsizei, gl.GLuint* ) nogil
ctypedef void ( __stdcall *glBindBuffer_pointer )( gl.GLenum, gl.GLuint ) nogil
ctypedef void ( __stdcall *glBufferData_pointer )( gl.GLenum, gl.GLsizei, gl.GLvoid*, gl.GLenum ) nogil

@cython.boundscheck( False )
def init_vertex_buffers(
    np.ndarray[ np.uint32_t ] triangle_vertex_buffers not None,
    np.ndarray[ np.uint32_t ] line_vertex_buffers not None,
    np.uint32_t start_index,
    np.uint32_t count,
    pygl,
):
    """
    Initialize the given vertex buffers by calling glGenBuffers().

    :param triangle_vertex_buffers: VBO handles for polygon triangle points
                                    (out parameter)
    :type triangle_vertex_buffers: scalar numpy array of type uint32 elements
    :param line_vertex_buffers: VBO handles for polygon line points
                                (out parameter)
    :type line_vertex_buffers: scalar numpy array of type uint32 elements
    :param start_index: buffer index where initialization should be started
    :type start_index: uint32
    :param count: number of elements to initialize
    :type count: uint32
    :param pygl: imported and initialized OpenGL module (from PyOpenGL)
    :type pygl: module
    """
    cdef glGenBuffers_pointer glGenBuffers

    if hasattr( pygl.platform.PLATFORM, "getExtensionProcedure" ):
        getExtensionProcedure = pygl.platform.PLATFORM.getExtensionProcedure
        glGenBuffers = \
            <glGenBuffers_pointer><size_t>getExtensionProcedure( "glGenBuffers" )
    else:
        glGenBuffers = <glGenBuffers_pointer><size_t>gl.glGenBuffersARB

    glGenBuffers(
        count,
        &( <gl.GLuint*>triangle_vertex_buffers.data )[ start_index ],
    )

    glGenBuffers(
        count,
        &( <gl.GLuint*>line_vertex_buffers.data )[ start_index ],
    )

@cython.boundscheck( False )
def tessellate(
    np.ndarray[ np.float64_t, ndim = 2 ] points not None,
    np.ndarray[ np.uint32_t ] point_adjacency not None,
    np.ndarray[ np.uint32_t ] point_polygons not None,
    np.ndarray[ np.uint32_t ] polygon_starts not None,
    np.ndarray[ np.uint32_t ] polygon_vertex_counts not None,
    np.ndarray[ np.uint32_t ] polygon_nan_counts not None,
    np.ndarray[ np.uint32_t ] polygon_groups not None,
    np.ndarray[ np.uint32_t ] triangle_vertex_buffers not None,
    np.ndarray[ np.uint32_t ] triangle_vertex_counts not None,
    np.ndarray[ np.uint32_t ] line_vertex_buffers not None,
    pygl,
):
    """
    A wrapper for GLU's built-in polygon tessellation functionality.

    Given a series of convex or concave polygons, return triangles
    representing their filled interior.

    :param points: geographic points comprising the polygons to tessellate
    :type points: Nx2 numpy array of type float32, where each row is the 2D
                  geographic coordinate of a point
    :param point_adjacency: adjacency list for polygon points: point index ->
                            index of next point in polygon
    :type point_adjacency: scalar numpy array of type uint32 elements
    :param point_polygons: indices of the polygon for each point: point index
                           -> index of polygon containing it
    :type point_polygons: scalar numpy array of type uint32 elements
    :param polygon_starts: indices of the start of each polygon
    :type polygon_starts: scalar numpy array of type uint32 elements
    :param polygon_vertex_counts: point count for each polygon
    :type polygon_vertex_counts: scalar numpy array of type uint32 elements
    :param polygon_nan_counts: count of NaN/deleted points for each polygon
                               (out parameter)
    :type polygon_nan_counts: scalar numpy array of type uint32 elements
    :param polygon_groups: group ids of each polygon
    :type polygon_groups: scalar numpy array of type uint32 elements
    :param triangle_vertex_buffers: VBO handles for polygon triangle points
                                    (out parameter)
    :type triangle_vertex_buffers: scalar numpy array of type uint32 elements
    :param triangle_vertex_counts: per-polygon counts for polygon triangle points 
                                   (out parameter)
    :type triangle_vertex_counts: scalar numpy array of type uint32 elements
    :param line_vertex_buffers: VBO handles for polygon line points
                                (out parameter)
    :type line_vertex_buffers: scalar numpy array of type uint32 elements
    :param pygl: imported and initialized OpenGL module (from PyOpenGL)
    :type pygl: module
    """
    cdef np.uint32_t loop_index, polygon_index, point_index, group_polygon_index
    cdef np.uint32_t polygon_point_index, start_point_index, polygon_nan_count
    cdef np.uint32_t group
    cdef np.uint32_t polygon_starts_count = polygon_starts.shape[ 0 ]

    cdef Double_point* double_points
    cdef Point* line_points
    cdef Point* raw_points = <Point*>points.data

    cdef gl.GLUtesselator* tess = gl.gluNewTess()
    cdef State state

    cdef glBindBuffer_pointer glBindBuffer
    cdef glBufferData_pointer glBufferData

    if hasattr( pygl.platform.PLATFORM, "getExtensionProcedure" ):
        getExtensionProcedure = pygl.platform.PLATFORM.getExtensionProcedure
        glBindBuffer = \
            <glBindBuffer_pointer><size_t>getExtensionProcedure( "glBindBuffer" )
        glBufferData = \
            <glBufferData_pointer><size_t>getExtensionProcedure( "glBufferData" )
    else:
        glBindBuffer = <glBindBuffer_pointer><size_t>gl.glBindBufferARB
        glBufferData = <glBufferData_pointer><size_t>gl.glBufferDataARB

    state.current_type = 0
    state.current_points = <Point*>stdlib.malloc( sizeof( Point ) * 256 )
    state.current_point_count = 0
    state.current_point_capacity = 256
    
    # small number of scratch points may be necessary if the tessellator needs
    # to create new points to handle an intersection
    state.scratch_points = <Point*>stdlib.malloc( sizeof( Point ) * 256 )
    state.scratch_point_count = 0
    state.scratch_point_capacity = 256

    gl.gluTessCallback( tess, gl.GLU_TESS_BEGIN_DATA, <_GLUfuncptr>begin )
    gl.gluTessCallback( tess, gl.GLU_TESS_END_DATA, <_GLUfuncptr>end )
    gl.gluTessCallback( tess, gl.GLU_TESS_VERTEX_DATA, <_GLUfuncptr>vertex )
    gl.gluTessCallback( tess, gl.GLU_TESS_COMBINE_DATA, <_GLUfuncptr>combine )
    gl.gluTessCallback( tess, gl.GLU_TESS_ERROR, <_GLUfuncptr>error )
    gl.gluTessProperty(
        tess, gl.GLU_TESS_WINDING_RULE, gl.GLU_TESS_WINDING_ODD,
    )

    # Indicate that all points are in the same x-y plane, thereby preventing
    # GLU from doing its own calculation of normals.
    gl.gluTessNormal( tess, 0.0, 0.0, 1.0 )

    for loop_index in range( polygon_starts_count ):
        state.points = <Point*>stdlib.malloc( sizeof( Point ) * 2048 )
        state.point_count = 0
        state.point_capacity = 2048
        
        # reset counter for scratch points since they aren't reused between
        # polygons
        state.scratch_point_count = 0

        start_point_index = polygon_starts[ loop_index ]
        polygon_index = point_polygons[ start_point_index ]
        point_index = start_point_index
        polygon_point_index = 0
        polygon_nan_count = 0

        # If a new group is being started, then end the current polygon and
        # start a new one. This makes all holes of a given outer boundary
        # render as separate contours within the same GLU polygon.
        if loop_index == 0 or polygon_groups[ loop_index ] != group:
            gl.gluTessBeginPolygon( tess, &state )
            group = polygon_groups[ loop_index ]
            group_polygon_index = polygon_index

        triangle_vertex_counts[ polygon_index ] = 0

        # GLU wants 3 doubles (x, y, z) per point, even though we've got 2 floats
        # (x, y) for each point.
        double_points = <Double_point*>stdlib.malloc(
            polygon_vertex_counts[ loop_index ] * sizeof( Double_point ),
        )

        line_points = <Point*>stdlib.malloc(
            polygon_vertex_counts[ loop_index ] * sizeof( Point ),
        )

        gl.gluTessBeginContour( tess )

        while True:
            # Note that when there's a NaN, polygon_point_index is not
            # incremented.
            if raw_points[ point_index ].x != raw_points[ point_index ].x:
                line_points[ polygon_point_index ].x = \
                    raw_points[ point_index ].x
                line_points[ polygon_point_index ].y = \
                    raw_points[ point_index ].x
                polygon_nan_count += 1
            else:
                double_points[ polygon_point_index ].x = \
                    <double>raw_points[ point_index ].x
                double_points[ polygon_point_index ].y = \
                    <double>raw_points[ point_index ].y
                double_points[ polygon_point_index ].z = <double>0

                line_points[ polygon_point_index ].x = \
                    raw_points[ point_index ].x
                line_points[ polygon_point_index ].y = \
                    raw_points[ point_index ].y

                gl.gluTessVertex(
                    tess,
                    <gl.GLdouble*>&double_points[ polygon_point_index ],
                    <gl.GLvoid*>&raw_points[ point_index ],
                )
                polygon_point_index += 1
#                print "Tessellator.pyx: loop #%d, point_index=%d capacity=%d count=%d" % (loop_index, polygon_point_index, state.point_capacity, state.point_count)

            point_index = point_adjacency[ point_index ]
            if point_index == start_point_index:
                break

        gl.gluTessEndContour( tess )
        if loop_index == polygon_starts_count - 1 or \
           polygon_groups[ loop_index + 1 ] != group:
            gl.gluTessEndPolygon( tess )
#            print "Tessellator.pyx: closed polygon #%d" % (loop_index)

        stdlib.free( double_points )

        polygon_nan_counts[ polygon_index ] = polygon_nan_count
        triangle_vertex_counts[ group_polygon_index ] += state.point_count

        # Make vertex and color buffers from the triangle point data.
#        print "Tessellator.pyx: group_index=%d capacity=%d count=%d" % (group_polygon_index, state.point_capacity, state.point_count)
        if state.point_count > 0:
            glBindBuffer(
                gl.GL_ARRAY_BUFFER,
                triangle_vertex_buffers[ group_polygon_index ],
            )
            glBufferData(
                gl.GL_ARRAY_BUFFER,
                state.point_count * sizeof( Point ),
                state.points,
                gl.GL_STATIC_DRAW,
            )
 
        # Make vertex buffers for boundary lines.
        glBindBuffer(
            gl.GL_ARRAY_BUFFER,
            line_vertex_buffers[ polygon_index ],
        )
        glBufferData(
            gl.GL_ARRAY_BUFFER,
            polygon_vertex_counts[ loop_index ] * sizeof( Point ),
            line_points,
            gl.GL_STATIC_DRAW,
        )
 
        stdlib.free( state.points )
        stdlib.free( line_points )

    glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )
    glBindBuffer( gl.GL_ELEMENT_ARRAY_BUFFER, 0 )

    stdlib.free( state.current_points )
    stdlib.free( state.scratch_points )

    gl.gluDeleteTess( tess )

cdef void __stdcall begin( np.uint32_t data_type, State* state ) nogil:
    state[ 0 ].current_type = data_type

cdef void __stdcall end( State* state ) nogil:
    cdef Point* first
    cdef Point* last
    cdef Point* next
    cdef np.uint32_t index
    cdef bint grow = False

    # Convert triangle fans and strips to just triangles. Ignore points for
    # unsupported geometry types.
    if state.current_type == gl.GL_TRIANGLE_FAN:
        first = &state[ 0 ].current_points[ 0 ]
        last = &state[ 0 ].current_points[ 1 ]

        # For each point in state.current_points (except for the first 2),
        # we're going to append 3 points onto state.points.
        while state[ 0 ].point_count + \
              ( state[ 0 ].current_point_count - 2 ) * 3 >= \
              state[ 0 ].point_capacity:
            state[ 0 ].point_capacity *= 2
            grow = True

        if grow:
            state[ 0 ].points = <Point*>stdlib.realloc(
                <void*>state[ 0 ].points,
                sizeof( Point ) * state[ 0 ].point_capacity,
            )

        # 2nd point to the last point
        for index in range( 2, state[ 0 ].current_point_count ):
            next = &state[ 0 ].current_points[ index ]
            state[ 0 ].points[ state.point_count ] = first[ 0 ]
            state[ 0 ].points[ state.point_count + 1 ] = last[ 0 ]
            state[ 0 ].points[ state.point_count + 2 ] = next[ 0 ]
            state[ 0 ].point_count += 3
            last = next

    elif state[ 0 ].current_type == gl.GL_TRIANGLE_STRIP:
        # For each point in state.current_points (except for the last 2),
        # we're going to append 3 points onto state.points.
        while state[ 0 ].point_count + \
              ( state[ 0 ].current_point_count - 2 ) * 3 >= \
              state[ 0 ].point_capacity:
            state[ 0 ].point_capacity *= 2
            grow = True

        if grow:
            state[ 0 ].points = <Point*>stdlib.realloc(
                <void*>state[ 0 ].points,
                sizeof( Point ) * state[ 0 ].point_capacity,
            )

        for index in range( state[ 0 ].current_point_count - 2 ):
            if index % 2:
                state[ 0 ].points[ state[ 0 ].point_count ] = state[ 0 ].current_points[ index ]
                state[ 0 ].points[ state[ 0 ].point_count + 1 ] = state[ 0 ].current_points[ index + 1 ]
                state[ 0 ].points[ state[ 0 ].point_count + 2 ] = state[ 0 ].current_points[ index + 2 ]
            else:
                state[ 0 ].points[ state[ 0 ].point_count ] = state[ 0 ].current_points[ index + 1 ]
                state[ 0 ].points[ state[ 0 ].point_count + 1 ] = state[ 0 ].current_points[ index ]
                state[ 0 ].points[ state[ 0 ].point_count + 2 ] = state[ 0 ].current_points[ index + 2 ]
            state[ 0 ].point_count += 3

    elif state[ 0 ].current_type == gl.GL_TRIANGLES:
        # We're going to append all the points in state.current_points onto
        # state.points.
        while state[ 0 ].point_count + state[ 0 ].current_point_count >= \
              state[ 0 ].point_capacity:
            state[ 0 ].point_capacity *= 2
            grow = True

        if grow:
            state[ 0 ].points = <Point*>stdlib.realloc(
                <void*>state[ 0 ].points,
                sizeof( Point ) * state[ 0 ].point_capacity,
            )

        for index in range( state[ 0 ].current_point_count ):
            state[ 0 ].points[ state[ 0 ].point_count + index ] = state[ 0 ].current_points[ index ]

        state[ 0 ].point_count += state[ 0 ].current_point_count

    state[ 0 ].current_type = 0
    state[ 0 ].current_point_count = 0

cdef void __stdcall vertex( np.float64_t* point, State* state ) nogil:
    if state[ 0 ].current_point_count == state[ 0 ].current_point_capacity:
        state[ 0 ].current_point_capacity *= 2
        state[ 0 ].current_points = <Point*>stdlib.realloc(
            <void*>state[ 0 ].current_points,
            sizeof( Point ) * state[ 0 ].current_point_capacity,
        )

#    stdio.printf("Tessellator.pyx: VERTEX CALLBACK! storing x=%f, y=%f\n", point[0], point[1])
    state[ 0 ].current_points[ state.current_point_count ] = ( <Point*>point )[ 0 ]
    state[ 0 ].current_point_count += 1

cdef void __stdcall combine( Double_point* position, Point* points,
                             gl.GLfloat* weights, void** outData, State* state ) nogil:
    cdef Point* new_pt

    # Need to create a point that is not on the stack because it may be
    # referenced much later in the tessellation process.  So, have to add it
    # to the scratch array and handle cleanup later.
    if state[0].scratch_point_count == state[0].scratch_point_capacity:
        state[0].scratch_point_capacity *= 2
        state[0].scratch_points = <Point*>stdlib.realloc(
            <void*>state[0].scratch_points,
            sizeof(Point) * state[0].scratch_point_capacity,
        )

    new_pt = &state[0].scratch_points[state.scratch_point_count]
    new_pt.x = position[0].x
    new_pt.y = position[0].y 
#    stdio.printf("Tessellator.pyx: COMBINE CALLBACK! x=%f, y=%f\n", new_pt.x, new_pt.y)
    state[0].scratch_point_count += 1
    outData[0] = <void *>new_pt

cdef void __stdcall error(gl.GLenum errno) nogil:
    stdio.printf("Tessellator.pyx: Tessellation error #%d\n", errno)
