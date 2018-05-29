import cython
import numpy as np
cimport numpy as np
cimport opengl as gl

ctypedef void ( __stdcall *glBindBuffer_pointer )( gl.GLenum, gl.GLuint ) nogil
cdef np.uint32_t RESET_COLOR = 4294967295UL # rgba = ( 1, 1, 1, 1 )

@cython.boundscheck( False )
def render_buffers_with_colors(
    np.ndarray[ np.uint32_t ] vertex_buffers not None,
    np.ndarray[ np.uint32_t ] colors not None,
    np.ndarray[ np.uint32_t ] vertex_counts not None,
    gl.GLenum primitive_type,
    pygl,
):
    """
    Given an array of vertex buffer handles, an array of colors, an array of
    vertex counts, and a GL primitive type, render all the vertex buffers with
    one color per buffer.
    """
    cdef np.int32_t buffer_index
    cdef glBindBuffer_pointer glBindBuffer

    glBindBuffer = <glBindBuffer_pointer>0
    if hasattr(pygl.platform.PLATFORM, "getExtensionProcedure"):
        getExtensionProcedure = pygl.platform.PLATFORM.getExtensionProcedure
        proc = getExtensionProcedure("glBindBuffer")
        if proc is not None:
            glBindBuffer = <glBindBuffer_pointer><size_t>proc
    if glBindBuffer == <glBindBuffer_pointer>0:
        glBindBuffer = <glBindBuffer_pointer><size_t>gl.glBindBufferARB

    gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated

    for buffer_index in range( len( vertex_buffers ) ):
        gl.glColor4ubv( <gl.GLubyte*>&colors[ buffer_index ] ) # FIXME: deprecated

        glBindBuffer(
            gl.GL_ARRAY_BUFFER,
            vertex_buffers[ buffer_index ],
        )
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, NULL ) # FIXME: deprecated

        gl.glDrawArrays(
            primitive_type,
            0,
            vertex_counts[ buffer_index ],
        )

    glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )

    gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated

    # Reset the color so it doesn't influence any other renderers.
    gl.glColor4ubv( <gl.GLubyte*>&RESET_COLOR ) # FIXME: deprecated

@cython.boundscheck( False )
def render_buffers_with_one_color(
    np.ndarray[ np.uint32_t ] vertex_buffers not None,
    np.uint32_t color,
    np.ndarray[ np.uint32_t ] vertex_counts not None,
    gl.GLenum primitive_type,
    pygl,
    np.uint32_t alternate_type_index,
    gl.GLenum alternate_primitive_type,
):
    """
    Given an array of vertex buffer handles, a single color, an array of vertex
    counts, and a GL primitive type, render all the vertex buffers with the
    same color.
    """
    cdef np.int32_t buffer_index
    cdef glBindBuffer_pointer glBindBuffer

    glBindBuffer = <glBindBuffer_pointer>0
    if hasattr(pygl.platform.PLATFORM, "getExtensionProcedure"):
        getExtensionProcedure = pygl.platform.PLATFORM.getExtensionProcedure
        proc = getExtensionProcedure("glBindBuffer")
        if proc is not None:
            glBindBuffer = <glBindBuffer_pointer><size_t>proc
    if glBindBuffer == <glBindBuffer_pointer>0:
        glBindBuffer = <glBindBuffer_pointer><size_t>gl.glBindBufferARB

    gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated

    gl.glColor4ubv( <gl.GLubyte*>&color ) # FIXME: deprecated

    for buffer_index in range( len( vertex_buffers ) ):
        glBindBuffer(
            gl.GL_ARRAY_BUFFER,
            vertex_buffers[ buffer_index ],
        )
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, NULL ) # FIXME: deprecated

        if buffer_index == alternate_type_index:
            gl.glDrawArrays(
                alternate_primitive_type,
                0,
                vertex_counts[ buffer_index ],
            )
        else:
            gl.glDrawArrays(
                primitive_type,
                0,
                vertex_counts[ buffer_index ],
            )

    glBindBuffer( gl.GL_ARRAY_BUFFER, 0 )

    gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated

    # Reset the color so it doesn't influence any other renderers.
    gl.glColor4ubv( <gl.GLubyte*>&RESET_COLOR ) # FIXME: deprecated
