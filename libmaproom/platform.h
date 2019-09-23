#if defined(_WIN32)
    #include "windows.h"
    #include "GL/gl.h"
    #include "GL/glu.h"
    // Microsoft hasn't updated GL headers since 1995, so these functions have
    // to be obtained as pointers from wglGetProcAddress()
    #define glGenBuffersARB 0
    #define glBindBufferARB 0
    #define glBufferDataARB 0

    #define GL_ARRAY_BUFFER 0x8892
    #define GL_ELEMENT_ARRAY_BUFFER 0x8893
    #define GL_STATIC_DRAW 0x88E4
    #define GL_DYNAMIC_DRAW 0x88E8
#elif defined(__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
#else
    #define GL_GLEXT_PROTOTYPES 1
    #include "GL/gl.h"
    #include "GL/glu.h"
    #include "GL/glext.h"
#endif
