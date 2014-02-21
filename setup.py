#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension

from glob import *
import os
import pyproj
import shutil
import subprocess
import sys

import maproom.Version as Version

# find the various headers, libs, etc.
import numpy
gl_include_dirs = [numpy.get_include()]
gl_library_dirs = []
gl_libraries = ["GL", "GLU"]

if sys.platform.startswith("win"):
    gl_libraries = ["opengl32", "glu32"]
elif sys.platform == "darwin":
    gl_include_dirs.append(
        "/System/Library/Frameworks/OpenGL.framework/Headers",
    )
    gl_library_dirs.append(
        "/System/Library/Frameworks/OpenGL.framework/Libraries",
    )

print gl_include_dirs
print gl_library_dirs
print gl_libraries

# Definintion of compiled extension code:
bitmap = Extension("maproom.library.Bitmap",
                   sources=["maproom/library/Bitmap.pyx"],
                   include_dirs=[numpy.get_include()],
                   )

shape = Extension("maproom.library.Shape",
                  sources=["maproom/library/Shape.pyx"],
                  include_dirs=[numpy.get_include()],
                  )

tree = Extension("maproom.library.scipy_ckdtree",
                 sources=["maproom/library/scipy_ckdtree.pyx"],
                 include_dirs=[numpy.get_include()],
                 )

tessellator = Extension("maproom.library.Opengl_renderer.Tessellator",
                        sources=["maproom/library/Opengl_renderer/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        )

render = Extension("maproom.library.Opengl_renderer.Render",
                   sources=["maproom/library/Opengl_renderer/Render.pyx"],
                   include_dirs=gl_include_dirs,
                   library_dirs=gl_library_dirs,
                   libraries=gl_libraries,
                   )

ext_modules = [bitmap, shape, tree, tessellator, render]
#ext_modules = [tessellator]

if sys.platform.startswith("win") and "py2exe" in sys.argv:
    import py2exe

    # Help py2exe find MSVCP90.DLL
    sys.path.append("c:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/redist/x86/Microsoft.VC90.CRT")


# Make pyproj's data into a resource instead of including it in the zip file
# so pyproj can actually find it. (Note: It's actually still included in the zip
# file.)
pyproj_data = glob(
    os.path.join(
        os.path.dirname(pyproj.__file__),
        "data", "*",
    )
)


full_version = Version.VERSION
spaceless_version = Version.VERSION.replace(" ", "_")

import peppy2
import maproom

data_files = []
data_files.extend(peppy2.get_py2exe_data_files())
data_files.extend(peppy2.get_py2exe_data_files(maproom))

import traitsui
data_files.extend(peppy2.get_py2exe_data_files(traitsui, excludes=["*/qt4/*"]))

import pyface
data_files.extend(peppy2.get_py2exe_data_files(pyface, excludes=["*/qt4/*", "*/pyface/images/*.jpg"]))

if sys.platform.startswith('win'):
    # with py2app, we just include the entire package and these files are
    # copied over
    data_files.extend([
        ("maproom/library/Opengl_renderer",
            glob("maproom/library/Opengl_renderer/*.pyd")
         ),
        ("maproom/library",
            glob("maproom/library/*.pyd")
         ),
    ])

    # Add missing DLL files that py2exe doesn't pull in automatically.
    # data_files.append(
    #    ( ".", [ "..\..\PROJ.4\workspace\src\proj.dll" ] ),
    #)


common_includes = [
    "ctypes",
    "ctypes.util",
    "wx.lib.pubsub.*",
    "wx.lib.pubsub.core.*",
    "wx.lib.pubsub.core.kwargs.*",
    "multiprocessing",
    
    "traits",
    
    "traitsui",
    "traitsui.editors",
    "traitsui.editors.*",
    "traitsui.extras",
    "traitsui.extras.*",
    "traitsui.wx",
    "traitsui.wx.*",
 
    "pyface",
    "pyface.*",
    "pyface.wx",
 
    "pyface.ui.wx",
    "pyface.ui.wx.init",
    "pyface.ui.wx.*",
    "pyface.ui.wx.grid.*",
    "pyface.ui.wx.action.*",
    "pyface.ui.wx.timer.*",
    "pyface.ui.wx.tasks.*",
    "pyface.ui.wx.workbench.*",
]
common_includes.extend(peppy2.get_py2exe_toolkit_includes())
common_includes.extend(peppy2.get_py2exe_toolkit_includes(maproom))
print common_includes

py2app_includes = [
    "OpenGL_accelerate",
    "OpenGL_accelerate.formathandler",
]

common_excludes = [
    "test",
#    "unittest", # needed for numpy
    "pydoc_data",
    "pyface.ui.qt4",
    "traitsui.qt4",
     "Tkconstants",
    "Tkinter", 
    "tcl", 
    "_imagingtk",
    "PIL._imagingtk",
    "ImageTk",
    "PIL.ImageTk",
    "FixTk",
    ]

py2exe_excludes = [
    "OpenGL",
    ]

base_dist_dir = "dist-%s" % spaceless_version
win_dist_dir = os.path.join(base_dist_dir, "win")
mac_dist_dir = os.path.join(base_dist_dir, "mac")

try:
    if sys.platform.startswith("win"):
        shutil.rmtree(win_dist_dir, ignore_errors=True)
    elif sys.platform.startswith('darwin'):
        shutil.rmtree(mac_dist_dir, ignore_errors=True)

    setup(
        name="Maproom",
        version=full_version,
        description="High-performance 2d mapping",
        author="NOAA",
        data_files=data_files,
        packages=find_packages(),
        app=["maproom.py"],
        entry_points = """

        [envisage.plugins]
        peppy2.tasks = maproom.plugin:MaproomPlugin

        """,
        windows=[dict(
            script="maproom.py",
            icon_resources=[(1, "maproom/ui/images/maproom.ico")],
        )],
        options=dict(
            py2app=dict(
                dist_dir=mac_dist_dir,
                argv_emulation=True,
                packages=['pyproj'],
                optimize=2,  # Equivalent to running "python -OO".
                semi_standalone=False,
                includes=common_includes + py2app_includes,
                excludes=common_excludes,
                iconfile="maproom/ui/images/maproom.icns",
                plist=dict(
                    CFBundleName="Maproom",
                    CFBundleTypeExtensions=["verdat", "kap", "bna"],
                    CFBundleTypeName="Geographic Document",
                    CFBundleTypeRole="Editor",
                    CFBundleShortVersionString=Version.VERSION,
                    CFBundleGetInfoString="Maproom %s" % Version.VERSION,
                    CFBundleExecutable="Maproom",
                    CFBUndleIdentifier="gov.noaa.maproom",
                )
            ),
            py2exe=dict(
                dist_dir=win_dist_dir,
                optimize=2,
                skip_archive=True,
                compressed=False,
                packages=['library'],
                # See http://www.py2exe.org/index.cgi/PyOpenGL
                # and http://www.py2exe.org/index.cgi/TkInter
                includes=common_includes,
                excludes=common_excludes + py2exe_excludes,
            ),
            build=dict(
                #compiler = "mingw32",
                compiler="msvc",
            ) if sys.platform.startswith("win") else {},
        )
    )

    if 'py2exe' in sys.argv and sys.platform.startswith("win"):
        try:
            import triangle
            import shutil
            print "*** copy Triangle module ***"
            triangle_path = triangle.__file__
            try:
                shutil.copy(triangle_path, win_dist_dir)
            except (OSError, WindowsError), error:
                if not "already exists" in str(error):
                    raise
        except ImportError:
            pass

        # See http://www.py2exe.org/index.cgi/PyOpenGL
        import OpenGL
        import OpenGL_accelerate
        import shutil
        print "*** copy PyOpenGL module ***"
        opengl_dir = os.path.dirname(OpenGL.__file__)
        opengl_accelerate_dir = os.path.dirname(OpenGL_accelerate.__file__)
        try:
            shutil.copytree(
                opengl_dir, os.path.join(win_dist_dir, "OpenGL"),
                ignore=shutil.ignore_patterns("GLUT", "Tk"),
            )
            shutil.copytree(
                opengl_accelerate_dir,
                os.path.join(win_dist_dir, "OpenGL_accelerate"),
            )
        except WindowsError, error:
            if not "already exists" in str(error):
                raise

        print "*** create installer ***"

        iss_filename = "%s\\maproom.iss" % win_dist_dir
        iss_file = open(iss_filename, "w")
        iss_file.write( """
[Setup]
AppId={{8AE5A4C3-B67E-4243-9F45-401C554A9019}
AppName=Maproom
AppVerName=Maproom %s
AppPublisher=NOAA
AppPublisherURL=http://www.noaa.gov/
AppSupportURL=http://www.noaa.gov/
AppUpdatesURL=http://www.noaa.gov/
DefaultDirName={pf}\Maproom
DefaultGroupName=Maproom
OutputBaseFilename=Maproom_%s
SetupIconFile=..\..\ui\images\maproom.ico
Compression=lzma
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "maproom.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\Maproom"; Filename: "{app}\maproom.exe"
Name: "{commondesktop}\Maproom"; Filename: "{app}\maproom.exe"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Maproom"; Filename: "{app}\maproom.exe"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\maproom.exe"; Description: "{cm:LaunchProgram,Maproom}"; Flags: nowait postinstall skipifsilent
""" % ( full_version, spaceless_version ) )
        iss_file.close()

        os.system(
            '"C:\Program Files (x86)\Inno Setup 5\ISCC.exe" %s' % iss_filename,
        )
    elif 'py2app' in sys.argv and sys.platform.startswith('darwin'):
        app_name = "%s/Maproom.app" % mac_dist_dir
        fat_app_name = "%s/Maproom.fat.app" % mac_dist_dir
        os.rename(app_name, fat_app_name)
        subprocess.call(['/usr/bin/ditto', '-arch', 'x86_64', fat_app_name, app_name])
        cwd = os.getcwd()
        os.chdir(mac_dist_dir)
        subprocess.call(['/usr/bin/zip', '-r', '-9', "Maproom-%s-darwin.zip" % spaceless_version, 'Maproom.app', ])
        os.chdir(cwd)
        shutil.rmtree(fat_app_name)
finally:
    pass
