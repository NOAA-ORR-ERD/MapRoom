#!/usr/bin/env python

"""
setup.py for MapRoom -- mostly to build the Cython extensions

right now - it mostly is to be used as follows:

python setup.py build_ext --inplace

"""


from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension

from glob import *
import os
import shutil
import subprocess
import sys

is_64bit = sys.maxsize > 2**32

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

tessellator = Extension("maproom.renderer.gl.Tessellator",
                        sources=["maproom/renderer/gl/Tessellator.pyx"],
                        include_dirs=gl_include_dirs,
                        library_dirs=gl_library_dirs,
                        libraries=gl_libraries,
                        )

render = Extension("maproom.renderer.gl.Render",
                   sources=["maproom/renderer/gl/Render.pyx"],
                   include_dirs=gl_include_dirs,
                   library_dirs=gl_library_dirs,
                   libraries=gl_libraries,
                   )

ext_modules = [bitmap, shape, tree, tessellator, render]
#ext_modules = [tessellator]

full_version = Version.VERSION
spaceless_version = Version.VERSION.replace(" ", "_")

import maproom

BUILD_APP = False
APP_TARGET = None

if sys.platform.startswith("win") and "py2exe" in sys.argv:
    BUILD_APP = True
    APP_TARGET = "win"
    import py2exe
    if is_64bit:
        # Help py2exe find MSVCP90.DLL
        sys.path.append("c:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/redist/amd64/Microsoft.VC90.CRT")
    else:
        # Help py2exe find MSVCP90.DLL
        sys.path.append("c:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/redist/x86/Microsoft.VC90.CRT")

elif sys.platform.startswith("darwin") and "py2app" in sys.argv:
    BUILD_APP = True
    APP_TARGET = "mac"


data_files = []
options = {}

base_dist_dir = "dist-%s" % spaceless_version
win_dist_dir = os.path.join(base_dist_dir, "win")
mac_dist_dir = os.path.join(base_dist_dir, "mac")

if BUILD_APP:
    import omnivore
    includes = []
    excludes = []
    
    data_files.extend(omnivore.get_py2exe_data_files())
    data_files.extend(omnivore.get_py2exe_data_files(maproom))

    import traitsui
    data_files.extend(omnivore.get_py2exe_data_files(traitsui, excludes=["*/qt4/*"]))

    import pyface
    data_files.extend(omnivore.get_py2exe_data_files(pyface, excludes=["*/qt4/*", "*/pyface/images/*.jpg"]))

    data_files.extend([
        ("maproom/templates",
            glob("maproom/templates/*.bna")
            ),
        ])
    
    includes = [
        "ctypes",
        "ctypes.util",
        "wx.lib.pubsub.*",
        "wx.lib.pubsub.core.*",
        "wx.lib.pubsub.core.kwargs.*",
        "multiprocessing",
        "pkg_resources",
        "configobj",
        
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
        
        "netCDF4",
        "netCDF4_utils",
        "netcdftime",

        "markdown"
    ]

    excludes = [
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

    if APP_TARGET == "win":
        shutil.rmtree(win_dist_dir, ignore_errors=True)

        # Make pyproj's data into a resource instead of including it in the zip file
        # so pyproj can actually find it. (Note: It's actually still included in the zip
        # file.)
        import pyproj
        pyproj_data = glob(
            os.path.join(
                os.path.dirname(pyproj.__file__),
                "data", "*",
            )
        )

        # with py2app, we just include the entire package and these files are
        # copied over
        import shapely
        libgeos = os.path.join(os.path.dirname(shapely.__file__), "geos_c.dll")
        import requests.certs
        data_files.extend([
            ("maproom/renderer/gl",
                glob("maproom/renderer/gl/*.pyd")
             ),
            ("maproom/library",
                glob("maproom/library/*.pyd")
             ),
            ("pyproj/data", pyproj_data),
            ("shapely", [libgeos]),
            ("requests",[requests.certs.where()]),
        ])

        # Add missing DLL files that py2exe doesn't pull in automatically.
        # data_files.append(
        #    ( ".", [ "..\..\PROJ.4\workspace\src\proj.dll" ] ),
        #)

        excludes.extend([
            "OpenGL",
            ])

        options = dict(
            py2exe=dict(
                dist_dir=win_dist_dir,
                optimize=2,
                skip_archive=True,
                compressed=False,
                packages=['maproom.renderer', 'maproom.library'],
                # See http://www.py2exe.org/index.cgi/PyOpenGL
                # and http://www.py2exe.org/index.cgi/TkInter
                includes=includes,
                excludes=excludes,
            ),
            build=dict(
                compiler="msvc",
            ),
            )

    elif APP_TARGET == "mac":
        shutil.rmtree(mac_dist_dir, ignore_errors=True)
        
        includes.extend([
            "OpenGL_accelerate",
            "OpenGL_accelerate.formathandler",
            ])

        options = dict(
            py2app=dict(
                dist_dir=mac_dist_dir,
                argv_emulation=True,
                packages=['pyproj'],
                optimize=2,  # Equivalent to running "python -OO".
                semi_standalone=False,
                includes=includes,
                excludes=excludes,
                frameworks=['libgeos_c.dylib'],
                iconfile="maproom/icons/maproom.icns",
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
            )


def remove_pyc(basedir):
    for curdir, dirlist, filelist in os.walk(basedir):
        print curdir
        for name in filelist:
            if name.endswith(".pyo"):
                c = name[:-1] + "c"
                cpath = os.path.join(curdir, c)
                print "  " + name
                # remove .pyc if .pyo exists
                if os.path.exists(cpath):
                    os.remove(cpath)
                # also remove .py if .pyo exists
                path = cpath[:-1]
                if os.path.exists(path) and "numpy" not in path:
                    os.remove(path)


def remove_numpy_tests(basedir):
    print basedir
    for f in glob("%s/*/tests" % basedir):
        print f
        shutil.rmtree(f)
    for f in glob("%s/tests" % basedir):
        print f
        shutil.rmtree(f)
    for f in ["tests", "f2py", "testing", "core/include", "core/lib", "distutils"]:
        path = os.path.join(basedir, f)
        shutil.rmtree(path, ignore_errors=True)
    testing = "%s/testing" % basedir
    os.mkdir(testing)
    
    tester_replace = """class Tester(object):
    def bench(self, label='fast', verbose=1, extra_argv=None):
        pass
    test = bench
"""
    fh = open("%s/__init__.py" % testing, "wb")
    fh.write(tester_replace)
    fh.close()

try:
    setup(
        name="Maproom",
        version=full_version,
        description="High-performance 2d mapping",
        author="NOAA",
        install_requires = [
            'numpy',
            'pyopengl',
            'pyopengl_accelerate',
            'pyproj',
            'cython',
            'shapely',
            'pytest',
            'coverage',
            'pytest-cov',
            'docutils',
            'markdown',
            'reportlab',
            'docutils',
            'pyparsing',
            'requests',
            'python-dateutil',
            ],
        data_files=data_files,
        packages=find_packages(),
        app=["maproom.py"],
        entry_points = """

        [envisage.plugins]
        omnivore.tasks = maproom.plugin:MaproomPlugin

        """,
        windows=[dict(
            script="maproom.py",
            icon_resources=[(1, "maproom/icons/maproom.ico")],
        )],
        options=options,
    )

    if APP_TARGET == "win":
        remove_pyc(win_dist_dir)
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

        if is_64bit:
            nsis_arch = """ArchitecturesAllowed=x64
    ArchitecturesInstallIn64BitMode=x64"""
            
            # copy manifest and app config files to work around side-by-side
            # errors
            for f in glob(r'pyinstaller/Microsoft.VC90.CRT-9.0.30729.6161/*'):
                print f
                shutil.copy(f, win_dist_dir)
        else:
            nsis_arch = ""
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
SetupIconFile=..\..\maproom\icons\maproom.ico
Compression=lzma
SolidCompression=yes
%s

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
""" % ( full_version, spaceless_version, nsis_arch ) )
        iss_file.close()

        os.system(
            '"C:\Program Files (x86)\Inno Setup 5\ISCC.exe" %s' % iss_filename,
        )
    elif APP_TARGET == "mac":
        remove_pyc(mac_dist_dir)
        app_name = "%s/Maproom.app" % mac_dist_dir
        
        # Strip out useless binary stuff from the site packages zip file.
        # Saves 3MB or so
        site_packages = "%s/Contents/Resources/lib/python2.7/site-packages.zip" % app_name
        subprocess.call(['/usr/bin/zip', '-d', site_packages, "distutils/command/*", "wx/locale/*", "*.c", "*.pyx", "*.png", "*.jpg", "*.ico", "*.xcf", "*.icns", "reportlab/fonts/*"])

        # fixup numpy
        numpy_dir = "%s/Contents/Resources/lib/python2.7/numpy" % app_name
        remove_numpy_tests(numpy_dir)
        
        fat_app_name = "%s/Maproom.fat.app" % mac_dist_dir
        os.rename(app_name, fat_app_name)
        subprocess.call(['/usr/bin/ditto', '-arch', 'x86_64', fat_app_name, app_name])
        cwd = os.getcwd()
        os.chdir(mac_dist_dir)
        subprocess.call(['/usr/bin/zip', '-r', '-9', '-q', "Maproom-%s-darwin.zip" % spaceless_version, 'Maproom.app', ])
        os.chdir(cwd)
        shutil.rmtree(fat_app_name)
finally:
    pass
