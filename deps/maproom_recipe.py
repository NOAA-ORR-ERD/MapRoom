{
    'settings' : {
        'darwin' : {
            'archs': ['i386'],
            'min-version': '10.5',
            'env_vars': {
                'CC': 'gcc-4.0',
                'CXX': 'g++-4.0',
            },
        }
        
    },
    'packages' : [
    {
        'name': 'proj',
        'version': '4.7.0',
        'source': 'http://download.osgeo.org/proj/proj-4.7.0.tar.gz',
        'darwin': {
            'ignore': True,
        },
    },
    {
        'name': 'libiconv',
        'version': '1.14',
        'source': 'http://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.14.tar.gz',
        'win32': {
            'build_args': ['NO_NLS=1',],
            'project_file': 'Makefile.msvc', 
        },
        'darwin': {
            'ignore': True,
        },
    },
    {
        'name': 'gfortran', # needed for scipy
        'version': '4.2.3',
        'darwin': {
            'dmg': 'http://r.research.att.com/gfortran-4.2.3.dmg',
            'installer': 'gfortran.pkg',
            'installer_requires_admin': True,
        },
        'win32': {
            'ignore': True,
        },
    },
    {
        'name': 'ogdi',
        'version': '3.2.0.beta2',
        'format': 'gnumake',
        'project_file': 'makefile',
        'source': 'http://sourceforge.net/projects/ogdi/files/ogdi/3.2.0beta2/ogdi-3.2.0.beta2.tar.gz/download',
        'darwin': {
            'ignore': True,
        },
        #'configure_args' : ["--with-proj='%(SRCDIR)s/..'", '--with-expat'],
        'win32': {
            'prebuild_script': """
import os
import shutil
libiconv_dir = '../../libiconv/workspace'
proj_dir = '../../PROJ.4/workspace'

if not os.path.exists(libiconv_dir):
    shutil.copytree('../libiconv-1.11', libiconv_dir)

if not os.path.exists(proj_dir):
    shutil.copytree('../proj-4.7.0', proj_dir)
            """,
            'postinstall_cmds' : [
                'cp bin/win32/*.dll ../dlls',
            ],
            'env_vars': {
                'TOPDIR' : '%(SRCDIR)s',
                'TARGET': 'win32',
                'MSVCDIR': '/cygdrive/c/Program\\ Files/Microsoft\\ Visual\\ Studio\\ 9.0/VC',
                'MSVCDIR_NATIVE' : '"c:\Program Files\Microsoft Visual Studio 9.0\VC"',
                'SDKDIR_NATIVE' : '"c:\Program Files\Microsoft SDKs\Windows\v6.0A"',
                'INCLUDE' : '%INCLUDE%;%(SRCDIR)s/../libiconv-1.11/include',
                'LIB' : '%LIB%;%(SRCDIR)s/../libiconv-1.11/lib',
                'PATH': os.path.dirname(sys.executable) + ';%PATH%;C:\\cygwin\\bin',
            },
        },
    },
    {
        'name': 'greenlet',
        'source': 'http://pypi.python.org/packages/source/g/greenlet/greenlet-0.3.4.zip#md5=530a69acebbb0d66eb5abd83523d8272',
        'version': '0.3.4',
        'build_type': 'python',
    },
    {
        'name': 'gdal',
        'version': '1.9.0',
        'configure_args': ['--with-unix-stdio-64=no'],
        'source': 'http://download.osgeo.org/gdal/gdal-1.9.0.tar.gz',
        'postinstall_cmds' : [
                'cd swig/python',
                sys.executable + ' setup.py install',
                'cd ../..',
        ],
        'win32': {
            'env_vars': {
                'INCLUDE': '%INCLUDE%;%(SRCDIR)s/../ogdi-3.2.0.beta2/ogdi/include;%(SRCDIR)s/../ogdi-3.2.0.beta2/proj;%(SRCDIR)s/../ogdi-3.2.0.beta2/include/win32',
            },
            'postinstall_cmds': ['copy gdal16.dll ..\\dlls\\gdal16.dll'], 
            'prebuild_script': """
import os
import shutil
ogdi_dir = '../../OGDI/workspace'

if not os.path.exists(ogdi_dir):
    shutil.copytree('../ogdi-3.2.0.beta2', ogdi_dir)
            """,
        },
    },
    {
        'name': 'Cython',
        'source': 'http://cython.org/release/Cython-0.15.1.tar.gz',
        'version': '0.15.1',
        'build_type': 'python',
    },
    {
        'name': 'numpy',
        'source': 'http://sourceforge.net/projects/numpy/files/NumPy/1.6.1/numpy-1.6.1.tar.gz/download',
        'version': '1.6.1',
        'build_type': 'python',
    },
    {
        'name': 'scipy',
        'source': 'http://sourceforge.net/projects/scipy/files/scipy/0.10.1/scipy-0.10.1.tar.gz/download',
        'version': '0.10.1',
        'build_type': 'python',
    },
    {
        'name': 'PyOpenGL',
        'source': 'http://pypi.python.org/packages/source/P/PyOpenGL/PyOpenGL-3.0.1.tar.gz#md5=cdf03284f24279b8d9914bb680a37b5e',
        'version': '3.0.1',
        'build_type': 'python',
    },
    {
        'name': 'PyOpenGL-accelerate',
        'source': 'http://pypi.python.org/packages/source/P/PyOpenGL-accelerate/PyOpenGL-accelerate-3.0.1.tar.gz#md5=4014cd203dd5f52109a76edc4c14a480',
        'version': '3.0.1',
        'build_type': 'python',
    },
    {
        'name': 'pyproj',
        'source': 'http://pyproj.googlecode.com/files/pyproj-1.9.0.tar.gz',
        'version': '1.9.0',
        'build_type': 'python',
    },
    {
        'name': 'pytriangle',
        'version': '2.6.1',
        'build_type': 'python',
    },
    {
        'name': 'pygarmin',
        'source': 'https://github.com/quentinsf/pygarmin.git',
        'version': '0.7',
        'build_type': 'python',
    },
    ],
}
