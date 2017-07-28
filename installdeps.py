#!/usr/bin/env python
import subprocess
import os
import sys

using_conda = "Continuum Analytics" in sys.version or "conda" in sys.version
if sys.platform.startswith("win"):
    needs_netcdf = False
    develop_instead_of_link = True
else:
    needs_netcdf = True
    develop_instead_of_link = False

deps = [
    [".", {'builddir': 'pytriangle-1.6.1', 'command': 'python setup.py install'}],
    ['https://github.com/NOAA-ORR-ERD/GnomeTools.git', {'builddir': 'post_gnome'}],
    ['https://github.com/robmcmullen/OWSLib.git',],
    ['https://github.com/fathat/glsvg.git',],
    ['https://github.com/robmcmullen/pyugrid.git',],
    ['https://github.com/robmcmullen/traits.git',],
    ['https://github.com/robmcmullen/pyface.git', {'branch':'omnivore'}],
    ['https://github.com/robmcmullen/traitsui.git',],
    ['https://github.com/enthought/apptools.git',],
    ['https://github.com/robmcmullen/envisage.git',],
    ['https://github.com/robmcmullen/pyfilesystem.git',],
    ['https://github.com/robmcmullen/omnivore.git', {'builddir': '.', 'command': ""}], # don't build extensions for omnivore since we are only using the pure python part
]

if needs_netcdf and not using_conda:
    # extra stuff isn't available through pypi or not easily built by hand
    deps.extend([
        ['https://github.com/MacPython/gattai.git',],
        ['https://github.com/robmcmullen/mac-builds.git', {'builddir': 'packages/netCDF4', 'command': 'gattai netcdf.gattai'}],
        ])


link_map = {
    "OWSLib": "owslib",
    "pyfilesystem": "fs",
    "GnomeTools": "post_gnome",
    "gattai": None,
}


real_call = subprocess.call
def git(args, branch=None):
    if sys.platform != "win32":
        real_args = ['git']
        real_args.extend(args)
        real_call(real_args)

dry_run = False
if dry_run:
    def dry_run_call(args):
        print "in %s: %s" % (os.getcwd(), " ".join(args))
    subprocess.call = dry_run_call
    def dry_run_symlink(source, name):
        print "in %s: %s -> %s" % (os.getcwd(), name, source)
    os.symlink = dry_run_symlink

if using_conda:
    # let conda manage the dependencies rather than using pip. setup.py can't
    # seem to find dependencies installed via conda and will try to install
    # them again.
    #setup = "python setup.py --no-deps "
    setup = "python setup.py "
else:
    setup = "python setup.py "

linkdir = os.getcwd()
topdir = os.path.join(os.getcwd(), "deps")

for dep in deps:
    os.chdir(topdir)
    try:
        repourl, options = dep
    except ValueError:
        repourl = dep[0]
        options = {}
    if repourl.startswith("http"):
        print "UPDATING %s" % repourl
        _, repo = os.path.split(repourl)
        repodir, _ = os.path.splitext(repo)
        if os.path.exists(repodir):
            os.chdir(repodir)
            git(['pull'])
        else:
            git(['clone', repourl])
    else:
        repodir = repourl

    builddir = options.get('builddir', ".")

    command = options.get('command',
        setup + "build_ext --inplace")
    link = repodir
    if "install" in command:
        link = None
    else:
        link = repodir
    if command:
        os.chdir(topdir)
        os.chdir(repodir)
        if 'branch' in options:
            git(['checkout', options['branch']])
        os.chdir(builddir)
        subprocess.call(command.split())
        if "install" not in command and develop_instead_of_link:
            subprocess.call(["python", "setup.py", "develop"])

    if link and sys.platform != "win32":
        os.chdir(linkdir)
        name = link_map.get(repodir, repodir)
        if name is None:
            print "No link for %s" % repodir
        else:
            src = os.path.normpath(os.path.join("deps", repodir, builddir, name))
            if os.path.islink(name):
                os.unlink(name)
            os.symlink(src, name)

os.chdir(linkdir)
subprocess.call(["python", "setup_library.py", "build_ext", "--inplace"])
