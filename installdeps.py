#!/usr/bin/env python
import subprocess
import os
import sys

using_conda = "Continuum Analytics" in sys.version or "conda" in sys.version
needs_netcdf = not sys.platform.startswith("win")
develop_instead_of_link = False

deps = [
    [".", 'pytriangle-1.6.1', 'python setup.py install'],
    ['https://github.com/NOAA-ORR-ERD/GnomeTools.git', 'post_gnome',],
    ['https://github.com/robmcmullen/OWSLib.git',],
    ['https://github.com/fathat/glsvg.git',],
    ['https://github.com/robmcmullen/pyugrid.git',],
    ['https://github.com/robmcmullen/traits.git',],
    ['https://github.com/robmcmullen/pyface.git',],
    ['https://github.com/robmcmullen/traitsui.git',],
    ['https://github.com/enthought/apptools.git',],
    ['https://github.com/robmcmullen/envisage.git',],
    ['https://github.com/robmcmullen/pyfilesystem.git',],
    ['https://github.com/robmcmullen/omnivore.git', '.', None], # don't build extensions for omnivore since we are only using the pure python part
]

if needs_netcdf and not using_conda:
    # extra stuff isn't available through pypi or not easily built by hand
    deps.extend([
        ['https://github.com/MacPython/gattai.git',],
        ['https://github.com/robmcmullen/mac-builds.git', 'packages/netCDF4', 'gattai netcdf.gattai'],
        ])


link_map = {
    "OWSLib": "owslib",
    "pyfilesystem": "fs",
    "GnomeTools": "post_gnome",
    "gattai": None,
}

# hack to overwrite files so I don't have to keep patching my git copy of
# pyface
overrides = {
    "deps/pyface/pyface/toolkit.py": """
from pyface.base_toolkit import Toolkit
toolkit_object = Toolkit('wx', 'pyface.ui.wx')
""",
}


real_call = subprocess.call
def git(args):
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
    command_extra_args = " --no-deps"
else:
    command_extra_args = ""

linkdir = os.getcwd()
topdir = os.path.join(os.getcwd(), "deps")

for dep in deps:
    os.chdir(topdir)
    repourl = dep[0]
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
    if len(dep) == 1:
        builddir = "."
        command = "python setup.py build_ext --inplace" + command_extra_args
        link = repodir
    elif len(dep) == 2:
        builddir = dep[1]
        command = "python setup.py build_ext --inplace" + command_extra_args
        link = repodir
    else:
        builddir = dep[1]
        command = dep[2]
        if command and "install" in command:
            link = None
        else:
            link = repodir
    if command:
        os.chdir(topdir)
        os.chdir(repodir)
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
for path, text in overrides.iteritems():
    print "Replacing %s" % path
    with open(path, "w") as fh:
        fh.write(text)

subprocess.call(["python", "setup_library.py", "build_ext", "--inplace"])
