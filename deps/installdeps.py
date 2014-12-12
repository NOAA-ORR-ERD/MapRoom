#!/usr/bin/env python
import subprocess
import os

deps = [
    ['https://github.com/NOAA-ORR-ERD/GnomeTools.git', 'post_gnome',],
    ['https://github.com/pyugrid/pyugrid.git',],
    ['https://github.com/enthought/traits.git',],
    ['https://github.com/robmcmullen/pyface.git',],
    ['https://github.com/robmcmullen/traitsui.git',],
    ['https://github.com/enthought/apptools.git',],
    ['https://github.com/robmcmullen/envisage.git',],
    ['https://github.com/robmcmullen/peppy2.git',],
]

topdir = os.getcwd()

for dep in deps:
    os.chdir(topdir)
    repourl = dep[0]
    print "UPDATING %s" % repourl
    _, repo = os.path.split(repourl)
    repodir, _ = os.path.splitext(repo)
    subdirs = dep[1:]
    if not subdirs:
        subdirs = ['.']
    if os.path.exists(repodir):
        os.chdir(repodir)
        subprocess.call(['git', 'pull'])
    else:
        subprocess.call(['git', 'clone', repourl])
    for subdir in subdirs:
        os.chdir(topdir)
        os.chdir(repodir)
        os.chdir(subdir)
        subprocess.call(['python', 'setup.py', 'develop'])
