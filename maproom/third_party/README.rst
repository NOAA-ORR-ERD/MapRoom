These are packages that aren't in conda or PyPI.

glsvg
=====

glsvg is cloned from https://github.com/fathat/glsvg.git

The reason it is included here is the PyPI package is very outdated and only
for python 2, and the python 3 support is only on the github page. In addition
to the python 3 changes, there are minor modifications to the import
statements allowing the package to be called with "maproom.third_party.glsvg"
instead of only "glsvg".


post_gnome
==========

post_gnome is a subdirectory of GnomeTools, which is from
https://github.com/NOAA-ORR-ERD/GnomeTools

It is only included here because there is no package on PyPI or Conda for it;
it is not modified.
