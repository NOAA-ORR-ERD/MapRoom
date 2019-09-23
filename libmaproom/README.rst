This package contains all Cython modules needed by MapRoom and was
moved into this standalone package so that it can be compiled into
a python wheel. This allows the main MapRoom package to be pure
python with no need to install compilers. Windows compilers have
proven to be a major obstacle to allowing developers to contribute
to MapRoom, so by isolating this (seldom-modified) code to its
own package, it is hoped that more people will be able to develop
and extend MapRoom.
