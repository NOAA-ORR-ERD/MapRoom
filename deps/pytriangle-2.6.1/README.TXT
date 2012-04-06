 = README for pytriangle =

pytriangle is a python wrapper around the "Triangle" program/library by Jonathan Richard Shewchuk.

For more information about Triangle, see:

http://www.cs.cmu.edu/~quake/triangle.html

pytriangle is written with Cython, for generating the C wrapper code:

http://cython.org/

currently being tested with Cython 1.4

 = Building =
 
 == Building Triangle ==

Triangle is not distributed with this code. You need to go get it yourself from the Triangle web site:

http://www.cs.cmu.edu/~quake/triangle.html
 
The rest of these instructions assume that you have unpacked the Triangle source in the pytriangle directory.

Triangle is built along with the binding code by the distutils setup.py script

 == Running Cython ==

If you are not changing hte code, cython should already have been run, and you can just build the extension. If there is a pytriangle.c file, then it has been done.

If you have changed the *.pyx file, you need to run cython to build the C code from the cython code. 

simple enough:

`$ cython src/pytriangle.pyx`

That should generate `src/pytriangle.c`

 == Building the Python extension ==
 
This should be the usual distutils dance:
 
 `python setup.py build`

 `python setup.py install`
 
 though if you are working on it, you might want to use:
 
 `python setup.py develop`

you can also do:
 `python setup.py build_ext -inplace
 
 if you just want to test.
 
If all goes well, this should compile triangle.c and link it all together for you.

 = Testing =
 
There isn't much test code yet, but running what's in the `tests` dir should at least let you know that it's compiled and installed properly.

 = LICENSE INFORMATION: =

Triangle itself is released with a non-standard license:
 - "Private, research, and institutional use is free."
 - "Distribution of this code as part of a commercial system is permissible ONLY BY DIRECT ARRANGEMENT
WITH THE AUTHOR."

Thus, the Triangle code itself will not be distributed with this python extension.

This extension code was developed by the United States Federal government employees and contractors, and thus is released under the "CC0", essentially released into the Public Domain.

http://creativecommons.org/publicdomain/zero/1.0/

