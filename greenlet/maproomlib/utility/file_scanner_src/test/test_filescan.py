#!/usr/bin/env python

# set up the data file:

import numpy as np
print "running test_filescan"

name = "JunkTiny.txt"

file(name, 'w').write("""23.4, 45.6, 354
13.23,  45.6
2.3   5.4   3.4
14;  32  ;  3
""")
Orig = np.array((23.4, 45.6, 354, 13.23, 45.6, 2.3, 5.4, 3.4, 14, 32, 3), dtype=np.float)

Orig_single = np.array((23.4, 45.6, 354, 13.23, 45.6, 2.3, 5.4, 3.4, 14, 32, 3), dtype=np.float32)

import file_scanner

def test_ScanN():
    f = file(name)
    A = file_scanner.FileScanN(f, 10)
    assert np.array_equal(Orig[:10], A)

def test_Scan():
    f = file(name)
    A = file_scanner.FileScan(f)
    assert np.array_equal(Orig, A)

def test_Scan_single():
    f = file(name)
    A = file_scanner.FileScanN_single(f, 11)
    print A.dtype
    print A
    assert A.dtype == np.float32
    assert np.array_equal(Orig_single, A)
    
#
#print f.readline()
#f.close()
#
#f = open("JunkTiny.txt")
#
#A = FileScanner.FileScan(f)
#print A.shape 
#for a in A: print a
