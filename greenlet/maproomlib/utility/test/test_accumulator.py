#!/usr/bin/env python

"""
tests for the accumulator class

designed to be run with nose
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from maproomlib.utility.accumulator import accumulator

class test_init(unittest.TestCase):
    
    def test_odd_shape(self):
        self.assertRaises(ValueError, accumulator,  ((1, 2),(3, 4, 5) )  )

    def test_empty(self):
        a = accumulator()
        self.assertEqual( len(a), 0 )
        
    def test_simple(self):
        a = accumulator( (1,2,3) )
        print a.shape
        print len(a)
        self.assertEqual( len(a), 3 )
        
    def test_buffer_init(self):
        a = accumulator()
        self.assertEqual( a.bufferlength, a.DEFAULT_BUFFER_SIZE )

    def test_dtype(self):
        a = accumulator(dtype=np.uint8)
        self.assertEqual( a.dtype, np.uint8 )

    def test_dtype_None(self):
        a = accumulator()
        self.assertEqual( a.dtype, np.float64 )

    def test_shape(self):
        a = accumulator((1,2,4.5), dtype=np.float64)
        self.assertEqual( a.shape, (3,) )
        
    def test_scalar(self):
        """
        passing a scalar in to the __init__ should give you a length-one array,
        as it doesn'tmake sesne to have a scalar accumulator
        """
        a = accumulator(5)
        self.assertEqual(len(a), 1)
        
class test_indexing(unittest.TestCase):
    
    def test_simple_index(self):
        a = accumulator( (1,2,3,4,5) )
        self.assertEqual(a[1], 2)

    def test_neg_index(self):
        a = accumulator( (1,2,3,4,5) )
        self.assertEqual(a[-1], 5)

    def test_index_too_big(self):
        a = accumulator( (1,2,3,4,5) )
        # I can't figure out how to use asserRaises for a non-callable
        try:
            a[6]
        except IndexError:
            pass
        else:
            raise Exception("This test didn't raise an IndexError")

    def test_append_then_index(self):
        a = accumulator( () )
        for i in range(20):
            a.append(i)
        self.assertEqual(a[15], 15)

    def test_indexs_then_resize(self):
       """
       this here to see if having a view on teh buffer causes problems
       """
       a = accumulator( (1,2,3,4,5) )
       b = a[4]
       a.resize(1000)


class test_slicing(unittest.TestCase):
    
    def test_simple_slice(self):
        a = accumulator( (1,2,3,4,5) )
        assert_array_equal(a[1:3], np.array([2, 3]))

    def test_too_big_slice(self):
        b = np.array( (1.0, 3, 4, 5, 6) )
        a = accumulator( b )
        assert_array_equal(a[1:10], b[1:10])

    def test_full_slice(self):
        b = np.array( (1.0, 3, 4, 5, 6) )
        a = accumulator( b )
        assert_array_equal(a[:], b[:])

    def test_slice_then_resize(self):
       """
       this here to see if having a view on th buffer causes problems
       """
       a = accumulator( (1,2,3,4,5) )
       b = a[2:4]
       a.resize(1000)


class test_append(unittest.TestCase):
    
    def test_append_length(self):
        a = accumulator( (1,2,3) )
        a.append(4)
        self.assertEqual(len(a), 4)

    def test_append_lots(self):
        """tests for buffer expansion"""
        a = accumulator( (1,2,3) )
        buff_size = a.bufferlength
        for i in range(buff_size + 1):
            a.append(4)
        self.assertEqual(len(a), 3 + buff_size + 1)


class test_extend(unittest.TestCase):
    
    def test_extend_length(self):
        a = accumulator( (1,2,3) )
        a.extend( (4, 5, 6) )
        self.assertEqual(len(a), 6)

    def test_extend_long(self):
        a = accumulator( range(100) )
        a.extend( range(100) )
        print len(a)
        self.assertEqual(len(a), 200)

    def test_extend_from_zero(self):
        a = accumulator( )
        a.extend( (4, 5, 6) )
        self.assertEqual(len(a), 3)

class test_resize(unittest.TestCase):
    
    def test_resize_longer(self):
        a = accumulator( (1,2,3) )
        a.resize(1000)
        self.assertEqual(len(a), 3)
        self.assertEqual(a.bufferlength, 1000)

    def test_resize_too_short(self):
        a = accumulator( (1,2,3,4,5,6,7,8) )
        self.assertRaises(ValueError, a.resize, 5)
        
    def test_fitbuffer(self):
        a = accumulator( (1,2,3) )
        a.fitbuffer()
        self.assertEqual(a.bufferlength, 3)

class test_special(unittest.TestCase):
    
    def test_asarray(self):
        a = accumulator( (1,2,3) )
        b = np.asarray(a)
        print b
        assert_array_equal(a[:], b)
    
    def test_asarray_dtype(self):
        a = accumulator( (1,2,3), dtype=np.uint32 )
        b = np.asarray(a, dtype=np.float)
        self.assertEqual(b.dtype, np.float)
        
    def test__eq__(self):
        a = accumulator( (1,2,3), dtype=np.uint32 )
        print a
        print a == (1, 4, 3)
        
        assert_array_equal(a == (1, 4, 3), (True, False, True) )
    
class test_strings(unittest.TestCase):

    def test_str(self):
        a = accumulator( (1,2,3), dtype=np.float )
        self.assertEqual(str(a), '[ 1.  2.  3.]')

    def test_repr(self):
        a = accumulator( (1,2,3), dtype=np.float )
        self.assertEqual(repr(a), 'accumulator([ 1.,  2.,  3.])')


class test_complex_data_types(unittest.TestCase):
    dt = np.dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])
    item = (5, [[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]])

    def test_dtype1(self):
        a = accumulator( self.item, dtype=self.dt )
        self.assertEqual(a[0][0], 5)

    def test_dtype2(self):
        a = accumulator( self.item, dtype=self.dt )
        assert_array_equal(a[0][1],
                           np.array([[3.2, 4.5, 6.5], [1.2, 4.2e7, -45.76]], dtype=np.float64)
                           )
                           
    def test_dtype3(self):
        a = accumulator(dtype=self.dt )
        for i in range(100):
            a.append(self.item)
        assert_array_equal( a[99][1][1,2], -45.76 )
        
class test_nd_array(unittest.TestCase):
    ## note: this needs more tests!
    def test_2d(self):
        b = np.array(((1,2),(3,4),(5,6)), dtype=np.float)
        a = accumulator( b )
        assert b.shape == (3, 2)

    def test_append_error_2d(self):
        b = np.array(((1,2),(3,4),(5,6)), dtype=np.float)
        a = accumulator( b )
        self.assertRaises( ValueError, a.append, (3,4,5) )

    def test_append_broadcast(self):
        b = np.array(((1,2),(3,4),(5,6)), dtype=np.float)
        a = accumulator( b )
        a.append(3.0)
        assert_array_equal( a[-1], (3.0, 3.0) )

    def test_3d(self):
        b = np.arange(24).reshape(2,3,4)
        a = accumulator( b )
        assert b.shape == (2, 3, 4)
    
    def test_init_empty(self):
        a = accumulator(block_shape=(2,3), dtype=np.float)
        a.append( ((1,2,3),(4,5,6)) )

    def test_nd_slice_1(self):
        b = np.array(((1,2),(3,4),(5,6),(7,8),(9,10)), dtype=np.float)
        a = accumulator( b )
        print a[2:4]
        assert_array_equal( a[2:4], (( 5., 6.), ( 7., 8.)) )
        
#    def test_nd_slice_2(self):
#        b = np.array(((1,2),(3,4),(5,6),(7,8),(9,10)), dtype=np.float)
#        a = accumulator( b )
#        print a[2:4,1]
#        assert_array_equal( a[2:4], (( 5., 6.), ( 7., 8.)) )
        
        