import numpy as np
import nose.tools
from maproomlib.utility.Property import Property


class Test_property:
    def setUp( self ):
        self.property = Property( "name", "value" )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == "value"
        assert self.property.type == None
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( "newvalue" )
        assert self.property.value == "newvalue"

    def test_update_with_int( self ):
        self.property.update( 7 )
        assert self.property.value == 7

    def test_str( self ):
        assert str( self.property ) == "value"


class Test_property_with_str_type:
    def setUp( self ):
        self.property = Property( "name", "value", type = str )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == "value"
        assert self.property.type == str
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( "newvalue" )
        assert self.property.value == "newvalue"

    def test_update_with_int( self ):
        self.property.update( 7 )
        assert self.property.value == "7"

    def test_str( self ):
        assert str( self.property ) == "value"


class Test_property_with_int_type:
    def setUp( self ):
        self.property = Property( "name", 7, type = int, min = -10, max = 10 )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == 7
        assert self.property.type == int
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( 8 )
        assert self.property.value == 8

    def test_update_with_float( self ):
        self.property.update( 5.5 )
        assert self.property.value == 5

    def test_update_with_str( self ):
        self.property.update( "7" )
        assert self.property.value == 7

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_small( self ):
        self.property.update( "-11" )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_big( self ):
        self.property.update( "11" )

    @nose.tools.raises( ValueError )
    def test_update_with_invalid_str( self ):
        self.property.update( "7f" )

    def test_str( self ):
        assert str( self.property ) == "7"


class Test_property_with_float_type:
    def setUp( self ):
        self.property = Property( "name", 7.7, type = float, min = -10, max = 10 )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == 7.7
        assert self.property.type == float
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( 8.8 )
        assert self.property.value == 8.8

    def test_update_with_int( self ):
        self.property.update( 5 )
        assert self.property.value == 5.0

    def test_update_with_str( self ):
        self.property.update( "1.123" )
        assert self.property.value == 1.123

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_small( self ):
        self.property.update( "-11" )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_big( self ):
        self.property.update( "11" )

    @nose.tools.raises( ValueError )
    def test_update_with_invalid_str( self ):
        self.property.update( "7.8f" )

    def test_str( self ):
        assert str( self.property ) == "7.7"


class Test_property_with_numpy_float_type:
    def setUp( self ):
        self.property = Property( "name", 7.7, type = np.float32, min = -10, max = 10 )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == np.float32( 7.7 )
        assert self.property.type == np.float32
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( 8.8 )
        assert self.property.value == np.float32( 8.8 )

    def test_update_with_int( self ):
        self.property.update( 5 )
        assert self.property.value == np.float32( 5.0 )

    def test_update_with_nan( self ):
        self.property.update( np.nan )
        assert np.isnan( self.property.value )

    def test_update_with_str( self ):
        self.property.update( "1.123" )
        assert self.property.value == np.float32( 1.123 )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_small( self ):
        self.property.update( "-11" )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_big( self ):
        self.property.update( "11" )

    @nose.tools.raises( ValueError )
    def test_update_with_invalid_str( self ):
        self.property.update( "7.8f" )

    def test_str( self ):
        assert str( self.property ) == "7.7"


class Test_property_with_numpy_float_type_nan:
    def setUp( self ):
        self.property = Property( "name", np.nan, type = np.float32, min = -10, max = 10 )

    def test_init( self ):
        assert self.property.name == "name"
        assert np.isnan( self.property.value )
        assert self.property.type == np.float32
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( 8.8 )
        assert self.property.value == np.float32( 8.8 )

    def test_update_with_int( self ):
        self.property.update( 5 )
        assert self.property.value == np.float32( 5.0 )

    def test_update_with_nan( self ):
        self.property.update( np.nan )
        assert np.isnan( self.property.value )

    def test_update_with_str( self ):
        self.property.update( "1.123" )
        assert self.property.value == np.float32( 1.123 )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_small( self ):
        self.property.update( "-11" )

    @nose.tools.raises( ValueError )
    def test_update_with_str_too_big( self ):
        self.property.update( "11" )

    @nose.tools.raises( ValueError )
    def test_update_with_invalid_str( self ):
        self.property.update( "7.8f" )

    def test_str( self ):
        assert str( self.property ) == ""


class Test_property_with_list_type:
    def setUp( self ):
        self.property = Property( "name", [ 1, 2, 3 ], type = list )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == [ 1, 2, 3 ]
        assert self.property.type == list
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( [ 4, 5, 6 ] )
        assert self.property.value == [ 4, 5, 6 ]

    def test_str( self ):
        assert str( self.property ) == "1, 2, 3"


class Test_property_with_tuple_type_long:
    def setUp( self ):
        self.property = Property( "name", ( 1, 2, 3, 4, 5 ), type = tuple )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == ( 1, 2, 3, 4, 5 )
        assert self.property.type == tuple
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( ( 4, 5, 6 ) )
        assert self.property.value == ( 4, 5, 6 )

    def test_str( self ):
        assert str( self.property ) == "1, 2, 3, 4, ..."


class Test_property_with_tuple_type:
    def setUp( self ):
        self.property = Property( "name", ( 1, 2, 3 ), type = tuple )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == ( 1, 2, 3 )
        assert self.property.type == tuple
        assert self.property.choices == None
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( ( 4, 5, 6 ) )
        assert self.property.value == ( 4, 5, 6 )

    def test_str( self ):
        assert str( self.property ) == "1, 2, 3"


class Test_property_with_choices:
    def setUp( self ):
        self.choices = ( 1, "2", 3.3 )
        self.property = Property( "name", 1, choices = self.choices )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == 1
        assert self.property.type == None
        assert self.property.choices == self.choices
        assert self.property.mutable == True

    def test_update( self ):
        self.property.update( "2" )
        assert self.property.value == "2"

    @nose.tools.raises( ValueError )
    def test_update_with_invalid_choice( self ):
        self.property.update( "invalid" )

    def test_str( self ):
        assert str( self.property ) == "1"


class Test_property_with_mutable_false:
    def setUp( self ):
        self.property = Property( "name", "value", mutable = False )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == "value"
        assert self.property.type == None
        assert self.property.choices == None
        assert self.property.mutable == False

    def test_update( self ):
        self.property.update( "newvalue" )
        assert self.property.value == "newvalue"

    def test_update_with_int( self ):
        self.property.update( 7 )
        assert self.property.value == 7

    def test_str( self ):
        assert str( self.property ) == "value"


class Test_property_with_extra_info:
    def setUp( self ):
        self.property = Property( "name", "value", foo = 7, bar = "8" )

    def test_init( self ):
        assert self.property.name == "name"
        assert self.property.value == "value"
        assert self.property.type == None
        assert self.property.choices == None
        assert self.property.mutable == True
        assert self.property.foo == 7
        assert self.property.bar == "8"

    def test_update( self ):
        self.property.update( "newvalue" )
        assert self.property.value == "newvalue"
        assert self.property.foo == 7
        assert self.property.bar == "8"

    def test_update_with_int( self ):
        self.property.update( 7 )
        assert self.property.value == 7
        assert self.property.foo == 7
        assert self.property.bar == "8"

    def test_str( self ):
        assert str( self.property ) == "value"
