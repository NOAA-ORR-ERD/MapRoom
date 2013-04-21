import numpy as np
import maproomlib.utility as utility


def assert_color_to_int( red, green, blue, alpha ):
    color = utility.color_to_int( red, green, blue, alpha )
    channels = np.array( [ color ] ).view( np.uint8 )

    CHANNEL_MAX = 255.0
    TOLERANCE = 0.01

    assert abs( channels[ 0 ] / CHANNEL_MAX - red ) < TOLERANCE
    assert abs( channels[ 1 ] / CHANNEL_MAX - green ) < TOLERANCE
    assert abs( channels[ 2 ] / CHANNEL_MAX - blue ) < TOLERANCE
    assert abs( channels[ 3 ] / CHANNEL_MAX - alpha ) < TOLERANCE


def test_color_to_int_red():
    assert_color_to_int( 1.0, 0, 0, 1.0 )


def test_color_to_int_green():
    assert_color_to_int( 0, 1.0, 0, 1.0 )


def test_color_to_int_blue():
    assert_color_to_int( 0, 0, 1.0, 1.0 )


def test_color_to_int_red_half_alpha():
    assert_color_to_int( 1.0, 0, 0, 0.5 )


def test_color_to_int_red_no_alpha():
    assert_color_to_int( 1.0, 0, 0, 1.0 )


def test_color_to_int_half_red():
    assert_color_to_int( 0.5, 0, 0, 1.0 )


def test_color_to_int_purplish():
    assert_color_to_int( 0.5, 0, 0.5, 1.0 )


def test_color_to_int_greyish():
    assert_color_to_int( 0.25, 0.25, 0.25, 1.0 )
