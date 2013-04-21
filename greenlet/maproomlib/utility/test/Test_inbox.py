import nose.tools
import maproomlib.utility
from maproomlib.utility.Inbox import Inbox, Receive_error
from maproomlib.utility import Timeout_error
from Mock_scheduler import Mock_scheduler


class Test_exception( Exception ):
    pass


class Test_inbox:
    def setUp( self ):
        self.inbox = Inbox()
        self.scheduler = Mock_scheduler()
        self.message = { "foo": 7, "bar": "baz" }
        self.message2 = { "foo": 8, "bar": "quux" }
        self.message3 = { "foo": 9, "qux": "corge" }

    def test_send( self ):
        self.inbox.send( **self.message )

        assert len( self.inbox.messages ) == 1
        assert self.inbox.messages[ 0 ] == self.message

    def test_send_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        assert len( self.inbox.messages ) == 2
        assert self.inbox.messages[ 0 ] == self.message2
        assert self.inbox.messages[ 1 ] == self.message

    def test_send_with_waiting_task( self ):
        self.dummy_task = "dummy"
        self.inbox.waiting = self.dummy_task
        self.inbox.waiting_scheduler = self.scheduler

        self.inbox.send( **self.message )

        assert len( self.scheduler.running ) == 1
        assert self.scheduler.running[ 0 ] == self.dummy_task
        assert len( self.scheduler.sleeping ) == 0

    def test_receive( self ):
        self.inbox.send( **self.message )

        # Sleep should not be called here.
        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler )

        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    @nose.tools.raises( Test_exception )
    def test_receive_exception( self ):
        self.inbox.send( exception = Test_exception() )

        # Sleep should not be called here.
        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler )

    def test_receive_with_pattern( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = 7 )

        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    @nose.tools.raises( Test_exception )
    def test_receive_exception_with_pattern( self ):
        self.inbox.send( exception = Test_exception() )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = 7 )

    def test_receive_with_pattern_with_list_value( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = [ 3, 7, 2 ] )

        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    def test_receive_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 1

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 8
        assert message.get( "bar" ) == "quux"
        assert len( self.inbox.messages ) == 0

    def test_receive_with_pattern_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = 8 )
        assert message.get( "foo" ) == 8
        assert message.get( "bar" ) == "quux"
        assert len( self.inbox.messages ) == 1

        message = self.inbox.receive( self.scheduler, bar = "baz" )
        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    def test_receive_with_pattern_with_list_value_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = [ 3, 8, 2 ] )
        assert message.get( "foo" ) == 8
        assert message.get( "bar" ) == "quux"
        assert len( self.inbox.messages ) == 1

        message = self.inbox.receive( self.scheduler, bar = [ "baz", "corge" ] )
        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    def test_receive_thrice_with_pattern_for_second_message( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )
        self.inbox.send( **self.message3 )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = 8 )
        assert message.get( "foo" ) == 8
        assert message.get( "bar" ) == "quux"
        assert len( self.inbox.messages ) == 2

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 1

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 9
        assert message.get( "qux" ) == "corge"
        assert len( self.inbox.messages ) == 0

    def test_receive_thrice_with_pattern_for_third_message( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )
        self.inbox.send( **self.message3 )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler, foo = 9 )
        assert message.get( "foo" ) == 9
        assert message.get( "qux" ) == "corge"
        assert len( self.inbox.messages ) == 2

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 1

        message = self.inbox.receive( self.scheduler )
        assert message.get( "foo" ) == 8
        assert message.get( "bar" ) == "quux"
        assert len( self.inbox.messages ) == 0

    @nose.tools.raises( Timeout_error )
    def test_receive_without_available_message( self ):
        self.inbox.receive( self.scheduler, timeout = 0.1 )

    def test_receive_after_waiting( self ):
        # After sleep is called, perform a send() to simulate waiting for a
        # message.
        def sleep_called():
            self.inbox.send( **self.message )

        self.scheduler.sleep_callback = sleep_called

        message = self.inbox.receive( self.scheduler )

        assert message.get( "foo" ) == 7
        assert message.get( "bar" ) == "baz"
        assert len( self.inbox.messages ) == 0

    @nose.tools.raises( Timeout_error )
    def test_receive_with_non_matching_pattern( self ):
        self.inbox.send( **self.message )
        self.inbox.receive( self.scheduler, unknown = "value", timeout = 0.1 )

    @nose.tools.raises( Receive_error )
    def test_receive_when_already_waiting( self ):
        self.inbox.send( **self.message )

        self.dummy_task = "dummy"
        self.inbox.waiting = self.dummy_task

        self.inbox.receive( self.scheduler )

    def test_discard( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

    def test_discard_with_pattern( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard( foo = 7 )
        assert len( self.inbox.messages ) == 0

    def test_discard_with_pattern_with_list_value( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard( foo = [ 3, 7, 2 ] )
        assert len( self.inbox.messages ) == 0

    def test_discard_with_non_matching_pattern( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard( nomatch = 19 )
        assert len( self.inbox.messages ) == 1

    def test_discard_twice( self ):
        self.inbox.send( **self.message )

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

    def test_discard_without_sending( self ):
        def sleep_called(): assert False
        self.scheduler.sleep_callback = sleep_called

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

    def test_discard_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

    def test_discard_with_pattern_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        self.inbox.discard( foo = 8 )
        assert len( self.inbox.messages ) == 1

        self.inbox.discard( bar = "baz" )
        assert len( self.inbox.messages ) == 0

    def test_discard_with_pattern_with_list_value_twice( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )

        self.inbox.discard( foo = [ 3, 8, 2 ] )
        assert len( self.inbox.messages ) == 1

        self.inbox.discard( bar = [ "baz", "corge" ] )
        assert len( self.inbox.messages ) == 0

    def test_discard_twice_with_three_messages( self ):
        self.inbox.send( **self.message )
        self.inbox.send( **self.message2 )
        self.inbox.send( **self.message3 )

        self.inbox.discard( foo = 8 )
        assert len( self.inbox.messages ) == 2

        self.inbox.discard()
        assert len( self.inbox.messages ) == 0

    def test_empty_true( self ):
        assert self.inbox.empty() is True

    def test_empty_true_after_send_and_receive( self ):
        self.inbox.send( **self.message )
        self.inbox.receive( self.scheduler )

        assert self.inbox.empty() is True

    def test_empty_false_after_send( self ):
        self.inbox.send( **self.message )

        assert self.inbox.empty() is False
