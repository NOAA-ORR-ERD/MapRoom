import nose.tools
from maproomlib.utility.Inbox import Inbox
from maproomlib.utility.Outbox import Outbox


class Test_outbox:
    def setUp( self ):
        self.outbox = Outbox()
        self.message = { "foo": 7, "bar": "baz" }
        self.message2 = { "foo": 8, "bar": "quux" }

    def test_subscribe( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 7 )

        assert len( self.outbox.inboxes ) == 1
        assert self.outbox.inboxes.get( inbox ) == { "foo": 7 }

    def test_subscribe_twice( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 7 )
        self.outbox.subscribe( inbox, foo = 7 )

        assert len( self.outbox.inboxes ) == 1
        assert self.outbox.inboxes.get( inbox ) == { "foo": 7 }

    def test_subscribe_twice_with_different_patterns( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 7 )
        self.outbox.subscribe( inbox, foo = 8 )

        assert len( self.outbox.inboxes ) == 1
        assert self.outbox.inboxes.get( inbox ) == { "foo": 7 }

    @nose.tools.raises( ValueError )
    def test_subscribe_without_pattern( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox )

    def test_unsubscribe( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 7 )
        self.outbox.unsubscribe( inbox )

        assert len( self.outbox.inboxes ) == 0

    def test_unsubscribe_without_subscribe( self ):
        self.dummy_task = "dummy"
        self.outbox.unsubscribe( self.dummy_task )

        assert len( self.outbox.inboxes ) == 0

    def test_send( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 7 )

        self.outbox.send( **self.message )

        assert len( inbox.messages ) == 1
        assert inbox.messages[ 0 ] == self.message

    def test_send_twice( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = ( 7, 8 ) )

        self.outbox.send( **self.message )
        self.outbox.send( **self.message2 )

        assert len( inbox.messages ) == 2
        assert inbox.messages[ 0 ] == self.message2
        assert inbox.messages[ 1 ] == self.message

    def test_send_with_multiple_subscribers( self ):
        inbox0 = Inbox()
        inbox1 = Inbox()
        self.outbox.subscribe( inbox0, foo = 7 )
        self.outbox.subscribe( inbox1, foo = 7 )

        self.outbox.send( **self.message )

        assert len( inbox0.messages ) == 1
        assert inbox0.messages[ 0 ] == self.message

        assert len( inbox1.messages ) == 1
        assert inbox1.messages[ 0 ] == self.message

    def test_send_twice_with_multiple_subscribers( self ):
        inbox0 = Inbox()
        inbox1 = Inbox()
        self.outbox.subscribe( inbox0, foo = ( 7, 8 ) )
        self.outbox.subscribe( inbox1, foo = ( 7, 8 ) )

        self.outbox.send( **self.message )
        self.outbox.send( **self.message2 )

        assert len( inbox0.messages ) == 2
        assert inbox0.messages[ 0 ] == self.message2
        assert inbox0.messages[ 1 ] == self.message

        assert len( inbox1.messages ) == 2
        assert inbox1.messages[ 0 ] == self.message2
        assert inbox1.messages[ 1 ] == self.message

    def test_send_with_non_matching_pattern( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = 27 )

        self.outbox.send( **self.message )

        assert len( inbox.messages ) == 0

    def test_send_with_non_matching_pattern_key( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, what = 7 )

        self.outbox.send( **self.message )

        assert len( inbox.messages ) == 0

    def test_send_with_partially_matching_pattern_sequence( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = ( 27, 7 ) )

        self.outbox.send( **self.message )

        assert len( inbox.messages ) == 1
        assert inbox.messages[ 0 ] == self.message

    def test_send_with_non_matching_pattern_sequence( self ):
        inbox = Inbox()
        self.outbox.subscribe( inbox, foo = ( 27, 28 ) )

        self.outbox.send( **self.message )

        assert len( inbox.messages ) == 0

    def test_send_with_non_matching_pattern_and_multiple_subscribers( self ):
        inbox0 = Inbox()
        inbox1 = Inbox()
        self.outbox.subscribe( inbox0, foo = 27 )
        self.outbox.subscribe( inbox1, foo = 7 )

        self.outbox.send( **self.message )

        assert len( inbox0.messages ) == 0

        assert len( inbox1.messages ) == 1
        assert inbox1.messages[ 0 ] == self.message

    def test_send_with_partially_matching_pattern_sequence_and_multiple_subscribers( self ):
        inbox0 = Inbox()
        inbox1 = Inbox()
        self.outbox.subscribe( inbox0, foo = 27 )
        self.outbox.subscribe( inbox1, foo = ( 27, 7 ) )

        self.outbox.send( **self.message )

        assert len( inbox0.messages ) == 0

        assert len( inbox1.messages ) == 1
        assert inbox1.messages[ 0 ] == self.message
