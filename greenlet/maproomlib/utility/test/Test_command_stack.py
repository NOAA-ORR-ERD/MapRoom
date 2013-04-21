import maproomlib.utility
from maproomlib.utility.Command_stack import Command_stack
from maproomlib.utility.Inbox import Inbox
from Mock_scheduler import Mock_scheduler


TIMEOUT = 0.5


class Test_command_stack:
    def setUp( self ):
        self.stack = Command_stack()
        self.scheduler = Mock_scheduler()
        self.inbox = Inbox()

        self.stack.outbox.subscribe(
            self.inbox,
            request = "undo_updated",
        )

        self.description = "test command"
        self.redone = None
        self.undone = None
        self.cleaned_up = None

    def redo( self, value ):
        self.redone = value

    def undo( self, value ):
        self.undone = value

    def cleanup( self, value ):
        self.cleaned_up = value

    def test_init( self ):
        assert self.stack.undo_index == -1

    def test_add( self ):
        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 1
        assert self.stack.undo_index == 0

        command = self.stack.stack[ 0 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == None

    def test_add_with_group( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 2
        assert self.stack.undo_index == 1

        command = self.stack.stack[ 1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == None

    def test_add_with_repeated_group( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 2
        assert self.stack.undo_index == 1

        command = self.stack.stack[ 1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == None

    def test_add_with_stack_overflow( self, with_cleanup = False ):
        # Fill up the stack with commands.
        for i in range( self.stack.MAX_COMMAND_COUNT ):
            # Putting this in a separate function because using a loop
            # variable in a lambda doesn't "capture" the current value.
            def send( i ):
                self.stack.inbox.send(
                    request = "add",
                    description = self.description + str( i ),
                    redo = lambda: self.redo( i ),
                    undo = lambda: self.undo( i + 0.5 ),
                    cleanup = lambda: self.cleanup( i + 0.75 ) if with_cleanup else None,
                )

            send( i )
            self.stack.run( self.scheduler, run_once = True )

            assert len( self.stack.stack ) == i + 1
            assert self.stack.undo_index == i

            command = self.stack.stack[ -1 ]
            assert command.description == self.description + str( i )

            assert self.redone is None
            assert self.undone is None
            assert self.cleaned_up is None

        # The command at the bottom of the stack should be the first command.
        assert len( self.stack.stack ) == self.stack.MAX_COMMAND_COUNT
        assert self.stack.undo_index == self.stack.MAX_COMMAND_COUNT - 1

        command = self.stack.stack[ 0 ]
        assert command.description == self.description + str( 0 )

        # Now overflow the stack.
        description = "last command"
        self.stack.inbox.send(
            request = "add",
            description = description,
            redo = lambda: self.redo( 100.1 ),
            undo = lambda: self.undo( 100.2 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == self.stack.MAX_COMMAND_COUNT
        assert self.stack.undo_index == self.stack.MAX_COMMAND_COUNT - 1

        command = self.stack.stack[ -1 ]
        assert command.description == description

        assert self.redone is None
        assert self.undone is None

        # Assert that the bottom of the stack had .cleanup() called.
        if with_cleanup:
            assert self.cleaned_up == 0.75

        # Make sure the command at the bottom of the stack was thrown away.
        command = self.stack.stack[ 0 ]
        assert command.description == self.description + str( 1 )

    def test_add_with_stack_overflow_and_cleanup( self ):
        self.test_add_with_stack_overflow( with_cleanup = True )

    def test_undo( self ):
        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 1
        assert self.stack.undo_index == -1

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is 1
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == None
        assert message.get( "next_redo_description" ) == self.description

    def test_undo_with_group( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 2
        assert self.stack.undo_index == -1

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is 1
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == None
        assert message.get( "next_redo_description" ) == self.description

    def test_undo_with_multiple_groups( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = "not undone yet",
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = "selected something",
            redo = lambda: self.redo( 2 ),
            undo = lambda: self.undo( 3 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 4 ),
            undo = lambda: self.undo( 5 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 5
        assert self.stack.undo_index == 1

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is 3
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == "not undone yet"
        assert message.get( "next_redo_description" ) == self.description

    def test_undo_with_empty_stack( self ):
        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 0
        assert self.stack.undo_index == -1

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        assert len( self.inbox.messages ) == 0

    def test_redo( self ):
        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )
        self.undone = None

        self.stack.inbox.send( request = "redo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 1
        assert self.stack.undo_index == 0

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is 0
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == None

    def test_redo_with_group( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )
        self.undone = None

        self.stack.inbox.send( request = "redo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 2
        assert self.stack.undo_index == 1

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is 0
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == None

    def test_redo_with_multiple_groups( self ):
        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = "another command",
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = "some command",
            redo = lambda: self.redo( 2 ),
            undo = lambda: self.undo( 3 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 4 ),
            undo = lambda: self.undo( 5 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "start_command",
        )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send(
            request = "add",
            description = "last command",
            redo = lambda: self.redo( 2 ),
            undo = lambda: self.undo( 3 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )
        self.undone = None

        self.stack.inbox.send( request = "redo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 7
        assert self.stack.undo_index == 4

        command = self.stack.stack[ self.stack.undo_index ]
        assert command.description == self.description

        assert self.redone is 4
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == self.description
        assert message.get( "next_redo_description" ) == "last command"

    def test_redo_with_empty_stack( self ):
        self.stack.inbox.send( request = "redo" )
        self.stack.run( self.scheduler, run_once = True )
        assert len( self.stack.stack ) == 0
        assert self.stack.undo_index == -1

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        assert len( self.inbox.messages ) == 0

    def test_redo_with_nothing_to_redo( self ):
        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "redo" )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 1
        assert self.stack.undo_index == 0

        command = self.stack.stack[ -1 ]
        assert command.description == self.description

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        assert len( self.inbox.messages ) == 0

    def test_add_after_undo( self ):
        self.stack.inbox.send(
            request = "add",
            description = self.description,
            redo = lambda: self.redo( 0 ),
            undo = lambda: self.undo( 1 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )

        self.stack.inbox.send( request = "undo" )
        self.stack.run( self.scheduler, run_once = True )

        self.inbox.discard( request = "undo_updated" )
        self.undone = None

        new_description = "new description"
        self.stack.inbox.send(
            request = "add",
            description = new_description,
            redo = lambda: self.redo( 2 ),
            undo = lambda: self.undo( 3 ),
        )
        self.stack.run( self.scheduler, run_once = True )

        assert len( self.stack.stack ) == 1
        assert self.stack.undo_index == 0

        command = self.stack.stack[ -1 ]
        assert command.description == new_description

        assert self.redone is None
        assert self.undone is None
        assert self.cleaned_up is None

        message = self.inbox.receive(
            request = "undo_updated",
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "next_undo_description" ) == new_description
        assert message.get( "next_redo_description" ) == None
