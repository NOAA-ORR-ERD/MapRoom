import collections
from Inbox import Inbox
from Outbox import Outbox


class Command:
    def __init__( self, description = None, redo = None, undo = None,
                  cleanup = None ):
        self.description = description
        self.redo = redo
        self.undo = undo
        self.cleanup = cleanup


class Command_stack:
    """
    A stack of undoable and redoable messages, as per the Command Pattern.

    .. function:: start_command()

        When a message with ``request = "start_command"`` is received within
        the :attr:`inbox`, a handler marks the current undo position as the
        start of a new command group (implemented by adding a start group
        command to the undo stack). Any subsequently added commands until the
        next ``start_command`` message will be undone or redone as if they
        were one command.

        This is useful when a single user-facing operation triggers multiple
        separate commands under the hood, and you want to group those commands
        together so the user need only press undo once to undo all the
        commands. In other words, you should send a ``start_command`` message
        immediately before each user-facing undoable operation is performed.

        If the command stack grows too large, then the oldest commands will be
        discarded. And if the current undo position is before the end of the
        stack when a ``start_command`` command is added, then all subsequent
        commands are discarded.

    .. function:: add( description, redo, undo )

        When a message with ``request = "add"`` is received within the
        :attr:`inbox`, a handler creates a command from the given parameters
        and adds it to the stack after the current undo position. The undo
        position is updated to point at this newly added command.

        :param description: user-facing description of the command
        :type description: str
        :param redo: what to add to redo the command
        :type redo: callable
        :param undo: what to add to undo the command
        :type undo: callable

        Additionally, a message will be sent to this object's :attr:`outbox`
        as follows::

            outbox.send( request = "undo_updated", next_undo_description, next_redo_description )

        If the command stack grows too large, then the oldest commands will be
        discarded. And if the current undo position is before the end of the
        stack when a command is added, then all subsequent commands are
        discarded.

    .. function:: undo()

        When a message with ``request = "undo"`` is received within the
        :attr:`inbox`, a handler undoes commands starting from the current
        undo position and going backwards until the last ``start_command``.
        This also moves the current undo position in the stack accordingly, so
        that the next call to undo() will undo the previous commands.

        Additionally, if there was a command to undo, a message will be sent
        to this object's :attr:`outbox` as follows::

            outbox.send( request = "undo_updated", next_undo_description, next_redo_description )

    .. function:: redo()

        When a message with ``request = "redo"`` is received within the
        :attr:`inbox`, a handler redoes the commands starting from immediately
        after the current undo position and continuing until the next
        ``start_command``. This also moves the current undo position in the
        stack accordingly, so that the next call to redo() will redo the
        next commands.

        Additionally, if there was a command to redo, a message will be sent
        to this object's :attr:`outbox` as follows::

            outbox.send( request = "undo_updated", next_undo_description, next_redo_description )
    """
    MAX_COMMAND_COUNT = 50

    def __init__( self ):
        self.stack = collections.deque()
        self.undo_index = -1 # position in stack of last add command
        self.inbox = Inbox()
        self.outbox = Outbox()
        self.group_sentinel = Command()

    def run( self, scheduler, run_once = False ):
        while True:
            message = self.inbox.receive(
                request = ( "add", "start_command", "undo", "redo" ),
            )
            request = message.pop( "request" )

            if request == "add":
                self.add( **message )
            elif request == "start_command":
                self.add_group_sentinel( **message )
            elif request == "undo":
                self.undo( **message )
            elif request == "redo":
                self.redo( **message )

            if run_once: break

    def add( self, description, redo, undo, cleanup = None ):
        self.add_command( Command( description, redo, undo, cleanup ) )

        self.send_descriptions()

    def add_group_sentinel( self ):
        # Collapse multiple group sentinels in a row into one sentinel.
        if self.undo_index != -1 and \
           self.stack[ self.undo_index ] == self.group_sentinel:
            return

        self.add_command( self.group_sentinel )

    def add_command( self, command ):
        # If there are redo commands past the current undo index, discard
        # them. (This is only linear undo, not branching.)
        while self.undo_index + 1 < len( self.stack ):
            self.stack.pop()

        self.stack.append( command )

        # If the stack gets too big, discard the oldest command. But first
        # execute any cleanup code it has.
        if len( self.stack ) > self.MAX_COMMAND_COUNT:
            if self.stack[ 0 ].cleanup:
                self.stack[ 0 ].cleanup()
            self.stack.popleft()
        else:
            self.undo_index += 1

    def undo( self ):
        if self.undo_index == -1:
            return

        while self.undo_index >= 0:
            command = self.stack[ self.undo_index ]
            if command.undo:
                command.undo()

            self.undo_index -= 1

            # Once we've undone all the commands in the group, we're done.
            if command == self.group_sentinel:
                break

        self.send_descriptions()

    def redo( self ):
        if self.undo_index + 1 >= len( self.stack ):
            return

        # Redo the first command (should be the group_sentinel).
        command = self.stack[ self.undo_index + 1 ]
        if command.redo:
            command.redo()

        self.undo_index += 1

        # Redo other commands on the stack up until the next sentinel.
        while self.undo_index + 1 < len( self.stack ):
            command = self.stack[ self.undo_index + 1 ]

            if command == self.group_sentinel:
                break

            if command.redo:
                command.redo()

            self.undo_index += 1

        self.send_descriptions()

    def send_descriptions( self ):
        next_undo_description = None
        description = None
        index = self.undo_index

        # Search backwards for an undo description to use. Stop as soon as we
        # find one, so the last valid description in the group is used.
        while index >= 0 and self.stack[ index ] != self.group_sentinel:
            description = self.stack[ index ].description

            # Since selections often occur as mere side-effects of other
            # operations, don't use any description containing the string
            # "selected" unless there are no other command descriptions to
            # use.
            if description and "selected" not in description.lower():
                next_undo_description = description
                break

            index -= 1

        if next_undo_description is None:
            next_undo_description = description

        next_redo_description = None
        description = None
        index = self.undo_index + 1

        # Advance past the start group sentinel.
        if index < len( self.stack ) and \
           self.stack[ index ] == self.group_sentinel:
            index += 1

        # Search forwards for a redo description to use. Don't stop until
        # we're at the end of the command group (or the end of the stack),
        # so the last valid description in group is used.
        while index < len( self.stack ) and \
              self.stack[ index ] != self.group_sentinel:
            description = self.stack[ index ].description

            if description and "selected" not in description.lower():
                next_redo_description = description

            index += 1

        if next_redo_description is None:
            next_redo_description = description

        self.outbox.send(
            request = "undo_updated",
            next_undo_description = next_undo_description,
            next_redo_description = next_redo_description,
        )
