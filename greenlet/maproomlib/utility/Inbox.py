import logging
import traceback
import threading
import collections
from Scheduler import Scheduler, Timeout_error


class Receive_error( Exception ):
    """
    An error occuring when attempting to receive a message with
    :meth:`Inbox.receive()`.

    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Inbox:
    """
    An incoming mailbox for receiving messages. Often used in conjuction
    with :class:`maproomlib.utility.Outbox`.
    """
    MESSAGE_WARNING_THRESHOLD = 200

    def __init__( self, leak_warnings = True ):
        self.leak_warnings = leak_warnings
        self.messages = collections.deque()
        self.waiting = None
        self.waiting_scheduler = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger( __name__ )

    def send( self, **message ):
        """
        Put a message into this inbox. If a receiver is waiting to
        :meth:`receive()` from this inbox, wake it up. This method is
        thread-safe.

        :param message: message contents
        :type message: dict
        """
        try:
            self.lock.acquire()
            self.messages.appendleft( message )

            waiting = self.waiting
            waiting_scheduler = self.waiting_scheduler
        finally:
            self.lock.release()

        if self.leak_warnings and \
           len( self.messages ) == self.MESSAGE_WARNING_THRESHOLD: # pragma: no cover
            self.logger.warning(
                "Inbox has %d messages. Potential message leak.\n" %
                len( self.messages ) +
                "Last message contents:\n%s\n" % message +
                "Current stack:\n%s" % "".join( traceback.format_stack() )
            )

        # Wake up the waiting task, if any. Use the task's own scheduler in
        # case it's different than the calling task's scheduler.
        if waiting and waiting_scheduler:
            waiting_scheduler.wake( waiting )

    def receive( self, scheduler = None, timeout = None, **pattern ):
        """
        Grab a message matching the given pattern and return the message. If
        no such messages are available now, sleep until one is available.
        This method is thread-safe.

        :param scheduler: the scheduler to use when sleeping a task
                          (defaults to the current task's parent)
        :type scheduler: Scheduler or NoneType
        :param timeout: seconds before this call times out (optional)
        :type timeout: float
        :param pattern: keys and corresponding values to look for in desired
                        messages
        :type pattern: { str: object, ... }
        :return: received message
        :rtype: dict

        The ``pattern`` matching works by first ensuring that each key in the
        pattern is present in the message. If a key's value is an iterable
        object, then the corresponding value in a message must equal at least
        one of the elements in the iterable. Otherwise, if the pattern's
        value is a non-iterable object, then the corresponding value in the
        message must equal it.

        If the optional ``timeout`` parameter is provided, then the call to
        :meth:`receive()` will time out after the given number of seconds
        by raising a :class:`Timeout_error` exception.
        """
        if self.waiting is not None:
            raise Receive_error(
                "Cannot receive when a task is already waiting to receive."
            )

        if self.leak_warnings and \
           len( self.messages ) == self.MESSAGE_WARNING_THRESHOLD: # pragma: no cover
            self.logger.warning(
                "Inbox has %d messages. Potential message leak.\n" %
                len( self.messages ) +
                "Last message contents:\n%s\n" % self.messages[ -1 ] +
                "Current stack:\n%s" % "".join( traceback.format_stack() )
            )

        # Messages that don't match the pattern.
        deferrals = collections.deque()

        while True:
            # Sleep the current task until there is an available message.
            self.lock.acquire()
            if len( self.messages ) == 0:
                self.wait( scheduler, timeout )

                # If we've woken up and there are still no messages for some
                # reason, keep waiting.
                if len( self.messages ) == 0:
                    continue # pragma: no cover
            else:
                self.lock.release()

            message = self.messages.pop()

            # Special case for turning a message into a raised exception.
            exception = message.get( "exception" )
            if exception:
                self.messages.extend( deferrals )
                raise exception

            # Determine whether the message matches the pattern.
            for ( key, value ) in pattern.items():
                if key not in message:
                    break

                if hasattr( value, "__iter__" ):
                    if message.get( key ) not in value:
                        break
                elif message.get( key ) != value:
                    break
            else:
                # We've made it through the entire loop without breaking out,
                # and the message matches the whole pattern. So put the
                # deferrals back and return the matching message.
                self.messages.extend( deferrals )

                return message

            # If the message does not match the pattern, then defer the
            # message.
            deferrals.appendleft( message )

    def discard( self, **pattern ):
        """
        Remove and discard all messages currently in the inbox that match the
        given pattern. This method is thread-safe.

        :param pattern: keys and corresponding values to look for in desired
                        messages
        :type pattern: { str: object, ... }

        The ``pattern`` matching works just as with :meth:`receive()`.
        """
        # Messages that don't match the pattern.
        deferrals = collections.deque()

        while len( self.messages ) > 0:
            message = self.messages.pop()

            # Determine whether the message matches the pattern.
            for ( key, value ) in pattern.items():
                if key not in message:
                    break

                if hasattr( value, "__iter__" ):
                    if message.get( key ) not in value:
                        break
                elif message.get( key ) != value:
                    break
            else:
                # We've made it through the entire loop without breaking out,
                # and the message matches the whole pattern. Continue looking
                # for matching messages.
                continue

            # If the message does not match the pattern, then defer the
            # message.
            deferrals.appendleft( message )

        self.messages.extend( deferrals )

    def wait( self, scheduler = None, timeout = None ):
        if scheduler is None: # pragma: no cover
            scheduler = Scheduler.current()

        self.waiting = scheduler.current_task()
        self.waiting_scheduler = scheduler
        self.lock.release()

        try:
            timed_out = scheduler.sleep( self.waiting, timeout = timeout )
        finally:
            self.waiting = None
            self.waiting_scheduler = None

        if timed_out:
            raise Timeout_error()

    def empty( self ):
        """
        Return whether this inbox is empty.

        :return: True if there are no messages in this inbox
        :rtype: bool
        """
        return len( self.messages ) == 0
