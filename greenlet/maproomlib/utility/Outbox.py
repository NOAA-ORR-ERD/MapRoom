import logging
import traceback


class Outbox:
    """
    An outgoing mailbox for broadcasting messages to multiple subscribers.
    """
    INBOX_WARNING_THRESHOLD = 30

    def __init__( self ):
        self.inboxes = {} # Inbox -> pattern dict
        self.logger = logging.getLogger( __name__ )

    def subscribe( self, inbox, **pattern ):
        """
        Subscribe to the messages sent from this outbox.

        :param inbox: the inbox to subscribe to this outbox
        :type inbox: Inbox
        :param pattern: pattern to filter messages by (cannot be empty)
        :type pattern: dict

        According to the provided pattern, then the subscriber will only
        receive matching messages rather than all broadcast messages. For
        information on how pattern matching works, see
        :meth:`Inbox.receive()`.
        """
        if pattern == {}:
            raise ValueError( "The pattern parameter is required." )

        if inbox not in self.inboxes:
            self.inboxes[ inbox ] = pattern

        if len( self.inboxes ) == self.INBOX_WARNING_THRESHOLD: # pragma: no cover
            self.logger.warning(
                "Outbox has %d subscriptions. Potential subscription leak.\n" %
                len( self.inboxes ) +
                "Current stack:\n%s" % "".join( traceback.format_stack() )
            )

    def unsubscribe( self, inbox ):
        """
        End a subscription to this outbox and stop receiving messages from
        it.

        :param inbox: the inbox to unsubscribe from this outbox
        :type inbox: Inbox
        """
        if inbox not in self.inboxes:
            return

        del( self.inboxes[ inbox ] )

    def send( self, **message ):
        """
        Broadcast a message to all of the subscribers of this outbox whose
        pattern matches the message.

        :param message: message contents
        :type message: dict
        """
        for ( inbox, pattern ) in self.inboxes.items():
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
                # We've made it through the entire loop without breaking
                # out, and the message matches the whole pattern. So send it.
                inbox.send( **message )
