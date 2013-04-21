import logging
from Inbox import Inbox


class Cache_bulk_setter:
    """
    Performs multiple sets into a cache.

    :param cache: cache into which data will be written
    :type cache: maproomlib.utility.Cache or similar
    :param paused: whether setting should start out paused
    :type paused: bool
    """
    def __init__( self, cache, paused = False ):
        self.cache = cache
        self.inbox = Inbox( leak_warnings = False )
        self.control_inbox = Inbox()
        self.paused = paused
        self.write_counter = 0
        self.logger = logging.getLogger( __name__ )

    def run( self, scheduler ):
        while True:
            if self.paused or not self.control_inbox.empty():
                message = self.control_inbox.receive(
                    request = ( "pause", "unpause" ),
                )
                request = message.get( "request" )

                if request == "pause":
                    self.paused = True
                    continue
                elif request == "unpause":
                    self.paused = False

            if self.inbox.empty() and self.write_counter > 0:
                if self.write_counter > 1:
                    self.logger.debug(
                        "Completed %d bulk writes to %s." %
                        ( self.write_counter, self.cache.__class__.__name__ )
                    )
                self.write_counter = 0

            message = self.inbox.receive( request = "set" )

            self.cache.set(
                message.get( "key" ),
                message.get( "value" ),
            )
            self.write_counter += 1

            if self.write_counter == 2:
                self.logger.debug(
                    "Starting bulk writes to %s." %
                    self.cache.__class__.__name__
                )

            scheduler.switch()
