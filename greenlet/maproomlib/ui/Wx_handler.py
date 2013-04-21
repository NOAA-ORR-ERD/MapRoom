import maproomlib.utility as utility


class Wx_handler:
    """
    An instance of :class:`Wx_handler` can serve as a wx event handler,
    transforming each event into a message in the given inbox. The purpose
    of doing this is so that all messages and events for a given task are
    received in a single place, thereby serializing event handling.

    :param inbox: where to put the message created from the event
    :type inbox: Maproomlib.Utility.Inbox
    :param request: request name of the created message
    :type request: str
    :param skip: whether to call event.Skip() before creating the message
    :type skip: bool
    :param message: additional key/value pairs to add to the sent message
    :type message: dict

    Example usage is as follows::

        handler = Wx_handler( inbox, "open_file", skip = True )
        ctrl.Bind( wx.EVT_MENU, handler, id = wx.ID_OPEN )
    """
    def __init__( self, inbox, request, skip = False, **message ):
        self.inbox = inbox
        self.request = request
        self.skip = skip
        self.message = message

    def __call__( self, event ):
        """
        Actually handle the event, add a new message to the inbox, and switch
        back to the scheduler.

        :param event: the event to handle
        :type event: wx.Event
        """
        if self.skip:
            event.Skip()

        self.inbox.send(
            request = self.request,
            event = event.Clone(), # In case the original is destroyed.
            **self.message
        )

        # To improve UI responsiveness, give the recipient of the event a
        # chance to handle it right away.
        utility.Scheduler.current().switch()
