import wx
import maproomlib.utility as utility


class Wx_inbox( utility.Inbox ):
    """
    A wx-aware variant of :class:`maproomlib.utility.Inbox` that issues
    a wx event every time a message is sent. This is intended to be used in
    conjuction with :class:`maproomlib.ui.Scheduled_application`.
    """
    def send( self, **message ):
        utility.Inbox.send( self, **message )

        # Send a "fake" event to convince wx to return from its MainLoop
        # Dispatch() function. This allows any task waiting for this
        # incoming message to be scheduled.
        wx.PostEvent( wx.GetApp().GetTopWindow(), wx.PyEvent() )
