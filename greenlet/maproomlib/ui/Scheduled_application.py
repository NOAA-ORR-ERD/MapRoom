import wx


class Scheduled_application( wx.App ):
    """
    A variant on the standard :class:`wx.App` class that can be scheduled via
    a :class:`maproomlib.utility.Scheduler`.
    """
    def __init__( self, *args, **kwargs ):
        wx.App.__init__( self, *args, **kwargs )
        self.done = False

    def MainLoop( self, scheduler = None ):
        """
        The default :meth:`wx.App.MainLoop()` does not return until the
        application exits, which means that non-UI work must be done in a
        separate thread or somehow hook into the wx idle event.

        In contrast, this variant :meth:`MainLoop()` can be added as a task to
        a :class:`Maproomlib.Utility.Scheduler` instance, thereby allowing
        it to be scheduled like any other task.

        If this is the only task currently running within the scheduler, then
        :meth:`MainLoop()` will block until there is an available wx event.
        This prevents this loop from spinning the CPU when polling for events.

        :param scheduler: the scheduler to schedule this application
        :type scheduler: Maproomlib.Utility.Scheduler or NoneType
        """

        event_loop = wx.EventLoop()
        original_event_loop = wx.EventLoop.GetActive()
        wx.EventLoop.SetActive( event_loop )

        while not self.done:
            while event_loop.Pending():
                event_loop.Dispatch()
                if scheduler:
                    scheduler.switch()

            self.ProcessIdle()

            # If this is the only running task, then consider the scheduler
            # idle. Calling Dispatch() will block until there is a wx event.
            if not self.done and scheduler and scheduler.running_alone():
                event_loop.Dispatch()

            if scheduler:
                scheduler.switch()

        wx.EventLoop.SetActive( original_event_loop )
        scheduler.shutdown()

    def shutdown( self ):
        """
        Close this application and shutdown the underlying scheduler as well.
        """
        self.done = True
