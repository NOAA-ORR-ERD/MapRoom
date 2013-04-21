try:
    import greenlet
except ImportError:
    import py.magic as greenlet

import time
import heapq
import weakref
import threading
import collections


class Timeout_error( Exception ):
    def __init__( self, message = None ):
        Exception.__init__(
            self,
            message,
        )


class Thread_error( Exception ):
    def __init__( self, message = None ):
        Exception.__init__(
            self,
            message or
                "This method was not called from the scheduler's thread.",
        )


class Scheduler:
    """
    A cooperative trampoline scheduler based on the greenlet library. You
    should create at most one scheduler per Python thread.
    """
    thread_local = threading.local()

    def __init__( self, idle_task = True ):
        self.running = set()
        self.sleeping = set()
        self.wakeups = collections.deque()
        self.timers = [] # a heapq priority queue sorted by time
        self.done = False
        self.task = greenlet.greenlet( self.run )
        self.something_to_run = threading.Event()
        self.thread = threading.currentThread()

    def switch( self ):
        """
        Start or continue running the scheduler.
        """
        self.task.switch()

    def run( self ):
        Scheduler.thread_local.scheduler_ref = weakref.ref( self )

        while not self.done:
            # If there's nothing to run, then wait until there is.
            if len( self.running ) == 0 and len( self.timers ) == 0:
                self.something_to_run.wait()
                self.something_to_run.clear()

            while len( self.wakeups ) > 0:
                self.add( self.wakeups.pop() )

            self.run_once()
            self.update_timers()

        Scheduler.thread_local.scheduler_ref = None
        self.running = None
        self.sleeping = None

    def run_once( self ):
        # Make a copy of self.running in case it changes during this loop.
        for task in self.running.copy():
            # If a task was removed from the running list out from under us,
            # then don't run that task.
            if task not in self.running:
                continue

            try:
                result = task.switch( self )
            except KeyboardInterrupt:
                self.done = True
                return

            # if the task is done, remove it
            if task.dead:
                self.remove( task )

            # TODO: do something with the result

    def update_timers( self ):
        if len( self.timers ) == 0:
            return

        now = time.time()

        while len( self.timers ) > 0:
            ( trigger_time, task ) = heapq.heappop( self.timers )

            # If the timer is not scheduled to fire yet, put it back and
            # bail since all subsequent tasks will have a later trigger time.
            if trigger_time > now:
                heapq.heappush( self.timers, ( trigger_time, task ) )
                return

            # Time to wake up!
            self.wake( task )
            task.throw( Timeout_error )

    def add( self, task ):
        """
        Add a task to be run by this scheduler. If the task is sleeping, then
        wake it up.

        :param task: the task to run or a callable to run as a task
        :type task: greenlet.greenlet or callable
        :return: the task that was passed in, or a task wrapping the callable
        :rtype: greenlet.greenlet

        The task's callable should expect this scheduler passed in as its
        only parameter. A typical task will loop repeatedly, performing some
        action each iteration such as receiving and handling incoming
        messages.

        At the end of each iteration, :meth:`scheduler.switch()` should be
        called to yield back to the scheduler. Or, you should call another
        function that itself calls :meth:`scheduler.switch()`, such as
        :meth:`maproomlib.utility.Inbox.receive()`.

        Example usage::

            def function( scheduler ):
                while True:
                    do_something()
                    scheduler.switch()

            scheduler.add( function )

        If a running task returns, then it will be considered dead and removed
        from the scheduler.

        Note that this method should only be called from the same Python
        thread that the scheduler is itself running in. Doing otherwise will
        result in a :class:`Thread_error` exception.
        """
        if threading.currentThread() != self.thread:
            raise Thread_error()

        if callable( task ):
            task = greenlet.greenlet( task )
        else:
            # If there are any timers scheduled to wake up this task, stop
            # them because the task is already awake.
            timers = [
                timer for timer in self.timers if timer[ 1 ] != task
            ]
            heapq.heapify( timers )
            self.timers = timers

        self.sleeping.discard( task )
        self.running.add( task )
        self.something_to_run.set()

        return task

    def remove( self, task ):
        """
        Remove a task from this scheduler so that it is no longer run nor
        sleeping.

        :param task: the task to remove
        :type task: greenlet.greenlet
        """
        if threading.currentThread() != self.thread:
            raise Thread_error()

        self.sleeping.discard( task )
        self.running.discard( task )

    def sleep( self, task = None, timeout = None ):
        """
        Put a task to sleep so that it is no longer actively run. If no task
        is given, then the current task is put to sleep.

        :param task: the task to put to sleep (or None for the current task)
        :type task: greenlet.greenlet (or NoneType)
        :param timeout: seconds before this task is woken up (optional)
        :type timeout: float
        :return: True if a timeout occurred, False if sleep was interrupted
        :rtype: bool

        If the optional ``timeout`` parameter is provided, then the call to
        :meth:`sleep()` will time out after the given number of seconds by
        raising a :class:`Timeout_error` exception within the slept task.
        """
        if threading.currentThread() != self.thread:
            raise Thread_error()

        if task is None:
            task = greenlet.greenlet.getcurrent()

        self.running.discard( task )
        self.sleeping.add( task )

        if timeout:
            now = time.time()
            heapq.heappush( self.timers, ( now + timeout, task ) )

        try:
            self.switch()
        except Timeout_error:
            return True
        except:
            # If some exception occurred and woke up the task before its timer
            # fired, then remove the timer so it doesn't fire later.
            if timeout:
                timers = [
                    timer for timer in self.timers if timer[ 1 ] != task
                ]
                heapq.heapify( timers )
                self.timers = timers
            raise

        return False

    def wake( self, task ):
        """
        Wake up a sleeping task so that it is actively run by this scheduler.

        :param task: the task to wake up
        :type task: greenlet.greenlet
        """
        if task not in self.sleeping:
            return

        self.wakeups.appendleft( task )
        self.something_to_run.set()

    def running_alone( self ):
        """
        Return whether the current task is running alone in the scheduler
        without any other running tasks or pending timers.

        :return: True if the current task is alone
        :rtype: bool
        """
        return len( self.running ) <= 1 and len( self.timers ) == 0

    def current_task( self ):
        """
        Return the currently running task.
        """
        return greenlet.greenlet.getcurrent()

    @staticmethod
    def current():
        """
        Return the currently running :class:`Scheduler` in this Python thread,
        or None if there isn't one.
        """
        if not hasattr( Scheduler.thread_local, "scheduler_ref" ):
            return None

        return Scheduler.thread_local.scheduler_ref()

    def shutdown( self ):
        """
        Stop the scheduler so that it no longer runs.
        """
        if threading.currentThread() != self.thread:
            raise Thread_error()

        self.done = True
        self.switch()
