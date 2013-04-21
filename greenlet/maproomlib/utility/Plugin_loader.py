import os
import os.path
import imp
import logging
import traceback
from Inbox import Inbox
from Outbox import Outbox
from Standard_paths import user_plugins_dir


class Load_plugin_error( Exception ):
    """ 
    An error occuring when attempting to load a plugin. Only raise this error
    when you want to prevent any subsequent plugins from attempting to load.
    This is useful, for instance, in a file loading plugin when you know the
    file is of the correct type, but is otherwise invalid or corrupt.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Plugin_loader:
    """
    Responsible for loading plugins and making them available to the
    application. Loads both built-in plugins and any user plugins found in
    ``$HOME/.maproom/plugins/`` or Linux, ``$APPDATA/Maproom/plugins/`` on
    Windows, and ``$HOME/Library/Application Support/Maproom/`` on Mac OS X.

    :param plugins_module: imported module of built-in plugins
    :type plugins_module: module
    :param environ: os.environ or similar (defaults to os.environ)
    :type environ: dict or NoneType
    :param file: standard Python file class or similar (defaults to file)
    :type file: object or NoneType
    :param exists: os.path.exists or similar (defaults to os.path.exists)
    :type exists: callable or NoneType
    :param listdir: os.listdir or similar (defaults to os.listdir)
    :type listdir: callable or NoneType
    :param isdir: os.path.isdir or similar (defaults to os.path.isdir)
    :type isdir: callable or NoneType
    :param imp: Python imp module or similar (defaults to imp)
    :type imp: object or NoneType

    .. function:: get_plugin( plugin_type, response_box, plugin_name = None, **params )

        When a message with ``request = "get_plugin"`` is received within the
        :attr:`inbox`, a handler attempts to instantiate a matching plugin and
        send it to the given ``response_box``.

        :param plugin_type: name of the desired plugin type
        :type plugin_type: str
        :param response_box: response message sent here
        :type response_box: Inbox
        :param plugin_name: name of a specific plugin if a particular plugin
            is desired (optional)
        :type plugin_name: str
        :param params: parameters sent to the plugin
        :type params: dict

        Getting a plugin works as follows. It is assumed that each plugin
        module contains a callable (such as a class or a function) of the
        same name as the plugin module itself. For instance, if the plugin
        is in a file called ``My_plugin.py``, then getting a plugin will
        look for a callable in that module called :attr:`My_plugin`.

        For each loaded plugin of the specified type and matching the given
        name (if any), its callable is invoked with the given parameters. If
        an exception is raised, for instance if the callable has no idea how
        to handle those parameters, then the next plugin's callable will be
        invoked, and so on. As soon as a callable is invoked without an
        exception, its return value is sent as a message as follows::

            response_box.send( request = "plugin", plugin )

        If a plugin raises its own :class:`Load_plugin_error` when its
        callable is invoked, then no other plugin loading attempts will be
        made, and the exception will be sent as a message. This is a useful
        when a plugin recognizes the given parameters, but needs to indicate
        that they are invalid or otherwise not capable of being used. This is
        also a way for the plugin to provide a textual message within its
        raised :class:`Load_plugin_error` to indicate what went wrong.

        If there is no plugin that can handle the given parameters, then
        a generic :class:`Load_plugin_error` is sent as a message.
    """
    def __init__( self, plugins_module, environ = None, file = file,
                  exists = os.path.exists, listdir = os.listdir,
                  isdir = os.path.isdir, imp = imp ):
        if environ is None: # pragma: no cover
            environ = os.environ

        self.file = file
        self.exists = exists
        self.listdir = listdir
        self.isdir = isdir
        self.imp = imp

        self.inbox = Inbox()
        self.outbox = Outbox()
        self.plugins = {}  # map from plugin type to list of plugins
        self.logger = logging.getLogger( __name__ )

        self.load_builtin_plugins( plugins_module )

        user_plugins_path = user_plugins_dir( environ )
        if exists( user_plugins_path ):
            self.load_path( user_plugins_path )

    def run( self, scheduler, run_once = False ):
        while True:
            message = self.inbox.receive( request = "get_plugin" )
            message.pop( "request" )
            self.get_plugin( **message )

            if run_once: break

    def load_builtin_plugins( self, plugins_module ):
        for plugin_name in dir( plugins_module ):
            if plugin_name.startswith( "__" ):
                continue

            plugin = getattr( plugins_module, plugin_name )
            if not hasattr( plugin, "PLUGIN_TYPE" ):
                continue

            if plugin.PLUGIN_TYPE in self.plugins:
                self.plugins[ plugin.PLUGIN_TYPE ].append( plugin )
            else:
                self.plugins[ plugin.PLUGIN_TYPE ] = [ plugin ]

    def load_path( self, dir_path ):
        """
        Load all plugins found within the given directory, thereby making
        them available for use.

        :param dir_path: full pathname of the plugin directory to load
        :type dir_path: str
        """
        for plugin_filename in self.listdir( dir_path ):
            plugin_path = os.path.join( dir_path, plugin_filename )

            if self.isdir( os.path.join( dir_path, plugin_filename ) ):
                plugin_name = plugin_filename
                plugin_path = os.path.join( plugin_path, "__init__.py" )

                if not self.exists( plugin_path ):
                    continue
            else:
                if not plugin_filename.endswith( ".py" ) or \
                       plugin_filename == "__init__.py":
                    continue

                plugin_name = plugin_filename[ : -3 ]

            plugin_file = self.file( plugin_path )

            try:
                module = self.imp.load_module(
                    plugin_name,
                    plugin_file,
                    plugin_path,
                    imp.get_suffixes()[ 2 ],
                )

                callable = getattr( module, plugin_name )

                if callable.PLUGIN_TYPE in self.plugins:
                    self.plugins[ callable.PLUGIN_TYPE ].append( callable )
                else:
                    self.plugins[ callable.PLUGIN_TYPE ] = [ callable ]
            finally:
                plugin_file.close()

    def get_plugin( self, plugin_type, response_box, plugin_name = None,
                    skip_call = False, **params ):
        pending_logs = []

        try:
            for plugin in self.plugins.get( plugin_type, () ):
                if plugin_name and plugin.__name__ != plugin_name:
                    continue

                try:
                    pending_logs.append(
                        "Attempting to load %s (%s)" %
                        ( plugin.__name__, plugin_type )
                    )

                    response_box.send(
                        request = "plugin",
                        plugin = skip_call and plugin or plugin( **params ),
                    )

                    # Found a class willing to handle the request, so we're done.
                    pending_logs = []
                    self.logger.debug(
                        "Loaded %s (%s)" %
                        ( plugin.__name__, plugin_type )
                    )
                    return
                except Load_plugin_error, error:
                    pending_logs.append( traceback.format_exc() )
                    response_box.send(
                        request = "plugin",
                        exception = error,
                    )
                    return
                except:
                    pending_logs.append( traceback.format_exc() )
        finally:
            for log in pending_logs:
                self.logger.debug( log )

        of_type = 'of type "%s"' % plugin_type
        with_name = ""
        if plugin_name is not None:
            with_name = ' with name "%s"' % plugin_name

        response_box.send(
            request = "plugin",
            exception = RuntimeError(
                "No plugin %s%s could be loaded." % ( of_type, with_name ),
            ),
        )
