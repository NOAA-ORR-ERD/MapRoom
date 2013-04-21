import sys
import os.path
import nose.tools
import maproomlib.utility
from maproomlib.utility.Plugin_loader import Plugin_loader, Load_plugin_error
from maproomlib.utility.Inbox import Inbox
from maproomlib.utility.Standard_paths import user_plugins_dir
from Mock_file import Mock_file
from Mock_imp import Mock_imp
from Mock_scheduler import Mock_scheduler


class Mock_plugins_module:
    def __init__( self, path ):
        self.path = path
        self.__path__ = [ self.path ]


class Mock_builtin_plugins_module( Mock_plugins_module ):
    class My_plugin:
        PLUGIN_TYPE = "testtype"
        def __init__( self, a, b ):
            self.a = a
            self.b = b

    class Other_plugin:
        PLUGIN_TYPE = "testtype"
        def __init__( self, foo ):
            self.foo = foo

    class Yet_another_plugin:
        PLUGIN_TYPE = "anothertype"
        def __init__( self, foo ):
            self.foo = foo

    class Error_plugin:
        PLUGIN_TYPE = "anothertype"
        def __init__( self, Exception_class ):
            raise Exception_class()


TIMEOUT = 0.5


class Test_plugin_loader:
    def setUp( self ):
        self.plugins_module = Mock_builtin_plugins_module( "path" )
        self.environ = {}

        self.imp = Mock_imp()
        self.inbox = Inbox()

        self.loader = Plugin_loader(
            self.plugins_module,
            self.environ,
            Mock_file,
            Mock_file.exists,
            Mock_file.listdir,
            Mock_file.isdir,
            self.imp,
        )

        self.scheduler = Mock_scheduler()

    def tearDown( self ):
        Mock_file.reset()

    def test_get_plugin( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            a = 1,
            b = 2,
        )
        self.loader.run( self.scheduler, run_once = True )

        message = self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "request" ) == "plugin"
        plugin = message.get( "plugin" )
        assert plugin
        assert plugin.__class__.__name__ == "My_plugin"
        assert plugin.a == 1
        assert plugin.b == 2

    def test_get_plugin_with_another_plugin_type( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "anothertype",
            response_box = self.inbox,
            foo = 17,
        )
        self.loader.run( self.scheduler, run_once = True )

        message = self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "request" ) == "plugin"
        plugin = message.get( "plugin" )
        assert plugin
        assert plugin.__class__.__name__ == "Yet_another_plugin"
        assert plugin.foo == 17

    def test_get_plugin_with_skip_call( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            skip_call = True,
        )
        self.loader.run( self.scheduler, run_once = True )

        message = self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "request" ) == "plugin"
        plugin = message.get( "plugin" )
        assert plugin
        assert not hasattr( plugin, "__class__" )
        assert plugin.__name__ in ( "My_plugin", "Other_plugin" )

    def test_get_plugin_with_plugin_name( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            plugin_name = "My_plugin",
            a = 1,
            b = 2,
        )
        self.loader.run( self.scheduler, run_once = True )

        message = self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "request" ) == "plugin"
        plugin = message.get( "plugin" )
        assert plugin
        assert plugin.__class__.__name__ == "My_plugin"
        assert plugin.a == 1
        assert plugin.b == 2

    def test_get_plugin_with_plugin_name_and_skip_call( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            plugin_name = "My_plugin",
            skip_call = True,
            a = 1,
            b = 2,
        )
        self.loader.run( self.scheduler, run_once = True )

        message = self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

        assert message.get( "request" ) == "plugin"
        plugin = message.get( "plugin" )
        assert plugin
        assert not hasattr( plugin, "__class__" )
        assert plugin.__name__ ==  "My_plugin"

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_plugin_name_and_non_matching_parameters( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            plugin_name = "My_plugin",
            a = 1,
            b = 2,
            c = 3,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_plugin_name_without_parameters( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            plugin_name = "My_plugin",
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_exception( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "anothertype",
            response_box = self.inbox,
            Exception_class = KeyError,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( Load_plugin_error )
    def test_get_plugin_with_load_plugin_error( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "anothertype",
            response_box = self.inbox,
            Exception_class = Load_plugin_error,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_unknown_plugin_type( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "unknowntype",
            response_box = self.inbox,
            a = 1,
            b = 2,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_unknown_plugin_name( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            plugin_name = "Unknown_plugin",
            a = 1,
            b = 2,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_with_non_matching_parameters( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
            a = 1,
            b = 2,
            c = 3,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )

    @nose.tools.raises( RuntimeError )
    def test_get_plugin_without_parameters( self ):
        self.loader.inbox.send(
            request = "get_plugin",
            plugin_type = "testtype",
            response_box = self.inbox,
        )
        self.loader.run( self.scheduler, run_once = True )

        self.inbox.receive(
            scheduler = self.scheduler,
            timeout = TIMEOUT,
        )


# The class within a class simulates a class within a module.
MY_PLUGIN_CONTENTS = \
"""
class My_plugin:
    class My_plugin:
        PLUGIN_TYPE = "testtype"
        def __init__( self, a, b ):
            self.a = a
            self.b = b
"""


OTHER_PLUGIN_CONTENTS = \
"""
class Other_plugin:
    class Other_plugin:
        PLUGIN_TYPE = "testtype"
        def __init__( self, foo ):
            self.foo = foo
"""


YET_ANOTHER_PLUGIN_CONTENTS = \
"""
class Yet_another_plugin:
    class Yet_another_plugin:
        PLUGIN_TYPE = "anothertype"
        def __init__( self, foo ):
            self.foo = foo
"""


DIRECTORY_PLUGIN_CONTENTS = \
"""
class Directory_plugin:
    class Directory_plugin:
        PLUGIN_TYPE = "directorytype"
        def __init__( self, foo ):
            self.foo = foo
"""


ERROR_PLUGIN_CONTENTS = \
"""
class Error_plugin:
    class Error_plugin:
        PLUGIN_TYPE = "anothertype"
        def __init__( self, Exception_class ):
            raise Exception_class()
"""


class Test_plugin_loader_with_user_plugins_path( Test_plugin_loader ):
    def setUp( self ):
        self.environ = {
            "APPDATA": "appdata",
            "HOME": "home",
        }

        user_plugins_path = user_plugins_dir( self.environ )

        self.plugins_module = Mock_plugins_module( user_plugins_path )

        my_plugin_filename = os.path.join(
            self.plugins_module.path, "My_plugin.py",
        )
        plugin_file = Mock_file( my_plugin_filename )
        plugin_file.write( MY_PLUGIN_CONTENTS )
        plugin_file.close()

        other_plugin_filename = os.path.join(
            self.plugins_module.path, "Other_plugin.py",
        )
        plugin_file = Mock_file( other_plugin_filename )
        plugin_file.write( OTHER_PLUGIN_CONTENTS )
        plugin_file.close()

        yet_another_plugin_filename = os.path.join(
            self.plugins_module.path, "Yet_another_plugin.py",
        )
        plugin_file = Mock_file( yet_another_plugin_filename )
        plugin_file.write( YET_ANOTHER_PLUGIN_CONTENTS )
        plugin_file.close()

        directory_plugin_filename = os.path.join(
            self.plugins_module.path, "Directory_plugin", "__init__.py",
        )
        plugin_file = Mock_file( directory_plugin_filename )
        plugin_file.write( DIRECTORY_PLUGIN_CONTENTS )
        plugin_file.close()

        fake_directory_plugin_filename = os.path.join(
            self.plugins_module.path, "Just_a_directory", "non_plugin.txt",
        )
        plugin_file = Mock_file( fake_directory_plugin_filename )
        plugin_file.write( "not a plugin!" )
        plugin_file.close()

        error_plugin_filename = os.path.join(
            self.plugins_module.path, "Error_plugin.py",
        )
        plugin_file = Mock_file( error_plugin_filename )
        plugin_file.write( ERROR_PLUGIN_CONTENTS )
        plugin_file.close()

        non_plugin_filename = os.path.join(
            self.plugins_module.path, "non_plugin.txt",
        )
        plugin_file = Mock_file( non_plugin_filename )
        plugin_file.write( "testing 1 2 3" )
        plugin_file.close()

        self.imp = Mock_imp()
        self.inbox = Inbox()

        self.loader = Plugin_loader(
            self.plugins_module,
            self.environ,
            Mock_file,
            Mock_file.exists,
            Mock_file.listdir,
            Mock_file.isdir,
            self.imp,
        )

        self.scheduler = Mock_scheduler()

    def tearDown( self ):
        Mock_file.reset()
