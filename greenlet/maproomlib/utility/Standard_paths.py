import os
import sys


def user_data_dir( environ ):
    """
    Return the path where Maproom user data is stored.
    """
    if sys.platform.startswith( "win" ):
        return os.path.join(
            environ.get( "APPDATA", "" ),
            "Maproom",
        )
    elif sys.platform.startswith( "darwin" ):
        return os.path.join(
            environ.get( "HOME", "" ),
            "Library", "Application Support", "Maproom",
        )

    return os.path.join(
        environ.get( "HOME", "" ),
        ".maproom",
    )


def user_plugins_dir( environ ):
    """
    Return the path from which Maproom user plugins are loaded.
    """
    return os.path.join( user_data_dir( environ ), "plugins" )
