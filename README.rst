=========
MapRoom 5
=========

A product of the Emergency Response Division of the `NOAA <http://www.noaa.gov/>`_ `Office of
Response and Restoration <http://response.restoration.noaa.gov/>`_.
Visit the `Response And Restoration Blog
<https://usresponserestoration.wordpress.com/>`_ to see some `examples of
MapRoom <https://usresponserestoration.wordpress.com/2015/12/16/on-the-hunt-for-shipping-containers-lost-off-california-coast/>`_
in use.

The newest versions for Mac OS X and Windows are on the `download page <https://gitlab.orr.noaa.gov/erd/MapRoom/wikis/downloads>`_.


Installation
============

Package management is through conda. Download the
`Miniconda binary installer <http://conda.pydata.org/miniconda.html>`_ and run it
to install the base environment. On Windows, open an **Anaconda Prompt** from the start menu.

Configure the conda global environment with::

    conda config --add channels conda-forge
    conda config --add channels NOAA-ORR-ERD
    conda install conda-build

Create and populate the MapRoom virtual environment with::

    conda create --name maproom-test python=3.8
    activate maproom-test

Install the dependencies that conda can install::

    conda install numpy pillow pytest-cov cython docutils markdown requests configobj netcdf4 reportlab python-dateutil gdal pyproj shapely pyopengl wxpython owslib scipy pyugrid

and the dependencies that aren't in conda::

    pip install sawx omnivore-framework

Additionally, on MacOS only, install the ``pythonw`` command that allows programs to use GUI frameworks (like wxPython)::

    conda install python.app


Usage
=====

Install the source code if you have not already::

    git clone git@gitlab.orr.noaa.gov:erd/MapRoom.git
    cd maproom

To develop, MapRoom must be installed as it uses entry points than must be registered with
the python interpreter::

    python setup.py develop

and then run MapRoom by::

    python maproom TestData/Verdat/000011.verdat


Building redistributable versions
=================================

MapRoom uses pyinstaller to build standalone/redistributable binary versions.

I (Rob McMullen) have not yet been successful creating pyinstaller bundles
using conda. I have been able to build pyinstaller bundles using pip virtual
environments, but this requires some by-hand building of some major
dependencies: GEOS and GDAL. There are notes on the wiki for both MacOS and
Windows:

* https://gitlab.orr.noaa.gov/erd/MapRoom/-/wikis/dev/MacOS:-building-app-bundles-without-conda
* https://gitlab.orr.noaa.gov/erd/MapRoom/-/wikis/dev/Windows-10:-building-app-bundles-without-conda

There is a script in the ``maproom/pyinstaller`` directory called
``build_pyinstaller.py`` that includes the configuration data to create a
bundle. On a non-conda install, this creates a working app bundle.

On a conda install, the operation to create the bundle completes successfully
and creates the application ``maproom/pyinstaller/dist/MapRoom_build.app``.
However, running this fails with a crash dialog box.

Trying to run executable the unpacked version in the
``maproom/pyinstaller/dist/MapRoom_build`` directory results in::

    $ ./MapRoom_build
    This program needs access to the screen. Please run with a
    Framework build of python, and only when you are logged in
    on the main display of your Mac.

Even using the ``--windowed`` flag to pyinstaller results in this same error.

Some references:

* https://github.com/chriskiehl/Gooey/issues/259

Debugging pyinstaller problems is very tedious, as it is difficult to get
error messages. On a non-conda install, running the application out of the
build folder would send error messages to the screen, but on a conda install
it doesn't get far enough because it can't seem to do the equivalent of the
the pythonw command.


Top-Level Directory Layout
=================================

TestData
------------------------

This directory contains many subdirectories of example data; see the
00-README.txt in that directory for more details.

libmaproom
-------------------

All the C and Cython code is contained in this directory, and it can be
compiled with a single call to ``python setup.py develop`` inside this
directory.


maproom
---------------

The pure-python code for MapRoom is contained in this directory.


tests
----------

The test framework, using ``pytest``, is contained in this directory. Changing
to this directory and running ``py.test`` will execute the test suite.

scripts
----------

* ``maproom.py`` -- The main script used to start MapRoom

* ``setup.py`` -- python install script

* ``release-and-tag-new-version.sh`` -- helper script to create a new version
  of the code (updating the version number in the code, tagging a new version
  in git, creating a ChangeLog, and building a new version using pyinstaller).
  This would be run on MacOS to create the new version number & tag it in git,
  and then the resulting repository would be checked out on a Windows machine
  to use pyinstaller there to create the executable for Windows.

* ``make-changelog.py`` -- helper script used by the above script to generate
  and update the ChangeLog


==========================
Project Architecture
==========================

A MapRoom project file represents graphic items on a lat/lon grid that can
create a product suitable for printing or display, representing spill data and
text & graphics showing predictions of future impacts of the spill.

The MapRoom program is a user interface to create this graphic file. Graphic
elements are divided into layers of the same types of elements. Only one layer
can be edited at a time, and the user interface changes depending on the
selected layer. The toolbar only shows tools available for the currently
selected layer.


Code Architecture - maproom/third_party
============================================

These are packages that aren't in conda or PyPI.

glsvg
------

glsvg is cloned from https://github.com/fathat/glsvg.git

The reason it is included here is the PyPI package is very outdated and only
for python 2, and the python 3 support is only on the github page. In addition
to the python 3 changes, there are minor modifications to the import
statements allowing the package to be called with "maproom.third_party.glsvg"
instead of only "glsvg".


post_gnome
------------------

post_gnome is a subdirectory of GnomeTools, which is from
https://github.com/NOAA-ORR-ERD/GnomeTools

It is only included here because there is no package on PyPI or Conda for it;
it is not modified.



Code Architecture - libmaproom
===================================

The libmaproom directory contains a separate python package that includes all
the Cython and C code used by MapRoom. There are 6 modules, 4 of which are
used directly by MapRoom to help accelerate rendering. The other 2 are
standalone modules for accelerating specific tasks: pytriangle for creating
triangular meshes, and py_contour for creating contours of particle layers.

libmaproom/libmaproom/*.pyx files
--------------------------------------

The 4 Cython files (.pyx) are helpers for OpenGL rendering.

libmaproom/libmaproom/py_contour/
--------------------------------------

This is a copy of the py_contour code found `here
<https://github.com/NOAA-ORR-ERD/py_contour>`_. There are no changes to the
code, it is just included here to streamline the install and development
process.

libmaproom/libmaproom/pytriangle-1.6.1/
-------------------------------------------

This is an implementation of `Richard Shewchuk's Triangle library
<http://www.cs.cmu.edu/~quake/triangle.html>`_ that is used for mesh
generation. It is Cython code, consisting of a Cython file
``libmaproom/libmaproom/pytriangle-1.6.1/src/triangle.pyx`` and Shewchuk's
original C source in the ``libmaproom/libmaproom/pytriangle-1.6.1/triangle/``
directory.

The ``triangle.pyx`` file is divided into two python functions, where
``triangulate_simple`` is the function designed to be called from user code,
where it uses the Multiprocess package to call ``triangulate_simple_child``
(which wraps the Shewchuk C code). If the C code were not run in another
process, it could kill the entire program as the C code uses the ``exit()``
system call.




Code Architecture - MapRoom Application Framework & File Loading
=====================================================================

The MapRoom program is started using the ``maproom.py`` script in the top
level directory. It contains the ``MapRoomApp`` class and the ``main``
function that is the driver for the whole program. 

Image Resources
--------------------

The ``get_image_path`` call in the main function is used to determine paths
for icons and other files located within the maproom file hierarchy in the
source distribution, but may be placed in different locations when bundled
using application bundlers like pyinstaller. It can be used to find any type
of file, not just images; for example, there is the concept of templates for
sample data, and a ``template_path`` argument is created using a call to
``get_image_path``.

Icons for toolbars and the About dialog are located in the "maproom/icons" and
"maproom/app_framework/icons" directories. They can be referenced using the
"icon://" URI when passed to the filesystem utility
``maproom.app_framework.filesystem.fsopen``.

Template Resources
----------------------

The default project files and some sample data files are stored in the
"maproom/templates" directory. These can be referenced by the "template://"
URI prefix when using the ``maproom.app_framework.filesystem.fsopen``
function. For example, the default project loaded when MapRoom is started is
the  file "maproom/templates/default_project.maproom" and referenced by
"template://default_project.maproom" in the code. That reference is in the
main application class, ``maproom.app_framework.application.MafApplication``,
in the ``default_uri`` class attribute.


Application Init
----------------------

The UI is built using the ``maproom.app_framework`` utilities. Its classes use
the ``Maf`` prefix. It supplies a multi-window interface, where each window
may have multiple tabs. Each tab represents a single project.

The application, ``MafApplication``, wraps the wx.App class. Its ``OnInit``
method sets up some initial data and event handling, but the main application
start occurs in the ``process_command_line_args`` method. This routine is
responsible for creating the first ``MafFrame`` window.

If no file is not specified on the command line, a default project will be
used. The command line supports loading a project file or a layer file; if a
layer only is specified, a default project will be created and the layer
loaded into that project. The ``MafFrame.load_file`` method is used to load
whichever type of file is specified, and once loaded the frame will be created
and displayed.


File Identification and Load
--------------------------------

Projects or files are loaded using the ``MafFrame.load_file`` method. The file
is identified through the ``maproom.app_framework.loader.identify_file``
routine to determine the loader that can parse the data, and the loader
creates the layers that are used for display.

At the start of the ``identify_file`` routine, the file data is loaded into
``maproom.app_framework.loader.FileGuess`` class instance that supplies
convenience functions for accessing the data in the file. It then loops over
every loader to find the best match. Each loader can test the data using
convenience methods of the FileGuess class without having to read the file
over again.

Loaders are registered as setuptools plugins with the entry point
"maproom.app_framework.loaders". Loaders are modules that implement a
module-level function called ``identify_loader`` that returns a dictionary
containing the MIME type and loader class that can handle the file, or None if
the loader can't handle that file.

The ``identify_file`` routine returns the "best" loader if an exact match is
found, or tries to supply a generic loader as a fallback.

At this point, the code is back in the ``MafFrame.load_file`` method with a
dictionary called ``file_metadata`` containing the loader class and the
FileGuess object. Here is where the difference between a project load and a
layer load is handled: if the attempted load is a project, the call to
``MafEditor.can_load_file`` will return false and a new project will be
created. If the file to be loaded can be represented as single layer (or group
of layers under a single layer like a NetCDF particles file), the layer will
be added to the current project.

``MafFrame.load_file`` contains a call to the function ``identify_document``
with the file metadata as an argument. It returns a document class that is
then used to create an editing window in a new tab of the frame. The framework
supports different types of documents with different editing UI elements for
each type. For example, the MapRoom graphic editor for MapRoom documents, a
text editor for text documents, etc. This is a layer of abstraction that
allows different viewers in each tab of the frame. It is largely unused in the
current version, but the idea whas that different editors could operate on
different types of documents, in different tabs in the same frame.


Document Identification
-------------------------------------

There is a distinction between documents and files because it is possible to
have different ways to view and edit the same type of file. For example, a
text file could be edited as a list of x, y points but that same file could
also be displayed as a set of particles in a MapRoom project. The document
provides the interface to access the data in a file. It is still possible that
different viewers use the same type of document; for instance, an HTML viewer
and a text editor use the same text document.

A ``MafDocument`` is the data container that is shown in an individual tab on
the user interface. The view of the data is supplied by the ``MafEditor``
class, which will be described in the next section. The framework is capable
of handling multiple document types, registered as setuptools plugins with the
entry point "maproom.app_framework.documents". Modules must supply at least
one ``MafDocument`` subclass. Each subclass must implement one or both of the
``can_load_file_exact`` and ``can_load_file_generic``, and return a boolean to
indicate if the document can load the file as specified by the
``file_metadata`` argument passed into the method. The
``maproom.app_framework.documents.text.TextDocument`` is a sample document
type that holds text data and can be viewed as HTML or plain text depending on
the format of the file. The title screen of MapRoom is an HTML document,
although the title screen is now not typically viewed since the change was
made to load an empty document at startup.

``maproom.layer_manager.LayerManager`` is the document class used to represent
a MapRoom project. More details on the inner workings of the ``LayerManager``
class below; but in summary, this class keeps references to all layers, the
stacking order, and any relationships between layers.

Once the document type is identified, the ``MafFrame.add_document`` method is
called in order to create a new editor tab for the specified document.


Editor Identification
----------------------------

The class ``maproom.app_framework.editor.MafEditor`` is the base class for the
user interface that is presented by a tab in the top level frames. It may be a
read-only viewer of a document, or it may provide both viewing and editing of
the document.

The ``maproom.app_framework.editor.MafEditor.find_editor_class_for_document``
module-level function searches through the list of available editors to find
the best match for the specified document. Editors are also setuptools
plugins, registered under the entry point "maproom.app_framework.editors".
Each plugin must provide at least one subclass of ``MafEditor``, and each
subclass must implement one or both of the class methods
``can_edit_document_exact`` or ``can_edit_document_generic``. The exact
matches are attempted first, so if an editor is a specific match for the
document (by MIME type provided in the document metadata, or by examining more
specific data in the document itself), those matches will happen before any
generic matches are considered.

Once an editor class is determined, the new tab is created in the frame and
the UI for the editor is instantiated. This happens in the
``maproom.app_framework.frame.MafFrame.add_editor`` method.


Code Architecture - Commands and the Undo Stack
===========================================================

MapRoom provides unlimited undo/redo capability through the
``maproom.command.UndoStack`` object created in the initialization of the
LayerManager. Each change to the document is recorded in a
``maproom.command.Command`` object, and recorded in the UndoStack. Each
command must include a way to restore the LayerManager to the previous state,
providing the undo capability.

The ``maproom.command.Command`` class is subclassed to provide individual
commands. There are 4 modules in the code that contain the available commands:

* menu_commands.py
* mouse_commands.py
* screen_object_commands.py
* vector_object_commands.py

A command is required to implement 3 methods: ``__init__``, ``perform``, and
``undo``. The ``perform`` method is used to make the change and the ``undo``
is used to revert the action.

Other features of commands are available, like coalescing commands. If two of
the same command are applied in a row, it is possible to combine them into a
single command such that only one command appears in the undo list. Commands
like viewport movement are coalesced so that each mouse movement isn't
recorded in a separate command.

There is another partially-implemented feature where commands could be
serialized into a text file and (theoretically) replayed to recreate the list
of commands. This capability is incomplete, but was planned and partially
implemented. It has, however, not been tested in quite a while. The
serialization of commands is mostly automated by a class attribute called
``serialize_order`` containing a list of the object instance attribute to save
and the type of data. The serialization of each of the data types is held in
the ``maproom.serializer`` module, so if new types are needed the
serialization code should be added in that module.

Command Initialization - __init__ method
---------------------------------------------

Each Command subclass can take its own argument list; the superclass __init__
method stores the layer as a layer invariant so that a reference to the Layer
object is not held with the Command object. This becomes important when
deleting a layer so that an old layer (with potentially a lot of memory) isn't
kept around. Deleting layers then restoring them will result in a new Layer
object reconstructed from the data in the Command object, not by restoring
references to the deleted layer.

Any data needed to perform the action should be stored in instance attributes
in the __init__ method.

Performing an Action - perform method
-----------------------------------------

Any change to the MapRoom project must happen in the perform method of a
Command. This complicates the code quite a bit, because instead of changing
the project where it happens in the UI code, the UI code must instead create a
Command object and then call the
``maproom.editor.ProjectEditor.process_command`` method. This will attempt the
operation, and if successful will record the command to the undo/redo
framework. If the operation fails, an error message will be generated. Raising
a ``maproom.errors.MapRoomError`` in the perform method is the way to report
an error.

There is a special subclass of ``MapRoomError`` called ``PointsError`` that
includes an extra argument called  ``points`` that will cause the editor to
highlight the points included in that list as the error conditions.

The perform method of a Command must create an ``maproom.command.UndoInfo``
object to hold any additional data necessary to construct the reverted state
should this command being undone.

The UndoInfo object also has a ``flags`` attribute, an instance of the
``maproom.command.CommandStatus`` class, that controls what aspects of the UI
is refreshed after the change. There are several boolean attributes of
``flags`` that can be set and are described in the class, and there is an
additional ``layer_flags`` list that uses a ``LayerStatus`` object that
contains a summary of all changes for each layer that is affected by this
command -- use the ``add_layer_flags`` method of the ``CommandStatus`` object
instead of appending to the ``layer_flags`` list directly.

There is an additional ``data`` attribute of the UndoInfo object that is for
arbitrary data that the ``undo`` method can use to restore the state of the
project.

The undo_info object should be returned at the end of the perform method.

Undoing an Action - undo method
------------------------------------

The state of the project must be restored to a functionally identical state as
before the ``perform`` method was called after the ``undo`` method completes.
Note that it is not necessary to be totally identical; for instance, some
arrays may have been resized to be larger during the ``perform`` method. It is
not necessary to undo that sort of operation -- as long as the working data is
presented as the same, the condition of a layer doesn't have to be identical
to the "before" state.

An undo_info object must be returned at the end of the method that contains
flags showing what has changed so the UI can be updated properly.


Code Architecture - Project Editor and Processing Commands
===========================================================

The ``maproom.editor.ProjectEditor`` is a subclass of the
``maproom.app_framework.editor.MafEditor`` and represents a tab in a top-level
``MafFrame``, which is a subclass of a wxPython Frame.

The ``process_command`` method takes the Command object and makes the change
described in its perform method. Assuming the change is successful, t flags
resulting from it are added to a ``BatchStatus`` object, the idea being that
multiple commands could be performed in a batch and the UI only updated after
all commands completed.

The call to ``perform_batch_flags`` is where the UI actually gets updated.


Code Architecture - Actions, Menu Bar, and Toolbar
===========================================================

The application framework doesn't use the normal wx method of a large if/else
block to decide what to UI function to perform. Rather, it uses a list of
actions for both menubar and toolbar specification, the definitions of which
are stored as class attributes of the ProjectEditor.

Actions are subclasses of the ``maproom.app_framework.action.MafAction`` that
hold the action description, icon, name, and trigger all in one place. There
is also the ability to perform differently if called using a keystroke or as a
UI callback.

There is a further subclass of ``MafAction``, ``maproom.actions.LayerAction``
that includes a convenience method ``perform_on_layer`` that includes the
active layer as an argument. Not all actions will subclass from
``LayerAction`` because not all actions apply to a single layer.

Menu Bar
-----------

Menubars are hierarchical, and are described in the ``menubar_desc`` class
attribute of a ``MafEditor``. Nested lists form sub-menus. The first item in
any nested list is the title of the menu, either the top level menu item if
it's a direct child of the ``menubar_desc`` list, or the sub-menu name if it's
a subsequent child. The following items in the list are the class names of
actions that will appear in sequence in the menu.

The class attribute ``module_search_order`` describes the modules in which
class names will be searched to populate the menubar. For instance, the source
for the ProjectEditor contains::

    menubar_desc = [
        ["File",
            "new_project",
            "new_empty_project",
            ["New Project From Template",
                "new_project_from_template",
            ],
            None,
            "open_file",
        ...
    ]

    module_search_order = ["maproom.actions", "maproom.toolbar", "maproom.app_framework.actions"]

The ``maproom.app_framework.menubar.MenubarDescription`` object is created
from this ``menubar_desc`` list, and stored in the ``menubar`` instance
attribute of the ``MafFrame`` instance. Note that when a new editor is made
active by chosing a different tab to be the active tab, this ``menubar``
instance attribute is updated to use the ``menubar_desc`` of the now-active
tab.

The "new_project" class will be searched for first in the ``maproom.actions``
module, then ``maproom.toolbar``, and finally the
``maproom.app_framework.actions`` module. The class may appear in any one of
the successively more generic modules formed by the name of the action where
it is split by the underscore character. For instance, "new_project" will be
searched for in the following order::

    maproom.actions
    maproom.toolbar
    maproom.app_framework.actions.new_project.py
    maproom.app_framework.actions.new.py
    maproom.app_framework.actions.__init__.py

In this example, it is found in the ``maproom.actions`` module and no further
seaching would be performed. If it had not been found there, the remaining
modules would be attempted. Because ``maproom.app_framework.actions`` has
sub-modules, the additional module searching based on the underscore splitting
would occur.

Toolbar
--------------

The toolbar definition works identically to the menubar, except there is no
hierarchy. A single list is all that is available, for example::

    toolbar_desc = [
        "open_file", "save_file", None, "undo", "redo", None, "copy", "cut", "paste"
    ]

Analogous to the menubar, the toolbar description object
``maproom.app_framework.toolbar.ToolbarDescription`` is stored in the
``MafFrame`` object as the ``toolbar`` instance attribute. This description
object is replaced every time a new tab is made active using the
``toolbar_desc`` list of the editor corresponding to the now-active tab.

Some tools should only be shown depending on the active layer, though, so
there is an additional routine in ProjectEditor called
``update_toolbar_for_mouse_mode`` that appends some additional tools onto the
end of the list that are useful for the active layer. This routine is called
at the end of the ``process_command`` method.

Each layer has a class attribute called ``mouse_mode_toolbar`` that references
a collection of toolbar items in the ``maproom.toolbar`` module. When a new
layer is made active, those toolbar actions listed in the named mouse mode are
appended to the toolbar and the UI is updated.

The toolbar icon is set through a function called ``calc_icon_name`` that
returns a resource name. Icon resources are described above and most are in
the "maproom/app_framework/icons" directory.

Key Bindings
------------------

Keyboard bindings are listed separately from toolbar and menubar descriptions.
Key binding actions may correspond to existing menubar or toolbar actions, or
may not have an equivalent. Either way, the actions are stored in a keybinding
description object and the actions are located in the same way as menubar and
toolbar actions. The description class attribute is a dictionary::

    keybinding_desc = {
        "new_file": "Ctrl+N",
        "open_file": "Ctrl+O",
        "save_file" : "Ctrl+S",
        "save_as" : "Shift+Ctrl+S",
        "cut": "Ctrl+X",
        "copy": "Ctrl+C",
        "paste": "Ctrl+V",
    }

The keybinding description object is stored in the ``keybinding`` instance
attribute of the ``MafFrame`` and is defined in
``maproom.app_framework.keybindings.KeyBindingDescription``.

Binding UI Actions
------------------------

The menubar, toolbar, and keybinding description objects are only created once
at the editor creation tab; that is, when a new tab is created.

The actions are bound to the menubar and toolbar during a call to
``MafFrame.sync_active_tab`` which is called whenever a tab is changed. The
entire mapping of menu ids is thrown out and recreated through this
function. The menubar (and toolbar) description objects have methods called
``sync_with_editor`` that loop through each action and call the
``wx.Menu.Append`` (or ``wx.ToolBar.AddTool``) methods linking an id value
with this action.

A mapping of id value to action is kept in the menubar (or toolbar)
description object called ``valid_id_map``, and the ``wx.EVT_MENU`` is bound
to the ``MafFrame.on_menu`` method. That method looks through first the
menubar then the toolbar description objects for the id value, and if found
calls that action as ``MafAction.perform_as_menu_item``.

Keybinding actions are handled in the ``wx.EVT_CHAR_HOOK`` binding, and if an
id value is found in the keybinding's ``valid_id_map``, the action's
``perform_as_keystroke`` method is called.


Menu Enabling & Disabling
------------------------------

One of the challenges of wxPython menubars and toolbars is efficiently
managing the code to enable or disable menu items depending on the state of
the application. For instance, the "Copy" item in the "Edit" menu should only
be enabled when there is something that can be copied to the clipboard,
otherwise it should remain grayed-out.

There are also dynamic menu items that change appearance or values depending
on the state of the application, including submenus that have the ability to
contain different numbers of menu items (which is discussed in the next
section).

The menu bar needs to be updated periodically in order to reflect these
dynamic updates. The ``wx.EVT_MENU_OPEN`` event is provided by wxPython to
handle this exact case: to update menu state just prior to being displayed.
However, there are platform differences on each of the 3 supported platforms.
A test is performed at the ``MafFrame.__init__`` method and the appropriate
method is bound to the ``wx.EVT_MENU_OPEN`` event.

The ``sync_menubar`` method is called as a result of the wx event handler,
which it turn calls the ``sync_with_editor`` method of the menubar description
object. This loops through each action and calls the
``sync_menu_item_from_editor`` method to determine the enabled/disabled state,
and also the checked state for radio/checkbox items.


Dynamic Submenus
----------------------

Submenus that have a variable number of entries depending on some aspect of
the current project are handled through the same
``sync_menu_item_from_editor`` method of each action.

The ``maproom.app_framework.action.MafListAction`` class is provided for
submenus that can have variable numbers of items. The first time the
``sync_menu_item_from_editor`` method is called, the object will create the
list of items to be contained in the submenu. The method ``calc_list_items``
must be overridden by the subclass to provide the items for the list. The list
does not have to be text items, a method ``calc_name`` is provided to return a
string that will be used as the menu item text.

The ``action_key`` is a text string that represents the specific menu item of
interest -- the root string of the ``action_key`` is the name of the menu
class, and for each menu item in the submenu, an underscore and the text
representation of an integer is appended. This compound action key is used by
the ``get_index`` method to return the position in the list items.

Every time the ``wx.EVT_MENU_OPEN`` event is called, the
``sync_menu_item_from_editor`` method is called to recreate the list of items.
If the items have changed, an
``maproom.app_framework.errors.RecreateDynamicMenuBar`` exception is raised
which causes the entire menu to be rebuilt, thereby creating the new menu that
includes the changed items.

Note that while this is not super efficient because it loops through the
entire menu system, recreating items that possibly don't need to be created,
it has the advantage of requiring a minimal amount of code. Modifying menus in
place would require careful track of identifying menus that were no longer
needed and deleting items from submenus. In practice, the speed of
regenerating menus has not been an issue.


Code Architecture - Layers and Layer Manager
==================================================

The ``maproom.layer_manager.LayerManager`` class is the ``MafDocument``
subclass that represents the MapRoom project. An object of this class holds
all the layers that make up the final image. Each layer is a subclass of the
``maproom.layers.base.Layer`` class.

Layer Manager
--------------------

The ``LayerManager`` object holds the layers in an arbitrarily deep array of
arrays that results in a tree-like structure. Internally, layers are referred
to by a "multi-index", which represents the location in the structure of the
layer. For example, in the source code is the following example: the array ``[
[ a, b ], [c, [ d, e ] ], f, [ g, h ] ]``. The multi_index ``[ 0 ]`` refers to
subtree ``[ a, b ]``, the multi_index ``[ 1, 1, 1 ]`` refers to the leaf
``e``, and the multi_index ``[ 3, 0 ]`` refers to the leaf ``g``.

Layers are also referenced by a unique number called an ``invariant``. This is
an integer used as id that doesn't change when the layer is renamed or
reordered. It gets created when the layer is added to a LayerManager. There
are special values that represent transient layers, the root layer, and other
layers created at project creation time.

There are various methods to find layers by id, multi-index, by layer type,
and by relationship to other layers. Layers must be added through the methods
provided in this class as there are many internal bookkeeping data that must
be updated as layers change.

LayerTreeControl
----------------------

This UI panel contains a tree control that allows the layers to be reordered
through drag-and-drop, thereby updating the LayerManager data structure (by
altering the "multi-index" of any affected layers).


Base Layer
------------

The ``maproom.layers.base.Layer`` abstract class must be subclassed before it
can be added to a LayerManager as a visible layer in the project. An example
of a simple layer is the ``maproom.layers.point.PointLayer`` layer, which
displays only points. A direct subclass is the
``maproom.layers.line.LineLayer`` which displays both points and lines in
files like ``.verdat`` and other "ugrid" file types. It is much more
complicated than the ``PointLayer`` because it includes editing functions:
moving, adding, and deleting points and lines. See the next section for more
information.

All layers use numpy arrays to hold coordinates to be mapped onto the lat/lon
project space. Some layers, like the LineLayer, have large arrays (one row per
point) that must be resized periodically if many points are added. Other
layers, like image layers, only store points for the 4 corners and store the
image data in OpenGL textures.

Annotation layers use the parent class
``maproom.layers.vector_object.VectorObjectLayer`` which is a further
subclasses of the LineLayer. They use the numpy array of points as the
bounding box of the layer, and some layers use additional points to represent
more points within the layer. Discussion of annotation layers is below.

Layers use class attributes to describe many characteristics, as quite a few
don't depend on the actual instance. They are described in comments in the
``maproom.layers.base`` module.

UGrid Layer
---------------

The most simple layer to display lat/lon data is the
``maproom.layers.line.LineLayer", capable of displaying point and lines.
Several file formats support line layers, including:

* Verdat (.verdat); see the ``maproom.loaders.verdat`` module
* NetCDF (.nc), without particle data; see ``maproom.loaders.ugrid``
* text holding rows of lat/lon data; see ``maproom.loaders.text``

