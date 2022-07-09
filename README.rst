=========
MapRoom 5
=========

MapRoom is a desktop application used to make maps to support trajecotry analysis and other support products. It supports a wide variety of base maps (Nautical Charts, assorted WMS and Tile services), The ability to present the results of the `NOAA GNOME Model <https://gnome.orr.noaa.gov>`_, and easy additon of geo-refrences notations.


MapRoom is a a product of the Emergency Response Division of the `NOAA <http://www.noaa.gov/>`_ `Office of
Response and Restoration <http://response.restoration.noaa.gov/>`_.
Visit the Office of Response And Restoration Blog to see some
`examples of MapRoom in use <https://response.restoration.noaa.gov/about/media/hunt-shipping-containers-lost-california-coast.html>`_


The newest versions for Mac OS X and Windows are on the `download page <https://gitlab.orr.noaa.gov/erd/MapRoom/wikis/downloads>`_.

.. toctree:
   :maxdepth: 2


Installation
============

Package management is through conda. Download the
`Miniconda binary installer <http://conda.pydata.org/miniconda.html>`_ and run it
to install the base environment. On Windows, open an **Anaconda Prompt** from the start menu.

Configure the conda global environment with::

    conda config --add channels conda-forge
    conda config --add channels NOAA-ORR-ERD

Create and populate the MapRoom virtual environment with::

    conda create --name maproom-test --file conda_requirements.txt
    activate maproom-test


NOTE: maybe still required ??? ``pip install sawx``

Additionally, on MacOS only, install the ``pythonw`` command that allows programs to use GUI frameworks (like wxPython)::

    conda install python.app


Usage
=====

Install the source code if you have not already::

    git clone git@gitlab.orr.noaa.gov:erd/MapRoom.git
    cd maproom

Build ``libmaproom`` (or install the wheel, if there's an appropriate one there)::

  cd libmaproom
  python setup.py install


To run or work on the code, MapRoom must be installed as it uses entry points than must be registered with the python interpreter::

    python setup.py develop

and then run MapRoom by::

    python maproom.py

or::

    pythonw maproom.py

on the Mac


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


Code Architecture - The Main User Interface
==================================================

The UI is divided into 3 main areas:

1. the menu bar
2. the toolbar
3. the top level frame containing tabbed views of editing windows

The menubar and toolbar are described in a subsequent section.

Each editing window, displayed in a tab in the main frame, is further divided
into 4 sections:

1. the main drawing area showing the map view
2. the left column of 3 panels
3. the vertically oriented popup menu list on the right border of the frame
4. timeline strip on the bottom.

The editing window is defined in ``maproom.editor.ProjectEditor``, a subclass
of ``MafEditor`` and represents a tab in a top-level ``MafFrame``, which is a
subclass of a wxPython Frame.

Main Drawing Area - LayerCanvas
------------------------------------

All the map data, annotations, and other graphical data that appear in layers
are rendered using OpenGL and are controlled by the
``maproom.layer_canvas.LayerCanvas`` object in the main portion of the window,
which is described in a section below. The UI for the frame is created in the
method ``create_layout``. The arrangement of the UI within the frame is
controlled by a tiling layout manager, the
``maproom.app_framework.ui.tilemanager.TileManager``, a custom control that
provides tiling for the main windows, sidebars with popout windows, and a
footer that holds the timeline control.

The ``LayerCanvas`` is more fully described in the OpenGL Rendering section
below.

Top Info Panel - LayerTreeControl
-------------------------------------

The left column of panels includes a tree view showing the stacking order of
layers, a list of layer parameters, and a list containing information about
the currently selected item in the main view.

The top-most panel on the left side of the frame is the
``maproom.layer_tree_control.LayerTreeControl``, a custom tree control
slightly modified from the ``wx.lib.agw.customtreecontrol.CustomTreeCtrl``
class. This UI panel allows the layers to be reordered through drag-and-drop.

The ``LayerTreeControl`` also has an event, contained in an attribute named
``current_layer_changed_event``, that is fired whenever the user selects a new
layer. The tree control is a single selection tree, so changing the selection
makes that layer the current editing layer. Other UI elements can add a method
to the event to get a callback when this happens. This is used for the points
list panel: ``maproom.panes.PointsList`` so that it can update its list using
points from the now-current layer.

The event handling class is ``maproom.app_framework.events.EventHandler``,
which is a small custom class that provides callback mechanisms.

Middle Info Panel - LayerInfoPanel
--------------------------------------

The middle panel on the left side, below the tree control, is the class
``maproom.ui.info_panels.LayerInfoPanel``. This displays information about the
currently selected layer, and provides controls to modify the characteristics
of the layer. The layer characteristics are described in a list of text
strings in the layer's class definition (see the Base Layer section below).

Each of the strings is the name of a field, and is used to create a control in
this info panel. For example, for a UGrid layer, the fields are defined by the
list::

    layer_info_panel = ["Point count", "Line segment count", "Show depth", "Flagged points", "Default depth", "Depth unit", "Color"]

The module ``maproom.ui.info_panels`` contains a large number of classes that
represent UI controls designed to display or modify layer parameters. For
example, the "Point count" field corresponds to a static text display that
shows the user the number of points in the layer and a toggle control that
allows the display or hiding of those points. This control is defined in the
class ``PointVisibilityField``.

"Line segment count" shows a similar control, except the number of line
segments instead of points. It is defined in the ``LineVisibilityField``.

Other fields will have different controls; for example, the "Depth unit" field
contains a drop-down list with a choice of units: "unknown", "meters", "feet",
or "fathoms".

See the docstrings of the ``maproom.ui.info_panels.InfoPanel`` object for more
details on how the controls for the fields are created and managed.

Lower Info Panel - SelectionInfoPanel
-----------------------------------------

This is the bottom panel on the left side and is similar in operation to the
LayerInfoPanel except that is displays data on the currently selected items in
the layer. Using the LineLayer as an example: if no points are selected, the
panel is blank. However, once one or more points are selected, details of the
selection are displayed.

Popup Menu List
--------------------

The right side of the frame contains popup windows that represent extra
information about the layer, or debugging info on the app itself. Hovering the
mouse pointer over one of the names on the list will display a popup dialog
with a data display.


Timeline Panel - TimelinePlaybackPanel
------------------------------------------

This control displays the timesteps available in all particle layers, and
playback controls to step through a visualization of the motion of the
particles.

The timeline itself is the ``maproom.panes.TimelinePanel`` class, subclassed
from the custom control ``maproom.app_framework.ui.zoomruler.ZoomRuler``. The
base class handles the scrolling, zooming, and selecting via the mouse and
uses callback functions to communicate the UI actions.


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

Command Initialization - ``__init__`` method
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

Running a Command - ``perform`` method
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

Undoing a Command - ``undo`` method
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


Processing Commands
--------------------------

The ``process_command`` method of the ``ProjectEditor`` takes the Command
object and makes the change described in its perform method. Assuming the
change is successful, t flags resulting from it are added to a ``BatchStatus``
object, the idea being that multiple commands could be performed in a batch
and the UI only updated after all commands completed.

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


Code Architecture - OpenGL Rendering 
==============================================

For speed, OpenGL is used to render all graphics in the main window. The
advantage of OpenGL is that the graphics card can hold most of the data in its
localized (fast) memory. Only when data changes (adding/deleting a point,
changing a coordinate, adding a line, etc.) does new data have to be loaded
into the graphics card memory.

The PyOpenGL package is used to interface with the operating system's native
OpenGL libraries.

Numpy record arrays are used as a further optimization, defined in
``maproom.renderer.gl.data_types`` for different use cases. For example,
``POINT_VIEW_DTYPE is used `` to access individual x, y, z coordinates
separately, and ``POINT_XY_VIEW_DTYPE is used to access `` the XY values
together. This ``POINT_XY_VIEW_DTYPE`` can be used, for instance, to set the
XY values in the record array from a regular python list of two-tuples.

There are convenience functions to create blank lists of points, lines, and
other items. Notice that ``numpy.NaN`` is used as a placeholder for undefined
values, and the drawing code will skip over those points. Arrays may be
allocated with extra members as buffer at the end so that additions can happen
by overwriting the ``NaN`` values at the end rather than continually
reallocating and resizing the array.

Layer Drawing
----------------------

Layers are drawn in the stacking order shown in the ``LayerTreeControl``
(described below), from the bottom to the top. Any opaque layers, like a WMS
layer, will obscure any layer below it.

Rendering happens in the ``render`` method of
``maproom.renderer.base.BaseCanvas``. The class is subclassed in the
``maproom.renderer.gl_immediate.screen_canvas.ScreenCanvas`` class that
provides the wxPython and OpenGL drawing area. The ``ScreenCanvas`` uses some
optimization and overrides the ``render`` method before calling the
``BaseCanvas.render`` method.

The ``ScreenCanvas`` is further subclassed in ``maproom.layer_canvas`` as the
``LayerCanvas`` object. A ``LayerCanvas`` object is created by the
``ProjectEditor`` main viewer during the instantiation process.

When drawing the screen, the layers are looped over from bottom to top, and
each layer's renderer object is called to draw that layer's contents. Layer
renderer objects are explained in the next section. There is an optional
overlay layer that will always be drawn on the top of the stacking order. The
overlay is used for certain user-interface modes (See the Mouse Handler
section below) like rubberbanding for selecting points.

An entire additional rendering pass is made after the drawing is complete, but
this time it is to create non-visible layer that is used to detect what object
is under the mouse. This is the picker framebuffer, and is described in a
subsequent section.

Layer Renderers
----------------------

Each layer has an object that controls how it is drawn, called the "layer
renderer", created by a call to ``LayerCanvas.new_renderer`` and held in the
dictionary attribute ``layer_renderers`` in the ``LayerCanvas``. It is of the
class ``maproom.renderer.gl_immediate.renderer.ImmediateModeRenderer``.

Any time a layer changes its representation (moving a point, changing a line,
adding or deleting an element), the layer renderer for that layer must be
updated through a call to ``update_renderer``. The usage of the word "update"
is a bit fuzzy, because it is the ``layer_renderers`` dictionary that is
updated; a new ``ImmediateModeRenderer`` object is created and stored in the
dictionary. The previous object referred to in the dictionary is garbage
collected.

The ``ImmediateModeRenderer`` object holds the OpenGL Vertex Buffer Objects
(VBO) for the data in the layer. These VBOs are representations of the data
held on the graphics card, so they must be loaded through calls line
``set_points``,``set_lines``, ``set_polygons`` and others. These routines do
the work of creating the VBOs and, behind the scenes, copy the values to the
graphics card.

It is because of this data transfer to the graphics card that the data types
in ``maproom.renderer.gl.data_types`` are used. They provide access to the raw
layer data in a format that can be easily converted into data that the
PyOpenGL methods need.

The ``ImmediateModeRenderer`` includes many convenience functions for drawing
on the OpenGL canvas. Some examples are: ``draw_points`` to draw small circles
for each non-NaN point in the layer; ``draw_selected_points`` which draws
larger circles for only those points specified in the argument to the
function; ``draw_image`` to draw texture mapped images after the images have
been set up with a call to one of the ``set_image_*`` methods; and many
others.

Note that all of the code here uses the now-deprecated OpenGL Immediate Mode
(hence the name ImmediateModeRenderer!), where OpenGL calls are bookended by
calls to ``glBegin`` and ``glEnd``. Modern OpenGL uses shaders for everything,
and the long term plan was to convert MapRoom to use shaders.

Examples of the usage of the layer renderers will be included in the layer
descriptions below.

Picker
----------------

All layer renderers include a picker object that is only active when
rendering the picker framebuffer.

The picker works by creating a separate pass through the rendering process,
but instead of drawing to the screen, it draws to an off-screen framebuffer.
In order to determine what object is under a specific mouse location, the
off-screen framebuffer stores a unique color value for every object that is
pickable. This color value doesn't relate to its color displayed on the
screen, instead it encodes the layer that it belongs to, the type of graphic
element within that layer, and an identifying number of that graphic element.

For instance, for a ``LineLayer`` (described below), the picker has to deal
with both points and lines. Each point renders to a circle with some radius in
pixels, so each one of those pixels gets assigned a unique color associated
with that point. Similarly, each line is rendered to a set of pixels, and the
color for each of those pixels will uniquely map back to the line on this
layer.

The class ``maproom.renderer.gl_immediate.picker.Picker`` contains this code.
During the second pass through rendering (the picker pass), a new ``Picker``
object is created and the picker colors are determined before each primitive
is drawn. The method ``get_next_color_block`` contains the logic for reserving
colors, and the ``Picker`` object contains the lists that are used to decode
the color value.

Internally, OpenGL uses a 32 bit integer to represent the color in red, green,
blue & alpha (RGBA) format. Because the alpha value allows color blending,
this would mess up the uniqueness of the mapping from color to pickable
object. So, the alpha value is left at zero which leaves 24 bits to map to
pickable objects.

An assumption is made here in the code: the machines will operate in
little-endian mode. Since most current computers are little endian (running on
Intel or AMD 64 bit processors), no code is added to check for big endian
machines. Red is stored in the least significant byte, green in the next, blue
next, and finally alpha in the most significant byte. For the numpy code used
here, the lowest 24 bits encode the color, and the highest 8 bits are alpha.
We must avoid the high 8 bits (we leave them at zero), but still we have 2^24
values, or 16.7 million possible unique color combinations allowing that many
unique objects to be decoded.

As each block of colors is reserved with a call to ``get_next_color_block``,
lists are maintained in order to reverse the mapping of color into layer type,
object type and object number. The method ``get_object_at_mouse_position``
takes the mouse position and reverses out the object info from the 24 bit
color value.


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

File Format
----------------

The ``LayerManager`` can also be considered the representation of the MapRoom
project file. Serialization to and from the project file is handled through
``save_all_zip`` and ``load_all_from_zip``. There is an older JSON-only text
file format accessed through ``load_all_from_json`` that is deprecated.

The zip file format puts each layer in its own directory, and includes
a few special files at the root directory to store additional
information, such as the metadata needed to specify the connections
between layers.

Examining the contents of the default project zip file shows these entries::

    Archive:  blank_project.maproom
     Length   Method    Size  Cmpr  Name
    --------  ------  ------- ----  ----
           2  Defl:N        4 100%  pre json data
        1978  Defl:N      432  78%  post json data
         376  Defl:N      206  45%  1-Graticule/json layer description
         422  Defl:N      215  49%  2-Scale/json layer description
        2180  Defl:N      441  80%  3/0-New Annotation/json layer description
        2180  Defl:N      448  79%  3/1-Rectangle/json layer description
    --------          -------  ---  -------
        7138             1746  76%  6 files

The "pre json data" file is processed before any layers are loaded, and the
"post json data" file is processed after all layers are loaded. Layers
themselves are directories. Directories that have only a number for a name are
folders, named a number plus a dash and a text value are normal layers.

Most layers are described in the file "json layer description". Image layers
will have additional file(s) with the image data.

Layers must be able to convert to and from JSON. They do this through their
``serialize_json`` and ``load_from_json`` methods.

Layer Overview
----------------------

The ``maproom.layers.base.Layer`` abstract class must be subclassed before it
can be added to a LayerManager as a visible layer in the project. An example
of a simple layer is the ``maproom.layers.point.PointLayer`` layer, which
displays only points. A direct subclass is the
``maproom.layers.line.LineLayer`` which displays both points and lines in
files like ``.verdat`` and other "ugrid" file types. It is much more
complicated than the ``PointLayer`` because it includes editing functions:
moving, adding, and deleting points and lines. See the UGrid section below for
more information.

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
``maproom.layers.base`` module. For example, the ``layer_info_panel``
attribute is a list of text identifiers that are used to display controls that
can modify layer characteristics.

Styles
----------

Annotation layers use a style object to hold the colors, line widths, font
sizes, etc. of all the shapes that they draw. There are default styles for
each layer type, and a style dialog to manage these. 

Other layers use the same style object to hold the point and line colors.
However, their styles aren't as customizable. UGrid layers rotate through a
set of colors as a new layer is created; particle layers use colors depending
on characteristics of the particle.

Styles are described in the ``maproom.styles.LayerStyle`` object, and are
serialized into text strings that are saved with the layer JSON data when
saving MapRoom project files.

As new style types were added to the class, backward compatibility was added
so old versions of MapRoom project files can still be loaded.

The ``LayerManager`` keeps a default style object, and as a new layer is
created a copy of this style object is used as the layer's style object. The
layer's style can then be changed without affecting other layers, but all
layers will start with the same styling. The style dialog changes the default
style object and can apply changes to current annotation layer objects.

Bounding Rectangles
----------------------

All layers have a boolean class attribute ``bounded`` which flags whether or
not the layer has finite lat/lon boundaries, or is unbounded. Bounded layers
are defined by an axis-aligned bounding rectangle that specifies lat/lon
coordinates for each corner.

Unbounded layers include the ``maproom.layers.tiles.TileLayer`` that hold the
WMS maps, sticky layers like the ``maproom.layers.title.TitleLayer`` or the
``maproom.layers.scale.Scale`` layer, and graticule layer
``maproom.layers.grid.Graticule``.

Vector object layers that don't scale with the map like the
``maproom.layers.vector_object.OverlayTextObject`` shouldn't technically be
bounded because the borders aren't stuck to 4 lat/lon coordinates. These
layers attach one control point to a lat/lon coordinate and maintain a fixed
size relative to the computer's display. They do not scale as the lat/lon area
is zoomed in or out. However, in the code they are bounded -- the bounds are
recalculated at every zoom to maintain the relative size. The layers were
written this way to be able to leverage the same rendering code and the same
code to use the mouse to move control points. It does lead to complications;
the method ``LayerManager.recalc_overlay_bounds`` is used to recompute the new
bounding box for each overlay layer every time the viewport is updated.

For normal bounded layers, the ``compute_bounding_rect`` method is called. The
``maproom.layers.points_base.PointBaseLayer``, which is the superclass for
most layers that use numpy arrays to store the point values, calculates the
the min/max values of the lat/lon of the set of points describes the bounding
rectangle.

For folder layers that are bounded, each child layer's bounding rectangle must
be calculated first. Once those sets of bounding rectangles, the folder's
bounding box becomes the bounding box of the union of those rectangles.
Because bounded folder layers can be resized, the child layers contained
within may need to be scaled to correspond to the new size. This is
accomplished using the ``set_data_from_bounds`` method on child layers, called
with the new bounding box size of the child folder that allows the child to
scale the location of the points to match the new bounding box location.

Valid Times
-----------------

Particle layers have a time associated with them, as each layer represents the
state of a set of points at a certain point in time.

The concept of time was extended to all layers, so all layers have a
``start_time`` and ``end_time`` value describing the period of time which is
valid to display this layer. Times are stored in floating point seconds, as
converted by the Python library function ``calendar.timegm``.

If the start and end times are zero, the layer is valid at all times.


Layer Rendering
------------------

Layers can either be drawn in projected space (zoomed in relative to the
visible layer on the map), or screen space (fixed relative to the computer
display). Verdat layers are drawn in projected space since they are a set of
lat/lon points plotted on a map. The Scale layer is drawn in screen space
since it always occupies the same position on screen regardless of zoom level.

Layers subclass from either ``maproom.layers.base.ProjectedLayer`` or
``maproom.layers.base.ScreenLayer`` which provides the ``render_projected`` or
``render_screen`` methods that are overridden by the subclass.

The call to ``maproom.layers.base.Layer.render`` handles the call to use
``render_projected`` or ``render_screen``.

Before the first time the layer is drawn or when the internal structure of the
layer changes (generally when items are added or deleted, but **not**
necessary when items are moved), the ``rebuild_renderer`` method is called. A
new, unpopulated ``ImmediateModeRenderer`` instance is passed to the function
allowing this method to call whatever setup is needed to add points, lines,
polygons, or other graphic primitive values.

The ``render_projected`` (or ``render_screen``) method must also be defined
for each layer, which calls methods on the ``ImmediateModeRenderer`` instance
passed into the method.

Renderers are passed into these methods and not stored in the objects for two
reasons:

1. the initial design of MapRoom called for the ability to have multiple tabs
   showing the same MapRoom project at different zoom levels or geographic
   locations.

2. there is the capability to generate PDF images of the current view. This is
   accomplished using the exact same interface: ``rebuild_renderer`` followed
   by ``render``, but this time using an
   ``maproom.renderer.pdf.renderer.ReportLabRenderer`` instance.

Layer Serialization
---------------------

JSON was chosen as the file format in which to save layer data. Some layer
data, like images, is extremely inefficient to save in JSON format, so
additional binary data may be used in some cases.

The ``serialize_json`` method in ``maproom.layers.base.Layer`` is the driver
to convert the layer data to a JSON dictionary. The ``unserialize_json``
method is the reverse: taking the JSON dictionary and repopulating the layer
with the correct data types represented by the JSON text encoding.

There is a simple list of attributes that will be saved for each layer, like
the type, invariant, and name. See the ``serialize_json`` method for the
complete list of simple attributes. Other attributes are marked for inclusion
in the JSON serialization by having a pair of methods in the class for
converting to and from JSON. These methods must be indicated by having the
``_to_json`` and ``_from_json`` strings appended to the attribute name.

For instance, the attribute ``start_time`` (indicating the first valid time
for the layer to appear on the timeline) has the companion methods
``start_time_to_json`` and ``start_time_from_json`` to handle converting the
time value to a JSON string and from the JSON string to a floating point
value, respectively.

Note that JSON is a special text format that is converted upon load to a
python dictionary where the keys are strings and the values can be python
primitives, lists or dictionaries. The ``json_data`` argument passed into the
``*_from_json`` is a python dictionary where the keywords will be the layer
attribute names.

Analogously, when saving to JSON format, MapRoom produces a dictionary that it
then converted to a text file and saved. Numpy values can give the Python
built-in ``json`` module difficulties and returns very vague error messages
claiming that value that looks like a normal floating point can't be
serialized. It usually turns out that this is a numpy value that gets printed
out as a normal looking string due to numpy's str() or repr() method, but is
actually a numpy data type. The ``*_to_json`` methods should return primitive
types (or lists of primitive types) that the ``json`` module will be able to
serialize.

The ``serialize_json`` method automatically scans the class definition for
attributes that have the matching ``_to_json`` and ``_from_json`` methods.
Adding a new attribute to the serialization process simply requires these two
methods. For backward compatibility, it is advised to handle the case where
the ``_from_json`` method is unable to find the value from the JSON encoded
data. For instance, the ``maproom.layers.vector_object.VectorObjectLayer``
base class has an attribute named ``rotation`` and both ``rotation_to_json``
and ``rotation_from_json``. Looking at the the method to read JSON data and
restore the layer value::

    def rotation_from_json(self, json_data):
        # Ignore when rotation isn't present
        try:
            self.rotation = json_data['rotation']
        except KeyError:
            self.rotation = 0.0

it includes the check that sets the rotation value to zero if the keyword
isn't present in the JSON data.

Code Architecture - UGrid Layer
==================================================

The layer ``maproom.layers.line.LineLayer" is capable of displaying point and
lines using lat/lon coordinates. Several file formats support line layers,
including:

* Verdat (.verdat); see the ``maproom.loaders.verdat`` module
* NetCDF (.nc), without particle data; see ``maproom.loaders.ugrid``
* text holding rows of lat/lon data; see ``maproom.loaders.text``

Points and lines are held in numpy record arrays (as explained above) as an
optimization to speed the OpenGL rendering. Both record arrays contain a
``state`` item that reflects a bitfield defined in ``maproom.layers.state``.
Selecting a point or line, for instance, sets the ``state.SELECTED`` bit and
the item will show in the UI with a selected border. The ``FLAGGED`` bit shows
up in the UI as a larger selection border around the time, and in a different
color than the selected state.

When inserting points or lines, the arrays are shifted and corresponding
indexes are updated. At the moment, there is only a routine to insert a single
point, not multiple points at one time. The UI would have to be changed for
more routines to be needed, since it only allows single points to be added at
once. In practice for verdat layers, this has not been a problem as of yet.

Deleting points or lines is similar, with the entire array being copied except
for the items to be removed. However, there is the facility to remove more
than one point or line at the same time, due to the UI allowing multiple
points & lines to be selected.

Lines are stored in the ``line_segment_indexes`` attribute, of type
``maproom.renderer.gl.data_types.LINE_SEGMENT_VIEW``. Lines will be referred
to by the index number into this list. Each line holds the index of the start
and end point, implying that this layer really holds a list of possibly
disjointed line segments. In order to determine if a subset of the line
segments makes a closed boundary, additional tools are needed. The
``maproom.library.Boundary`` module provides the ``Boundaries`` class which
can determine a set of ``Boundary`` objects that correspond to closed
boundaries in the layer.


Code Architecture - Shapefile Layer
==================================================

There are two modules defining polygon layers: the older type in
``maproom.layers.polygon`` and the newer module supporting editable polygons
in ``maproom.layers.shapefile``.

The older module originally was used to display all polygon layers, but is now
only used to display the RNC selection map. Because it is working and debugged
for this purpose, it was not rearchitected into the ``shapefile`` module. The
older module is not documented here; this description is for the newer
``shapefile`` module.

Editable polygons are supported by the ``shapefile`` module in the
``ShapefileLayer``. It is a subclass of the ``PointLayer``, using the points
numpy array to store all the lat/lon coordinates of each point. The polygons
are broken up into rings, each ring having an index in the ``rings`` attribute
which is of the ``POLYGON_DTYPE`` numpy recarray type. The ring descriptions
are stored in the ``point_adjacency_array`` and ``ring_adjacency`` attributes,
of type ``POINT_ADJACENCY_DTYPE`` and ``RING_ADJACENCY_DTYPE``, respectively.
These two arrays are the same size as the points array and track information
about the rings. They are two different arrays for historical reasons, as the
old ``PolygonLayer`` from ``maproom.layers.polygon`` uses the
``POINT_ADJACENCY_DTYPE`` and the drawing code in the
``ImmediateModeRenderer`` uses this array for the OpenGL VBO data.

Rings are defined as a contiguous block of points in the ``points`` array. The
starting point and number of points in the ring are defined in the
``ring_adjacency`` array, which is the same length as the points array. The
ring size is encoded into the ``point_flag`` attribute. The numpy recarray is
defined in ``maproom.renderer.gl.data_types`` as::

    RING_ADJACENCY_DTYPE = np.dtype([  # parallels the points array
        ("point_flag", np.int32),
        ("state", np.int32),
    ])

where the ``state`` is the same selection state bitfield as other data types.
The ``point_flag`` is a 32 bit integer that uses bits to encode several types
of data.

* bit 31: if set, results in a negative number. Indicates the start of a new
  ring, where the negative value is the number of points
* bit 0: if set, connect previous point to this point
* bit 1: last point
* bit 2: only checked on last point: connect to starting point and any points
  after this but before the next ring are unconnected points

In the code, the polygon start and count arrays are determined by::

    polygon_starts = np.where(self.ring_adjacency['point_flag'] < 0)[0]
    polygon_counts = -self.ring_adjacency[polygon_starts]['point_flag']

The ``state`` flag in this array is used to indicate several aspects of the
ring, including the selected state of the entire ring. Because this array is
the same size as the points array, there are a lot of entries in this array
that can be unused. (NOTE: This was a design decision early on, and I can't
remember why now it is done this way instead of a smaller array that is just
tracked on a per-ring array. But, at any rate, this is how it works now.) The
meaning of ``state`` depends on its position in the points array. The entry in
``state`` at the same index as the first point in the ring holds the selection
state for entire polygon. The next entry (at ``index + 1``) is the feature
code, which is an integer indicating what type of ring it is: water, land,
etc. If the integer is negative, then the ring indicates a hole in the parent
polygon. The next entry, ``index + 2``, holds fill color for entire ring. The
remaining entries corresponding to this ring are unused.

Rings can be edited individually; in the UI, right-clicking inside a ring will
bring up a context menu to modify the ring. Ring data is adjusted using the
``replace_ring`` method, which shifts ring data after an inserted ring and
adjusts the ring adjacency. The ``rings`` attribute is recreated and the
renderer is flagged as needing to be rebuilt.

There is also the concept of the ``geometry_list``, which is defined an a
NamedTuple called ``GeomInfo`` in ``maproom.library.shapefile_utils``, and
keeps track of the ring state in addition to text strings that aren't stored
anywhere else. These text strings corresponding to the name of the polygon as
read out of the source data file (some loaders, like the GDAL loader, can have
text names for these), as well as text names for the ``feature_code`` and
``feature_name``.

A ``feature_list`` is a list of items, where each item is itself a list. Each
sub-list consists of a string identifier and one or more GeomInfo objects. For
example, this feature_list contains 2 entries: a polygon and a polygon with a
hole.

    [
       ['Polygon', GeomInfo(start_index=0, count=5, name='', feature_code=1, feature_name='1')],
       ['Polygon', GeomInfo(start_index=5, count=4, name='', feature_code=1, feature_name='1'),
                   GeomInfo(start_index=9, count=4, name='', feature_code=-1, feature_name='1')],
    ]

The feature list is used when exporting to a shapefile.


Code Architecture - Vector Object Layers
==================================================

Annotation objects are defined in ``maproom.layers.vector_object_layer``. They
include graphical elements like lines, rectangles, circles, and polygons that
scale with the current zoom level, and the objects that act as if they are
stuck to the display, like the text box and the combo text-arrow boxes.

The ``VectorObjectLayer`` base class defines the abstract class used for all
vector objects, based on the ``LineLayer``, so using the same numpy points
array as the line layers. However, the points array is used for control points
for the annotation objects. The control points will define a rectangle that is
the bounding box of the layer.

Some subclasses like the polyline classes use additional points to draw the
polyline contained within the control point boundary box.

Annotation layer objects can also be grouped together; the ``AnnotationLayer``
class is provided that is both a folder object and an annotation layer object.
It doesn't draw anything itself, merely provides a container for other
objects. These folder objects can be nested, and are also used to implement
the ``ArrowTextBoxLayer`` and ``ArrowTextIconLayer``.

LineVectorObject
------------------

The simplest vector object is the ``LineVectorObject``, a line segment with 3
control points: one at each end and one at the midpoint.

The line is defined in the UI by the starting and ending point. The center
control point is not displayed (because the ``display_center_control_point``
class attribute is False). Clicking and dragging one of the control points
moves that point, stretching or shrinking the line in response. The other
control point remains anchored in place.

The class attribute ``anchor_of`` returns the value of the opposite control
point, the point that should remain in place when dragging the index point.
So, for instance, dragging control point zero would use zero as the index
value into this array, returning the value ``1`` as the control point that
remains in place.

The ``anchor_dxdy`` is an array that describes how each control point is
affected when a control point is dragged. This is a two-dimensional array, the
first index indicates the control point that is being dragged. The second
index contains values for all control points and supplies the scaling values
to be applied to each control point as the dragging point is moved.

The dragging operation itself is a command object called from the mouse
handler. The ``MoveControlPointCommand`` is defined in
``maproom.vector_object_commands`` and handles additional details like control
points that are bound to other vector objects ("snap-to-layer"), which is
discussed later.

After every control point move, the bounding box must be updated. When an
annotation object is inside a folder, the bounding box may be forced to be
updated. In this case the object is entirely defined by the points array and
the containing folder will resize the points. Other cases will require the
call to ``fit_to_bounding_box``, which in this case is just an empty method.

Rendering annotation layer objects requires an additional step, called
rasterizing. This sets the renderer with the points needed to describe this
object. Rasterizing lines does not require an extra step because the line can
be fully described by the control points. but rasterizing a circle needs extra
steps since the circle doesn't pass through any of the control points.

Line objects also have markers that can be added to the beginning or ending of
the line. These are stored in the style instance attribute of the layer, and
is drawn by the renderer.

Ends of lines can be snapped to other annotation layer control points. This is
maintained by a mapping in the LayerManager that keeps track of which control
points are linked to other layers. The method
``LayerManager.set_control_point_link`` sets up a "truth" point that is the
source of the location and a "dependent" point, such that any movement of the
source location is propagated to the dependent point. In the
``BaseCanvas.render`` method, there is a call to
``LayerManager.update_linked_control_points`` that, before drawing any layers,
checks to see if any dependent control points need to be move in response to
either: 1) a source control point moving, or 2) the zoom level of the view has
changed, forcing an update of an overlay object.

RectangleVectorObject
-------------------------

The rectangle object is used as the base class for other annotation layer
objects because it defines a rectangular set of control points that can be
used to constrain other objects, like a circle or ellipse.

The addition of more control points requires new ``anchor_of`` and
``anchor_dxdy`` arrays, and the ``compute_constrained_control_points``
function which is needed to fill in the mid-edge control points when new
control points are calculated. The function
``get_control_points_from_corners`` is used when the layer must be fit into a
specified area (specified by opposite corners); this is typically called when
a layer is inside another annotation folder and that folder is resized.


Polylines and Polygons
-------------------------

These objects include extra points after the control points that define the
line segments making up the object. The control points will be adjusted to the
minimum necessary bounding box if the line segment are modified to go outside
the original boundary.

Polylines may have markers at the start and end of the line, while polygons
are always closed shapes.


Overlay Objects
-----------------

Overlay objects are those that are drawn relative to the computer screen and
do not scale with the lat/lon map. They use the OverlayMixin class that
handles updating the control points to keep the objects fixed on screen.

The way overlay objects work is that a lat/lon position is calculated for each
control point at creation time. At every viewport change (zoom or pan), the
lat/lon position of the the control points are recalculated to maintain the
relative position on the screen.

Overlay objects always have one control point fixed to the lat/lon map; the
other control points are recalculated based on some fixed sizes in pixels -
the width and height of the screen object. The control point that is fixed can
be changed, and this changes the location of the object relative to other
lat/lon objects when the map is zoomed in or out.

