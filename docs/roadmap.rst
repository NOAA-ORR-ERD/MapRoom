================
Code Walkthrough
================

MapRoom uses the omnivore framework to provide multi-window, multi-frame user
interface.  See the documentation in `omnivore-overview`.


Startup
=======

The function :func:`.main` defines the MapRoom main plugin and
extra file type recognizer plugins, then passes them on to the
:func:`omnivore.framework.application.run` function which initializes the omnivore
and Enthought frameworks.  This runs the omnivore application that shows the
MapRoom editing windows by default because the Task ID for maproom is also
passed into the ``run`` function that requests the MapRoom task as the startup
task.


MapRoom User Interface Construction
===================================

Plugin
------

The MapRoom task must be referenced in an Enthought Plugin so the framework will
know about it.  This is defined in :class:`maproom.plugin.MaproomPlugin`.  It
defines the MaproomProjectTask and hooks for the preferences, and should not
need to be modified further.

Task
----

The task itself is defined in :class:`maproom.task.MaproomProjectTask`
and includes concrete implementations of the abstract methods in
:class:`omnivore.framework.task.FrameworkTask`, as well as some additional
methods to setup the menu bars, toolbars, and sidebar panes.  The editor is
defined in the :class:`maproom.project_editor_wx.ProjectEditor` as the wx-
specific control :class:`maproom.layer_control_wx.LayerControl`.

The :meth:`maproom.task.MaproomProjectTask.can_edit` class method defines the
MIME types that can be loaded by the Editor::

    @classmethod
    def can_edit(cls, mime):
        return mime.startswith("image") or mime.startswith("application/x-maproom-") or mime == "application/x-hdf"

Editor
------

The :class:`maproom.project_editor_wx.ProjectEditor` is the glue between the
omnivore/Enthought framework and the wx controls that the user interacts with.
These are described in the next section.


Sidebar Panes
-------------

The sidebar creation is localized in :mod:`maproom.pane_layout`.
This also defines the value of the Task ID in the attribute
:data:`maproom.pane_layout.task_id_with_pane_layout`, so if any changes are
made to the pane layout the window save/restore functionality won't attempt to
try to restore an old layout.

Pane IDs are referenced in :meth:`maproom.pane_layout.pane_layout`
as if they were all visible, and initial visibility is defined in
:meth:`maproom.pane_layout.pane_initially_visible`.  Panes are constructed in
the :meth:`maproom.pane_layout.pane_create` method.  The panes are defined in
the file ``maproom.panes.py``.

MapRoom Editing Controls
========================

Main Window: LayerControl
-------------------------

Layer Selection: Layer_tree_control
-----------------------------------

Layer Info: LayerInfoPanel
--------------------------

Info on Selection: SelectionInfoPanel
-------------------------------------

Mouse Mode Toolbar
------------------

Each layer type can specify toolbar items that can be used to edit the data
within the layer.  The :data:`maproom.layer.Layer.mouse_mode_toolbar` attribute
specifies the name of the toolbar item collection, which is in turn defined in
:attribute:`maproom.toolbar.valid_mouse_modes` dictionary.  For example::

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'BaseLayerToolBar': [PanMode, ZoomRectMode],
        }

shows two special toolbar collections and the default collection.  Multiple
layers can use the same toolbar mouse modes.

The mouse modes in the list (e.g.  :class:`maproom.MouseHandler.PanMode`,
:class:`maproom.MouseHandler.ZoomRectMode`, etc.) are subclasses of
:class:`maproom.MouseHandler` that process the mouse and keyboard events
through methods like :meth:`maproom.MouseHandler.process_mouse_motion_down`
and :meth:`maproom.MouseHandler.process_mouse_down`.

Adding a new mouse mode to the applicable dict entry (or entries if the mode can
work with multiple toolbars) of :attribute:`maproom.toolbar.valid_mouse_modes`
is all that is required for it to appear in the UI.  The toolbar is now
automatically generated from the list of mouse modes.

Adding New Toolbar Items
~~~~~~~~~~~~~~~~~~~~~~~~

To add a new toolbar item to an existing mouse mode, create the new subclass
of :class:`maproom.MouseHandler` and add that to the appropriate list in
the :attribute:`maproom.LayerCanvas.valid_mouse_modes` dictionary.

If you are creating a new mouse handler, see the :ref:`Mouse Handler` section
that describes the process of extending a :class:`maproom.MouseHandler` class
to perform additional functions.  The new mouse handler must be added to the
toolbar definition, so for this example, `RulerMode` is added
to the all the layers::

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, RulerMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, RulerMode, CropRectMode],
        'BaseLayerToolBar': [PanMode, ZoomRectMode, RulerMode],
        }

There is glue code that takes mouse modes and turns them into toolbar items.
The function :attribute:`maproom.toolbar.get_all_toolbars` does this
automatically it task creation time, so no additional code must we written for
the toolbar itself.


Adding a New Toolbar
~~~~~~~~~~~~~~~~~~~~

Adding a new toolbar requires a new text string identifier that ends in
"ToolBar".  This text string is used to refer to the toolbar type in the
layers that support this toolbar.  This text string must be added as a key to
the :attribute:`maproom.toolbar.valid_mouse_modes` dictionary.

In this key/value pair, the value is a list of mouse mode
classes.  (See the above section on adding a new mouse
mode.) Note that :class:`maproom.MouseHandler.PanMode` and
:class:`maproom.MouseHandler.ZoomRectMode` are common to most modes and so
they should be included as the first two items unless there is some layer-
specific reason not to.

You must also add the toolbar string identifier to the
:data:`maproom.layer.Layer.mouse_mode_toolbar` attribute in all the layers
that use this toolbar.  E.g.  if you were adding the ``AnnotationLayerToolBar``
toolbar to the :class:`maproom.layers.VectorObjectLayer` class, you would use
this code::

    mouse_mode_toolbar = Str("AnnotationLayerToolBar")


Mouse Handler
-------------

Mouse handlers are objects that process mouse and keyboard handling to
provide customization based on the layer.  The :class:`maproom.MouseHandler`
provides an abstraction into the mouse processing with overridable
methods like :meth:`maproom.MouseHandler.process_mouse_down` and
:meth:`maproom.MouseHandler.process_mouse_motion_up`.

Several UI features, like the icon and menu name, are class attributes (note
that icons should be located in the `maprooms/icons` subdirectory), so when
defining the subclass, be sure to change those to describe the new handler.

Mouse handlers have limited lifetime: an instance is created every time the
user clicks on a different mouse mode in the UI.  So, the mouse mode should
not be used to store any state information because it will be lost when
switching to a new mode.


MapRoom Layers
==============

Point Layer
-----------

Line Layer
----------

Polygon Layer
-------------

Raster Layer
------------

Adding a New Layer
==================

The code for layers resides in the :mod:`maproom.layers` module, and unless some special case is needed should subclass from :class:`maproom.layers.ScreenLayer` or :class:`maproom.layers.ProjectedLayer`.

Most layers will need methods to save to disk and load from disk, so the file
format recoginition must be added.  This process is documented in :ref:`Adding
a New MIME Type`.  Once the file type is recognized by MapRoom, a loader can
be defined in the module :mod:`maproom.layers.loaders`.

For your new layer, create a new class that extends from
:class:`BaseLayerLoader` and change the class attributes to match the file
type information as defined in your new MIME type handler.  E.g.  for the
FloatCanvas annotation layer, the new class attributes are::

    class FloatCanvasJSONLoader(BaseLayerLoader):
        mime = "application/x-float_canvas"
        layer_types = ["annotation"]
        extensions = [".fc"]
        name = "FloatCanvas JSON Layer"

The `load_layers` method expects a list of layers to be returned, so your
loader can return multiple layers if that's supported by the file format.
The metadata object passed to it in the arguments list contains a pointer
to the uri (filename) and the other argument passed to it is the manager
object needed as a parameter for all `Layer` constructors.  Again, using the
FloatCanvas annotation layer as a simple example, the `load_layer` method is::

    def load_layers(self, metadata, manager):
        layers = []
        with open(metadata.uri, "r") as fh:
            text = fh.read()
            layer = AnnotationLayer(manager=manager)
            layer.load_fc_json(text)
            layers.append(layer)
        return layers

The :meth:`maproom.layers.AnnotationLayer.load_fc_json` method takes the
MapRoom formatted text string loaded above in the load_layers method, and
calls the :meth:`FloatCanvas.Unserialize` method to restore the graphic
objects to the annotation layer.


MapRoom File Types
==================

Adding a New MIME Type
----------------------

In order to display a new file type, MapRoom must be programmed
to recognize the new file type.  Three actions are needed:

First: add a new :class:`omnivore.file_type.i_file_recognizer.IFileRecognizer`
that can return a MIME type based on either a scan of the beginning of the
file, or as a last resort based on the filename itself.  These classes reside
in the :mod:`maproom.file_type` module.  E.g., for the FloatCanvas annotation
layer, the class :class:`maproom.file_type.FloatCanvasJSONRecognizer` was
added::

    @provides(IFileRecognizer)
    class FloatCanvasJSONRecognizer(HasTraits):
        """Finds FloatCanvas JSON files using the text header
        
        """
        id = "application/x-float_canvas_json"
        
        before = "text/plain"
        
        def identify(self, guess):
            byte_stream = guess.get_utf8()
            if byte_stream.startswith("FloatCanvas JSON Format"):
                return self.id

The `@provides` decorator is a Traits feature that marks this class as a plugin.

Adding a new recognizer in the :mod:`maproom.file_type` module and rerunning
the cog script contained in :file:`maproom.file_type.__init__.py` will add the
new recognizer class into the automatically scanned list of recognizers.  This
must be run once and the new version of __init__.py checked in to the source
code repository so that it doesn't have to be run again and cog doesn't have
to be a dependency of the project at runtime. It is run by::

    cd maproom/file_type
    cog.py -r __init__.py

Second: the :meth:`maproom.task.MaproomProjectTask.can_edit` class method must be modified to accept the new MIME type.

Third: a layer loader must be added to parse the file and return the correct
layer type based on the data.  If the new file type can not be displayed
by a current layer, you will have to create a new layer type.  See above
:ref:`Adding a New Layer`


MapRoom load process
====================

Due to the complexity resulting from the flexibility of the Omnivore framework, some things aren't easy to follow. Like loading a file.

When a file is loaded from File -> Open or from the command line it goes through this process:

* FrameworkApplication.load_file is called with the path name (really URI), any keyword arguments sent to the function, and some optional stuff that isn't necessary to discuss here.
* attempt to get a document that can edit that type of file using a FileGuess object and the FileRecognizerDriver and raises an error here if a compatible document type isn't found.
* find a task that can edit this document
* calls Task.new on that document, passing through the keyword arguments from load_file
* Task.new checks document and if it is an entire MapRoom project file, it is loaded into a new tab by creating a new :class:`ProjectEditor`.
* If it creates a new tab, it then calls FrameworkEditor.activate_editor which is a call into the Pyface library, and a side effect of that is a call back to :class:`ProjectEditor.create` which sets up the controls in the UI. This create method is a good place to put any initialization code that can happen before the document is loaded
* :meth:`FrameworkEditor.load_omnivore_document` is called to process the document by finding a :class:`maproom.layers.loaders.common.BaseLoader` instance that can handle the file type.
