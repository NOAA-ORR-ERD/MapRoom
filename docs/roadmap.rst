================
Code Walkthrough
================

MapRoom uses the Peppy2 framework to provide multi-window, multi-frame user
interface.  See the documentation in `peppy2-overview`.


Startup
=======

The function :func:`.main` defines the MapRoom main plugin and
extra file type recognizer plugins, then passes them on to the
:func:`peppy2.framework.application.run` function which initializes the peppy2
and Enthought frameworks.  This runs the Peppy2 application that shows the
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
:class:`peppy2.framework.task.FrameworkTask`, as well as some additional
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
Peppy2/Enthought framework and the wx controls that the user interacts with.
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
:attribute:`maproom.LayerCanvas.valid_mouse_modes` dictionary.  For example::

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'default': [PanMode, ZoomRectMode],
        }

shows two special toolbar collections and the default collection.  Multiple
layers can use the same toolbar mouse modes.

The mouse modes in the list (e.g.  :class:`maproom.MouseHandler.PanMode`,
:class:`maproom.MouseHandler.ZoomRectMode`, etc.) are subclasses of
:class:`maproom.MouseHandler` that process the mouse and keyboard events
through methods like :meth:`maproom.MouseHandler.process_mouse_motion_down`
and :meth:`maproom.MouseHandler.process_mouse_down`.

Adding a New Mouse Mode
~~~~~~~~~~~~~~~~~~~~~~~

Adding a new mouse mode requires a new text string identifier that ends in
"ToolBar".  This text string is used to refer to the toolbar type in several
places.

First, this text string must be added as a key to the
:attribute:`maproom.LayerCanvas.valid_mouse_modes` dictionary.
The value refered to by this key must be a list of mouse mode
classes.  Note that :class:`maproom.MouseHandler.PanMode` and
:class:`maproom.MouseHandler.ZoomRectMode` are common to most modes, and
should be included as the first two items unless there is some layer-specific
reason not to.

Second, the text string must also be added as the id
keyword to a new :class:`SToolBar` Enthought object in the
:meth:`maproom.MaproomProjectTask._tool_bars_default` method.  This SToolBar
instance specifies a set of Enthought :class:`Action`s that correspond one-to-
one to the MouseHandler objects for the layer.  E.g.  the SToolBar object for
the `VectorLayerToolBar` is::

    SToolBar(Group(ZoomModeAction(),
                   PanModeAction(),
                   AddPointsAction(),
                   AddLinesAction()),
             show_tool_names=False,
             id="VectorLayerToolBar",),

where the actions correspond to the mouse handlers as so:

.. csv-table:: Action to Mouse Handle Mapping
   :header: "Action", "MouseHandler"

   ZoomModeAction, ZoomRectMode
   PanModeAction, PanMode
   AddPointsAction, PointSelectionMode
   AddLinesAction, LineSelectionMode

The Action classes provide the Enthought code framework with the the icon and
menu item name for the user interface, and the code to track which mouse mode
is active at the moment.

So, adding a new mode would proceed like this:

The vector object layers need to have modes to move, create and group objects. The first task is to create a new text id for the mode, which will be called `AnnotationLayerToolBar`. This text id is placed in the LayerCanvas::

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'AnnotationLayerToolBar': [PanMode, ZoomRectMode],
        'default': [PanMode, ZoomRectMode],
        }

For testing purposes, only the existing PanMode and ZoomRectModes are listed in the AnnotationLayerToolBar. We'll define a new mouse handler in the next section.

Finally, the :data:`maproom.layer.Layer.mouse_mode_toolbar` attribute must be
set in the :class:`maproom.layers.VectorObjectLayer` to tell MapRoom when to
display this toolbar::

    mouse_mode_toolbar = Str("AnnotationLayerToolBar")




Adding New Toolbar Items
~~~~~~~~~~~~~~~~~~~~~~~~

To add a new toolbar item to an existing mouse mode, create the new subclass
of :class:`maproom.MouseHandler` and add that to the appropriate list in
the :attribute:`maproom.LayerCanvas.valid_mouse_modes` dictionary.  In this
case, we'll create `ControlPointSelectionMode` and it will be added to the
`AnnotationLayerToolBar`::

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'AnnotationLayerToolBar': [PanMode, ZoomRectMode, ControlPointSelectionMode],
        'default': [PanMode, ZoomRectMode],
        }

To get the toolbar item to appear, an action must be defined in task.py::

    class ControlPointAction(MouseHandlerBaseAction):
        handler = ControlPointSelectionMode
        name = handler.menu_item_name
        tooltip = handler.menu_item_tooltip
        image = ImageResource(handler.icon)

This class references the new handler ControlPointSelectionMode and the rest
of the boilerplate stuff in this class essentially just supplies defaults to
the MouseHandlerBaseClass, which in turn provides all the glue code to the
Enthought framework for the UI stuff.  More than likely this action class won't
have to be modified any more than the above for any action that you create.

In the toolbar creation code in :meth:`maproom.MaproomProjectTask._tool_bars_default`, the Action needs to be referenced::

    SToolBar(Group(ZoomModeAction(),
                   PanModeAction(),
                   ControlPointAction()),
             show_tool_names=False,
             id="AnnotationLayerToolBar",),

Note that the order in which the actions are defined is the order that they will appear in the UI.


Mouse Handler
-------------

Mouse handlers are objects that process mouse and keyboard handling to provide
customization based on the layer.

Icons should be located in the `maprooms/icons` subdirectory.

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

First: add a new :class:`peppy2.file_type.i_file_recognizer.IFileRecognizer`
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
