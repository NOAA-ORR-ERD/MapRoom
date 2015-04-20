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


Rendering
=========

Tutorials

* `Modern OpenGL tutorial by Nicolas Rougier <http://www.loria.fr/~rougier/teaching/opengl/>`_
* `PyOpenGL tutorials, including a set of Shader tutorials <http://pyopengl.sourceforge.net/context/tutorials/>`_
* https://bitbucket.org/rndblnch/opengl-programmable/src

Useful OpenGL links:

* http://en.wikibooks.org/wiki/GLSL_Programming/Vector_and_Matrix_Operations
* https://www.opengl.org/wiki/Common_Mistakes
* https://www.opengl.org/wiki/Vertex_Specification
* https://www.opengl.org/wiki/Vertex_Specification_Best_Practices
* https://github.com/vispy/vispy/pull/782

Interesting shader examples

* https://github.com/mike-lawrence/actichampy/blob/master/actichampy.py
* http://www.lighthouse3d.com/cg-topics/code-samples/opengl-3-3-glsl-1-5-sample/
