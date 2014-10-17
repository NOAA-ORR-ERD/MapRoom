"""Sample panes for Skeleton

"""
# Enthought library imports.
from pyface.tasks.api import TaskLayout, PaneItem, HSplitter, VSplitter

from panes import *


# The project ID must be changed when the pane layout changes, otherwise
# the new panes won't be displayed because the previous saved state of the
# application will be loaded.  Changing the project ID forces the framework to
# honor the new layout, because there won't be a saved state of this new ID.

# The saved state is stored in ~/.config/Peppy2/tasks/wx/application_memento

# Removing this file will cause the default layout to be used.  The saved state
# is only updated when quitting the application; if the application is killed
# (or crashes!) the saved state is not updated.

task_id_with_pane_layout = 'maproom.project.v4'

def pane_layout():
    """ Create the default task layout, which is overridded by the user's save
    state if it exists.
    """
    return TaskLayout(
        left=VSplitter(
            PaneItem('maproom.layer_selection_pane'),
            PaneItem('maproom.layer_info_pane'),
            PaneItem('maproom.selection_info_pane'),
            ),
        right=HSplitter(
            PaneItem('maproom.triangulate_pane'),
            PaneItem('maproom.merge_points_pane'),
            ),
        )

def pane_initially_visible():
    """ List of initially visible panes; all others (like the triangulate
    pane) will be hidden.
    """
    
    return {
        'maproom.layer_selection_pane',
        'maproom.layer_info_pane',
        'maproom.selection_info_pane',
        }

def pane_create():
    """ Create all the pane objects available for the task (regardless
    of visibility -- visibility is handled in the task activation method
    MaproomTask.activated)
    """
    return [ LayerSelectionPane(), LayerInfoPane(), SelectionInfoPane(),
             TriangulatePane(), MergePointsPane() ]
