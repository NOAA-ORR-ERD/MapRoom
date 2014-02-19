"""Sample panes for Skeleton

"""
# Enthought library imports.
from pyface.tasks.api import DockPane

class LayerSelectionPane(DockPane):
    #### TaskPane interface ###################################################

    id = 'maproom.layer_selection_pane'
    name = 'Layers'

class LayerInfoPane(DockPane):
    #### TaskPane interface ###################################################

    id = 'maproom.layer_info_pane'
    name = 'Info'
