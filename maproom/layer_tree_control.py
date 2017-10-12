import sys
import wx
import omnivore.utils.wx.customtreectrl as treectrl

from layers import Layer
from menu_commands import MoveLayerCommand


import logging
log = logging.getLogger(__name__)


class IndexableTree(treectrl.CustomTreeCtrl):
    def get_item_by_indexes(self, *args):
        args = list(args)  # copy since it could be an iterator or unmodifiable
        print(args)
        if args[0] > 0:
            raise IndexError("Invalid root index")
        parent = self.GetRootItem()
        log.debug("root @ %s: %s" % (args[0:1], parent.GetText()))
        parent = self.GetRootItem()
        used = []
        for level, index in enumerate(args[1:]):
            # sanity check first
            if index >= self.GetChildrenCount(parent):
                raise IndexError("Invalid index at %s" % str(used))
            item, cookie = self.GetFirstChild(parent)
            for i in range(1, index):
                item, cookie = self.GetNextChild(item, cookie)
            parent = item
            log.debug("child @ %s: %s" % (args[0:level + 2], parent.GetText()))
        return parent


class LayerTreeControl(wx.Panel):

    dragged_item = None

    def __init__(self, parent, project, size=(-1, -1)):
        wx.Panel.__init__(self, parent, wx.ID_ANY, size=size)

        self.tree = IndexableTree(self, wx.ID_ANY, style=treectrl.TR_DEFAULT_STYLE,
                                            agwStyle=treectrl.TR_HIDE_ROOT | treectrl.TR_NO_LINES | treectrl.TR_HAS_BUTTONS | treectrl.TR_FULL_ROW_HIGHLIGHT)
        self.project = project

        self.tree.Bind(treectrl.EVT_TREE_ITEM_CHECKED, self.handle_item_checked)
        self.tree.Bind(treectrl.EVT_TREE_BEGIN_DRAG, self.handle_begin_drag)
        self.tree.Bind(treectrl.EVT_TREE_END_DRAG, self.handle_end_drag)
        self.tree.Bind(treectrl.EVT_TREE_SEL_CHANGED, self.handle_selection_changed)

        """
        self.state_image_list = wx.ImageList( self.IMAGE_SIZE, self.IMAGE_SIZE )
        #self.state_image_list.Add( wx.Bitmap( "maproom/ui/images/maproom.png", wx.BITMAP_TYPE_PNG ) )
        #self.SetImageList( self.state_image_list )
        """

        self.tree.Bind(wx.EVT_LEFT_DOWN, self.mouse_pressed)
        self.tree.Bind(wx.EVT_RIGHT_DOWN, self.mouse_pressed)
        # self.Bind( wx.EVT_RIGHT_UP, self.mouse_right_released )
        if sys.platform.startswith("win"):
            self.tree.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.Fit()

    def set_project(self, project):
        self.project = project
        self.rebuild()

    def get_edit_layer(self):
        item = self.tree.GetSelection()
        if (item is None):
            return None
        (layer, ) = self.tree.GetItemData(item)


        return layer

    def is_edit_layer(self, layer):
        item = self.tree.GetSelection()
        if (item is None):
            return False
        (selected, ) = self.tree.GetItemData(item)
        return layer == selected

    def set_edit_layer(self, layer):
        if self.project is None:
            return
        self.project.clear_all_selections(False)

        if (layer is None):
            self.tree.UnselectAll()
        else:
            self.set_edit_layer_recursive(layer, self.tree.GetRootItem())

    def set_edit_layer_recursive(self, layer, item):
        (item_layer, ) = self.tree.GetItemData(item)

        if (item_layer == layer):
            self.tree.SelectItem(item, True)
            # also make sure the layer's name is up-to-date
            self.tree.SetItemText(item, layer.name)
            layer.set_visibility_when_selected(self.project.layer_visibility[layer])

            return True

        if (not self.tree.ItemHasChildren(item)):
            return False

        n = self.tree.GetChildrenCount(item, False)
        child, cookie = self.tree.GetFirstChild(item)

        while (n > 0):
            if (self.set_edit_layer_recursive(layer, child)):
                return True

            child = self.tree.GetNextSibling(child)
            n -= 1

        return False

    def get_expanded_state(self):
        state = dict()
        self.get_expanded_state_recursive(self.tree.GetRootItem(), state)
        return state

    def get_expanded_state_recursive(self, item, state):
        if item is None:
            return
        (item_layer, ) = self.tree.GetItemData(item)
        state[item_layer] = item.IsExpanded()
        if (not self.tree.ItemHasChildren(item)):
            return

        n = self.tree.GetChildrenCount(item, False)
        child, cookie = self.tree.GetFirstChild(item)

        while (n > 0):
            self.get_expanded_state_recursive(child, state)
            child = self.tree.GetNextSibling(child)
            n -= 1

    def clear_all_items(self):
        self.tree.DeleteAllItems()

    def rebuild(self):
        # rebuild the tree from the layer manager's data
        if self.project is None:
            # Wait till there's an active project
            return
        selected = self.get_edit_layer()
        expanded_state = self.get_expanded_state()
        lm = self.project.layer_manager
        self.tree.DeleteAllItems()
        # self.Freeze()
        log.debug("LAYER_TREE: rebuiding layers = " + str(lm.layers))
        self.add_layers_recursive(lm.layers, None, expanded_state)
        # self.Thaw()
        if selected:
            self.set_edit_layer(selected)
            self.project.update_layer_selection_ui(selected)

    def add_layers_recursive(self, layer_tree, parent, expanded_state):
        if (len(layer_tree) == 0):
            return

        # we assume the layer at the start of each list is a folder
        folder_node = self.add_layer(layer_tree[0], parent, expanded_state)
        # import code; code.interact( local = locals() )

        for item in layer_tree[1:]:
            if (isinstance(item, Layer)):
                self.add_layer(item, folder_node, expanded_state)
            else:
                self.add_layers_recursive(item, folder_node, expanded_state)

    def add_layer(self, layer, parent, expanded_state):
        log.debug("LAYER_TREE: adding layer = " + str(layer.name))
        data = (layer, )
        if (parent is None):
            return self.tree.AddRoot(layer.name, data=data)

        vis = self.project.layer_visibility[layer]
        item = self.tree.AppendItem(parent, layer.name, ct_type=treectrl.TREE_ITEMTYPE_CHECK, data=data)
        self.tree.CheckItem2(item, vis["layer"])
        if layer.is_folder():
            # Force the appearance of expand button on folders to be used as
            # drop target for dropping inside a folder
            item.SetHasPlus(True)

        if layer in expanded_state:
            expanded = expanded_state[layer]
            if expanded:
                item.Expand()
            else:
                item.Collapse()
        else:
            # expand by default if not listed
            item.Expand()

        return item

    def remove_layer(self, layer, parent=None):
        item = self.layer_to_item.get(layer)
        if item is None:
            return

        self.tree.Freeze()
        self.tree.Delete(item)

        if self.tree.GetChildrenCount(self.root) == 0:
            self.none_item = self.tree.AppendItem(self.root, "None")

        self.tree.Thaw()

        self.layer_to_item.pop(layer, None)

    def update_checked_from_visibility(self):
        self.update_checked_from_visibility_recursive(self.tree.GetRootItem())

    def update_checked_from_visibility_recursive(self, item):
        if item is None:
            return
        (layer, ) = self.tree.GetItemData(item)
        checked = self.project.layer_visibility[layer]["layer"]
        self.tree.CheckItem2(item, checked, True)
        if (not self.tree.ItemHasChildren(item)):
            return

        n = self.tree.GetChildrenCount(item, False)
        child, cookie = self.tree.GetFirstChild(item)

        while (n > 0):
            self.update_checked_from_visibility_recursive(child)
            child = self.tree.GetNextSibling(child)
            n -= 1

    def handle_item_checked(self, event):
        (layer, ) = self.tree.GetItemData(event.GetItem())
        item = event.GetItem()
        checked = self.tree.IsItemChecked(item)
        layer.set_visibility_when_checked(checked, self.project.layer_visibility)
        self.update_checked_from_visibility_recursive(item)
        self.project.refresh()
        event.Skip()

    def handle_begin_drag(self, event):
        (layer, ) = self.tree.GetItemData(event.GetItem())
        item = event.GetItem()
        if not layer.is_root():
            event.Allow()
            self.dragged_item = item

    def handle_end_drag(self, event):
        item = event.GetItem()
        local_dragged_item = self.dragged_item
        self.dragged_item = None

        # if we dropped somewhere that isn't on top of an item, ignore the event
        if item is None or not item.IsOk():
            return

        (target_layer, ) = self.tree.GetItemData(item)
        (source_layer, ) = self.tree.GetItemData(local_dragged_item)
        lm = self.project.layer_manager
        mi_source = lm.get_multi_index_of_layer(source_layer)
        mi_target = lm.get_multi_index_of_layer(target_layer)

        if (len(mi_target) > len(mi_source) and mi_target[0: len(mi_source)] == mi_source):
            self.project.window.error("You cannot move folder into one of its sub-folders.", "Invalid Layer Move")
            self.tree.Refresh()
            return

        before = event.IsDroppedBeforeItem()
        in_folder = event.IsDroppedInFolder()
        cmd = MoveLayerCommand(source_layer, target_layer, before, in_folder)
        self.project.process_command(cmd)

    def handle_selection_changed(self, event):
        self.project.clear_all_selections(False)
        layer = self.get_edit_layer()
        self.project.update_layer_selection_ui(layer)
        layer.set_visibility_when_selected(self.project.layer_visibility[layer])
        self.project.refresh()
        lm = self.project.layer_manager
        sel = lm.get_multi_index_of_layer(layer)
        log.debug("Multi-index of selected layer: %s" % sel)

    def raise_selected_layer(self):
        self.move_selected_layer(-1)

    def raise_to_top(self):
        self.move_selected_layer(-1, True)

    def lower_to_bottom(self):
        self.move_selected_layer(1, True)

    def lower_selected_layer(self):
        self.move_selected_layer(1)

    def move_selected_layer(self, delta, to_extreme=False):
        item = self.tree.GetSelection()
        (layer, ) = self.tree.GetItemData(item)
        lm = self.project.layer_manager
        mi_source = lm.get_multi_index_of_layer(layer)
        mi_target = mi_source[: len(mi_source) - 1]
        if to_extreme:
            if delta < 0:
                mi_target.append(1)  # zeroth index is folder
            else:
                mi2 = mi_source[: len(mi_source) - 1]
                parent_list = lm.get_layer_by_multi_index(mi2)
                total = len(parent_list)
                mi_target.append(total)
        else:
            mi_target.append(mi_source[len(mi_source) - 1] + delta)

        # here we "re-get" the source layer so that if it's a folder layer what we'll get is the
        # folder's list, not just the folder pseudo-layer
        source_layer = lm.get_layer_by_multi_index(mi_source)
        lm.remove_layer_at_multi_index(mi_source)
        lm.insert_layer(mi_target, source_layer)
        self.project.clear_all_selections(False)
        self.set_edit_layer(layer)
        self.rebuild()

    def mouse_pressed(self, event):
        # If a selected item is clicked, unselect it so that it will be
        # selected again. This allows the user to click on an
        # already-selected layer to display its properties, for instance.
        event.Skip()
        selected_item = self.tree.GetSelection()
        if selected_item is None:
            return

        (clicked_item, flags) = self.tree.HitTest(event.GetPosition())

        if clicked_item != selected_item or \
           flags & wx.TREE_HITTEST_ONITEMLABEL == 0:
            return

        self.tree.ToggleItemSelection(selected_item)

    def on_mouse_wheel_scroll(self, event):
        screen_point = event.GetPosition()
        size = self.GetSize()
        if screen_point.x < 0 or screen_point.y < 0 or screen_point.x > size.x or screen_point.y > size.y:
            # print "Mouse not over Tree: trying map!"
            if self.project is not None:
                self.project.control.on_mouse_wheel_scroll(event)
            return

        event.Skip()
