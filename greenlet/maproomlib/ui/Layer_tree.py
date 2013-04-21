import wx

try:
    import wx.lib.agw.customtreectrl as treectrl
except ImportError:
    import wx.lib.customtreectrl as treectrl

import maproomlib.ui
import maproomlib.utility


class Layer_tree( treectrl.CustomTreeCtrl ):
    """
    A tree control for displaying a hierarchical group of layers. Supports
    toggling individual layer visibility.
    """
    IMAGE_SIZE = 16
    NON_CHECK_LAYERS = set( (
        "Line_layer",
    ) )

    def __init__( self, parent, root_layer, command_stack ):
        treectrl.CustomTreeCtrl.__init__(
            self, parent, wx.ID_ANY,
            style = treectrl.TR_DEFAULT_STYLE | wx.BORDER_SUNKEN,
        )
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.layer_to_item = {} # layer -> corresponding treectrl item
        self.inbox = maproomlib.ui.Wx_inbox()
        self.outbox = maproomlib.utility.Outbox()

        self.root = self.AddRoot( "Layers" )
        self.none_item = self.AppendItem( self.root, "None" )
        self.state_image_list = wx.ImageList( self.IMAGE_SIZE, self.IMAGE_SIZE )
        #self.state_image_list.Add( wx.Bitmap( "maproom/ui/images/maproom.png", wx.BITMAP_TYPE_PNG ) )

        #self.SetImageList( self.state_image_list )
        self.Expand( self.root )

        self.Bind(
            treectrl.EVT_TREE_ITEM_CHECKING,
            self.toggle_visibility,
        )
        self.Bind(
            treectrl.EVT_TREE_SEL_CHANGED,
            self.selection_changed,
        )
        self.Bind(
            wx.EVT_LEFT_DOWN,
            self.mouse_pressed,
        )
        self.Bind(
            wx.EVT_RIGHT_DOWN,
            self.mouse_pressed,
        )
        self.Bind(
            wx.EVT_RIGHT_UP,
            self.mouse_right_released,
        )

    def run( self, scheduler ):
        self.root_layer.inbox.send(
            request = "get_layers",
            response_box = self.inbox
        )

        message = self.inbox.receive( request = "layers" )
        hidden_layers = message.get( "hidden_layers" )

        for layer in message.get( "layers" ):
            self.add( layer, hidden = layer in hidden_layers )

        self.root_layer.outbox.subscribe(
            self.inbox,
            request = (
                "layer_added", "layer_removed",
                "layer_shown", "layer_hidden",
                "layer_raised", "layer_lowered",
                "raise_layer", "lower_layer",
                "selection_updated",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "layer_added", "layer_removed",
                    "layer_shown", "layer_hidden",
                    "layer_raised", "layer_lowered",
                    "raise_layer", "lower_layer",
                    "remove_layer",
                    "property_updated",
                    "selection_updated",
                ),
            )
            request = message.pop( "request" )

            if request == "layer_added":
                self.add( **message )
            elif request == "layer_removed":
                self.remove( **message )
            elif request == "layer_shown":
                self.show( **message )
            elif request == "layer_hidden":
                self.hide( **message )
            elif request == "layer_raised":
                self.raise_layer( **message )
            elif request == "layer_lowered":
                self.lower_layer( **message )
            elif request == "raise_layer":
                self.raise_current( **message )
            elif request == "lower_layer":
                self.lower_current( **message )
            elif request == "property_updated":
                self.property_updated( **message )
            elif request == "selected_updated":
                self.selected_updated( **message )

    def add( self, layer, hidden = False, insert_index = None,
             parent = None ):
        if hasattr( layer, "outbox" ):
            layer.outbox.subscribe(
                self.inbox,
                request = (
                    "property_updated",
                    "layer_added",
                    "layer_removed",
                )
            )

        if self.none_item:
            self.Delete( self.none_item )
            self.none_item = None

        parent_item = None

        if parent:
            parent_item = self.layer_to_item.get( parent )
        if parent_item is None:
            parent_item = self.root

        self.Freeze()

        item = self.add_item(
            layer, parent = parent_item, hidden = hidden,
            insert_index = insert_index,
        )
        self.Expand( parent_item )
        if item is not None:
            self.ToggleItemSelection( item )

        self.Thaw()

    def add_item( self, layer, parent, hidden = False, insert_index = None ):
        import maproomlib.plugin

        if maproomlib.plugin.Layer_selection_layer.ghost_layer( layer ):
           return None

        if layer.__class__.__name__ in self.NON_CHECK_LAYERS:
            item_type = treectrl.TREE_ITEMTYPE_NORMAL
        else:
            item_type = treectrl.TREE_ITEMTYPE_CHECK

        data = wx.TreeItemData()
        data.SetData( layer )

        if insert_index is None:
            item = self.PrependItem(
                parent,
                str( layer.name ),
                ct_type = item_type,
                data = data,
            )
        else:
            # The insert_index assumes the inclusion of ghost layers, but this
            # layer tree excludes ghost layers. So in order to convert the
            # given insert_index into an index compatible with this layer
            # tree, calculate an offset based on the number of ghost layers
            # that come before the given layer.
            siblings = layer.parent.children

            for sibling in siblings[ : insert_index ]:
                if maproomlib.plugin.Layer_selection_layer.ghost_layer( sibling ):
                    insert_index -= 1

            # Flip the index so that it's from the start of the list rather
            # than the bottom of the list.
            insert_index = \
                self.GetChildrenCount( parent, recursively = False ) - \
                insert_index

            item = self.InsertItemByIndex(
                parent,
                insert_index,
                str( layer.name ),
                ct_type = item_type,
                data = data,
            )

        self.CheckItem2( item, checked = not hidden )
        self.layer_to_item[ layer ] = item

        for child in layer.children:
            self.add_item(
                child, parent = item, hidden = child in layer.hidden_children,
            )

        self.Expand( item )

        return item

    def remove( self, layer, parent = None ):
        if hasattr( layer, "outbox" ):
            layer.outbox.unsubscribe( self.inbox )

        item = self.layer_to_item.get( layer )
        if item is None: return

        self.Freeze()
        self.Delete( item )

        if self.GetChildrenCount( self.root ) == 0:
            self.none_item = self.AppendItem( self.root, "None" )

        self.Thaw()

        self.layer_to_item.pop( layer, None )

    def toggle_visibility( self, event ):
        layer = self.GetItemPyData( event.GetItem() ).Data
        item = event.GetItem()
        checked = self.IsItemChecked( item )

        self.command_stack.inbox.send(
            request = "start_command",
        )

        if checked:
            self.root_layer.inbox.send(
                request = "hide_layer",
                layer = layer,
            )
        else:
            self.root_layer.inbox.send(
                request = "show_layer",
                layer = layer,
            )

    def show( self, layer ):
        item = self.layer_to_item.get( layer )
        if item is None: return

        self.Freeze()
        self.CheckItem2( item, checked = True )
        self.EnableChildren( item, enable = True )
        self.Thaw()

    def hide( self, layer ):
        item = self.layer_to_item.get( layer )
        if item is None: return

        self.Freeze()
        self.CheckItem2( item, checked = False )
        self.EnableChildren( item, enable = False )
        self.Thaw()

    def raise_current( self ):
        item = self.GetSelection()
        if item is None: return
        layer = self.GetItemPyData( item ).Data

        # If it's already the top child, it can't be raised any higher.
        previous_item = self.GetPrevSibling( item )
        if previous_item is None: return

        self.command_stack.inbox.send(
            request = "start_command",
        )
        self.root_layer.inbox.send(
            request = "raise_layer",
            layer = layer,
        )

    def lower_current( self ):
        item = self.GetSelection()
        if item is None: return
        layer = self.GetItemPyData( item ).Data

        # If it's already the bottom child, it can't be lowered any further.
        next_item = self.GetNextSibling( item )
        if next_item is None: return

        self.command_stack.inbox.send(
            request = "start_command",
        )
        self.root_layer.inbox.send(
            request = "lower_layer",
            layer = layer,
        )

    def raise_layer( self, layer ):
        item = self.layer_to_item.get( layer )
        if item is None: return

        previous_item = self.GetPrevSibling( item )
        if previous_item is None: return

        # Move the previous sibling to be after the given layer item.
        self.Freeze()
        self.move_item_after( previous_item, after_item = item )
        self.Delete( previous_item )
        self.Thaw()

        self.selection_changed()

    def lower_layer( self, layer ):
        item = self.layer_to_item.get( layer )
        if item is None: return

        next_item = self.GetNextSibling( item )
        if next_item is None: return

        # Move the given layer item to be after the next sibling.
        self.Freeze()
        self.move_item_after( item, after_item = next_item )
        self.Delete( item )
        self.Thaw()

        self.selection_changed()

    def move_item_after( self, item, after_item ):
        new_item = self.InsertItem(
            after_item.GetParent(),
            after_item,
            item.GetText(),
            ct_type = item.GetType(),
            data = item.GetData(),
            image = item.GetImage(),
        )

        self.replace_item( item, new_item )

    def reparent_item( self, item, new_parent ):
        new_item = self.AppendItem(
            new_parent,
            item.GetText(),
            ct_type = item.GetType(),
            data = item.GetData(),
            image = item.GetImage(),
        )

        self.replace_item( item, new_item )

    def replace_item( self, item, new_item ):
        if item == self.GetSelection():
            self.ToggleItemSelection( new_item )
        self.CheckItem2( new_item, checked = self.IsItemChecked( item ) )

        self.reparent_children( item, new_item )

        if self.IsExpanded( item ):
            self.Expand( new_item )

        layer = self.GetItemPyData( item ).Data
        self.layer_to_item[ layer ] = new_item

    def reparent_children( self, old_parent, new_parent ):
        ( old_child, token ) = self.GetFirstChild( old_parent )

        while old_child:
            self.reparent_item( old_child, new_parent )
            ( old_child, token ) = self.GetNextChild( old_parent, token )

    def selection_changed( self, event = None ):
        item = self.GetSelection()
        if item is None: return

        layer = self.GetItemPyData( item )
        if layer is None:
            layer = self.root_layer
        else:
            layer = layer.Data

        self.root_layer.inbox.send(
            request = "replace_selection",
            layer = layer,
            object_indices = (),
            record_undo = False,
        )

    def mouse_pressed( self, event ):
        # If a selected item is clicked, unselect it so that it will be
        # selected again. This allows the user to click on an
        # already-selected layer to display its properties, for instance.
        event.Skip()
        selected_item = self.GetSelection()
        if selected_item is None: return

        ( clicked_item, flags ) = self.HitTest( event.GetPosition() )

        if clicked_item != selected_item or \
           flags & wx.TREE_HITTEST_ONITEMLABEL == 0:
            return

        self.ToggleItemSelection( selected_item ) 

    def mouse_right_released( self, event ):
        event.Skip()

        selected_item = self.GetSelection()
        ( clicked_item, flags ) = self.HitTest( event.GetPosition() )

        if clicked_item != selected_item or \
           flags & wx.TREE_HITTEST_ONITEMLABEL == 0:
            return

        layer = self.GetItemPyData( selected_item )
        if layer is None: return
        layer = layer.Data

        raisable = self.GetPrevSibling( selected_item ) is not None
        lowerable = self.GetNextSibling( selected_item ) is not None

        # For now, only top-level layers are deletable.
        layer_deletable = (
            selected_item.GetParent() == self.GetRootItem() and
            selected_item != self.none_item
        )

        self.PopupMenu(
            maproomlib.ui.Layer_context_menu(
                layer, raisable, lowerable, layer_deletable,
            ),
            event.GetPosition(),
        )

    def selection_updated( self, selections = None, raisable = False,
                           lowerable = False, deletable = False,
                           layer_deletable = None ):
        if len( selections ) != 1:
            return

        ( layer, indices ) = selections[ 0 ]

        if len( indices ) != 0:
            return

        new_item = self.layer_to_item.get( layer )
        if new_item is None: return

        item = self.GetSelection()
        if item and self.IsSelected( item ):
            self.ToggleItemSelection( item ) 

        self.ToggleItemSelection( new_item ) 

    def property_updated( self, layer, property ):
        if property.name != "Layer name":
            return

        item = self.layer_to_item.get( layer )
        if item is None: return

        self.SetItemText( item, property.value )
