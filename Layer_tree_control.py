import os
import wx
try:
    import wx.lib.agw.customtreectrl as treectrl
except ImportError:
    import wx.lib.customtreectrl as treectrl
import Layer
import app_globals

class Layer_tree_control( treectrl.CustomTreeCtrl ):
    
    dragged_item = None
    
    def __init__( self, parent_window ):
        treectrl.CustomTreeCtrl.__init__(
            self, parent_window, wx.ID_ANY,
            style = treectrl.TR_DEFAULT_STYLE | wx.BORDER_SUNKEN
        )
        
        self.Bind( treectrl.EVT_TREE_ITEM_CHECKED, self.handle_item_checked )
        self.Bind( treectrl.EVT_TREE_ITEM_COLLAPSED, self.handle_item_collapsed )
        self.Bind( treectrl.EVT_TREE_ITEM_EXPANDED, self.handle_item_expanded )
        self.Bind( treectrl.EVT_TREE_BEGIN_DRAG, self.handle_begin_drag )
        self.Bind( treectrl.EVT_TREE_END_DRAG, self.handle_end_drag )
        self.Bind( treectrl.EVT_TREE_SEL_CHANGED, self.handle_selection_changed )
        
        """
        self.state_image_list = wx.ImageList( self.IMAGE_SIZE, self.IMAGE_SIZE )
        #self.state_image_list.Add( wx.Bitmap( "maproom/ui/images/maproom.png", wx.BITMAP_TYPE_PNG ) )
        #self.SetImageList( self.state_image_list )
        
        self.Bind( wx.EVT_LEFT_DOWN, self.mouse_pressed )
        self.Bind( wx.EVT_RIGHT_DOWN, self.mouse_pressed )
        self.Bind( wx.EVT_RIGHT_UP, self.mouse_right_released )
        """
    
    def get_selected_layer( self ):
        item = self.GetSelection()
        if ( item == None ):
            return None
        ( category, layer ) = self.GetItemPyData( item ).Data
        
        return layer
    
    def select_layer( self, layer ):
        app_globals.editor.esc_key_pressed()
        
        if ( layer == None ):
            self.UnselectAll()
        else:
            self.select_layer_recursive( layer, self.GetRootItem() )
    
    def select_layer_recursive( self, layer, item ):
        ( category, item_layer ) = self.GetItemPyData( item ).Data
        
        if ( item_layer == layer ):
            self.SelectItem( item, True )
            # also make sure the layer's name is up-to-date
            self.SetItemText( item, layer.name )
            
            return True
        
        if ( not self.ItemHasChildren( item ) ):
            return False
        
        n = self.GetChildrenCount( item, False )
        # apparently GetFirstChild() returns a tuple of the
        # child and an integer (I think the integer is the "cookie" for calling GetNextChild())
        child = self.GetFirstChild( item )[ 0 ]
        
        while ( n > 0 ):
            if ( self.select_layer_recursive( layer, child ) ):
                return True
            
            child = self.GetNextSibling( child )
            n -= 1
        
        return False
    
    def rebuild( self ):
        # rebuild the tree from the layer manager's data
        lm = app_globals.layer_manager
        self.DeleteAllItems()
        # self.Freeze()
        print "rebuiding layers = " + str( lm.layers )
        self.add_layers_recursive( lm.layers, None )
        # self.Thaw()
        app_globals.application.layer_tree_selection_changed()
    
    def add_layers_recursive( self, layer_tree, parent ):
        if ( len( layer_tree ) == 0 ):
            return
        
        # we assume the layer at the start of each list is a folder
        folder_node = self.add_layer( layer_tree[ 0 ], parent )
        # import code; code.interact( local = locals() )
        
        for item in layer_tree[ 1 : ]:
            if ( isinstance( item, Layer.Layer ) ):
                node = self.add_layer( item, folder_node )
            else:
                self.add_layers_recursive( item, folder_node )
        
        if ( layer_tree[ 0 ].is_expanded ):
            self.Expand( folder_node )
    
    def add_layer( self, layer, parent ):
        if ( parent == None ):
            data = wx.TreeItemData()
            data.SetData( ( "root", layer ) )
            return self.AddRoot( layer.name, data = data )
        
        data = wx.TreeItemData()
        if ( layer.type == "folder" ):
            data.SetData( ( "folder", layer ) )
        else:
            data.SetData( ( "layer", layer ) )
        
        item = self.AppendItem( parent, layer.name, ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
        self.CheckItem2( item, layer.is_visible )
        
        # add sub-items depending on what the layer has
        
        if ( layer.type == ".bna" ):
            return item
        
        if ( layer.images != None ):
            data = wx.TreeItemData()
            data.SetData( ( "images", layer ) )
            subitem = self.AppendItem( item, "images", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.images_visible )
        
        if ( layer.polygons != None ):
            data = wx.TreeItemData()
            data.SetData( ( "polygons", layer ) )
            subitem = self.AppendItem( item, "polygons", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.polygons_visible )
        
        if ( layer.points != None ):
            data = wx.TreeItemData()
            data.SetData( ( "points", layer ) )
            subitem = self.AppendItem( item, "points", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.points_visible )
        
        if ( layer.line_segment_indexes != None ):
            data = wx.TreeItemData()
            data.SetData( ( "lines", layer ) )
            subitem = self.AppendItem( item, "line segments", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.line_segments_visible )
        
        if ( layer.triangles != None ):
            data = wx.TreeItemData()
            data.SetData( ( "triangles", layer ) )
            subitem = self.AppendItem( item, "triangles", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.triangles_visible )
        
        if ( layer.label_set_renderer != None ):
            data = wx.TreeItemData()
            data.SetData( ( "labels", layer ) )
            subitem = self.AppendItem( item, "labels", ct_type = treectrl.TREE_ITEMTYPE_CHECK, data = data )
            self.CheckItem2( subitem, layer.labels_visible )
        
        if ( layer.is_expanded ):
            self.Expand( item )
        
        return item
    
    def handle_item_checked( self, event ):
        ( category, layer ) = self.GetItemPyData( event.GetItem() ).Data
        item = event.GetItem()
        checked = self.IsItemChecked( item )
        if ( category == "layer" ):
            layer.is_visible = checked
        elif ( category == "images" ):
            layer.images_visible = checked
        elif ( category == "polygons" ):
            layer.polygons_visible = checked
        elif ( category == "points" ):
            layer.points_visible = checked
        elif ( category == "lines" ):
            layer.line_segments_visible = checked
        elif ( category == "triangles" ):
            layer.triangles_visible = checked
        elif ( category == "labels" ):
            layer.labels_visible = checked
        app_globals.application.refresh()
        event.Skip()
    
    def handle_item_collapsed( self, event ):
        pd = self.GetItemPyData( event.GetItem() )
        if ( pd == None ):
            return
        
        ( category, layer ) = pd.Data
        layer.is_expanded = False
    
    def handle_item_expanded( self, event ):
        pd = self.GetItemPyData( event.GetItem() )
        if ( pd == None ):
            return
        
        ( category, layer ) = pd.Data
        layer.is_expanded = True
    
    def handle_begin_drag( self, event ):
        ( category, layer ) = self.GetItemPyData( event.GetItem() ).Data
        item = event.GetItem()
        checked = self.IsItemChecked( item )
        if ( category == "folder" or category == "layer" ):
            event.Allow()
            self.dragged_item = item
    
    def handle_end_drag( self, event ):
        item = event.GetItem()
        local_dragged_item = self.dragged_item
        self.dragged_item = None
        
        # if we dropped somewhere that isn't on top of an item, ignore the event
        if not item.IsOk():
            return
        
        ( target_category, target_layer ) = self.GetItemPyData( item ).Data
        if ( target_category != "root" and target_category != "folder" and target_category != "layer" ):
            wx.MessageDialog(
                app_globals.application.frame,
                caption = "Invalid Layer Drag",
                message = "You can only drag a layer onto another layer, a folder, or the tree root.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            
            return
        
        ( source_category, source_layer ) = self.GetItemPyData( local_dragged_item ).Data
        lm = app_globals.layer_manager
        mi_source = lm.get_multi_index_of_layer( source_layer )
        # here we "re-get" the source layer so that if it's a folder layer what we'll get is the
        # folder's list, not just the folder pseudo-layer
        source_layer = lm.get_layer_by_multi_index( mi_source )
        
        mi_target = lm.get_multi_index_of_layer( target_layer )
        
        if ( mi_target == mi_source ):
            return
        
        if ( len( mi_target ) > len( mi_source ) and mi_target[ 0 : len( mi_source ) ] == mi_source ):
            wx.MessageDialog(
                app_globals.application.frame,
                caption = "Invalid Layer Move",
                message = "You cannot move folder into one of its sub-folders.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            
            return
        
        lm.remove_layer( mi_source )
        
        # re-get the multi_index for the target, because it may have changed when the layer was removed
        mi_target = lm.get_multi_index_of_layer( target_layer )
        # if we are inserting onto a folder, insert as the second item in the folder
        # (the first item in the folder is the folder pseudo-layer)
        if ( target_category == "root" ):
            mi_target = [ 1 ]
        if ( target_category == "folder" ):
            mi_target.append( 1 )
        lm.insert_layer( mi_target, source_layer )
    
    def handle_selection_changed( self, event ):
        app_globals.editor.esc_key_pressed()
        app_globals.application.layer_tree_selection_changed()
    
    def is_selected_layer_raisable( self ):
        item = self.GetSelection()
        if item:
            ( category, layer ) = self.GetItemPyData( item ).Data
            if ( category != "layer" and category != "folder" ):
                return False
            lm = app_globals.layer_manager
            mi = lm.get_multi_index_of_layer( layer )
        
            return mi[ len( mi ) - 1 ] >= 2
        return False
    
    def is_selected_layer_lowerable( self ):
        item = self.GetSelection()
        if item: 
            ( category, layer ) = self.GetItemPyData( item ).Data
            if ( category != "layer" and category != "folder" ):
                return False
            lm = app_globals.layer_manager
            mi = lm.get_multi_index_of_layer( layer )
            n = mi[ len( mi ) - 1 ]
            mi2 = mi[ : len( mi ) - 1 ]
            parent_list = lm.get_layer_by_multi_index( mi2 )
            total = len( parent_list )
            
            return n < ( total - 1 )
        return False
    
    def raise_selected_layer( self ):
        if ( not self.is_selected_layer_raisable() ):
            return
        
        self.move_selected_layer( -1 )
    
    def lower_selected_layer( self ):
        if ( not self.is_selected_layer_lowerable() ):
            return
        
        self.move_selected_layer( 1 )
    
    def move_selected_layer( self, delta ):
        item = self.GetSelection()
        ( category, layer ) = self.GetItemPyData( item ).Data
        lm = app_globals.layer_manager
        mi_source = lm.get_multi_index_of_layer( layer )
        mi_target = mi_source[ : len( mi_source ) - 1 ]
        mi_target.append( mi_source[ len( mi_source ) - 1 ] + delta )
        
        # here we "re-get" the source layer so that if it's a folder layer what we'll get is the
        # folder's list, not just the folder pseudo-layer
        source_layer = lm.get_layer_by_multi_index( mi_source )
        lm.remove_layer( mi_source )
        lm.insert_layer( mi_target, source_layer )
        app_globals.editor.esc_key_pressed()
    
    """
    def add( self, layer, hidden = False, insert_index = None, parent = None ):
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
            insert_index = self.GetChildrenCount( parent, recursively = False ) - insert_index

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

        if clicked_item != selected_item or flags & wx.TREE_HITTEST_ONITEMLABEL == 0:
            return

        self.ToggleItemSelection( selected_item ) 

    def mouse_right_released( self, event ):
        event.Skip()

        selected_item = self.GetSelection()
        ( clicked_item, flags ) = self.HitTest( event.GetPosition() )

        if clicked_item != selected_item or flags & wx.TREE_HITTEST_ONITEMLABEL == 0:
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
    """
