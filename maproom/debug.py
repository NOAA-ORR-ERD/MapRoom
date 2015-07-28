def debug_objects(lm):
    import layers
#    a = layers.OverlayImageObject(manager=lm)
#    a.set_location((-16.6637485204,-1.40163099748))
#    a.set_style(lm.default_style)
#    lm.insert_layer([2], a)
#    
    a = layers.OverlayTextObject(manager=lm)
    a.set_location((6.6637485204,5.40163099748))
    a.set_style(lm.default_style)
    lm.insert_layer([2], a)
    
    #a = layers.ScaledImageObject(manager=lm)
    b = layers.RectangleVectorObject(manager=lm)
    b.set_opposite_corners(
        (-16.6637485204,-1.40163099748),
        (9.65688930428,-19.545688433))
    b.set_style(lm.default_style)
    lm.insert_layer([2], b)
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((0,0), (0,3))
    c.copy_control_point_from(0, a, 0)
    c.copy_control_point_from(1, b, 1)
    c.set_style(lm.default_style)
    lm.insert_layer([2], c)
    lm.set_control_point_link(c, 0, a, 0)
    lm.update_linked_control_points()
    lm.set_control_point_link(c, 1, b, 1)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-20,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    lm.insert_layer([2], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-15,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    lm.insert_layer([2], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-10,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    lm.insert_layer([2], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
#    a = layers.PolylineObject(manager=lm)
#    a.set_points([
#        (-15,-2),
#        (5, -8),
#        (10, -20),
#        (8, -5),
#        (-17, -10),
#        ])
#    a.set_style(lm.default_style)
#    a.style.fill_style = 0
#    lm.insert_layer([2], a)
