def debug_objects(lm):
    import layers
    
    a = layers.AnnotationLayer(manager=lm)
    lm.insert_layer([3], a)
    
    a = layers.OverlayTextObject(manager=lm)
    a.set_location((6.6637485204,5.40163099748))
    a.set_style(lm.default_style)
    lm.insert_layer([3, 999], a)
    
    #a = layers.ScaledImageObject(manager=lm)
    b = layers.RectangleVectorObject(manager=lm)
    b.set_opposite_corners(
        (-16.6637485204,-1.40163099748),
        (9.65688930428,-19.545688433))
    b.set_style(lm.default_style)
    lm.insert_layer([3, 999], b)
    
    b = layers.CircleVectorObject(manager=lm)
    b.set_opposite_corners(
        (14.0, 4.0),
        (18.0, 8.0))
    b.set_style(lm.default_style)
    lm.insert_layer([3, 999], b)
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((0,0), (0,3))
    c.copy_control_point_from(0, a, 0)
    c.copy_control_point_from(1, b, 1)
    c.set_style(lm.default_style)
    c.style.line_start_marker = 1
    c.style.line_end_marker = 2
    lm.insert_layer([3, 999], c)
    lm.set_control_point_link(c, 0, a, 0)
    lm.update_linked_control_points()
    lm.set_control_point_link(c, 1, b, 1)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-20,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    c.style.line_start_marker = 1
    c.style.line_end_marker = 2
    lm.insert_layer([3, 999], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-15,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    c.style.line_start_marker = 1
    c.style.line_end_marker = 2
    lm.insert_layer([3, 999], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
    c = layers.LineVectorObject(manager=lm)
    c.set_opposite_corners((-10,10), (0,3))
    c.copy_control_point_from(1, b, 3)
    c.set_style(lm.default_style)
    c.style.line_start_marker = 1
    c.style.line_end_marker = 2
    lm.insert_layer([3, 999], c)
    lm.set_control_point_link(c, 1, b, 3)
    lm.update_linked_control_points()
    
    a = layers.PolylineObject(manager=lm)
    a.set_points([
        (15,12),
        (25, 18),
        (10, 30),
        (8, 15),
        (17, 10),
        ])
    a.set_style(lm.default_style)
    a.style.fill_style = 0
    lm.insert_layer([3, 999], a)
    
    a = layers.PolygonObject(manager=lm)
    a.set_points([
        (0, 25),
        (-10, 30),
        (-20, 20),
        (-10, 15),
        (2, 20),
        ])
    a.set_style(lm.default_style)
    a.style.line_stipple = 0
    a.style.fill_color = a.style.default_colors[2]
    lm.insert_layer([3, 999], a)
    
    a = layers.OverlayIconObject(manager=lm)
    a.set_location((20, -10))
    a.set_style(lm.default_style)
    a.style.icon_marker = 186
    lm.insert_layer([3, 999], a)
    
#    a = layers.TileLayer(manager=lm)
#    a = layers.WMSLayer(manager=lm)
#    lm.insert_layer([4], a)
