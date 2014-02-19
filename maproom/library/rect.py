# functions for rects (screen rects, world rects, etc.)
# a rect is a rect stored as ( ( l, b ), ( r, t ) )

NONE_RECT = ((None, None), (None, None))
EMPTY_RECT = ((0, 0), (0, 0))


def is_rect_empty(r):
    return r[ 0 ][ 0 ] == None or r[ 0 ][ 1 ] == None or r[ 1 ][ 0 ] == None or r[ 1 ][ 1 ] == None or \
        width(r) <= 0 or height(r) <= 0


def width(r):
    return r[1][0] - r[0][0]


def height(r):
    return r[1][1] - r[0][1]


def size(r):
    return (width(r), height(r))


def center(r):
    return ((r[0][0] + r[1][0]) / 2.0, (r[0][1] + r[1][1]) / 2.0)


def contains_point(r, p):
    return p[0] >= r[0][0] and p[0] <= r[1][0] and p[1] >= r[0][1] and p[1] <= r[1][1]

# does r1 fully contain r2?


def contains_rect(r1, r2):
    return contains_point(r1, r2[0]) and contains_point(r1, r2[1])


def zoom_in_2(r):
    w = width(r)
    h = height(r)
    c = center(r)

    return ((c[0] - w / 4, c[1] - h / 4), (c[0] + w / 4, c[1] + h / 4))


def zoom_out_2(r):
    w = width(r)
    h = height(r)
    c = center(r)

    return ((c[0] - w, c[1] - h), (c[0] + w, c[1] + h))


def accumulate_rect(r1, r2):
    # print "in accumulate_rect() r1 = " + str( r1 ) + ", r2 = " + str( r2 )
    if (r1[0][0] == None):
        l = r2[0][0]
    elif (r2[0][0] == None):
        l = r1[0][0]
    else:
        l = min(r1[0][0], r2[0][0])

    if (r1[1][0] == None):
        r = r2[1][0]
    elif (r2[1][0] == None):
        r = r1[1][0]
    else:
        r = max(r1[1][0], r2[1][0])

    if (r1[0][1] == None):
        b = r2[0][1]
    elif (r2[0][1] == None):
        b = r1[0][1]
    else:
        b = min(r1[0][1], r2[0][1])

    if (r1[1][1] == None):
        t = r2[1][1]
    elif (r2[1][1] == None):
        t = r1[1][1]
    else:
        t = max(r1[1][1], r2[1][1])

    return ((l, b), (r, t))


def accumulate_point(r, p):
    if (r[0][0] == None):
        l = p[0]
    elif (p[0] == None):
        l = r[0][0]
    else:
        l = min(r[0][0], p[0])

    if (r[1][0] == None):
        rt = p[0]
    elif (p[0] == None):
        rt = r[1][0]
    else:
        rt = max(r[1][0], p[0])

    if (r[0][1] == None):
        b = p[1]
    elif (p[1] == None):
        b = r[0][1]
    else:
        b = min(r[0][1], p[1])

    if (r[1][1] == None):
        t = p[1]
    elif (p[1] == None):
        t = r[1][1]
    else:
        t = max(r[1][1], p[1])

    return ((l, b), (rt, t))


def rect_accumulate_rect_safe(r1, r2):
    l = min(r1[0][0], r2[0][0])
    r = max(r1[1][0], r2[1][0])
    b = min(r1[0][1], r2[0][1])
    t = max(r1[1][1], r2[1][1])

    return ((l, b), (r, t))


def rect_accumulate_point_safe(r, p):
    l = min(r[0][0], p[0])
    r = max(r[1][0], p[0])
    b = min(r[0][1], p[1])
    t = max(r[1][1], p[1])

    return ((l, b), (r, t))


def get_normalized_coordinates(point_a, point_b):
    return (min(point_a[0], point_b[0]),
            min(point_a[1], point_b[1]),
            max(point_a[0], point_b[0]),
            max(point_a[1], point_b[1]))
