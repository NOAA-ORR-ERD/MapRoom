import itertools


def parse_int_string(nputstr=""):
    """Return list of integers from comma separated ranges
    
    Modified from http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-
    of-numbers-in-python.html to return ranges in order specified, rather than
    sorting the entire resulting list
    
    >>> parse_int_string("1,4,9-12")
    [1, 4, 9, 10, 11, 12]
    >>> parse_int_string("4,5,1,6")
    [4, 5, 1, 6]
    >>> parse_int_string("4,6-8,1,5,2")
    [4, 6, 7, 8, 1, 5, 2]
    """
    selection = []
    error = ""
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.append(int(i))
        except ValueError:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.append(x)
            except ValueError:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    if invalid:
        error = u"Invalid range: " + " ".join([unicode(i) for i in invalid])
    # ordered = list(selection)
    # ordered.sort()
    return selection, error


def int_list_to_string(values):
    # from http://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
    def ranges(i):
        for a, b in itertools.groupby(enumerate(i), lambda (x, y): y - x):
            b = list(b)
            yield b[0][1], b[-1][1]

    items = []
    for r in ranges(values):
        if r[0] < r[1]:
            items.append("%d-%d" % (r[0], r[1]))
        else:
            items.append(str(r[0]))
    return ",".join(items)
