
convert_units_map = {
    "feet": {
        "feet": 1.0,
        "fathoms": 1.0 / 6.0,
        "meters": 0.3048,
        },
    "fathoms": {
        "feet": 6.0,
        "fathoms": 1.0,
        "meters": 1.8288,
        },
    "meters": {
        "feet": 3.28084,
        "fathoms": 0.546807,
        "meters": 1.0,
        },
}


def convert_units(depths, from_units, to_units):
    if not from_units or from_units == "unknown" or not to_units or to_units == "unknown":
        return
    factor = convert_units_map[from_units][to_units]
    depths *= factor
