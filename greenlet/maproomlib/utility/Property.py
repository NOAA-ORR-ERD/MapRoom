import numpy as np


class Property:
    """
    A value wrapped with some basic metadata and validation.
    """
    MAX_LIST_ELEMENTS_TO_SHOW_IN_STR = 4

    def __init__( self, name, value, type = None, choices = None,
                  min = None, max = None, mutable = True, **extra_info ):
        self.name = name
        self.type = type
        self.choices = choices
        self.min = min
        self.max = max

        # The mutable flag is not actually enforced here. It's present so
        # that the UI can make the displayed field non-mutable if necessary.
        self.mutable = mutable

        for ( key, val ) in extra_info.items():
            setattr( self, key, val )

        self.value = self.validate( value )

    def update( self, value ):
        self.value = self.validate( value )

    def validate( self, value ):
        if self.type is not None:
            try:
                value = self.type( value )
            except ValueError:
                raise ValueError(
                    "The value is invalid."
                )

        if self.choices and value not in self.choices:
            raise ValueError(
                "The value is not one of the available choices.",
            )

        if self.min is not None and value < self.min:
            raise ValueError(
                "The value is too small.",
            )

        if self.max is not None and value > self.max:
            raise ValueError(
                "The value is too large.",
            )

        return value

    def __str__( self ):
        if self.type == np.float32 and np.isnan( self.value ):
            return ""

        if hasattr( self.value, "__iter__" ):
            if len( self.value ) > self.MAX_LIST_ELEMENTS_TO_SHOW_IN_STR:
                value = self.value[ : self.MAX_LIST_ELEMENTS_TO_SHOW_IN_STR ]
                return ", ".join( [ str( x ) for x in sorted( value ) ] ) + \
                       ", ..."
            else:
                return ", ".join( [ str( x ) for x in sorted( self.value ) ] )

        return self.value.__str__()
