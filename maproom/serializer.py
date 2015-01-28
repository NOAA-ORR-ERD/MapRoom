import os
import re

from peppy2.utils.runtime import get_all_subclasses


# shlex quote routine modified from python 3 to allow [ and ] unquoted for lists
_find_unsafe = re.compile(r'[^\w@%+=:,./[\]-]').search

def quote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s
    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return "'" + s.replace("'", "'\"'\"'") + "'"


class Serializer(object):
    def __init__(self):
        self.serialized_commands = []
    
    def __str__(self):
        lines = []
        for cmd in self.serialized_commands:
            lines.append(str(cmd))
        return "\n".join(lines)
    
    def add(self, cmd):
        sc = SerializedCommand(cmd)
        self.serialized_commands.append(sc)


class ArgumentConverter(object):
    stype = None
    
    def get_args(self, instance):
        """Return list of strings that can be used to reconstruct the instance
        """
        return str(instance),
    
    def instance_from_args(self, args):
        pass

class FileMetadataConverter(ArgumentConverter):
    stype = "file_metadata"
    
    def get_args(self, instance):
        return instance.uri, instance.mime
    
    def instance_from_args(self, args):
        pass


class LayerConverter(ArgumentConverter):
    stype = "layer"
    
    def get_args(self, instance):
        return instance.name,
    
    def instance_from_args(self, args):
        pass


class IntConverter(ArgumentConverter):
    stype = "int"
    
    def instance_from_args(self, args):
        pass


class FloatConverter(ArgumentConverter):
    stype = "float"
    
    def instance_from_args(self, args):
        # Note: handle None for e.g. triangle params q and a
        pass


class PointConverter(ArgumentConverter):
    stype = "point"
    
    def get_args(self, instance):
        return instance  # already a tuple
    
    def instance_from_args(self, args):
        pass


class RectConverter(ArgumentConverter):
    stype = "rect"
    
    def get_args(self, instance):
        (x1, y1), (x2, y2) = instance
        return x1, y1, x2, y2
    
    def instance_from_args(self, args):
        pass


class ListIntConverter(ArgumentConverter):
    stype = "rect"
    
    def get_args(self, instance):
        text = ",".join([str(i) for i in instance])
        return "[%s]" % text,
    
    def instance_from_args(self, args):
        pass


def get_converters():
    s = get_all_subclasses(ArgumentConverter)
    c = {}
    for cls in s:
        if cls.stype is not None:
            c[cls.stype] = cls()
    return c

class SerializedCommand(object):
    converters = get_converters()
    
    def __init__(self, cmd):
        self.cmd_name = cmd.get_serialized_name()
        p = []
        for name, stype in [(n[0], n[1]) for n in cmd.serialize_order]:
            p.append((stype, getattr(cmd, name)))
        self.params = p

    def __str__(self):
        output = []
        for stype, value in self.params:
            print stype, value
            try:
                c = self.converters[stype]
                values = c.get_args(value)
            except KeyError:
                values = [value]
            string_values = [quote(str(v)) for v in values]
            output.append(" ".join(string_values))
        
        text = " ".join(output)
        return "%s %s" % (self.cmd_name, text)
