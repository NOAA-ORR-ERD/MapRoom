import os
import re
import shlex

from peppy2.utils.runtime import get_all_subclasses
from peppy2.utils.file_guess import FileMetadata

import command
from layers.style import LayerStyle

magic_version = 1
magic_template = "# NOAA MapRoom Command File, v"
magic_header = "%s%d" % (magic_template, magic_version)

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


class UnknownCommandError(RuntimeError):
    pass


class Serializer(object):
    known_commands = None
    
    def __init__(self):
        self.serialized_commands = []
    
    def __str__(self):
        lines = [magic_header]
        for cmd in self.serialized_commands:
            lines.append(str(cmd))
        return "\n".join(lines)
    
    def add(self, cmd):
        sc = SerializedCommand(cmd)
        self.serialized_commands.append(sc)
    
    @classmethod
    def get_command(cls, short_name):
        if cls.known_commands is None:
            cls.known_commands = command.get_known_commands()
        try:
            return cls.known_commands[short_name]
        except KeyError:
            return UnknownCommandError(short_name)


class TextDeserializer(object):
    def __init__(self, text, layer_offset=0):
        self.layer_offset = layer_offset
        lines = text.splitlines()
        self.header = lines.pop(0)
        self.lines = lines
        if not self.header.startswith(magic_template):
            raise RuntimeError("Not a MapRoom log file!")
    
    def iter_cmds(self, manager):
        for line in self.lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cmd = self.unserialize_line(line, manager)
            yield cmd
    
    def unserialize_line(self, line, manager):
        text_args = shlex.split(line)
        short_name = text_args.pop(0)
        cmd_cls = Serializer.get_command(short_name)
        cmd_args = []
        for name, stype in cmd_cls.serialize_order:
            converter = SerializedCommand.get_converter(stype)
            arg = converter.instance_from_args(text_args, manager, self)
            cmd_args.append(arg)
        cmd = "COMMAND: %s(%s)" % (cmd_cls.__name__, ",".join([str(a) for a in cmd_args]))
        cmd = cmd_cls(*cmd_args)
        return cmd


class ArgumentConverter(object):
    stype = None  # Default converter just uses strings
    
    def get_args(self, instance):
        """Return list of strings that can be used to reconstruct the instance
        """
        return str(instance),
    
    def instance_from_args(self, args, manager, deserializer):
        arg = args.pop(0)
        return arg


class FileMetadataConverter(ArgumentConverter):
    stype = "file_metadata"
    
    def get_args(self, instance):
        # Force forward slashes on windows so to prevent backslash escape chars
        return os.path.normpath(instance.uri).replace('\\', '/'), instance.mime
    
    def instance_from_args(self, args, manager, deserializer):
        uri = args.pop(0)
        mime = args.pop(0)
        return FileMetadata(uri=uri, mime=mime)


class LayerConverter(ArgumentConverter):
    stype = "layer"
    
    def get_args(self, instance):
        return instance,
    
    def instance_from_args(self, args, manager, deserializer):
        val = args.pop(0)
        try:
            id = int(val)
            layer = manager.get_layer_by_invariant(id + deserializer.layer_offset)
        except:
            # Old way: save layer references by name
            layer = manager.get_layer_by_name(val)
        return layer


class IntConverter(ArgumentConverter):
    stype = "int"
    
    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        if text == "None":
            return None
        return int(text)


class FloatConverter(ArgumentConverter):
    stype = "float"
    
    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        if text == "None":
            return None
        return float(text)


class PointConverter(ArgumentConverter):
    stype = "point"
    
    def get_args(self, instance):
        return instance  # already a tuple
    
    def instance_from_args(self, args, manager, deserializer):
        lon = args.pop(0)
        lat = args.pop(0)
        return (float(lon), float(lat))


class RectConverter(ArgumentConverter):
    stype = "rect"
    
    def get_args(self, instance):
        (x1, y1), (x2, y2) = instance
        return x1, y1, x2, y2
    
    def instance_from_args(self, args, manager, deserializer):
        x1 = args.pop(0)
        y1 = args.pop(0)
        x2 = args.pop(0)
        y2 = args.pop(0)
        return ((x1, y1), (x2, y2))


class ListIntConverter(ArgumentConverter):
    stype = "list_int"
    
    def get_args(self, instance):
        text = ",".join([str(i) for i in instance])
        return "[%s]" % text,
    
    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        if text.startswith("["):
            text = text[1:]
        if text.endswith("]"):
            text = text[:-1]
        if text:
            vals = text.split(",")
            return [int(i) for i in vals]
        return []


class StyleConverter(ArgumentConverter):
    stype = "style"
    
    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        style = LayerStyle()
        style.parse(text)
        return style


def get_converters():
    s = get_all_subclasses(ArgumentConverter)
    c = {}
    for cls in s:
        c[cls.stype] = cls()
    c[None] = ArgumentConverter()  # Include default converter
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
    
    @classmethod
    def get_converter(cls, stype):
        try:
            return cls.converters[stype]
        except KeyError:
            return cls.converters[None]
