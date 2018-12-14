import re
import shlex

from omnivore_framework.utils.runtime import get_all_subclasses
from omnivore_framework.utils.file_guess import FileMetadata

from . import command
from . import magic
from .layers.style import LayerStyle

import logging
log = logging.getLogger(__name__)


class UnknownCommandError(RuntimeError):
    pass


class Serializer(object):
    known_commands = None

    def __init__(self):
        self.serialized_commands = []

    def __str__(self):
        lines = [magic.magic_header]
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
        lines = text.splitlines(True)
        self.header = lines.pop(0)
        self.lines = lines
        if not self.header.startswith(magic.magic_template):
            raise RuntimeError("Not a MapRoom log file!")
        self.version = int(self.header[len(magic.magic_template):])
        if self.version == 1:
            self.layer_offset -= 1

    def iter_cmds(self, manager):
        build_multiline = ""
        for line in self.lines:
            if build_multiline:
                line = build_multiline + line
                build_multiline = ""
            else:
                if not line.strip() or line.startswith("#"):
                    continue
            try:
                text_args = shlex.split(line.strip())
            except ValueError:
                build_multiline = line
                continue
            cmd = self.unserialize_line(text_args, manager)
            yield cmd

    def unserialize_line(self, text_args, manager):
        short_name = text_args.pop(0)
        log.debug("unserialize: short_name=%s, args=%s" % (short_name, text_args))
        cmd_cls = Serializer.get_command(short_name)
        cmd_args = []
        for name, stype, default_val in [(n[0], n[1], n[2] if len(n) > 2 else None) for n in cmd_cls.serialize_order]:
            log.debug("  name=%s, type=%s" % (name, stype))
            converter = SerializedCommand.get_converter(stype)
            try:
                arg = converter.instance_from_args(text_args, manager, self)
                log.debug("  converter=%s: %s" % (converter.__class__.__name__, repr(arg)))
            except IndexError:
                arg = default_val
                log.debug("  converter=%s: %s (using default value)" % (converter.__class__.__name__, repr(arg)))
            cmd_args.append(arg)
        log.debug("COMMAND: %s(%s)" % (cmd_cls.__name__, ",".join([repr(a) for a in cmd_args])))
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
        return instance.uri, instance.mime

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
            invariant = int(val)
            log.debug(f"LayerConverter: deserialized invariant={invariant}")
            if invariant < 0:
                if deserializer.version == 1 and invariant == -3:
                    invariant = manager.transient_invariant
            else:
                # only add layer offset on permanent layers. Transient layers
                # will have invariants less than zero and must be referenced
                # that way.
                invariant += deserializer.layer_offset
            log.debug(f"LayerConverter: actual invariant={invariant}")
            layer = manager.get_layer_by_invariant(invariant)
        except:
            # Old way: save layer references by name
            layer = manager.get_layer_by_name(val)
        return layer


class TextConverter(ArgumentConverter):
    stype = "text"

    def get_args(self, instance):
        """Return list of strings that can be used to reconstruct the instance
        """
        return instance,

    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        return text


class BoolConverter(ArgumentConverter):
    stype = "bool"

    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        if text == "None":
            return None
        if text == "True" or text == "1":
            return True
        return False


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
        if instance is None:
            return None,
        return instance  # already a tuple

    def instance_from_args(self, args, manager, deserializer):
        lon = args.pop(0)
        if lon == "None":
            return None
        lat = args.pop(0)
        return (float(lon), float(lat))


class PointsConverter(ArgumentConverter):
    stype = "points"

    def get_args(self, instance):
        text = ",".join(["(%s,%s)" % (str(i[0]), str(i[1])) for i in instance])
        return "[%s]" % text,

    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0).lstrip("[").rstrip("]")
        if text:
            text = text.lstrip("(").rstrip(")")
            tuples = text.split("),(")
            points = []
            for t in tuples:
                lon, lat = t.split(",", 1)
                points.append((float(lon), float(lat)))
            return points
        return []


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

    def get_values_from_list(self, vals, manager, deserializer):
        return [int(i) for i in vals]

    def instance_from_args(self, args, manager, deserializer):
        text = args.pop(0)
        if text.startswith("["):
            text = text[1:]
        if text.endswith("]"):
            text = text[:-1]
        if text:
            vals = text.split(",")
            return self.get_values_from_list(vals, manager, deserializer)
        return []


class LayersConverter(ListIntConverter):
    stype = "layers"

    def get_values_from_list(self, vals, manager, deserializer):
        layers = []
        for i in vals:
            invariant = int(i) + deserializer.layer_offset
            layer = manager.get_layer_by_invariant(invariant)
            print("invariant=%d, layer=%s" % (invariant, layer))
            layers.append(layer)
        return layers


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
