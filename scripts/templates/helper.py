"""
 Copyright (C) 2022-2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import re
import sys
import util

# allow imports from top-level scripts directory
sys.path.append("..")

"""
    Extracts traits from a spec object
"""
class obj_traits:

    @staticmethod
    def is_function(obj):
        try:
            return True if re.match(r"function", obj['type']) else False
        except:
            return False

    @staticmethod
    def is_class(obj):
        try:
            return True if re.match(r"class", obj['type']) else False
        except:
            return False

    @staticmethod
    def is_handle(obj):
        try:
            return True if re.match(r"handle", obj['type']) else False
        except:
            return False

    @staticmethod
    def is_enum(obj):
        try:
            return True if re.match(r"enum", obj['type']) else False
        except:
            return False

    @staticmethod
    def is_experimental(obj):
        try:
            return True if re.search("Exp$", obj['name']) else False
        except:
            return False

    @staticmethod
    def class_name(obj):
        try:
            return obj['class']
        except:
            return None

    @staticmethod
    def is_loader_only(obj):
        try:
            return obj['loader_only']
        except:
            return False


"""
    Extracts traits from a class name
"""
class class_traits:

    @staticmethod
    def is_global(name, tags):
        try:
            return True if name in tags else False
        except:
            return False

    @staticmethod
    def is_namespace(name, namespace, tags):
        try:
            return tags[name] == namespace
        except:
            return False

    @staticmethod
    def is_singleton(item):
        try:
            return "singleton" == item['attribute']
        except:
            return False

    @staticmethod
    def get_handle(item, meta):
        try:
            return meta['class'][item['name']]['handle'][0]
        except:
            return ""

"""
    Extracts traits from a type name
"""
class type_traits:
    RE_HANDLE   = r"(.*)handle_t"
    RE_IPC      = r"(.*)ipc(.*)handle_t"
    RE_POINTER  = r"(.*\w+)\*+"
    RE_PPOINTER = r"(.*\w+)\*{2,}"
    RE_DESC     = r"(.*)desc_t.*"
    RE_PROPS    = r"(.*)properties_t.*"
    RE_FLAGS    = r"(.*)flags_t"
    RE_ARRAY    = r"(.*)\[([1-9][0-9]*)\]"

    @staticmethod
    def base(name):
        return _remove_const_ptr(name)

    @classmethod
    def is_handle(cls, name):
        try:
            return True if re.match(cls.RE_HANDLE, name) else False
        except:
            return False

    @classmethod
    def is_pointer_to_pointer(cls, name):
        try:
            return True if re.match(cls.RE_PPOINTER, name) else False
        except:
            return False

    @classmethod
    def is_ipc_handle(cls, name):
        try:
            return True if re.match(cls.RE_IPC, name) else False
        except:
            return False

    @staticmethod
    def is_class_handle(name, meta):
        try:
            name = _remove_const_ptr(name)
            return len(meta['handle'][name]['class']) > 0
        except:
            return False

    @classmethod
    def is_pointer(cls, name):
        try:
            return True if re.match(cls.RE_POINTER, name) else False
        except:
            return False

    @classmethod
    def is_descriptor(cls, name):
        try:
            return True if re.match(cls.RE_DESC, name) else False
        except:
            return False

    @classmethod
    def is_properties(cls, name):
        try:
            return True if re.match(cls.RE_PROPS, name) else False
        except:
            return False

    @classmethod
    def is_flags(cls, name):
        try:
            return True if re.match(cls.RE_FLAGS, name) else False
        except:
            return False

    @classmethod
    def get_flag_type(cls, name):
        return re.sub(r"(\w+)_flags_t", r"\1_flag_t", name)

    @staticmethod
    def is_known(name, meta):
        try:
            name = _remove_const_ptr(name)
            for group in meta:
                if name in meta[group]:
                    return True
            return False
        except:
            return False

    @staticmethod
    def is_enum(name, meta):
        try:
            name = _remove_const_ptr(name)
            if name in meta['enum']:
                return True
            return False
        except:
            return False

    @staticmethod
    def is_funcptr(name, meta):
        return name in meta['fptr_typedef']

    @staticmethod
    def is_struct(name, meta):
        try:
            name = _remove_const_ptr(name)
            if name in meta['struct']:
                return True
            return False
        except:
            return False

    @staticmethod
    def find_class_name(name, meta):
        try:
            name = _remove_const_ptr(name)
            for group in meta:
                if name in meta[group]:
                    return meta[group][name]['class']
            return None
        except:
            return None

    @classmethod
    def is_array(cls, name):
        try:
            return True if re.match(cls.RE_ARRAY, name) else False
        except:
            return False
        
    @classmethod
    def get_array_length(cls, name):
        if not cls.is_array(name):
            raise Exception("Cannot find array length of non-array type.")

        match = re.match(cls.RE_ARRAY, name)
        return match.groups()[1]
    
    @classmethod
    def get_array_element_type(cls, name):
        if not cls.is_array(name):
            raise Exception("Cannot find array type of non-array type.")

        match = re.match(cls.RE_ARRAY, name)
        return match.groups()[0]

"""
    Extracts traits from a value name
"""
class value_traits:
    RE_VERSION  = r"\$X_MAKE_VERSION\(\s*(\d+)\s*\,\s*(\d+)\s*\)"
    RE_BIT      = r".*BIT\(\s*(.*)\s*\)"
    RE_HEX      = r"0x\w+"
    RE_MACRO    = r"(\$\w+)\(.*\)"
    RE_ARRAY    = r"(.*)\[(.*)\]"

    @classmethod
    def is_ver(cls, name):
        try:
            return True if re.match(cls.RE_VERSION, name) else False
        except:
            return False

    @classmethod
    def get_major_ver(cls, name):
        try:
            return int(re.sub(cls.RE_VERSION, r"\1", name))
        except:
            return 0

    @classmethod
    def get_minor_ver(cls, name):
        try:
            return int(re.sub(cls.RE_VERSION, r"\2", name))
        except:
            return 0

    @classmethod
    def is_bit(cls, name):
        try:
            return True if re.match(cls.RE_BIT, name) else False
        except:
            return False

    @classmethod
    def get_bit_count(cls, name):
        try:
            return int(re.sub(cls.RE_BIT, r"\1", name))
        except:
            return 0

    @classmethod
    def is_hex(cls, name):
        try:
            return True if re.match(cls.RE_HEX, name) else False
        except:
            return False

    @classmethod
    def is_macro(cls, name, meta):
        try:
            name = cls.get_macro_name(name)
            name = cls.get_array_length(name)
            return True if name in meta['macro'] else False
        except:
            return False

    @classmethod
    def get_macro_name(cls, name):
        try:
            return re.sub(cls.RE_MACRO, r"\1", name)    # 'NAME()' -> 'NAME'
        except:
            return name

    @classmethod
    def is_array(cls, name):
        try:
            return True if re.match(cls.RE_ARRAY, name) else False
        except:
            return False

    @classmethod
    def get_array_name(cls, name):
        try:
            return re.sub(cls.RE_ARRAY, r"\1", name)    # 'name[len]' -> 'name'
        except:
            return name

    @classmethod
    def get_array_length(cls, name):
        try:
            return re.sub(cls.RE_ARRAY, r"\2", name)    # 'name[len]' -> 'len'
        except:
            return name

    @classmethod
    def find_enum_name(cls, name, meta):
        try:
            name = cls.get_array_name(name)
            # if the value is an etor, return the name of the enum
            for e in meta['enum']:
                if name in meta['enum'][e]['etors']:
                    return e
            return None
        except:
            return None

"""
    Extracts traits from a parameter object
"""
class param_traits:
    RE_MBZ      = r".*\[mbz\].*"
    RE_IN       = r"^\[in\].*"
    RE_OUT      = r"^\[out\].*"
    RE_INOUT    = r"^\[in,out\].*"
    RE_OPTIONAL = r".*\[optional\].*"
    RE_NOCHECK  = r".*\[nocheck\].*"
    RE_RANGE    = r".*\[range\((.+),\s*(.+)\)\][\S\s]*"
    RE_RELEASE  = r".*\[release\].*"
    RE_TYPENAME = r".*\[typename\((.+),\s(.+)\)\].*"
    RE_TAGGED   = r".*\[tagged_by\((.+)\)].*"
    RE_BOUNDS   = r".*\[bounds\((.+),\s*(.+)\)].*"

    @classmethod
    def is_mbz(cls, item):
        try:
            return True if re.match(cls.RE_MBZ, item['desc']) else False
        except:
            return False

    @classmethod
    def is_input(cls, item):
        try:
            return True if re.match(cls.RE_IN, item['desc']) else False
        except:
            return False

    @classmethod
    def is_output(cls, item):
        try:
            return True if re.match(cls.RE_OUT, item['desc']) else False
        except:
            return False

    @classmethod
    def is_inoutput(cls, item):
        try:
            return True if re.match(cls.RE_INOUT, item['desc']) else False
        except:
            return False

    @classmethod
    def is_optional(cls, item):
        try:
            return True if re.match(cls.RE_OPTIONAL, item['desc']) else False
        except:
            return False

    @classmethod
    def is_nocheck(cls, item):
        try:
            return True if re.match(cls.RE_NOCHECK, item['desc']) else False
        except:
            return False

    @classmethod
    def is_range(cls, item):
        try:
            return True if re.match(cls.RE_RANGE, item['desc']) else False
        except:
            return False

    @classmethod
    def is_tagged(cls, item):
        try:
            return True if re.match(cls.RE_TAGGED, item['desc']) else False
        except:
            return False

    @classmethod
    def is_bounds(cls, item):
        try:
            return True if re.match(cls.RE_BOUNDS, item['desc']) else False
        except:
            return False
    
    @classmethod
    def tagged_member(cls, item):
        try:
            return re.sub(cls.RE_TAGGED, r"\1", item['desc'])
        except:
            return None

    @classmethod
    def range_start(cls, item):
        try:
            return re.sub(cls.RE_RANGE, r"\1", item['desc'])
        except:
            return None

    @classmethod
    def range_end(cls, item):
        try:
            return re.sub(cls.RE_RANGE, r"\2", item['desc'])
        except:
            return None

    @classmethod
    def is_release(cls, item):
        try:
            return True if re.match(cls.RE_RELEASE, item['desc']) else False
        except:
            return False

    @classmethod
    def is_typename(cls, item):
        try:
            return True if re.match(cls.RE_TYPENAME, item['desc']) else False
        except:
            return False

    @classmethod
    def typename(cls, item):
        match = re.match(cls.RE_TYPENAME, item['desc'])
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def typename_size(cls, item):
        match = re.match(cls.RE_TYPENAME, item['desc'])
        if match:
            return match.group(2)
        else:
            return None

    @classmethod
    def bounds_offset(cls, item):
        match = re.match(cls.RE_BOUNDS, item['desc'])
        if match:
            return match.group(1)
        else:
            return None

    @classmethod
    def bounds_size(cls, item):
        match = re.match(cls.RE_BOUNDS, item['desc'])
        if match:
            return match.group(2)
        else:
            return None

"""
    Extracts traits from a function object
"""
class function_traits:

    @staticmethod
    def is_static(item):
        try:
            return True if re.match(r"static", item['decl']) else False
        except:
            return False

    @staticmethod
    def is_global(item, tags):
        try:
            return True if item['class'] in tags else False
        except:
            return False


"""
Public:
    substitutes each tag['key'] with tag['value']
    if comment, then insert doxygen '::' notation at beginning (for autogen links)
"""
def subt(namespace, tags, string, comment=False, remove_namespace=False):
    for key, value in tags.items():
        if remove_namespace:
            repl = ""                                                           # remove namespace; e.g. "$x" -> ""
            string = re.sub(r"%s_?"%re.escape(key), repl, string)
            string = re.sub(r"%s_?"%re.escape(key.upper()), repl.upper(), string)
        else:
            string = re.sub(r"-%s"%re.escape(key), "-"+value, string)           # hack for compile options
            repl = "::"+value if comment and "$OneApi" != key else value        # replace tag; e.g., "$x" -> "ur"
            string = re.sub(re.escape(key), repl, string)
            string = re.sub(re.escape(key.upper()), repl.upper(), string)
    return string

"""
Public:
    appends whitespace (in multiples of 4) to the end of the string,
    until len(string) > count
"""
def append_ws(string, count):
    while len(string) > count:
        count = count + 4
    string = '{str: <{width}}'.format(str=string, width=count)
    return string

"""
Public:
    split the line of text into a list of strings,
    where each length of each entry is less-than count
"""
def split_line(line, ch_count):
    if not line:
        return [""]

    RE_NEWLINE = r"(.*)\n(.*)"

    words           = line.split(" ")
    lines           = []
    word_list       = []

    for word in words:
        if re.match(RE_NEWLINE, word):
            prologue = re.sub(RE_NEWLINE,r"\1",word)
            epilogue = re.sub(RE_NEWLINE,r"\2",word)
            word_list.append(prologue)
            lines.append(" ".join(word_list))
            word_list = []
            if len(epilogue):
                word_list.append(epilogue)

        elif sum(map(len, word_list)) + len(word_list) + len(word) <= ch_count:
            word_list.append(word)

        else:
            lines.append(" ".join(word_list))
            word_list = [word]

    if len(word_list):
        lines.append(" ".join(word_list))
    return lines

"""
Private:
    converts string from camelCase to snake_case
"""
def _camel_to_snake(name):
    return util.to_snake_case(name).lower()

"""
Public:
    removes items from the list with the key and whose value do not match filter
"""
def filter_items(lst, key, filter):
    flst = []
    for item in lst:
        if key in item:
            if filter == item[key]:
                flst.append(item)
    return flst

"""
Public:
    returns a list of items with key from a list of dict
"""
def extract_items(lst, key):
    klst = []
    for item in lst:
        if key in item:
            klst.append(item[key])
    return klst

"""
Public:
    returns a list of all objects of type in all specs
"""
def extract_objs(specs, value):
    objs = []
    for s in specs:
        for obj in s['objects']:
            if re.match(value, obj['type']):
                objs.append(obj)
    return objs

"""
Public:
    returns a list of all adapter functions
"""
def get_adapter_functions(specs):
    objs = []
    for s in specs:
        for obj in s['objects']:
            if obj_traits.is_function(obj) and not obj_traits.is_loader_only(obj):
                objs.append(obj)
    return objs

"""
Public:
    returns a list of all adapter handles
"""
def get_adapter_handles(specs):
    objs = []
    for s in specs:
        for obj in s['objects']:
            if obj_traits.is_handle(obj) and not obj_traits.is_loader_only(obj):
                objs.append(obj)

    return objs

"""
Private:
    removes 'const' from c++ type
"""
def _remove_const(name):
    name = name.split(" ")[-1]
    return name

"""
Private:
    removes '*' from c++ type
"""
def _remove_ptr(name, last=True):
    if last:
        name = re.sub(r"(.*)\*", r"\1", name) # removes only last '*'
    else:
        name = re.sub(r"\*", "", name) # removes all '*'
    return name

"""
Private:
    removes 'const' and '*' from c++ type
"""
def _remove_const_ptr(name):
    name = _remove_ptr(_remove_const(name))
    return name

"""
Public:
    returns c/c++ name of macro
"""
def make_macro_name(namespace, tags, obj, params=True):
    if params:
        return subt(namespace, tags, obj['name'])
    else:
        name = re.sub(r"(.*)\(.*", r"\1", obj['name'])  # remove '()' part
        return subt(namespace, tags, name)

"""
Public:
    returns c/c++ name of enums, structs, unions, typedefs...
"""
def make_type_name(namespace, tags, obj):
    name = subt(namespace, tags, obj['name'])
    return name

"""
Public:
    returns c/c++ name of enums...
"""
def make_enum_name(namespace, tags, obj):
    name = make_type_name(namespace, tags, obj)
    if type_traits.is_flags(obj['name']):
        name = re.sub(r"flags", r"flag", name)
    return name

"""
    returns c/c++ definition of flags_t bit mask.
"""
def make_flags_bitmask(namespace, tags, obj, meta):
    etor_meta = meta[obj['type']][obj['name']]
    if 'bit_mask' not in etor_meta.keys():
        return ""
    macro_def = "#define"
    macro_name = make_type_name(namespace, tags, obj).upper()[:-2] + "_MASK"
    mask = etor_meta['bit_mask']
    return "%s %s %s"%(macro_def, macro_name, mask)

"""
Public:
    returns c/c++ name of etor
"""
def make_etor_name(namespace, tags, enum, etor, meta=None):
    return subt(namespace, tags, etor)

"""
Private:
    returns the associated type of an etor from a typed enum
"""
def etor_get_associated_type(namespace, tags, item):
    match = re.match(r'^\[(.+)\]\s', item['desc'])
    if match:
        associated_type = match.group(1)
        return subt(namespace, tags, associated_type)
    else:
        return None

"""
Private:
    returns c/c++ name of value
"""
def _get_value_name(namespace, tags, value):
    value = subt(namespace, tags, value)
    return value

"""
Public:
    returns a list of strings for declaring each enumerator in an enumeration
    c++ format: "ETOR_NAME = VALUE, ///< DESCRIPTION"
"""
def make_etor_lines(namespace, tags, obj, meta=None):
    lines = []
    for item in obj['etors']:
        name = make_etor_name(namespace, tags, obj['name'], item['name'], meta)

        if 'value' in item:
            delim = ","
            value = _get_value_name(namespace, tags, item['value'])
            prologue = "%s = %s%s"%(name, value, delim)
        else:
            prologue = "%s,"%(name)

        for line in split_line(subt(namespace, tags, item['desc'], True), 70):
            lines.append("%s%s %s"%(append_ws(prologue, 48), "///<", line))
            prologue = ""

    lines += [
        "/// @cond",
        "%sFORCE_UINT32 = 0x7fffffff"%make_enum_name(namespace, tags, obj)[:-1].upper(),
        "/// @endcond",
    ]

    return lines

"""
Private:
    returns c/c++ name of any type
"""
def _get_type_name(namespace, tags, obj, item):
    type = item['type']
    if type_traits.is_array(type):
        type = type_traits.get_array_element_type(type)
    name = subt(namespace, tags, type,)
    return name

"""
Public:
    returns c/c++ name of member of struct/class
"""
def make_member_name(namespace, tags, item, prefix="", remove_array=False):
    name = subt(namespace, tags, prefix+item['name'])

    if remove_array:
        name = value_traits.get_array_name(name)

    return name

"""
Public:
    returns a list of strings for each member of a structure or class
    c++ format: "TYPE NAME = INIT, ///< DESCRIPTION"
"""
def make_member_lines(namespace, tags, obj, prefix="", meta=None):
    lines = []
    if 'members' not in obj:
        return lines

    for i, item in enumerate(obj['members']):
        name = make_member_name(namespace, tags, item, prefix)
        tname = _get_type_name(namespace, tags, obj, item)

        array_suffix = f"[{type_traits.get_array_length(item['type'])}]" if type_traits.is_array(item['type']) else ""
        prologue = "%s %s %s;"%(tname, name, array_suffix)

        for line in split_line(subt(namespace, tags, item['desc'], True), 70):
            lines.append("%s%s %s"%(append_ws(prologue, 48), "///<", line))
            prologue = ""
    return lines

"""
Private:
    returns c/c++ name of parameter
"""
def _get_param_name(namespace, tags, item):
    name = subt(namespace, tags, item['name'])
    return name

"""
Public:
    returns a list of c++ strings for each parameter of a function
    format: "TYPE NAME = INIT, ///< DESCRIPTION"
"""
def make_param_lines(namespace, tags, obj, decl=False, meta=None, format=["type", "name", "delim", "desc"], delim=",", replacements={}):
    lines = []

    params = obj['params']
    fptr_types = [] # This is done so that we dont have to try/catch for defined
    if meta is not None and "fptr_typedef" in meta:
        fptr_types = list(meta['fptr_typedef'].keys())

    for i, item in enumerate(params):
        name = _get_param_name(namespace, tags, item)
        if replacements.get(name):
            name = replacements[name]

        tname = _get_type_name(namespace, tags, obj, item)

        words = []
        if "type*" in format:
            words.append(tname+"*")
            name = "p"+name
        elif "type" in format:
            words.append(tname)
        if "name" in format:
            words.append(name)

        prologue = " ".join(words)
        if "delim" in format:
            if i < len(params)-1:
                prologue += delim

        if "desc" in format:
            desc = item['desc']
            for line in split_line(subt(namespace, tags, desc, True), 70):
                lines.append("%s///< %s"%(append_ws(prologue, 48), line))
                prologue = ""
        else:
            lines.append(prologue)

    if "type" in format and len(lines) == 0:
        lines = ["void"]
    return lines

"""
Public:
    returns a list of strings for the description
    format: "@brief DESCRIPTION"
"""
def make_desc_lines(namespace, tags, obj):
    lines = []
    prologue = "@brief"
    for line in split_line(subt(namespace, tags, obj['desc'], True), 70):
        lines.append("%s %s"%(prologue, line))
        prologue = "      "
    return lines

"""
Public:
    returns a list of strings for the detailed description
    format: "@details DESCRIPTION"
"""
def make_details_lines(namespace, tags, obj):
    lines = []
    if 'details' in obj:
        lines.append("")
        lines.append("@details")

        # If obj['details'] is a list of bullet points, add the bullet point formatting to the lines.
        if isinstance(obj['details'], list):
            for item in obj['details']:
                if isinstance(item, dict):
                    for key, values in item.items():
                        prologue = "    -"
                        for line in split_line(subt(namespace, tags, key, True), 70):
                            lines.append("%s %s"%(prologue, line))
                            prologue = "     "
                        for val in values:
                            prologue = "        +"
                            for line in split_line(subt(namespace, tags, val, True), 66):
                                lines.append("%s %s"%(prologue, line))
                                prologue = "         "
                else:
                    prologue = "    -"
                    for line in split_line(subt(namespace, tags, item, True), 70):
                        lines.append("%s %s"%(prologue, line))
                        prologue = "     "
        # If obj['details'] is a string, then assume that it's already correctly formatted using markdown.
        else:
            for line in obj['details'].splitlines(False):
                lines.append(subt(namespace, tags, line, True))

    if 'analogue' in obj:
        lines.append("")
        lines.append("@remarks")
        lines.append("  _Analogues_")
        for line in obj['analogue']:
            lines.append("    - %s"%line)
    return lines

"""
Public:
    returns a list of strings for possible return values
"""
def make_returns_lines(namespace, tags, obj, meta=None):
    lines = []
    lines.append("@returns")
    for item in obj.get('returns', []):
        for key, values in item.items():
            lines.append("    - %s"%subt(namespace, tags, key, True))
            for val in values:
                lines.append("        + %s"%subt(namespace, tags, val, True))

    return lines

"""
Public:
    returns the name of a function
"""
def make_func_name(namespace, tags, obj):
    cname = obj_traits.class_name(obj)
    if cname == None: # If can't find the class name append nothing
        cname = ''
    return subt(namespace, tags, "%s%s"%(cname, obj['name']))

"""
Public:
    returns the etor of a function
"""
def make_func_etor(namespace, tags, obj):
    etags = tags.copy()
    etags['$x'] += 'Function'
    return util.to_snake_case(make_func_name(namespace, etags, obj)).upper()

"""
Public:
    returns the name of a function pointer
"""
def make_pfn_name(namespace, tags, obj):

    return subt(namespace, tags, "pfn%s"%obj['name'])

"""
Public:
    returns the name of a function pointer
"""
def make_pfncb_name(namespace, tags, obj):

    return subt(namespace, tags, "pfn%sCb"%obj['name'])

"""
Public:
    returns the name of a function pointer
"""
def make_pfn_type(namespace, tags, obj, epilogue=""):
    newtags = dict()
    for key, value in tags.items():
        if re.match(namespace, value):
            newtags[key] = "pfn"
    return "%s_%s%s_t"%(namespace, make_func_name(namespace, newtags, obj), epilogue)

"""
Public:
    returns the name of a function pointer
"""
def make_pfncb_type(namespace, tags, obj):

    return make_pfn_type(namespace, tags, obj, epilogue="Cb")

"""
Public:
    returns the name of a function pointer
"""
def make_pfncb_param_type(namespace, tags, obj):

    return "%s_params_t"%_camel_to_snake(make_func_name(namespace, tags, obj))

"""
Public:
    returns an appropriate bounds helper function call for an entry point
    parameter with the [bounds] tag
"""
def get_bounds_check(param, bounds_error):
    # Images need their own helper, since function signature wise they would be
    # identical to buffer rect
    bounds_function = 'boundsImage' if 'image' in param['name'].lower() else 'bounds'
    bounds_check = "auto {0} = {1}({2}, {3}, {4})".format(
        bounds_error,
        bounds_function,
        param["name"],
        param_traits.bounds_offset(param),
        param_traits.bounds_size(param),
    )
    bounds_check += '; {0} != UR_RESULT_SUCCESS'.format(bounds_error)

    # USM bounds checks need the queue handle parameter to be able to use the
    # GetMemAllocInfo entry point
    if type_traits.is_pointer(param['type']):
        # If no `hQueue` parameter exists that should have been caught at spec
        # generation.
        return re.sub(r'bounds\(', 'bounds(hQueue, ', bounds_check)

    return bounds_check

"""
Public:
    returns a dict of auto-generated c++ parameter validation checks for the
    given function (specified by `obj`)
"""
def make_param_checks(namespace, tags, obj, cpp=False, meta=None):
    checks = {}
    for item in obj.get('returns', []):
        for key, values in item.items():
            key = subt(namespace, tags, key, False, cpp)
            for val in values:
                code = re.match(r"^\`(.*)\`$", val)
                if code:
                    if key not in checks:
                        checks[key] = []
                    checks[key].append(subt(namespace, tags, code.group(1), False, cpp))

    for p in obj.get('params', []):
        if param_traits.is_bounds(p):
            if 'boundsError' not in checks:
                checks['boundsError'] = []
            checks['boundsError'].append(get_bounds_check(p, 'boundsError'))

    return checks

"""
Public:
    returns a list of all function objs for the specified class.
"""
def get_class_function_objs(specs, cname, version = None):
    objects = []
    for s in specs:
        for obj in s['objects']:
            is_function = obj_traits.is_function(obj)
            match_cls = cname == obj_traits.class_name(obj)
            if is_function and match_cls:
                if version is None:
                    objects.append(obj)
                elif float(obj.get('version',"1.0")) <= version:
                    objects.append(obj)
    return sorted(objects, key=lambda obj: (float(obj.get('version',"1.0"))*10000) + int(obj.get('ordinal',"100")))

"""
Public:
    returns a list of all non-experimental function objs and a list of experimental function objs for the specified class
"""
def get_class_function_objs_exp(specs, cname):
    objects = []
    exp_objects = []
    for s in specs:
        for obj in s['objects']:
            is_function = obj_traits.is_function(obj)
            match_cls = cname == obj_traits.class_name(obj)
            if is_function and match_cls:
                if obj_traits.is_experimental(obj):
                    exp_objects.append(obj)
                else:
                    objects.append(obj)
    objects = sorted(objects, key=lambda obj: (float(obj.get('version',"1.0"))*10000) + int(obj.get('ordinal',"100")))
    exp_objects = sorted(exp_objects, key=lambda obj: (float(obj.get('version',"1.0"))*10000) + int(obj.get('ordinal',"100")))
    return objects, exp_objects

"""
Public:
    returns string name of table for function object
"""
def get_table_name(namespace, tags, obj):
    cname = obj_traits.class_name(obj)
    if obj_traits.is_experimental(obj):
        cname=cname+"Exp"
    name = subt(namespace, tags, cname, remove_namespace=True) # i.e., "$x" -> ""
    name = name if len(name) > 0 else "Global"
    return name

"""
Public:
    returns a list of dict of each pfntables needed
"""
def get_pfntables(specs, meta, namespace, tags):
    tables = []
    for cname in sorted(meta['class'], key=lambda x: meta['class'][x]['ordinal']):
        objs, exp_objs = get_class_function_objs_exp(specs, cname)
        objs = list(filter(lambda obj: not obj_traits.is_loader_only(obj), objs))
        exp_objs = list(filter(lambda obj: not obj_traits.is_loader_only(obj), exp_objs))

        if len(objs) > 0:
            name = get_table_name(namespace, tags, objs[0])
            table = "%s_%s_dditable_t"%(namespace, _camel_to_snake(name))

            params = []
            params.append({
                'type': "$x_api_version_t",
                'name': "version",
                'desc': "[in] API version requested"
                })
            params.append({
                'type': "%s*"%table,
                'name': "pDdiTable",
                'desc': "[in,out] pointer to table of DDI function pointers"
                })
            export = {
                'name': "%sGet%sProcAddrTable"%(namespace, name),
                'params': params
                }

            pfn = "%s_pfnGet%sProcAddrTable_t"%(namespace, name)

            tables.append({
                'name': name,
                'type': table,
                'export': export,
                'pfn': pfn,
                'functions': objs,
                'experimental': False
            })
        if len(exp_objs) > 0:
            name = get_table_name(namespace, tags, exp_objs[0])
            table = "%s_%s_dditable_t"%(namespace, _camel_to_snake(name))

            params = []
            params.append({
                'type': "$x_api_version_t",
                'name': "version",
                'desc': "[in] API version requested"
                })
            params.append({
                'type': "%s*"%table,
                'name': "pDdiTable",
                'desc': "[in,out] pointer to table of DDI function pointers"
                })
            export = {
                'name': "%sGet%sProcAddrTable"%(namespace, name),
                'params': params
                }

            pfn = "%s_pfnGet%sProcAddrTable_t"%(namespace, name)

            tables.append({
                'name': name,
                'type': table,
                'export': export,
                'pfn': pfn,
                'functions': exp_objs,
                'experimental': True
            })


    return tables


"""
Private:
    returns the list of parameters, filtering based on desc tags
"""
def _filter_param_list(params, filters1=["[in]", "[in,out]", "[out]"], filters2=[""]):
    lst = []
    for p in params:
        for f1 in filters1:
            if f1 in p['desc']:
                for f2 in filters2:
                    if f2 in p['desc']:
                        lst.append(p)
                        break
                break
    return lst

"""
Public:
    returns a list of dict of each pfntables needed
"""
def get_pfncbtables(specs, meta, namespace, tags):
    tables = []
    for cname in sorted(meta['class'], key=lambda x: meta['class'][x]['ordinal']):
        objs = get_class_function_objs(specs, cname, 1.0)
        if len(objs) > 0:
            name = get_table_name(namespace, tags, {'class': cname})
            print(name)
            table = "%s_%s_callbacks_t"%(namespace, _camel_to_snake(name))
            tables.append({
                'name': name,
                'type': table,
                'functions': objs
            })
    return tables

"""
Public:
    returns a list of dict for converting loader input parameters
"""
def get_loader_prologue(namespace, tags, obj, meta):
    prologue = []

    params = _filter_param_list(obj['params'], ["[in]"])
    for item in params:
        if param_traits.is_mbz(item):
            continue
        if type_traits.is_class_handle(item['type'], meta):
            name = subt(namespace, tags, item['name'])
            tname = _remove_const_ptr(subt(namespace, tags, item['type']))

            # e.g., "xe_device_handle_t" -> "xe_device_object_t"
            obj_name = re.sub(r"(\w+)_handle_t", r"\1_object_t", tname)
            fty_name = re.sub(r"(\w+)_handle_t", r"\1_factory", tname)

            if type_traits.is_pointer(item['type']):
                range_start = param_traits.range_start(item)
                range_end   = param_traits.range_end(item)
                prologue.append({
                    'name': name,
                    'obj': obj_name,
                    'range': (range_start, range_end),
                    'type': tname,
                    'factory': fty_name,
                    'pointer' : "*"
                })
            else:
                prologue.append({
                    'name': name,
                    'obj': obj_name,
                    'optional': param_traits.is_optional(item),
                    'pointer' : ""
                })

    return prologue

"""
Public:
    returns an enum object with the given name
"""
def get_enum_by_name(specs, namespace, tags, name, only_typed):
    for s in specs:
        for obj in s['objects']:
            if obj_traits.is_enum(obj) and make_enum_name(namespace, tags, obj) == name:
                typed = obj.get('typed_etors', False) is True
                if only_typed:
                    if typed:
                        return obj
                    else:
                        return None
                else:
                    return obj
    return None

"""
Public:
    returns a list of dict for converting loader output parameters
"""
def get_loader_epilogue(specs, namespace, tags, obj, meta):
    epilogue = []

    for i, item in enumerate(obj['params']):
        if param_traits.is_mbz(item):
            continue

        name = subt(namespace, tags, item['name'])
        tname = _remove_const_ptr(subt(namespace, tags, item['type']))

        obj_name = re.sub(r"(\w+)_handle_t", r"\1_object_t", tname)
        fty_name = re.sub(r"(\w+)_handle_t", r"\1_factory", tname)

        if param_traits.is_release(item) or param_traits.is_output(item) or param_traits.is_inoutput(item):
            if type_traits.is_class_handle(item['type'], meta):
                if param_traits.is_range(item):
                    range_start = param_traits.range_start(item)
                    range_end   = param_traits.range_end(item)
                    epilogue.append({
                        'name': name,
                        'type': tname,
                        'obj': obj_name,
                        'factory': fty_name,
                        'release': param_traits.is_release(item),
                        'range': (range_start, range_end)
                    })
                else:
                    epilogue.append({
                        'name': name,
                        'type': tname,
                        'obj': obj_name,
                        'factory': fty_name,
                        'release': param_traits.is_release(item),
                        'optional': param_traits.is_optional(item)
                    })
            elif param_traits.is_typename(item):
                typename = param_traits.typename(item)
                underlying_type = None
                for inner in obj['params']:
                    iname = _get_param_name(namespace, tags, inner)
                    if iname == typename:
                        underlying_type = _get_type_name(namespace, tags, obj, inner)
                if underlying_type is None:
                    continue

                prop_size = param_traits.typename_size(item)
                enum = get_enum_by_name(specs, namespace, tags, underlying_type, True)
                handle_etors = []
                for etor in enum['etors']:
                    associated_type = etor_get_associated_type(namespace, tags, etor)
                    if 'handle' in associated_type:
                        is_array = False
                        if value_traits.is_array(associated_type):
                            associated_type = value_traits.get_array_name(associated_type)
                            is_array = True

                        etor_name = make_etor_name(namespace, tags, enum['name'], etor['name'])
                        obj_name = re.sub(r"(\w+)_handle_t", r"\1_object_t", associated_type)
                        fty_name = re.sub(r"(\w+)_handle_t", r"\1_factory", associated_type)
                        handle_etors.append({'name': etor_name,
                                             'type': associated_type,
                                             'obj': obj_name,
                                             'factory': fty_name,
                                             'is_array': is_array})

                if handle_etors:
                    epilogue.append({
                                     'name': name,
                                     'obj': obj_name,
                                     'release': False,
                                     'typename': typename,
                                     'size': prop_size,
                                     'etors': handle_etors})

    return epilogue

"""
Public:
    returns a dictionary with lists of create, retain and release functions
"""
def get_create_retain_release_functions(specs, namespace, tags):
    funcs = []
    for s in specs:
        for obj in s['objects']:
            if re.match(r"function", obj['type']):
                funcs.append(make_func_name(namespace, tags, obj))

    create_suffixes = r"(Create[A-Za-z]*){1}"
    retain_suffixes = r"(Retain){1}"
    release_suffixes = r"(Release){1}"

    create_exp = namespace + r"([A-Za-z]+)" + create_suffixes
    retain_exp = namespace + r"([A-Za-z]+)" + retain_suffixes
    release_exp = namespace + r"([A-Za-z]+)" + release_suffixes

    create_funcs, retain_funcs, release_funcs = (
        list(filter(lambda f: re.match(create_exp, f), funcs)),
        list(filter(lambda f: re.match(retain_exp, f), funcs)),
        list(filter(lambda f: re.match(release_exp, f), funcs)),
    )

    create_funcs, retain_funcs = (
        list(filter(lambda f: re.sub(create_suffixes, "Release", f) in release_funcs, create_funcs)),
        list(filter(lambda f: re.sub(retain_suffixes, "Release", f) in release_funcs, retain_funcs)),
    )

    return {"create": create_funcs, "retain": retain_funcs, "release": release_funcs}


def get_event_wait_list_functions(specs, namespace, tags):
    funcs = []
    for s in specs:
        for obj in s['objects']:
            if re.match(r"function", obj['type']):
                if any(x['name'] == 'phEventWaitList' for x in obj['params']) and any(
                        x['name'] == 'numEventsInWaitList' for x in obj['params']):
                    funcs.append(make_func_name(namespace, tags, obj))
    return funcs
