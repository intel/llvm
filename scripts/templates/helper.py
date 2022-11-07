"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

"""
import re

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
    RE_DESC     = r"(.*)desc_t.*"
    RE_PROPS    = r"(.*)properties_t.*"
    RE_FLAGS    = r"(.*)flags_t"

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
    RE_IN       = r"^\[in\].*"
    RE_OUT      = r"^\[out\].*"
    RE_INOUT    = r"^\[in,out\].*"
    RE_OPTIONAL = r".*\[optional\].*"
    RE_RANGE    = r".*\[range\((.+),\s*(.+)\)\][\S\s]*"
    RE_RELEASE  = r".*\[release\].*"

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
    def is_range(cls, item):
        try:
            return True if re.match(cls.RE_RANGE, item['desc']) else False
        except:
            return False

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
    substitues each tag['key'] with tag['value']
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
            repl = "::"+value if comment and "$OneApi" != key else value        # replace tag; e.g., "$x" -> "xe"
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
    str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', str).lower()
    return str

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
        name = re.sub(r"(.*)\*$", r"\1", name) # removes only last '*'
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
Public:
    returns c/c++ name of etor
"""
def make_etor_name(namespace, tags, enum, etor, py=False, meta=None):
    if py:
        # e.g., "ENUM_NAME_ETOR_NAME" -> "ETOR_NAME"
        if type_traits.is_flags(enum):
            prefix = re.sub(r"(\w+)_flags_t", r"\1_flag", subt(namespace, tags, enum)).upper()
        else:
            prefix = re.sub(r"(\w+)_t", r"\1", subt(namespace, tags, enum)).upper()
        name = re.sub(r"%s_(\w+)"%prefix, r"\1", subt(namespace, tags, etor))
        name = re.sub(r"^(\d+\w*)", r"_\1", name)
    else:
        name = subt(namespace, tags, etor)
    return name

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
    python format: "ETOR_NAME = VALUE, ## DESCRIPTION"
"""
def make_etor_lines(namespace, tags, obj, py=False, meta=None):
    lines = []
    for item in obj['etors']:
        name = make_etor_name(namespace, tags, obj['name'], item['name'], py, meta)

        if 'value' in item:
            delim = "," if not py else ""
            value = _get_value_name(namespace, tags, item['value'])
            prologue = "%s = %s%s"%(name, value, delim)
        elif py:
            prologue = "%s = auto()"%(name)
        else:
            prologue = "%s,"%(name)

        comment_style = "##" if py else "///<"
        for line in split_line(subt(namespace, tags, item['desc'], True), 70):
            lines.append("%s%s %s"%(append_ws(prologue, 48), comment_style, line))
            prologue = ""

    if not py:
        lines.append("%sFORCE_UINT32 = 0x7fffffff"%make_enum_name(namespace, tags, obj)[:-1].upper())

    return lines

"""
Private:
    returns c/c++ name of any type
"""
def _get_type_name(namespace, tags, obj, item):
    name = subt(namespace, tags, item['type'],)
    return name

"""
Private:
    returns python c_type name of any type
"""
def get_ctype_name(namespace, tags, item):
    name = subt(namespace, tags, item['type'])
    name = _remove_const(name)
    name = re.sub(r"void\*", "c_void_p", name)
    name = re.sub(r"char\*", "c_char_p", name)
    name = re.sub(r"uint8_t", "c_ubyte", name)
    name = re.sub(r"uint16_t", "c_ushort", name)
    name = re.sub(r"uint32_t", "c_ulong", name)
    name = re.sub(r"uint64_t", "c_ulonglong", name)
    name = re.sub(r"size_t", "c_size_t", name)
    name = re.sub(r"float", "c_float", name)
    name = re.sub(r"double", "c_double", name)
    name = re.sub(r"\bchar", "c_char", name)
    name = re.sub(r"\bint", "c_int", name)

    if type_traits.is_pointer(name):
        name = _remove_ptr(name)
        name = "POINTER(%s)"%name

    elif 'name' in item and value_traits.is_array(item['name']):
        length = subt(namespace, tags, value_traits.get_array_length(item['name']))
        name = "%s * %s"%(name, length)

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
    python format: "("NAME", TYPE)" ## DESCRIPTION"
"""
def make_member_lines(namespace, tags, obj, prefix="", py=False, meta=None):
    lines = []
    if 'members' not in obj:
        return lines

    for i, item in enumerate(obj['members']):
        name = make_member_name(namespace, tags, item, prefix, remove_array=py)

        if py:
            tname = get_ctype_name(namespace, tags, item)
        else:
            tname = _get_type_name(namespace, tags, obj, item)

        if py:
            delim = "," if i < (len(obj['members'])-1) else ""
            prologue = "(\"%s\", %s)%s"%(name, tname, delim)
        else:
            prologue = "%s %s;"%(tname, name)

        comment_style = "##" if py else "///<"
        ws_count = 64 if py else 48
        for line in split_line(subt(namespace, tags, item['desc'], True), 70):
            lines.append("%s%s %s"%(append_ws(prologue, ws_count), comment_style, line))
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
def make_param_lines(namespace, tags, obj, py=False, decl=False, meta=None, format=["type", "name", "delim", "desc"], delim=","):
    lines = []

    params = obj['params']

    for i, item in enumerate(params):
        name = _get_param_name(namespace, tags, item)
        if py:
            tname = get_ctype_name(namespace, tags, item)
        else:
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

    if "type" in format and len(lines) == 0 and not py:
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
    return subt(namespace, tags, "%s%s"%(cname, obj['name']))

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
                'functions': objs
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
                'functions': exp_objs
            })
        
        
    return tables

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


