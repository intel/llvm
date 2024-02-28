# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Please follow GDB coding standards and use 'black' for formatting:
# https://sourceware.org/gdb/wiki/Internals%20GDB-Python-Coding-Standards


import re
import gdb
import gdb.xmethod
import gdb.printing

### XMethod implementations ###


class Accessor:
    """Generalized base class for buffer index calculation"""

    def memory_range(self, dim):
        pass

    def offset(self, dim):
        pass

    def data(self):
        pass

    def __init__(self, obj, result_type, depth):
        self.obj = obj
        self.result_type = result_type
        self.depth = depth

    def index(self, arg):
        if arg.type.unqualified().strip_typedefs().code == gdb.TYPE_CODE_INT:
            return int(arg)
        # unwrap if inside item
        try:
            arg = arg["MImpl"]["MIndex"]
        except:
            pass
        result = 0
        for dim in range(self.depth):
            result = (
                result * self.memory_range(dim)
                + self.offset(dim)
                + arg["common_array"][dim]
            )
        return result

    def value(self, arg):
        return self.data()[self.index(arg)]


class HostAccessor(Accessor):
    """For Host device memory layout"""

    def memory_range(self, dim):
        eval_string = (
            "((" + str(self.obj.type) + ")" + str(self.obj) + ")->getMemoryRange()"
        )
        return gdb.parse_and_eval(eval_string)["common_array"][dim]

    def offset(self, dim):
        eval_string = "((" + str(self.obj.type) + ")" + str(self.obj) + ")->getOffset()"
        return gdb.parse_and_eval(eval_string)["common_array"][dim]

    def data(self):
        eval_string = "((" + str(self.obj.type) + ")" + str(self.obj) + ")->getPtr()"
        return gdb.parse_and_eval(eval_string)


class HostAccessorLocal(HostAccessor):
    """For Host device memory layout"""

    def memory_range(self, dim):
        eval_string = "((" + str(self.obj.type) + ")" + str(self.obj) + ")->getSize()"
        return gdb.parse_and_eval(eval_string)["common_array"][dim]

    def index(self, arg):
        if arg.type.code == gdb.TYPE_CODE_INT:
            return int(arg)
        result = 0
        for dim in range(self.depth):
            result = result * self.memory_range(dim) + arg["common_array"][dim]
        return result


class DeviceAccessor(Accessor):
    """For CPU/GPU memory layout"""

    def memory_range(self, dim):
        return self.obj["impl"]["MemRange"]["common_array"][dim]

    def offset(self, dim):
        return self.obj["impl"]["Offset"]["common_array"][dim]

    def data(self):
        return self.obj["MData"]


class AccessorOpIndex(gdb.xmethod.XMethodWorker):
    """Generic implementation for N-dimensional ID"""

    def __init__(self, class_type, result_type, depth):
        self.class_type = class_type
        self.result_type = result_type
        self.depth = depth

    def get_arg_types(self):
        try:
            return gdb.lookup_type("sycl::_V1::id<%s>" % self.depth)
        except:
            pass
        return None

    def get_result_type(self, *args):
        return self.result_type

    def __call__(self, obj, arg):
        # No way to easily figure out which devices is currently being used,
        # try all accessor implementations until one of them works:
        accessors = [
            DeviceAccessor(obj, self.result_type, self.depth),
            HostAccessor(obj, self.result_type, self.depth),
            HostAccessorLocal(obj, self.result_type, self.depth),
        ]
        for accessor in accessors:
            try:
                return accessor.value(arg)
            except:
                pass

        print("Failed to call '%s.operator[](%s)'" % (obj.type, arg.type))

        return None


class AccessorOpIndex1D(AccessorOpIndex):
    """Introduces an extra overload for 1D case that takes plain size_t"""

    def get_arg_types(self):
        assert self.depth == 1
        return gdb.lookup_type("size_t")


class AccessorOpIndexItemTrue(AccessorOpIndex):
    """Introduces an extra overload for item wrapper"""

    def get_arg_types(self):
        return gdb.lookup_type("sycl::_V1::item<%s, true>" % self.depth)


class AccessorOpIndexItemFalse(AccessorOpIndex):
    """Introduces an extra overload for item wrapper"""

    def get_arg_types(self):
        return gdb.lookup_type("sycl::_V1::item<%s, false>" % self.depth)


class AccessorMatcher(gdb.xmethod.XMethodMatcher):
    """Entry point for sycl::_V1::(local_)accessor"""

    def __init__(self):
        gdb.xmethod.XMethodMatcher.__init__(self, "AccessorMatcher")

    def match(self, class_type, method_name):
        if method_name != "operator[]":
            return None

        result = re.match("^sycl::_V1::(?:local_)?accessor<.+>$", class_type.tag)
        if result is None:
            return None

        depth = int(class_type.template_argument(1))
        result_type = class_type.template_argument(0)

        methods = [AccessorOpIndex(class_type, result_type, depth)]
        try:
            method = AccessorOpIndexItemTrue(class_type, result_type, depth)
            method.get_arg_types()
            methods.append(method)
        except:
            pass
        try:
            method = AccessorOpIndexItemFalse(class_type, result_type, depth)
            method.get_arg_types()
            methods.append(method)
        except:
            pass
        if depth == 1:
            methods.append(AccessorOpIndex1D(class_type, result_type, depth))
        return methods


class PrivateMemoryOpCall(gdb.xmethod.XMethodWorker):
    """Provides operator() overload for h_item argument"""

    class ItemBase:
        """Wrapper for sycl::_V1::detail::ItemBase which reimplements index calculation"""

        def __init__(
            self,
            obj,
        ):
            result = re.match(
                "^sycl::_V1::detail::ItemBase<(.+), (.+)>$", str(obj.type)
            )
            self.dim = int(result[1])
            self.with_offset = result[2] == "true"
            self.obj = obj

        def get_linear_id(self):
            index = self.obj["MIndex"]["common_array"]
            extent = self.obj["MExtent"]["common_array"]

            if self.with_offset:
                offset = self.obj["MOffset"]["common_array"]
                if self.dim == 1:
                    return index[0] - offset[0]
                elif self.dim == 2:
                    return (index[0] - offset[0]) * extent[1] + (index[1] - offset[1])
                else:
                    return (
                        ((index[0] - offset[0]) * extent[1] * extent[2])
                        + ((index[1] - offset[1]) * extent[2])
                        + (index[2] - offset[2])
                    )
            else:
                if self.dim == 1:
                    return index[0]
                elif self.dim == 2:
                    return index[0] * extent[1] + index[1]
                else:
                    return (
                        (index[0] * extent[1] * extent[2])
                        + (index[1] * extent[2])
                        + index[2]
                    )

    def __init__(self, result_type, dim):
        self.result_type = result_type
        self.dim = dim

    def get_arg_types(self):
        return gdb.lookup_type("sycl::_V1::h_item<%s>" % self.dim)

    def get_result_type(self, *args):
        return self.result_type

    def __call__(self, obj, *args):
        if obj["Val"].type.tag.endswith(self.result_type):
            # On device private_memory is a simple wrapper over actual value
            return obj["Val"]
        else:
            # On host it wraps a unique_ptr to an array of items
            item_base = args[0]["localItem"]["MImpl"]
            item_base = self.ItemBase(item_base)
            index = item_base.get_linear_id()

            eval_string = "((" + str(obj.type) + ")" + str(obj) + ")->Val.get()"
            return gdb.parse_and_eval(eval_string)[index]


class PrivateMemoryMatcher(gdb.xmethod.XMethodMatcher):
    """Entry point for sycl::_V1::private_memory"""

    def __init__(self):
        gdb.xmethod.XMethodMatcher.__init__(self, "PrivateMemoryMatcher")

    def match(self, class_type, method_name):
        if method_name != "operator()":
            return None

        result = re.match(
            "^sycl::_V1::private_memory<((cl::)?(sycl::_V1::)?id<.+>), (.+)>$",
            class_type.tag,
        )
        if result is None:
            return None
        return PrivateMemoryOpCall(result[1], result[4])


gdb.xmethod.register_xmethod_matcher(None, AccessorMatcher(), replace=True)
gdb.xmethod.register_xmethod_matcher(None, PrivateMemoryMatcher(), replace=True)

### Pretty-printer implementations ###


class SyclArrayPrinter:
    """Print an object deriving from sycl::_V1::detail::array"""

    class ElementIterator:
        def __init__(self, data, size):
            self.data = data
            self.size = size
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.count == self.size:
                raise StopIteration
            count = self.count
            self.count = self.count + 1
            try:
                elt = self.data[count]
            except:
                elt = "<error reading variable>"
            return ("[%d]" % count, elt)

    def __init__(self, value):
        self.value = value
        if self.value.type.code == gdb.TYPE_CODE_REF:
            self.type = value.referenced_value().type.unqualified().strip_typedefs()
        else:
            self.type = value.type.unqualified().strip_typedefs()

        self.dimensions = self.type.template_argument(0)

    def children(self):
        try:
            return self.ElementIterator(self.value["common_array"], self.dimensions)
        except:
            # There is no way to return an error from this method. Return an
            # empty iterable to make GDB happy and rely on to_string method
            # to take care of formatting.
            return []

    def to_string(self):
        try:
            # Check if accessing array value will succeed and resort to
            # error message otherwise. Individual array element access failures
            # will be caught by iterator itself.
            _ = self.value["common_array"]
            if self.value.type.code == gdb.TYPE_CODE_REF:
                return "({tag} &) @{address}: {tag}".format(
                    tag=self.type.tag, address=self.value.address
                )
            return self.type.tag
        except:
            return "<error reading variable>"

    def display_hint(self):
        return "array"


class SyclBufferPrinter:
    """Print a sycl::_V1::buffer"""

    def __init__(self, value):
        self.value = value
        if self.value.type.code == gdb.TYPE_CODE_REF:
            self.type = value.referenced_value().type.unqualified().strip_typedefs()
        else:
            self.type = value.type.unqualified().strip_typedefs()

        self.elt_type = value.type.template_argument(0)
        self.dimensions = value.type.template_argument(1)
        self.typeregex = re.compile("^([a-zA-Z0-9_:]+)(<.*>)?$")

    def to_string(self):
        match = self.typeregex.match(self.type.tag)
        if not match:
            return "<error parsing type>"
        r_value = "{{impl={address}}}".format(address=self.value["impl"].address)
        r_type = "{group}<{elt_type}, {dim}>".format(
            group=match.group(1), elt_type=self.elt_type, dim=self.dimensions
        )
        if self.value.type.code == gdb.TYPE_CODE_REF:
            return "({type} &) @{address}: {type} = {value}".format(
                type=r_type, address=self.value.address, value=r_value
            )
        return "{type} = {value}".format(type=r_type, value=r_value)


sycl_printer = gdb.printing.RegexpCollectionPrettyPrinter("SYCL")
sycl_printer.add_printer("sycl::_V1::id", "^sycl::_V1::id<.*$", SyclArrayPrinter)
sycl_printer.add_printer("sycl::_V1::range", "^sycl::_V1::range<.*$", SyclArrayPrinter)
sycl_printer.add_printer(
    "sycl::_V1::buffer", "^sycl::_V1::buffer<.*$", SyclBufferPrinter
)
gdb.printing.register_pretty_printer(None, sycl_printer, True)
