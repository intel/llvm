# ===-- sycl/gdb/libsycl.so-gdb.py - SYCL Pretty Printers -----------------===#
#
# Part of the LLVM Project, under the Apache license v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#
#
# Please follow the LLVM coding standard for python formatting under section
# "Python version and Source Code Formatting" here:
#   https://llvm.org/docs/CodingStandards.html
#
# This file has been formatted with the 'black' utility version 23.
#
# ===-----------------------------------------------------------------------===#

import gdb.printing
import gdb.xmethod


def strip_version(name):
    return name.replace("sycl::_V1::", "sycl::")


class SYCLType:
    """Wrapper around a gdb.Type."""

    """ Cached basic type information."""
    _char_type = None
    _size_type = None
    _void_type = None

    def __init__(self, gdb_type):
        self._gdb_type = gdb_type

    def gdb_type(self):
        return self._gdb_type

    @staticmethod
    def char_type():
        if SYCLType._char_type is None:
            SYCLType._char_type = gdb.lookup_type("char")
        return SYCLType._char_type

    @staticmethod
    def size_type():
        if SYCLType._size_type is None:
            SYCLType._size_type = gdb.lookup_type("size_t")
        return SYCLType._size_type

    @staticmethod
    def void_type():
        if SYCLType._void_type is None:
            SYCLType._void_type = gdb.lookup_type("void")
        return SYCLType._void_type


class SYCLAccessorType(SYCLType):
    """Provide type information for a sycl::accessor from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def base_type(self):
        return self.gdb_type().template_argument(0)

    def dimensions(self):
        return self.gdb_type().template_argument(1)

    def access_mode(self):
        return self.gdb_type().template_argument(2)


class SYCLBufferType(SYCLType):
    """Provide type information for a sycl::buffer from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def base_type(self):
        return self.gdb_type().template_argument(0)

    def dimensions(self):
        return self.gdb_type().template_argument(1)


class SYCLIdType(SYCLType):
    """Provides type information for a sycl::id from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def dimensions(self):
        return self.gdb_type().template_argument(0)


class SYCLLocalAccessorType(SYCLType):
    """Provide type information for a sycl::local_accessor from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def base_type(self):
        return self.gdb_type().template_argument(0)

    def dimensions(self):
        return self.gdb_type().template_argument(1)


class SYCLRangeType(SYCLType):
    """Provides type information for a sycl::range from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def dimensions(self):
        return self.gdb_type().template_argument(0)


class SYCLPrivateMemoryType(SYCLType):
    """Provides type information for a sycl::private_memory from a gdb.Type."""

    def __init__(self, gdb_type):
        super().__init__(gdb_type)

    def base_type(self):
        return self.gdb_type().template_argument(0)

    def dimensions(self):
        return self.gdb_type().template_argument(1)


class SYCLValue:
    """Wrapper around a gdb.Value."""

    def __init__(self, gdb_value):
        self._gdb_value = gdb_value

    def gdb_value(self):
        return self._gdb_value

    def gdb_type(self):
        return self._gdb_value.type

    """ Convenience method for converting an array pointer to a vector."""

    @staticmethod
    def cast_pointer_to_vector(pointer_value, vector_type):
        return pointer_value.cast(vector_type.pointer()).dereference()


class SYCLAccessor(SYCLValue):
    """Provides information about a sycl::accessor from a gdb.Value."""

    IMPL_OFFSET_TO_MOBJ = 0x78
    MOBJ_OFFSET_TO_SIZE = 0x70
    MOBJ_OFFSET_TO_DATA = 0x78

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def type(self):
        return SYCLAccessorType(self.gdb_type())

    def base_type(self):
        return self.type().base_type()

    def dimensions(self):
        return self.type().dimensions()

    def access_mode(self):
        return self.type().access_mode()

    def subscript_int(self, subscript):
        return self.data()[subscript]

    def subscript_id(self, subscript):
        id_array = SYCLId(subscript).array()
        result = self.data()
        for dimension in range(self.dimensions()):
            index = id_array[dimension]
            result = result[index]
        return result

    def subscript_item(self, subscript):
        return self.subscript_id(SYCLItem(subscript).index())

    def __getitem__(self, subscript):
        code = subscript.type.unqualified().code
        if code == gdb.TYPE_CODE_INT:
            return self.subscript_int(subscript)
        elif (
            code == gdb.TYPE_CODE_STRUCT
            and subscript.type.name.startswith("sycl::_V1::item<")
            and subscript.type.name.endswith(">")
        ):
            return self.subscript_item(subscript)
        elif (
            code == gdb.TYPE_CODE_STRUCT
            and subscript.type.name.startswith("sycl::_V1::id<")
            and subscript.type.name.endswith(">")
        ):
            return self.subscript_id(subscript)
        else:
            raise AttributeError("Unsupported sycl::accessor subscript type.")

    def impl(self):
        return self.gdb_value()["impl"]

    def impl_ptr(self):
        # Host only.
        return self.impl()["_M_ptr"]

    def offset(self):
        try:
            # __SYCL_DEVICE_ONLY__
            return self.impl()["Offset"]
        except:
            pass
        try:
            # Host
            return self.gdb_value()["MAccData"].dereference()["MOffset"]
        except:
            pass
        raise AttributeError("Unable to locate sycl::accessor offset.")

    def access_range(self):
        try:
            # __SYCL_DEVICE_ONLY__
            return self.impl()["AccessRange"]
        except:
            pass
        try:
            # Host
            return self.gdb_value()["MAccData"].dereference()["MAccessRange"]
        except:
            pass
        raise AttributeError("Unable to locate sycl::accessor access range.")

    def memory_range(self):
        try:
            # __SYCL_DEVICE_ONLY__
            return self.impl()["MemRange"]
        except:
            pass
        # Host
        try:
            return self.gdb_value()["MAccData"].dereference()["MMemoryRange"]
        except:
            pass
        raise AttributeError("Unable to locate sycl::accessor memory range.")

    def data_ptr(self):
        try:
            # __SYCL_DEVICE_ONLY__
            return self.gdb_value()["MData"]
        except:
            pass
        try:
            # Host
            base_type = self.base_type()
            char_ptr_type = SYCLType.char_type().pointer()
            size_ptr_type = SYCLType.size_type().pointer()
            base_ptr_type = base_type.pointer()
            impl_ptr = self.impl_ptr()
            cast_impl = impl_ptr.cast(char_ptr_type)
            addr_mobj = cast_impl + SYCLAccessor.IMPL_OFFSET_TO_MOBJ
            mobj_ptr = addr_mobj.cast(char_ptr_type.pointer()).dereference()
            addr_size = mobj_ptr + SYCLAccessor.MOBJ_OFFSET_TO_SIZE
            size = addr_size.cast(size_ptr_type).dereference()
            upperbound = size / base_type.sizeof
            inclusive_upperbound = upperbound - 1
            addr_data = mobj_ptr + SYCLAccessor.MOBJ_OFFSET_TO_DATA
            ptr = addr_data.cast(base_ptr_type.pointer()).dereference()
            return ptr
        except:
            pass
        raise AttributeError("Unable to locate sycl::accessor data.")

    def data(self):
        data_ptr = self.data_ptr()
        base_type = self.base_type()
        dims = self.dimensions()
        ranges_array = SYCLRange(self.access_range()).array()
        vector_type = SYCLPrinter.data_vector_type(base_type, dims, ranges_array)
        data = SYCLValue.cast_pointer_to_vector(data_ptr, vector_type)
        return data


class SYCLLocalAccessor(SYCLValue):
    """Provides information about a sycl::local_accessor from a gdb.Value."""

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def type(self):
        return SYCLLocalAccessorType(self.gdb_type())

    def base_type(self):
        return self.type().base_type()

    def dimensions(self):
        return self.type().dimensions()

    def impl(self):
        return self.gdb_value()["impl"]

    def memory_range(self):
        # __SYCL_DEVICE_ONLY__
        return self.impl()["MemRange"]

    def data_ptr(self):
        # __SYCL_DEVICE_ONLY__
        return self.gdb_value()["MData"]

    def data(self):
        data_ptr = self.data_ptr()
        base_type = self.base_type()
        dims = self.dimensions()
        array = SYCLRange(self.memory_range()).array()
        vector_type = SYCLPrinter.data_vector_type(base_type, dims, array)
        data = SYCLValue.cast_pointer_to_vector(data_ptr, vector_type)
        return data

    def subscript_sizet(self, subscript):
        return self.data()[subscript]

    def subscript_id(self, subscript):
        id_array = SYCLId(subscript).array()
        result = self.data()
        for dimension in range(self.dimensions()):
            index = id_array[dimension]
            result = result[index]
        return result

    def subscript_item(self, subscript):
        return self.subscript_id(SYCLItem(subscript).index())


class SYCLBuffer(SYCLValue):
    """Provides information about a sycl::buffer from a gdb.Value."""

    IMPL_OFFSET_TO_POINTER = 0x78

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def type(self):
        return SYCLBufferType(self.gdb_type())

    def base_type(self):
        return self.type().base_type()

    def dimensions(self):
        return self.type().dimensions()

    def impl_ptr(self):
        return self.gdb_value()["impl"]["_M_ptr"]

    def host_ptr(self):
        base_type = self.base_type()
        char_pointer = SYCLType.char_type().pointer()
        impl = self.impl_ptr().cast(char_pointer)
        host_address = impl.cast(char_pointer) + self.IMPL_OFFSET_TO_POINTER
        host_casted = host_address.cast(base_type.pointer().pointer())
        host_pointer = host_casted.dereference()
        return host_pointer

    def range(self):
        return self.gdb_value()["Range"]

    def range_common_array(self):
        return self.range()["common_array"]


class SYCLDevice(SYCLValue):
    """Provides information about a sycl::device from a gdb.Value."""

    IMPL_OFFSET_TO_DEVICE_TYPE = 0x8
    IMPL_OFFSET_TO_PLATFORM = 0x18
    PLATFORM_OFFSET_TO_BACKEND = 0x10

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def impl_ptr(self):
        return self.gdb_value()["impl"]["_M_ptr"]

    def device_type(self):
        char_ptr = SYCLType.char_type().pointer()
        device_addr = self.impl_ptr().cast(char_ptr)
        device_type_addr = device_addr + self.IMPL_OFFSET_TO_DEVICE_TYPE
        uint32t_ptr = gdb.lookup_type("uint32_t").pointer()
        device_type = device_type_addr.cast(uint32t_ptr).dereference()
        return device_type

    def backend(self):
        char_ptr = SYCLType.char_type().pointer()
        impl_ptr = self.impl_ptr().cast(char_ptr)
        platform_addr = impl_ptr + self.IMPL_OFFSET_TO_PLATFORM
        platform = platform_addr.cast(char_ptr.pointer()).dereference()
        backend_addr = platform + self.PLATFORM_OFFSET_TO_BACKEND
        backend = backend_addr.dereference()
        return backend


class SYCLId(SYCLValue):
    """Provides information about a sycl::id from gdb.Value."""

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def type(self):
        return SYCLIdType(self.gdb_type())

    def dimensions(self):
        return self.type().dimensions()

    def array(self):
        return self.gdb_value()["common_array"]


class SYCLItem(SYCLValue):
    """Provides information about a sycl::item from a gdb.Value."""

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def impl(self):
        return self.gdb_value()["MImpl"]

    def index(self):
        return self.impl()["MIndex"]

    def extent(self):
        return self.impl()["MExtent"]

    def offset(self):
        return self.impl()["MOffset"]


class SYCLQueue(SYCLValue):
    """Provides information about a sycl::queue from a gdb.Value."""

    DEVICE_TYPE_NAME = "sycl::_V1::device"
    IMPL_OFFSET_TO_DEVICE = 0x28

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def impl_ptr(self):
        return self.gdb_value()["impl"]["_M_ptr"]

    def device(self):
        char_ptr = SYCLType.char_type().pointer()
        impl_ptr = self.impl_ptr().cast(char_ptr)
        device_addr = impl_ptr.cast(char_ptr) + self.IMPL_OFFSET_TO_DEVICE
        device_ptr = gdb.lookup_type(SYCLQueue.DEVICE_TYPE_NAME).pointer()
        device = device_addr.cast(device_ptr).dereference()
        return device


class SYCLRange(SYCLValue):
    """Provides information about a sycl::range from a gdb.Value."""

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def type(self):
        return SYCLRangeType(self.gdb_type())

    def dimensions(self):
        return self.type().dimensions()

    def array(self):
        return self.gdb_value()["common_array"]


class SYCLTypePrinter:
    """SYCL wrapper around a gdb type printer."""

    class Recognizer:
        def __init__(self, printer):
            self._printer = printer

        def recognize(self, gdb_type, first=True):
            if gdb_type.code == gdb.TYPE_CODE_REF:
                result = self.recognize(gdb_type.target(), False)
                if result:
                    return result + "&"
            elif gdb_type.code == gdb.TYPE_CODE_RVALUE_REF:
                result = self.recognize(gdb_type.target(), False)
                if result:
                    return result + "&&"
            elif gdb_type.code == gdb.TYPE_CODE_PTR:
                result = self.recognize(gdb_type.target(), False)
                if result:
                    return result + "*"
            elif self._printer.matches(gdb_type):
                result = self._printer.to_string(gdb_type)
                if not first:
                    result = result + " "
                return result
            return None

    def to_string(self, gdb_type):
        return strip_version(gdb_type.name)

    def __init__(self, name):
        self.enabled = True
        self.name = name

    def matches(self, type):
        return type.name == self.name or type.name.startswith(self.name + "<")

    def instantiate(self):
        return SYCLTypePrinter.Recognizer(self)


class SYCLAccessorTypePrinter(SYCLTypePrinter):
    """Type printer for a sycl::accessor.

    Examples:
      (gdb) whatis accessor_1d_read_only
      type = sycl::accessor<int, 1, read, device>

      (gdb) whatis accessor_2d_read_write
      type = sycl::accessor<int, 2, read_write, device>

      (gdb) whatis accessor_1d_reference
      type = sycl::accessor<int, 1, read, device> &
    """

    def __init__(self, name):
        super().__init__(name)

    def to_string(self, gdb_type):
        string = gdb_type.name
        string = self.substitute_access_mode(string)
        string = self.substitute_access_target(string)
        string = self.substitute_default_args(string)
        string = strip_version(string)
        return string

    @staticmethod
    def substitute_access_mode(string):
        access_mode = "(sycl::_V1::access::mode)"
        substitutions = [
            (access_mode + "1024", "read"),
            (access_mode + "1025", "write"),
            (access_mode + "1026", "read_write"),
            (access_mode + "1027", "discard_write"),
            (access_mode + "1028", "discard_read_write"),
            (access_mode + "1029", "atomic"),
        ]
        for old, new in substitutions:
            string = string.replace(old, new)
        return string

    @staticmethod
    def substitute_access_target(string):
        access_target = "(sycl::_V1::access::target)"
        substitutions = [
            (access_target + "2014", "device"),
            (access_target + "2015", "constant_buffer"),
            (access_target + "2016", "local"),
            (access_target + "2017", "image"),
            (access_target + "2018", "host_buffer"),
            (access_target + "2019", "host_image"),
            (access_target + "2020", "image_array"),
            (access_target + "2021", "host_task"),
        ]
        for old, new in substitutions:
            string = string.replace(old, new)
        return string

    @staticmethod
    def substitute_default_args(string):
        substitutions = [
            (", sycl::_V1::ext::oneapi::accessor_property_list<> ", ""),
            (", (sycl::_V1::access::placeholder)0", ""),
        ]
        for old, new in substitutions:
            string = string.replace(old, new)
        return string


class SYCLBufferTypePrinter(SYCLTypePrinter):
    """Type printer for a sycl::buffer.

    Examples:
      (gdb) whatis buffer_1d
      type = sycl::buffer<int>

      (gdb) whatis buffer_2d
      type = sycl::buffer<int, 2>

      (gdb) whatis buffer_with_custom_allocator
      type = sycl::buffer<int, 1, std::allocator<int>>
    """

    def __init__(self, name):
        super().__init__(name)

    def to_string(self, gdb_type):
        base_type = SYCLBufferType(gdb_type).base_type()
        string = gdb_type.name
        string = self.substitute_default_args(string, base_type)
        string = strip_version(string)
        return string

    @staticmethod
    def substitute_default_args(string, base_type):
        substitutions = [
            (", void>", ">"),
            (", sycl::_V1::detail::aligned_allocator<" + str(base_type) + ">>", ">"),
            (", 1>", ">"),
        ]
        for old, new in substitutions:
            string = string.replace(old, new)
        return string


class SYCLItemTypePrinter(SYCLTypePrinter):
    """Type printer for a sycl::item.

    Examples:
      (gdb) whatis item
      type = sycl::item<1>
    """

    def __init__(self, name):
        super().__init__(name)

    def to_string(self, gdb_type):
        string = gdb_type.name
        string = self.substitute_default_args(string)
        string = strip_version(string)
        return string

    @staticmethod
    def substitute_default_args(string):
        substitutions = [(", true>", ">")]
        for old, new in substitutions:
            string = string.replace(old, new)
        return string


class SYCLTypePrinters:
    PRINTERS = [
        (SYCLAccessorTypePrinter, "sycl::_V1::accessor"),
        (SYCLBufferTypePrinter, "sycl::_V1::buffer"),
        (SYCLTypePrinter, "sycl::_V1::device"),
        (SYCLTypePrinter, "sycl::_V1::exception"),
        (SYCLTypePrinter, "sycl::_V1::handler"),
        (SYCLTypePrinter, "sycl::_V1::id"),
        (SYCLItemTypePrinter, "sycl::_V1::item"),
        (SYCLTypePrinter, "sycl::_V1::local_accessor"),
        (SYCLTypePrinter, "sycl::_V1::queue"),
        (SYCLTypePrinter, "sycl::_V1::range"),
    ]

    def register(self):
        for printer, name in SYCLTypePrinters.PRINTERS:
            gdb.types.register_type_printer(None, printer(name))


class SYCLPrinter:
    """Base class for SYCL pretty printers."""

    UNKNOWN = "unknown"

    def __init__(self, gdb_value):
        self._gdb_value = gdb_value

    def gdb_value(self):
        return self._gdb_value

    def gdb_type(self):
        return self._gdb_value.type

    @staticmethod
    def data_vector_type(base_type, dimensions, ranges_array):
        vector_type = base_type
        for index in range(dimensions):
            upperbound = ranges_array[index]
            inclusive_upperbound = upperbound - 1
            vector_type = vector_type.vector(inclusive_upperbound)
        return vector_type

    @staticmethod
    def type_name(gdb_type):
        result = None
        if gdb_type.code == gdb.TYPE_CODE_REF:
            result = SYCLPrinter.type_name(gdb_type.target())
            if result:
                result += "&"
        elif gdb_type.code == gdb.TYPE_CODE_RVALUE_REF:
            result = SYCLPrinter.type_name(gdb_type.target())
            if result:
                result += "&&"
        elif gdb_type.code == gdb.TYPE_CODE_PTR:
            result = SYCLPrinter.type_name(gdb_type.target())
            if result:
                result += "*"
        elif gdb_type.code == gdb.TYPE_CODE_STRUCT:
            name = strip_version(str(gdb_type))
            result = name.partition("<")[0]
        return result


class SYCLAccessModePrinter(SYCLPrinter):
    """Pretty printer for a sycl::_V1::access::mode

    Currently, this printer is only used indirectly by the SYCLAccessorPrinter.
    """

    def __init__(self, value):
        super().__init__(value)

    def to_string(self):
        string = str(self.gdb_value())
        string = string.removeprefix("sycl::_V1::access::mode::")
        return string


class SYCLAccessorPrinter(SYCLPrinter):
    """Pretty printer for a sycl::accessor.

    Examples:
      (gdb) p accessor_1d_read_only
      $1 = sycl::accessor read range 10 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

      (gdb) p acessor_2d_read_write
      $2 = sycl::accessor read_write range {3, 3} = {{1, 2, 3}, {4, 5, 6}, ...}
    """

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def display_hint(self):
        return "array"

    def to_string(self):
        sycl_accessor = SYCLAccessor(self.gdb_value())
        string = self.type_name(self.gdb_type())
        mode = sycl_accessor.access_mode()
        string += " " + SYCLAccessModePrinter(mode).to_string()
        try:
            access = SYCLRangePrinter(sycl_accessor.access_range()).value_as_string()
            string += " range " + access
        except:
            string += " range " + self.UNKNOWN
        try:
            offset = SYCLIdPrinter(sycl_accessor.offset()).value_as_string()
            if offset not in ["0", "{0, 0}", "{0, 0, 0}"]:
                string += " offset " + offset
        except:
            string += " offset " + self.UNKNOWN
        return string

    def children(self):
        try:
            sycl_accessor = SYCLAccessor(self.gdb_value())
            data = sycl_accessor.data()
            (low, inclusive_high) = data.type.range()
            high = inclusive_high + 1
            for index in range(low, high):
                yield (f"[{index}]", data[index])
        except:
            return None


class SYCLLocalAccessorPrinter(SYCLPrinter):
    """Pretty printer for a sycl::local_accessor.

    Examples:
      # In device code.
      (gdb) p local_accessor_1d
      $1 = sycl::local_accessor range 10 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

      # In host code.
      (gdb) p local_accessor_1d
      $2 = sycl::local_accessor undefined
    """

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def display_hint(self):
        return "array"

    def to_string(self):
        local_accessor = SYCLLocalAccessor(self.gdb_value())
        string = self.type_name(self.gdb_type())
        try:
            mem_range = local_accessor.memory_range()
            range_string = SYCLRangePrinter(mem_range).value_as_string()
            string += " range " + range_string
        except:
            string += " undefined"
        return string

    def children(self):
        try:
            local_accessor = SYCLLocalAccessor(self.gdb_value())
            data = local_accessor.data()
            (low, inclusive_high) = data.type.range()
            high = inclusive_high + 1
            for index in range(low, high):
                yield (f"[{index}]", data[index])
        except:
            return None


class SYCLBufferPrinter(SYCLPrinter):
    """Pretty printer for a sycl::buffer.

    Examples:
      (gdb) p buffer_1d
      $1 = sycl::buffer range 10 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

      (gdb) p buffer_3d
      $2 = sycl::buffer range {2, 2, 2} = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}

      (gdb) p buffer_ref
      $3 = sycl::buffer& range 10 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    """

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def display_hint(self):
        return "array"

    def to_string(self):
        buffer = SYCLBuffer(self.gdb_value())
        name = self.type_name(self.gdb_type())
        dims = SYCLRangePrinter(buffer.range()).value_as_string()
        return f"{name} range {dims}"

    def children(self):
        try:
            buffer = SYCLBuffer(self.gdb_value())
            base_type = buffer.base_type()
            dims = buffer.dimensions()
            host_ptr = buffer.host_ptr()
            ranges_array = buffer.range_common_array()
            vector_type = SYCLPrinter.data_vector_type(base_type, dims, ranges_array)
            vector = SYCLValue.cast_pointer_to_vector(host_ptr, vector_type)
            (low, inclusive_high) = vector_type.range()
            high = inclusive_high + 1
            for index in range(low, high):
                yield (f"[{index}]", vector[index])
        except:
            return None


class SYCLDevicePrinter(SYCLPrinter):
    """Pretty printer for a sycl::device.

    Example:
      (gdb) p q.get_device()
      $1 = sycl::device using opencl on cpu = {
        impl = 0x012345678
      }
    """

    def __init__(self, value):
        super().__init__(value)

    def device_type_as_string(self):
        sycl_device = SYCLDevice(self.gdb_value())
        try:
            device_type = sycl_device.device_type()
            string = {
                "1": "default",
                "2": "all",
                "3": "gpu",
                "4": "cpu",
                "5": "fpga",
                "6": "mca",
                "7": "vpu",
            }[str(device_type)]
        except:
            string = self.UNKNOWN
        return string

    def backend_as_string(self):
        sycl_device = SYCLDevice(self.gdb_value())
        try:
            backend = sycl_device.backend()
            string = {
                "0": "host",
                "1": "opencl",
                "2": "level zero",
                "3": "cuda",
                "4": "all",
                "5": "intel esimd emulator",
                "6": "oneapi hip",
                "7": "oneapi native cpu",
            }[str(int(backend))]
        except:
            string = self.UNKNOWN
        return string

    def to_string(self):
        string = self.type_name(self.gdb_type())
        backend = self.backend_as_string()
        if backend != self.UNKNOWN:
            string += " using " + backend
        device = self.device_type_as_string()
        if device != self.UNKNOWN:
            string += " on " + device
        if backend == self.UNKNOWN and device == self.UNKNOWN:
            string = " " + self.UNKNOWN
        return string

    def children(self):
        sycl_device = SYCLDevice(self.gdb_value())
        impl = sycl_device.impl_ptr()
        return [("impl", impl)]


class SYCLExceptionPrinter(SYCLPrinter):
    """Pretty printer for a sycl::exception.

    Example:
      (gdb) print runtime_exception
      $1 = sycl::exception = {
        MErrC = sycl::_V1::errc::runtime,
        MMsg= 0x4b9f70 "Runtime Message"
      }
    """

    def __init__(self, value):
        super().__init__(value)

    def merrc(self):
        return self.gdb_value()["MErrC"]

    def error_code(self):
        errc_type = gdb.lookup_type("sycl::_V1::errc")
        return self.merrc()["_M_value"].cast(errc_type)

    def mmsg_ptr(self):
        return self.gdb_value()["MMsg"]["_M_ptr"]

    def message(self):
        mmsg = self.mmsg_ptr()
        type = SYCLType.char_type().pointer()
        message = mmsg.cast(type.pointer()).dereference()
        return message

    def to_string(self):
        return self.type_name(self.gdb_type())

    def children(self):
        try:
            errcode = self.error_code()
        except:
            errcode = self.UNKNOWN
        try:
            message = self.message()
        except:
            message = self.UNKNOWN
        return [("MErrC", errcode), ("MMsg", message)]


class SYCLHandlerPrinter(SYCLPrinter):
    """Pretty printer for a sycl::handler"""

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def impl_ptr(self):
        return self.gdb_value()["impl"]["_M_ptr"]

    def to_string(self):
        return self.type_name(self.gdb_type())

    def children(self):
        try:
            impl = self.impl_ptr()
        except:
            impl = self.UNKNOWN
        return [("impl", impl)]


class SYCLIdPrinter(SYCLPrinter):
    """Pretty printer for a sycl::id.

    Examples:
      (gdb) p id1
      $1 = sycl::id<1> = 9
      (gdb) p id2
      $2 = sycl::id<2> = {1, 0}
    """

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def value_as_string(self):
        try:
            sycl_id = SYCLId(self.gdb_value())
            value = sycl_id.array()
            if sycl_id.dimensions() == 1:
                value = value[0]
            value = str(value)
        except:
            value = self.UNKNOWN
        return value

    def to_string(self):
        name = self.type_name(self.gdb_type())
        value = self.value_as_string()
        return f"{name} = {value}"


class SYCLItemPrinter(SYCLPrinter):
    """Pretty printer for a sycl::item.

    Example:
      (gdb) p item
      $1 = sycl::item range {2, 2, 2} = {1, 1, 1}
    """

    def __init__(self, value):
        super().__init__(value)

    def value_as_string(self):
        index = SYCLItem(self.gdb_value()).index()
        return SYCLIdPrinter(index).value_as_string()

    def to_string(self):
        sycl_item = SYCLItem(self.gdb_value())
        string = self.type_name(self.gdb_type())
        extent = SYCLRangePrinter(sycl_item.extent()).value_as_string()
        string += " range " + extent
        offset_id = SYCLItem(self.gdb_value()).offset()
        offset = SYCLIdPrinter(offset_id).value_as_string()
        if offset not in ["0", "{0, 0}", "{0, 0, 0}"]:
            string += ", offset " + offset
        string += " = " + self.value_as_string()
        return string


class SYCLQueuePrinter(SYCLPrinter):
    """Pretty printer for a sycl::queue.

    Example:
      (gdb) print queue
      $1 = sycl::queue = {
        [device] = sycl::device using opencl on cpu = {
          impl = 0x719940
        }
      }
    """

    def __init__(self, value):
        super().__init__(value)

    def to_string(self):
        return self.type_name(self.gdb_type())

    def children(self):
        device = SYCLQueue(self.gdb_value()).device()
        return [("[device]", device)]


class SYCLRangePrinter(SYCLPrinter):
    """Pretty printer for a sycl::range.

    Example:
      (gdb) p range_2d
      $1 = sycl::range = {3, 3}
    """

    def __init__(self, gdb_value):
        super().__init__(gdb_value)

    def value_as_string(self):
        try:
            sycl_range = SYCLRange(self.gdb_value())
            value = sycl_range.array()
            if sycl_range.dimensions() == 1:
                value = value[0]
            value = str(value)
        except:
            value = self.UNKNOWN
        return value

    def to_string(self):
        name = self.type_name(self.gdb_type())
        value = self.value_as_string()
        return f"{name} = {value}"


class SYCLPrettyPrinters:
    """Collection of all SYCL pretty printers"""

    PRINTERS = [
        (SYCLAccessorPrinter, [("sycl::_V1::accessor", "^sycl::_V1::accessor<.*>$")]),
        (SYCLBufferPrinter, [("sycl::_V1::buffer", "^sycl::_V1::buffer<.*>$")]),
        (SYCLDevicePrinter, [("sycl::_V1::device", "^sycl::_V1::device$")]),
        (SYCLExceptionPrinter, [("sycl::_V1::exception", "^sycl::_V1::exception$")]),
        (SYCLHandlerPrinter, [("sycl::_V1::handler", "^sycl::_V1::handler$")]),
        (SYCLIdPrinter, [("sycl::_V1::id", "^sycl::_V1::id<.*>$")]),
        (SYCLItemPrinter, [("sycl::_V1::item", "^sycl::_V1::item<.*>$")]),
        (
            SYCLLocalAccessorPrinter,
            [("sycl::_V1::local_accessor", "^sycl::_V1::local_accessor<.*>$")],
        ),
        (SYCLQueuePrinter, [("sycl::_V1::queue", "^sycl::_V1::queue$")]),
        (SYCLRangePrinter, [("sycl::_V1::range", "^sycl::_V1::range<.*>$")]),
    ]

    def __init__(self):
        self._printer = gdb.printing.RegexpCollectionPrettyPrinter("libsycl")
        for reference, descriptions in self.PRINTERS:
            for name, regx in descriptions:
                self._printer.add_printer(name, regx, reference)

    def register(self):
        replace = True
        gdb.printing.register_pretty_printer(None, self._printer, replace)


class SYCLXMethod(gdb.xmethod.XMethod):
    """SYCL interface for a GDB XMethod."""

    def __init__(self, display_name, name, worker):
        super().__init__(display_name)
        self._name = name
        self._worker = worker

    def xdisplay_name(self):
        return self.name

    def xname(self):
        return self._name

    def xenabled(self):
        return self.enabled

    def xworker(self):
        return self._worker


class SYCLXMethodMatcher(gdb.xmethod.XMethodMatcher):
    """SYCL interface to the GDB xmethod matcher."""

    def __init__(self, name, workers):
        super().__init__(name)
        self.methods = [SYCLXMethod(D, N, W) for (D, N, W) in workers]

    def match(self, class_type, method_name):
        result = []
        for xm in self.methods:
            if xm.xenabled() and xm.xname() == method_name:
                worker = xm.xworker()(class_type, method_name)
                result.append(worker)
        return result


class SYCLXMethodWorker(gdb.xmethod.XMethodWorker):
    """SYCL interface to the GDB xmethod worker.

    Derived objects must provide the following routines:
      - def get_arg_types(self):
      - def get_result_type(self, *args):
      - def __call__(self, *args):
    """

    def __init__(self, class_type, method_name):
        super().__init__()
        self._class_type = class_type
        self._method_name = method_name

    def class_type(self):
        return self._class_type

    def method_name(self):
        return self._method_name


class SYCLAccessorSubscript(SYCLXMethodWorker):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_result_type(self, *args):
        return SYCLAccessorType(self.class_type()).base_type()


class SYCLAccessorSubscriptSizeT(SYCLAccessorSubscript):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_arg_types(self):
        return self.size_type()

    def size_type(self):
        return SYCLType.size_type()

    def __call__(self, accessor_ptr, subscript):
        return SYCLAccessor(accessor_ptr.dereference())[subscript]


class SYCLAccessorSubscriptID(SYCLAccessorSubscript):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_arg_types(self):
        return self.id_type()

    def id_type(self):
        dimensions = SYCLAccessorType(self.class_type()).dimensions()
        return gdb.lookup_type(f"sycl::_V1::id<{dimensions}>")

    def __call__(self, accessor_ptr, subscript):
        return SYCLAccessor(accessor_ptr.dereference())[subscript]


class SYCLAccessorSubscriptItem(SYCLAccessorSubscript):
    def __init__(self, class_type, method_name, with_offset):
        super().__init__(class_type, method_name)
        self._with_offset = with_offset

    def with_offset(self):
        return self._with_offset

    def get_arg_types(self):
        return self.item_type()

    def item_type(self):
        try:
            dim = str(SYCLAccessorType(self.class_type()).dimensions())
            off = "true" if self.with_offset() else "false"
            return gdb.lookup_type(f"sycl::_V1::item<{dim}, {off}>")
        except:
            return None

    def __call__(self, accessor_ptr, subscript):
        return SYCLAccessor(accessor_ptr.dereference())[subscript]


class SYCLAccessorSubscriptItemOffset(SYCLAccessorSubscriptItem):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name, True)


class SYCLAccessorSubscriptItemNoOffset(SYCLAccessorSubscriptItem):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name, False)


class SYCLAccessorMatcher(SYCLXMethodMatcher):
    WORKERS = [
        ("operator[size_t]", "operator[]", SYCLAccessorSubscriptSizeT),
        ("operator[id]", "operator[]", SYCLAccessorSubscriptID),
        ("operator[item<true>]", "operator[]", SYCLAccessorSubscriptItemOffset),
        ("operator[item<false>]", "operator[]", SYCLAccessorSubscriptItemNoOffset),
    ]

    def __init__(self, name):
        super().__init__(name, self.WORKERS)

    def match(self, class_type, method_name):
        if not class_type.tag.startswith(
            "sycl::_V1::accessor<"
        ) or not class_type.tag.endswith(">"):
            return None
        return super().match(class_type, method_name)


class SYCLLocalAccessorSubscript(SYCLXMethodWorker):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_result_type(self, *args):
        return SYCLLocalAccessorType(self.class_type()).base_type()


class SYCLLocalAccessorSubscriptSizeT(SYCLAccessorSubscript):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_arg_types(self):
        return SYCLType.size_type()

    def __call__(self, ptr, subscript):
        return SYCLLocalAccessor(ptr.dereference()).subscript_sizet(subscript)


class SYCLLocalAccessorSubscriptID(SYCLLocalAccessorSubscript):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_arg_types(self):
        return self.id_type()

    def id_type(self):
        dimensions = SYCLLocalAccessorType(self.class_type()).dimensions()
        return gdb.lookup_type(f"sycl::_V1::id<{dimensions}>")

    def __call__(self, ptr, subscript):
        return SYCLLocalAccessor(ptr.dereference()).subscript_id(subscript)


class SYCLLocalAccessorSubscriptItem(SYCLLocalAccessorSubscript):
    def __init__(self, class_type, method_name, with_offset):
        super().__init__(class_type, method_name)
        self._with_offset = with_offset

    def with_offset(self):
        return self._with_offset

    def get_arg_types(self):
        return self.item_type()

    def item_type(self):
        try:
            dim = str(SYCLLocalAccessorType(self.class_type()).dimensions())
            off = "true" if self.with_offset() else "false"
            return gdb.lookup_type(f"sycl::_V1::item<{dim}, {off}>")
        except:
            return None

    def __call__(self, ptr, subscript):
        return SYCLLocalAccessor(ptr.dereference()).subscript_item(subscript)


class SYCLLocalAccessorSubscriptItemOffset(SYCLLocalAccessorSubscriptItem):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name, True)


class SYCLLocalAccessorSubscriptItemNoOffset(SYCLLocalAccessorSubscriptItem):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name, False)


class SYCLLocalAccessorMatcher(SYCLXMethodMatcher):
    WORKERS = [
        ("operator[size_t]", "operator[]", SYCLLocalAccessorSubscriptSizeT),
        ("operator[id]", "operator[]", SYCLLocalAccessorSubscriptID),
        ("operator[item<true>]", "operator[]", SYCLLocalAccessorSubscriptItemOffset),
        ("operator[item<false>]", "operator[]", SYCLLocalAccessorSubscriptItemNoOffset),
    ]

    def __init__(self, name):
        super().__init__(name, self.WORKERS)

    def match(self, class_type, method_name):
        if not class_type.tag.startswith(
            "sycl::_V1::local_accessor<"
        ) or not class_type.tag.endswith(">"):
            return None
        return super().match(class_type, method_name)


class SYCLPrivateMemoryCallHItem(SYCLXMethodWorker):
    def __init__(self, class_type, method_name):
        super().__init__(class_type, method_name)

    def get_result_type(self, *args):
        return SYCLPrivateMemoryType(self.class_type()).base_type().reference()

    def get_arg_types(self):
        dimensions = str(SYCLPrivateMemoryType(self.class_type()).dimensions())
        h_item_type = gdb.lookup_type(f"sycl::_V1::h_item<{dimensions}>")
        return h_item_type

    def __call__(self, private_memory, h_item):
        return private_memory["Val"].reference_value()


class SYCLPrivateMemoryMatcher(SYCLXMethodMatcher):
    WORKERS = [("operator()(sycl::h_item)", "operator()", SYCLPrivateMemoryCallHItem)]

    def __init__(self, name):
        super().__init__(name, self.WORKERS)

    def match(self, class_type, method_name):
        if not class_type.tag.startswith(
            "sycl::_V1::private_memory<"
        ) or not class_type.tag.endswith(">"):
            return None
        return super().match(class_type, method_name)


class SYCLXMethods:
    """Handles registration of the SYCL XMethods."""

    MATCHERS = [
        ("sycl::accessor", SYCLAccessorMatcher),
        ("sycl::local_accessor", SYCLLocalAccessorMatcher),
        ("sycl::private_memory", SYCLPrivateMemoryMatcher),
    ]

    def register(self):
        replace = True
        for name, matcher in self.MATCHERS:
            gdb.xmethod.register_xmethod_matcher(None, matcher(name), replace)


print("Registering SYCL extensions for gdb")
SYCLPrettyPrinters().register()
SYCLTypePrinters().register()
SYCLXMethods().register()
