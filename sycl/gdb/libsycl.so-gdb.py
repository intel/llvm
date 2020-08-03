# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gdb
import gdb.xmethod
import gdb.printing
import itertools
import re

### XMethod implementations ###

"""
Generalized base class for buffer index calculation
"""
class Accessor:
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
        if arg.type.code == gdb.TYPE_CODE_INT:
            return int(arg)
        # https://github.com/intel/llvm/blob/97272b7ebd569bfa13811913a31e30f926559217/sycl/include/CL/sycl/accessor.hpp#L678-L690
        result = 0
        for dim in range(self.depth):
            result = result * self.memory_range(dim) + \
                self.offset(dim) + \
                arg['common_array'][dim]
        return result

    def value(self, arg):
        return self.data().cast(self.result_type.pointer())[self.index(arg)]


"""
For Host device memory layout
"""
class HostAccessor(Accessor):
    def payload(self):
        return self.obj['impl']['_M_ptr'].dereference()

    def memory_range(self, dim):
        return self.payload()['MMemoryRange']['common_array'][dim]

    def offset(self, dim):
        return self.payload()['MOffset']['common_array'][dim]

    def data(self):
        return self.payload()['MData']

"""
For CPU/GPU memory layout
"""
class DeviceAccessor(Accessor):
    def memory_range(self, dim):
        return self.obj['impl']['MemRange']['common_array'][dim]

    def offset(self, dim):
        return self.obj['impl']['Offset']['common_array'][dim]

    def data(self):
        return self.obj['MData']


"""
Generic implementation for N-dimensional ID
"""
class AccessorOpIndex(gdb.xmethod.XMethodWorker):
    def __init__(self, class_type, result_type, depth):
        self.class_type = class_type
        self.result_type = result_type
        self.depth = depth

    def get_arg_types(self):
        return gdb.lookup_type("cl::sycl::id<%s>" % self.depth)

    def get_result_type(self):
        return self.result_type

    def __call__(self, obj, arg):
        # No way to wasily figure out which devices is currently being used,
        # try all accessor implementations until one of them works:
        accessors = [
            DeviceAccessor(obj, self.result_type, self.depth),
            HostAccessor(obj, self.result_type, self.depth)
        ]
        for accessor in accessors:
            try:
                return accessor.value(arg)
            except:
                pass

        print("Failed to call '%s.operator[](%s)" % (obj.type, arg.type))

        return None


"""
Introduces an extra overload for 1D case that takes plain size_t
"""
class AccessorOpIndex1D(AccessorOpIndex):
    def get_arg_types(self):
        assert(self.depth == 1)
        return gdb.lookup_type('size_t')


class AccessorOpIndexMatcher(gdb.xmethod.XMethodMatcher):
    def __init__(self):
        gdb.xmethod.XMethodMatcher.__init__(self, 'AccessorOpIndexMatcher')

    def match(self, class_type, method_name):
        if method_name != 'operator[]':
            return None

        result = re.match('^cl::sycl::accessor<.+>$', class_type.tag)
        if (result == None):
            return None

        depth = int(class_type.template_argument(1))
        result_type = class_type.template_argument(0)

        methods = [
            AccessorOpIndex(class_type, result_type, depth)
        ]
        if depth == 1:
            methods.append(AccessorOpIndex1D(class_type, result_type, depth))
        return methods


gdb.xmethod.register_xmethod_matcher(None, AccessorOpIndexMatcher(), replace=True)

### Pretty-printer implementations ###

"""
Print an object deriving from cl::sycl::detail::array
"""
class SyclArrayPrinter:
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
            return ('[%d]' % count, elt)

    def __init__(self, value):
        if value.type.code == gdb.TYPE_CODE_REF:
            if hasattr(gdb.Value,"referenced_value"):
                value = value.referenced_value()

        self.value = value
        self.type = value.type.unqualified().strip_typedefs()
        self.dimensions = self.type.template_argument(0)

    def children(self):
        try:
            return self.ElementIterator(self.value['common_array'], self.dimensions)
        except:
            # There is no way to return an error from this method. Return an
            # empty iterable to make GDB happy and rely on to_string method
            # to take care of formatting.
            return [ ]

    def to_string(self):
        try:
            # Check if accessing array value will succeed and resort to
            # error message otherwise. Individual array element access failures
            # will be caught by iterator itself.
            _ = self.value['common_array']
            return self.type.tag
        except:
            return "<error reading variable>"

    def display_hint(self):
        return 'array'

"""
Print a cl::sycl::buffer
"""
class SyclBufferPrinter:
    def __init__(self, value):
        self.value = value
        self.type = value.type.unqualified().strip_typedefs()
        self.elt_type = value.type.template_argument(0)
        self.dimensions = value.type.template_argument(1)
        self.typeregex = re.compile('^([a-zA-Z0-9_:]+)(<.*>)?$')

    def to_string(self):
        match = self.typeregex.match(self.type.tag)
        if not match:
            return "<error parsing type>"
        return ('%s<%s, %s> = {impl=%s}'
                % (match.group(1), self.elt_type, self.dimensions,
                   self.value['impl'].address))

sycl_printer = gdb.printing.RegexpCollectionPrettyPrinter("SYCL")
sycl_printer.add_printer("cl::sycl::id",     '^cl::sycl::id<.*$',     SyclArrayPrinter)
sycl_printer.add_printer("cl::sycl::range",  '^cl::sycl::range<.*$',  SyclArrayPrinter)
sycl_printer.add_printer("cl::sycl::buffer", '^cl::sycl::buffer<.*$', SyclBufferPrinter)
gdb.printing.register_pretty_printer(None, sycl_printer, True)

