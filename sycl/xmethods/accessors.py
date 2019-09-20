# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gdb.xmethod
import re

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


gdb.xmethod.register_xmethod_matcher(None, AccessorOpIndexMatcher())
