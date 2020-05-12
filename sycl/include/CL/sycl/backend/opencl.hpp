
//==---------------- opencl.hpp - SYCL OpenCL backend ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/cl.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <> struct interop<backend::opencl, queue> {
  using type = cl_command_queue;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl, accessor<DataT, Dimensions, AccessMode,
                                         access::target::global_buffer,
                                         access::placeholder::false_t>> {
  using type = cl_mem;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
