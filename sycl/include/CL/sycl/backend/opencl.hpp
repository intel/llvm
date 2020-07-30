
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

template <> struct interop<backend::opencl, platform> {
  using type = cl_platform_id;
};

template <> struct interop<backend::opencl, device> {
  using type = cl_device_id;
};

template <> struct interop<backend::opencl, context> {
  using type = cl_context;
};

template <> struct interop<backend::opencl, queue> {
  using type = cl_command_queue;
};

template <> struct interop<backend::opencl, program> {
  using type = cl_program;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl, accessor<DataT, Dimensions, AccessMode,
                                         access::target::global_buffer,
                                         access::placeholder::false_t>> {
  using type = cl_mem;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl, accessor<DataT, Dimensions, AccessMode,
                                         access::target::constant_buffer,
                                         access::placeholder::false_t>> {
  using type = cl_mem;
};

namespace opencl {

// Implementation of various "make" functions resides in SYCL RT because
// creating SYCL objects requires knowing details not acessible here.
// Note that they take opaque pi_native_handle that real OpenCL handles
// are casted to.
//
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle);
__SYCL_EXPORT context make_context(pi_native_handle NativeHandle);
__SYCL_EXPORT program make_program(const context &Context,
                                   pi_native_handle NativeHandle);
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle InteropHandle);

// Construction of SYCL platform.
template <typename T, typename std::enable_if<
                          std::is_same<T, platform>::value>::type * = nullptr>
T make(typename interop<backend::opencl, T>::type Interop) {
  return make_platform(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T, typename std::enable_if<
                          std::is_same<T, device>::value>::type * = nullptr>
T make(typename interop<backend::opencl, T>::type Interop) {
  return make_device(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL context.
template <typename T, typename std::enable_if<
                          std::is_same<T, context>::value>::type * = nullptr>
T make(typename interop<backend::opencl, T>::type Interop) {
  return make_context(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL program.
template <typename T, typename std::enable_if<
                          std::is_same<T, program>::value>::type * = nullptr>
T make(const context &Context,
       typename interop<backend::opencl, T>::type Interop) {
  return make_program(Context, detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL queue.
template <typename T, typename std::enable_if<
                          std::is_same<T, queue>::value>::type * = nullptr>
T make(const context &Context,
       typename interop<backend::opencl, T>::type Interop) {
  return make_queue(Context, detail::pi::cast<pi_native_handle>(Interop));
}

} // namespace opencl
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
