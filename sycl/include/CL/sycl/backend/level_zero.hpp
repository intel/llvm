//==--------- level_zero.hpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <> struct interop<backend::level_zero, platform> {
  using type = ze_driver_handle_t;
};

template <> struct interop<backend::level_zero, device> {
  using type = ze_device_handle_t;
};

template <> struct interop<backend::level_zero, queue> {
  using type = ze_command_queue_handle_t;
};

template <> struct interop<backend::level_zero, program> {
  using type = ze_module_handle_t;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::level_zero, accessor<DataT, Dimensions, AccessMode,
                                             access::target::global_buffer,
                                             access::placeholder::false_t>> {
  using type = char *;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::level_zero, accessor<DataT, Dimensions, AccessMode,
                                             access::target::constant_buffer,
                                             access::placeholder::false_t>> {
  using type = char *;
};

namespace level_zero {

// Implementation of various "make" functions resides in libsycl.so
platform make_platform(pi_native_handle NativeHandle);
device make_device(const platform &Platform, pi_native_handle NativeHandle);
program make_program(const context &Context, pi_native_handle NativeHandle);
queue make_queue(const context &Context, pi_native_handle InteropHandle);

// Construction of SYCL platform.
template <typename T, typename std::enable_if<
                          std::is_same<T, platform>::value>::type * = nullptr>
T make(typename interop<backend::level_zero, T>::type Interop) {
  return make_platform(reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T, typename std::enable_if<
                          std::is_same<T, device>::value>::type * = nullptr>
T make(const platform &Platform,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_device(Platform, reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL program.
template <typename T, typename std::enable_if<
                          std::is_same<T, program>::value>::type * = nullptr>
T make(const context &Context,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_program(Context, reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL queue.
template <typename T, typename std::enable_if<
                          std::is_same<T, queue>::value>::type * = nullptr>
T make(const context &Context,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_queue(Context, reinterpret_cast<pi_native_handle>(Interop));
}

} // namespace level_zero
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
