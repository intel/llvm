//===---- backend_traits_level_zero.hpp - Backend traits for Level Zero ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the specializations of the sycl::detail::interop,
// sycl::detail::BackendInput, sycl::detail::BackendReturn and
// sycl::detail::InteropFeatureSupportMap class templates for the Level Zero
// backend.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/queue.hpp>
#include <sycl/ext/oneapi/backend/level_zero_ownership.hpp>

typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_event_handle_t *ze_event_handle_t;
typedef struct _ze_image_handle_t *ze_image_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes. The interop<backend, queue> specialization
// is also used in the get_queue() method of the deprecated class
// interop_handler and also can be removed after API cleanup.
template <> struct interop<backend::ext_oneapi_level_zero, context> {
  using type = ze_context_handle_t;
};

template <> struct interop<backend::ext_oneapi_level_zero, device> {
  using type = ze_device_handle_t;
};

template <> struct interop<backend::ext_oneapi_level_zero, event> {
  using type = ze_event_handle_t;
};

template <> struct interop<backend::ext_oneapi_level_zero, queue> {
  using type = ze_command_queue_handle_t;
};

template <> struct interop<backend::ext_oneapi_level_zero, platform> {
  using type = ze_driver_handle_t;
};

#ifdef __SYCL_INTERNAL_API
template <> struct interop<backend::ext_oneapi_level_zero, program> {
  using type = ze_module_handle_t;
};
#endif

// TODO the interops for accessor is used in the already deprecated class
// interop_handler and can be removed after API cleanup.
template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::ext_oneapi_level_zero,
               accessor<DataT, Dimensions, AccessMode, access::target::device,
                        access::placeholder::false_t>> {
  using type = char *;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<
    backend::ext_oneapi_level_zero,
    accessor<DataT, Dimensions, AccessMode, access::target::constant_buffer,
             access::placeholder::false_t>> {
  using type = char *;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::ext_oneapi_level_zero,
               accessor<DataT, Dimensions, AccessMode, access::target::image,
                        access::placeholder::false_t>> {
  using type = ze_image_handle_t;
};

template <> struct interop<backend::ext_oneapi_level_zero, kernel> {
  using type = ze_kernel_handle_t;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, context> {
  struct type {
    interop<backend::ext_oneapi_level_zero, context>::type NativeHandle;
    std::vector<device> DeviceList;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, context> {
  using type = ze_context_handle_t;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, device> {
  using type = ze_device_handle_t;
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, device> {
  using type = ze_device_handle_t;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, event> {
  struct type {
    interop<backend::ext_oneapi_level_zero, event>::type NativeHandle;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, event> {
  using type = ze_event_handle_t;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, queue> {
  struct type {
    interop<backend::ext_oneapi_level_zero, queue>::type NativeHandle;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, queue> {
  using type = ze_command_queue_handle_t;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, platform> {
  using type = ze_driver_handle_t;
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, platform> {
  using type = ze_driver_handle_t;
};

#ifdef __SYCL_INTERNAL_API
template <> struct BackendInput<backend::ext_oneapi_level_zero, program> {
  using type = ze_module_handle_t;
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, program> {
  using type = ze_module_handle_t;
};
#endif

template <bundle_state State>
struct BackendInput<backend::ext_oneapi_level_zero, kernel_bundle<State>> {
  struct type {
    ze_module_handle_t NativeHandle;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <bundle_state State>
struct BackendReturn<backend::ext_oneapi_level_zero, kernel_bundle<State>> {
  using type = std::vector<ze_module_handle_t>;
};

template <> struct BackendInput<backend::ext_oneapi_level_zero, kernel> {
  struct type {
    kernel_bundle<bundle_state::executable> KernelBundle;
    ze_kernel_handle_t NativeHandle;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, kernel> {
  using type = ze_kernel_handle_t;
};

template <> struct InteropFeatureSupportMap<backend::ext_oneapi_level_zero> {
  static constexpr bool MakePlatform = true;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = true;
  static constexpr bool MakeQueue = true;
  static constexpr bool MakeEvent = true;
  static constexpr bool MakeKernelBundle = true;
  static constexpr bool MakeKernel = true;
  static constexpr bool MakeBuffer = false;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
