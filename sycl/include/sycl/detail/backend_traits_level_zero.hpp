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

#include <sycl/accessor.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/backend_traits.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/backend/level_zero_ownership.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/queue.hpp>

typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_event_handle_t *ze_event_handle_t;
typedef struct _ze_image_handle_t *ze_image_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// Forward declarations
class device_impl;

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
    ext::oneapi::level_zero::ownership Ownership;

    device Device;

    type(interop<backend::ext_oneapi_level_zero, queue>::type nativeHandle,
         device dev,
         ext::oneapi::level_zero::ownership ownership =
             ext::oneapi::level_zero::ownership::transfer)
        : NativeHandle(nativeHandle), Ownership(ownership), Device(dev) {}
  };
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_level_zero,
                    buffer<DataT, Dimensions, AllocatorT>> {
  struct type {
    void *NativeHandle;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_level_zero,
                     buffer<DataT, Dimensions, AllocatorT>> {
  using type = void *;
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
  static constexpr bool MakeBuffer = true;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
