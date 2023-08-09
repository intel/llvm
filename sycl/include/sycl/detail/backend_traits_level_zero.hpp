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

#include <sycl/backend_types.hpp>                           // for backend
#include <sycl/context.hpp>                                 // for context
#include <sycl/detail/backend_traits.hpp>                   // for BackendI...
#include <sycl/device.hpp>                                  // for device
#include <sycl/event.hpp>                                   // for event
#include <sycl/ext/oneapi/backend/level_zero_ownership.hpp> // for ownership
#include <sycl/handler.hpp>                                 // for buffer
#include <sycl/image.hpp>                                   // for image
#include <sycl/kernel.hpp>                                  // for kernel
#include <sycl/kernel_bundle.hpp>                           // for kernel_b...
#include <sycl/kernel_bundle_enums.hpp>                     // for bundle_s...
#include <sycl/platform.hpp>                                // for platform
#include <sycl/property_list.hpp>                           // for property...
#include <sycl/queue.hpp>                                   // for queue
#include <sycl/range.hpp>                                   // for range

#include <variant> // for variant
#include <vector>  // for vector

typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;
typedef struct _ze_command_list_handle_t *ze_command_list_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_event_handle_t *ze_event_handle_t;
typedef struct _ze_image_handle_t *ze_image_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;

namespace sycl {
inline namespace _V1 {
namespace detail {

// Forward declarations
class device_impl;

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes.
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
  using type =
      std::variant<ze_command_queue_handle_t, ze_command_list_handle_t>;
};

template <> struct interop<backend::ext_oneapi_level_zero, platform> {
  using type = ze_driver_handle_t;
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
    property_list Properties;

    device Device;

    type(interop<backend::ext_oneapi_level_zero, queue>::type nativeHandle,
         device dev,
         ext::oneapi::level_zero::ownership ownership =
             ext::oneapi::level_zero::ownership::transfer,
         property_list properties = {})
        : NativeHandle(nativeHandle), Ownership(ownership),
          Properties(properties), Device(dev) {}
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

template <int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_level_zero,
                    image<Dimensions, AllocatorT>> {
  // LevelZero has no way of getting image description FROM a ZeImageHandle so
  // it must be provided.
  struct type {
    ze_image_handle_t ZeImageHandle;
    sycl::image_channel_order ChanOrder;
    sycl::image_channel_type ChanType;
    range<Dimensions> Range;
    ext::oneapi::level_zero::ownership Ownership{
        ext::oneapi::level_zero::ownership::transfer};
  };
};

template <int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_level_zero,
                     image<Dimensions, AllocatorT>> {
  using type = ze_image_handle_t;
};

template <> struct BackendReturn<backend::ext_oneapi_level_zero, queue> {
  using type =
      std::variant<ze_command_queue_handle_t, ze_command_list_handle_t>;
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
  static constexpr bool MakeImage = true;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
