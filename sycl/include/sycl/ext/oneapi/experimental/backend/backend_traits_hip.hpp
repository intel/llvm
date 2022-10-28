//===------- backend_traits_hip.hpp - Backend traits for HIP ---*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the specializations of the sycl::detail::interop,
// sycl::detail::BackendInput and sycl::detail::BackendReturn class templates
// for the HIP backend but there is no sycl::detail::InteropFeatureSupportMap
// specialization for the HIP backend.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/backend_traits.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/queue.hpp>

#include <vector>

typedef int HIPdevice;
typedef struct HIPctx_st *HIPcontext;
typedef struct HIPstream_st *HIPstream;
typedef struct HIPevent_st *HIPevent;
typedef struct HIPmod_st *HIPmodule;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes. The interop<backend, queue> specialization
// is also used in the get_queue() method of the deprecated class
// interop_handler and also can be removed after API cleanup.
template <> struct interop<backend::ext_oneapi_hip, context> {
  using type = HIPcontext;
};

template <> struct interop<backend::ext_oneapi_hip, device> {
  using type = HIPdevice;
};

template <> struct interop<backend::ext_oneapi_hip, event> {
  using type = HIPevent;
};

template <> struct interop<backend::ext_oneapi_hip, queue> {
  using type = HIPstream;
};

template <> struct interop<backend::ext_oneapi_hip, platform> {
  using type = std::vector<HIPdevice>;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_hip,
                    buffer<DataT, Dimensions, AllocatorT>> {
  using type = DataT *;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_hip,
                     buffer<DataT, Dimensions, AllocatorT>> {
  using type = DataT *;
};

template <> struct BackendInput<backend::ext_oneapi_hip, context> {
  using type = HIPcontext;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, context> {
  using type = std::vector<HIPcontext>;
};

template <> struct BackendInput<backend::ext_oneapi_hip, device> {
  using type = HIPdevice;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, device> {
  using type = HIPdevice;
};

template <> struct BackendInput<backend::ext_oneapi_hip, event> {
  using type = HIPevent;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, event> {
  using type = HIPevent;
};

template <> struct BackendInput<backend::ext_oneapi_hip, queue> {
  using type = HIPstream;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, queue> {
  using type = HIPstream;
};

template <> struct BackendInput<backend::ext_oneapi_hip, platform> {
  using type = std::vector<HIPdevice>;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, platform> {
  using type = std::vector<HIPdevice>;
};

template <> struct InteropFeatureSupportMap<backend::ext_oneapi_hip> {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = true;
  static constexpr bool MakeQueue = true;
  static constexpr bool MakeEvent = true;
  static constexpr bool MakeBuffer = false;
  static constexpr bool MakeKernel = false;
  static constexpr bool MakeKernelBundle = false;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
