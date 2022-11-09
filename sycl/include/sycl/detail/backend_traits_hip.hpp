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

typedef int HIPdevice;
typedef struct ihipCtx_t *HIPcontext;
typedef struct ihipStream_t *HIPstream;
typedef struct ihipEvent_t *HIPevent;
typedef struct ihipModule_t *HIPmodule;
typedef void *HIPdeviceptr;

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

// TODO the interops for accessor is used in the already deprecated class
// interop_handler and can be removed after API cleanup.
template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::ext_oneapi_hip,
               accessor<DataT, Dimensions, AccessMode, access::target::device,
                        access::placeholder::false_t>> {
  using type = HIPdeviceptr;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<
    backend::ext_oneapi_hip,
    accessor<DataT, Dimensions, AccessMode, access::target::constant_buffer,
             access::placeholder::false_t>> {
  using type = HIPdeviceptr;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_hip,
                    buffer<DataT, Dimensions, AllocatorT>> {
  using type = HIPdeviceptr;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_hip,
                     buffer<DataT, Dimensions, AllocatorT>> {
  using type = HIPdeviceptr;
};

template <> struct BackendInput<backend::ext_oneapi_hip, context> {
  using type = HIPcontext;
};

template <> struct BackendReturn<backend::ext_oneapi_hip, context> {
  using type = HIPcontext;
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

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
