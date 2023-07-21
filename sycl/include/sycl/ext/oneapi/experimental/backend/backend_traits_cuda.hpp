//===------- backend_traits_cuda.hpp - Backend traits for CUDA ---*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the specializations of the sycl::detail::interop,
// sycl::detail::BackendInput and sycl::detail::BackendReturn class templates
// for the CUDA backend but there is no sycl::detail::InteropFeatureSupportMap
// specialization for the CUDA backend.
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

typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUmod_st *CUmodule;

namespace sycl {
inline namespace _V1 {
namespace detail {

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes. The interop<backend, queue> specialization
// is also used in the get_queue() method of the deprecated class
// interop_handler and also can be removed after API cleanup.
template <> struct interop<backend::ext_oneapi_cuda, context> {
  using type = CUcontext;
};

template <> struct interop<backend::ext_oneapi_cuda, device> {
  using type = CUdevice;
};

template <> struct interop<backend::ext_oneapi_cuda, event> {
  using type = CUevent;
};

template <> struct interop<backend::ext_oneapi_cuda, queue> {
  using type = CUstream;
};

template <> struct interop<backend::ext_oneapi_cuda, platform> {
  using type = std::vector<CUdevice>;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_cuda,
                    buffer<DataT, Dimensions, AllocatorT>> {
  using type = DataT *;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_cuda,
                     buffer<DataT, Dimensions, AllocatorT>> {
  using type = DataT *;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, context> {
  using type = CUcontext;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, context> {
  using type = std::vector<CUcontext>;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, device> {
  using type = CUdevice;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, device> {
  using type = CUdevice;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, event> {
  using type = CUevent;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, event> {
  using type = CUevent;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, queue> {
  using type = CUstream;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, queue> {
  using type = CUstream;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, platform> {
  using type = std::vector<CUdevice>;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, platform> {
  using type = std::vector<CUdevice>;
};

template <> struct InteropFeatureSupportMap<backend::ext_oneapi_cuda> {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = false;
  static constexpr bool MakeQueue = true;
  static constexpr bool MakeEvent = true;
  static constexpr bool MakeBuffer = false;
  static constexpr bool MakeKernel = false;
  static constexpr bool MakeKernelBundle = false;
  static constexpr bool MakeImage = false;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
