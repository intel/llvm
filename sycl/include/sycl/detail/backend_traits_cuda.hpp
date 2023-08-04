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

#include <sycl/context.hpp>
#include <sycl/detail/backend_traits.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/queue.hpp>

typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUmod_st *CUmodule;

// As defined in the CUDA 10.1 header file. This requires CUDA version > 3.2
#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes.
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

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::ext_oneapi_cuda,
                    buffer<DataT, Dimensions, AllocatorT>> {
  using type = CUdeviceptr;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::ext_oneapi_cuda,
                     buffer<DataT, Dimensions, AllocatorT>> {
  using type = CUdeviceptr;
};

template <> struct BackendInput<backend::ext_oneapi_cuda, context> {
  using type = CUcontext;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, context> {
  using type = CUcontext;
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

} // namespace detail
} // namespace _V1
} // namespace sycl
