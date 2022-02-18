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

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/queue.hpp>

#include <vector>

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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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

#ifdef __SYCL_INTERNAL_API
template <> struct interop<backend::ext_oneapi_cuda, program> {
  using type = CUmodule;
};
#endif

// TODO the interops for accessor is used in the already deprecated class
// interop_handler and can be removed after API cleanup.
template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::ext_oneapi_cuda,
               accessor<DataT, Dimensions, AccessMode, access::target::device,
                        access::placeholder::false_t>> {
  using type = CUdeviceptr;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<
    backend::ext_oneapi_cuda,
    accessor<DataT, Dimensions, AccessMode, access::target::constant_buffer,
             access::placeholder::false_t>> {
  using type = CUdeviceptr;
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

#ifdef __SYCL_INTERNAL_API
template <> struct BackendInput<backend::ext_oneapi_cuda, program> {
  using type = CUmodule;
};

template <> struct BackendReturn<backend::ext_oneapi_cuda, program> {
  using type = CUmodule;
};
#endif

template <> struct InteropFeatureSupportMap<backend::ext_oneapi_cuda> {
  static constexpr bool MakePlatform = true;
  static constexpr bool MakeDevice = true;
  static constexpr bool MakeContext = true;
  static constexpr bool MakeQueue = true;
  static constexpr bool MakeEvent = true;
  static constexpr bool MakeBuffer = true;
  static constexpr bool MakeKernel = true;
  static constexpr bool MakeKernelBundle = true;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
