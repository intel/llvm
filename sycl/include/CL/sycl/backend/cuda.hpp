
//==---------------- cuda.hpp - SYCL CUDA backend --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/queue.hpp>

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

template <> struct interop<backend::cuda, device> { using type = CUdevice; };

template <> struct interop<backend::cuda, context> { using type = CUcontext; };

template <> struct interop<backend::cuda, queue> { using type = CUstream; };

template <> struct interop<backend::cuda, event> { using type = CUevent; };

template <> struct interop<backend::cuda, program> { using type = CUmodule; };

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::cuda, accessor<DataT, Dimensions, AccessMode,
                                       access::target::global_buffer,
                                       access::placeholder::false_t>> {
  using type = CUdeviceptr;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::cuda, accessor<DataT, Dimensions, AccessMode,
                                       access::target::constant_buffer,
                                       access::placeholder::false_t>> {
  using type = CUdeviceptr;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct interop<backend::cuda, buffer<DataT, Dimensions, AllocatorT>> {
  using type = CUdeviceptr;
};

} // namespace sycl
} // namespace cl
