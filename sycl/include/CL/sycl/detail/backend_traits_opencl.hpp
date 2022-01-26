//===---- backend_traits_opencl.hpp - Backend traits for OpenCL --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the specializations of the sycl::detail::interop,
// sycl::detail::BackendInput, sycl::detail::BackendReturn and
// sycl::detail::InteropFeatureSupportMap class templates for the OpenCL
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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// TODO the interops for context, device, event, platform and program
// may be removed after removing the deprecated 'get_native()' methods
// from the corresponding classes. The interop<backend, queue> specialization
// is also used in the get_queue() method of the deprecated class
// interop_handler and also can be removed after API cleanup.
template <> struct interop<backend::opencl, context> {
  using type = cl_context;
};

template <> struct interop<backend::opencl, device> {
  using type = cl_device_id;
};

template <> struct interop<backend::opencl, queue> {
  using type = cl_command_queue;
};

template <> struct interop<backend::opencl, platform> {
  using type = cl_platform_id;
};

#ifdef __SYCL_INTERNAL_API
template <> struct interop<backend::opencl, program> {
  using type = cl_program;
};
#endif

// TODO the interops for accessor is used in the already deprecated class
// interop_handler and can be removed after API cleanup.
template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl,
               accessor<DataT, Dimensions, AccessMode, access::target::device,
                        access::placeholder::false_t>> {
  using type = cl_mem;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl, accessor<DataT, Dimensions, AccessMode,
                                         access::target::constant_buffer,
                                         access::placeholder::false_t>> {
  using type = cl_mem;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::opencl,
               accessor<DataT, Dimensions, AccessMode, access::target::image,
                        access::placeholder::false_t>> {
  using type = cl_mem;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendInput<backend::opencl, buffer<DataT, Dimensions, AllocatorT>> {
  using type = cl_mem;
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BackendReturn<backend::opencl, buffer<DataT, Dimensions, AllocatorT>> {
  using type = cl_mem;
};

template <> struct BackendInput<backend::opencl, context> {
  using type = cl_context;
};

template <> struct BackendReturn<backend::opencl, context> {
  using type = cl_context;
};

template <> struct BackendInput<backend::opencl, device> {
  using type = cl_device_id;
};

template <> struct BackendReturn<backend::opencl, device> {
  using type = cl_device_id;
};

#ifdef SYCL2020_CONFORMANT_APIS
template <> struct interop<backend::opencl, event> {
  using type = std::vector<cl_event>;
  using value_type = cl_event;
};
template <> struct BackendInput<backend::opencl, event> {
  using type = std::vector<cl_event>;
  using value_type = cl_event;
};
template <> struct BackendReturn<backend::opencl, event> {
  using type = std::vector<cl_event>;
  using value_type = cl_event;
};
#else
template <> struct interop<backend::opencl, event> { using type = cl_event; };
template <> struct BackendInput<backend::opencl, event> {
  using type = cl_event;
};
template <> struct BackendReturn<backend::opencl, event> {
  using type = cl_event;
};
#endif

template <> struct BackendInput<backend::opencl, queue> {
  using type = cl_command_queue;
};

template <> struct BackendReturn<backend::opencl, queue> {
  using type = cl_command_queue;
};

template <> struct BackendInput<backend::opencl, platform> {
  using type = cl_platform_id;
};

template <> struct BackendReturn<backend::opencl, platform> {
  using type = cl_platform_id;
};

#ifdef __SYCL_INTERNAL_API
template <> struct BackendInput<backend::opencl, program> {
  using type = cl_program;
};

template <> struct BackendReturn<backend::opencl, program> {
  using type = cl_program;
};
#endif

template <bundle_state State>
struct BackendInput<backend::opencl, kernel_bundle<State>> {
  using type = cl_program;
};

template <bundle_state State>
struct BackendReturn<backend::opencl, kernel_bundle<State>> {
  using type = std::vector<cl_program>;
};

template <> struct BackendInput<backend::opencl, kernel> {
  using type = cl_kernel;
};

template <> struct BackendReturn<backend::opencl, kernel> {
  using type = cl_kernel;
};

template <> struct InteropFeatureSupportMap<backend::opencl> {
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
