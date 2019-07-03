//==-------------- usm_impl.hpp - SYCL USM Utils ---------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/usm.hpp>

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

template <>
void* alignedAlloc<alloc::host>(size_t alignment,
                                size_t bytes,
                                const context* ctxt,
                                const device* dev) {
  cl_int error;
  cl_context c = detail::getSyclObjImpl(*ctxt)->getHandleRef();
  
  void* ret = clHostMemAllocINTEL(c,
                                  nullptr,
                                  size,
                                  alignment,
                                  &error);

  CHECK_OCL_CODE_THROW(error, "SYCL host allocation error");

  return ret;
}

template <>
void* alignedAlloc<alloc::device>(size_t alignment,
                                  size_t bytes,
                                  const context* ctxt,
                                  const device* dev) {
  cl_device_id id = dev->get();
  cl_int error;
  cl_context c = detail::getSyclObjImpl(*ctxt)->getHandleRef();
  
  void* ret = clDeviceMemAllocINTEL(c,
                                    id,
                                    nullptr,
                                    size,
                                    alignment,
                                    &error);
  
  CHECK_OCL_CODE_THROW(error, "SYCL device allocation error");
  
  return ret;
}

template <>
void* alignedAlloc<alloc::shared>(size_t alignment,
                                  size_t bytes,
                                  const context* ctxt,
                                  const device* dev) {
  cl_device_id id = dev->get();
  cl_int error;
  cl_context c = detail::getSyclObjImpl(*ctxt)->getHandleRef();
  
  void* ret = clSharedMemAllocINTEL(c,
                                    id,
                                    nullptr,
                                    size,
                                    alignment,
                                    &error);

  CHECK_OCL_CODE_THROW(error, "SYCL shared allocation error");

  return ret;
}
  
template <alloc Kind>
void* alignedAlloc(size_t alignment,
                   size_t bytes,
                   const context* ctxt,
                   const device* dev) {
  // Only use template specializations of this func
  return nullptr;
}

void free(void* ptr, const context* ctxt) {
  cl_int error;
  cl_context c = detail::getSyclObjImpl(*ctxt)->getHandleRef();

  error = clMemFreeINTEL(c, ptr);

  CHECK_OCL_CODE_THROW(error, "SYCL free deallocation error");
}

} // namespace usm
} // namespace detail
} // namespace sycl
} // namespace cl
