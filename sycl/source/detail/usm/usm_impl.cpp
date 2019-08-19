//==---------------- usm_impl.cpp - USM API Utils  -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/usm.hpp>

namespace cl {
namespace sycl {

using alloc = cl::sycl::usm::alloc;

namespace detail {
namespace usm {

void *alignedAlloc(size_t alignment, size_t size, const context *ctxt,
                   const device *dev, alloc kind) {
  cl_int error;
  cl_context c =
      pi::cast<cl_context>(detail::getSyclObjImpl(*ctxt)->getHandleRef());
  cl_device_id id;

  void *retVal = nullptr;

  switch (kind) {
  case alloc::host: {
    retVal = clHostMemAllocINTEL(c, nullptr, size, alignment, &error);
    break;
  }
  case alloc::device: {
    id = dev->get();
    retVal = clDeviceMemAllocINTEL(c, id, nullptr, size, alignment, &error);
    break;
  }
  case alloc::shared: {
    id = dev->get();
    retVal = clSharedMemAllocINTEL(c, id, nullptr, size, alignment, &error);
    break;
  }
  case alloc::unknown: {
    retVal = nullptr;
    error = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    break;
  }
  }

  CHECK_OCL_CODE_THROW(error, runtime_error);

  return retVal;
}

void free(void *ptr, const context *ctxt) {
  cl_int error;
  cl_context c =
      pi::cast<cl_context>(detail::getSyclObjImpl(*ctxt)->getHandleRef());

  error = clMemFreeINTEL(c, ptr);

  CHECK_OCL_CODE_THROW(error, runtime_error);
}

} // namespace usm
} // namespace detail

void *malloc_device(size_t size, const device &dev, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, &ctxt, &dev, alloc::device);
}

void *aligned_alloc_device(size_t alignment, size_t size, const device &dev,
                           const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, &ctxt, &dev, alloc::device);
}

void free(void *ptr, const context &ctxt) {
  return detail::usm::free(ptr, &ctxt);
}

///
// Restricted USM
///
void *malloc_host(size_t size, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, &ctxt, nullptr, alloc::host);
}

void *malloc_shared(size_t size, const device &dev, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, &ctxt, &dev, alloc::shared);
}

void *aligned_alloc_host(size_t alignment, size_t size, const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, &ctxt, nullptr,
                                   alloc::host);
}

void *aligned_alloc_shared(size_t alignment, size_t size, const device &dev,
                           const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, &ctxt, &dev, alloc::shared);
}

// single form

void *malloc(size_t size, const device &dev, const context &ctxt, alloc kind) {
  return detail::usm::alignedAlloc(0, size, &ctxt, &dev, kind);
}

void *aligned_alloc(size_t alignment, size_t size, const device &dev,
                    const context &ctxt, alloc kind) {
  return detail::usm::alignedAlloc(alignment, size, &ctxt, &dev, kind);
}

} // namespace sycl
} // namespace cl
