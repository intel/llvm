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

void *alignedAlloc(size_t Alignment, size_t Size, const context &Ctxt,
                   const device *Dev, alloc Kind) {
  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;
  pi_device Id;
  void *RetVal = nullptr;

  switch (Kind) {
  case alloc::host: {
    RetVal = Dispatch->hostMemAlloc(C, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::device: {
    Id = detail::getSyclObjImpl(*Dev)->getHandleRef();
    RetVal = Dispatch->deviceMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::shared: {
    Id = detail::getSyclObjImpl(*Dev)->getHandleRef();
    RetVal = Dispatch->sharedMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::unknown: {
    RetVal = nullptr;
    Error = PI_INVALID_VALUE;
    break;
  }
  }

  PI_CHECK(Error);

  return RetVal;
}

void free(void *Ptr, const context &Ctxt) {
  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;

  Error = Dispatch->memFree(C, Ptr);

  PI_CHECK(Error);
}

} // namespace usm
} // namespace detail

void *malloc_device(size_t size, const device &dev, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, ctxt, &dev, alloc::device);
}

void *aligned_alloc_device(size_t alignment, size_t size, const device &dev,
                           const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, ctxt, &dev, alloc::device);
}

void free(void *ptr, const context &ctxt) {
  return detail::usm::free(ptr, ctxt);
}

///
// Restricted USM
///
void *malloc_host(size_t size, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, ctxt, nullptr, alloc::host);
}

void *malloc_shared(size_t size, const device &dev, const context &ctxt) {
  return detail::usm::alignedAlloc(0, size, ctxt, &dev, alloc::shared);
}

void *aligned_alloc_host(size_t alignment, size_t size, const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, ctxt, nullptr, alloc::host);
}

void *aligned_alloc_shared(size_t alignment, size_t size, const device &dev,
                           const context &ctxt) {
  return detail::usm::alignedAlloc(alignment, size, ctxt, &dev, alloc::shared);
}

// single form

void *malloc(size_t size, const device &dev, const context &ctxt, alloc kind) {
  return detail::usm::alignedAlloc(0, size, ctxt, &dev, kind);
}

void *aligned_alloc(size_t alignment, size_t size, const device &dev,
                    const context &ctxt, alloc kind) {
  return detail::usm::alignedAlloc(alignment, size, ctxt, &dev, kind);
}

} // namespace sycl
} // namespace cl
