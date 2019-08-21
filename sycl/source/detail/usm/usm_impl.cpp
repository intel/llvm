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
                   alloc Kind) {
  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;
  void *RetVal = nullptr;

  switch (Kind) {
  case alloc::host: {
    RetVal = Dispatch->hostMemAlloc(C, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::device:
  case alloc::shared:
  case alloc::unknown: {
    RetVal = nullptr;
    Error = PI_INVALID_VALUE;
    break;
  }
  }

  PI_CHECK(Error);

  return RetVal;
}

void *alignedAlloc(size_t Alignment, size_t Size, const context &Ctxt,
                   const device &Dev, alloc Kind) {
  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;
  pi_device Id;
  void *RetVal = nullptr;

  switch (Kind) {
  case alloc::device: {
    Id = detail::getSyclObjImpl(Dev)->getHandleRef();
    RetVal = Dispatch->deviceMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::shared: {
    Id = detail::getSyclObjImpl(Dev)->getHandleRef();
    RetVal = Dispatch->sharedMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
    break;
  }
  case alloc::host:
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
  pi_result Error = Dispatch->memFree(C, Ptr);

  PI_CHECK(Error);
}

} // namespace usm
} // namespace detail

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device);
}

void free(void *ptr, const context &Ctxt) {
  return detail::usm::free(ptr, Ctxt);
}

///
// Restricted USM
///
void *malloc_host(size_t Size, const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, alloc::host);
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, alloc::host);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared);
}

// single form

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind);
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind);
}

} // namespace sycl
} // namespace cl
