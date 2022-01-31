//==---------------- usm_impl.cpp - USM API Utils  -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/usm.hpp>
#include <detail/queue_impl.hpp>

#include <cstdlib>
#include <memory>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.hpp"

#define XPTI_CREATE_TRACEPOINT(CL)                                             \
  std::unique_ptr<xpti::framework::tracepoint_t> _TP(nullptr);                 \
  if (xptiTraceEnabled()) {                                                    \
    xpti::payload_t Payload{CL.functionName(), CL.fileName(),                  \
                            static_cast<int>(CL.lineNumber()),                 \
                            static_cast<int>(CL.columnNumber()), nullptr};     \
    _TP = std::make_unique<xpti::framework::tracepoint_t>(&Payload);           \
  }                                                                            \
  (void)_TP;
#else
#define XPTI_CREATE_TRACEPOINT(CL)
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

using alloc = cl::sycl::usm::alloc;

namespace detail {
namespace usm {

void *alignedAllocHost(size_t Alignment, size_t Size, const context &Ctxt,
                       alloc Kind, const detail::code_location &CL) {
  XPTI_CREATE_TRACEPOINT(CL);
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;
  if (Ctxt.is_host()) {
    if (!Alignment) {
      // worst case default
      Alignment = 128;
    }

    aligned_allocator<char> Alloc(Alignment);
    try {
      RetVal = Alloc.allocate(Size);
    } catch (const std::bad_alloc &) {
      // Conform with Specification behavior
      RetVal = nullptr;
    }
  } else {
    std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
    pi_context C = CtxImpl->getHandleRef();
    const detail::plugin &Plugin = CtxImpl->getPlugin();
    pi_result Error;

    switch (Kind) {
    case alloc::host: {
      Error = Plugin.call_nocheck<PiApiKind::piextUSMHostAlloc>(
          &RetVal, C, nullptr, Size, Alignment);
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

    // Error is for debugging purposes.
    // The spec wants a nullptr returned, not an exception.
    if (Error != PI_SUCCESS)
      return nullptr;
  }
  return RetVal;
}

void *alignedAlloc(size_t Alignment, size_t Size, const context &Ctxt,
                   const device &Dev, alloc Kind,
                   const detail::code_location &CL) {
  XPTI_CREATE_TRACEPOINT(CL);
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;
  if (Ctxt.is_host()) {
    if (Kind == alloc::unknown) {
      RetVal = nullptr;
    } else {
      if (!Alignment) {
        // worst case default
        Alignment = 128;
      }

      aligned_allocator<char> Alloc(Alignment);
      try {
        RetVal = Alloc.allocate(Size);
      } catch (const std::bad_alloc &) {
        // Conform with Specification behavior
        RetVal = nullptr;
      }
    }
  } else {
    std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
    pi_context C = CtxImpl->getHandleRef();
    const detail::plugin &Plugin = CtxImpl->getPlugin();
    pi_result Error;
    pi_device Id;

    switch (Kind) {
    case alloc::device: {
      Id = detail::getSyclObjImpl(Dev)->getHandleRef();
      Error = Plugin.call_nocheck<PiApiKind::piextUSMDeviceAlloc>(
          &RetVal, C, Id, nullptr, Size, Alignment);
      break;
    }
    case alloc::shared: {
      Id = detail::getSyclObjImpl(Dev)->getHandleRef();
      Error = Plugin.call_nocheck<PiApiKind::piextUSMSharedAlloc>(
          &RetVal, C, Id, nullptr, Size, Alignment);
      break;
    }
    case alloc::host:
    case alloc::unknown: {
      RetVal = nullptr;
      Error = PI_INVALID_VALUE;
      break;
    }
    }

    // Error is for debugging purposes.
    // The spec wants a nullptr returned, not an exception.
    if (Error != PI_SUCCESS)
      return nullptr;
  }
  return RetVal;
}

void free(void *Ptr, const context &Ctxt, const detail::code_location &CL) {
  XPTI_CREATE_TRACEPOINT(CL);
  if (Ptr == nullptr)
    return;
  if (Ctxt.is_host()) {
    // need to use alignedFree here for Windows
    detail::OSUtil::alignedFree(Ptr);
  } else {
    std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
    pi_context C = CtxImpl->getHandleRef();
    const detail::plugin &Plugin = CtxImpl->getPlugin();
    Plugin.call<PiApiKind::piextUSMFree>(C, Ptr);
  }
}

// For ABI compatibility
// TODO remove once ABI breakages are allowed.
__SYCL_EXPORT void *alignedAllocHost(size_t Alignment, size_t Size,
                                     const context &Ctxt, alloc Kind) {
  return alignedAllocHost(Alignment, Size, Ctxt, Kind, detail::code_location{});
}

__SYCL_EXPORT void free(void *Ptr, const context &Ctxt) {
  detail::usm::free(Ptr, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void *alignedAlloc(size_t Alignment, size_t Size,
                                 const context &Ctxt, const device &Dev,
                                 alloc Kind) {
  return alignedAlloc(Alignment, Size, Ctxt, Dev, Kind,
                      detail::code_location{});
}

} // namespace usm
} // namespace detail

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt,
                    const detail::code_location CL) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device, CL);
}

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt,
                    const property_list &, const detail::code_location CL) {
  return malloc_device(Size, Dev, Ctxt, CL);
}

void *malloc_device(size_t Size, const queue &Q,
                    const detail::code_location CL) {
  return malloc_device(Size, Q.get_device(), Q.get_context(), CL);
}

void *malloc_device(size_t Size, const queue &Q, const property_list &PropList,
                    const detail::code_location CL) {
  return malloc_device(Size, Q.get_device(), Q.get_context(), PropList, CL);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt,
                           const detail::code_location CL) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device,
                                   CL);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt, const property_list &,
                           const detail::code_location CL) {
  return aligned_alloc_device(Alignment, Size, Dev, Ctxt, CL);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const queue &Q,
                           const detail::code_location CL) {
  return aligned_alloc_device(Alignment, Size, Q.get_device(), Q.get_context(),
                              CL);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const queue &Q,
                           const property_list &PropList,
                           const detail::code_location CL) {
  return aligned_alloc_device(Alignment, Size, Q.get_device(), Q.get_context(),
                              PropList, CL);
}

void free(void *ptr, const context &Ctxt, const detail::code_location CL) {
  return detail::usm::free(ptr, Ctxt, CL);
}

void free(void *ptr, const queue &Q, const detail::code_location CL) {
  return free(ptr, Q.get_context(), CL);
}

///
// Restricted USM
///
void *malloc_host(size_t Size, const context &Ctxt,
                  const detail::code_location CL) {
  return detail::usm::alignedAllocHost(0, Size, Ctxt, alloc::host, CL);
}

void *malloc_host(size_t Size, const context &Ctxt, const property_list &,
                  const detail::code_location CL) {
  return malloc_host(Size, Ctxt, CL);
}

void *malloc_host(size_t Size, const queue &Q, const detail::code_location CL) {
  return malloc_host(Size, Q.get_context(), CL);
}

void *malloc_host(size_t Size, const queue &Q, const property_list &PropList,
                  const detail::code_location CL) {
  return malloc_host(Size, Q.get_context(), PropList, CL);
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt,
                    const detail::code_location CL) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared, CL);
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt,
                    const property_list &, const detail::code_location CL) {
  return malloc_shared(Size, Dev, Ctxt, CL);
}

void *malloc_shared(size_t Size, const queue &Q,
                    const detail::code_location CL) {
  return malloc_shared(Size, Q.get_device(), Q.get_context(), CL);
}

void *malloc_shared(size_t Size, const queue &Q, const property_list &PropList,
                    const detail::code_location CL) {
  return malloc_shared(Size, Q.get_device(), Q.get_context(), PropList, CL);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt,
                         const detail::code_location CL) {
  return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, alloc::host, CL);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt,
                         const property_list &,
                         const detail::code_location CL) {
  return aligned_alloc_host(Alignment, Size, Ctxt, CL);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const queue &Q,
                         const detail::code_location CL) {
  return aligned_alloc_host(Alignment, Size, Q.get_context(), CL);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const queue &Q,
                         const property_list &PropList,
                         const detail::code_location CL) {
  return aligned_alloc_host(Alignment, Size, Q.get_context(), PropList, CL);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt,
                           const detail::code_location CL) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared,
                                   CL);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt, const property_list &,
                           const detail::code_location CL) {
  return aligned_alloc_shared(Alignment, Size, Dev, Ctxt, CL);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const queue &Q,
                           const detail::code_location CL) {
  return aligned_alloc_shared(Alignment, Size, Q.get_device(), Q.get_context(),
                              CL);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const queue &Q,
                           const property_list &PropList,
                           const detail::code_location CL) {
  return aligned_alloc_shared(Alignment, Size, Q.get_device(), Q.get_context(),
                              PropList, CL);
}

// single form

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind,
             const detail::code_location CL) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(0, Size, Ctxt, Kind, CL);
  } else {
    RetVal = detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind, CL);
  }

  return RetVal;
}

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind,
             const property_list &, const detail::code_location CL) {
  return malloc(Size, Dev, Ctxt, Kind, CL);
}

void *malloc(size_t Size, const queue &Q, alloc Kind,
             const detail::code_location CL) {
  return malloc(Size, Q.get_device(), Q.get_context(), Kind, CL);
}

void *malloc(size_t Size, const queue &Q, alloc Kind,
             const property_list &PropList, const detail::code_location CL) {
  return malloc(Size, Q.get_device(), Q.get_context(), Kind, PropList, CL);
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind,
                    const detail::code_location CL) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(Alignment, Size, Ctxt, Kind, CL);
  } else {
    RetVal = detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind, CL);
  }

  return RetVal;
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind, const property_list &,
                    const detail::code_location CL) {
  return aligned_alloc(Alignment, Size, Dev, Ctxt, Kind, CL);
}

void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q, alloc Kind,
                    const detail::code_location CL) {
  return aligned_alloc(Alignment, Size, Q.get_device(), Q.get_context(), Kind,
                       CL);
}

void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q, alloc Kind,
                    const property_list &PropList,
                    const detail::code_location CL) {
  return aligned_alloc(Alignment, Size, Q.get_device(), Q.get_context(), Kind,
                       PropList, CL);
}

// Pointer queries
/// Query the allocation type from a USM pointer
/// Returns alloc::host for all pointers in a host context.
///
/// \param Ptr is the USM pointer to query
/// \param Ctxt is the sycl context the ptr was allocated in
alloc get_pointer_type(const void *Ptr, const context &Ctxt) {
  if (!Ptr)
    return alloc::unknown;

  // Everything on a host device is just system malloc so call it host
  if (Ctxt.is_host())
    return alloc::host;

  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  pi_context PICtx = CtxImpl->getHandleRef();
  pi_usm_type AllocTy;

  // query type using PI function
  const detail::plugin &Plugin = CtxImpl->getPlugin();
  RT::PiResult Err =
      Plugin.call_nocheck<detail::PiApiKind::piextUSMGetMemAllocInfo>(
          PICtx, Ptr, PI_MEM_ALLOC_TYPE, sizeof(pi_usm_type), &AllocTy,
          nullptr);

  // PI_INVALID_VALUE means USM doesn't know about this ptr
  if (Err == PI_INVALID_VALUE)
    return alloc::unknown;
  // otherwise PI_SUCCESS is expected
  if (Err != PI_SUCCESS) {
    Plugin.reportPiError(Err, "get_pointer_type()");
  }

  alloc ResultAlloc;
  switch (AllocTy) {
  case PI_MEM_TYPE_HOST:
    ResultAlloc = alloc::host;
    break;
  case PI_MEM_TYPE_DEVICE:
    ResultAlloc = alloc::device;
    break;
  case PI_MEM_TYPE_SHARED:
    ResultAlloc = alloc::shared;
    break;
  default:
    ResultAlloc = alloc::unknown;
    break;
  }

  return ResultAlloc;
}

/// Queries the device against which the pointer was allocated
///
/// \param Ptr is the USM pointer to query
/// \param Ctxt is the sycl context the ptr was allocated in
device get_pointer_device(const void *Ptr, const context &Ctxt) {
  // Check if ptr is a valid USM pointer
  if (get_pointer_type(Ptr, Ctxt) == alloc::unknown)
    throw runtime_error("Ptr not a valid USM allocation!", PI_INVALID_VALUE);

  // Just return the host device in the host context
  if (Ctxt.is_host())
    return Ctxt.get_devices()[0];

  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);

  // Check if ptr is a host allocation
  if (get_pointer_type(Ptr, Ctxt) == alloc::host) {
    auto Devs = CtxImpl->getDevices();
    if (Devs.size() == 0)
      throw runtime_error("No devices in passed context!", PI_INVALID_VALUE);

    // Just return the first device in the context
    return Devs[0];
  }

  pi_context PICtx = CtxImpl->getHandleRef();
  pi_device DeviceId;

  // query device using PI function
  const detail::plugin &Plugin = CtxImpl->getPlugin();
  Plugin.call<detail::PiApiKind::piextUSMGetMemAllocInfo>(
      PICtx, Ptr, PI_MEM_ALLOC_DEVICE, sizeof(pi_device), &DeviceId, nullptr);

  for (const device &Dev : CtxImpl->getDevices()) {
    // Try to find the real sycl device used in the context
    if (detail::getSyclObjImpl(Dev)->getHandleRef() == DeviceId)
      return Dev;
  }

  throw runtime_error("Cannot find device associated with USM allocation!",
                      PI_INVALID_OPERATION);
}

// For ABI compatibility

__SYCL_EXPORT void *malloc_device(size_t Size, const device &Dev,
                                  const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device,
                                   detail::code_location{});
}

__SYCL_EXPORT void *malloc_device(size_t Size, const device &Dev,
                                  const context &Ctxt, const property_list &) {
  return malloc_device(Size, Dev, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void *malloc_device(size_t Size, const queue &Q) {
  return malloc_device(Size, Q.get_device(), Q.get_context(),
                       detail::code_location{});
}

__SYCL_EXPORT void *malloc_device(size_t Size, const queue &Q,
                                  const property_list &PropList) {
  return malloc_device(Size, Q.get_device(), Q.get_context(), PropList,
                       detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_device(size_t Alignment, size_t Size,
                                         const device &Dev,
                                         const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device,
                                   detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_device(size_t Alignment, size_t Size,
                                         const device &Dev, const context &Ctxt,
                                         const property_list &) {
  return aligned_alloc_device(Alignment, Size, Dev, Ctxt,
                              detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_device(size_t Alignment, size_t Size,
                                         const queue &Q) {
  return aligned_alloc_device(Alignment, Size, Q.get_device(), Q.get_context(),
                              detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_device(size_t Alignment, size_t Size,
                                         const queue &Q,
                                         const property_list &PropList) {
  return aligned_alloc_device(Alignment, Size, Q.get_device(), Q.get_context(),
                              PropList, detail::code_location{});
}

__SYCL_EXPORT void free(void *ptr, const context &Ctxt) {
  return detail::usm::free(ptr, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void free(void *ptr, const queue &Q) {
  return free(ptr, Q.get_context(), detail::code_location{});
}

///
// Restricted USM
///
__SYCL_EXPORT void *malloc_host(size_t Size, const context &Ctxt) {
  return detail::usm::alignedAllocHost(0, Size, Ctxt, alloc::host,
                                       detail::code_location{});
}

__SYCL_EXPORT void *malloc_host(size_t Size, const context &Ctxt,
                                const property_list &) {
  return malloc_host(Size, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void *malloc_host(size_t Size, const queue &Q) {
  return malloc_host(Size, Q.get_context(), detail::code_location{});
}

__SYCL_EXPORT void *malloc_host(size_t Size, const queue &Q,
                                const property_list &PropList) {
  return malloc_host(Size, Q.get_context(), PropList, detail::code_location{});
}

__SYCL_EXPORT void *malloc_shared(size_t Size, const device &Dev,
                                  const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared,
                                   detail::code_location{});
}

__SYCL_EXPORT void *malloc_shared(size_t Size, const device &Dev,
                                  const context &Ctxt, const property_list &) {
  return malloc_shared(Size, Dev, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void *malloc_shared(size_t Size, const queue &Q) {
  return malloc_shared(Size, Q.get_device(), Q.get_context(),
                       detail::code_location{});
}

__SYCL_EXPORT void *malloc_shared(size_t Size, const queue &Q,
                                  const property_list &PropList) {
  return malloc_shared(Size, Q.get_device(), Q.get_context(), PropList,
                       detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_host(size_t Alignment, size_t Size,
                                       const context &Ctxt) {
  return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, alloc::host,
                                       detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_host(size_t Alignment, size_t Size,
                                       const context &Ctxt,
                                       const property_list &) {
  return aligned_alloc_host(Alignment, Size, Ctxt, detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_host(size_t Alignment, size_t Size,
                                       const queue &Q) {
  return aligned_alloc_host(Alignment, Size, Q.get_context(),
                            detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_host(size_t Alignment, size_t Size,
                                       const queue &Q,
                                       const property_list &PropList) {
  return aligned_alloc_host(Alignment, Size, Q.get_context(), PropList,
                            detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_shared(size_t Alignment, size_t Size,
                                         const device &Dev,
                                         const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared,
                                   detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_shared(size_t Alignment, size_t Size,
                                         const device &Dev, const context &Ctxt,
                                         const property_list &) {
  return aligned_alloc_shared(Alignment, Size, Dev, Ctxt,
                              detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_shared(size_t Alignment, size_t Size,
                                         const queue &Q) {
  return aligned_alloc_shared(Alignment, Size, Q.get_device(), Q.get_context(),
                              detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc_shared(size_t Alignment, size_t Size,
                                         const queue &Q,
                                         const property_list &PropList) {
  return aligned_alloc_shared(Alignment, Size, Q.get_device(), Q.get_context(),
                              PropList, detail::code_location{});
}

// single form

__SYCL_EXPORT void *malloc(size_t Size, const device &Dev, const context &Ctxt,
                           alloc Kind) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(0, Size, Ctxt, Kind,
                                           detail::code_location{});
  } else {
    RetVal = detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind,
                                       detail::code_location{});
  }

  return RetVal;
}

__SYCL_EXPORT void *malloc(size_t Size, const device &Dev, const context &Ctxt,
                           alloc Kind, const property_list &) {
  return malloc(Size, Dev, Ctxt, Kind, detail::code_location{});
}

__SYCL_EXPORT void *malloc(size_t Size, const queue &Q, alloc Kind) {
  return malloc(Size, Q.get_device(), Q.get_context(), Kind,
                detail::code_location{});
}

__SYCL_EXPORT void *malloc(size_t Size, const queue &Q, alloc Kind,
                           const property_list &PropList) {
  return malloc(Size, Q.get_device(), Q.get_context(), Kind, PropList,
                detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc(size_t Alignment, size_t Size,
                                  const device &Dev, const context &Ctxt,
                                  alloc Kind) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(Alignment, Size, Ctxt, Kind,
                                           detail::code_location{});
  } else {
    RetVal = detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind,
                                       detail::code_location{});
  }

  return RetVal;
}

__SYCL_EXPORT void *aligned_alloc(size_t Alignment, size_t Size,
                                  const device &Dev, const context &Ctxt,
                                  alloc Kind, const property_list &) {
  return aligned_alloc(Alignment, Size, Dev, Ctxt, Kind,
                       detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q,
                                  alloc Kind) {
  return aligned_alloc(Alignment, Size, Q.get_device(), Q.get_context(), Kind,
                       detail::code_location{});
}

__SYCL_EXPORT void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q,
                                  alloc Kind, const property_list &PropList) {
  return aligned_alloc(Alignment, Size, Q.get_device(), Q.get_context(), Kind,
                       PropList, detail::code_location{});
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
