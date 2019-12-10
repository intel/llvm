//==---------------- usm_impl.cpp - USM API Utils  -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/cl_usm_ext.h>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/usm.hpp>

#include <cstdlib>
#include <tuple>

namespace cl {
namespace sycl {

using alloc = cl::sycl::usm::alloc;

namespace detail {
namespace usm {

void *alignedAllocHost(size_t Alignment, size_t Size, const context &Ctxt,
                       alloc Kind) {
  void *RetVal = nullptr;
  if (Ctxt.is_host()) {
    if (!Alignment) {
      // Worst case default
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
    std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
    pi_context C = CtxImpl->getHandleRef();
    pi_result Error;

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

    // Error is for debugging purposes.
    // The spec wants a nullptr returned, not an exception.
    if (Error != PI_SUCCESS)
      return nullptr;
  }
  return RetVal;
}

void *alignedAlloc(size_t Alignment, size_t Size, const context &Ctxt,
                   const device &Dev, alloc Kind) {
  void *RetVal = nullptr;
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
    std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
    pi_context C = CtxImpl->getHandleRef();
    pi_result Error;
    pi_device Id;

    switch (Kind) {
    case alloc::device: {
      Id = detail::getSyclObjImpl(Dev)->getHandleRef();
      RetVal =
          Dispatch->deviceMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
      break;
    }
    case alloc::shared: {
      Id = detail::getSyclObjImpl(Dev)->getHandleRef();
      RetVal =
          Dispatch->sharedMemAlloc(C, Id, nullptr, Size, Alignment, &Error);
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

void free(void *Ptr, const context &Ctxt) {
  if (Ctxt.is_host()) {
    // need to use alignedFree here for Windows
    detail::OSUtil::alignedFree(Ptr);
  } else {
    std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
    std::shared_ptr<USMDispatcher> Dispatch = CtxImpl->getUSMDispatch();
    pi_context C = CtxImpl->getHandleRef();
    pi_result Error = Dispatch->memFree(C, Ptr);

    RT::checkPiResult(Error);
  }
}


} // namespace usm
} // namespace detail

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device);
}

void *malloc_device(size_t Size, const queue &Q) {
  return malloc_device(Size, Q.get_device(), Q.get_context());
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const queue &Q) {
  return aligned_alloc_device(Alignment, Size, Q.get_device(), Q.get_context());
}

void free(void *ptr, const context &Ctxt) {
  return detail::usm::free(ptr, Ctxt);
}

void free(void *ptr, const queue &Q) { return free(ptr, Q.get_context()); }

///
// Restricted USM
///
void *malloc_host(size_t Size, const context &Ctxt) {
  return detail::usm::alignedAllocHost(0, Size, Ctxt, alloc::host);
}

void *malloc_host(size_t Size, const queue &Q) {
  return malloc_host(Size, Q.get_context());
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared);
}

void *malloc_shared(size_t Size, const queue &Q) {
  return malloc_shared(Size, Q.get_device(), Q.get_context());
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt) {
  return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, alloc::host);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const queue &Q) {
  return aligned_alloc_host(Alignment, Size, Q.get_context());
}  

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const queue &Q) {
  return aligned_alloc_shared(Alignment, Size, Q.get_device(), Q.get_context());
}

// single form

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(0, Size, Ctxt, Kind);
  } else {
    RetVal = detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind);
  }

  return RetVal;
}

void *malloc(size_t Size, const queue &Q, alloc Kind) {
  return malloc(Size, Q.get_device(), Q.get_context(), Kind);
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind) {
  void *RetVal = nullptr;

  if (Kind == alloc::host) {
    RetVal = detail::usm::alignedAllocHost(Alignment, Size, Ctxt, Kind);
  } else {
    RetVal = detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind);
  }

  return RetVal;
}

void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q, alloc Kind) {
  return aligned_alloc(Alignment, Size, Q.get_device(), Q.get_context(), Kind);
}

// get_pointer_info
std::tuple<alloc, device> get_pointer_info(const void *Ptr,
                                           const context &Ctxt) {
  if (Ctxt.is_host()) {
    return std::make_tuple(alloc::host, Ctxt.get_devices()[0]);
  }
  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  std::shared_ptr<detail::usm::USMDispatcher> Dispatch =
      CtxImpl->getUSMDispatch();
  pi_context C = CtxImpl->getHandleRef();
  pi_usm_type AllocTy;
  pi_device DeviceId;

  // All these CL enums should be replicated in PI
  detail::pi::checkPiResult(Dispatch->getMemAllocInfo(
      C, Ptr, PI_MEM_ALLOC_TYPE, sizeof(pi_mem_info), &AllocTy, nullptr));
  detail::pi::checkPiResult(Dispatch->getMemAllocInfo(
      C, Ptr, PI_MEM_ALLOC_INFO_DEVICE, sizeof(pi_device), &DeviceId, nullptr));

  alloc ResultAlloc = alloc::unknown;;
  if (AllocTy == PI_MEM_TYPE_HOST) {
    ResultAlloc = alloc::host;
  } else if (AllocTy == PI_MEM_TYPE_DEVICE) {
    ResultAlloc = alloc::device;
  } else if (AllocTy == PI_MEM_TYPE_SHARED) {
    ResultAlloc = alloc::shared;
  }

  // Check device id
  device Result;
  if (DeviceId == nullptr) {
    // Host allocs don't return a device id.
    // It's irrelevant, so don't do anything in this case.
  } else  {
    auto Devs = Ctxt.get_devices();
    bool Found = false;
    for (auto D : Devs) {
      // Try to find the real sycl device we used in the Context
      if (detail::pi::cast<pi_device>(D.get()) == DeviceId) {
        Result = D;
        Found = true;
        break;
      }
    }
    if (!Found) {
      throw runtime_error("Cannot find device associated with USM allocation!");
    }
  }

  return std::make_tuple(ResultAlloc, Result);
}

} // namespace sycl
} // namespace cl
