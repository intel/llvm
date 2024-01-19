//==---------------- usm_impl.cpp - USM API Utils  -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <detail/queue_impl.hpp>
#include <detail/usm/usm_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/aligned_allocator.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm.hpp>

#include <array>
#include <cassert>
#include <cstdlib>
#include <memory>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.hpp"
#include <detail/xpti_registry.hpp>
#endif

namespace sycl {
inline namespace _V1 {

using alloc = sycl::usm::alloc;

namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GSYCLGraphEvent;
#endif
namespace usm {

void *alignedAllocHost(size_t Alignment, size_t Size, const context &Ctxt,
                       alloc Kind, const property_list &PropList,
                       const detail::code_location &CodeLoc) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Stash the code location information and propagate
  detail::tls_code_loc_t CL(CodeLoc);
  XPTIScope PrepareNotify((void *)alignedAllocHost,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_MEM_ALLOC_STREAM_NAME, "malloc_host");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device_name", std::string("Host"));
    xpti::addMetadata(TEvent, "sycl_device", 0);
    xpti::addMetadata(TEvent, "memory_size", Size);
  });
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify(
      (uint16_t)xpti::trace_point_type_t::mem_alloc_begin);
#endif
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;

  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  if (CtxImpl->is_host()) {
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
    pi_context C = CtxImpl->getHandleRef();
    const PluginPtr &Plugin = CtxImpl->getPlugin();
    pi_result Error = PI_ERROR_INVALID_VALUE;

    switch (Kind) {
    case alloc::host: {
      std::array<pi_usm_mem_properties, 3> Props;
      auto PropsIter = Props.begin();

      if (PropList.has_property<sycl::ext::intel::experimental::property::usm::
                                    buffer_location>() &&
          Ctxt.get_platform().has_extension(
              "cl_intel_mem_alloc_buffer_location")) {
        *PropsIter++ = PI_MEM_USM_ALLOC_BUFFER_LOCATION;
        *PropsIter++ = PropList
                           .get_property<sycl::ext::intel::experimental::
                                             property::usm::buffer_location>()
                           .get_buffer_location();
      }

      assert(PropsIter >= Props.begin() && PropsIter < Props.end());
      *PropsIter++ = 0; // null-terminate property list

      Error = Plugin->call_nocheck<PiApiKind::piextUSMHostAlloc>(
          &RetVal, C, Props.data(), Size, Alignment);

      break;
    }
    case alloc::device:
    case alloc::shared:
    case alloc::unknown: {
      RetVal = nullptr;
      Error = PI_ERROR_INVALID_VALUE;
      break;
    }
    }

    // Error is for debugging purposes.
    // The spec wants a nullptr returned, not an exception.
    if (Error != PI_SUCCESS)
      return nullptr;
  }
#ifdef XPTI_ENABLE_INSTRUMENTATION
  xpti::addMetadata(PrepareNotify.traceEvent(), "memory_ptr",
                    reinterpret_cast<size_t>(RetVal));
#endif
  return RetVal;
}

void *alignedAllocInternal(size_t Alignment, size_t Size,
                           const context_impl *CtxImpl,
                           const device_impl *DevImpl, alloc Kind,
                           const property_list &PropList) {
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;

  if (CtxImpl->is_host()) {
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
    pi_context C = CtxImpl->getHandleRef();
    const PluginPtr &Plugin = CtxImpl->getPlugin();
    pi_result Error = PI_ERROR_INVALID_VALUE;
    pi_device Id;

    switch (Kind) {
    case alloc::device: {
      //if (Kind == alloc::device &&
    //!DevImpl->has(sycl::aspect::usm_device_allocations)) {
      //  throw sycl::exception(sycl::errc::feature_not_supported,
        //    "Device does not support Unified Shared Memory!");
  //}
      Id = DevImpl->getHandleRef();

      std::array<pi_usm_mem_properties, 3> Props;
      auto PropsIter = Props.begin();

      // Buffer location is only supported on FPGA devices
      if (PropList.has_property<sycl::ext::intel::experimental::property::usm::
                                    buffer_location>() &&
          DevImpl->has_extension("cl_intel_mem_alloc_buffer_location")) {
        *PropsIter++ = PI_MEM_USM_ALLOC_BUFFER_LOCATION;
        *PropsIter++ = PropList
                           .get_property<sycl::ext::intel::experimental::
                                             property::usm::buffer_location>()
                           .get_buffer_location();
      }

      assert(PropsIter >= Props.begin() && PropsIter < Props.end());
      *PropsIter++ = 0; // null-terminate property list

      Error = Plugin->call_nocheck<PiApiKind::piextUSMDeviceAlloc>(
          &RetVal, C, Id, Props.data(), Size, Alignment);

      break;
    }
    case alloc::shared: {
      Id = DevImpl->getHandleRef();

      std::array<pi_usm_mem_properties, 5> Props;
      auto PropsIter = Props.begin();

      if (PropList.has_property<
              sycl::ext::oneapi::property::usm::device_read_only>()) {
        *PropsIter++ = PI_MEM_ALLOC_FLAGS;
        *PropsIter++ = PI_MEM_ALLOC_DEVICE_READ_ONLY;
      }

      if (PropList.has_property<sycl::ext::intel::experimental::property::usm::
                                    buffer_location>() &&
          DevImpl->has_extension("cl_intel_mem_alloc_buffer_location")) {
        *PropsIter++ = PI_MEM_USM_ALLOC_BUFFER_LOCATION;
        *PropsIter++ = PropList
                           .get_property<sycl::ext::intel::experimental::
                                             property::usm::buffer_location>()
                           .get_buffer_location();
      }

      assert(PropsIter >= Props.begin() && PropsIter < Props.end());
      *PropsIter++ = 0; // null-terminate property list

      Error = Plugin->call_nocheck<PiApiKind::piextUSMSharedAlloc>(
          &RetVal, C, Id, Props.data(), Size, Alignment);

      break;
    }
    case alloc::host:
    case alloc::unknown: {
      RetVal = nullptr;
      Error = PI_ERROR_INVALID_VALUE;
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
                   const device &Dev, alloc Kind, const property_list &PropList,
                   const detail::code_location &CodeLoc) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Stash the code location information and propagate
  detail::tls_code_loc_t CL(CodeLoc);
  XPTIScope PrepareNotify((void *)alignedAlloc,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_MEM_ALLOC_STREAM_NAME, "usm::alignedAlloc");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device_name",
                      Dev.get_info<info::device::name>());
    // Need to determine how to get the device handle reference
    // xpti::addMetadata(TEvent, "sycl_device", Dev.getHandleRef()));
    xpti::addMetadata(TEvent, "memory_size", Size);
  });
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify(
      (uint16_t)xpti::trace_point_type_t::mem_alloc_begin);
#endif
  void *RetVal =
      alignedAllocInternal(Alignment, Size, getSyclObjImpl(Ctxt).get(),
                           getSyclObjImpl(Dev).get(), Kind, PropList);
#ifdef XPTI_ENABLE_INSTRUMENTATION
  xpti::addMetadata(PrepareNotify.traceEvent(), "memory_ptr",
                    reinterpret_cast<size_t>(RetVal));
#endif
  return RetVal;
}

void freeInternal(void *Ptr, const context_impl *CtxImpl) {
  if (Ptr == nullptr)
    return;
  if (CtxImpl->is_host()) {
    // need to use alignedFree here for Windows
    detail::OSUtil::alignedFree(Ptr);
  } else {
    pi_context C = CtxImpl->getHandleRef();
    const PluginPtr &Plugin = CtxImpl->getPlugin();
    Plugin->call<PiApiKind::piextUSMFree>(C, Ptr);
  }
}

void free(void *Ptr, const context &Ctxt,
          const detail::code_location &CodeLoc) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // Stash the code location information and propagate
  detail::tls_code_loc_t CL(CodeLoc);
  XPTIScope PrepareNotify((void *)free,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_MEM_ALLOC_STREAM_NAME, "usm::free");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "memory_ptr", reinterpret_cast<size_t>(Ptr));
  });
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify(
      (uint16_t)xpti::trace_point_type_t::mem_release_begin);
#endif
  freeInternal(Ptr, detail::getSyclObjImpl(Ctxt).get());
}

} // namespace usm
} // namespace detail

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device,
                                   property_list{}, CodeLoc);
}

void *malloc_device(size_t Size, const device &Dev, const context &Ctxt,
                    const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::device, PropList,
                                   CodeLoc);
}

void *malloc_device(size_t Size, const queue &Q,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   alloc::device, property_list{}, CodeLoc);
}

void *malloc_device(size_t Size, const queue &Q, const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   alloc::device, PropList, CodeLoc);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device,
                                   property_list{}, CodeLoc);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt, const property_list &PropList,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::device,
                                   PropList, CodeLoc);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const queue &Q,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), alloc::device,
                                   property_list{}, CodeLoc);
}

void *aligned_alloc_device(size_t Alignment, size_t Size, const queue &Q,
                           const property_list &PropList,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), alloc::device, PropList,
                                   CodeLoc);
}

void free(void *ptr, const context &Ctxt,
          const detail::code_location &CodeLoc) {
  return detail::usm::free(ptr, Ctxt, CodeLoc);
}

void free(void *ptr, const queue &Q, const detail::code_location &CodeLoc) {
  return detail::usm::free(ptr, Q.get_context(), CodeLoc);
}

void *malloc_host(size_t Size, const context &Ctxt,
                  const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(0, Size, Ctxt, alloc::host,
                                       property_list{}, CodeLoc);
}

void *malloc_host(size_t Size, const context &Ctxt,
                  const property_list &PropList,
                  const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(0, Size, Ctxt, alloc::host, PropList,
                                       CodeLoc);
}

void *malloc_host(size_t Size, const queue &Q,
                  const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(0, Size, Q.get_context(), alloc::host,
                                       property_list{}, CodeLoc);
}

void *malloc_host(size_t Size, const queue &Q, const property_list &PropList,
                  const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(0, Size, Q.get_context(), alloc::host,
                                       PropList, CodeLoc);
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared,
                                   property_list{}, CodeLoc);
}

void *malloc_shared(size_t Size, const device &Dev, const context &Ctxt,
                    const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, alloc::shared, PropList,
                                   CodeLoc);
}

void *malloc_shared(size_t Size, const queue &Q,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   alloc::shared, property_list{}, CodeLoc);
}

void *malloc_shared(size_t Size, const queue &Q, const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   alloc::shared, PropList, CodeLoc);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt,
                         const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, alloc::host,
                                       property_list{}, CodeLoc);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const context &Ctxt,
                         const property_list &PropList,
                         const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, alloc::host,
                                       PropList, CodeLoc);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const queue &Q,
                         const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(Alignment, Size, Q.get_context(),
                                       alloc::host, property_list{}, CodeLoc);
}

void *aligned_alloc_host(size_t Alignment, size_t Size, const queue &Q,
                         const property_list &PropList,
                         const detail::code_location &CodeLoc) {
  return detail::usm::alignedAllocHost(Alignment, Size, Q.get_context(),
                                       alloc::host, PropList, CodeLoc);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared,
                                   property_list{}, CodeLoc);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const device &Dev,
                           const context &Ctxt, const property_list &PropList,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, alloc::shared,
                                   PropList, CodeLoc);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const queue &Q,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), alloc::shared,
                                   property_list{}, CodeLoc);
}

void *aligned_alloc_shared(size_t Alignment, size_t Size, const queue &Q,
                           const property_list &PropList,
                           const detail::code_location &CodeLoc) {
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), alloc::shared, PropList,
                                   CodeLoc);
}

// single form

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind,
             const property_list &PropList,
             const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(0, Size, Ctxt, Kind, PropList,
                                         CodeLoc);
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind, PropList, CodeLoc);
}

void *malloc(size_t Size, const device &Dev, const context &Ctxt, alloc Kind,
             const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(0, Size, Ctxt, Kind, property_list{},
                                         CodeLoc);
  return detail::usm::alignedAlloc(0, Size, Ctxt, Dev, Kind, property_list{},
                                   CodeLoc);
}

void *malloc(size_t Size, const queue &Q, alloc Kind,
             const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(0, Size, Q.get_context(), Kind,
                                         property_list{}, CodeLoc);
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   Kind, property_list{}, CodeLoc);
}

void *malloc(size_t Size, const queue &Q, alloc Kind,
             const property_list &PropList,
             const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(0, Size, Q.get_context(), Kind,
                                         PropList, CodeLoc);
  return detail::usm::alignedAlloc(0, Size, Q.get_context(), Q.get_device(),
                                   Kind, PropList, CodeLoc);
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind,
                    const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, Kind,
                                         property_list{}, CodeLoc);

  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind,
                                   property_list{}, CodeLoc);
}

void *aligned_alloc(size_t Alignment, size_t Size, const device &Dev,
                    const context &Ctxt, alloc Kind,
                    const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(Alignment, Size, Ctxt, Kind, PropList,
                                         CodeLoc);
  return detail::usm::alignedAlloc(Alignment, Size, Ctxt, Dev, Kind, PropList,
                                   CodeLoc);
}

void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q, alloc Kind,
                    const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(Alignment, Size, Q.get_context(), Kind,
                                         property_list{}, CodeLoc);
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), Kind, property_list{},
                                   CodeLoc);
}

void *aligned_alloc(size_t Alignment, size_t Size, const queue &Q, alloc Kind,
                    const property_list &PropList,
                    const detail::code_location &CodeLoc) {
  if (Kind == alloc::host)
    return detail::usm::alignedAllocHost(Alignment, Size, Q.get_context(), Kind,
                                         PropList, CodeLoc);
  return detail::usm::alignedAlloc(Alignment, Size, Q.get_context(),
                                   Q.get_device(), Kind, PropList, CodeLoc);
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

  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);

  // Everything on a host device is just system malloc so call it host
  if (CtxImpl->is_host())
    return alloc::host;

  pi_context PICtx = CtxImpl->getHandleRef();
  pi_usm_type AllocTy;

  // query type using PI function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  sycl::detail::pi::PiResult Err =
      Plugin->call_nocheck<detail::PiApiKind::piextUSMGetMemAllocInfo>(
          PICtx, Ptr, PI_MEM_ALLOC_TYPE, sizeof(pi_usm_type), &AllocTy,
          nullptr);

  // PI_ERROR_INVALID_VALUE means USM doesn't know about this ptr
  if (Err == PI_ERROR_INVALID_VALUE)
    return alloc::unknown;
  // otherwise PI_SUCCESS is expected
  if (Err != PI_SUCCESS) {
    Plugin->reportPiError(Err, "get_pointer_type()");
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
    throw runtime_error("Ptr not a valid USM allocation!",
                        PI_ERROR_INVALID_VALUE);

  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);

  // Just return the host device in the host context
  if (CtxImpl->is_host())
    return Ctxt.get_devices()[0];

  // Check if ptr is a host allocation
  if (get_pointer_type(Ptr, Ctxt) == alloc::host) {
    auto Devs = CtxImpl->getDevices();
    if (Devs.size() == 0)
      throw runtime_error("No devices in passed context!",
                          PI_ERROR_INVALID_VALUE);

    // Just return the first device in the context
    return Devs[0];
  }

  pi_context PICtx = CtxImpl->getHandleRef();
  pi_device DeviceId;

  // query device using PI function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::PiApiKind::piextUSMGetMemAllocInfo>(
      PICtx, Ptr, PI_MEM_ALLOC_DEVICE, sizeof(pi_device), &DeviceId, nullptr);

  // The device is not necessarily a member of the context, it could be a
  // member's descendant instead. Fetch the corresponding device from the cache.
  std::shared_ptr<detail::platform_impl> PltImpl = CtxImpl->getPlatformImpl();
  std::shared_ptr<detail::device_impl> DevImpl =
      PltImpl->getDeviceImpl(DeviceId);
  if (DevImpl)
    return detail::createSyclObjFromImpl<device>(DevImpl);
  throw runtime_error("Cannot find device associated with USM allocation!",
                      PI_ERROR_INVALID_OPERATION);
}

// Device copy enhancement APIs, prepare_for and release_from USM.

static void prepare_for_usm_device_copy(const void *Ptr, size_t Size,
                                        const context &Ctxt) {
  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  pi_context PICtx = CtxImpl->getHandleRef();
  // Call the PI function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::PiApiKind::piextUSMImport>(Ptr, Size, PICtx);
}

static void release_from_usm_device_copy(const void *Ptr, const context &Ctxt) {
  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  pi_context PICtx = CtxImpl->getHandleRef();
  // Call the PI function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::PiApiKind::piextUSMRelease>(Ptr, PICtx);
}

namespace ext::oneapi::experimental {
void prepare_for_device_copy(const void *Ptr, size_t Size,
                             const context &Ctxt) {
  prepare_for_usm_device_copy(Ptr, Size, Ctxt);
}

void prepare_for_device_copy(const void *Ptr, size_t Size, const queue &Queue) {
  prepare_for_usm_device_copy(Ptr, Size, Queue.get_context());
}

void release_from_device_copy(const void *Ptr, const context &Ctxt) {
  release_from_usm_device_copy(Ptr, Ctxt);
}

void release_from_device_copy(const void *Ptr, const queue &Queue) {
  release_from_usm_device_copy(Ptr, Queue.get_context());
}
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
