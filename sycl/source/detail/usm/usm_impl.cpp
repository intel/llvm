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
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
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
  const auto &devices = Ctxt.get_devices();
  if (!std::any_of(devices.begin(), devices.end(), [&](const auto &device) {
        return device.has(sycl::aspect::usm_host_allocations);
      })) {
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "No device in this context supports USM host allocations!");
  }
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;

  std::shared_ptr<context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  ur_context_handle_t C = CtxImpl->getHandleRef();
  const PluginPtr &Plugin = CtxImpl->getPlugin();
  ur_result_t Error = UR_RESULT_ERROR_INVALID_VALUE;

  switch (Kind) {
  case alloc::host: {
    ur_usm_desc_t UsmDesc{};
    UsmDesc.align = Alignment;

    ur_usm_alloc_location_desc_t UsmLocationDesc{};
    UsmLocationDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

    if (PropList.has_property<
            sycl::ext::intel::experimental::property::usm::buffer_location>() &&
        Ctxt.get_platform().has_extension(
            "cl_intel_mem_alloc_buffer_location")) {
      UsmLocationDesc.location = static_cast<uint32_t>(
          PropList
              .get_property<sycl::ext::intel::experimental::property::usm::
                                buffer_location>()
              .get_buffer_location());
      UsmDesc.pNext = &UsmLocationDesc;
    }

    Error = Plugin->call_nocheck<sycl::detail::UrApiKind::urUSMHostAlloc>(
        C, &UsmDesc,
        /* pool= */ nullptr, Size, &RetVal);

    break;
  }
  case alloc::device:
  case alloc::shared:
  case alloc::unknown: {
    RetVal = nullptr;
    Error = UR_RESULT_ERROR_INVALID_VALUE;
    break;
  }
  }

  // Error is for debugging purposes.
  // The spec wants a nullptr returned, not an exception.
  if (Error != UR_RESULT_SUCCESS)
    return nullptr;
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
  if (Kind == alloc::device &&
      !DevImpl->has(sycl::aspect::usm_device_allocations)) {
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Device does not support USM device allocations!");
  }
  if (Kind == alloc::shared &&
      !DevImpl->has(sycl::aspect::usm_shared_allocations)) {
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Device does not support shared USM allocations!");
  }
  void *RetVal = nullptr;
  if (Size == 0)
    return nullptr;

  ur_context_handle_t C = CtxImpl->getHandleRef();
  const PluginPtr &Plugin = CtxImpl->getPlugin();
  ur_result_t Error = UR_RESULT_ERROR_INVALID_VALUE;
  ur_device_handle_t Dev;

  switch (Kind) {
  case alloc::device: {
    Dev = DevImpl->getHandleRef();

    ur_usm_desc_t UsmDesc{};
    UsmDesc.align = Alignment;

    ur_usm_alloc_location_desc_t UsmLocationDesc{};
    UsmLocationDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

    // Buffer location is only supported on FPGA devices
    if (PropList.has_property<
            sycl::ext::intel::experimental::property::usm::buffer_location>() &&
        DevImpl->has_extension("cl_intel_mem_alloc_buffer_location")) {
      UsmLocationDesc.location = static_cast<uint32_t>(
          PropList
              .get_property<sycl::ext::intel::experimental::property::usm::
                                buffer_location>()
              .get_buffer_location());
      UsmDesc.pNext = &UsmLocationDesc;
    }

    Error = Plugin->call_nocheck<detail::UrApiKind::urUSMDeviceAlloc>(
        C, Dev, &UsmDesc,
        /*pool=*/nullptr, Size, &RetVal);

    break;
  }
  case alloc::shared: {
    Dev = DevImpl->getHandleRef();

    ur_usm_desc_t UsmDesc{};
    UsmDesc.align = Alignment;

    ur_usm_alloc_location_desc_t UsmLocationDesc{};
    UsmLocationDesc.stype = UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC;

    ur_usm_device_desc_t UsmDeviceDesc{};
    UsmDeviceDesc.stype = UR_STRUCTURE_TYPE_USM_DEVICE_DESC;
    UsmDeviceDesc.flags = 0;

    UsmDesc.pNext = &UsmDeviceDesc;

    if (PropList.has_property<
            sycl::ext::oneapi::property::usm::device_read_only>()) {
      UsmDeviceDesc.flags |= UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;
    }

    if (PropList.has_property<
            sycl::ext::intel::experimental::property::usm::buffer_location>() &&
        DevImpl->has_extension("cl_intel_mem_alloc_buffer_location")) {
      UsmLocationDesc.location = static_cast<uint32_t>(
          PropList
              .get_property<sycl::ext::intel::experimental::property::usm::
                                buffer_location>()
              .get_buffer_location());
      UsmDeviceDesc.pNext = &UsmLocationDesc;
    }

    Error = Plugin->call_nocheck<detail::UrApiKind::urUSMSharedAlloc>(
        C, Dev, &UsmDesc,
        /*pool=*/nullptr, Size, &RetVal);

    break;
  }
  case alloc::host:
  case alloc::unknown: {
    RetVal = nullptr;
    Error = UR_RESULT_ERROR_INVALID_VALUE;
    break;
  }
  }

  // Error is for debugging purposes.
  // The spec wants a nullptr returned, not an exception.
  if (Error != UR_RESULT_SUCCESS)
    return nullptr;
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
  ur_context_handle_t C = CtxImpl->getHandleRef();
  const PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::UrApiKind::urUSMFree>(C, Ptr);
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

  ur_context_handle_t URCtx = CtxImpl->getHandleRef();
  ur_usm_type_t AllocTy;

  // query type using UR function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  ur_result_t Err =
      Plugin->call_nocheck<detail::UrApiKind::urUSMGetMemAllocInfo>(
          URCtx, Ptr, UR_USM_ALLOC_INFO_TYPE, sizeof(ur_usm_type_t), &AllocTy,
          nullptr);

  // UR_RESULT_ERROR_INVALID_VALUE means USM doesn't know about this ptr
  if (Err == UR_RESULT_ERROR_INVALID_VALUE)
    return alloc::unknown;
  // otherwise UR_RESULT_SUCCESS is expected
  if (Err != UR_RESULT_SUCCESS) {
    throw detail::set_ur_error(
        exception(make_error_code(errc::runtime), "get_pointer_type() failed"),
        Err);
  }

  alloc ResultAlloc;
  switch (AllocTy) {
  case UR_USM_TYPE_HOST:
    ResultAlloc = alloc::host;
    break;
  case UR_USM_TYPE_DEVICE:
    ResultAlloc = alloc::device;
    break;
  case UR_USM_TYPE_SHARED:
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
    throw exception(make_error_code(errc::invalid),
                    "Ptr not a valid USM allocation!");

  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);

  // Check if ptr is a host allocation
  if (get_pointer_type(Ptr, Ctxt) == alloc::host) {
    auto Devs = CtxImpl->getDevices();
    if (Devs.size() == 0)
      throw exception(make_error_code(errc::invalid),
                      "No devices in passed context!");

    // Just return the first device in the context
    return Devs[0];
  }

  ur_context_handle_t URCtx = CtxImpl->getHandleRef();
  ur_device_handle_t DeviceId;

  // query device using UR function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::UrApiKind::urUSMGetMemAllocInfo>(
      URCtx, Ptr, UR_USM_ALLOC_INFO_DEVICE, sizeof(ur_device_handle_t),
      &DeviceId, nullptr);

  // The device is not necessarily a member of the context, it could be a
  // member's descendant instead. Fetch the corresponding device from the cache.
  std::shared_ptr<detail::platform_impl> PltImpl = CtxImpl->getPlatformImpl();
  std::shared_ptr<detail::device_impl> DevImpl =
      PltImpl->getDeviceImpl(DeviceId);
  if (DevImpl)
    return detail::createSyclObjFromImpl<device>(DevImpl);
  throw exception(make_error_code(errc::runtime),
                  "Cannot find device associated with USM allocation!");
}

// Device copy enhancement APIs, prepare_for and release_from USM.

static void prepare_for_usm_device_copy(const void *Ptr, size_t Size,
                                        const context &Ctxt) {
  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  ur_context_handle_t URCtx = CtxImpl->getHandleRef();
  // Call the UR function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::UrApiKind::urUSMImportExp>(
      URCtx, const_cast<void *>(Ptr), Size);
}

static void release_from_usm_device_copy(const void *Ptr, const context &Ctxt) {
  std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctxt);
  ur_context_handle_t URCtx = CtxImpl->getHandleRef();
  // Call the UR function
  const detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  Plugin->call<detail::UrApiKind::urUSMReleaseExp>(URCtx,
                                                   const_cast<void *>(Ptr));
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
