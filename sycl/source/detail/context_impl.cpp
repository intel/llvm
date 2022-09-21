//==---------------- context_impl.cpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <detail/context_impl.hpp>
#include <detail/context_info.hpp>
#include <detail/platform_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>
#include <sycl/properties/context_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler,
                           const property_list &PropList)
    : MAsyncHandler(AsyncHandler), MDevices(1, Device), MContext(nullptr),
      MPlatform(), MPropList(PropList),
      MHostContext(detail::getSyclObjImpl(Device)->is_host()),
      MSupportBufferLocationByDevices(NotChecked) {
  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(const std::vector<sycl::device> Devices,
                           async_handler AsyncHandler,
                           const property_list &PropList)
    : MAsyncHandler(AsyncHandler), MDevices(Devices), MContext(nullptr),
      MPlatform(), MPropList(PropList), MHostContext(false),
      MSupportBufferLocationByDevices(NotChecked) {
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  std::vector<RT::PiDevice> DeviceIds;
  for (const auto &D : MDevices) {
    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  const auto Backend = getPlugin().getBackend();
  if (Backend == backend::ext_oneapi_cuda) {
    const bool UseCUDAPrimaryContext = MPropList.has_property<
        ext::oneapi::cuda::property::context::use_primary_context>();
    const pi_context_properties Props[] = {
        static_cast<pi_context_properties>(
            __SYCL_PI_CONTEXT_PROPERTIES_CUDA_PRIMARY),
        static_cast<pi_context_properties>(UseCUDAPrimaryContext), 0};

    getPlugin().call<PiApiKind::piContextCreate>(
        Props, DeviceIds.size(), DeviceIds.data(), nullptr, nullptr, &MContext);
  } else {
    getPlugin().call<PiApiKind::piContextCreate>(nullptr, DeviceIds.size(),
                                                 DeviceIds.data(), nullptr,
                                                 nullptr, &MContext);
  }

  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(RT::PiContext PiContext, async_handler AsyncHandler,
                           const plugin &Plugin)
    : MAsyncHandler(AsyncHandler), MDevices(), MContext(PiContext), MPlatform(),
      MHostContext(false), MSupportBufferLocationByDevices(NotChecked) {

  std::vector<RT::PiDevice> DeviceIds;
  size_t DevicesNum = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piContextGetInfo>(
      MContext, PI_CONTEXT_INFO_NUM_DEVICES, sizeof(DevicesNum), &DevicesNum,
      nullptr);
  DeviceIds.resize(DevicesNum);
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin.call<PiApiKind::piContextGetInfo>(MContext, PI_CONTEXT_INFO_DEVICES,
                                           sizeof(RT::PiDevice) * DevicesNum,
                                           &DeviceIds[0], nullptr);

  if (!DeviceIds.empty()) {
    std::shared_ptr<detail::platform_impl> Platform =
        platform_impl::getPlatformFromPiDevice(DeviceIds[0], Plugin);
    for (RT::PiDevice Dev : DeviceIds) {
      MDevices.emplace_back(createSyclObjFromImpl<device>(
          Platform->getOrMakeDeviceImpl(Dev, Platform)));
    }
    MPlatform = Platform;
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  // getPlugin() will be the same as the Plugin passed. This should be taken
  // care of when creating device object.
  //
  // TODO: Move this backend-specific retain of the context to SYCL-2020 style
  //       make_context<backend::opencl> interop, when that is created.
  if (getPlugin().getBackend() == sycl::backend::opencl) {
    getPlugin().call<PiApiKind::piContextRetain>(MContext);
  }
  MKernelProgramCache.setContextPtr(this);
}

cl_context context_impl::get() const {
  if (MHostContext) {
    throw invalid_object_error(
        "This instance of context doesn't support OpenCL interoperability.",
        PI_ERROR_INVALID_CONTEXT);
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  getPlugin().call<PiApiKind::piContextRetain>(MContext);
  return pi::cast<cl_context>(MContext);
}

bool context_impl::is_host() const { return MHostContext; }

context_impl::~context_impl() {
  for (auto LibProg : MCachedLibPrograms) {
    assert(LibProg.second && "Null program must not be kept in the cache");
    getPlugin().call<PiApiKind::piProgramRelease>(LibProg.second);
  }
  if (!MHostContext) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    getPlugin().call<PiApiKind::piContextRelease>(MContext);
  }
}

const async_handler &context_impl::get_async_handler() const {
  return MAsyncHandler;
}

template <>
uint32_t context_impl::get_info<info::context::reference_count>() const {
  if (is_host())
    return 0;
  return get_context_info<info::context::reference_count>(this->getHandleRef(),
                                                          this->getPlugin());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  if (is_host())
    return createSyclObjFromImpl<platform>(
        platform_impl::getHostPlatformImpl());
  return createSyclObjFromImpl<platform>(MPlatform);
}
template <>
std::vector<sycl::device>
context_impl::get_info<info::context::devices>() const {
  return MDevices;
}
template <>
std::vector<sycl::memory_order>
context_impl::get_info<info::context::atomic_memory_order_capabilities>()
    const {
  if (is_host())
    return {sycl::memory_order::relaxed, sycl::memory_order::acquire,
            sycl::memory_order::release, sycl::memory_order::acq_rel,
            sycl::memory_order::seq_cst};

  pi_memory_order_capabilities Result;
  getPlugin().call<PiApiKind::piContextGetInfo>(
      MContext,
      PiInfoCode<info::context::atomic_memory_order_capabilities>::value,
      sizeof(Result), &Result, nullptr);
  return readMemoryOrderBitfield(Result);
}
template <>
std::vector<sycl::memory_scope>
context_impl::get_info<info::context::atomic_memory_scope_capabilities>()
    const {
  if (is_host())
    return {sycl::memory_scope::work_item, sycl::memory_scope::sub_group,
            sycl::memory_scope::work_group, sycl::memory_scope::device,
            sycl::memory_scope::system};

  pi_memory_scope_capabilities Result;
  getPlugin().call<PiApiKind::piContextGetInfo>(
      MContext,
      PiInfoCode<info::context::atomic_memory_scope_capabilities>::value,
      sizeof(Result), &Result, nullptr);
  return readMemoryScopeBitfield(Result);
}

RT::PiContext &context_impl::getHandleRef() { return MContext; }
const RT::PiContext &context_impl::getHandleRef() const { return MContext; }

KernelProgramCache &context_impl::getKernelProgramCache() const {
  return MKernelProgramCache;
}

bool context_impl::hasDevice(
    std::shared_ptr<detail::device_impl> Device) const {
  for (auto D : MDevices)
    if (getSyclObjImpl(D) == Device)
      return true;
  return false;
}

DeviceImplPtr
context_impl::findMatchingDeviceImpl(RT::PiDevice &DevicePI) const {
  for (device D : MDevices)
    if (getSyclObjImpl(D)->getHandleRef() == DevicePI)
      return getSyclObjImpl(D);

  return nullptr;
}

pi_native_handle context_impl::getNative() const {
  auto Plugin = getPlugin();
  if (Plugin.getBackend() == backend::opencl)
    Plugin.call<PiApiKind::piContextRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextContextGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

bool context_impl::isBufferLocationSupported() const {
  if (MSupportBufferLocationByDevices != NotChecked)
    return MSupportBufferLocationByDevices == Supported ? true : false;
  // Check that devices within context have support of buffer location
  MSupportBufferLocationByDevices = Supported;
  for (auto &Device : MDevices) {
    if (!Device.has_extension("cl_intel_mem_alloc_buffer_location")) {
      MSupportBufferLocationByDevices = NotSupported;
      break;
    }
  }
  return MSupportBufferLocationByDevices == Supported ? true : false;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
