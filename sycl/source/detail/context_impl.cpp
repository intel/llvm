//==---------------- context_impl.cpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/cuda_definitions.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/properties/context_properties.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/context_impl.hpp>
#include <detail/context_info.hpp>
#include <detail/platform_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler,
                           const property_list &PropList)
    : MAsyncHandler(AsyncHandler), MDevices(1, Device), MContext(nullptr),
      MPlatform(), MPropList(PropList), MHostContext(Device.is_host()) {
  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(const vector_class<cl::sycl::device> Devices,
                           async_handler AsyncHandler,
                           const property_list &PropList)
    : MAsyncHandler(AsyncHandler), MDevices(Devices), MContext(nullptr),
      MPlatform(), MPropList(PropList), MHostContext(false) {
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  vector_class<RT::PiDevice> DeviceIds;
  for (const auto &D : MDevices) {
    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  const auto Backend = getPlugin().getBackend();
  if (Backend == backend::cuda) {
    const bool UseCUDAPrimaryContext =
        MPropList.has_property<property::context::cuda::use_primary_context>();
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
      MHostContext(false) {

  vector_class<RT::PiDevice> DeviceIds;
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
  if (getPlugin().getBackend() == cl::sycl::backend::opencl) {
    getPlugin().call<PiApiKind::piContextRetain>(MContext);
  }
  MKernelProgramCache.setContextPtr(this);
}

cl_context context_impl::get() const {
  if (MHostContext) {
    throw invalid_object_error(
        "This instance of context doesn't support OpenCL interoperability.",
        PI_INVALID_CONTEXT);
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
cl_uint context_impl::get_info<info::context::reference_count>() const {
  if (is_host())
    return 0;
  return get_context_info<info::context::reference_count>::get(
      this->getHandleRef(), this->getPlugin());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  if (is_host())
    return platform();
  return createSyclObjFromImpl<platform>(MPlatform);
}
template <>
vector_class<cl::sycl::device>
context_impl::get_info<info::context::devices>() const {
  return MDevices;
}

RT::PiContext &context_impl::getHandleRef() { return MContext; }
const RT::PiContext &context_impl::getHandleRef() const { return MContext; }

KernelProgramCache &context_impl::getKernelProgramCache() const {
  return MKernelProgramCache;
}

bool context_impl::hasDevice(
    shared_ptr_class<detail::device_impl> Device) const {
  for (auto D : MDevices)
    if (getSyclObjImpl(D) == Device)
      return true;
  return false;
}

pi_native_handle context_impl::getNative() const {
  auto Plugin = getPlugin();
  if (Plugin.getBackend() == backend::opencl)
    Plugin.call<PiApiKind::piContextRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextContextGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
