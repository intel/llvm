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
#include <CL/sycl/stl.hpp>
#include <detail/context_impl.hpp>
#include <detail/context_info.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler,
                           bool UseCUDAPrimaryContext)
    : MAsyncHandler(AsyncHandler), MDevices(1, Device), MContext(nullptr),
      MPlatform(), MPluginInterop(false), MHostContext(true),
      MUseCUDAPrimaryContext(UseCUDAPrimaryContext) {
  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(const vector_class<cl::sycl::device> Devices,
                           async_handler AsyncHandler, bool UseCUDAPrimaryContext)
    : MAsyncHandler(AsyncHandler), MDevices(Devices), MContext(nullptr),
      MPlatform(), MPluginInterop(true), MHostContext(false),
      MUseCUDAPrimaryContext(UseCUDAPrimaryContext) {
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  vector_class<RT::PiDevice> DeviceIds;
  for (const auto &D : MDevices) {
    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  const auto Backend = getPlugin().getBackend();
  if (Backend == backend::cuda) {
#if USE_PI_CUDA
    const pi_context_properties props[] = {
        static_cast<pi_context_properties>(PI_CONTEXT_PROPERTIES_CUDA_PRIMARY),
        static_cast<pi_context_properties>(UseCUDAPrimaryContext), 0};

    getPlugin().call<PiApiKind::piContextCreate>(props, DeviceIds.size(), 
	  	  DeviceIds.data(), nullptr, nullptr, &MContext);
#else
    cl::sycl::detail::pi::die("CUDA support was not enabled at compilation time");
#endif
  } else {
    getPlugin().call<PiApiKind::piContextCreate>(nullptr, DeviceIds.size(), 
	  	  DeviceIds.data(), nullptr, nullptr, &MContext);
  }

  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(RT::PiContext PiContext, async_handler AsyncHandler,
                           const plugin &Plugin)
    : MAsyncHandler(AsyncHandler), MDevices(), MContext(PiContext), MPlatform(),
      MPluginInterop(true), MHostContext(false) {

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

  for (auto Dev : DeviceIds) {
    MDevices.emplace_back(createSyclObjFromImpl<device>(
        std::make_shared<device_impl>(Dev, Plugin)));
  }
  // TODO What if m_Devices if empty? m_Devices[0].get_platform()
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  // TODO catch an exception and put it to list of asynchronous exceptions
  // getPlugin() will be the same as the Plugin passed. This should be taken
  // care of when creating device object.
  getPlugin().call<PiApiKind::piContextRetain>(MContext);
  MKernelProgramCache.setContextPtr(this);
}

cl_context context_impl::get() const {
  if (MPluginInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    getPlugin().call<PiApiKind::piContextRetain>(MContext);
    return pi::cast<cl_context>(MContext);
  }
  throw invalid_object_error(
      "This instance of context doesn't support OpenCL interoperability.",
      PI_INVALID_CONTEXT);
}

bool context_impl::is_host() const { return MHostContext || !MPluginInterop; }

context_impl::~context_impl() {
  if (MPluginInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    getPlugin().call<PiApiKind::piContextRelease>(MContext);
  }
  for (auto LibProg : MCachedLibPrograms) {
    assert(LibProg.second && "Null program must not be kept in the cache");
    getPlugin().call<PiApiKind::piProgramRelease>(LibProg.second);
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

bool
context_impl::hasDevice(shared_ptr_class<detail::device_impl> Device) const {
  for (auto D : MDevices)
    if (getSyclObjImpl(D) == Device)
      return true;
  return false;
}

pi_native_handle context_impl::getNative() const {
  auto Plugin = getPlugin();
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextContextGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
