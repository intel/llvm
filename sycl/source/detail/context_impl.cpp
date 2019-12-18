//==---------------- context_impl.cpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl/detail/clusm.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/context_info.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler)
    : MAsyncHandler(AsyncHandler), MDevices(1, Device), MContext(nullptr),
      MPlatform(), MPluginInterop(false), MHostContext(true) {}

context_impl::context_impl(const vector_class<cl::sycl::device> Devices,
                           async_handler AsyncHandler)
    : MAsyncHandler(AsyncHandler), MDevices(Devices), MContext(nullptr),
      MPlatform(), MPluginInterop(true), MHostContext(false) {
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  vector_class<RT::PiDevice> DeviceIds;
  for (const auto &D : MDevices) {
    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  PI_CALL(piContextCreate)(nullptr, DeviceIds.size(), DeviceIds.data(), nullptr,
                           nullptr, &MContext);
}

context_impl::context_impl(RT::PiContext PiContext, async_handler AsyncHandler)
    : MAsyncHandler(AsyncHandler), MDevices(), MContext(PiContext), MPlatform(),
      MPluginInterop(true), MHostContext(false) {

  vector_class<RT::PiDevice> DeviceIds;
  size_t DevicesNum = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piContextGetInfo)(MContext, PI_CONTEXT_INFO_NUM_DEVICES,
                            sizeof(DevicesNum), &DevicesNum, nullptr);
  DeviceIds.resize(DevicesNum);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piContextGetInfo)(MContext, PI_CONTEXT_INFO_DEVICES,
                            sizeof(RT::PiDevice) * DevicesNum, &DeviceIds[0],
                            nullptr);

  for (auto Dev : DeviceIds) {
    MDevices.emplace_back(
        createSyclObjFromImpl<device>(std::make_shared<device_impl>(Dev)));
  }
  // TODO What if m_Devices if empty? m_Devices[0].get_platform()
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(piContextRetain)(MContext);
}

cl_context context_impl::get() const {
  if (MPluginInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(piContextRetain)(MContext);
    return pi::cast<cl_context>(MContext);
  }
  throw invalid_object_error(
      "This instance of context doesn't support OpenCL interoperability.");
}

bool context_impl::is_host() const { return MHostContext || !MPluginInterop; }

context_impl::~context_impl() {
  if (MPluginInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(piContextRelease)(MContext);
  }
  // Release all programs and kernels created with this context
  for (auto ProgIt : MCachedPrograms) {
    RT::PiProgram ToBeDeleted = ProgIt.second;
    for (auto KernIt : MCachedKernels[ToBeDeleted])
      PI_CALL(piKernelRelease)(KernIt.second);
    PI_CALL(piProgramRelease)(ToBeDeleted);
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
      this->getHandleRef());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}
template <>
vector_class<cl::sycl::device>
context_impl::get_info<info::context::devices>() const {
  return MDevices;
}

RT::PiContext &context_impl::getHandleRef() { return MContext; }
const RT::PiContext &context_impl::getHandleRef() const { return MContext; }

} // namespace detail
} // namespace sycl
} // namespace cl
