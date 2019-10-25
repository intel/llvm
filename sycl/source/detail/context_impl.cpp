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
    : m_AsyncHandler(AsyncHandler), m_Devices(1, Device), m_Context(nullptr),
      m_Platform(), m_OpenCLInterop(false), m_HostContext(true) {}

context_impl::context_impl(const vector_class<cl::sycl::device> Devices,
                           async_handler AsyncHandler)
    : m_AsyncHandler(AsyncHandler), m_Devices(Devices), m_Context(nullptr),
      m_Platform(), m_OpenCLInterop(true), m_HostContext(false) {
  m_Platform = m_Devices[0].get_platform();
  vector_class<RT::PiDevice> DeviceIds;
  for (const auto &D : m_Devices) {
    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  PI_CALL(
      RT::piContextCreate(0, DeviceIds.size(), DeviceIds.data(), 0, 0, &m_Context));

  m_USMDispatch.reset(new usm::USMDispatcher(m_Platform.get(), DeviceIds));
}

context_impl::context_impl(cl_context ClContext, async_handler AsyncHandler)
    : m_AsyncHandler(AsyncHandler), m_Devices(),
      m_Platform(), m_OpenCLInterop(true), m_HostContext(false) {

  m_Context = pi::cast<RT::PiContext>(ClContext);
  vector_class<RT::PiDevice> DeviceIds;
  size_t DevicesNum = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(RT::piContextGetInfo(m_Context, PI_CONTEXT_INFO_NUM_DEVICES,
                               sizeof(DevicesNum), &DevicesNum, nullptr));
  DeviceIds.resize(DevicesNum);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(RT::piContextGetInfo(
      m_Context, PI_CONTEXT_INFO_DEVICES,
      sizeof(RT::PiDevice) * DevicesNum, &DeviceIds[0], nullptr));

  for (auto Dev : DeviceIds) {
    m_Devices.emplace_back(
        createSyclObjFromImpl<device>(std::make_shared<device_impl>(Dev)));
  }
  // TODO What if m_Devices if empty? m_Devices[0].get_platform()
  m_Platform = platform(m_Devices[0].get_platform());
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(RT::piContextRetain(m_Context));
}

cl_context context_impl::get() const {
  if (m_OpenCLInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piContextRetain(m_Context));
    return pi::cast<cl_context>(m_Context);
  }
  throw invalid_object_error(
      "This instance of context doesn't support OpenCL interoperability.");
}

bool context_impl::is_host() const { return m_HostContext || !m_OpenCLInterop; }
platform context_impl::get_platform() const { return m_Platform; }
vector_class<device> context_impl::get_devices() const { return m_Devices; }

context_impl::~context_impl() {
  if (m_OpenCLInterop) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piContextRelease(m_Context));
  }
  // Release all programs and kernels created with this context
  for (auto ProgIt : m_CachedPrograms) {
    RT::PiProgram ToBeDeleted = ProgIt.second;
    for (auto KernIt : m_CachedKernels[ToBeDeleted]) {
      PI_CALL(RT::piKernelRelease(KernIt.second));
    }
    PI_CALL(RT::piProgramRelease(ToBeDeleted));
  }
}

const async_handler &context_impl::get_async_handler() const {
  return m_AsyncHandler;
}

template <>
cl_uint context_impl::get_info<info::context::reference_count>() const {
  if (is_host()) {
    return 0;
  }
  return get_context_info<info::context::reference_count>::_(
      this->getHandleRef());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  return get_platform();
}
template <>
vector_class<cl::sycl::device>
context_impl::get_info<info::context::devices>() const {
  return get_devices();
}

RT::PiContext &context_impl::getHandleRef() { return m_Context; }
const RT::PiContext &context_impl::getHandleRef() const { return m_Context; }

std::shared_ptr<usm::USMDispatcher> context_impl::getUSMDispatch() const {
  return m_USMDispatch;
}

} // namespace detail
} // namespace sycl
} // namespace cl
