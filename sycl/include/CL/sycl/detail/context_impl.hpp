//==---------------- context_impl.hpp - SYCL context -----------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
#include <memory>

namespace cl {
namespace sycl {
// Forward declaration
class device;
namespace detail {
class context_impl {
public:
  /// Constructs a context_impl using a single SYCL devices.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// @param Device is an instance of SYCL device.
  /// @param AsyncHandler is an instance of async_handler.
  context_impl(const device &Device, async_handler AsyncHandler);

  /// Constructs a context_impl using a list of SYCL devices.
  ///
  /// Newly created instance will save each SYCL device in the list. This
  /// requres that all devices in the list are associated with the same
  /// SYCL platform.
  /// The constructed context_impl will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// @param DeviceList is a list of SYCL device instances.
  /// @param AsyncHandler is an instance of async_handler.
  context_impl(const vector_class<cl::sycl::device> Devices,
               async_handler AsyncHandler);

  /// Construct a context_impl using plug-in interoperability handle.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to handle
  /// exceptions.
  ///
  /// @param PiContext is an instance of a valid plug-in context handle.
  /// @param AsyncHandler is an instance of async_handler.
  context_impl(RT::PiContext PiContext, async_handler AsyncHandler);

  ~context_impl();

  /// Gets OpenCL interoperability context handle.
  ///
  /// @return an instance of OpenCL cl_context.
  cl_context get() const;

  /// Checks if this context is a host context.
  ///
  /// @return true if this context is a host context.
  bool is_host() const;

  /// Gets asynchronous exception handler.
  ///
  /// @return an instance of SYCL async_handler.
  const async_handler &get_async_handler() const;

  /// Queries this context for information.
  ///
  /// The return type depends on information being queried.
  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const;

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// @return an instance of raw plug-in context handle.
  RT::PiContext &getHandleRef();

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// @return an instance of raw plug-in context handle.
  const RT::PiContext &getHandleRef() const;

  /// Gets cached programs.
  ///
  /// @return a map of cached programs.
  std::map<KernelSetId, RT::PiProgram> &getCachedPrograms() {
    return MCachedPrograms;
  }

  /// Gets cached kernels.
  ///
  /// @return a map of cached kernels.
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> &
  getCachedKernels() {
    return MCachedKernels;
  }

private:
  async_handler MAsyncHandler;
  vector_class<device> MDevices;
  RT::PiContext MContext;
  platform MPlatform;
  bool MPluginInterop;
  bool MHostContext;
  std::map<KernelSetId, RT::PiProgram> MCachedPrograms;
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> MCachedKernels;
};

} // namespace detail
} // namespace sycl
} // namespace cl
