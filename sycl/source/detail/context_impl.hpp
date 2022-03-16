//==---------------- context_impl.hpp - SYCL context ------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_program_cache.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <map>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
class device;
namespace detail {
using PlatformImplPtr = std::shared_ptr<detail::platform_impl>;
class context_impl {
public:
  /// Constructs a context_impl using a single SYCL devices.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  /// PropList carries the properties of the constructed context_impl.
  ///
  /// \param Device is an instance of SYCL device.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  context_impl(const device &Device, async_handler AsyncHandler,
               const property_list &PropList);

  /// Constructs a context_impl using a list of SYCL devices.
  ///
  /// Newly created instance will save each SYCL device in the list. This
  /// requres that all devices in the list are associated with the same
  /// SYCL platform.
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  /// PropList carries the properties of the constructed context_impl.
  ///
  /// \param DeviceList is a list of SYCL device instances.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  context_impl(const std::vector<cl::sycl::device> DeviceList,
               async_handler AsyncHandler, const property_list &PropList);

  /// Construct a context_impl using plug-in interoperability handle.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  ///
  /// \param PiContext is an instance of a valid plug-in context handle.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param Plugin is the reference to the underlying Plugin that this
  /// context is associated with.
  context_impl(RT::PiContext PiContext, async_handler AsyncHandler,
               const plugin &Plugin);

  ~context_impl();

  /// Checks if this context_impl has a property of type propertyT.
  ///
  /// \return true if this context_impl has a property of type propertyT.
  template <typename propertyT> bool has_property() const {
    return MPropList.has_property<propertyT>();
  }

  /// Gets the specified property of this context_impl.
  ///
  /// Throws invalid_object_error if this context_impl does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  /// Gets OpenCL interoperability context handle.
  ///
  /// \return an instance of OpenCL cl_context.
  cl_context get() const;

  /// Checks if this context is a host context.
  ///
  /// \return true if this context is a host context.
  bool is_host() const;

  /// Gets asynchronous exception handler.
  ///
  /// \return an instance of SYCL async_handler.
  const async_handler &get_async_handler() const;

  /// \return the Plugin associated with the platform of this context.
  const plugin &getPlugin() const { return MPlatform->getPlugin(); }

  /// \return the PlatformImpl associated with this context.
  PlatformImplPtr getPlatformImpl() const { return MPlatform; }

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
  /// \return an instance of raw plug-in context handle.
  RT::PiContext &getHandleRef();

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// \return an instance of raw plug-in context handle.
  const RT::PiContext &getHandleRef() const;

  /// Unlike `get_info<info::context::devices>', this function returns a
  /// reference.
  const std::vector<device> &getDevices() const { return MDevices; }

  /// In contrast to user programs, which are compiled from user code, library
  /// programs come from the SYCL runtime. They are identified by the
  /// corresponding extension:
  ///
  ///  cl_intel_devicelib_assert -> #<pi_program with assert functions>
  ///  cl_intel_devicelib_complex -> #<pi_program with complex functions>
  ///  etc.
  ///
  /// See `doc/design/DeviceLibExtensions.rst' for
  /// more details.
  ///
  /// \returns a map with device library programs.
  std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram> &
  getCachedLibPrograms() {
    return MCachedLibPrograms;
  }

  KernelProgramCache &getKernelProgramCache() const;

  /// Returns true if and only if context contains the given device.
  bool hasDevice(std::shared_ptr<detail::device_impl> Device) const;

  /// Gets the native handle of the SYCL context.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

private:
  async_handler MAsyncHandler;
  std::vector<device> MDevices;
  RT::PiContext MContext;
  PlatformImplPtr MPlatform;
  property_list MPropList;
  bool MHostContext;
  std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
      MCachedLibPrograms;
  mutable KernelProgramCache MKernelProgramCache;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
