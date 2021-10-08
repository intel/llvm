//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/program.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_info.hpp>

#include <cassert>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// Forward declaration
class program_impl;
class kernel_bundle_impl;

using ContextImplPtr = std::shared_ptr<context_impl>;
using ProgramImplPtr = std::shared_ptr<program_impl>;
using KernelBundleImplPtr = std::shared_ptr<kernel_bundle_impl>;
class kernel_impl {
public:
  /// Constructs a SYCL kernel instance from a PiKernel
  ///
  /// This constructor is used for plug-in interoperability. It always marks
  /// kernel as being created from source and creates a new program_impl
  /// instance.
  ///
  /// \param Kernel is a valid PiKernel instance
  /// \param Context is a valid SYCL context
  kernel_impl(RT::PiKernel Kernel, ContextImplPtr Context);

  /// Constructs a SYCL kernel instance from a SYCL program and a PiKernel
  ///
  /// This constructor creates a new instance from PiKernel and saves
  /// the provided SYCL program. If context of PiKernel differs from
  /// context of the SYCL program, an invalid_parameter_error exception is
  /// thrown.
  ///
  /// \param Kernel is a valid PiKernel instance
  /// \param ContextImpl is a valid SYCL context
  /// \param ProgramImpl is a valid instance of program_impl
  /// \param IsCreatedFromSource is a flag that indicates whether program
  /// is created from source code
  kernel_impl(RT::PiKernel Kernel, ContextImplPtr ContextImpl,
              ProgramImplPtr ProgramImpl, bool IsCreatedFromSource);

  /// Constructs a SYCL kernel_impl instance from a SYCL device_image,
  /// kernel_bundle and / PiKernel.
  ///
  /// \param Kernel is a valid PiKernel instance
  /// \param ContextImpl is a valid SYCL context
  /// \param ProgramImpl is a valid instance of kernel_bundle_impl
  kernel_impl(RT::PiKernel Kernel, ContextImplPtr ContextImpl,
              DeviceImageImplPtr DeviceImageImpl,
              KernelBundleImplPtr KernelBundleImpl);

  /// Constructs a SYCL kernel for host device
  ///
  /// \param Context is a valid SYCL context
  /// \param ProgramImpl is a valid instance of program_impl
  kernel_impl(ContextImplPtr Context, ProgramImplPtr ProgramImpl);

  // This section means the object is non-movable and non-copyable
  // There is no need of move and copy constructors in kernel_impl.
  // If they need to be added, piKernelRetain method for MKernel
  // should be present.
  kernel_impl(const kernel_impl &) = delete;
  kernel_impl(kernel_impl &&) = delete;
  kernel_impl &operator=(const kernel_impl &) = delete;
  kernel_impl &operator=(kernel_impl &&) = delete;

  ~kernel_impl();

  /// Gets a valid OpenCL kernel handle
  ///
  /// If this kernel encapsulates an instance of OpenCL kernel, a valid
  /// cl_kernel will be returned. If this kernel is a host kernel,
  /// an invalid_object_error exception will be thrown.
  ///
  /// \return a valid cl_kernel instance
  cl_kernel get() const {
    if (is_host()) {
      throw invalid_object_error(
          "This instance of kernel doesn't support OpenCL interoperability.",
          PI_INVALID_KERNEL);
    }
    getPlugin().call<PiApiKind::piKernelRetain>(MKernel);
    return pi::cast<cl_kernel>(MKernel);
  }

  /// Check if the associated SYCL context is a SYCL host context.
  ///
  /// \return true if this SYCL kernel is a host kernel.
  bool is_host() const { return MContext->is_host(); }

  const plugin &getPlugin() const { return MContext->getPlugin(); }

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// \return depends on information being queried.
  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const;

  /// Query device-specific information from a kernel object using the
  /// info::kernel_device_specific descriptor.
  ///
  /// \param Device is a valid SYCL device to query info for.
  /// \return depends on information being queried.
  template <info::kernel_device_specific param>
  typename info::param_traits<info::kernel_device_specific, param>::return_type
  get_info(const device &Device) const;

  /// Query device-specific information from a kernel using the
  /// info::kernel_device_specific descriptor for a specific device and value.
  ///
  /// \param Device is a valid SYCL device.
  /// \param Value depends on information being queried.
  /// \return depends on information being queried.
  template <info::kernel_device_specific param>
  typename info::param_traits<info::kernel_device_specific, param>::return_type
  get_info(const device &Device,
           typename info::param_traits<info::kernel_device_specific,
                                       param>::input_type Value) const;

  /// Query work-group information from a kernel using the
  /// info::kernel_work_group descriptor for a specific device.
  ///
  /// \param Device is a valid SYCL device.
  /// \return depends on information being queried.
  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device.
  ///
  /// \param Device is a valid SYCL device
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const;

  /// Query sub-group information from a kernel using the
  /// info::kernel_sub_group descriptor for a specific device and value.
  ///
  /// \param Device is a valid SYCL device.
  /// \param Value depends on information being queried.
  /// \return depends on information being queried.
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const;

  /// Get a reference to a raw kernel object.
  ///
  /// \return a reference to a valid PiKernel instance with raw kernel object.
  RT::PiKernel &getHandleRef() { return MKernel; }
  /// Get a constant reference to a raw kernel object.
  ///
  /// \return a constant reference to a valid PiKernel instance with raw
  /// kernel object.
  const RT::PiKernel &getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// \return true if kernel was created from source.
  bool isCreatedFromSource() const;

  const DeviceImageImplPtr &getDeviceImage() const { return MDeviceImageImpl; }

  pi_native_handle getNative() const {
    const plugin &Plugin = MContext->getPlugin();

    if (Plugin.getBackend() == backend::opencl)
      Plugin.call<PiApiKind::piKernelRetain>(MKernel);

    pi_native_handle NativeKernel = 0;
    Plugin.call<PiApiKind::piextKernelGetNativeHandle>(MKernel, &NativeKernel);

    return NativeKernel;
  }

  KernelBundleImplPtr get_kernel_bundle() const { return MKernelBundleImpl; }

private:
  RT::PiKernel MKernel;
  const ContextImplPtr MContext;
  const ProgramImplPtr MProgramImpl;
  bool MCreatedFromSource = true;
  const DeviceImageImplPtr MDeviceImageImpl;
  const KernelBundleImplPtr MKernelBundleImpl;
};

template <info::kernel param>
inline typename info::param_traits<info::kernel, param>::return_type
kernel_impl::get_info() const {
  if (is_host()) {
    // TODO implement
    assert(0 && "Not implemented");
  }
  return get_kernel_info<
      typename info::param_traits<info::kernel, param>::return_type,
      param>::get(this->getHandleRef(), getPlugin());
}

template <>
inline context kernel_impl::get_info<info::kernel::context>() const {
  return createSyclObjFromImpl<context>(MContext);
}

template <>
inline program kernel_impl::get_info<info::kernel::program>() const {
  return createSyclObjFromImpl<program>(MProgramImpl);
}

template <info::kernel_device_specific param>
inline typename info::param_traits<info::kernel_device_specific,
                                   param>::return_type
kernel_impl::get_info(const device &Device) const {
  if (is_host()) {
    return get_kernel_device_specific_info_host<param>(Device);
  }
  return get_kernel_device_specific_info<
      typename info::param_traits<info::kernel_device_specific,
                                  param>::return_type,
      param>::get(this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(),
                  getPlugin());
}

template <info::kernel_device_specific param>
inline typename info::param_traits<info::kernel_device_specific,
                                   param>::return_type
kernel_impl::get_info(
    const device &Device,
    typename info::param_traits<info::kernel_device_specific, param>::input_type
        Value) const {
  if (is_host()) {
    throw runtime_error("Sub-group feature is not supported on HOST device.",
                        PI_INVALID_DEVICE);
  }
  return get_kernel_device_specific_info_with_input<param>::get(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(), Value,
      getPlugin());
}

template <info::kernel_work_group param>
inline typename info::param_traits<info::kernel_work_group, param>::return_type
kernel_impl::get_work_group_info(const device &Device) const {
  return get_info<
      info::compatibility_param_traits<info::kernel_work_group, param>::value>(
      Device);
}

template <info::kernel_sub_group param>
inline typename info::param_traits<info::kernel_sub_group, param>::return_type
kernel_impl::get_sub_group_info(const device &Device) const {
  return get_info<
      info::compatibility_param_traits<info::kernel_sub_group, param>::value>(
      Device);
}

template <info::kernel_sub_group param>
inline typename info::param_traits<info::kernel_sub_group, param>::return_type
kernel_impl::get_sub_group_info(
    const device &Device,
    typename info::param_traits<info::kernel_sub_group, param>::input_type
        Value) const {
  return get_info<
      info::compatibility_param_traits<info::kernel_sub_group, param>::value>(
      Device, Value);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
