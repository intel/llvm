//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/kernel_info.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/info/info_desc.hpp>

#include <cassert>
#include <memory>

namespace sycl {
inline namespace _V1 {
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
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(sycl::detail::pi::PiKernel Kernel, ContextImplPtr Context,
              KernelBundleImplPtr KernelBundleImpl,
              const KernelArgMask *ArgMask = nullptr);

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
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(sycl::detail::pi::PiKernel Kernel, ContextImplPtr ContextImpl,
              ProgramImplPtr ProgramImpl, bool IsCreatedFromSource,
              KernelBundleImplPtr KernelBundleImpl,
              const KernelArgMask *ArgMask);

  /// Constructs a SYCL kernel_impl instance from a SYCL device_image,
  /// kernel_bundle and / PiKernel.
  ///
  /// \param Kernel is a valid PiKernel instance
  /// \param ContextImpl is a valid SYCL context
  /// \param KernelBundleImpl is a valid instance of kernel_bundle_impl
  kernel_impl(sycl::detail::pi::PiKernel Kernel, ContextImplPtr ContextImpl,
              DeviceImageImplPtr DeviceImageImpl,
              KernelBundleImplPtr KernelBundleImpl,
              const KernelArgMask *ArgMask, std::mutex *CacheMutex);

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
          PI_ERROR_INVALID_KERNEL);
    }
    getPlugin()->call<PiApiKind::piKernelRetain>(MKernel);
    return pi::cast<cl_kernel>(MKernel);
  }

  /// Check if the associated SYCL context is a SYCL host context.
  ///
  /// \return true if this SYCL kernel is a host kernel.
  bool is_host() const { return MContext->is_host(); }

  const PluginPtr &getPlugin() const { return MContext->getPlugin(); }

  /// Query information from the kernel object using the info::kernel_info
  /// descriptor.
  ///
  /// \return depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Query device-specific information from a kernel object using the
  /// info::kernel_device_specific descriptor.
  ///
  /// \param Device is a valid SYCL device to query info for.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_info(const device &Device) const;

  /// Query device-specific information from a kernel using the
  /// info::kernel_device_specific descriptor for a specific device and value.
  /// max_sub_group_size is the only valid descriptor for this function.
  ///
  /// \param Device is a valid SYCL device.
  /// \param WGSize is the work-group size the sub-group size is requested for.
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_info(const device &Device,
                                       const range<3> &WGSize) const;

  /// Get a reference to a raw kernel object.
  ///
  /// \return a reference to a valid PiKernel instance with raw kernel object.
  sycl::detail::pi::PiKernel &getHandleRef() { return MKernel; }
  /// Get a constant reference to a raw kernel object.
  ///
  /// \return a constant reference to a valid PiKernel instance with raw
  /// kernel object.
  const sycl::detail::pi::PiKernel &getHandleRef() const { return MKernel; }

  /// Check if kernel was created from a program that had been created from
  /// source.
  ///
  /// \return true if kernel was created from source.
  bool isCreatedFromSource() const;

  const DeviceImageImplPtr &getDeviceImage() const { return MDeviceImageImpl; }

  pi_native_handle getNative() const {
    const PluginPtr &Plugin = MContext->getPlugin();

    if (MContext->getBackend() == backend::opencl)
      Plugin->call<PiApiKind::piKernelRetain>(MKernel);

    pi_native_handle NativeKernel = 0;
    Plugin->call<PiApiKind::piextKernelGetNativeHandle>(MKernel, &NativeKernel);

    return NativeKernel;
  }

  KernelBundleImplPtr get_kernel_bundle() const { return MKernelBundleImpl; }

  bool isInterop() const { return MIsInterop; }

  ProgramImplPtr getProgramImpl() const { return MProgramImpl; }
  ContextImplPtr getContextImplPtr() const { return MContext; }

  std::mutex &getNoncacheableEnqueueMutex() {
    return MNoncacheableEnqueueMutex;
  }

  const KernelArgMask *getKernelArgMask() const { return MKernelArgMaskPtr; }
  std::mutex *getCacheMutex() const { return MCacheMutex; }

private:
  sycl::detail::pi::PiKernel MKernel;
  const ContextImplPtr MContext;
  const ProgramImplPtr MProgramImpl;
  bool MCreatedFromSource = true;
  const DeviceImageImplPtr MDeviceImageImpl;
  const KernelBundleImplPtr MKernelBundleImpl;
  bool MIsInterop = false;
  std::mutex MNoncacheableEnqueueMutex;
  const KernelArgMask *MKernelArgMaskPtr;
  std::mutex *MCacheMutex = nullptr;

  bool isBuiltInKernel(const device &Device) const;
  void checkIfValidForNumArgsInfoQuery() const;
};

template <typename Param>
inline typename Param::return_type kernel_impl::get_info() const {
  static_assert(is_kernel_info_desc<Param>::value,
                "Invalid kernel information descriptor");
  if (is_host()) {
    // TODO implement
    assert(0 && "Not implemented");
  }

  if constexpr (std::is_same_v<Param, info::kernel::num_args>)
    checkIfValidForNumArgsInfoQuery();

  return get_kernel_info<Param>(this->getHandleRef(), getPlugin());
}

template <>
inline context kernel_impl::get_info<info::kernel::context>() const {
  return createSyclObjFromImpl<context>(MContext);
}

template <typename Param>
inline typename Param::return_type
kernel_impl::get_info(const device &Device) const {
  if constexpr (std::is_same_v<
                    Param, info::kernel_device_specific::global_work_size>) {
    bool isDeviceCustom = Device.get_info<info::device::device_type>() ==
                          info::device_type::custom;
    if (!isDeviceCustom && !isBuiltInKernel(Device))
      throw exception(
          sycl::make_error_code(errc::invalid),
          "info::kernel_device_specific::global_work_size descriptor may only "
          "be used if the device type is device_type::custom or if the kernel "
          "is a built-in kernel.");
  }

  if (is_host()) {
    return get_kernel_device_specific_info_host<Param>(Device);
  }
  return get_kernel_device_specific_info<Param>(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(),
      getPlugin());
}

template <typename Param>
inline typename Param::return_type
kernel_impl::get_info(const device &Device,
                      const sycl::range<3> &WGSize) const {
  if (is_host()) {
    throw runtime_error("Sub-group feature is not supported on HOST device.",
                        PI_ERROR_INVALID_DEVICE);
  }
  return get_kernel_device_specific_info_with_input<Param>(
      this->getHandleRef(), getSyclObjImpl(Device)->getHandleRef(), WGSize,
      getPlugin());
}

} // namespace detail
} // namespace _V1
} // namespace sycl
