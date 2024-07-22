//==------- kernel_impl.cpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

kernel_impl::kernel_impl(sycl::detail::pi::PiKernel Kernel,
                         ContextImplPtr ContextImpl,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask)
    : MKernel(Kernel), MContext(ContextImpl),
      MProgram(ProgramManager::getInstance().getPiProgramFromPiKernel(
          Kernel, ContextImpl)),
      MCreatedFromSource(true), MKernelBundleImpl(std::move(KernelBundleImpl)),
      MIsInterop(true), MKernelArgMaskPtr{ArgMask} {
  sycl::detail::pi::PiContext Context = nullptr;
  // Using the plugin from the passed ContextImpl
  getPlugin()->call<PiApiKind::piKernelGetInfo>(
      MKernel, PI_KERNEL_INFO_CONTEXT, sizeof(Context), &Context, nullptr);
  if (ContextImpl->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_kernel");

  // Enable USM indirect access for interoperability kernels.
  // Some PI Plugins (like OpenCL) require this call to enable USM
  // For others, PI will turn this into a NOP.
  if (ContextImpl->getPlatformImpl()->supports_usm())
    getPlugin()->call<PiApiKind::piKernelSetExecInfo>(
        MKernel, PI_USM_INDIRECT_ACCESS, sizeof(pi_bool), &PI_TRUE);
}

kernel_impl::kernel_impl(sycl::detail::pi::PiKernel Kernel,
                         ContextImplPtr ContextImpl,
                         DeviceImageImplPtr DeviceImageImpl,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask, PiProgram ProgramPI,
                         std::mutex *CacheMutex)
    : MKernel(Kernel), MContext(std::move(ContextImpl)), MProgram(ProgramPI),
      MCreatedFromSource(false), MDeviceImageImpl(std::move(DeviceImageImpl)),
      MKernelBundleImpl(std::move(KernelBundleImpl)),
      MKernelArgMaskPtr{ArgMask}, MCacheMutex{CacheMutex} {
  MIsInterop = MKernelBundleImpl->isInterop();
}

kernel_impl::~kernel_impl() {
  try {
    // TODO catch an exception and put it to list of asynchronous exceptions
    getPlugin()->call<PiApiKind::piKernelRelease>(MKernel);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~kernel_impl", e);
  }
}

bool kernel_impl::isCreatedFromSource() const {
  // TODO it is not clear how to understand whether the SYCL kernel is created
  // from source code or not when the SYCL kernel is created using
  // the interoperability constructor.
  // Here a strange case which does not work now:
  // context Context;
  // program Program(Context);
  // Program.build_with_kernel_type<class A>();
  // kernel FirstKernel= Program.get_kernel<class A>();
  // cl_kernel ClKernel = FirstKernel.get();
  // kernel SecondKernel = kernel(ClKernel, Context);
  // clReleaseKernel(ClKernel);
  // FirstKernel.isCreatedFromSource() != FirstKernel.isCreatedFromSource();
  return MCreatedFromSource;
}

bool kernel_impl::isBuiltInKernel(const device &Device) const {
  auto BuiltInKernels = Device.get_info<info::device::built_in_kernel_ids>();
  if (BuiltInKernels.empty())
    return false;
  std::string KernelName = get_info<info::kernel::function_name>();
  return (std::any_of(
      BuiltInKernels.begin(), BuiltInKernels.end(),
      [&KernelName](kernel_id &Id) { return Id.get_name() == KernelName; }));
}

void kernel_impl::checkIfValidForNumArgsInfoQuery() const {
  if (MKernelBundleImpl->isInterop())
    return;
  auto Devices = MKernelBundleImpl->get_devices();
  if (std::any_of(Devices.begin(), Devices.end(),
                  [this](device &Device) { return isBuiltInKernel(Device); }))
    return;

  throw sycl::exception(
      sycl::make_error_code(errc::invalid),
      "info::kernel::num_args descriptor may only be used to query a kernel "
      "that resides in a kernel bundle constructed using a backend specific"
      "interoperability function or to query a device built-in kernel");
}

template <>
typename info::platform::version::return_type
kernel_impl::get_backend_info<info::platform::version>() const {
  if (MContext->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  auto Devices = MKernelBundleImpl->get_devices();
  return Devices[0].get_platform().get_info<info::platform::version>();
}

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices);

template <>
typename info::device::version::return_type
kernel_impl::get_backend_info<info::device::version>() const {
  if (MContext->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  auto Devices = MKernelBundleImpl->get_devices();
  if (Devices.empty()) {
    return "No available device";
  }
  // Use default selector to pick a device.
  return select_device(default_selector_v, Devices)
      .get_info<info::device::version>();
}

template <>
typename info::device::backend_version::return_type
kernel_impl::get_backend_info<info::device::backend_version>() const {
  if (MContext->getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}

} // namespace detail
} // namespace _V1
} // namespace sycl
