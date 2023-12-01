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
#include <detail/program_impl.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

kernel_impl::kernel_impl(sycl::detail::pi::PiKernel Kernel,
                         ContextImplPtr Context,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask)
    : kernel_impl(Kernel, Context,
                  std::make_shared<program_impl>(Context, Kernel),
                  /*IsCreatedFromSource*/ true, KernelBundleImpl, ArgMask) {
  // Enable USM indirect access for interoperability kernels.
  // Some PI Plugins (like OpenCL) require this call to enable USM
  // For others, PI will turn this into a NOP.
  getPlugin()->call<PiApiKind::piKernelSetExecInfo>(
      MKernel, PI_USM_INDIRECT_ACCESS, sizeof(pi_bool), &PI_TRUE);

  // This constructor is only called in the interoperability kernel constructor.
  MIsInterop = true;
}

kernel_impl::kernel_impl(sycl::detail::pi::PiKernel Kernel,
                         ContextImplPtr ContextImpl, ProgramImplPtr ProgramImpl,
                         bool IsCreatedFromSource,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask)
    : MKernel(Kernel), MContext(ContextImpl),
      MProgramImpl(std::move(ProgramImpl)),
      MCreatedFromSource(IsCreatedFromSource),
      MKernelBundleImpl(std::move(KernelBundleImpl)),
      MKernelArgMaskPtr{ArgMask} {

  sycl::detail::pi::PiContext Context = nullptr;
  // Using the plugin from the passed ContextImpl
  getPlugin()->call<PiApiKind::piKernelGetInfo>(
      MKernel, PI_KERNEL_INFO_CONTEXT, sizeof(Context), &Context, nullptr);
  if (ContextImpl->getHandleRef() != Context)
    throw sycl::invalid_parameter_error(
        "Input context must be the same as the context of cl_kernel",
        PI_ERROR_INVALID_CONTEXT);

  MIsInterop = MProgramImpl->isInterop();
}

kernel_impl::kernel_impl(sycl::detail::pi::PiKernel Kernel,
                         ContextImplPtr ContextImpl,
                         DeviceImageImplPtr DeviceImageImpl,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask, std::mutex *CacheMutex)
    : MKernel(Kernel), MContext(std::move(ContextImpl)), MProgramImpl(nullptr),
      MCreatedFromSource(false), MDeviceImageImpl(std::move(DeviceImageImpl)),
      MKernelBundleImpl(std::move(KernelBundleImpl)),
      MKernelArgMaskPtr{ArgMask}, MCacheMutex{CacheMutex} {
  MIsInterop = MKernelBundleImpl->isInterop();
}

kernel_impl::kernel_impl(ContextImplPtr Context, ProgramImplPtr ProgramImpl)
    : MContext(Context), MProgramImpl(std::move(ProgramImpl)) {}

kernel_impl::~kernel_impl() {
  // TODO catch an exception and put it to list of asynchronous exceptions
  if (!is_host()) {
    getPlugin()->call<PiApiKind::piKernelRelease>(MKernel);
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

} // namespace detail
} // namespace _V1
} // namespace sycl
