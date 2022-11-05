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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

kernel_impl::kernel_impl(RT::PiKernel Kernel, ContextImplPtr Context,
                         KernelBundleImplPtr KernelBundleImpl)
    : kernel_impl(Kernel, Context,
                  std::make_shared<program_impl>(Context, Kernel),
                  /*IsCreatedFromSource*/ true, KernelBundleImpl) {
  // Enable USM indirect access for interoperability kernels.
  // Some PI Plugins (like OpenCL) require this call to enable USM
  // For others, PI will turn this into a NOP.
  getPlugin().call<PiApiKind::piKernelSetExecInfo>(
      MKernel, PI_USM_INDIRECT_ACCESS, sizeof(pi_bool), &PI_TRUE);

  // This constructor is only called in the interoperability kernel constructor.
  MIsInterop = true;
}

kernel_impl::kernel_impl(RT::PiKernel Kernel, ContextImplPtr ContextImpl,
                         ProgramImplPtr ProgramImpl, bool IsCreatedFromSource,
                         KernelBundleImplPtr KernelBundleImpl)
    : MKernel(Kernel), MContext(ContextImpl),
      MProgramImpl(std::move(ProgramImpl)),
      MCreatedFromSource(IsCreatedFromSource),
      MKernelBundleImpl(std::move(KernelBundleImpl)) {

  RT::PiContext Context = nullptr;
  // Using the plugin from the passed ContextImpl
  getPlugin().call<PiApiKind::piKernelGetInfo>(
      MKernel, PI_KERNEL_INFO_CONTEXT, sizeof(Context), &Context, nullptr);
  if (ContextImpl->getHandleRef() != Context)
    throw sycl::invalid_parameter_error(
        "Input context must be the same as the context of cl_kernel",
        PI_ERROR_INVALID_CONTEXT);

  MIsInterop = MProgramImpl->isInterop();
}

kernel_impl::kernel_impl(RT::PiKernel Kernel, ContextImplPtr ContextImpl,
                         DeviceImageImplPtr DeviceImageImpl,
                         KernelBundleImplPtr KernelBundleImpl)
    : MKernel(Kernel), MContext(std::move(ContextImpl)), MProgramImpl(nullptr),
      MCreatedFromSource(false), MDeviceImageImpl(std::move(DeviceImageImpl)),
      MKernelBundleImpl(std::move(KernelBundleImpl)) {

  // kernel_impl shared ownership of kernel handle
  if (!is_host()) {
    getPlugin().call<PiApiKind::piKernelRetain>(MKernel);
  }

  MIsInterop = MKernelBundleImpl->isInterop();
}

kernel_impl::kernel_impl(ContextImplPtr Context, ProgramImplPtr ProgramImpl)
    : MContext(Context), MProgramImpl(std::move(ProgramImpl)) {}

kernel_impl::~kernel_impl() {
  // TODO catch an exception and put it to list of asynchronous exceptions
  if (!is_host()) {
    getPlugin().call<PiApiKind::piKernelRelease>(MKernel);
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

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
