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

kernel_impl::kernel_impl(ur_kernel_handle_t Kernel, ContextImplPtr Context,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask)
    : MKernel(Kernel), MContext(Context),
      MProgram(ProgramManager::getInstance().getUrProgramFromUrKernel(Kernel,
                                                                      Context)),
      MCreatedFromSource(true), MKernelBundleImpl(std::move(KernelBundleImpl)),
      MIsInterop(true), MKernelArgMaskPtr{ArgMask} {
  ur_context_handle_t UrContext = nullptr;
  // Using the adapter from the passed ContextImpl
  getAdapter()->call<UrApiKind::urKernelGetInfo>(
      MKernel, UR_KERNEL_INFO_CONTEXT, sizeof(UrContext), &UrContext, nullptr);
  if (Context->getHandleRef() != UrContext)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_kernel");

  // Enable USM indirect access for interoperability kernels.
  // Some UR Adapters (like OpenCL) require this call to enable USM
  // For others, UR will turn this into a NOP.
  if (Context->getPlatformImpl()->supports_usm()) {
    bool EnableAccess = true;
    getAdapter()->call<UrApiKind::urKernelSetExecInfo>(
        MKernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
        nullptr, &EnableAccess);
  }
}

kernel_impl::kernel_impl(ur_kernel_handle_t Kernel, ContextImplPtr ContextImpl,
                         DeviceImageImplPtr DeviceImageImpl,
                         KernelBundleImplPtr KernelBundleImpl,
                         const KernelArgMask *ArgMask,
                         ur_program_handle_t Program, std::mutex *CacheMutex)
    : MKernel(Kernel), MContext(std::move(ContextImpl)), MProgram(Program),
      MCreatedFromSource(false), MDeviceImageImpl(std::move(DeviceImageImpl)),
      MKernelBundleImpl(std::move(KernelBundleImpl)),
      MKernelArgMaskPtr{ArgMask}, MCacheMutex{CacheMutex} {
  MIsInterop = MKernelBundleImpl->isInterop();
}

kernel_impl::~kernel_impl() {
  try {
    // TODO catch an exception and put it to list of asynchronous exceptions
    getAdapter()->call<UrApiKind::urKernelRelease>(MKernel);
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

bool kernel_impl::exceedsOccupancyResourceLimits(
    const device &Device, const range<3> &WorkGroupSize,
    size_t DynamicLocalMemorySize) const {
  // Respect occupancy limits for WorkGroupSize and DynamicLocalMemorySize.
  // Generally, exceeding hardware resource limits will yield in an error when
  // the kernel is launched.
  const size_t MaxWorkGroupSize =
      get_info<info::kernel_device_specific::work_group_size>(Device);
  const size_t MaxLocalMemorySizeInBytes =
      Device.get_info<info::device::local_mem_size>();

  if (WorkGroupSize.size() > MaxWorkGroupSize)
    return true;

  if (DynamicLocalMemorySize > MaxLocalMemorySizeInBytes)
    return true;

  // It will be impossible to launch a kernel for Cuda when the hardware limit
  // for the 32-bit registers page file size is exceeded.
  if (Device.get_backend() == backend::ext_oneapi_cuda) {
    const uint32_t RegsPerWorkItem =
        get_info<info::kernel_device_specific::ext_codeplay_num_regs>(Device);
    const uint32_t MaxRegsPerWorkGroup =
        Device.get_info<ext::codeplay::experimental::info::device::
                            max_registers_per_work_group>();
    if ((MaxWorkGroupSize * RegsPerWorkItem) > MaxRegsPerWorkGroup)
      return true;
  }

  return false;
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
