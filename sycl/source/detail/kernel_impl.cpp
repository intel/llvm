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

static CompileTimeKernelInfoTy
createCompileTimeKernelInfo(std::string_view KernelName = {}) {
  return CompileTimeKernelInfoTy{KernelName};
}

kernel_impl::kernel_impl(Managed<ur_kernel_handle_t> &&Kernel,
                         context_impl &Context,
                         kernel_bundle_impl *KernelBundleImpl,
                         const KernelArgMask *ArgMask)
    : MKernel(std::move(Kernel)), MContext(Context.shared_from_this()),
      MProgram(ProgramManager::getInstance().getUrProgramFromUrKernel(MKernel,
                                                                      Context)),
      MCreatedFromSource(true),
      MKernelBundleImpl(KernelBundleImpl ? KernelBundleImpl->shared_from_this()
                                         : nullptr),
      MIsInterop(true), MKernelArgMaskPtr{ArgMask},
      MInteropDeviceKernelInfo(createCompileTimeKernelInfo(getName())) {
  ur_context_handle_t UrContext = nullptr;
  // Using the adapter from the passed ContextImpl
  getAdapter().call<UrApiKind::urKernelGetInfo>(
      MKernel, UR_KERNEL_INFO_CONTEXT, sizeof(UrContext), &UrContext, nullptr);
  if (Context.getHandleRef() != UrContext)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_kernel");

  // Enable USM indirect access for interoperability kernels.
  enableUSMIndirectAccess();
}

kernel_impl::kernel_impl(Managed<ur_kernel_handle_t> &&Kernel,
                         context_impl &ContextImpl,
                         std::shared_ptr<device_image_impl> &&DeviceImageImpl,
                         const kernel_bundle_impl &KernelBundleImpl,
                         const KernelArgMask *ArgMask,
                         ur_program_handle_t Program, std::mutex *CacheMutex)
    : MKernel(std::move(Kernel)), MContext(ContextImpl.shared_from_this()),
      MProgram(Program),
      MCreatedFromSource(DeviceImageImpl->isNonSYCLSourceBased()),
      MDeviceImageImpl(std::move(DeviceImageImpl)),
      MKernelBundleImpl(KernelBundleImpl.shared_from_this()),
      MIsInterop(MDeviceImageImpl->getOriginMask() & ImageOriginInterop),
      MKernelArgMaskPtr{ArgMask}, MCacheMutex{CacheMutex},
      MInteropDeviceKernelInfo(MIsInterop
                                   ? createCompileTimeKernelInfo(getName())
                                   : createCompileTimeKernelInfo()) {
  // Enable USM indirect access for interop and non-sycl-jit source kernels.
  // sycl-jit kernels will enable this if needed through the regular kernel
  // path.
  if (MCreatedFromSource || MIsInterop)
    enableUSMIndirectAccess();
}

#ifdef _MSC_VER
#pragma warning(push)
// https://developercommunity.visualstudio.com/t/False-C4297-warning-while-using-function/1130300
// https://godbolt.org/z/xsMvKf84f
#pragma warning(disable : 4297)
#endif
kernel_impl::~kernel_impl() try {
} catch (std::exception &e) {
  // TODO put it to list of asynchronous exceptions
  __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~kernel_impl", e);
  return; // Don't re-throw.
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

bool kernel_impl::isInteropOrSourceBased() const noexcept {
  return isInterop() ||
         (MDeviceImageImpl &&
          (MDeviceImageImpl->getOriginMask() & ImageOriginKernelCompiler));
}

bool kernel_impl::hasSYCLMetadata() const noexcept {
  return !isInteropOrSourceBased() ||
         (MDeviceImageImpl &&
          MDeviceImageImpl->isFromSourceLanguage(
              sycl::ext::oneapi::experimental::source_language::sycl));
}

// TODO this is how kernel_impl::get_info<function_name> should behave instead.
std::string_view kernel_impl::getName() const {
  if (MName.empty())
    MName = get_info<info::kernel::function_name>();
  return MName;
}

bool kernel_impl::isBuiltInKernel(device_impl &Device) const {
  auto BuiltInKernels = Device.get_info<info::device::built_in_kernel_ids>();
  if (BuiltInKernels.empty())
    return false;
  std::string KernelName = get_info<info::kernel::function_name>();
  return (std::any_of(
      BuiltInKernels.begin(), BuiltInKernels.end(),
      [&KernelName](kernel_id &Id) { return Id.get_name() == KernelName; }));
}

void kernel_impl::checkIfValidForNumArgsInfoQuery() const {
  if (isInteropOrSourceBased())
    return;
  devices_range Devices = MKernelBundleImpl->get_devices();
  if (std::any_of(Devices.begin(), Devices.end(), [this](device_impl &Device) {
        return isBuiltInKernel(Device);
      }))
    return;

  throw sycl::exception(
      sycl::make_error_code(errc::invalid),
      "info::kernel::num_args descriptor may only be used to query a kernel "
      "that resides in a kernel bundle constructed using a backend specific"
      "interoperability function or to query a device built-in kernel");
}

std::optional<unsigned> kernel_impl ::getFreeFuncKernelArgSize() const {
  return MKernelBundleImpl->tryGetKernelArgsSize(getName());
}

void kernel_impl::enableUSMIndirectAccess() const {
  if (!MContext->getPlatformImpl().supports_usm())
    return;

  // Some UR Adapters (like OpenCL) require this call to enable USM
  // For others, UR will turn this into a NOP.
  bool EnableAccess = true;
  getAdapter().call<UrApiKind::urKernelSetExecInfo>(
      MKernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
      nullptr, &EnableAccess);
}

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices);

} // namespace detail
} // namespace _V1
} // namespace sycl
