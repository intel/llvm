//==----- program_impl.cpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_impl.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/pi.h>
#include <sycl/kernel.hpp>
#include <sycl/property_list.hpp>

#include <algorithm>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>

namespace sycl {
inline namespace _V1 {
namespace detail {

program_impl::program_impl(ContextImplPtr Context,
                           const property_list &PropList)
    : program_impl(Context, Context->get_info<info::context::devices>(),
                   PropList) {}

program_impl::program_impl(ContextImplPtr Context,
                           std::vector<device> DeviceList,
                           const property_list &PropList)
    : MContext(Context), MDevices(DeviceList), MPropList(PropList) {
  if (Context->getDevices().size() > 1) {
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "multiple devices within a context are not supported with "
        "sycl::program and sycl::kernel");
  }
}

program_impl::program_impl(
    std::vector<std::shared_ptr<program_impl>> ProgramList,
    std::string LinkOptions, const property_list &PropList)
    : MState(program_state::linked), MPropList(PropList),
      MLinkOptions(LinkOptions), MBuildOptions(LinkOptions) {
  // Verify arguments
  if (ProgramList.empty()) {
    throw runtime_error("Non-empty vector of programs expected",
                        UR_RESULT_ERROR_INVALID_VALUE);
  }

  // Sort the programs to avoid deadlocks due to locking multiple mutexes &
  // verify that all programs are unique.
  std::sort(ProgramList.begin(), ProgramList.end());
  auto It = std::unique(ProgramList.begin(), ProgramList.end());
  if (It != ProgramList.end()) {
    throw runtime_error("Attempting to link a program with itself",
                        UR_RESULT_ERROR_INVALID_PROGRAM);
  }

  MContext = ProgramList[0]->MContext;
  if (MContext->getDevices().size() > 1) {
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "multiple devices within a context are not supported with "
        "sycl::program and sycl::kernel");
  }
  MDevices = ProgramList[0]->MDevices;
  std::vector<device> DevicesSorted;
  if (!is_host()) {
    DevicesSorted = sort_devices_by_cl_device_id(MDevices);
  }
  check_device_feature_support<info::device::is_linker_available>(MDevices);
  std::list<std::lock_guard<std::mutex>> Locks;
  for (const auto &Prg : ProgramList) {
    Locks.emplace_back(Prg->MMutex);
    Prg->throw_if_state_is_not(program_state::compiled);
    if (Prg->MContext != MContext) {
      throw invalid_object_error(
          "Not all programs are associated with the same context",
          UR_RESULT_ERROR_INVALID_PROGRAM);
    }
    if (!is_host()) {
      std::vector<device> PrgDevicesSorted =
          sort_devices_by_cl_device_id(Prg->MDevices);
      if (PrgDevicesSorted != DevicesSorted) {
        throw invalid_object_error(
            "Not all programs are associated with the same devices",
            UR_RESULT_ERROR_INVALID_PROGRAM);
      }
    }
  }

  if (!is_host()) {
    // std::vector<ur_device_handle_t> Devices(get_ur_devices());
    std::vector<ur_program_handle_t> Programs;
    bool NonInterOpToLink = false;
    for (const auto &Prg : ProgramList) {
      if (!Prg->MLinkable && NonInterOpToLink)
        continue;
      NonInterOpToLink |= !Prg->MLinkable;
      Programs.push_back(Prg->MURProgram);
    }
    const PluginPtr &Plugin = getPlugin();
    ur_result_t Err = Plugin->call_nocheck(
        urProgramLink, MContext->getUrHandleRef(), Programs.size(),
        Programs.data(), LinkOptions.c_str(), &MURProgram);
    Plugin->checkUrResult<compile_program_error>(Err);
  }
}

program_impl::program_impl(ContextImplPtr Context,
                           ur_native_handle_t InteropProgram)
    : program_impl(Context, InteropProgram, nullptr) {
  MIsInterop = true;
}

program_impl::program_impl(ContextImplPtr Context,
                           ur_native_handle_t InteropProgram,
                           ur_program_handle_t Program)
    : MURProgram(Program), MContext(Context), MLinkable(true) {
  const PluginPtr &Plugin = getPlugin();
  if (MURProgram == nullptr) {
    assert(InteropProgram &&
           "No InteropProgram/PiProgram defined with piextProgramFromNative");
    // Translate the raw program handle into PI program.
    Plugin->call(urProgramCreateWithNativeHandle, InteropProgram,
                 MContext->getUrHandleRef(), nullptr, &MURProgram);
  } else
    Plugin->call(urProgramRetain, Program);

  // TODO handle the case when cl_program build is in progress
  pi_uint32 NumDevices;
  Plugin->call(urProgramGetInfo, MURProgram, UR_PROGRAM_INFO_NUM_DEVICES,
               sizeof(pi_uint32), &NumDevices, nullptr);
  std::vector<ur_device_handle_t> UrDevices(NumDevices);
  Plugin->call(urProgramGetInfo, MURProgram, UR_PROGRAM_INFO_DEVICES,
               sizeof(ur_device_handle_t) * NumDevices, UrDevices.data(),
               nullptr);

  std::vector<device> PlatformDevices =
      MContext->getPlatformImpl()->get_devices();
  // Keep only the subset of the devices (associated with context) that
  // were actually used to create the program.
  // This is possible when clCreateProgramWithBinary is used.
  auto NewEnd = std::remove_if(
      PlatformDevices.begin(), PlatformDevices.end(),
      [&UrDevices](const sycl::device &Dev) {
        return UrDevices.end() ==
               std::find(UrDevices.begin(), UrDevices.end(),
                         detail::getSyclObjImpl(Dev)->getUrHandleRef());
      });
  PlatformDevices.erase(NewEnd, PlatformDevices.end());
  MDevices = PlatformDevices;
  assert(!MDevices.empty() && "No device found for this program");
  ur_device_handle_t Device = UrDevices[0];
  // TODO check build for each device instead
  ur_program_binary_type_t BinaryType = UR_PROGRAM_BINARY_TYPE_NONE;
  Plugin->call(urProgramGetBuildInfo, MURProgram, Device,
               UR_PROGRAM_BUILD_INFO_BINARY_TYPE,
               sizeof(ur_program_binary_type_t), &BinaryType, nullptr);
  if (BinaryType == UR_PROGRAM_BINARY_TYPE_NONE) {
    throw invalid_object_error(
        "The native program passed to the program constructor has to be either "
        "compiled or linked",
        PI_ERROR_INVALID_PROGRAM);
  }
  size_t Size = 0;
  Plugin->call(urProgramGetBuildInfo,
      MURProgram, Device, UR_PROGRAM_BUILD_INFO_OPTIONS, 0, nullptr, &Size);
  std::vector<char> OptionsVector(Size);
  Plugin->call(urProgramGetBuildInfo, MURProgram, Device,
               UR_PROGRAM_BUILD_INFO_OPTIONS, Size, OptionsVector.data(),
               nullptr);
  std::string Options(OptionsVector.begin(), OptionsVector.end());
  switch (BinaryType) {
  case UR_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
    MState = program_state::compiled;
    MCompileOptions = Options;
    MBuildOptions = Options;
    return;
  case UR_PROGRAM_BINARY_TYPE_LIBRARY:
  case UR_PROGRAM_BINARY_TYPE_EXECUTABLE:
    MState = program_state::linked;
    MLinkOptions = "";
    MBuildOptions = Options;
    return;
  default:
    break;
  }
  assert(false && "BinaryType is invalid.");
}

program_impl::program_impl(ContextImplPtr Context, ur_kernel_handle_t Kernel)
    : program_impl(Context, static_cast<ur_native_handle_t>(nullptr),
                   ProgramManager::getInstance().getUrProgramFromUrKernel(
                       Kernel, Context)) {
  MIsInterop = true;
}

program_impl::~program_impl() {
  // TODO catch an exception and put it to list of asynchronous exceptions
  if (!is_host() && MURProgram != nullptr) {
    const PluginPtr &Plugin = getPlugin();
    Plugin->call(urProgramRelease, MURProgram);
  }
}

cl_program program_impl::get() const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    throw invalid_object_error(
        "This instance of program doesn't support OpenCL interoperability.",
        UR_RESULT_ERROR_INVALID_PROGRAM);
  }
  getPlugin()->call(urProgramRetain, MURProgram);
  ur_native_handle_t nativeHandle = nullptr;
  getPlugin()->call(urProgramGetNativeHandle, MURProgram, &nativeHandle);
  return pi::cast<cl_program>(nativeHandle);
}

void program_impl::compile_with_kernel_name(std::string KernelName,
                                            std::string CompileOptions) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::none);
  if (!is_host()) {
    create_ur_program_with_kernel_name(
        KernelName,
        /*JITCompilationIsRequired=*/(!CompileOptions.empty()));
    compile(CompileOptions);
  }
  MState = program_state::compiled;
}

void program_impl::link(std::string LinkOptions) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::compiled);
  if (!is_host()) {
    check_device_feature_support<info::device::is_linker_available>(MDevices);
    std::vector<ur_device_handle_t> Devices(get_ur_devices());
    const PluginPtr &Plugin = getPlugin();
    const char *LinkOpts = SYCLConfig<SYCL_PROGRAM_LINK_OPTIONS>::get();
    if (!LinkOpts) {
      LinkOpts = LinkOptions.c_str();
    }

    // Plugin resets MProgram with a new pi_program as a result of the call to
    // "piProgramLink". Thus, we need to release MProgram before the call to
    // piProgramLink.
    if (MURProgram != nullptr)
      Plugin->call(urProgramRelease, MURProgram);

    ur_result_t Err = Plugin->call_nocheck(
        urProgramLinkExp, MContext->getUrHandleRef(), Devices.size(),
        Devices.data(),
        /*num_input_programs*/ 1, &MURProgram, LinkOpts, &MURProgram);
    if (Err == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Err = Plugin->call_nocheck(
        urProgramLink, MContext->getUrHandleRef(),
        /*num_input_programs*/ 1, &MURProgram, LinkOpts, &MURProgram);
    }
    Plugin->checkUrResult<compile_program_error>(Err);
    MLinkOptions = LinkOptions;
    MBuildOptions = LinkOptions;
  }
  MState = program_state::linked;
}

bool program_impl::has_kernel(std::string KernelName,
                              bool IsCreatedFromSource) const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    return !IsCreatedFromSource;
  }

  std::vector<ur_device_handle_t> Devices(get_ur_devices());
  void *function_ptr;
  const PluginPtr &Plugin = getPlugin();

  ur_result_t Err = UR_RESULT_SUCCESS;
  for (ur_device_handle_t Device : Devices) {
    Err = Plugin->call_nocheck(urProgramGetFunctionPointer, Device, MURProgram,
                               KernelName.c_str(), &function_ptr);
    if (Err != UR_RESULT_SUCCESS &&
        Err != UR_RESULT_ERROR_INVALID_FUNCTION_NAME &&
        Err != UR_RESULT_ERROR_INVALID_KERNEL_NAME)
      throw runtime_error(
          "Error from urProgramGetFunctionPointer when called by program", Err);
    // TODO: This seems wrong......
    if (Err == UR_RESULT_SUCCESS ||
        Err == UR_RESULT_ERROR_INVALID_FUNCTION_NAME)
      return true;
  }

  return false;
}

kernel program_impl::get_kernel(std::string KernelName,
                                std::shared_ptr<program_impl> PtrToSelf,
                                bool IsCreatedFromSource) const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    if (IsCreatedFromSource)
      throw invalid_object_error("This instance of program is a host instance",
                                 PI_ERROR_INVALID_PROGRAM);

    return createSyclObjFromImpl<kernel>(
        std::make_shared<kernel_impl>(MContext, PtrToSelf));
  }
  auto [Kernel, ArgMask] = get_ur_kernel_arg_mask_pair(KernelName);
  return createSyclObjFromImpl<kernel>(std::make_shared<kernel_impl>(
      Kernel, MContext, PtrToSelf, IsCreatedFromSource, nullptr, ArgMask));
}

std::vector<std::vector<char>> program_impl::get_binaries() const {
  throw_if_state_is(program_state::none);
  if (is_host())
    return {};

  std::vector<std::vector<char>> Result;
  const PluginPtr &Plugin = getPlugin();
  std::vector<size_t> BinarySizes(MDevices.size());
  Plugin->call(urProgramGetInfo,
      MURProgram, UR_PROGRAM_INFO_BINARY_SIZES,
      sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

  std::vector<char *> Pointers;
  for (size_t I = 0; I < BinarySizes.size(); ++I) {
    Result.emplace_back(BinarySizes[I]);
    Pointers.push_back(Result[I].data());
  }
  // TODO: This result isn't used?
  Plugin->call(urProgramGetInfo, MURProgram, UR_PROGRAM_INFO_BINARIES,
               sizeof(char *) * Pointers.size(), Pointers.data(), nullptr);
  return Result;
}

void program_impl::compile(const std::string &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  std::vector<ur_device_handle_t> Devices(get_ur_devices());
  const PluginPtr &Plugin = getPlugin();
  const char *CompileOpts = SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  if (!CompileOpts) {
    CompileOpts = Options.c_str();
  }
  // TODO: Use urProgramCompileExt?
  ur_result_t Err = Plugin->call_nocheck(
      urProgramCompile, MContext->getUrHandleRef(), MURProgram, CompileOpts);

  if (Err != UR_RESULT_SUCCESS) {
    throw compile_program_error(
        "Program compilation error:\n" +
            ProgramManager::getProgramBuildLog(MURProgram, MContext),
        Err);
  }
  MCompileOptions = Options;
  MBuildOptions = Options;
}

void program_impl::build(const std::string &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  std::vector<ur_device_handle_t> Devices(get_ur_devices());
  const PluginPtr &Plugin = getPlugin();
  ProgramManager::getInstance().flushSpecConstants(*this);
  ur_result_t Err =
      Plugin->call_nocheck(urProgramBuildExp, MURProgram, Devices.size(),
                           Devices.data(), Options.c_str());

  if (Err == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    Err = Plugin->call_nocheck(urProgramBuild, MContext->getUrHandleRef(),
                               MURProgram, Options.c_str());
  }

  if (Err != UR_RESULT_SUCCESS) {
    throw compile_program_error(
        "Program build error:\n" +
            ProgramManager::getProgramBuildLog(MURProgram, MContext),
        Err);
  }
  MBuildOptions = Options;
}

std::vector<ur_device_handle_t> program_impl::get_ur_devices() const {
  std::vector<ur_device_handle_t> UrDevices;
  for (const auto &Device : MDevices) {
    UrDevices.push_back(getSyclObjImpl(Device)->getUrHandleRef());
  }
  return UrDevices;
}

std::pair<ur_kernel_handle_t, const KernelArgMask *>
program_impl::get_ur_kernel_arg_mask_pair(const std::string &KernelName) const {
  std::pair<ur_kernel_handle_t, const KernelArgMask *> Result;

  const PluginPtr &Plugin = getPlugin();
  ur_result_t Err = Plugin->call_nocheck(urKernelCreate, MURProgram,
                                         KernelName.c_str(), &Result.first);
  if (Err == UR_RESULT_ERROR_INVALID_KERNEL_NAME) {
    throw invalid_object_error(
        "This instance of program does not contain the kernel requested", Err);
  }
    Plugin->checkUrResult(Err);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    if (getContextImplPtr()->getPlatformImpl()->supports_usm())
      Plugin->call(urKernelSetExecInfo, Result.first,
                   UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
                   nullptr, &PI_TRUE);

    return Result;
}

std::vector<device>
program_impl::sort_devices_by_cl_device_id(std::vector<device> Devices) {
  std::sort(Devices.begin(), Devices.end(),
            [](const device &id1, const device &id2) {
              return (detail::getSyclObjImpl(id1)->getUrHandleRef() <
                      detail::getSyclObjImpl(id2)->getUrHandleRef());
            });
  return Devices;
}

void program_impl::throw_if_state_is(program_state State) const {
  if (MState == State) {
    throw invalid_object_error("Invalid program state",
                               PI_ERROR_INVALID_PROGRAM);
  }
}

void program_impl::throw_if_state_is_not(program_state State) const {
  if (MState != State) {
    throw invalid_object_error("Invalid program state",
                               PI_ERROR_INVALID_PROGRAM);
  }
}

// TODO(pi2ur): Rename?
void program_impl::create_ur_program_with_kernel_name(
    const std::string &KernelName, bool JITCompilationIsRequired) {
  assert(!MProgram && "This program already has an encapsulated PI program");
  ProgramManager &PM = ProgramManager::getInstance();
  const device FirstDevice = get_devices()[0];
  RTDeviceBinaryImage &Img = PM.getDeviceImage(
      KernelName, get_context(), FirstDevice, JITCompilationIsRequired);
  MURProgram = PM.createURProgram(Img, get_context(), {FirstDevice});
}

void program_impl::flush_spec_constants(
    const RTDeviceBinaryImage &Img,
    ur_program_handle_t NativePrg) const {
  // iterate via all specialization constants the program's image depends on,
  // and set each to current runtime value (if any)
  const RTDeviceBinaryImage::PropertyRange &SCRange = Img.getSpecConstants();
  ContextImplPtr Ctx = getSyclObjImpl(get_context());
  using SCItTy = RTDeviceBinaryImage::PropertyRange::ConstIterator;

  auto LockGuard = Ctx->getKernelProgramCache().acquireCachedPrograms();
  NativePrg = NativePrg ? NativePrg : getUrHandleRef();

  for (SCItTy SCIt : SCRange) {
    auto SCEntry = SpecConstRegistry.find((*SCIt)->Name);
    if (SCEntry == SpecConstRegistry.end())
      // spec constant has not been set in user code - SPIR-V will use default
      continue;
    const spec_constant_impl &SC = SCEntry->second;
    assert(SC.isSet() && "uninitialized spec constant");
    ByteArray Descriptors = DeviceBinaryProperty(*SCIt).asByteArray();

    // First 8 bytes are consumed by the size of the property.
    Descriptors.dropBytes(8);

    // Expected layout is vector of 3-component tuples (flattened into a
    // vector of scalars), where each tuple consists of: ID of a scalar spec
    // constant, (which might be a member of the composite); offset, which
    // is used to calculate location of scalar member within the composite
    // or zero for scalar spec constants; size of a spec constant.
    // TODO(pi2ur): Do in one call if possible?
    while (!Descriptors.empty()) {
      auto [Id, Offset, Size] =
          Descriptors.consume<uint32_t, uint32_t, uint32_t>();

      ur_specialization_constant_info_t SpecConst = {Id, Size,
                                                     SC.getValuePtr() + Offset};
      Ctx->getPlugin()->call(urProgramSetSpecializationConstants, NativePrg, 1,
                             &SpecConst);
    }
  }
}

ur_native_handle_t program_impl::getNative() const {
  const auto &Plugin = getPlugin();
  if (getContextImplPtr()->getBackend() == backend::opencl)
    Plugin->call(urProgramRetain, MURProgram);
  ur_native_handle_t Handle = nullptr;
  Plugin->call(urProgramGetNativeHandle, MURProgram, &Handle);
  return Handle;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
