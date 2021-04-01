//==----- program_impl.cpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/property_list.hpp>
#include <detail/config.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_impl.hpp>
#include <detail/spec_constant_impl.hpp>

#include <algorithm>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

program_impl::program_impl(ContextImplPtr Context,
                           const property_list &PropList)
    : program_impl(Context, Context->get_info<info::context::devices>(),
                   PropList) {}

program_impl::program_impl(ContextImplPtr Context,
                           vector_class<device> DeviceList,
                           const property_list &PropList)
    : MContext(Context), MDevices(DeviceList), MPropList(PropList) {
  if (Context->getDevices().size() > 1) {
    throw feature_not_supported(
        "multiple devices within a context are not supported with "
        "sycl::program and sycl::kernel",
        PI_INVALID_OPERATION);
  }
}

program_impl::program_impl(
    vector_class<shared_ptr_class<program_impl>> ProgramList,
    string_class LinkOptions, const property_list &PropList)
    : MState(program_state::linked), MPropList(PropList),
      MLinkOptions(LinkOptions), MBuildOptions(LinkOptions) {
  // Verify arguments
  if (ProgramList.empty()) {
    throw runtime_error("Non-empty vector of programs expected",
                        PI_INVALID_VALUE);
  }

  // Sort the programs to avoid deadlocks due to locking multiple mutexes &
  // verify that all programs are unique.
  std::sort(ProgramList.begin(), ProgramList.end());
  auto It = std::unique(ProgramList.begin(), ProgramList.end());
  if (It != ProgramList.end()) {
    throw runtime_error("Attempting to link a program with itself",
                        PI_INVALID_PROGRAM);
  }

  MContext = ProgramList[0]->MContext;
  if (MContext->getDevices().size() > 1) {
    throw feature_not_supported(
        "multiple devices within a context are not supported with "
        "sycl::program and sycl::kernel",
        PI_INVALID_OPERATION);
  }
  MDevices = ProgramList[0]->MDevices;
  vector_class<device> DevicesSorted;
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
          PI_INVALID_PROGRAM);
    }
    if (!is_host()) {
      vector_class<device> PrgDevicesSorted =
          sort_devices_by_cl_device_id(Prg->MDevices);
      if (PrgDevicesSorted != DevicesSorted) {
        throw invalid_object_error(
            "Not all programs are associated with the same devices",
            PI_INVALID_PROGRAM);
      }
    }
  }

  if (!is_host()) {
    vector_class<RT::PiDevice> Devices(get_pi_devices());
    vector_class<RT::PiProgram> Programs;
    bool NonInterOpToLink = false;
    for (const auto &Prg : ProgramList) {
      if (!Prg->MLinkable && NonInterOpToLink)
        continue;
      NonInterOpToLink |= !Prg->MLinkable;
      Programs.push_back(Prg->MProgram);
    }
    const detail::plugin &Plugin = getPlugin();
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramLink>(
        MContext->getHandleRef(), Devices.size(), Devices.data(),
        LinkOptions.c_str(), Programs.size(), Programs.data(), nullptr, nullptr,
        &MProgram);
    Plugin.checkPiResult<compile_program_error>(Err);
  }
}

program_impl::program_impl(ContextImplPtr Context,
                           pi_native_handle InteropProgram)
    : program_impl(Context, InteropProgram, nullptr) {}

program_impl::program_impl(ContextImplPtr Context,
                           pi_native_handle InteropProgram,
                           RT::PiProgram Program)
    : MProgram(Program), MContext(Context), MLinkable(true) {
  const detail::plugin &Plugin = getPlugin();
  if (MProgram == nullptr) {
    assert(InteropProgram &&
           "No InteropProgram/PiProgram defined with piextProgramFromNative");
    // Translate the raw program handle into PI program.
    Plugin.call<PiApiKind::piextProgramCreateWithNativeHandle>(
        InteropProgram, MContext->getHandleRef(), &MProgram);
  } else
    Plugin.call<PiApiKind::piProgramRetain>(Program);

  // TODO handle the case when cl_program build is in progress
  pi_uint32 NumDevices;
  Plugin.call<PiApiKind::piProgramGetInfo>(
      MProgram, PI_PROGRAM_INFO_NUM_DEVICES, sizeof(pi_uint32), &NumDevices,
      nullptr);
  vector_class<RT::PiDevice> PiDevices(NumDevices);
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_DEVICES,
                                           sizeof(RT::PiDevice) * NumDevices,
                                           PiDevices.data(), nullptr);
  vector_class<device> SyclContextDevices =
      MContext->get_info<info::context::devices>();

  // Keep only the subset of the devices (associated with context) that
  // were actually used to create the program.
  // This is possible when clCreateProgramWithBinary is used.
  auto NewEnd = std::remove_if(
      SyclContextDevices.begin(), SyclContextDevices.end(),
      [&PiDevices](const sycl::device &Dev) {
        return PiDevices.end() ==
               std::find(PiDevices.begin(), PiDevices.end(),
                         detail::getSyclObjImpl(Dev)->getHandleRef());
      });
  SyclContextDevices.erase(NewEnd, SyclContextDevices.end());
  MDevices = SyclContextDevices;
  RT::PiDevice Device = getSyclObjImpl(MDevices[0])->getHandleRef();
  assert(!MDevices.empty() && "No device found for this program");
  // TODO check build for each device instead
  cl_program_binary_type BinaryType;
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(
      MProgram, Device, CL_PROGRAM_BINARY_TYPE, sizeof(cl_program_binary_type),
      &BinaryType, nullptr);
  if (BinaryType == CL_PROGRAM_BINARY_TYPE_NONE) {
    throw invalid_object_error(
        "The native program passed to the program constructor has to be either "
        "compiled or linked",
        PI_INVALID_PROGRAM);
  }
  size_t Size = 0;
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(
      MProgram, Device, CL_PROGRAM_BUILD_OPTIONS, 0, nullptr, &Size);
  std::vector<char> OptionsVector(Size);
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(MProgram, Device,
                                                CL_PROGRAM_BUILD_OPTIONS, Size,
                                                OptionsVector.data(), nullptr);
  string_class Options(OptionsVector.begin(), OptionsVector.end());
  switch (BinaryType) {
  case CL_PROGRAM_BINARY_TYPE_NONE:
    assert(false);
    break;
  case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
    MState = program_state::compiled;
    MCompileOptions = Options;
    MBuildOptions = Options;
    break;
  case CL_PROGRAM_BINARY_TYPE_LIBRARY:
  case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
    MState = program_state::linked;
    MLinkOptions = "";
    MBuildOptions = Options;
  }
}

program_impl::program_impl(ContextImplPtr Context, RT::PiKernel Kernel)
    : program_impl(Context, reinterpret_cast<pi_native_handle>(nullptr),
                   ProgramManager::getInstance().getPiProgramFromPiKernel(
                       Kernel, Context)) {}

program_impl::~program_impl() {
  // TODO catch an exception and put it to list of asynchronous exceptions
  if (!is_host() && MProgram != nullptr) {
    const detail::plugin &Plugin = getPlugin();
    Plugin.call<PiApiKind::piProgramRelease>(MProgram);
  }
}

cl_program program_impl::get() const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    throw invalid_object_error(
        "This instance of program doesn't support OpenCL interoperability.",
        PI_INVALID_PROGRAM);
  }
  getPlugin().call<PiApiKind::piProgramRetain>(MProgram);
  return pi::cast<cl_program>(MProgram);
}

void program_impl::compile_with_kernel_name(string_class KernelName,
                                            string_class CompileOptions,
                                            OSModuleHandle M) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::none);
  MProgramModuleHandle = M;
  if (!is_host()) {
    create_pi_program_with_kernel_name(
        M, KernelName,
        /*JITCompilationIsRequired=*/(!CompileOptions.empty()));
    compile(CompileOptions);
  }
  MState = program_state::compiled;
}

void program_impl::compile_with_source(string_class KernelSource,
                                       string_class CompileOptions) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::none);
  // TODO should it throw if it's host?
  if (!is_host()) {
    create_cl_program_with_source(KernelSource);
    compile(CompileOptions);
  }
  MState = program_state::compiled;
}

void program_impl::build_with_kernel_name(string_class KernelName,
                                          string_class BuildOptions,
                                          OSModuleHandle Module) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::none);
  MProgramModuleHandle = Module;
  if (!is_host()) {
    MProgramAndKernelCachingAllowed = true;
    MBuildOptions = BuildOptions;
    MProgram = ProgramManager::getInstance().getBuiltPIProgram(
        Module, get_context(), get_devices()[0], KernelName, this,
        /*JITCompilationIsRequired=*/(!BuildOptions.empty()));
    const detail::plugin &Plugin = getPlugin();
    Plugin.call<PiApiKind::piProgramRetain>(MProgram);
  }
  MState = program_state::linked;
}

void program_impl::build_with_source(string_class KernelSource,
                                     string_class BuildOptions) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::none);
  // TODO should it throw if it's host?
  if (!is_host()) {
    create_cl_program_with_source(KernelSource);
    build(BuildOptions);
  }
  MState = program_state::linked;
}

void program_impl::link(string_class LinkOptions) {
  std::lock_guard<std::mutex> Lock(MMutex);
  throw_if_state_is_not(program_state::compiled);
  if (!is_host()) {
    check_device_feature_support<info::device::is_linker_available>(MDevices);
    vector_class<RT::PiDevice> Devices(get_pi_devices());
    const detail::plugin &Plugin = getPlugin();
    const char *LinkOpts = SYCLConfig<SYCL_PROGRAM_LINK_OPTIONS>::get();
    if (!LinkOpts) {
      LinkOpts = LinkOptions.c_str();
    }
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramLink>(
        MContext->getHandleRef(), Devices.size(), Devices.data(), LinkOpts,
        /*num_input_programs*/ 1, &MProgram, nullptr, nullptr, &MProgram);
    Plugin.checkPiResult<compile_program_error>(Err);
    MLinkOptions = LinkOptions;
    MBuildOptions = LinkOptions;
  }
  MState = program_state::linked;
}

bool program_impl::has_kernel(string_class KernelName,
                              bool IsCreatedFromSource) const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    return !IsCreatedFromSource;
  }
  return has_cl_kernel(KernelName);
}

kernel program_impl::get_kernel(string_class KernelName,
                                shared_ptr_class<program_impl> PtrToSelf,
                                bool IsCreatedFromSource) const {
  throw_if_state_is(program_state::none);
  if (is_host()) {
    if (IsCreatedFromSource)
      throw invalid_object_error("This instance of program is a host instance",
                                 PI_INVALID_PROGRAM);

    return createSyclObjFromImpl<kernel>(
        std::make_shared<kernel_impl>(MContext, PtrToSelf));
  }
  return createSyclObjFromImpl<kernel>(std::make_shared<kernel_impl>(
      get_pi_kernel(KernelName), MContext, PtrToSelf,
      /*IsCreatedFromSource*/ IsCreatedFromSource));
}

vector_class<vector_class<char>> program_impl::get_binaries() const {
  throw_if_state_is(program_state::none);
  if (is_host())
    return {};

  vector_class<vector_class<char>> Result;
  const detail::plugin &Plugin = getPlugin();
  vector_class<size_t> BinarySizes(MDevices.size());
  Plugin.call<PiApiKind::piProgramGetInfo>(
      MProgram, PI_PROGRAM_INFO_BINARY_SIZES,
      sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

  vector_class<char *> Pointers;
  for (size_t I = 0; I < BinarySizes.size(); ++I) {
    Result.emplace_back(BinarySizes[I]);
    Pointers.push_back(Result[I].data());
  }
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_BINARIES,
                                           sizeof(char *) * Pointers.size(),
                                           Pointers.data(), nullptr);
  return Result;
}

void program_impl::create_cl_program_with_source(const string_class &Source) {
  assert(!MProgram && "This program already has an encapsulated cl_program");
  const char *Src = Source.c_str();
  size_t Size = Source.size();
  const detail::plugin &Plugin = getPlugin();
  RT::PiResult Err =
      Plugin.call_nocheck<PiApiKind::piclProgramCreateWithSource>(
          MContext->getHandleRef(), 1, &Src, &Size, &MProgram);

  if (Err == PI_INVALID_OPERATION) {
    throw feature_not_supported(
        "program::compile_with_source is not supported by the selected backend",
        PI_INVALID_OPERATION);
  }
}

void program_impl::compile(const string_class &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  vector_class<RT::PiDevice> Devices(get_pi_devices());
  const detail::plugin &Plugin = getPlugin();
  const char *CompileOpts = SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  if (!CompileOpts) {
    CompileOpts = Options.c_str();
  }
  RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
      MProgram, Devices.size(), Devices.data(), CompileOpts, 0, nullptr,
      nullptr, nullptr, nullptr);

  if (Err != PI_SUCCESS) {
    throw compile_program_error(
        "Program compilation error:\n" +
            ProgramManager::getProgramBuildLog(MProgram, MContext),
        Err);
  }
  MCompileOptions = Options;
  MBuildOptions = Options;
}

void program_impl::build(const string_class &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  vector_class<RT::PiDevice> Devices(get_pi_devices());
  const detail::plugin &Plugin = getPlugin();
  ProgramManager::getInstance().flushSpecConstants(*this);
  RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramBuild>(
      MProgram, Devices.size(), Devices.data(), Options.c_str(), nullptr,
      nullptr);

  if (Err != PI_SUCCESS) {
    throw compile_program_error(
        "Program build error:\n" +
            ProgramManager::getProgramBuildLog(MProgram, MContext),
        Err);
  }
  MBuildOptions = Options;
}

vector_class<RT::PiDevice> program_impl::get_pi_devices() const {
  vector_class<RT::PiDevice> PiDevices;
  for (const auto &Device : MDevices) {
    PiDevices.push_back(getSyclObjImpl(Device)->getHandleRef());
  }
  return PiDevices;
}

bool program_impl::has_cl_kernel(const string_class &KernelName) const {
  size_t Size;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(
      MProgram, PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr, &Size);
  string_class ClResult(Size, ' ');
  Plugin.call<PiApiKind::piProgramGetInfo>(
      MProgram, PI_PROGRAM_INFO_KERNEL_NAMES, ClResult.size(), &ClResult[0],
      nullptr);
  // Get rid of the null terminator
  ClResult.pop_back();
  vector_class<string_class> KernelNames(split_string(ClResult, ';'));
  for (const auto &Name : KernelNames) {
    if (Name == KernelName) {
      return true;
    }
  }
  return false;
}

RT::PiKernel program_impl::get_pi_kernel(const string_class &KernelName) const {
  RT::PiKernel Kernel = nullptr;

  if (is_cacheable()) {
    std::tie(Kernel, std::ignore) =
        ProgramManager::getInstance().getOrCreateKernel(
            MProgramModuleHandle, get_context(), get_devices()[0], KernelName,
            this);
    getPlugin().call<PiApiKind::piKernelRetain>(Kernel);
  } else {
    const detail::plugin &Plugin = getPlugin();
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piKernelCreate>(
        MProgram, KernelName.c_str(), &Kernel);
    if (Err == PI_INVALID_KERNEL_NAME) {
      throw invalid_object_error(
          "This instance of program does not contain the kernel requested",
          Err);
    }
    Plugin.checkPiResult(Err);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    Plugin.call<PiApiKind::piKernelSetExecInfo>(Kernel, PI_USM_INDIRECT_ACCESS,
                                                sizeof(pi_bool), &PI_TRUE);
  }

  return Kernel;
}

vector_class<device>
program_impl::sort_devices_by_cl_device_id(vector_class<device> Devices) {
  std::sort(Devices.begin(), Devices.end(),
            [](const device &id1, const device &id2) {
              return (detail::getSyclObjImpl(id1)->getHandleRef() <
                      detail::getSyclObjImpl(id2)->getHandleRef());
            });
  return Devices;
}

void program_impl::throw_if_state_is(program_state State) const {
  if (MState == State) {
    throw invalid_object_error("Invalid program state", PI_INVALID_PROGRAM);
  }
}

void program_impl::throw_if_state_is_not(program_state State) const {
  if (MState != State) {
    throw invalid_object_error("Invalid program state", PI_INVALID_PROGRAM);
  }
}

void program_impl::create_pi_program_with_kernel_name(
    OSModuleHandle Module, const string_class &KernelName,
    bool JITCompilationIsRequired) {
  assert(!MProgram && "This program already has an encapsulated PI program");
  ProgramManager &PM = ProgramManager::getInstance();
  const device FirstDevice = get_devices()[0];
  RTDeviceBinaryImage &Img = PM.getDeviceImage(
      Module, KernelName, get_context(), FirstDevice, JITCompilationIsRequired);
  MProgram = PM.createPIProgram(Img, get_context(), {FirstDevice});
}

template <>
cl_uint program_impl::get_info<info::program::reference_count>() const {
  if (is_host()) {
    throw invalid_object_error("This instance of program is a host instance",
                               PI_INVALID_PROGRAM);
  }
  pi_uint32 Result;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram,
                                           PI_PROGRAM_INFO_REFERENCE_COUNT,
                                           sizeof(pi_uint32), &Result, nullptr);
  return Result;
}

template <> context program_impl::get_info<info::program::context>() const {
  return get_context();
}

template <>
vector_class<device> program_impl::get_info<info::program::devices>() const {
  return get_devices();
}

void program_impl::set_spec_constant_impl(const char *Name, const void *ValAddr,
                                          size_t ValSize) {
  if (MState != program_state::none)
    throw cl::sycl::ONEAPI::experimental::spec_const_error(
        "Invalid program state", PI_INVALID_PROGRAM);
  // Reuse cached programs lock as opposed to introducing a new lock.
  auto LockGuard = MContext->getKernelProgramCache().acquireCachedPrograms();
  spec_constant_impl &SC = SpecConstRegistry[Name];
  SC.set(ValSize, ValAddr);
}

void program_impl::flush_spec_constants(const RTDeviceBinaryImage &Img,
                                        RT::PiProgram NativePrg) const {
  // iterate via all specialization constants the program's image depends on,
  // and set each to current runtime value (if any)
  const pi::DeviceBinaryImage::PropertyRange &SCRange = Img.getSpecConstants();
  ContextImplPtr Ctx = getSyclObjImpl(get_context());
  using SCItTy = pi::DeviceBinaryImage::PropertyRange::ConstIterator;

  auto LockGuard = Ctx->getKernelProgramCache().acquireCachedPrograms();
  NativePrg = NativePrg ? NativePrg : getHandleRef();

  for (SCItTy SCIt : SCRange) {
    auto SCEntry = SpecConstRegistry.find((*SCIt)->Name);
    if (SCEntry == SpecConstRegistry.end())
      // spec constant has not been set in user code - SPIR-V will use default
      continue;
    const spec_constant_impl &SC = SCEntry->second;
    assert(SC.isSet() && "uninitialized spec constant");
    pi::ByteArray Descriptors = pi::DeviceBinaryProperty(*SCIt).asByteArray();
    // First 8 bytes are consumed by size of the property
    assert(Descriptors.size() > 8 && "Unexpected property size");
    // Expected layout is vector of 3-component tuples (flattened into a vector
    // of scalars), where each tuple consists of: ID of a scalar spec constant,
    // (which might be a member of the composite); offset, which is used to
    // calculate location of scalar member within the composite or zero for
    // scalar spec constants; size of a spec constant
    assert(((Descriptors.size() - 8) / sizeof(std::uint32_t)) % 3 == 0 &&
           "unexpected layout of composite spec const descriptors");
    auto *It = reinterpret_cast<const std::uint32_t *>(&Descriptors[8]);
    auto *End = reinterpret_cast<const std::uint32_t *>(&Descriptors[0] +
                                                        Descriptors.size());
    while (It != End) {
      Ctx->getPlugin().call<PiApiKind::piextProgramSetSpecializationConstant>(
          NativePrg, /* ID */ It[0], /* Size */ It[2],
          SC.getValuePtr() + /* Offset */ It[1]);
      It += 3;
    }
  }
}

pi_native_handle program_impl::getNative() const {
  const auto &Plugin = getPlugin();
  if (Plugin.getBackend() == backend::opencl)
    Plugin.call<PiApiKind::piProgramRetain>(MProgram);
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextProgramGetNativeHandle>(MProgram, &Handle);
  return Handle;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
