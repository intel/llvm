//==----- program_impl.cpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/kernel.hpp>
#include <detail/program_impl.hpp>

#include <algorithm>
#include <fstream>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

program_impl::program_impl(ContextImplPtr Context)
    : program_impl(Context, Context->get_info<info::context::devices>()) {}

program_impl::program_impl(ContextImplPtr Context,
                           vector_class<device> DeviceList)
    : MContext(Context), MDevices(DeviceList) {}

program_impl::program_impl(
    vector_class<shared_ptr_class<program_impl>> ProgramList,
    string_class LinkOptions)
    : MState(program_state::linked), MLinkOptions(LinkOptions),
      MBuildOptions(LinkOptions) {
  // Verify arguments
  if (ProgramList.empty()) {
    throw runtime_error("Non-empty vector of programs expected");
  }
  MContext = ProgramList[0]->MContext;
  MDevices = ProgramList[0]->MDevices;
  vector_class<device> DevicesSorted;
  if (!is_host()) {
    DevicesSorted = sort_devices_by_cl_device_id(MDevices);
  }
  check_device_feature_support<info::device::is_linker_available>(MDevices);
  for (const auto &Prg : ProgramList) {
    Prg->throw_if_state_is_not(program_state::compiled);
    if (Prg->MContext != MContext) {
      throw invalid_object_error(
          "Not all programs are associated with the same context");
    }
    if (!is_host()) {
      vector_class<device> PrgDevicesSorted =
          sort_devices_by_cl_device_id(Prg->MDevices);
      if (PrgDevicesSorted != DevicesSorted) {
        throw invalid_object_error(
            "Not all programs are associated with the same devices");
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

program_impl::program_impl(ContextImplPtr Context, RT::PiProgram Program)
    : MProgram(Program), MContext(Context), MLinkable(true) {

  // TODO handle the case when cl_program build is in progress
  cl_uint NumDevices;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(
      Program, PI_PROGRAM_INFO_NUM_DEVICES, sizeof(cl_uint), &NumDevices, nullptr);
  vector_class<RT::PiDevice> PiDevices(NumDevices);
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES,
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
  // TODO check build for each device instead
  cl_program_binary_type BinaryType;
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(
      Program, Device, CL_PROGRAM_BINARY_TYPE, sizeof(cl_program_binary_type),
      &BinaryType, nullptr);
  size_t Size = 0;
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(
      Program, Device, CL_PROGRAM_BUILD_OPTIONS, 0, nullptr, &Size);
  std::vector<char> OptionsVector(Size);
  Plugin.call<PiApiKind::piProgramGetBuildInfo>(Program, Device,
                                                CL_PROGRAM_BUILD_OPTIONS, Size,
                                                OptionsVector.data(), nullptr);
  string_class Options(OptionsVector.begin(), OptionsVector.end());
  switch (BinaryType) {
  case CL_PROGRAM_BINARY_TYPE_NONE:
    MState = program_state::none;
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
  Plugin.call<PiApiKind::piProgramRetain>(Program);
}

program_impl::program_impl(ContextImplPtr Context, RT::PiKernel Kernel)
    : program_impl(Context,
                   ProgramManager::getInstance().getClProgramFromClKernel(
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
    throw invalid_object_error("This instance of program is a host instance");
  }
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piProgramRetain>(MProgram);
  return pi::cast<cl_program>(MProgram);
}

void program_impl::compile_with_kernel_name(string_class KernelName,
                                            string_class CompileOptions,
                                            OSModuleHandle M) {
  throw_if_state_is_not(program_state::none);
  MProgramModuleHandle = M;
  if (!is_host()) {
    create_pi_program_with_kernel_name(M, KernelName);
    compile(CompileOptions);
  }
  MState = program_state::compiled;
}

void program_impl::compile_with_source(string_class KernelSource,
                                       string_class CompileOptions) {
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
  throw_if_state_is_not(program_state::none);
  MProgramModuleHandle = Module;
  if (!is_host()) {
    // If there are no build options, program can be safely cached
    if (is_cacheable_with_options(BuildOptions)) {
      MProgramAndKernelCachingAllowed = true;
      MProgram = ProgramManager::getInstance().getBuiltPIProgram(
          Module, get_context(), KernelName);
      const detail::plugin &Plugin = getPlugin();
      Plugin.call<PiApiKind::piProgramRetain>(MProgram);
    } else {
      create_pi_program_with_kernel_name(Module, KernelName);
      build(BuildOptions);
    }
  }
  MState = program_state::linked;
}

void program_impl::build_with_source(string_class KernelSource,
                                     string_class BuildOptions) {
  throw_if_state_is_not(program_state::none);
  // TODO should it throw if it's host?
  if (!is_host()) {
    create_cl_program_with_source(KernelSource);
    build(BuildOptions);
  }
  MState = program_state::linked;
}

void program_impl::link(string_class LinkOptions) {
  throw_if_state_is_not(program_state::compiled);
  if (!is_host()) {
    check_device_feature_support<info::device::is_linker_available>(MDevices);
    vector_class<RT::PiDevice> Devices(get_pi_devices());
    const detail::plugin &Plugin = getPlugin();
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramLink>(
        MContext->getHandleRef(), Devices.size(), Devices.data(),
        LinkOptions.c_str(), 1, &MProgram, nullptr, nullptr, &MProgram);
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
      throw invalid_object_error("This instance of program is a host instance");

    return createSyclObjFromImpl<kernel>(
        std::make_shared<kernel_impl>(MContext, PtrToSelf));
  }
  return createSyclObjFromImpl<kernel>(std::make_shared<kernel_impl>(
      get_pi_kernel(KernelName), MContext, PtrToSelf,
      /*IsCreatedFromSource*/ IsCreatedFromSource));
}

vector_class<vector_class<char>> program_impl::get_binaries() const {
  throw_if_state_is(program_state::none);
  vector_class<vector_class<char>> Result;
  const detail::plugin &Plugin = getPlugin();
  if (!is_host()) {
    vector_class<size_t> BinarySizes(MDevices.size());
    Plugin.call<PiApiKind::piProgramGetInfo>(
        MProgram, PI_PROGRAM_INFO_BINARY_SIZES, sizeof(size_t) * BinarySizes.size(),
        BinarySizes.data(), nullptr);

    vector_class<char *> Pointers;
    for (size_t I = 0; I < BinarySizes.size(); ++I) {
      Result.emplace_back(BinarySizes[I]);
      Pointers.push_back(Result[I].data());
    }
    Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_BINARIES,
                                             sizeof(char *) * Pointers.size(),
                                             Pointers.data(), nullptr);
  }
  return Result;
}

void program_impl::create_cl_program_with_source(const string_class &Source) {
  assert(!MProgram && "This program already has an encapsulated cl_program");
  const char *Src = Source.c_str();
  size_t Size = Source.size();
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piclProgramCreateWithSource>(
      MContext->getHandleRef(), 1, &Src, &Size, &MProgram);
}

void program_impl::compile(const string_class &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  vector_class<RT::PiDevice> Devices(get_pi_devices());
  const detail::plugin &Plugin = getPlugin();
  RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
      MProgram, Devices.size(), Devices.data(), Options.c_str(), 0, nullptr,
      nullptr, nullptr, nullptr);

  if (Err != PI_SUCCESS) {
    throw compile_program_error(
        "Program compilation error:\n" +
        ProgramManager::getProgramBuildLog(MProgram, MContext));
  }
  MCompileOptions = Options;
}

void program_impl::build(const string_class &Options) {
  check_device_feature_support<info::device::is_compiler_available>(MDevices);
  vector_class<RT::PiDevice> Devices(get_pi_devices());
  const detail::plugin &Plugin = getPlugin();
  RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piProgramBuild>(
      MProgram, Devices.size(), Devices.data(), Options.c_str(), nullptr,
      nullptr);

  if (Err != PI_SUCCESS) {
    throw compile_program_error(
        "Program build error:\n" +
        ProgramManager::getProgramBuildLog(MProgram, MContext));
  }
  MBuildOptions = Options;
  MCompileOptions = Options;
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
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_KERNEL_NAMES, 0,
                                           nullptr, &Size);
  string_class ClResult(Size, ' ');
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_KERNEL_NAMES,
                                           ClResult.size(), &ClResult[0],
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
  RT::PiKernel Kernel;

  if (is_cacheable()) {
    Kernel = ProgramManager::getInstance().getOrCreateKernel(
        MProgramModuleHandle, get_context(), KernelName);
  } else {
    const detail::plugin &Plugin = getPlugin();
    RT::PiResult Err = Plugin.call_nocheck<PiApiKind::piKernelCreate>(
        MProgram, KernelName.c_str(), &Kernel);
    if (Err == PI_RESULT_INVALID_KERNEL_NAME) {
      throw invalid_object_error(
          "This instance of program does not contain the kernel requested");
    }
    Plugin.checkPiResult(Err);
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
    throw invalid_object_error("Invalid program state");
  }
}

void program_impl::throw_if_state_is_not(program_state State) const {
  if (MState != State) {
    throw invalid_object_error("Invalid program state");
  }
}

void program_impl::create_pi_program_with_kernel_name(
    OSModuleHandle Module, const string_class &KernelName) {
  assert(!MProgram && "This program already has an encapsulated PI program");
  ProgramManager &PM = ProgramManager::getInstance();
  DeviceImage &Img = PM.getDeviceImage(Module, KernelName, get_context());
  MProgram = PM.createPIProgram(Img, get_context());
}

template <>
cl_uint program_impl::get_info<info::program::reference_count>() const {
  if (is_host()) {
    throw invalid_object_error("This instance of program is a host instance");
  }
  cl_uint Result;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(MProgram, PI_PROGRAM_INFO_REFERENCE_COUNT,
                                           sizeof(cl_uint), &Result, nullptr);
  return Result;
}

template <> context program_impl::get_info<info::program::context>() const {
  return get_context();
}

template <>
vector_class<device> program_impl::get_info<info::program::devices>() const {
  return get_devices();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
