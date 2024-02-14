//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/aspects.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/kernel_properties.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/stl.hpp>

#include <sycl/ext/oneapi/matrix/query-types.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <variant>

namespace sycl {
inline namespace _V1 {
namespace detail {

using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;

static constexpr int DbgProgMgr = 0;

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

/// This function enables ITT annotations in SPIR-V module by setting
/// a specialization constant if INTEL_LIBITTNOTIFY64 env variable is set.
static void
enableITTAnnotationsIfNeeded(const sycl::detail::pi::PiProgram &Prog,
                             const PluginPtr &Plugin) {
  if (SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::get() != nullptr) {
    constexpr char SpecValue = 1;
    Plugin->call<PiApiKind::piextProgramSetSpecializationConstant>(
        Prog, ITTSpecConstId, sizeof(char), &SpecValue);
  }
}

ProgramManager &ProgramManager::getInstance() {
  return GlobalHandler::instance().getProgramManager();
}

static sycl::detail::pi::PiProgram
createBinaryProgram(const ContextImplPtr Context, const device &Device,
                    const unsigned char *Data, size_t DataLen,
                    const std::vector<pi_device_binary_property> Metadata) {
  const PluginPtr &Plugin = Context->getPlugin();
#ifndef _NDEBUG
  pi_uint32 NumDevices = 0;
  Plugin->call<PiApiKind::piContextGetInfo>(Context->getHandleRef(),
                                            PI_CONTEXT_INFO_NUM_DEVICES,
                                            sizeof(NumDevices), &NumDevices,
                                            /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  sycl::detail::pi::PiProgram Program;
  const sycl::detail::pi::PiDevice PiDevice =
      getSyclObjImpl(Device)->getHandleRef();
  pi_int32 BinaryStatus = CL_SUCCESS;
  Plugin->call<PiApiKind::piProgramCreateWithBinary>(
      Context->getHandleRef(), 1 /*one binary*/, &PiDevice, &DataLen, &Data,
      Metadata.size(), Metadata.data(), &BinaryStatus, &Program);

  if (BinaryStatus != CL_SUCCESS) {
    throw runtime_error("Creating program with binary failed.", BinaryStatus);
  }

  return Program;
}

static sycl::detail::pi::PiProgram
createSpirvProgram(const ContextImplPtr Context, const unsigned char *Data,
                   size_t DataLen) {
  sycl::detail::pi::PiProgram Program = nullptr;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piProgramCreate>(Context->getHandleRef(), Data,
                                           DataLen, &Program);
  return Program;
}

// TODO replace this with a new PI API function
static bool
isDeviceBinaryTypeSupported(const context &C,
                            sycl::detail::pi::PiDeviceBinaryType Format) {
  // All formats except PI_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  const backend ContextBackend = detail::getSyclObjImpl(C)->getBackend();

  // The CUDA backend cannot use SPIR-V
  if (ContextBackend == backend::ext_oneapi_cuda)
    return false;

  std::vector<device> Devices = C.get_devices();

  // Program type is SPIR-V, so we need a device compiler to do JIT.
  for (const device &D : Devices) {
    if (!D.get_info<info::device::is_compiler_available>())
      return false;
  }

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (ContextBackend == backend::opencl) {
    std::string ver = C.get_platform().get_info<info::platform::version>();
    if (ver.find("OpenCL 1.0") == std::string::npos &&
        ver.find("OpenCL 1.1") == std::string::npos &&
        ver.find("OpenCL 1.2") == std::string::npos &&
        ver.find("OpenCL 2.0") == std::string::npos)
      return true;
  }

  for (const device &D : Devices) {
    // We need cl_khr_il_program extension to be present
    // and we can call clCreateProgramWithILKHR using the extension
    std::vector<std::string> Extensions =
        D.get_info<info::device::extensions>();
    if (Extensions.end() ==
        std::find(Extensions.begin(), Extensions.end(), "cl_khr_il_program"))
      return false;
  }

  return true;
}

static const char *getFormatStr(sycl::detail::pi::PiDeviceBinaryType Format) {
  switch (Format) {
  case PI_DEVICE_BINARY_TYPE_NONE:
    return "none";
  case PI_DEVICE_BINARY_TYPE_NATIVE:
    return "native";
  case PI_DEVICE_BINARY_TYPE_SPIRV:
    return "SPIR-V";
  case PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE:
    return "LLVM IR";
  }
  assert(false && "Unknown device image format");
  return "unknown";
}

sycl::detail::pi::PiProgram
ProgramManager::createPIProgram(const RTDeviceBinaryImage &Img,
                                const context &Context, const device &Device) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::createPIProgram(" << &Img << ", "
              << getRawSyclObjImpl(Context) << ", " << getRawSyclObjImpl(Device)
              << ")\n";
  const pi_device_binary_struct &RawImg = Img.getRawData();

  // perform minimal sanity checks on the device image and the descriptor
  if (RawImg.BinaryEnd < RawImg.BinaryStart) {
    throw runtime_error("Malformed device program image descriptor",
                        PI_ERROR_INVALID_VALUE);
  }
  if (RawImg.BinaryEnd == RawImg.BinaryStart) {
    throw runtime_error("Invalid device program image: size is zero",
                        PI_ERROR_INVALID_VALUE);
  }
  size_t ImgSize = Img.getSize();

  // TODO if the binary image is a part of the fat binary, the clang
  //   driver should have set proper format option to the
  //   clang-offload-wrapper. The fix depends on AOT compilation
  //   implementation, so will be implemented together with it.
  //   Img->Format can't be updated as it is inside of the in-memory
  //   OS module binary.
  sycl::detail::pi::PiDeviceBinaryType Format = Img.getFormat();

  if (Format == PI_DEVICE_BINARY_TYPE_NONE)
    Format = pi::getBinaryImageFormat(RawImg.BinaryStart, ImgSize);
  // sycl::detail::pi::PiDeviceBinaryType Format = Img->Format;
  // assert(Format != PI_DEVICE_BINARY_TYPE_NONE && "Image format not set");

  if (!isDeviceBinaryTypeSupported(Context, Format))
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "SPIR-V online compilation is not supported in this context");

  // Get program metadata from properties
  auto ProgMetadata = Img.getProgramMetadata();
  std::vector<pi_device_binary_property> ProgMetadataVector{
      ProgMetadata.begin(), ProgMetadata.end()};

  // Load the image
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  sycl::detail::pi::PiProgram Res =
      Format == PI_DEVICE_BINARY_TYPE_SPIRV
          ? createSpirvProgram(Ctx, RawImg.BinaryStart, ImgSize)
          : createBinaryProgram(Ctx, Device, RawImg.BinaryStart, ImgSize,
                                ProgMetadataVector);

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    // associate the PI program with the image it was created for
    NativePrograms[Res] = &Img;
  }

  Ctx->addDeviceGlobalInitializer(Res, {Device}, &Img);

  if (DbgProgMgr > 1)
    std::cerr << "created program: " << Res
              << "; image format: " << getFormatStr(Format) << "\n";

  return Res;
}

static void appendLinkOptionsFromImage(std::string &LinkOpts,
                                       const RTDeviceBinaryImage &Img) {
  static const char *LinkOptsEnv = SYCLConfig<SYCL_PROGRAM_LINK_OPTIONS>::get();
  // Update only if link options are not overwritten by environment variable
  if (!LinkOptsEnv) {
    const char *TemporaryStr = Img.getLinkOptions();
    if (TemporaryStr != nullptr) {
      if (!LinkOpts.empty())
        LinkOpts += " ";
      LinkOpts += std::string(TemporaryStr);
    }
  }
}

static bool getUint32PropAsBool(const RTDeviceBinaryImage &Img,
                                const char *PropName) {
  pi_device_binary_property Prop = Img.getProperty(PropName);
  return Prop && (DeviceBinaryProperty(Prop).asUint32() != 0);
}

static std::string getUint32PropAsOptStr(const RTDeviceBinaryImage &Img,
                                         const char *PropName) {
  pi_device_binary_property Prop = Img.getProperty(PropName);
  std::stringstream ss;
  if (!Prop)
    return "";
  int optLevel = DeviceBinaryProperty(Prop).asUint32();
  if (optLevel < 0 || optLevel > 3)
    return "";
  ss << "-O" << optLevel;
  std::string temp = ss.str();
  return temp;
}

static void
appendCompileOptionsForGRFSizeProperties(std::string &CompileOpts,
                                         const RTDeviceBinaryImage &Img,
                                         bool IsEsimdImage) {
  // TODO: sycl-register-alloc-mode is deprecated and should be removed in the
  // next ABI break.
  pi_device_binary_property RegAllocModeProp =
      Img.getProperty("sycl-register-alloc-mode");
  pi_device_binary_property GRFSizeProp = Img.getProperty("sycl-grf-size");

  if (!RegAllocModeProp && !GRFSizeProp)
    return;
  // The mutual exclusivity of these properties should have been checked in
  // sycl-post-link.
  assert(!RegAllocModeProp || !GRFSizeProp);
  bool IsLargeGRF = false;
  bool IsAutoGRF = false;
  if (RegAllocModeProp) {
    uint32_t RegAllocModePropVal =
        DeviceBinaryProperty(RegAllocModeProp).asUint32();
    IsLargeGRF = RegAllocModePropVal ==
                 static_cast<uint32_t>(register_alloc_mode_enum::large);
    IsAutoGRF = RegAllocModePropVal ==
                static_cast<uint32_t>(register_alloc_mode_enum::automatic);
  } else {
    assert(GRFSizeProp);
    uint32_t GRFSizePropVal = DeviceBinaryProperty(GRFSizeProp).asUint32();
    IsLargeGRF = GRFSizePropVal == 256;
    IsAutoGRF = GRFSizePropVal == 0;
  }
  if (IsLargeGRF) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    // This option works for both LO AND OCL backends.
    CompileOpts += IsEsimdImage ? "-doubleGRF" : "-ze-opt-large-register-file";
  }
  if (IsAutoGRF) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    // This option works for both LO AND OCL backends.
    CompileOpts += "-ze-intel-enable-auto-large-GRF-mode";
  }
}

static void appendCompileOptionsFromImage(std::string &CompileOpts,
                                          const RTDeviceBinaryImage &Img,
                                          const std::vector<device> &Devs,
                                          const PluginPtr &) {
  // Build options are overridden if environment variables are present.
  // Environment variables are not changed during program lifecycle so it
  // is reasonable to use static here to read them only once.
  static const char *CompileOptsEnv =
      SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  // Update only if compile options are not overwritten by environment
  // variable
  if (!CompileOptsEnv) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    const char *TemporaryStr = Img.getCompileOptions();
    if (TemporaryStr != nullptr)
      CompileOpts += std::string(TemporaryStr);
  }
  bool isEsimdImage = getUint32PropAsBool(Img, "isEsimdImage");
  // The -vc-codegen option is always preserved for ESIMD kernels, regardless
  // of the contents SYCL_PROGRAM_COMPILE_OPTIONS environment variable.
  if (isEsimdImage) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    CompileOpts += "-vc-codegen";
    // Allow warning and performance hints from vc/finalizer if the RT warning
    // level is at least 1.
    if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() == 0)
      CompileOpts += " -disable-finalizer-msg";
  }

  appendCompileOptionsForGRFSizeProperties(CompileOpts, Img, isEsimdImage);

  const auto &PlatformImpl = detail::getSyclObjImpl(Devs[0].get_platform());

  // Add optimization flags.
  auto str = getUint32PropAsOptStr(Img, "optLevel");
  const char *optLevelStr = str.c_str();
  // TODO: Passing these options to vector compiler causes build failure in
  // backend. Will pass the flags once backend compilation issue is resolved.
  // Update only if compile options are not overwritten by environment
  // variable.
  if (!isEsimdImage && !CompileOptsEnv && optLevelStr != nullptr &&
      optLevelStr[0] != '\0') {
    // Making sure all devices have the same platform.
    assert(!Devs.empty() &&
           std::all_of(Devs.begin(), Devs.end(), [&](const device &Dev) {
             return Dev.get_platform() == Devs[0].get_platform();
           }));
    const char *backend_option = nullptr;
    // Empty string is returned in backend_option when no appropriate backend
    // option is available for a given frontend option.
    PlatformImpl->getBackendOption(optLevelStr, &backend_option);
    if (backend_option && backend_option[0] != '\0') {
      if (!CompileOpts.empty())
        CompileOpts += " ";
      CompileOpts += std::string(backend_option);
    }
  }
  bool IsIntelGPU =
      (PlatformImpl->getBackend() == backend::ext_oneapi_level_zero ||
       PlatformImpl->getBackend() == backend::opencl) &&
      std::all_of(Devs.begin(), Devs.end(), [](const device &Dev) {
        return Dev.is_gpu() &&
               Dev.get_info<info::device::vendor_id>() == 0x8086;
      });
  if (!CompileOptsEnv) {
    static const char *TargetCompileFast = "-ftarget-compile-fast";
    if (auto Pos = CompileOpts.find(TargetCompileFast);
        Pos != std::string::npos) {
      const char *BackendOption = nullptr;
      if (IsIntelGPU)
        PlatformImpl->getBackendOption(TargetCompileFast, &BackendOption);
      auto OptLen = strlen(TargetCompileFast);
      if (IsIntelGPU && BackendOption && BackendOption[0] != '\0')
        CompileOpts.replace(Pos, OptLen, BackendOption);
      else
        CompileOpts.erase(Pos, OptLen);
    }
    static const std::string TargetRegisterAllocMode =
        "-ftarget-register-alloc-mode=";
    auto OptPos = CompileOpts.find(TargetRegisterAllocMode);
    while (OptPos != std::string::npos) {
      auto EndOfOpt = CompileOpts.find(" ", OptPos);
      // Extract everything after the equals until the end of the option
      auto OptValue = CompileOpts.substr(
          OptPos + TargetRegisterAllocMode.size(),
          EndOfOpt - OptPos - TargetRegisterAllocMode.size());
      auto ColonPos = OptValue.find(":");
      auto Device = OptValue.substr(0, ColonPos);
      std::string BackendStrToAdd;
      bool IsPVC =
          std::all_of(Devs.begin(), Devs.end(), [&](const device &Dev) {
            return IsIntelGPU &&
                   (Dev.get_info<ext::intel::info::device::device_id>() &
                    0xFF00) == 0x0B00;
          });
      // Currently 'pvc' is the only supported device.
      if (Device == "pvc" && IsPVC)
        BackendStrToAdd = " " + OptValue.substr(ColonPos + 1) + " ";

      // Extract everything before this option
      std::string NewCompileOpts =
          CompileOpts.substr(0, OptPos) + BackendStrToAdd;
      // Extract everything after this option and add it to the above.
      if (EndOfOpt != std::string::npos)
        NewCompileOpts += CompileOpts.substr(EndOfOpt);
      CompileOpts = NewCompileOpts;
      OptPos = CompileOpts.find(TargetRegisterAllocMode);
    }
  }
}

static void
appendCompileEnvironmentVariablesThatAppend(std::string &CompileOpts) {
  static const char *AppendCompileOptsEnv =
      SYCLConfig<SYCL_PROGRAM_APPEND_COMPILE_OPTIONS>::get();
  if (AppendCompileOptsEnv) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    CompileOpts += AppendCompileOptsEnv;
  }
}
static void appendLinkEnvironmentVariablesThatAppend(std::string &LinkOpts) {
  static const char *AppendLinkOptsEnv =
      SYCLConfig<SYCL_PROGRAM_APPEND_LINK_OPTIONS>::get();
  if (AppendLinkOptsEnv) {
    if (!LinkOpts.empty())
      LinkOpts += " ";
    LinkOpts += AppendLinkOptsEnv;
  }
}

static void applyOptionsFromImage(std::string &CompileOpts,
                                  std::string &LinkOpts,
                                  const RTDeviceBinaryImage &Img,
                                  const std::vector<device> &Devices,
                                  const PluginPtr &Plugin) {
  appendCompileOptionsFromImage(CompileOpts, Img, Devices, Plugin);
  appendLinkOptionsFromImage(LinkOpts, Img);
}

static void applyCompileOptionsFromEnvironment(std::string &CompileOpts) {
  // Environment variables are not changed during program lifecycle so it
  // is reasonable to use static here to read them only once.
  static const char *CompileOptsEnv =
      SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  if (CompileOptsEnv) {
    CompileOpts = CompileOptsEnv;
  }
}

static void applyLinkOptionsFromEnvironment(std::string &LinkOpts) {
  // Environment variables are not changed during program lifecycle so it
  // is reasonable to use static here to read them only once.
  static const char *LinkOptsEnv = SYCLConfig<SYCL_PROGRAM_LINK_OPTIONS>::get();
  if (LinkOptsEnv) {
    LinkOpts = LinkOptsEnv;
  }
}

static void applyOptionsFromEnvironment(std::string &CompileOpts,
                                        std::string &LinkOpts) {
  // Build options are overridden if environment variables are present.
  applyCompileOptionsFromEnvironment(CompileOpts);
  applyLinkOptionsFromEnvironment(LinkOpts);
}

std::pair<sycl::detail::pi::PiProgram, bool>
ProgramManager::getOrCreatePIProgram(const RTDeviceBinaryImage &Img,
                                     const context &Context,
                                     const device &Device,
                                     const std::string &CompileAndLinkOptions,
                                     SerializedObj SpecConsts) {
  sycl::detail::pi::PiProgram NativePrg;

  auto BinProg = PersistentDeviceCodeCache::getItemFromDisc(
      Device, Img, SpecConsts, CompileAndLinkOptions);
  if (BinProg.size()) {
    // Get program metadata from properties
    auto ProgMetadata = Img.getProgramMetadata();
    std::vector<pi_device_binary_property> ProgMetadataVector{
        ProgMetadata.begin(), ProgMetadata.end()};

    // TODO: Build for multiple devices once supported by program manager
    NativePrg = createBinaryProgram(getSyclObjImpl(Context), Device,
                                    (const unsigned char *)BinProg[0].data(),
                                    BinProg[0].size(), ProgMetadataVector);
  } else {
    NativePrg = createPIProgram(Img, Context, Device);
  }
  return {NativePrg, BinProg.size()};
}

/// Emits information about built programs if the appropriate contitions are
/// met, namely when SYCL_RT_WARNING_LEVEL is greater than or equal to 2.
static void emitBuiltProgramInfo(const pi_program &Prog,
                                 const ContextImplPtr &Context) {
  if (SYCLConfig<SYCL_RT_WARNING_LEVEL>::get() >= 2) {
    std::string ProgramBuildLog =
        ProgramManager::getProgramBuildLog(Prog, Context);
    std::clog << ProgramBuildLog << std::endl;
  }
}

// When caching is enabled, the returned PiProgram will already have
// its ref count incremented.
sycl::detail::pi::PiProgram ProgramManager::getBuiltPIProgram(
    const ContextImplPtr &ContextImpl, const DeviceImplPtr &DeviceImpl,
    const std::string &KernelName, const NDRDescT &NDRDesc,
    bool JITCompilationIsRequired) {
  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  std::string CompileOpts;
  std::string LinkOpts;

  applyOptionsFromEnvironment(CompileOpts, LinkOpts);

  SerializedObj SpecConsts;

  // Check if we can optimize program builds for sub-devices by using a program
  // built for the root device
  DeviceImplPtr RootDevImpl = DeviceImpl;
  while (!RootDevImpl->isRootDevice()) {
    auto ParentDev = detail::getSyclObjImpl(
        RootDevImpl->get_info<info::device::parent_device>());
    // Sharing is allowed within a single context only
    if (!ContextImpl->hasDevice(ParentDev))
      break;
    RootDevImpl = ParentDev;
  }

  pi_bool MustBuildOnSubdevice = PI_TRUE;
  ContextImpl->getPlugin()->call<PiApiKind::piDeviceGetInfo>(
      RootDevImpl->getHandleRef(), PI_DEVICE_INFO_BUILD_ON_SUBDEVICE,
      sizeof(pi_bool), &MustBuildOnSubdevice, nullptr);

  DeviceImplPtr Dev =
      (MustBuildOnSubdevice == PI_TRUE) ? DeviceImpl : RootDevImpl;
  auto Context = createSyclObjFromImpl<context>(ContextImpl);
  auto Device = createSyclObjFromImpl<device>(Dev);
  const RTDeviceBinaryImage &Img =
      getDeviceImage(KernelName, Context, Device, JITCompilationIsRequired);

  // Check that device supports all aspects used by the kernel
  if (auto exception = checkDevSupportDeviceRequirements(Device, Img, NDRDesc))
    throw *exception;

  auto BuildF = [this, &Img, &Context, &ContextImpl, &Device, &CompileOpts,
                 &LinkOpts, SpecConsts] {
    const PluginPtr &Plugin = ContextImpl->getPlugin();
    applyOptionsFromImage(CompileOpts, LinkOpts, Img, {Device}, Plugin);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOpts);
    appendLinkEnvironmentVariablesThatAppend(LinkOpts);
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreatePIProgram(
        Img, Context, Device, CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache) {
      if (Img.supportsSpecConstants())
        enableITTAnnotationsIfNeeded(NativePrg, Plugin);
    }

    ProgramPtr ProgramManaged(
        NativePrg, Plugin->getPiPlugin().PiFunctionTable.piProgramRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs (after AOT compilation or read from persitent
    // cache) are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    if (!DeviceCodeWasInCache &&
        Img.getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV &&
        !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get())
      DeviceLibReqMask = getDeviceLibReqMask(Img);

    ProgramPtr BuiltProgram =
        build(std::move(ProgramManaged), ContextImpl, CompileOpts, LinkOpts,
              getRawSyclObjImpl(Device)->getHandleRef(), DeviceLibReqMask);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms[BuiltProgram.get()] = &Img;
    }

    ContextImpl->addDeviceGlobalInitializer(BuiltProgram.get(), {Device}, &Img);

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache)
      PersistentDeviceCodeCache::putItemToDisc(
          Device, Img, SpecConsts, CompileOpts + LinkOpts, BuiltProgram.get());
    return BuiltProgram.release();
  };

  uint32_t ImgId = Img.getImageID();
  const sycl::detail::pi::PiDevice PiDevice = Dev->getHandleRef();
  auto CacheKey =
      std::make_pair(std::make_pair(std::move(SpecConsts), ImgId), PiDevice);

  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get())
    return BuildF();

  auto BuildResult =
      Cache.getOrBuild<compile_program_error>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  // If caching is enabled, one copy of the program handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // program.
  ContextImpl->getPlugin()->call<PiApiKind::piProgramRetain>(BuildResult->Val);
  return BuildResult->Val;
}

// When caching is enabled, the returned PiProgram and PiKernel will
// already have their ref count incremented.
std::tuple<sycl::detail::pi::PiKernel, std::mutex *, const KernelArgMask *,
           sycl::detail::pi::PiProgram>
ProgramManager::getOrCreateKernel(const ContextImplPtr &ContextImpl,
                                  const DeviceImplPtr &DeviceImpl,
                                  const std::string &KernelName,
                                  const NDRDescT &NDRDesc) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << ContextImpl.get()
              << ", " << DeviceImpl.get() << ", " << KernelName << ")\n";
  }

  using KernelArgMaskPairT = KernelProgramCache::KernelArgMaskPairT;

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  std::string CompileOpts, LinkOpts;
  SerializedObj SpecConsts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);
  // Should always come last!
  appendCompileEnvironmentVariablesThatAppend(CompileOpts);
  appendLinkEnvironmentVariablesThatAppend(LinkOpts);
  const sycl::detail::pi::PiDevice PiDevice = DeviceImpl->getHandleRef();

  auto key = std::make_tuple(std::move(SpecConsts), PiDevice,
                             CompileOpts + LinkOpts, KernelName);
  if (SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    auto ret_tuple = Cache.tryToGetKernelFast(key);
    constexpr size_t Kernel = 0;  // see KernelFastCacheValT tuple
    constexpr size_t Program = 3; // see KernelFastCacheValT tuple
    if (std::get<Kernel>(ret_tuple)) {
      // Pulling a copy of a kernel and program from the cache,
      // so we need to retain those resources.
      ContextImpl->getPlugin()->call<PiApiKind::piKernelRetain>(
          std::get<Kernel>(ret_tuple));
      ContextImpl->getPlugin()->call<PiApiKind::piProgramRetain>(
          std::get<Program>(ret_tuple));
      return ret_tuple;
    }
  }

  sycl::detail::pi::PiProgram Program =
      getBuiltPIProgram(ContextImpl, DeviceImpl, KernelName, NDRDesc);

  auto BuildF = [this, &Program, &KernelName, &ContextImpl] {
    sycl::detail::pi::PiKernel Kernel = nullptr;

    const PluginPtr &Plugin = ContextImpl->getPlugin();
    Plugin->call<errc::kernel_not_supported, PiApiKind::piKernelCreate>(
        Program, KernelName.c_str(), &Kernel);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    Plugin->call<PiApiKind::piKernelSetExecInfo>(Kernel, PI_USM_INDIRECT_ACCESS,
                                                 sizeof(pi_bool), &PI_TRUE);

    const KernelArgMask *ArgMask = nullptr;
    if (!m_UseSpvFile)
      ArgMask = getEliminatedKernelArgMask(Program, KernelName);
    return std::make_pair(Kernel, ArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    // The built kernel cannot be shared between multiple
    // threads when caching is disabled, so we can return
    // nullptr for the mutex.
    auto [Kernel, ArgMask] = BuildF();
    return make_tuple(Kernel, nullptr, ArgMask, Program);
  }

  auto BuildResult =
      Cache.getOrBuild<invalid_object_error>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  const KernelArgMaskPairT &KernelArgMaskPair = BuildResult->Val;
  auto ret_val = std::make_tuple(KernelArgMaskPair.first,
                                 &(BuildResult->MBuildResultMutex),
                                 KernelArgMaskPair.second, Program);
  // If caching is enabled, one copy of the kernel handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // kernel.
  ContextImpl->getPlugin()->call<PiApiKind::piKernelRetain>(
      KernelArgMaskPair.first);
  Cache.saveKernel(key, ret_val);
  return ret_val;
}

sycl::detail::pi::PiProgram
ProgramManager::getPiProgramFromPiKernel(sycl::detail::pi::PiKernel Kernel,
                                         const ContextImplPtr Context) {
  sycl::detail::pi::PiProgram Program;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piKernelGetInfo>(Kernel, PI_KERNEL_INFO_PROGRAM,
                                           sizeof(sycl::detail::pi::PiProgram),
                                           &Program, nullptr);
  return Program;
}

std::string
ProgramManager::getProgramBuildLog(const sycl::detail::pi::PiProgram &Program,
                                   const ContextImplPtr Context) {
  size_t PIDevicesSize = 0;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, 0,
                                            nullptr, &PIDevicesSize);
  std::vector<sycl::detail::pi::PiDevice> PIDevices(
      PIDevicesSize / sizeof(sycl::detail::pi::PiDevice));
  Plugin->call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES,
                                            PIDevicesSize, PIDevices.data(),
                                            nullptr);
  std::string Log = "The program was built for " +
                    std::to_string(PIDevices.size()) + " devices";
  for (sycl::detail::pi::PiDevice &Device : PIDevices) {
    std::string DeviceBuildInfoString;
    size_t DeviceBuildInfoStrSize = 0;
    Plugin->call<PiApiKind::piProgramGetBuildInfo>(
        Program, Device, PI_PROGRAM_BUILD_INFO_LOG, 0, nullptr,
        &DeviceBuildInfoStrSize);
    if (DeviceBuildInfoStrSize > 0) {
      std::vector<char> DeviceBuildInfo(DeviceBuildInfoStrSize);
      Plugin->call<PiApiKind::piProgramGetBuildInfo>(
          Program, Device, PI_PROGRAM_BUILD_INFO_LOG, DeviceBuildInfoStrSize,
          DeviceBuildInfo.data(), nullptr);
      DeviceBuildInfoString = std::string(DeviceBuildInfo.data());
    }

    std::string DeviceNameString;
    size_t DeviceNameStrSize = 0;
    Plugin->call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME, 0,
                                             nullptr, &DeviceNameStrSize);
    if (DeviceNameStrSize > 0) {
      std::vector<char> DeviceName(DeviceNameStrSize);
      Plugin->call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME,
                                               DeviceNameStrSize,
                                               DeviceName.data(), nullptr);
      DeviceNameString = std::string(DeviceName.data());
    }
    Log += "\nBuild program log for '" + DeviceNameString + "':\n" +
           DeviceBuildInfoString;
  }
  return Log;
}

// TODO device libraries may use scpecialization constants, manifest files, etc.
// To support that they need to be delivered in a different container - so that
// pi_device_binary_struct can be created for each of them.
static bool loadDeviceLib(const ContextImplPtr Context, const char *Name,
                          sycl::detail::pi::PiProgram &Prog) {
  std::string LibSyclDir = OSUtil::getCurrentDSODir();
  std::ifstream File(LibSyclDir + OSUtil::DirSep + Name,
                     std::ifstream::in | std::ifstream::binary);
  if (!File.good()) {
    return false;
  }

  File.seekg(0, std::ios::end);
  size_t FileSize = File.tellg();
  File.seekg(0, std::ios::beg);
  std::vector<char> FileContent(FileSize);
  File.read(&FileContent[0], FileSize);
  File.close();

  Prog =
      createSpirvProgram(Context, (unsigned char *)&FileContent[0], FileSize);
  return Prog != nullptr;
}

// For each extension, a pair of library names. The first uses native support,
// the second emulates functionality in software.
static const std::map<DeviceLibExt, std::pair<const char *, const char *>>
    DeviceLibNames = {
        {DeviceLibExt::cl_intel_devicelib_assert,
         {nullptr, "libsycl-fallback-cassert.spv"}},
        {DeviceLibExt::cl_intel_devicelib_math,
         {nullptr, "libsycl-fallback-cmath.spv"}},
        {DeviceLibExt::cl_intel_devicelib_math_fp64,
         {nullptr, "libsycl-fallback-cmath-fp64.spv"}},
        {DeviceLibExt::cl_intel_devicelib_complex,
         {nullptr, "libsycl-fallback-complex.spv"}},
        {DeviceLibExt::cl_intel_devicelib_complex_fp64,
         {nullptr, "libsycl-fallback-complex-fp64.spv"}},
        {DeviceLibExt::cl_intel_devicelib_cstring,
         {nullptr, "libsycl-fallback-cstring.spv"}},
        {DeviceLibExt::cl_intel_devicelib_imf,
         {nullptr, "libsycl-fallback-imf.spv"}},
        {DeviceLibExt::cl_intel_devicelib_imf_fp64,
         {nullptr, "libsycl-fallback-imf-fp64.spv"}},
        {DeviceLibExt::cl_intel_devicelib_imf_bf16,
         {nullptr, "libsycl-fallback-imf-bf16.spv"}},
        {DeviceLibExt::cl_intel_devicelib_bfloat16,
         {"libsycl-native-bfloat16.spv", "libsycl-fallback-bfloat16.spv"}}};

static const char *getDeviceLibFilename(DeviceLibExt Extension, bool Native) {
  auto LibPair = DeviceLibNames.find(Extension);
  const char *Lib = nullptr;
  if (LibPair != DeviceLibNames.end())
    Lib = Native ? LibPair->second.first : LibPair->second.second;
  if (Lib == nullptr)
    throw compile_program_error("Unhandled (new?) device library extension",
                                PI_ERROR_INVALID_OPERATION);
  return Lib;
}

// For each extension understood by the SYCL runtime, the string representation
// of its name. Names with devicelib in them are internal to the runtime. Others
// are actual OpenCL extensions.
static const std::map<DeviceLibExt, const char *> DeviceLibExtensionStrs = {
    {DeviceLibExt::cl_intel_devicelib_assert, "cl_intel_devicelib_assert"},
    {DeviceLibExt::cl_intel_devicelib_math, "cl_intel_devicelib_math"},
    {DeviceLibExt::cl_intel_devicelib_math_fp64,
     "cl_intel_devicelib_math_fp64"},
    {DeviceLibExt::cl_intel_devicelib_complex, "cl_intel_devicelib_complex"},
    {DeviceLibExt::cl_intel_devicelib_complex_fp64,
     "cl_intel_devicelib_complex_fp64"},
    {DeviceLibExt::cl_intel_devicelib_cstring, "cl_intel_devicelib_cstring"},
    {DeviceLibExt::cl_intel_devicelib_imf, "cl_intel_devicelib_imf"},
    {DeviceLibExt::cl_intel_devicelib_imf_fp64, "cl_intel_devicelib_imf_fp64"},
    {DeviceLibExt::cl_intel_devicelib_imf_bf16, "cl_intel_devicelib_imf_bf16"},
    {DeviceLibExt::cl_intel_devicelib_bfloat16,
     "cl_intel_bfloat16_conversions"}};

static const char *getDeviceLibExtensionStr(DeviceLibExt Extension) {
  auto Ext = DeviceLibExtensionStrs.find(Extension);
  if (Ext == DeviceLibExtensionStrs.end())
    throw compile_program_error("Unhandled (new?) device library extension",
                                PI_ERROR_INVALID_OPERATION);
  return Ext->second;
}

static sycl::detail::pi::PiProgram
loadDeviceLibFallback(const ContextImplPtr Context, DeviceLibExt Extension,
                      const sycl::detail::pi::PiDevice &Device,
                      bool UseNativeLib) {

  auto LibFileName = getDeviceLibFilename(Extension, UseNativeLib);

  auto LockedCache = Context->acquireCachedLibPrograms();
  auto CachedLibPrograms = LockedCache.get();
  auto CacheResult = CachedLibPrograms.emplace(
      std::make_pair(std::make_pair(Extension, Device), nullptr));
  bool Cached = !CacheResult.second;
  auto LibProgIt = CacheResult.first;
  sycl::detail::pi::PiProgram &LibProg = LibProgIt->second;

  if (Cached)
    return LibProg;

  if (!loadDeviceLib(Context, LibFileName, LibProg)) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(std::string("Failed to load ") + LibFileName,
                                PI_ERROR_INVALID_VALUE);
  }

  const PluginPtr &Plugin = Context->getPlugin();
  // TODO no spec constants are used in the std libraries, support in the future
  sycl::detail::pi::PiResult Error =
      Plugin->call_nocheck<PiApiKind::piProgramCompile>(
          LibProg,
          /*num devices = */ 1, &Device,
          // Do not use compile options for library programs: it is not clear
          // if user options (image options) are supposed to be applied to
          // library program as well, and what actually happens to a SPIR-V
          // program if we apply them.
          "", 0, nullptr, nullptr, nullptr, nullptr);
  if (Error != PI_SUCCESS) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(
        ProgramManager::getProgramBuildLog(LibProg, Context), Error);
  }

  return LibProg;
}

ProgramManager::ProgramManager() : m_AsanFoundInImage(false) {
  const char *SpvFile = std::getenv(UseSpvEnv);
  // If a SPIR-V file is specified with an environment variable,
  // register the corresponding image
  if (SpvFile) {
    m_UseSpvFile = true;
    // The env var requests that the program is loaded from a SPIR-V file on
    // disk
    std::ifstream File(SpvFile, std::ios::binary);

    if (!File.is_open())
      throw runtime_error(std::string("Can't open file specified via ") +
                              UseSpvEnv + ": " + SpvFile,
                          PI_ERROR_INVALID_VALUE);
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    std::unique_ptr<char[]> Data(new char[Size]);
    File.seekg(0);
    File.read(Data.get(), Size);
    File.close();
    if (!File.good())
      throw runtime_error(std::string("read from ") + SpvFile +
                              std::string(" failed"),
                          PI_ERROR_INVALID_VALUE);
    // No need for a mutex here since all access to these private fields is
    // blocked until the construction of the ProgramManager singleton is
    // finished.
    m_SpvFileImage =
        make_unique_ptr<DynRTDeviceBinaryImage>(std::move(Data), Size);

    if (DbgProgMgr > 0) {
      std::cerr << "loaded device image binary from " << SpvFile << "\n";
      std::cerr << "format: " << getFormatStr(m_SpvFileImage->getFormat())
                << "\n";
    }
  }
}

void CheckJITCompilationForImage(const RTDeviceBinaryImage *const &Image,
                                 bool JITCompilationIsRequired) {
  if (!JITCompilationIsRequired)
    return;
  // If the image is already compiled with AOT, throw an exception.
  const pi_device_binary_struct &RawImg = Image->getRawData();
  if ((strcmp(RawImg.DeviceTargetSpec,
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
      (strcmp(RawImg.DeviceTargetSpec,
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
      (strcmp(RawImg.DeviceTargetSpec,
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0)) {
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Recompiling AOT image is not supported");
  }
}

template <typename StorageKey>
RTDeviceBinaryImage *getBinImageFromMultiMap(
    const std::unordered_multimap<StorageKey, RTDeviceBinaryImage *> &ImagesSet,
    const StorageKey &Key, const context &Context, const device &Device) {
  auto [ItBegin, ItEnd] = ImagesSet.equal_range(Key);
  if (ItBegin == ItEnd)
    return nullptr;

  std::vector<pi_device_binary> RawImgs(std::distance(ItBegin, ItEnd));
  auto It = ItBegin;
  for (unsigned I = 0; It != ItEnd; ++It, ++I)
    RawImgs[I] = const_cast<pi_device_binary>(&It->second->getRawData());

  pi_uint32 ImgInd = 0;
  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  getSyclObjImpl(Context)
      ->getPlugin()
      ->call<PiApiKind::piextDeviceSelectBinary>(
          getSyclObjImpl(Device)->getHandleRef(), RawImgs.data(),
          (pi_uint32)RawImgs.size(), &ImgInd);
  std::advance(ItBegin, ImgInd);
  return ItBegin->second;
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(const std::string &KernelName,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(\"" << KernelName << "\", "
              << getRawSyclObjImpl(Context) << ", " << getRawSyclObjImpl(Device)
              << ", " << JITCompilationIsRequired << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }

  if (m_UseSpvFile) {
    assert(m_SpvFileImage);
    return getDeviceImage(
        std::unordered_set<RTDeviceBinaryImage *>({m_SpvFileImage.get()}),
        Context, Device, JITCompilationIsRequired);
  }

  RTDeviceBinaryImage *Img = nullptr;
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    if (auto KernelId = m_KernelName2KernelIDs.find(KernelName);
        KernelId != m_KernelName2KernelIDs.end()) {
      // Kernel ID presence guarantees that we have bin image in the storage.
      Img = getBinImageFromMultiMap(m_KernelIDs2BinImage, KernelId->second,
                                    Context, Device);
      assert(Img && "No binary image found for kernel id");
    } else {
      Img = getBinImageFromMultiMap(m_ServiceKernels, KernelName, Context,
                                    Device);
    }
  }
  if (Img) {
    CheckJITCompilationForImage(Img, JITCompilationIsRequired);

    if (DbgProgMgr > 0) {
      std::cerr << "selected device image: " << &Img->getRawData() << "\n";
      Img->print();
    }
    return *Img;
  }

  throw runtime_error("No kernel named " + KernelName + " was found",
                      PI_ERROR_INVALID_KERNEL_NAME);
}

RTDeviceBinaryImage &ProgramManager::getDeviceImage(
    const std::unordered_set<RTDeviceBinaryImage *> &ImageSet,
    const context &Context, const device &Device,
    bool JITCompilationIsRequired) {
  assert(ImageSet.size() > 0);

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(Custom SPV file "
              << getRawSyclObjImpl(Context) << ", " << getRawSyclObjImpl(Device)
              << ", " << JITCompilationIsRequired << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }

  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
  std::vector<pi_device_binary> RawImgs(ImageSet.size());
  auto ImageIterator = ImageSet.begin();
  for (size_t i = 0; i < ImageSet.size(); i++, ImageIterator++)
    RawImgs[i] = const_cast<pi_device_binary>(&(*ImageIterator)->getRawData());
  pi_uint32 ImgInd = 0;
  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  getSyclObjImpl(Context)
      ->getPlugin()
      ->call<PiApiKind::piextDeviceSelectBinary>(
          getSyclObjImpl(Device)->getHandleRef(), RawImgs.data(),
          (pi_uint32)RawImgs.size(), &ImgInd);

  ImageIterator = ImageSet.begin();
  std::advance(ImageIterator, ImgInd);

  CheckJITCompilationForImage(*ImageIterator, JITCompilationIsRequired);

  if (DbgProgMgr > 0) {
    std::cerr << "selected device image: " << &(*ImageIterator)->getRawData()
              << "\n";
    (*ImageIterator)->print();
  }
  return **ImageIterator;
}

static bool isDeviceLibRequired(DeviceLibExt Ext, uint32_t DeviceLibReqMask) {
  uint32_t Mask =
      0x1 << (static_cast<uint32_t>(Ext) -
              static_cast<uint32_t>(DeviceLibExt::cl_intel_devicelib_assert));
  return ((DeviceLibReqMask & Mask) == Mask);
}

static std::vector<sycl::detail::pi::PiProgram>
getDeviceLibPrograms(const ContextImplPtr Context,
                     const sycl::detail::pi::PiDevice &Device,
                     uint32_t DeviceLibReqMask) {
  std::vector<sycl::detail::pi::PiProgram> Programs;

  std::pair<DeviceLibExt, bool> RequiredDeviceLibExt[] = {
      {DeviceLibExt::cl_intel_devicelib_assert,
       /* is fallback loaded? */ false},
      {DeviceLibExt::cl_intel_devicelib_math, false},
      {DeviceLibExt::cl_intel_devicelib_math_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_complex, false},
      {DeviceLibExt::cl_intel_devicelib_complex_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_cstring, false},
      {DeviceLibExt::cl_intel_devicelib_imf, false},
      {DeviceLibExt::cl_intel_devicelib_imf_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_imf_bf16, false},
      {DeviceLibExt::cl_intel_devicelib_bfloat16, false}};

  // Disable all devicelib extensions requiring fp64 support if at least
  // one underlying device doesn't support cl_khr_fp64.
  std::string DevExtList =
      Context->getPlatformImpl()->getDeviceImpl(Device)->get_device_info_string(
          PiInfoCode<info::device::extensions>::value);
  const bool fp64Support = (DevExtList.npos != DevExtList.find("cl_khr_fp64"));

  // Load a fallback library for an extension if the device does not
  // support it.
  for (auto &Pair : RequiredDeviceLibExt) {
    DeviceLibExt Ext = Pair.first;
    bool &FallbackIsLoaded = Pair.second;

    if (FallbackIsLoaded) {
      continue;
    }

    if (!isDeviceLibRequired(Ext, DeviceLibReqMask)) {
      continue;
    }

    if ((Ext == DeviceLibExt::cl_intel_devicelib_math_fp64 ||
         Ext == DeviceLibExt::cl_intel_devicelib_complex_fp64 ||
         Ext == DeviceLibExt::cl_intel_devicelib_imf_fp64) &&
        !fp64Support) {
      continue;
    }

    auto ExtName = getDeviceLibExtensionStr(Ext);

    bool InhibitNativeImpl = false;
    if (const char *Env = getenv("SYCL_DEVICELIB_INHIBIT_NATIVE")) {
      InhibitNativeImpl = strstr(Env, ExtName) != nullptr;
    }

    bool DeviceSupports = DevExtList.npos != DevExtList.find(ExtName);
    if (!DeviceSupports || InhibitNativeImpl) {
      Programs.push_back(
          loadDeviceLibFallback(Context, Ext, Device, /*UseNativeLib=*/false));
      FallbackIsLoaded = true;
    } else {
      // bfloat16 needs native library if device supports it
      if (Ext == DeviceLibExt::cl_intel_devicelib_bfloat16) {
        Programs.push_back(
            loadDeviceLibFallback(Context, Ext, Device, /*UseNativeLib=*/true));
        FallbackIsLoaded = true;
      }
    }
  }
  return Programs;
}

ProgramManager::ProgramPtr ProgramManager::build(
    ProgramPtr Program, const ContextImplPtr Context,
    const std::string &CompileOptions, const std::string &LinkOptions,
    const sycl::detail::pi::PiDevice &Device, uint32_t DeviceLibReqMask) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", ... " << Device
              << ")\n";
  }

  bool LinkDeviceLibs = (DeviceLibReqMask != 0);

  // TODO: this is a temporary workaround for GPU tests for ESIMD compiler.
  // We do not link with other device libraries, because it may fail
  // due to unrecognized SPIR-V format of those libraries.
  if (CompileOptions.find(std::string("-cmc")) != std::string::npos ||
      CompileOptions.find(std::string("-vc-codegen")) != std::string::npos)
    LinkDeviceLibs = false;

  std::vector<sycl::detail::pi::PiProgram> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Device, DeviceLibReqMask);
  }

  static const char *ForceLinkEnv = std::getenv("SYCL_FORCE_LINK");
  static bool ForceLink = ForceLinkEnv && (*ForceLinkEnv == '1');

  const PluginPtr &Plugin = Context->getPlugin();
  if (LinkPrograms.empty() && !ForceLink) {
    const std::string &Options = LinkOptions.empty()
                                     ? CompileOptions
                                     : (CompileOptions + " " + LinkOptions);
    sycl::detail::pi::PiResult Error =
        Plugin->call_nocheck<PiApiKind::piProgramBuild>(
            Program.get(), /*num devices =*/1, &Device, Options.c_str(),
            nullptr, nullptr);
    if (Error != PI_SUCCESS)
      throw compile_program_error(getProgramBuildLog(Program.get(), Context),
                                  Error);
    return Program;
  }

  // Include the main program and compile/link everything together
  Plugin->call<PiApiKind::piProgramCompile>(Program.get(), /*num devices =*/1,
                                            &Device, CompileOptions.c_str(), 0,
                                            nullptr, nullptr, nullptr, nullptr);
  LinkPrograms.push_back(Program.get());

  sycl::detail::pi::PiProgram LinkedProg = nullptr;
  auto doLink = [&] {
    return Plugin->call_nocheck<PiApiKind::piProgramLink>(
        Context->getHandleRef(), /*num devices =*/1, &Device,
        LinkOptions.c_str(), LinkPrograms.size(), LinkPrograms.data(), nullptr,
        nullptr, &LinkedProg);
  };
  sycl::detail::pi::PiResult Error = doLink();
  if (Error == PI_ERROR_OUT_OF_RESOURCES) {
    Context->getKernelProgramCache().reset();
    Error = doLink();
  }

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise. Release the original (user) program.
  Program.reset(LinkedProg);
  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg, and throw.
      throw compile_program_error(getProgramBuildLog(LinkedProg, Context),
                                  Error);
    }
    Plugin->checkPiResult(Error);
  }
  return Program;
}

void ProgramManager::cacheKernelUsesAssertInfo(RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &AssertUsedRange =
      Img.getAssertUsed();
  if (AssertUsedRange.isAvailable())
    for (const auto &Prop : AssertUsedRange)
      m_KernelUsesAssert.insert(Prop->Name);
}

bool ProgramManager::kernelUsesAssert(const std::string &KernelName) const {
  return m_KernelUsesAssert.find(KernelName) != m_KernelUsesAssert.end();
}

void ProgramManager::addImages(pi_device_binaries DeviceBinary) {
  const bool DumpImages = std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile;
  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);
    const _pi_offload_entry EntriesB = RawImg->EntriesBegin;
    const _pi_offload_entry EntriesE = RawImg->EntriesEnd;
    // Treat the image as empty one
    if (EntriesB == EntriesE)
      continue;

    auto Img = make_unique_ptr<RTDeviceBinaryImage>(RawImg);
    static uint32_t SequenceID = 0;

    // Fill the kernel argument mask map
    const RTDeviceBinaryImage::PropertyRange &KPOIRange =
        Img->getKernelParamOptInfo();
    if (KPOIRange.isAvailable()) {
      KernelNameToArgMaskMap &ArgMaskMap =
          m_EliminatedKernelArgMasks[Img.get()];
      for (const auto &Info : KPOIRange)
        ArgMaskMap[Info->Name] =
            createKernelArgMask(DeviceBinaryProperty(Info).asByteArray());
    }

    // Fill maps for kernel bundles
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

    // Register all exported symbols
    auto ExportedSymbols = Img->getExportedSymbols();
    for (const pi_device_binary_property &ExportedSymbol : ExportedSymbols)
      m_ExportedSymbols.insert(ExportedSymbol->Name);

    if (DumpImages) {
      const bool NeedsSequenceID = std::any_of(
          m_BinImg2KernelIDs.begin(), m_BinImg2KernelIDs.end(),
          [&](auto &CurrentImg) {
            return CurrentImg.first->getFormat() == Img->getFormat();
          });
      dumpImage(*Img, NeedsSequenceID ? ++SequenceID : 0);
    }

    m_BinImg2KernelIDs[Img.get()].reset(new std::vector<kernel_id>);

    for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
         ++EntriesIt) {

      // Skip creating unique kernel ID if it is a service kernel.
      // SYCL service kernels are identified by having
      // __sycl_service_kernel__ in the mangled name, primarily as part of
      // the namespace of the name type.
      if (std::strstr(EntriesIt->name, "__sycl_service_kernel__")) {
        m_ServiceKernels.insert(std::make_pair(EntriesIt->name, Img.get()));
        continue;
      }

      // Skip creating unique kernel ID if it is an exported device
      // function. Exported device functions appear in the offload entries
      // among kernels, but are identifiable by being listed in properties.
      if (m_ExportedSymbols.find(EntriesIt->name) != m_ExportedSymbols.end())
        continue;

      // ... and create a unique kernel ID for the entry
      auto It = m_KernelName2KernelIDs.find(EntriesIt->name);
      if (It == m_KernelName2KernelIDs.end()) {
        std::shared_ptr<detail::kernel_id_impl> KernelIDImpl =
            std::make_shared<detail::kernel_id_impl>(EntriesIt->name);
        sycl::kernel_id KernelID =
            detail::createSyclObjFromImpl<sycl::kernel_id>(KernelIDImpl);

        It = m_KernelName2KernelIDs.emplace_hint(It, EntriesIt->name, KernelID);
      }
      m_KernelIDs2BinImage.insert(std::make_pair(It->second, Img.get()));
      m_BinImg2KernelIDs[Img.get()]->push_back(It->second);
    }

    cacheKernelUsesAssertInfo(*Img);

    // check if kernel uses asan
    {
      pi_device_binary_property Prop = Img->getProperty("asanUsed");
      m_AsanFoundInImage |=
          Prop && (detail::DeviceBinaryProperty(Prop).asUint32() != 0);
    }

    // Sort kernel ids for faster search
    std::sort(m_BinImg2KernelIDs[Img.get()]->begin(),
              m_BinImg2KernelIDs[Img.get()]->end(), LessByHash<kernel_id>{});

    // ... and initialize associated device_global information
    {
      std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);

      auto DeviceGlobals = Img->getDeviceGlobals();
      for (const pi_device_binary_property &DeviceGlobal : DeviceGlobals) {
        ByteArray DeviceGlobalInfo =
            DeviceBinaryProperty(DeviceGlobal).asByteArray();

        // The supplied device_global info property is expected to contain:
        // * 8 bytes - Size of the property.
        // * 4 bytes - Size of the underlying type in the device_global.
        // * 4 bytes - 0 if device_global has device_image_scope and any value
        //             otherwise.
        DeviceGlobalInfo.dropBytes(8);
        auto [TypeSize, DeviceImageScopeDecorated] =
            DeviceGlobalInfo.consume<std::uint32_t, std::uint32_t>();
        assert(DeviceGlobalInfo.empty() && "Extra data left!");

        // Give the image pointer as an identifier for the image the
        // device-global is associated with.

        auto ExistingDeviceGlobal = m_DeviceGlobals.find(DeviceGlobal->Name);
        if (ExistingDeviceGlobal != m_DeviceGlobals.end()) {
          // If it has already been registered we update the information.
          ExistingDeviceGlobal->second->initialize(Img.get(), TypeSize,
                                                   DeviceImageScopeDecorated);
        } else {
          // If it has not already been registered we create a new entry.
          // Note: Pointer to the device global is not available here, so it
          //       cannot be set until registration happens.
          auto EntryUPtr = std::make_unique<DeviceGlobalMapEntry>(
              DeviceGlobal->Name, Img.get(), TypeSize,
              DeviceImageScopeDecorated);
          m_DeviceGlobals.emplace(DeviceGlobal->Name, std::move(EntryUPtr));
        }
      }
    }
    // ... and initialize associated host_pipe information
    {
      std::lock_guard<std::mutex> HostPipesGuard(m_HostPipesMutex);
      auto HostPipes = Img->getHostPipes();
      for (const pi_device_binary_property &HostPipe : HostPipes) {
        ByteArray HostPipeInfo = DeviceBinaryProperty(HostPipe).asByteArray();

        // The supplied host_pipe info property is expected to contain:
        // * 8 bytes - Size of the property.
        // * 4 bytes - Size of the underlying type in the host_pipe.
        // Note: Property may be padded.

        HostPipeInfo.dropBytes(8);
        auto TypeSize = HostPipeInfo.consume<std::uint32_t>();
        assert(HostPipeInfo.empty() && "Extra data left!");

        auto ExistingHostPipe = m_HostPipes.find(HostPipe->Name);
        if (ExistingHostPipe != m_HostPipes.end()) {
          // If it has already been registered we update the information.
          ExistingHostPipe->second->initialize(TypeSize);
          ExistingHostPipe->second->initialize(Img.get());
        } else {
          // If it has not already been registered we create a new entry.
          // Note: Pointer to the host pipe is not available here, so it
          //       cannot be set until registration happens.
          auto EntryUPtr =
              std::make_unique<HostPipeMapEntry>(HostPipe->Name, TypeSize);
          EntryUPtr->initialize(Img.get());
          m_HostPipes.emplace(HostPipe->Name, std::move(EntryUPtr));
        }
      }
    }
    m_DeviceImages.insert(std::move(Img));
  }
}

void ProgramManager::debugPrintBinaryImages() const {
  for (const auto &ImgIt : m_BinImg2KernelIDs) {
    ImgIt.first->print();
  }
}

void ProgramManager::dumpImage(const RTDeviceBinaryImage &Img,
                               uint32_t SequenceID) const {
  const char *Prefix = std::getenv("SYCL_DUMP_IMAGES_PREFIX");
  std::string Fname(Prefix ? Prefix : "sycl_");
  const pi_device_binary_struct &RawImg = Img.getRawData();
  Fname += RawImg.DeviceTargetSpec;
  if (SequenceID)
    Fname += '_' + std::to_string(SequenceID);
  std::string Ext;

  sycl::detail::pi::PiDeviceBinaryType Format = Img.getFormat();
  if (Format == PI_DEVICE_BINARY_TYPE_SPIRV)
    Ext = ".spv";
  else if (Format == PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE)
    Ext = ".bc";
  else
    Ext = ".bin";
  Fname += Ext;

  std::ofstream F(Fname, std::ios::binary);

  if (!F.is_open()) {
    throw runtime_error("Can not write " + Fname, PI_ERROR_UNKNOWN);
  }
  Img.dump(F);
  F.close();
}

void ProgramManager::flushSpecConstants(const program_impl &Prg,
                                        sycl::detail::pi::PiProgram NativePrg,
                                        const RTDeviceBinaryImage *Img) {
  if (DbgProgMgr > 2) {
    std::cerr << ">>> ProgramManager::flushSpecConstants(" << Prg.get()
              << ",...)\n";
  }
  if (!Prg.hasSetSpecConstants())
    return; // nothing to do
  pi::PiProgram PrgHandle = Prg.getHandleRef();
  // program_impl can't correspond to two different native programs
  assert(!NativePrg || !PrgHandle || (NativePrg == PrgHandle));
  NativePrg = NativePrg ? NativePrg : PrgHandle;

  if (!Img) {
    // caller hasn't provided the image object - find it
    { // make sure NativePrograms map access is synchronized
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      auto It = NativePrograms.find(NativePrg);
      if (It == NativePrograms.end())
        throw sycl::exception(
            sycl::errc::invalid,
            "spec constant is set in a program w/o a binary image");
      Img = It->second;
    }
    if (!Img->supportsSpecConstants()) {
      if (DbgProgMgr > 0)
        std::cerr << ">>> ProgramManager::flushSpecConstants: binary image "
                  << &Img->getRawData() << " doesn't support spec constants\n";
      // This device binary image does not support runtime setting of
      // specialization constants; compiler must have generated default values.
      // NOTE: Can't throw here, as it would always take place with AOT
      //-compiled code. New Khronos 2020 spec should fix this inconsistency.
      return;
    }
  }
  Prg.flush_spec_constants(*Img, NativePrg);
}

uint32_t ProgramManager::getDeviceLibReqMask(const RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &DLMRange =
      Img.getDeviceLibReqMask();
  if (DLMRange.isAvailable())
    return DeviceBinaryProperty(*(DLMRange.begin())).asUint32();
  else
    return 0x0;
}

const KernelArgMask *
ProgramManager::getEliminatedKernelArgMask(pi::PiProgram NativePrg,
                                           const std::string &KernelName) {
  // Bail out if there are no eliminated kernel arg masks in our images
  if (m_EliminatedKernelArgMasks.empty())
    return nullptr;

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    auto ImgIt = NativePrograms.find(NativePrg);
    if (ImgIt != NativePrograms.end()) {
      auto MapIt = m_EliminatedKernelArgMasks.find(ImgIt->second);
      if (MapIt != m_EliminatedKernelArgMasks.end()) {
        auto ArgMaskMapIt = MapIt->second.find(KernelName);
        if (ArgMaskMapIt != MapIt->second.end())
          return &MapIt->second[KernelName];
      }
      return nullptr;
    }
  }

  // If the program was not cached iterate over all available images looking for
  // the requested kernel
  for (auto &Elem : m_EliminatedKernelArgMasks) {
    auto ArgMask = Elem.second.find(KernelName);
    if (ArgMask != Elem.second.end())
      return &ArgMask->second;
  }

  // The kernel is not generated by DPCPP stack, so a mask doesn't exist for it
  return nullptr;
}

static bundle_state getBinImageState(const RTDeviceBinaryImage *BinImage) {
  auto IsAOTBinary = [](const char *Format) {
    return (
        (strcmp(Format, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
        (strcmp(Format, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
        (strcmp(Format, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0));
  };

  // There are only two initial states so far - SPIRV which needs to be compiled
  // and linked and fully compiled(AOTed) binary

  const bool IsAOT = IsAOTBinary(BinImage->getRawData().DeviceTargetSpec);

  return IsAOT ? sycl::bundle_state::executable : sycl::bundle_state::input;
}

static bool compatibleWithDevice(RTDeviceBinaryImage *BinImage,
                                 const device &Dev) {
  const std::shared_ptr<detail::device_impl> &DeviceImpl =
      detail::getSyclObjImpl(Dev);
  auto &Plugin = DeviceImpl->getPlugin();

  const sycl::detail::pi::PiDevice &PIDeviceHandle = DeviceImpl->getHandleRef();

  // Call piextDeviceSelectBinary with only one image to check if an image is
  // compatible with implementation. The function returns invalid index if no
  // device images are compatible.
  pi_uint32 SuitableImageID = std::numeric_limits<pi_uint32>::max();
  pi_device_binary DevBin =
      const_cast<pi_device_binary>(&BinImage->getRawData());
  sycl::detail::pi::PiResult Error =
      Plugin->call_nocheck<PiApiKind::piextDeviceSelectBinary>(
          PIDeviceHandle, &DevBin,
          /*num bin images = */ (pi_uint32)1, &SuitableImageID);
  if (Error != PI_SUCCESS && Error != PI_ERROR_INVALID_BINARY)
    throw runtime_error("Invalid binary image or device",
                        PI_ERROR_INVALID_VALUE);

  return (0 == SuitableImageID);
}

kernel_id ProgramManager::getSYCLKernelID(const std::string &KernelName) {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  auto KernelID = m_KernelName2KernelIDs.find(KernelName);
  if (KernelID == m_KernelName2KernelIDs.end())
    throw runtime_error("No kernel found with the specified name",
                        PI_ERROR_INVALID_KERNEL_NAME);

  return KernelID->second;
}

bool ProgramManager::hasCompatibleImage(const device &Dev) {
  std::lock_guard<std::mutex> Guard(m_KernelIDsMutex);

  return std::any_of(
      m_BinImg2KernelIDs.cbegin(), m_BinImg2KernelIDs.cend(),
      [&](std::pair<RTDeviceBinaryImage *,
                    std::shared_ptr<std::vector<kernel_id>>>
              Elem) { return compatibleWithDevice(Elem.first, Dev); });
}

std::vector<kernel_id> ProgramManager::getAllSYCLKernelIDs() {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  std::vector<sycl::kernel_id> AllKernelIDs;
  AllKernelIDs.reserve(m_KernelName2KernelIDs.size());
  for (std::pair<std::string, kernel_id> KernelID : m_KernelName2KernelIDs) {
    AllKernelIDs.push_back(KernelID.second);
  }
  return AllKernelIDs;
}

kernel_id ProgramManager::getBuiltInKernelID(const std::string &KernelName) {
  std::lock_guard<std::mutex> BuiltInKernelIDsGuard(m_BuiltInKernelIDsMutex);

  auto KernelID = m_BuiltInKernelIDs.find(KernelName);
  if (KernelID == m_BuiltInKernelIDs.end()) {
    auto Impl = std::make_shared<kernel_id_impl>(KernelName);
    auto CachedID = createSyclObjFromImpl<kernel_id>(Impl);
    KernelID = m_BuiltInKernelIDs.insert({KernelName, CachedID}).first;
  }

  return KernelID->second;
}

void ProgramManager::addOrInitDeviceGlobalEntry(const void *DeviceGlobalPtr,
                                                const char *UniqueId) {
  std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);

  auto ExistingDeviceGlobal = m_DeviceGlobals.find(UniqueId);
  if (ExistingDeviceGlobal != m_DeviceGlobals.end()) {
    // Update the existing information and add the entry to the pointer map.
    ExistingDeviceGlobal->second->initialize(DeviceGlobalPtr);
    m_Ptr2DeviceGlobal.insert(
        {DeviceGlobalPtr, ExistingDeviceGlobal->second.get()});
    return;
  }

  auto EntryUPtr =
      std::make_unique<DeviceGlobalMapEntry>(UniqueId, DeviceGlobalPtr);
  auto NewEntry = m_DeviceGlobals.emplace(UniqueId, std::move(EntryUPtr));
  m_Ptr2DeviceGlobal.insert({DeviceGlobalPtr, NewEntry.first->second.get()});
}

std::set<RTDeviceBinaryImage *>
ProgramManager::getRawDeviceImages(const std::vector<kernel_id> &KernelIDs) {
  std::set<RTDeviceBinaryImage *> BinImages;
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
  for (const kernel_id &KID : KernelIDs) {
    auto Range = m_KernelIDs2BinImage.equal_range(KID);
    for (auto It = Range.first, End = Range.second; It != End; ++It)
      BinImages.insert(It->second);
  }
  return BinImages;
}

DeviceGlobalMapEntry *
ProgramManager::getDeviceGlobalEntry(const void *DeviceGlobalPtr) {
  std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);
  auto Entry = m_Ptr2DeviceGlobal.find(DeviceGlobalPtr);
  assert(Entry != m_Ptr2DeviceGlobal.end() && "Device global entry not found");
  return Entry->second;
}

std::vector<DeviceGlobalMapEntry *> ProgramManager::getDeviceGlobalEntries(
    const std::vector<std::string> &UniqueIds,
    bool ExcludeDeviceImageScopeDecorated) {
  std::vector<DeviceGlobalMapEntry *> FoundEntries;
  FoundEntries.reserve(UniqueIds.size());

  std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);
  for (const std::string &UniqueId : UniqueIds) {
    auto DeviceGlobalEntry = m_DeviceGlobals.find(UniqueId);
    assert(DeviceGlobalEntry != m_DeviceGlobals.end() &&
           "Device global not found in map.");
    if (!ExcludeDeviceImageScopeDecorated ||
        !DeviceGlobalEntry->second->MIsDeviceImageScopeDecorated)
      FoundEntries.push_back(DeviceGlobalEntry->second.get());
  }
  return FoundEntries;
}

void ProgramManager::addOrInitHostPipeEntry(const void *HostPipePtr,
                                            const char *UniqueId) {
  std::lock_guard<std::mutex> HostPipesGuard(m_HostPipesMutex);

  auto ExistingHostPipe = m_HostPipes.find(UniqueId);
  if (ExistingHostPipe != m_HostPipes.end()) {
    ExistingHostPipe->second->initialize(HostPipePtr);
    m_Ptr2HostPipe.insert({HostPipePtr, ExistingHostPipe->second.get()});
    return;
  }

  auto EntryUPtr = std::make_unique<HostPipeMapEntry>(UniqueId, HostPipePtr);
  auto NewEntry = m_HostPipes.emplace(UniqueId, std::move(EntryUPtr));
  m_Ptr2HostPipe.insert({HostPipePtr, NewEntry.first->second.get()});
}

HostPipeMapEntry *
ProgramManager::getHostPipeEntry(const std::string &UniqueId) {
  std::lock_guard<std::mutex> HostPipesGuard(m_HostPipesMutex);
  auto Entry = m_HostPipes.find(UniqueId);
  assert(Entry != m_HostPipes.end() && "Host pipe entry not found");
  return Entry->second.get();
}

HostPipeMapEntry *ProgramManager::getHostPipeEntry(const void *HostPipePtr) {
  std::lock_guard<std::mutex> HostPipesGuard(m_HostPipesMutex);
  auto Entry = m_Ptr2HostPipe.find(HostPipePtr);
  assert(Entry != m_Ptr2HostPipe.end() && "Host pipe entry not found");
  return Entry->second;
}

device_image_plain ProgramManager::getDeviceImageFromBinaryImage(
    RTDeviceBinaryImage *BinImage, const context &Ctx, const device &Dev) {
  const bundle_state ImgState = getBinImageState(BinImage);

  assert(compatibleWithDevice(BinImage, Dev));

  std::shared_ptr<std::vector<sycl::kernel_id>> KernelIDs;
  // Collect kernel names for the image.
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    KernelIDs = m_BinImg2KernelIDs[BinImage];
  }

  DeviceImageImplPtr Impl = std::make_shared<detail::device_image_impl>(
      BinImage, Ctx, std::vector<device>{Dev}, ImgState, KernelIDs,
      /*PIProgram=*/nullptr);

  return createSyclObjFromImpl<device_image_plain>(Impl);
}

std::vector<device_image_plain>
ProgramManager::getSYCLDeviceImagesWithCompatibleState(
    const context &Ctx, const std::vector<device> &Devs,
    bundle_state TargetState, const std::vector<kernel_id> &KernelIDs) {

  // Collect unique raw device images taking into account kernel ids passed
  // TODO: Can we avoid repacking?
  std::set<RTDeviceBinaryImage *> BinImages;
  if (!KernelIDs.empty()) {
    for (const auto &KID : KernelIDs) {
      bool isCompatibleWithAtLeastOneDev =
          std::any_of(Devs.begin(), Devs.end(), [&KID](const auto &Dev) {
            return sycl::is_compatible({KID}, Dev);
          });
      if (!isCompatibleWithAtLeastOneDev)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Kernel is incompatible with all devices in devs");
    }
    BinImages = getRawDeviceImages(KernelIDs);
  } else {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    for (auto &ImageUPtr : m_BinImg2KernelIDs) {
      BinImages.insert(ImageUPtr.first);
    }
  }

  // Ignore images with incompatible state. Image is considered compatible
  // with a target state if an image is already in the target state or can
  // be brought to target state by compiling/linking/building.
  //
  // Example: an image in "executable" state is not compatible with
  // "input" target state - there is no operation to convert the image it
  // to "input" state. An image in "input" state is compatible with
  // "executable" target state because it can be built to get into
  // "executable" state.
  for (auto It = BinImages.begin(); It != BinImages.end();) {
    if (getBinImageState(*It) > TargetState)
      It = BinImages.erase(It);
    else
      ++It;
  }

  std::vector<device_image_plain> SYCLDeviceImages;

  // If a non-input state is requested, we can filter out some compatible
  // images and return only those with the highest compatible state for each
  // device-kernel pair. This map tracks how many kernel-device pairs need each
  // image, so that any unneeded ones are skipped.
  // TODO this has no effect if the requested state is input, consider having
  // a separate branch for that case to avoid unnecessary tracking work.
  struct DeviceBinaryImageInfo {
    std::shared_ptr<std::vector<sycl::kernel_id>> KernelIDs;
    bundle_state State = bundle_state::input;
    int RequirementCounter = 0;
  };
  std::unordered_map<RTDeviceBinaryImage *, DeviceBinaryImageInfo> ImageInfoMap;

  for (const sycl::device &Dev : Devs) {
    // Track the highest image state for each requested kernel.
    using StateImagesPairT =
        std::pair<bundle_state, std::vector<RTDeviceBinaryImage *>>;
    using KernelImageMapT =
        std::map<kernel_id, StateImagesPairT, LessByNameComp>;
    KernelImageMapT KernelImageMap;
    if (!KernelIDs.empty())
      for (const kernel_id &KernelID : KernelIDs)
        KernelImageMap.insert({KernelID, {}});

    for (RTDeviceBinaryImage *BinImage : BinImages) {
      if (!compatibleWithDevice(BinImage, Dev) ||
          !doesDevSupportDeviceRequirements(Dev, *BinImage))
        continue;

      auto InsertRes = ImageInfoMap.insert({BinImage, {}});
      DeviceBinaryImageInfo &ImgInfo = InsertRes.first->second;
      if (InsertRes.second) {
        ImgInfo.State = getBinImageState(BinImage);
        // Collect kernel names for the image
        {
          std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
          ImgInfo.KernelIDs = m_BinImg2KernelIDs[BinImage];
        }
      }
      const bundle_state ImgState = ImgInfo.State;
      const std::shared_ptr<std::vector<sycl::kernel_id>> &ImageKernelIDs =
          ImgInfo.KernelIDs;
      int &ImgRequirementCounter = ImgInfo.RequirementCounter;

      // If the image does not contain any non-service kernels we can skip it.
      if (!ImageKernelIDs || ImageKernelIDs->empty())
        continue;

      // Update tracked information.
      for (kernel_id &KernelID : *ImageKernelIDs) {
        StateImagesPairT *StateImagesPair;
        // If only specific kernels are requested, ignore the rest.
        if (!KernelIDs.empty()) {
          auto It = KernelImageMap.find(KernelID);
          if (It == KernelImageMap.end())
            continue;
          StateImagesPair = &It->second;
        } else
          StateImagesPair = &KernelImageMap[KernelID];

        auto &[KernelImagesState, KernelImages] = *StateImagesPair;

        if (KernelImages.empty()) {
          KernelImagesState = ImgState;
          KernelImages.push_back(BinImage);
          ++ImgRequirementCounter;
        } else if (KernelImagesState < ImgState) {
          for (RTDeviceBinaryImage *Img : KernelImages) {
            auto It = ImageInfoMap.find(Img);
            assert(It != ImageInfoMap.end());
            assert(It->second.RequirementCounter > 0);
            --(It->second.RequirementCounter);
          }
          KernelImages.clear();
          KernelImages.push_back(BinImage);
          KernelImagesState = ImgState;
          ++ImgRequirementCounter;
        } else if (KernelImagesState == ImgState) {
          KernelImages.push_back(BinImage);
          ++ImgRequirementCounter;
        }
      }
    }
  }

  for (const auto &ImgInfoPair : ImageInfoMap) {
    if (ImgInfoPair.second.RequirementCounter == 0)
      continue;

    DeviceImageImplPtr Impl = std::make_shared<detail::device_image_impl>(
        ImgInfoPair.first, Ctx, Devs, ImgInfoPair.second.State,
        ImgInfoPair.second.KernelIDs, /*PIProgram=*/nullptr);

    SYCLDeviceImages.push_back(createSyclObjFromImpl<device_image_plain>(Impl));
  }

  return SYCLDeviceImages;
}

void ProgramManager::bringSYCLDeviceImagesToState(
    std::vector<device_image_plain> &DeviceImages, bundle_state TargetState) {

  for (device_image_plain &DevImage : DeviceImages) {
    const bundle_state DevImageState = getSyclObjImpl(DevImage)->get_state();

    // At this time, there is no circumstance where a device image should ever
    // be in the source state. That not good.
    assert(DevImageState != bundle_state::ext_oneapi_source);

    switch (TargetState) {
    case bundle_state::ext_oneapi_source:
      // This case added for switch statement completion. We should not be here.
      assert(DevImageState == bundle_state::ext_oneapi_source);
      break;
    case bundle_state::input:
      // Do nothing since there is no state which can be upgraded to the input.
      assert(DevImageState == bundle_state::input);
      break;
    case bundle_state::object:
      if (DevImageState == bundle_state::input) {
        DevImage = compile(DevImage, getSyclObjImpl(DevImage)->get_devices(),
                           /*PropList=*/{});
        break;
      }
      // Device image is expected to be object state then.
      assert(DevImageState == bundle_state::object);
      break;
    case bundle_state::executable: {
      switch (DevImageState) {
      case bundle_state::ext_oneapi_source:
        // This case added for switch statement completion.
        // We should not be here.
        assert(DevImageState != bundle_state::ext_oneapi_source);
        break;
      case bundle_state::input:
        DevImage = build(DevImage, getSyclObjImpl(DevImage)->get_devices(),
                         /*PropList=*/{});
        break;
      case bundle_state::object: {
        std::vector<device_image_plain> LinkedDevImages =
            link({DevImage}, getSyclObjImpl(DevImage)->get_devices(),
                 /*PropList=*/{});
        // Since only one device image is passed here one output device image is
        // expected
        assert(LinkedDevImages.size() == 1 && "Expected one linked image here");
        DevImage = LinkedDevImages[0];
        break;
      }
      case bundle_state::executable:
        DevImage = build(DevImage, getSyclObjImpl(DevImage)->get_devices(),
                         /*PropList=*/{});
        break;
      }
      break;
    }
    }
  }
}

std::vector<device_image_plain>
ProgramManager::getSYCLDeviceImages(const context &Ctx,
                                    const std::vector<device> &Devs,
                                    bundle_state TargetState) {
  // Collect device images with compatible state
  std::vector<device_image_plain> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState);
  // Bring device images with compatible state to desired state.
  bringSYCLDeviceImagesToState(DeviceImages, TargetState);
  return DeviceImages;
}

std::vector<device_image_plain> ProgramManager::getSYCLDeviceImages(
    const context &Ctx, const std::vector<device> &Devs,
    const DevImgSelectorImpl &Selector, bundle_state TargetState) {
  // Collect device images with compatible state
  std::vector<device_image_plain> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState);

  // Filter out images that are rejected by Selector
  auto It = std::remove_if(DeviceImages.begin(), DeviceImages.end(),
                           [&Selector](const device_image_plain &Image) {
                             return !Selector(getSyclObjImpl(Image));
                           });
  DeviceImages.erase(It, DeviceImages.end());

  // The spec says that the function should not call online compiler or linker
  // to translate device images into target state
  return DeviceImages;
}

std::vector<device_image_plain> ProgramManager::getSYCLDeviceImages(
    const context &Ctx, const std::vector<device> &Devs,
    const std::vector<kernel_id> &KernelIDs, bundle_state TargetState) {
  // Fast path for when no kernel IDs are requested
  if (KernelIDs.empty())
    return {};

  {
    std::lock_guard<std::mutex> BuiltInKernelIDsGuard(m_BuiltInKernelIDsMutex);

    for (auto &It : m_BuiltInKernelIDs) {
      if (std::find(KernelIDs.begin(), KernelIDs.end(), It.second) !=
          KernelIDs.end())
        throw sycl::exception(make_error_code(errc::kernel_argument),
                              "Attempting to use a built-in kernel. They are "
                              "not fully supported");
    }
  }

  // Collect device images with compatible state
  std::vector<device_image_plain> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState, KernelIDs);

  // Bring device images with compatible state to desired state.
  bringSYCLDeviceImagesToState(DeviceImages, TargetState);
  return DeviceImages;
}

static void
setSpecializationConstants(const std::shared_ptr<device_image_impl> &InputImpl,
                           sycl::detail::pi::PiProgram Prog,
                           const PluginPtr &Plugin) {
  // Set ITT annotation specialization constant if needed.
  enableITTAnnotationsIfNeeded(Prog, Plugin);

  std::lock_guard<std::mutex> Lock{InputImpl->get_spec_const_data_lock()};
  const std::map<std::string, std::vector<device_image_impl::SpecConstDescT>>
      &SpecConstData = InputImpl->get_spec_const_data_ref();
  const SerializedObj &SpecConsts = InputImpl->get_spec_const_blob_ref();

  // Set all specialization IDs from descriptors in the input device image.
  for (const auto &[SpecConstNames, SpecConstDescs] : SpecConstData) {
    std::ignore = SpecConstNames;
    for (const device_image_impl::SpecConstDescT &SpecIDDesc : SpecConstDescs) {
      if (SpecIDDesc.IsSet) {
        Plugin->call<PiApiKind::piextProgramSetSpecializationConstant>(
            Prog, SpecIDDesc.ID, SpecIDDesc.Size,
            SpecConsts.data() + SpecIDDesc.BlobOffset);
      }
    }
  }
}

device_image_plain
ProgramManager::compile(const device_image_plain &DeviceImage,
                        const std::vector<device> &Devs,
                        const property_list &) {

  // TODO: Extract compile options from property list once the Spec clarifies
  // how they can be passed.

  // TODO: Probably we could have cached compiled device images.
  const std::shared_ptr<device_image_impl> &InputImpl =
      getSyclObjImpl(DeviceImage);

  const PluginPtr &Plugin =
      getSyclObjImpl(InputImpl->get_context())->getPlugin();

  // TODO: Add support for creating non-SPIRV programs from multiple devices.
  if (InputImpl->get_bin_image_ref()->getFormat() !=
          PI_DEVICE_BINARY_TYPE_SPIRV &&
      Devs.size() > 1)
    sycl::runtime_error(
        "Creating a program from AOT binary for multiple device is not "
        "supported",
        PI_ERROR_INVALID_OPERATION);

  // Device is not used when creating program from SPIRV, so passing only one
  // device is OK.
  sycl::detail::pi::PiProgram Prog = createPIProgram(
      *InputImpl->get_bin_image_ref(), InputImpl->get_context(), Devs[0]);

  if (InputImpl->get_bin_image_ref()->supportsSpecConstants())
    setSpecializationConstants(InputImpl, Prog, Plugin);

  DeviceImageImplPtr ObjectImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), InputImpl->get_context(), Devs,
      bundle_state::object, InputImpl->get_kernel_ids_ptr(), Prog,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  std::vector<pi_device> PIDevices;
  PIDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    PIDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

  // TODO: Handle zero sized Device list.
  std::string CompileOptions;
  applyCompileOptionsFromEnvironment(CompileOptions);
  appendCompileOptionsFromImage(
      CompileOptions, *(InputImpl->get_bin_image_ref()), Devs, Plugin);
  // Should always come last!
  appendCompileEnvironmentVariablesThatAppend(CompileOptions);
  sycl::detail::pi::PiResult Error =
      Plugin->call_nocheck<PiApiKind::piProgramCompile>(
          ObjectImpl->get_program_ref(), /*num devices=*/Devs.size(),
          PIDevices.data(), CompileOptions.c_str(),
          /*num_input_headers=*/0, /*input_headers=*/nullptr,
          /*header_include_names=*/nullptr,
          /*pfn_notify=*/nullptr, /*user_data*/ nullptr);
  if (Error != PI_SUCCESS)
    throw sycl::exception(
        make_error_code(errc::build),
        getProgramBuildLog(ObjectImpl->get_program_ref(),
                           getSyclObjImpl(ObjectImpl->get_context())));

  return createSyclObjFromImpl<device_image_plain>(ObjectImpl);
}

std::vector<device_image_plain>
ProgramManager::link(const device_image_plain &DeviceImage,
                     const std::vector<device> &Devs,
                     const property_list &PropList) {
  (void)PropList;

  std::vector<pi_program> PIPrograms;
  PIPrograms.push_back(getSyclObjImpl(DeviceImage)->get_program_ref());

  std::vector<pi_device> PIDevices;
  PIDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    PIDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

  std::string LinkOptionsStr;
  applyLinkOptionsFromEnvironment(LinkOptionsStr);
  if (LinkOptionsStr.empty()) {
    const std::shared_ptr<device_image_impl> &InputImpl =
        getSyclObjImpl(DeviceImage);
    appendLinkOptionsFromImage(LinkOptionsStr,
                               *(InputImpl->get_bin_image_ref()));
  }
  // Should always come last!
  appendLinkEnvironmentVariablesThatAppend(LinkOptionsStr);
  const context &Context = getSyclObjImpl(DeviceImage)->get_context();
  const ContextImplPtr ContextImpl = getSyclObjImpl(Context);
  const PluginPtr &Plugin = ContextImpl->getPlugin();

  sycl::detail::pi::PiProgram LinkedProg = nullptr;
  auto doLink = [&] {
    return Plugin->call_nocheck<PiApiKind::piProgramLink>(
        ContextImpl->getHandleRef(), PIDevices.size(), PIDevices.data(),
        /*options=*/LinkOptionsStr.c_str(), PIPrograms.size(),
        PIPrograms.data(),
        /*pfn_notify=*/nullptr,
        /*user_data=*/nullptr, &LinkedProg);
  };
  sycl::detail::pi::PiResult Error = doLink();
  if (Error == PI_ERROR_OUT_OF_RESOURCES) {
    ContextImpl->getKernelProgramCache().reset();
    Error = doLink();
  }

  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      const std::string ErrorMsg = getProgramBuildLog(LinkedProg, ContextImpl);
      throw sycl::exception(make_error_code(errc::build), ErrorMsg);
    }
    Plugin->reportPiError(Error, "link()");
  }

  std::shared_ptr<std::vector<kernel_id>> KernelIDs{new std::vector<kernel_id>};
  std::vector<unsigned char> NewSpecConstBlob;
  device_image_impl::SpecConstMapT NewSpecConstMap;

  std::shared_ptr<device_image_impl> DeviceImageImpl =
      getSyclObjImpl(DeviceImage);

  // Duplicates are not expected here, otherwise piProgramLink should fail
  KernelIDs->insert(KernelIDs->end(),
                    DeviceImageImpl->get_kernel_ids_ptr()->begin(),
                    DeviceImageImpl->get_kernel_ids_ptr()->end());

  // To be able to answer queries about specialziation constants, the new
  // device image should have the specialization constants from all the linked
  // images.
  {
    const std::lock_guard<std::mutex> SpecConstLock(
        DeviceImageImpl->get_spec_const_data_lock());

    // Copy all map entries to the new map. Since the blob will be copied to
    // the end of the new blob we need to move the blob offset of each entry.
    for (const auto &SpecConstIt : DeviceImageImpl->get_spec_const_data_ref()) {
      std::vector<device_image_impl::SpecConstDescT> &NewDescEntries =
          NewSpecConstMap[SpecConstIt.first];
      assert(NewDescEntries.empty() &&
             "Specialization constant already exists in the map.");
      NewDescEntries.reserve(SpecConstIt.second.size());
      for (const device_image_impl::SpecConstDescT &SpecConstDesc :
           SpecConstIt.second) {
        device_image_impl::SpecConstDescT NewSpecConstDesc = SpecConstDesc;
        NewSpecConstDesc.BlobOffset += NewSpecConstBlob.size();
        NewDescEntries.push_back(std::move(NewSpecConstDesc));
      }
    }

    // Copy the blob from the device image into the new blob. This moves the
    // offsets of the following blobs.
    NewSpecConstBlob.insert(NewSpecConstBlob.end(),
                            DeviceImageImpl->get_spec_const_blob_ref().begin(),
                            DeviceImageImpl->get_spec_const_blob_ref().end());
  }

  // device_image_impl expects kernel ids to be sorted for fast search
  std::sort(KernelIDs->begin(), KernelIDs->end(), LessByHash<kernel_id>{});

  auto BinImg = getSyclObjImpl(DeviceImage)->get_bin_image_ref();
  DeviceImageImplPtr ExecutableImpl =
      std::make_shared<detail::device_image_impl>(
          BinImg, Context, Devs, bundle_state::executable, std::move(KernelIDs),
          LinkedProg, std::move(NewSpecConstMap), std::move(NewSpecConstBlob));

  // TODO: Make multiple sets of device images organized by devices they are
  // compiled for.
  return {createSyclObjFromImpl<device_image_plain>(ExecutableImpl)};
}

// The function duplicates most of the code from existing getBuiltPIProgram.
// The differences are:
// Different API - uses different objects to extract required info
// Supports caching of a program built for multiple devices
device_image_plain ProgramManager::build(const device_image_plain &DeviceImage,
                                         const std::vector<device> &Devs,
                                         const property_list &PropList) {
  (void)PropList;

  const std::shared_ptr<device_image_impl> &InputImpl =
      getSyclObjImpl(DeviceImage);

  const context Context = InputImpl->get_context();

  const ContextImplPtr ContextImpl = getSyclObjImpl(Context);

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  std::string CompileOpts;
  std::string LinkOpts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);

  const RTDeviceBinaryImage *ImgPtr = InputImpl->get_bin_image_ref();
  const RTDeviceBinaryImage &Img = *ImgPtr;

  SerializedObj SpecConsts = InputImpl->get_spec_const_blob_ref();

  // TODO: Unify this code with getBuiltPIProgram
  auto BuildF = [this, &Context, &Img, &Devs, &CompileOpts, &LinkOpts,
                 &InputImpl, SpecConsts] {
    ContextImplPtr ContextImpl = getSyclObjImpl(Context);
    const PluginPtr &Plugin = ContextImpl->getPlugin();
    applyOptionsFromImage(CompileOpts, LinkOpts, Img, Devs, Plugin);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOpts);
    appendLinkEnvironmentVariablesThatAppend(LinkOpts);
    // TODO: Add support for creating non-SPIRV programs from multiple devices.
    if (InputImpl->get_bin_image_ref()->getFormat() !=
            PI_DEVICE_BINARY_TYPE_SPIRV &&
        Devs.size() > 1)
      sycl::runtime_error(
          "Creating a program from AOT binary for multiple device is not "
          "supported",
          PI_ERROR_INVALID_OPERATION);

    // Device is not used when creating program from SPIRV, so passing only one
    // device is OK.
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreatePIProgram(
        Img, Context, Devs[0], CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache &&
        InputImpl->get_bin_image_ref()->supportsSpecConstants())
      setSpecializationConstants(InputImpl, NativePrg, Plugin);

    ProgramPtr ProgramManaged(
        NativePrg, Plugin->getPiPlugin().PiFunctionTable.piProgramRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    if (Img.getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV &&
        !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get())
      DeviceLibReqMask = getDeviceLibReqMask(Img);

    ProgramPtr BuiltProgram =
        build(std::move(ProgramManaged), ContextImpl, CompileOpts, LinkOpts,
              getRawSyclObjImpl(Devs[0])->getHandleRef(), DeviceLibReqMask);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms[BuiltProgram.get()] = &Img;
    }

    ContextImpl->addDeviceGlobalInitializer(BuiltProgram.get(), Devs, &Img);

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache)
      PersistentDeviceCodeCache::putItemToDisc(
          Devs[0], Img, SpecConsts, CompileOpts + LinkOpts, BuiltProgram.get());

    return BuiltProgram.release();
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    auto ResProgram = BuildF();
    DeviceImageImplPtr ExecImpl = std::make_shared<detail::device_image_impl>(
        InputImpl->get_bin_image_ref(), Context, Devs, bundle_state::executable,
        InputImpl->get_kernel_ids_ptr(), ResProgram,
        InputImpl->get_spec_const_data_ref(),
        InputImpl->get_spec_const_blob_ref());

    return createSyclObjFromImpl<device_image_plain>(ExecImpl);
  }

  uint32_t ImgId = Img.getImageID();
  const sycl::detail::pi::PiDevice PiDevice =
      getRawSyclObjImpl(Devs[0])->getHandleRef();
  auto CacheKey =
      std::make_pair(std::make_pair(std::move(SpecConsts), ImgId), PiDevice);

  // CacheKey is captured by reference so when we overwrite it later we can
  // reuse this function.
  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  // TODO: Throw SYCL2020 style exception
  auto BuildResult =
      Cache.getOrBuild<compile_program_error>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  sycl::detail::pi::PiProgram ResProgram = BuildResult->Val;

  // Cache supports key with once device only, but here we have multiple
  // devices a program is built for, so add the program to the cache for all
  // other devices.
  const PluginPtr &Plugin = ContextImpl->getPlugin();
  auto CacheOtherDevices = [ResProgram, &Plugin]() {
    Plugin->call<PiApiKind::piProgramRetain>(ResProgram);
    return ResProgram;
  };

  // The program for device "0" is already added to the cache during the first
  // call to getOrBuild, so starting with "1"
  for (size_t Idx = 1; Idx < Devs.size(); ++Idx) {
    const sycl::detail::pi::PiDevice PiDeviceAdd =
        getRawSyclObjImpl(Devs[Idx])->getHandleRef();

    // Change device in the cache key to reduce copying of spec const data.
    CacheKey.second = PiDeviceAdd;
    Cache.getOrBuild<compile_program_error>(GetCachedBuildF, CacheOtherDevices);
    // getOrBuild is not supposed to return nullptr
    assert(BuildResult != nullptr && "Invalid build result");
  }

  // devive_image_impl shares ownership of PIProgram with, at least, program
  // cache. The ref counter will be descremented in the destructor of
  // device_image_impl
  Plugin->call<PiApiKind::piProgramRetain>(ResProgram);

  DeviceImageImplPtr ExecImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), Context, Devs, bundle_state::executable,
      InputImpl->get_kernel_ids_ptr(), ResProgram,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  return createSyclObjFromImpl<device_image_plain>(ExecImpl);
}

// When caching is enabled, the returned PiKernel will already have
// its ref count incremented.
std::tuple<sycl::detail::pi::PiKernel, std::mutex *, const KernelArgMask *>
ProgramManager::getOrCreateKernel(const context &Context,
                                  const std::string &KernelName,
                                  const property_list &PropList,
                                  sycl::detail::pi::PiProgram Program) {

  (void)PropList;

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto BuildF = [this, &Program, &KernelName, &Ctx] {
    sycl::detail::pi::PiKernel Kernel = nullptr;

    const PluginPtr &Plugin = Ctx->getPlugin();
    Plugin->call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                            &Kernel);

    Plugin->call<PiApiKind::piKernelSetExecInfo>(Kernel, PI_USM_INDIRECT_ACCESS,
                                                 sizeof(pi_bool), &PI_TRUE);

    // Ignore possible m_UseSpvFile for now.
    // TODO consider making m_UseSpvFile interact with kernel bundles as well.
    const KernelArgMask *KernelArgMask =
        getEliminatedKernelArgMask(Program, KernelName);

    return std::make_pair(Kernel, KernelArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    // The built kernel cannot be shared between multiple
    // threads when caching is disabled, so we can return
    // nullptr for the mutex.
    auto [Kernel, ArgMask] = BuildF();
    return make_tuple(Kernel, nullptr, ArgMask);
  }

  auto BuildResult =
      Cache.getOrBuild<invalid_object_error>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  // If caching is enabled, one copy of the kernel handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // kernel.
  Ctx->getPlugin()->call<PiApiKind::piKernelRetain>(BuildResult->Val.first);
  return std::make_tuple(BuildResult->Val.first,
                         &(BuildResult->MBuildResultMutex),
                         BuildResult->Val.second);
}

bool doesDevSupportDeviceRequirements(const device &Dev,
                                      const RTDeviceBinaryImage &Img) {
  return !checkDevSupportDeviceRequirements(Dev, Img).has_value();
}

static std::string getAspectNameStr(sycl::aspect AspectNum) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case aspect::ASPECT:                                                         \
    return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
// We don't need "case aspect::usm_allocator" here because it will duplicate
// "case aspect::usm_system_allocations", therefore leave this macro empty
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
  switch (AspectNum) {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
  }
  throw sycl::exception(errc::kernel_not_supported,
                        "Unknown aspect " +
                            std::to_string(static_cast<unsigned>(AspectNum)));
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
}

// Check if the multiplication over unsigned integers overflows
template <typename T>
static std::enable_if_t<std::is_unsigned_v<T>, std::optional<T>>
multiply_with_overflow_check(T x, T y) {
  if (y == 0)
    return 0;
  if (x > std::numeric_limits<T>::max() / y)
    return {};
  else
    return x * y;
}

namespace matrix_ext = ext::oneapi::experimental::matrix;

// Matrix type string to matrix_type enum value conversion
// Note: matrix type strings are defined in template specialization for
// convertTypeToMatrixTypeString above
std::optional<matrix_ext::matrix_type>
convertMatrixTypeStringMatrixTypeEnumValue(
    const std::string &MatrixTypeString) {
  assert(!MatrixTypeString.empty() &&
         "MatrixTypeString type string can't be empty. Check if required "
         "template specialization for convertTypeToMatrixTypeString exists.");
  std::string_view MatrixTypeStringView = MatrixTypeString;
  std::string Prefix("matrix_type::");
  assert((MatrixTypeStringView.substr(0, Prefix.size()) == Prefix) &&
         "MatrixTypeString has incorrect prefix, should be \"matrix_type::\".");
  MatrixTypeStringView.remove_prefix(Prefix.size());
  if ("bf16" == MatrixTypeStringView)
    return matrix_ext::matrix_type::bf16;
  else if ("fp16" == MatrixTypeStringView)
    return matrix_ext::matrix_type::fp16;
  else if ("tf32" == MatrixTypeStringView)
    return matrix_ext::matrix_type::tf32;
  else if ("fp32" == MatrixTypeStringView)
    return matrix_ext::matrix_type::fp32;
  else if ("fp64" == MatrixTypeStringView)
    return matrix_ext::matrix_type::fp64;
  else if ("sint8" == MatrixTypeStringView)
    return matrix_ext::matrix_type::sint8;
  else if ("sint16" == MatrixTypeStringView)
    return matrix_ext::matrix_type::sint16;
  else if ("sint32" == MatrixTypeStringView)
    return matrix_ext::matrix_type::sint32;
  else if ("sint64" == MatrixTypeStringView)
    return matrix_ext::matrix_type::sint64;
  else if ("uint8" == MatrixTypeStringView)
    return matrix_ext::matrix_type::uint8;
  else if ("uint16" == MatrixTypeStringView)
    return matrix_ext::matrix_type::uint16;
  else if ("uint32" == MatrixTypeStringView)
    return matrix_ext::matrix_type::uint32;
  else if ("uint64" == MatrixTypeStringView)
    return matrix_ext::matrix_type::uint64;
  return std::nullopt;
}

bool isMatrixSupportedByHW(const std::string &MatrixTypeStrUser,
                           size_t RowsUser, size_t ColsUser,
                           matrix_ext::matrix_type MatrixTypeRuntime,
                           size_t MaxRowsRuntime, size_t MaxColsRuntime,
                           size_t RowsRuntime, size_t ColsRuntime) {
  std::optional<matrix_ext::matrix_type> MatrixTypeUserOpt =
      convertMatrixTypeStringMatrixTypeEnumValue(MatrixTypeStrUser);
  if (!MatrixTypeUserOpt)
    return false;
  bool IsMatrixTypeSupported = (MatrixTypeUserOpt.value() == MatrixTypeRuntime);
  bool IsRowsSupported = ((RowsRuntime != 0) ? (RowsUser == RowsRuntime)
                                             : (RowsUser <= MaxRowsRuntime));
  bool IsColsSupported = ((ColsRuntime != 0) ? (ColsUser == ColsRuntime)
                                             : (ColsUser <= MaxColsRuntime));
  return IsMatrixTypeSupported && IsRowsSupported && IsColsSupported;
}

std::optional<sycl::exception> checkDevSupportJointMatrix(
    const std::string &JointMatrixProStr,
    const std::vector<ext::oneapi::experimental::matrix::combination>
        &SupportedMatrixCombinations) {
  std::istringstream JointMatrixStrStream(JointMatrixProStr);
  std::string SingleJointMatrix;

  // start to parse the value which is generated by
  // SYCLPropagateJointMatrixUsage pass
  while (std::getline(JointMatrixStrStream, SingleJointMatrix, ';')) {
    std::istringstream SingleJointMatrixStrStream(SingleJointMatrix);
    std::vector<std::string> JointMatrixVec;
    std::string Item;

    while (std::getline(SingleJointMatrixStrStream, Item, ',')) {
      JointMatrixVec.push_back(Item);
    }

    assert(JointMatrixVec.size() == 4 &&
           "Property set is corrupted, it must have 4 elements.");

    const std::string &MatrixTypeUser = JointMatrixVec[0];
    const std::string &UseStrUser = JointMatrixVec[1];
    size_t RowsUser, ColsUser = 0;
    try {
      RowsUser = std::stoi(JointMatrixVec[2]);
      ColsUser = std::stoi(JointMatrixVec[3]);
    } catch (std::logic_error &) {
      // ignore exceptions, one way or another a user will see sycl::exception
      // with the message about incorrect rows or cols, because they are
      // initialized with 0 above
    }

    bool IsMatrixCompatible = false;

    for (const auto &Combination : SupportedMatrixCombinations) {
      std::optional<ext::oneapi::experimental::matrix::use> Use =
          detail::convertMatrixUseStringToEnum(UseStrUser.c_str());
      assert(Use && "Property set has empty matrix::use value.");
      switch (Use.value()) {
      case matrix_ext::use::a:
        IsMatrixCompatible |= isMatrixSupportedByHW(
            MatrixTypeUser, RowsUser, ColsUser, Combination.atype,
            Combination.max_msize, Combination.max_ksize, Combination.msize,
            Combination.ksize);
        break;
      case matrix_ext::use::b:
        IsMatrixCompatible |= isMatrixSupportedByHW(
            MatrixTypeUser, RowsUser, ColsUser, Combination.btype,
            Combination.max_ksize, Combination.max_nsize, Combination.ksize,
            Combination.nsize);
        break;
      case matrix_ext::use::accumulator: {
        IsMatrixCompatible |= isMatrixSupportedByHW(
            MatrixTypeUser, RowsUser, ColsUser, Combination.ctype,
            Combination.max_msize, Combination.max_nsize, Combination.msize,
            Combination.nsize);
        IsMatrixCompatible |= isMatrixSupportedByHW(
            MatrixTypeUser, RowsUser, ColsUser, Combination.dtype,
            Combination.max_msize, Combination.max_nsize, Combination.msize,
            Combination.nsize);
        break;
      }
      }

      // early exit if we have a match
      if (IsMatrixCompatible)
        break;
    }

    if (!IsMatrixCompatible)
      return sycl::exception(make_error_code(errc::kernel_not_supported),
                             "joint_matrix with parameters " + MatrixTypeUser +
                                 ", " + UseStrUser +
                                 ", Rows=" + std::to_string(RowsUser) +
                                 ", Cols=" + std::to_string(ColsUser) +
                                 " is not supported on this device");
  }
  return std::nullopt;
}

std::optional<sycl::exception> checkDevSupportJointMatrixMad(
    const std::string &JointMatrixProStr,
    const std::vector<ext::oneapi::experimental::matrix::combination>
        &SupportedMatrixCombinations) {
  std::istringstream JointMatrixMadStrStream(JointMatrixProStr);
  std::string SingleJointMatrixMad;

  // start to parse the value which is generated by
  // SYCLPropagateJointMatrixUsage pass
  while (std::getline(JointMatrixMadStrStream, SingleJointMatrixMad, ';')) {
    std::istringstream SingleJointMatrixMadStrStream(SingleJointMatrixMad);
    std::vector<std::string> JointMatrixMadVec;
    std::string Item;

    while (std::getline(SingleJointMatrixMadStrStream, Item, ',')) {
      JointMatrixMadVec.push_back(Item);
    }

    assert(JointMatrixMadVec.size() == 7 &&
           "Property set is corrupted, it must have 7 elements.");

    const std::string &MatrixTypeAStrUser = JointMatrixMadVec[0];
    const std::string &MatrixTypeBStrUser = JointMatrixMadVec[1];
    const std::string &MatrixTypeCStrUser = JointMatrixMadVec[2];
    const std::string &MatrixTypeDStrUser = JointMatrixMadVec[3];
    size_t MSizeUser, KSizeUser, NSizeUser = 0;
    try {
      MSizeUser = std::stoi(JointMatrixMadVec[4]);
      KSizeUser = std::stoi(JointMatrixMadVec[5]);
      NSizeUser = std::stoi(JointMatrixMadVec[6]);
    } catch (std::logic_error &) {
      // ignore exceptions, one way or another a user will see sycl::exception
      // with the message about incorrect size(s), because they are
      // initialized with 0 above
    }

    std::optional<matrix_ext::matrix_type> MatrixTypeAUserOpt =
        convertMatrixTypeStringMatrixTypeEnumValue(MatrixTypeAStrUser);
    std::optional<matrix_ext::matrix_type> MatrixTypeBUserOpt =
        convertMatrixTypeStringMatrixTypeEnumValue(MatrixTypeBStrUser);
    std::optional<matrix_ext::matrix_type> MatrixTypeCUserOpt =
        convertMatrixTypeStringMatrixTypeEnumValue(MatrixTypeCStrUser);
    std::optional<matrix_ext::matrix_type> MatrixTypeDUserOpt =
        convertMatrixTypeStringMatrixTypeEnumValue(MatrixTypeDStrUser);

    bool IsMatrixMadCompatible = false;

    for (const auto &Combination : SupportedMatrixCombinations) {
      if (!MatrixTypeAUserOpt || !MatrixTypeBUserOpt || !MatrixTypeCUserOpt ||
          !MatrixTypeDUserOpt)
        continue;

      bool IsMatrixTypeACompatible =
          (MatrixTypeAUserOpt.value() == Combination.atype);
      bool IsMatrixTypeBCompatible =
          (MatrixTypeBUserOpt.value() == Combination.btype);
      bool IsMatrixTypeCCompatible =
          (MatrixTypeCUserOpt.value() == Combination.ctype);
      bool IsMatrixTypeDCompatible =
          (MatrixTypeDUserOpt.value() == Combination.dtype);
      bool IsMSizeCompatible =
          ((Combination.msize != 0) ? (MSizeUser == Combination.msize)
                                    : (MSizeUser <= Combination.max_msize));
      bool IsKSizeCompatible =
          ((Combination.ksize != 0) ? (KSizeUser == Combination.ksize)
                                    : (KSizeUser <= Combination.max_ksize));
      bool IsNSizeCompatible =
          ((Combination.nsize != 0) ? (NSizeUser == Combination.nsize)
                                    : (NSizeUser <= Combination.max_nsize));

      IsMatrixMadCompatible =
          IsMatrixTypeACompatible && IsMatrixTypeBCompatible &&
          IsMatrixTypeCCompatible && IsMatrixTypeDCompatible &&
          IsMSizeCompatible && IsKSizeCompatible && IsNSizeCompatible;

      // early exit if we have a match
      if (IsMatrixMadCompatible)
        break;
    }

    if (!IsMatrixMadCompatible)
      return sycl::exception(
          make_error_code(errc::kernel_not_supported),
          "joint_matrix_mad function with parameters atype=" +
              MatrixTypeAStrUser + ", btype=" + MatrixTypeBStrUser +
              ", ctype=" + MatrixTypeCStrUser + ", dtype=" +
              MatrixTypeDStrUser + ", M=" + std::to_string(MSizeUser) + ", K=" +
              std::to_string(KSizeUser) + ", N=" + std::to_string(NSizeUser) +
              " is not supported on this "
              "device");
  }
  return std::nullopt;
}

std::optional<sycl::exception>
checkDevSupportDeviceRequirements(const device &Dev,
                                  const RTDeviceBinaryImage &Img,
                                  const NDRDescT &NDRDesc) {
  auto getPropIt = [&Img](const std::string &PropName) {
    const RTDeviceBinaryImage::PropertyRange &PropRange =
        Img.getDeviceRequirements();
    RTDeviceBinaryImage::PropertyRange::ConstIterator PropIt = std::find_if(
        PropRange.begin(), PropRange.end(),
        [&PropName](RTDeviceBinaryImage::PropertyRange::ConstIterator &&Prop) {
          return (*Prop)->Name == PropName;
        });
    return (PropIt == PropRange.end())
               ? std::nullopt
               : std::optional<
                     RTDeviceBinaryImage::PropertyRange::ConstIterator>{PropIt};
  };

  auto AspectsPropIt = getPropIt("aspects");
  auto JointMatrixPropIt = getPropIt("joint_matrix");
  auto JointMatrixMadPropIt = getPropIt("joint_matrix_mad");
  auto ReqdWGSizeUint32TPropIt = getPropIt("reqd_work_group_size");
  auto ReqdWGSizeUint64TPropIt = getPropIt("reqd_work_group_size_uint64_t");
  auto ReqdSubGroupSizePropIt = getPropIt("reqd_sub_group_size");

  // Checking if device supports defined aspects
  if (AspectsPropIt) {
    ByteArray Aspects =
        DeviceBinaryProperty(*(AspectsPropIt.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    Aspects.dropBytes(8);
    while (!Aspects.empty()) {
      aspect Aspect = Aspects.consume<aspect>();
      if (!Dev.has(Aspect))
        return sycl::exception(errc::kernel_not_supported,
                               "Required aspect " + getAspectNameStr(Aspect) +
                                   " is not supported on the device");
    }
  }

  // TODO: remove checks for CUDA and HIP from if-statement below when runtime
  // query for them in matrix_combinations is implemented
  if (JointMatrixPropIt &&
      (Dev.get_backend() != sycl::backend::ext_oneapi_cuda) &&
      (Dev.get_backend() != sycl::backend::ext_oneapi_hip)) {
    std::vector<ext::oneapi::experimental::matrix::combination> Combinations =
        Dev.get_info<
            ext::oneapi::experimental::info::device::matrix_combinations>();

    if (Combinations.empty())
      return sycl::exception(make_error_code(errc::kernel_not_supported),
                             "no matrix hardware on the target device, "
                             "joint_matrix is not supported");

    ByteArray JointMatrixByteArray =
        DeviceBinaryProperty(*(JointMatrixPropIt.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    JointMatrixByteArray.dropBytes(8);
    std::string JointMatrixByteArrayToStr;
    while (!JointMatrixByteArray.empty()) {
      JointMatrixByteArrayToStr += JointMatrixByteArray.consume<char>();
    }
    std::optional<sycl::exception> Result =
        checkDevSupportJointMatrix(JointMatrixByteArrayToStr, Combinations);
    if (Result)
      return Result.value();
  }

  // TODO: remove checks for CUDA and HIP from if-statement below when runtime
  // query for them in matrix_combinations is implemented
  if (JointMatrixMadPropIt &&
      (Dev.get_backend() != sycl::backend::ext_oneapi_cuda) &&
      (Dev.get_backend() != sycl::backend::ext_oneapi_hip)) {
    std::vector<ext::oneapi::experimental::matrix::combination> Combinations =
        Dev.get_info<
            ext::oneapi::experimental::info::device::matrix_combinations>();

    if (Combinations.empty())
      return sycl::exception(make_error_code(errc::kernel_not_supported),
                             "no matrix hardware on the target device, "
                             "joint_matrix_mad is not supported");

    ByteArray JointMatrixMadByteArray =
        DeviceBinaryProperty(*(JointMatrixMadPropIt.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    JointMatrixMadByteArray.dropBytes(8);
    std::string JointMatrixMadByteArrayToStr;
    while (!JointMatrixMadByteArray.empty()) {
      JointMatrixMadByteArrayToStr += JointMatrixMadByteArray.consume<char>();
    }
    std::optional<sycl::exception> Result = checkDevSupportJointMatrixMad(
        JointMatrixMadByteArrayToStr, Combinations);
    if (Result)
      return Result.value();
  }

  // Checking if device supports defined required work group size
  if (ReqdWGSizeUint32TPropIt || ReqdWGSizeUint64TPropIt) {
    /// TODO: Before intel/llvm#10620, the reqd_work_group_size attribute
    // stores its values as uint32_t, but this needed to be expanded to
    // uint64_t.  However, this change did not happen in ABI-breaking
    // window, so we attach the required work-group size as the
    // reqd_work_group_size_uint64_t attribute. At the next ABI-breaking
    // window, we can remove the logic for the 32 bit property.
    bool usingUint64_t = ReqdWGSizeUint64TPropIt.has_value();
    auto it = usingUint64_t ? ReqdWGSizeUint64TPropIt : ReqdWGSizeUint32TPropIt;

    ByteArray ReqdWGSize = DeviceBinaryProperty(*(it.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    ReqdWGSize.dropBytes(8);
    uint64_t ReqdWGSizeAllDimsTotal = 1;
    std::vector<uint64_t> ReqdWGSizeVec;
    int Dims = 0;
    while (!ReqdWGSize.empty()) {
      uint64_t SingleDimSize = usingUint64_t ? ReqdWGSize.consume<uint64_t>()
                                             : ReqdWGSize.consume<uint32_t>();
      if (auto res = multiply_with_overflow_check(ReqdWGSizeAllDimsTotal,
                                                  SingleDimSize))
        ReqdWGSizeAllDimsTotal = *res;
      else
        return sycl::exception(
            sycl::errc::kernel_not_supported,
            "Required work-group size is not supported"
            " (total number of work-items requested can't fit into size_t)");
      ReqdWGSizeVec.push_back(SingleDimSize);
      Dims++;
    }

    if (NDRDesc.Dims != 0 && NDRDesc.Dims != static_cast<size_t>(Dims))
      return sycl::exception(
          sycl::errc::nd_range,
          "The local size dimension of submitted nd_range doesn't match the "
          "required work-group size dimension");

    // The SingleDimSize was computed in an uint64_t; size_t does not
    // necessarily have to be the same uint64_t (but should fit in an
    // uint64_t).
    if (ReqdWGSizeAllDimsTotal >
        Dev.get_info<info::device::max_work_group_size>())
      return sycl::exception(sycl::errc::kernel_not_supported,
                             "Required work-group size " +
                                 std::to_string(ReqdWGSizeAllDimsTotal) +
                                 " is not supported on the device");
    // Creating std::variant to call max_work_item_sizes one time to avoid
    // performance drop
    std::variant<id<1>, id<2>, id<3>> MaxWorkItemSizesVariant;
    if (Dims == 1)
      MaxWorkItemSizesVariant =
          Dev.get_info<info::device::max_work_item_sizes<1>>();
    else if (Dims == 2)
      MaxWorkItemSizesVariant =
          Dev.get_info<info::device::max_work_item_sizes<2>>();
    else // (Dims == 3)
      MaxWorkItemSizesVariant =
          Dev.get_info<info::device::max_work_item_sizes<3>>();
    for (int i = 0; i < Dims; i++) {
      // Extracting value from std::variant to avoid dealing with type-safety
      // issues after that
      if (Dims == 1) {
        // ReqdWGSizeVec is in reverse order compared to MaxWorkItemSizes
        if (ReqdWGSizeVec[i] >
            std::get<id<1>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return sycl::exception(sycl::errc::kernel_not_supported,
                                 "Required work-group size " +
                                     std::to_string(ReqdWGSizeVec[i]) +
                                     " is not supported");
      } else if (Dims == 2) {
        if (ReqdWGSizeVec[i] >
            std::get<id<2>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return sycl::exception(sycl::errc::kernel_not_supported,
                                 "Required work-group size " +
                                     std::to_string(ReqdWGSizeVec[i]) +
                                     " is not supported");
      } else // (Dims == 3)
        if (ReqdWGSizeVec[i] >
            std::get<id<3>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return sycl::exception(sycl::errc::kernel_not_supported,
                                 "Required work-group size " +
                                     std::to_string(ReqdWGSizeVec[i]) +
                                     " is not supported");
    }
  }

  // Check if device supports required sub-group size.
  if (ReqdSubGroupSizePropIt) {
    auto ReqdSubGroupSize =
        DeviceBinaryProperty(*(ReqdSubGroupSizePropIt.value())).asUint32();
    auto SupportedSubGroupSizes = Dev.get_info<info::device::sub_group_sizes>();
    // !getUint32PropAsBool(Img, "isEsimdImage") is a WA for ESIMD,
    // as ESIMD images have a reqd-sub-group-size of 1, but currently
    // no backend currently includes 1 as a valid sub-group size.
    // This can be removed if backends add 1 as a valid sub-group size.
    if (!getUint32PropAsBool(Img, "isEsimdImage") &&
        std::none_of(SupportedSubGroupSizes.cbegin(),
                     SupportedSubGroupSizes.cend(),
                     [=](auto s) { return s == ReqdSubGroupSize; }))
      return sycl::exception(sycl::errc::kernel_not_supported,
                             "Sub-group size " +
                                 std::to_string(ReqdSubGroupSize) +
                                 " is not supported on the device");
  }

  return {};
}

} // namespace detail
} // namespace _V1
} // namespace sycl

extern "C" void __sycl_register_lib(pi_device_binaries desc) {
  sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(pi_device_binaries desc) {
  (void)desc;
  // TODO implement the function
}
