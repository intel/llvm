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
#include <sycl/ext/oneapi/experimental/spec_constant.hpp>
#include <sycl/stl.hpp>

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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;

static constexpr int DbgProgMgr = 0;

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

/// This function enables ITT annotations in SPIR-V module by setting
/// a specialization constant if INTEL_LIBITTNOTIFY64 env variable is set.
static void enableITTAnnotationsIfNeeded(const RT::PiProgram &Prog,
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

static RT::PiProgram
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

  RT::PiProgram Program;
  const RT::PiDevice PiDevice = getSyclObjImpl(Device)->getHandleRef();
  pi_int32 BinaryStatus = CL_SUCCESS;
  Plugin->call<PiApiKind::piProgramCreateWithBinary>(
      Context->getHandleRef(), 1 /*one binary*/, &PiDevice, &DataLen, &Data,
      Metadata.size(), Metadata.data(), &BinaryStatus, &Program);

  if (BinaryStatus != CL_SUCCESS) {
    throw runtime_error("Creating program with binary failed.", BinaryStatus);
  }

  return Program;
}

static RT::PiProgram createSpirvProgram(const ContextImplPtr Context,
                                        const unsigned char *Data,
                                        size_t DataLen) {
  RT::PiProgram Program = nullptr;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piProgramCreate>(Context->getHandleRef(), Data,
                                           DataLen, &Program);
  return Program;
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(OSModuleHandle M, const std::string &KernelName,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \""
              << KernelName << "\", " << getRawSyclObjImpl(Context) << ", "
              << getRawSyclObjImpl(Device) << ", " << JITCompilationIsRequired
              << ")\n";

  KernelSetId KSId = getKernelSetId(M, KernelName);
  return getDeviceImage(M, KSId, Context, Device, JITCompilationIsRequired);
}

/// Try to fetch entity (kernel or program) from cache. If there is no such
/// entity try to build it. Throw any exception build process may throw.
/// This method eliminates unwanted builds by employing atomic variable with
/// build state and waiting until the entity is built in another thread.
/// If the building thread has failed the awaiting thread will fail either.
/// Exception thrown by build procedure are rethrown.
///
/// \tparam RetT type of entity to get
/// \tparam ExceptionT type of exception to throw on awaiting thread if the
///         building thread fails build step.
/// \tparam KeyT key (in cache) to fetch built entity with
/// \tparam AcquireFT type of function which will acquire the locked version of
///         the cache. Accept reference to KernelProgramCache.
/// \tparam GetCacheFT type of function which will fetch proper cache from
///         locked version. Accepts reference to locked version of cache.
/// \tparam BuildFT type of function which will build the entity if it is not in
///         cache. Accepts nothing. Return pointer to built entity.
///
/// \return a pointer to cached build result, return value must not be nullptr.
template <typename RetT, typename ExceptionT, typename GetCachedBuildFT,
          typename BuildFT>
KernelProgramCache::BuildResult<RetT> *
getOrBuild(KernelProgramCache &KPCache, GetCachedBuildFT &&GetCachedBuild,
           BuildFT &&Build) {
  using BuildState = KernelProgramCache::BuildState;

  auto [BuildResult, InsertionTookPlace] = GetCachedBuild();

  // no insertion took place, thus some other thread has already inserted smth
  // in the cache
  if (!InsertionTookPlace) {
    for (;;) {
      RetT *Result = KPCache.waitUntilBuilt<ExceptionT>(BuildResult);

      if (Result)
        return BuildResult;

      // Previous build is failed. There was no SYCL exception though.
      // We might try to build once more.
      BuildState Expected = BuildState::BS_Failed;
      BuildState Desired = BuildState::BS_InProgress;

      if (BuildResult->State.compare_exchange_strong(Expected, Desired))
        break; // this thread is the building thread now
    }
  }

  // only the building thread will run this
  try {
    BuildResult->Val = Build();
    RetT *Desired = &BuildResult->Val;

#ifndef NDEBUG
    RetT *Expected = nullptr;

    if (!BuildResult->Ptr.compare_exchange_strong(Expected, Desired))
      // We've got a funny story here
      assert(false && "We've build an entity that is already have been built.");
#else
    BuildResult->Ptr.store(Desired);
#endif

    {
      // Even if shared variable is atomic, it must be modified under the mutex
      // in order to correctly publish the modification to the waiting thread
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BuildState::BS_Done);
    }

    KPCache.notifyAllBuild(*BuildResult);

    return BuildResult;
  } catch (const exception &Ex) {
    BuildResult->Error.Msg = Ex.what();
    BuildResult->Error.Code = Ex.get_cl_code();

    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BuildState::BS_Failed);
    }

    KPCache.notifyAllBuild(*BuildResult);

    std::rethrow_exception(std::current_exception());
  } catch (...) {
    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BuildState::BS_Failed);
    }

    KPCache.notifyAllBuild(*BuildResult);

    std::rethrow_exception(std::current_exception());
  }
}

// TODO replace this with a new PI API function
static bool isDeviceBinaryTypeSupported(const context &C,
                                        RT::PiDeviceBinaryType Format) {
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

static const char *getFormatStr(RT::PiDeviceBinaryType Format) {
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

RT::PiProgram ProgramManager::createPIProgram(const RTDeviceBinaryImage &Img,
                                              const context &Context,
                                              const device &Device) {
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
  RT::PiDeviceBinaryType Format = Img.getFormat();

  if (Format == PI_DEVICE_BINARY_TYPE_NONE)
    Format = pi::getBinaryImageFormat(RawImg.BinaryStart, ImgSize);
  // RT::PiDeviceBinaryType Format = Img->Format;
  // assert(Format != PI_DEVICE_BINARY_TYPE_NONE && "Image format not set");

  if (!isDeviceBinaryTypeSupported(Context, Format))
    throw feature_not_supported(
        "SPIR-V online compilation is not supported in this context",
        PI_ERROR_INVALID_OPERATION);

  // Get program metadata from properties
  auto ProgMetadata = Img.getProgramMetadata();
  std::vector<pi_device_binary_property> ProgMetadataVector{
      ProgMetadata.begin(), ProgMetadata.end()};

  // Load the image
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  RT::PiProgram Res = Format == PI_DEVICE_BINARY_TYPE_SPIRV
                          ? createSpirvProgram(Ctx, RawImg.BinaryStart, ImgSize)
                          : createBinaryProgram(Ctx, Device, RawImg.BinaryStart,
                                                ImgSize, ProgMetadataVector);

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

static void appendCompileOptionsForRegAllocMode(std::string &CompileOpts,
                                                const RTDeviceBinaryImage &Img,
                                                bool IsEsimdImage) {
  pi_device_binary_property Prop = Img.getProperty("sycl-register-alloc-mode");
  if (!Prop)
    return;
  uint32_t PropVal = DeviceBinaryProperty(Prop).asUint32();
  if (PropVal == static_cast<uint32_t>(register_alloc_mode_enum::large)) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    // This option works for both LO AND OCL backends.
    CompileOpts += IsEsimdImage ? "-doubleGRF" : "-ze-opt-large-register-file";
  }
  // TODO: Support Auto GRF for ESIMD once vc supports it.
  if (PropVal == static_cast<uint32_t>(register_alloc_mode_enum::automatic) &&
      !IsEsimdImage) {
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

  appendCompileOptionsForRegAllocMode(CompileOpts, Img, isEsimdImage);

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
  if ((PlatformImpl->getBackend() == backend::ext_oneapi_level_zero ||
       PlatformImpl->getBackend() == backend::opencl) &&
      std::all_of(Devs.begin(), Devs.end(),
                  [](const device &Dev) { return Dev.is_gpu(); }) &&
      Img.getDeviceGlobals().size() != 0) {
    // If the image has device globals we need to add the
    // -ze-take-global-address option to tell IGC to record addresses of these.
    if (!CompileOpts.empty())
      CompileOpts += " ";
    CompileOpts += "-ze-take-global-address";
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

std::pair<RT::PiProgram, bool> ProgramManager::getOrCreatePIProgram(
    const RTDeviceBinaryImage &Img, const context &Context,
    const device &Device, const std::string &CompileAndLinkOptions,
    SerializedObj SpecConsts) {
  RT::PiProgram NativePrg;

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

RT::PiProgram ProgramManager::getBuiltPIProgram(
    OSModuleHandle M, const ContextImplPtr &ContextImpl,
    const DeviceImplPtr &DeviceImpl, const std::string &KernelName,
    const program_impl *Prg, bool JITCompilationIsRequired) {
  // TODO: Make sure that KSIds will be different for the case when the same
  // kernel built with different options is present in the fat binary.
  KernelSetId KSId = getKernelSetId(M, KernelName);

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  std::string CompileOpts;
  std::string LinkOpts;
  if (Prg) {
    CompileOpts = Prg->get_build_options();
  }

  applyOptionsFromEnvironment(CompileOpts, LinkOpts);

  SerializedObj SpecConsts;
  if (Prg)
    Prg->stableSerializeSpecConstRegistry(SpecConsts);

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
      getDeviceImage(M, KSId, Context, Device, JITCompilationIsRequired);

  // Check that device supports all aspects used by the kernel
  const RTDeviceBinaryImage::PropertyRange &ARange =
      Img.getDeviceRequirements();

#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case aspect::ASPECT:                                                         \
    return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
// We don't need "case aspect::usm_allocator" here because it will duplicate
// "case aspect::usm_system_allocations", therefore leave this macro empty
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
  auto getAspectNameStr = [](aspect AspectNum) -> std::string {
    switch (AspectNum) {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
    }
    throw sycl::exception(errc::kernel_not_supported,
                          "Unknown aspect " +
                              std::to_string(static_cast<unsigned>(AspectNum)));
  };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT

  for (RTDeviceBinaryImage::PropertyRange::ConstIterator It : ARange) {
    using namespace std::literals;
    if ((*It)->Name != "aspects"sv)
      continue;
    ByteArray Aspects = DeviceBinaryProperty(*It).asByteArray();
    // 8 because we need to skip 64-bits of size of the byte array
    auto *AIt = reinterpret_cast<const std::uint32_t *>(&Aspects[8]);
    auto *AEnd =
        reinterpret_cast<const std::uint32_t *>(&Aspects[0] + Aspects.size());
    while (AIt != AEnd) {
      auto Aspect = static_cast<aspect>(*AIt);
      // Strict check for fp64 is disabled temporarily to avoid confusion.
      if (!Dev->has(Aspect))
        throw sycl::exception(errc::kernel_not_supported,
                              "Required aspect " + getAspectNameStr(Aspect) +
                                  " is not supported on the device");
      ++AIt;
    }
  }

  auto BuildF = [this, &Img, &Context, &ContextImpl, &Device, Prg, &CompileOpts,
                 &LinkOpts, SpecConsts] {
    const PluginPtr &Plugin = ContextImpl->getPlugin();
    applyOptionsFromImage(CompileOpts, LinkOpts, Img, {Device}, Plugin);

    auto [NativePrg, DeviceCodeWasInCache] = getOrCreatePIProgram(
        Img, Context, Device, CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache) {
      if (Prg)
        flushSpecConstants(*Prg, NativePrg, &Img);
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
  const RT::PiDevice PiDevice = Dev->getHandleRef();
  auto CacheKey =
      std::make_pair(std::make_pair(std::move(SpecConsts), ImgId),
                     std::make_pair(PiDevice, CompileOpts + LinkOpts));

  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  auto BuildResult = getOrBuild<RT::PiProgram, compile_program_error>(
      Cache, GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  return *BuildResult->Ptr.load();
}

std::tuple<RT::PiKernel, std::mutex *, const KernelArgMask *, RT::PiProgram>
ProgramManager::getOrCreateKernel(OSModuleHandle M,
                                  const ContextImplPtr &ContextImpl,
                                  const DeviceImplPtr &DeviceImpl,
                                  const std::string &KernelName,
                                  const program_impl *Prg) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << ContextImpl.get() << ", " << DeviceImpl.get() << ", "
              << KernelName << ")\n";
  }

  using KernelArgMaskPairT = KernelProgramCache::KernelArgMaskPairT;

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  std::string CompileOpts, LinkOpts;
  SerializedObj SpecConsts;
  if (Prg) {
    CompileOpts = Prg->get_build_options();
    Prg->stableSerializeSpecConstRegistry(SpecConsts);
  }
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);
  const RT::PiDevice PiDevice = DeviceImpl->getHandleRef();

  auto key = std::make_tuple(std::move(SpecConsts), M, PiDevice,
                             CompileOpts + LinkOpts, KernelName);
  auto ret_tuple = Cache.tryToGetKernelFast(key);
  if (std::get<0>(ret_tuple))
    return ret_tuple;

  RT::PiProgram Program =
      getBuiltPIProgram(M, ContextImpl, DeviceImpl, KernelName, Prg);

  auto BuildF = [this, &Program, &KernelName, &ContextImpl, M] {
    RT::PiKernel Kernel = nullptr;

    const PluginPtr &Plugin = ContextImpl->getPlugin();
    Plugin->call<errc::kernel_not_supported, PiApiKind::piKernelCreate>(
        Program, KernelName.c_str(), &Kernel);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    Plugin->call<PiApiKind::piKernelSetExecInfo>(Kernel, PI_USM_INDIRECT_ACCESS,
                                                 sizeof(pi_bool), &PI_TRUE);

    const KernelArgMask *ArgMask =
        getEliminatedKernelArgMask(M, Program, KernelName);
    return std::make_pair(Kernel, ArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  auto BuildResult = getOrBuild<KernelArgMaskPairT, invalid_object_error>(
      Cache, GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  const KernelArgMaskPairT &KernelArgMaskPair = *BuildResult->Ptr.load();
  auto ret_val = std::make_tuple(KernelArgMaskPair.first,
                                 &(BuildResult->MBuildResultMutex),
                                 KernelArgMaskPair.second, Program);
  Cache.saveKernel(key, ret_val);
  return ret_val;
}

RT::PiProgram
ProgramManager::getPiProgramFromPiKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context) {
  RT::PiProgram Program;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piKernelGetInfo>(
      Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(RT::PiProgram), &Program, nullptr);
  return Program;
}

std::string ProgramManager::getProgramBuildLog(const RT::PiProgram &Program,
                                               const ContextImplPtr Context) {
  size_t PIDevicesSize = 0;
  const PluginPtr &Plugin = Context->getPlugin();
  Plugin->call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, 0,
                                            nullptr, &PIDevicesSize);
  std::vector<RT::PiDevice> PIDevices(PIDevicesSize / sizeof(RT::PiDevice));
  Plugin->call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES,
                                            PIDevicesSize, PIDevices.data(),
                                            nullptr);
  std::string Log = "The program was built for " +
                    std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
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
                          RT::PiProgram &Prog) {
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

static RT::PiProgram loadDeviceLibFallback(const ContextImplPtr Context,
                                           DeviceLibExt Extension,
                                           const RT::PiDevice &Device,
                                           bool UseNativeLib) {

  auto LibFileName = getDeviceLibFilename(Extension, UseNativeLib);

  auto LockedCache = Context->acquireCachedLibPrograms();
  auto CachedLibPrograms = LockedCache.get();
  auto CacheResult = CachedLibPrograms.emplace(
      std::make_pair(std::make_pair(Extension, Device), nullptr));
  bool Cached = !CacheResult.second;
  auto LibProgIt = CacheResult.first;
  RT::PiProgram &LibProg = LibProgIt->second;

  if (Cached)
    return LibProg;

  if (!loadDeviceLib(Context, LibFileName, LibProg)) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(std::string("Failed to load ") + LibFileName,
                                PI_ERROR_INVALID_VALUE);
  }

  const PluginPtr &Plugin = Context->getPlugin();
  // TODO no spec constants are used in the std libraries, support in the future
  RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piProgramCompile>(
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

ProgramManager::ProgramManager() {
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
    auto ImgPtr = make_unique_ptr<DynRTDeviceBinaryImage>(
        std::move(Data), Size, OSUtil::DummyModuleHandle);

    if (DbgProgMgr > 0) {
      std::cerr << "loaded device image binary from " << SpvFile << "\n";
      std::cerr << "format: " << getFormatStr(ImgPtr->getFormat()) << "\n";
    }
    // No need for a mutex here since all access to these private fields is
    // blocked until the construction of the ProgramManager singleton is
    // finished.
    m_DeviceImages[SpvFileKSId].reset(
        new std::vector<RTDeviceBinaryImageUPtr>());
    m_DeviceImages[SpvFileKSId]->push_back(std::move(ImgPtr));
  }
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \"" << KSId
              << "\", " << getRawSyclObjImpl(Context) << ", "
              << getRawSyclObjImpl(Device) << ", " << JITCompilationIsRequired
              << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  auto It = m_DeviceImages.find(KSId);
  assert(It != m_DeviceImages.end() &&
         "No device image found for the given kernel set id");
  std::vector<RTDeviceBinaryImageUPtr> &Imgs = *It->second;
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  pi_uint32 ImgInd = 0;
  RTDeviceBinaryImage *Img = nullptr;

  // TODO: There may be cases with sycl::program class usage in source code
  // that will result in a multi-device context. This case needs to be handled
  // here or at the program_impl class level

  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  std::vector<pi_device_binary> RawImgs(Imgs.size());
  for (unsigned I = 0; I < Imgs.size(); I++)
    RawImgs[I] = const_cast<pi_device_binary>(&Imgs[I]->getRawData());

  Ctx->getPlugin()->call<PiApiKind::piextDeviceSelectBinary>(
      getSyclObjImpl(Device)->getHandleRef(), RawImgs.data(),
      (pi_uint32)RawImgs.size(), &ImgInd);

  if (JITCompilationIsRequired) {
    // If the image is already compiled with AOT, throw an exception.
    const pi_device_binary_struct &RawImg = Imgs[ImgInd]->getRawData();
    if ((strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
        (strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
        (strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0)) {
      throw feature_not_supported("Recompiling AOT image is not supported",
                                  PI_ERROR_INVALID_OPERATION);
    }
  }

  Img = Imgs[ImgInd].get();

  if (DbgProgMgr > 0) {
    std::cerr << "selected device image: " << &Img->getRawData() << "\n";
    Img->print();
  }
  return *Img;
}

static bool isDeviceLibRequired(DeviceLibExt Ext, uint32_t DeviceLibReqMask) {
  uint32_t Mask =
      0x1 << (static_cast<uint32_t>(Ext) -
              static_cast<uint32_t>(DeviceLibExt::cl_intel_devicelib_assert));
  return ((DeviceLibReqMask & Mask) == Mask);
}

static std::vector<RT::PiProgram>
getDeviceLibPrograms(const ContextImplPtr Context, const RT::PiDevice &Device,
                     uint32_t DeviceLibReqMask) {
  std::vector<RT::PiProgram> Programs;

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
         Ext == DeviceLibExt::cl_intel_devicelib_complex_fp64) &&
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

ProgramManager::ProgramPtr
ProgramManager::build(ProgramPtr Program, const ContextImplPtr Context,
                      const std::string &CompileOptions,
                      const std::string &LinkOptions,
                      const RT::PiDevice &Device, uint32_t DeviceLibReqMask) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", ... " << Device
              << ")\n";
  }

  // TODO: old sycl compiler always marks cassert fallback device library as
  // "required", this will lead to compatibilty issue when we enable online
  // link in SYCL runtime. If users compile their code with old compiler and run
  // their executable with latest SYCL runtime, cassert fallback spv file will
  // always be loaded which is not expected, cassert device library development
  // is still in progress, the unexpected loading may lead to runtime problem.
  // So, we clear bit 0 in device library require mask to avoid loading cassert
  // fallback device library and will revert this when cassert development is
  // done.
  DeviceLibReqMask &= 0xFFFFFFFE;
  bool LinkDeviceLibs = (DeviceLibReqMask != 0);

  // TODO: this is a temporary workaround for GPU tests for ESIMD compiler.
  // We do not link with other device libraries, because it may fail
  // due to unrecognized SPIR-V format of those libraries.
  if (CompileOptions.find(std::string("-cmc")) != std::string::npos ||
      CompileOptions.find(std::string("-vc-codegen")) != std::string::npos)
    LinkDeviceLibs = false;

  std::vector<RT::PiProgram> LinkPrograms;
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
    RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piProgramBuild>(
        Program.get(), /*num devices =*/1, &Device, Options.c_str(), nullptr,
        nullptr);
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

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piProgramLink>(
      Context->getHandleRef(), /*num devices =*/1, &Device, LinkOptions.c_str(),
      LinkPrograms.size(), LinkPrograms.data(), nullptr, nullptr, &LinkedProg);

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

void ProgramManager::cacheKernelUsesAssertInfo(OSModuleHandle M,
                                               RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &AssertUsedRange =
      Img.getAssertUsed();
  if (AssertUsedRange.isAvailable())
    for (const auto &Prop : AssertUsedRange) {
      KernelNameWithOSModule Key{Prop->Name, M};
      m_KernelUsesAssert.insert(Key);
    }
}

bool ProgramManager::kernelUsesAssert(OSModuleHandle M,
                                      const std::string &KernelName) const {
  KernelNameWithOSModule Key{KernelName, M};
  return m_KernelUsesAssert.find(Key) != m_KernelUsesAssert.end();
}

void ProgramManager::addImages(pi_device_binaries DeviceBinary) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  const bool DumpImages = std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile;
  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(RawImg);
    const _pi_offload_entry EntriesB = RawImg->EntriesBegin;
    const _pi_offload_entry EntriesE = RawImg->EntriesEnd;
    auto Img = make_unique_ptr<RTDeviceBinaryImage>(RawImg, M);
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
    if (EntriesB != EntriesE) {
      std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

      // Register all exported symbols
      auto ExportedSymbols = Img->getExportedSymbols();
      for (const pi_device_binary_property &ExportedSymbol : ExportedSymbols)
        m_ExportedSymbols.insert(ExportedSymbol->Name);

      m_BinImg2KernelIDs[Img.get()].reset(new std::vector<kernel_id>);

      for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
           ++EntriesIt) {

        // Skip creating unique kernel ID if it is a service kernel.
        // SYCL service kernels are identified by having
        // __sycl_service_kernel__ in the mangled name, primarily as part of
        // the namespace of the name type.
        if (std::strstr(EntriesIt->name, "__sycl_service_kernel__")) {
          m_ServiceKernels.insert(EntriesIt->name);
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

          It = m_KernelName2KernelIDs.emplace_hint(It, EntriesIt->name,
                                                   KernelID);
        }

        m_KernelIDs2BinImage.insert(std::make_pair(It->second, Img.get()));
        m_BinImg2KernelIDs[Img.get()]->push_back(It->second);
      }

      // Sort kernel ids for faster search
      std::sort(m_BinImg2KernelIDs[Img.get()]->begin(),
                m_BinImg2KernelIDs[Img.get()]->end(), LessByHash<kernel_id>{});
    }

    // TODO: Remove the code below once program manager works trought kernel
    // bundles only
    // Use the entry information if it's available
    if (EntriesB != EntriesE) {
      // The kernel sets for any pair of images are either disjoint or
      // identical, look up the kernel set using the first kernel name...
      StrToKSIdMap &KSIdMap = m_KernelSets[M];
      auto KSIdIt = KSIdMap.find(EntriesB->name);
      if (KSIdIt != KSIdMap.end()) {
        auto &Imgs = m_DeviceImages[KSIdIt->second];
        assert(Imgs && "Device image vector should have been already created");
        if (DumpImages) {
          const bool NeedsSequenceID =
              std::any_of(Imgs->begin(), Imgs->end(), [&](auto &I) {
                return I->getFormat() == Img->getFormat();
              });
          dumpImage(*Img, KSIdIt->second, NeedsSequenceID ? ++SequenceID : 0);
        }

        cacheKernelUsesAssertInfo(M, *Img);

        Imgs->push_back(std::move(Img));
        continue;
      }
      // ... or create the set first if it hasn't been
      KernelSetId KSId = getNextKernelSetId();
      {
        std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

        for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
             ++EntriesIt) {
          KSIdMap.insert(std::make_pair(EntriesIt->name, KSId));
        }
      }
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
          uintptr_t ImgId = reinterpret_cast<uintptr_t>(Img.get());

          auto ExistingDeviceGlobal = m_DeviceGlobals.find(DeviceGlobal->Name);
          if (ExistingDeviceGlobal != m_DeviceGlobals.end()) {
            // If it has already been registered we update the information.
            ExistingDeviceGlobal->second->initialize(ImgId, KSId, TypeSize,
                                                     DeviceImageScopeDecorated);
          } else {
            // If it has not already been registered we create a new entry.
            // Note: Pointer to the device global is not available here, so it
            //       cannot be set until registration happens.
            auto EntryUPtr = std::make_unique<DeviceGlobalMapEntry>(
                DeviceGlobal->Name, ImgId, KSId, TypeSize,
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
      m_DeviceImages[KSId].reset(new std::vector<RTDeviceBinaryImageUPtr>());
      cacheKernelUsesAssertInfo(M, *Img);

      if (DumpImages)
        dumpImage(*Img, KSId);
      m_DeviceImages[KSId]->push_back(std::move(Img));

      continue;
    }
    // Otherwise assume that the image contains all kernels associated with the
    // module
    KernelSetId &KSId = m_OSModuleKernelSets[M];
    if (KSId == 0)
      KSId = getNextKernelSetId();

    auto &Imgs = m_DeviceImages[KSId];
    if (!Imgs)
      Imgs.reset(new std::vector<RTDeviceBinaryImageUPtr>());

    cacheKernelUsesAssertInfo(M, *Img);

    if (DumpImages)
      dumpImage(*Img, KSId);
    Imgs->push_back(std::move(Img));
  }
}

void ProgramManager::debugPrintBinaryImages() const {
  for (const auto &ImgVecIt : m_DeviceImages) {
    std::cerr << "  ++++++ Kernel set: " << ImgVecIt.first << "\n";
    for (const auto &Img : *ImgVecIt.second)
      Img.get()->print();
  }
}

KernelSetId ProgramManager::getNextKernelSetId() const {
  // No need for atomic, should be guarded by the caller
  static KernelSetId Result = LastKSId;
  return ++Result;
}

KernelSetId
ProgramManager::getKernelSetId(OSModuleHandle M,
                               const std::string &KernelName) const {
  // If the env var instructs to use image from a file,
  // return the kernel set associated with it
  if (m_UseSpvFile && M == OSUtil::ExeModuleHandle)
    return SpvFileKSId;
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  auto KSIdMapIt = m_KernelSets.find(M);
  if (KSIdMapIt != m_KernelSets.end()) {
    const StrToKSIdMap &KSIdMap = KSIdMapIt->second;
    auto KSIdIt = KSIdMap.find(KernelName);
    // If the kernel has been assigned to a kernel set, return it
    if (KSIdIt != KSIdMap.end())
      return KSIdIt->second;
  }
  // If no kernel set was found check if there is a kernel set containing
  // all kernels in the given module
  auto ModuleKSIdIt = m_OSModuleKernelSets.find(M);
  if (ModuleKSIdIt != m_OSModuleKernelSets.end())
    return ModuleKSIdIt->second;

  throw runtime_error("No kernel named " + KernelName + " was found",
                      PI_ERROR_INVALID_KERNEL_NAME);
}

void ProgramManager::dumpImage(const RTDeviceBinaryImage &Img, KernelSetId KSId,
                               uint32_t SequenceID) const {
  std::string Fname("sycl_");
  const pi_device_binary_struct &RawImg = Img.getRawData();
  Fname += RawImg.DeviceTargetSpec;
  Fname += std::to_string(KSId);
  if (SequenceID)
    Fname += '_' + std::to_string(SequenceID);
  std::string Ext;

  RT::PiDeviceBinaryType Format = Img.getFormat();
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
                                        RT::PiProgram NativePrg,
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
        throw sycl::ext::oneapi::experimental::spec_const_error(
            "spec constant is set in a program w/o a binary image",
            PI_ERROR_INVALID_OPERATION);
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

// If the kernel is loaded from spv file, it may not include DeviceLib require
// mask, sycl runtime won't know which fallback device libraries are needed. In
// such case, the safest way is to load all fallback device libraries.
uint32_t ProgramManager::getDeviceLibReqMask(const RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &DLMRange =
      Img.getDeviceLibReqMask();
  if (DLMRange.isAvailable())
    return DeviceBinaryProperty(*(DLMRange.begin())).asUint32();
  else
    return 0xFFFFFFFF;
}

// This version does not check m_UseSpvFile, but it's used in the kernel_bundle
// path, which does not currently check it and always uses images from the fat
// binary anyway.
// TODO consider making m_UseSpvFile interact with kernel bundles as well.
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

// TODO consider another approach with storing the masks in the integration
// header instead.
const KernelArgMask *ProgramManager::getEliminatedKernelArgMask(
    OSModuleHandle M, pi::PiProgram NativePrg, const std::string &KernelName) {
  // If instructed to use a spv file, assume no eliminated arguments.
  if (m_UseSpvFile && M == OSUtil::ExeModuleHandle)
    return nullptr;
  return getEliminatedKernelArgMask(NativePrg, KernelName);
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

  const RT::PiDevice &PIDeviceHandle = DeviceImpl->getHandleRef();

  // Call piextDeviceSelectBinary with only one image to check if an image is
  // compatible with implementation. The function returns invalid index if no
  // device images are compatible.
  pi_uint32 SuitableImageID = std::numeric_limits<pi_uint32>::max();
  pi_device_binary DevBin =
      const_cast<pi_device_binary>(&BinImage->getRawData());
  RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piextDeviceSelectBinary>(
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
    std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
    for (auto &ImagesSets : m_DeviceImages) {
      auto &ImagesUPtrs = *ImagesSets.second.get();
      for (auto &ImageUPtr : ImagesUPtrs)
        BinImages.insert(ImageUPtr.get());
    }
  }
  assert(BinImages.size() > 0 && "Expected to find at least one device image");

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

    switch (TargetState) {
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
                           RT::PiProgram Prog, const PluginPtr &Plugin) {
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
  RT::PiProgram Prog = createPIProgram(*InputImpl->get_bin_image_ref(),
                                       InputImpl->get_context(), Devs[0]);

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
  RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piProgramCompile>(
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
  const context &Context = getSyclObjImpl(DeviceImage)->get_context();
  const ContextImplPtr ContextImpl = getSyclObjImpl(Context);
  const PluginPtr &Plugin = ContextImpl->getPlugin();

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin->call_nocheck<PiApiKind::piProgramLink>(
      ContextImpl->getHandleRef(), PIDevices.size(), PIDevices.data(),
      /*options=*/LinkOptionsStr.c_str(), PIPrograms.size(), PIPrograms.data(),
      /*pfn_notify=*/nullptr,
      /*user_data=*/nullptr, &LinkedProg);

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

  uint32_t ImgId = Img.getImageID();
  const RT::PiDevice PiDevice = getRawSyclObjImpl(Devs[0])->getHandleRef();
  auto CacheKey =
      std::make_pair(std::make_pair(std::move(SpecConsts), ImgId),
                     std::make_pair(PiDevice, CompileOpts + LinkOpts));

  // CacheKey is captured by reference so when we overwrite it later we can
  // reuse this function.
  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  // TODO: Throw SYCL2020 style exception
  auto BuildResult = getOrBuild<RT::PiProgram, compile_program_error>(
      Cache, GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  RT::PiProgram ResProgram = *BuildResult->Ptr.load();

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
    const RT::PiDevice PiDeviceAdd =
        getRawSyclObjImpl(Devs[Idx])->getHandleRef();

    // Change device in the cache key to reduce copying of spec const data.
    CacheKey.second.first = PiDeviceAdd;
    getOrBuild<RT::PiProgram, compile_program_error>(Cache, GetCachedBuildF,
                                                     CacheOtherDevices);
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

std::tuple<RT::PiKernel, std::mutex *, const KernelArgMask *>
ProgramManager::getOrCreateKernel(const context &Context,
                                  const std::string &KernelName,
                                  const property_list &PropList,
                                  RT::PiProgram Program) {

  (void)PropList;

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto BuildF = [this, &Program, &KernelName, &Ctx] {
    RT::PiKernel Kernel = nullptr;

    const PluginPtr &Plugin = Ctx->getPlugin();
    Plugin->call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                            &Kernel);

    Plugin->call<PiApiKind::piKernelSetExecInfo>(Kernel, PI_USM_INDIRECT_ACCESS,
                                                 sizeof(pi_bool), &PI_TRUE);

    const KernelArgMask *KernelArgMask =
        getEliminatedKernelArgMask(Program, KernelName);
    return std::make_pair(Kernel, KernelArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  auto BuildResult =
      getOrBuild<KernelProgramCache::KernelArgMaskPairT, invalid_object_error>(
          Cache, GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  return std::make_tuple(BuildResult->Ptr.load()->first,
                         &(BuildResult->MBuildResultMutex),
                         BuildResult->Ptr.load()->second);
}

bool doesDevSupportDeviceRequirements(const device &Dev,
                                      const RTDeviceBinaryImage &Img) {
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
  auto ReqdWGSizePropIt = getPropIt("reqd_work_group_size");

  if (!AspectsPropIt && !ReqdWGSizePropIt)
    return true;

  // Checking if device supports defined aspects
  if (AspectsPropIt) {
    ByteArray Aspects =
        DeviceBinaryProperty(*(AspectsPropIt.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    Aspects.dropBytes(8);
    while (!Aspects.empty()) {
      aspect Aspect = Aspects.consume<aspect>();
      // Strict check for fp64 is disabled temporarily to avoid confusion.
      if (!Dev.has(Aspect))
        return false;
    }
  }

  // Checking if device supports defined required work group size
  if (ReqdWGSizePropIt) {
    ByteArray ReqdWGSize =
        DeviceBinaryProperty(*(ReqdWGSizePropIt.value())).asByteArray();
    // Drop 8 bytes describing the size of the byte array.
    ReqdWGSize.dropBytes(8);
    int ReqdWGSizeAllDimsTotal = 1;
    std::vector<int> ReqdWGSizeVec;
    int Dims = 0;
    while (!ReqdWGSize.empty()) {
      int SingleDimSize = ReqdWGSize.consume<int>();
      ReqdWGSizeAllDimsTotal *= SingleDimSize;
      ReqdWGSizeVec.push_back(SingleDimSize);
      Dims++;
    }
    if (static_cast<size_t>(ReqdWGSizeAllDimsTotal) >
        Dev.get_info<info::device::max_work_group_size>())
      return false;
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
        if (static_cast<size_t>(ReqdWGSizeVec[i]) >
            std::get<id<1>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return false;
      } else if (Dims == 2) {
        if (static_cast<size_t>(ReqdWGSizeVec[i]) >
            std::get<id<2>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return false;
      } else // (Dims == 3)
        if (static_cast<size_t>(ReqdWGSizeVec[i]) >
            std::get<id<3>>(MaxWorkItemSizesVariant)[Dims - i - 1])
          return false;
    }
  }
  return true;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

extern "C" void __sycl_register_lib(pi_device_binaries desc) {
  sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(pi_device_binaries desc) {
  (void)desc;
  // TODO implement the function
}
