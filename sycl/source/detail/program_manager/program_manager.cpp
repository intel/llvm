//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/program_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/ext/oneapi/experimental/spec_constant.hpp>

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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;

static constexpr int DbgProgMgr = 0;

enum BuildState { BS_InProgress, BS_Done, BS_Failed };

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

/// This function enables ITT annotations in SPIR-V module by setting
/// a specialization constant if INTEL_LIBITTNOTIFY64 env variable is set.
static void enableITTAnnotationsIfNeeded(const RT::PiProgram &Prog,
                                         const plugin &Plugin) {
  if (SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::get() != nullptr) {
    constexpr char SpecValue = 1;
    Plugin.call<PiApiKind::piextProgramSetSpecializationConstant>(
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
  const detail::plugin &Plugin = Context->getPlugin();
#ifndef _NDEBUG
  pi_uint32 NumDevices = 0;
  Plugin.call<PiApiKind::piContextGetInfo>(Context->getHandleRef(),
                                           PI_CONTEXT_INFO_NUM_DEVICES,
                                           sizeof(NumDevices), &NumDevices,
                                           /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  RT::PiProgram Program;
  const RT::PiDevice PiDevice = getSyclObjImpl(Device)->getHandleRef();
  pi_int32 BinaryStatus = CL_SUCCESS;
  Plugin.call<PiApiKind::piProgramCreateWithBinary>(
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
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piProgramCreate>(Context->getHandleRef(), Data,
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

template <typename ExceptionT, typename RetT>
RetT *waitUntilBuilt(KernelProgramCache &Cache,
                     KernelProgramCache::BuildResult<RetT> *BuildResult) {
  // any thread which will find nullptr in cache will wait until the pointer
  // is not null anymore
  Cache.waitUntilBuilt(*BuildResult, [BuildResult]() {
    int State = BuildResult->State.load();

    return State == BS_Done || State == BS_Failed;
  });

  if (BuildResult->Error.isFilledIn()) {
    const KernelProgramCache::BuildError &Error = BuildResult->Error;
    throw ExceptionT(Error.Msg, Error.Code);
  }

  RetT *Result = BuildResult->Ptr.load();

  return Result;
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
template <typename RetT, typename ExceptionT, typename KeyT, typename AcquireFT,
          typename GetCacheFT, typename BuildFT>
KernelProgramCache::BuildResult<RetT> *
getOrBuild(KernelProgramCache &KPCache, KeyT &&CacheKey, AcquireFT &&Acquire,
           GetCacheFT &&GetCache, BuildFT &&Build) {
  bool InsertionTookPlace;
  KernelProgramCache::BuildResult<RetT> *BuildResult;

  {
    auto LockedCache = Acquire(KPCache);
    auto &Cache = GetCache(LockedCache);
    auto Inserted =
        Cache.emplace(std::piecewise_construct, std::forward_as_tuple(CacheKey),
                      std::forward_as_tuple(nullptr, BS_InProgress));

    InsertionTookPlace = Inserted.second;
    BuildResult = &Inserted.first->second;
  }

  // no insertion took place, thus some other thread has already inserted smth
  // in the cache
  if (!InsertionTookPlace) {
    for (;;) {
      RetT *Result = waitUntilBuilt<ExceptionT>(KPCache, BuildResult);

      if (Result)
        return BuildResult;

      // Previous build is failed. There was no SYCL exception though.
      // We might try to build once more.
      int Expected = BS_Failed;
      int Desired = BS_InProgress;

      if (BuildResult->State.compare_exchange_strong(Expected, Desired))
        break; // this thread is the building thread now
    }
  }

  // only the building thread will run this
  try {
    RetT *Desired = Build();

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
      BuildResult->State.store(BS_Done);
    }

    KPCache.notifyAllBuild(*BuildResult);

    return BuildResult;
  } catch (const exception &Ex) {
    BuildResult->Error.Msg = Ex.what();
    BuildResult->Error.Code = Ex.get_cl_code();

    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BS_Failed);
    }

    KPCache.notifyAllBuild(*BuildResult);

    std::rethrow_exception(std::current_exception());
  } catch (...) {
    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BS_Failed);
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

  const backend ContextBackend =
      detail::getSyclObjImpl(C)->getPlugin().getBackend();

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
                        PI_INVALID_VALUE);
  }
  if (RawImg.BinaryEnd == RawImg.BinaryStart) {
    throw runtime_error("Invalid device program image: size is zero",
                        PI_INVALID_VALUE);
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
        PI_INVALID_OPERATION);

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

static void appendCompileOptionsFromImage(std::string &CompileOpts,
                                          const RTDeviceBinaryImage &Img) {
  // Build options are overridden if environment variables are present.
  // Environment variables are not changed during program lifecycle so it
  // is reasonable to use static here to read them only once.
  static const char *CompileOptsEnv =
      SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  pi_device_binary_property isEsimdImage = Img.getProperty("isEsimdImage");
  // Update only if compile options are not overwritten by environment
  // variable
  if (!CompileOptsEnv) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    const char *TemporaryStr = Img.getCompileOptions();
    if (TemporaryStr != nullptr)
      CompileOpts += std::string(TemporaryStr);
  }
  // The -vc-codegen option is always preserved for ESIMD kernels, regardless
  // of the contents SYCL_PROGRAM_COMPILE_OPTIONS environment variable.
  if (isEsimdImage && pi::DeviceBinaryProperty(isEsimdImage).asUint32()) {
    if (!CompileOpts.empty())
      CompileOpts += " ";
    CompileOpts += "-vc-codegen";
  }
}

static void applyOptionsFromImage(std::string &CompileOpts,
                                  std::string &LinkOpts,
                                  const RTDeviceBinaryImage &Img) {
  appendCompileOptionsFromImage(CompileOpts, Img);
  appendLinkOptionsFromImage(LinkOpts, Img);
}

static void applyOptionsFromEnvironment(std::string &CompileOpts,
                                        std::string &LinkOpts) {
  // Build options are overridden if environment variables are present.
  // Environment variables are not changed during program lifecycle so it
  // is reasonable to use static here to read them only once.
  static const char *CompileOptsEnv =
      SYCLConfig<SYCL_PROGRAM_COMPILE_OPTIONS>::get();
  if (CompileOptsEnv) {
    CompileOpts = CompileOptsEnv;
  }
  static const char *LinkOptsEnv = SYCLConfig<SYCL_PROGRAM_LINK_OPTIONS>::get();
  if (LinkOptsEnv) {
    LinkOpts = LinkOptsEnv;
  }
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

  using PiProgramT = KernelProgramCache::PiProgramT;
  using ProgramCacheT = KernelProgramCache::ProgramCacheT;

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireCachedPrograms();
  };
  auto GetF = [](const Locked<ProgramCacheT> &LockedCache) -> ProgramCacheT & {
    return LockedCache.get();
  };

  std::string CompileOpts;
  std::string LinkOpts;
  if (Prg) {
    CompileOpts = Prg->get_build_options();
  }

  applyOptionsFromEnvironment(CompileOpts, LinkOpts);

  SerializedObj SpecConsts;
  if (Prg)
    Prg->stableSerializeSpecConstRegistry(SpecConsts);

  auto BuildF = [this, &M, &KSId, &ContextImpl, &DeviceImpl, Prg, &CompileOpts,
                 &LinkOpts, &JITCompilationIsRequired, SpecConsts] {
    auto Context = createSyclObjFromImpl<context>(ContextImpl);
    auto Device = createSyclObjFromImpl<device>(DeviceImpl);

    const RTDeviceBinaryImage &Img =
        getDeviceImage(M, KSId, Context, Device, JITCompilationIsRequired);

    applyOptionsFromImage(CompileOpts, LinkOpts, Img);

    const detail::plugin &Plugin = ContextImpl->getPlugin();
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreatePIProgram(
        Img, Context, Device, CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache) {
      if (Prg)
        flushSpecConstants(*Prg, NativePrg, &Img);
      if (Img.supportsSpecConstants())
        enableITTAnnotationsIfNeeded(NativePrg, Plugin);
    }

    ProgramPtr ProgramManaged(
        NativePrg, Plugin.getPiPlugin().PiFunctionTable.piProgramRelease);

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
              getRawSyclObjImpl(Device)->getHandleRef(),
              ContextImpl->getCachedLibPrograms(), DeviceLibReqMask);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms[BuiltProgram.get()] = &Img;
    }

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache)
      PersistentDeviceCodeCache::putItemToDisc(
          Device, Img, SpecConsts, CompileOpts + LinkOpts, BuiltProgram.get());
    return BuiltProgram.release();
  };

  const RT::PiDevice PiDevice = DeviceImpl->getHandleRef();

  auto BuildResult = getOrBuild<PiProgramT, compile_program_error>(
      Cache,
      std::make_pair(std::make_pair(std::move(SpecConsts), KSId),
                     std::make_pair(PiDevice, CompileOpts + LinkOpts)),
      AcquireF, GetF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  return BuildResult->Ptr.load();
}

std::tuple<RT::PiKernel, std::mutex *, RT::PiProgram>
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

  using PiKernelT = KernelProgramCache::PiKernelT;
  using KernelCacheT = KernelProgramCache::KernelCacheT;
  using KernelByNameT = KernelProgramCache::KernelByNameT;

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

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireKernelsPerProgramCache();
  };
  auto GetF =
      [&Program](const Locked<KernelCacheT> &LockedCache) -> KernelByNameT & {
    return LockedCache.get()[Program];
  };
  auto BuildF = [&Program, &KernelName, &ContextImpl] {
    PiKernelT *Result = nullptr;

    // TODO need some user-friendly error/exception
    // instead of currently obscure one
    const detail::plugin &Plugin = ContextImpl->getPlugin();
    Plugin.call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                           &Result);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    Plugin.call<PiApiKind::piKernelSetExecInfo>(Result, PI_USM_INDIRECT_ACCESS,
                                                sizeof(pi_bool), &PI_TRUE);

    return Result;
  };

  auto BuildResult = getOrBuild<PiKernelT, invalid_object_error>(
      Cache, KernelName, AcquireF, GetF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  auto ret_val = std::make_tuple(BuildResult->Ptr.load(),
                                 &(BuildResult->MBuildResultMutex), Program);
  Cache.saveKernel(key, ret_val);
  return ret_val;
}

RT::PiProgram
ProgramManager::getPiProgramFromPiKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context) {
  RT::PiProgram Program;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piKernelGetInfo>(
      Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(RT::PiProgram), &Program, nullptr);
  return Program;
}

std::string ProgramManager::getProgramBuildLog(const RT::PiProgram &Program,
                                               const ContextImplPtr Context) {
  size_t PIDevicesSize = 0;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, 0,
                                           nullptr, &PIDevicesSize);
  std::vector<RT::PiDevice> PIDevices(PIDevicesSize / sizeof(RT::PiDevice));
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES,
                                           PIDevicesSize, PIDevices.data(),
                                           nullptr);
  std::string Log = "The program was built for " +
                    std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
    std::string DeviceBuildInfoString;
    size_t DeviceBuildInfoStrSize = 0;
    Plugin.call<PiApiKind::piProgramGetBuildInfo>(
        Program, Device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
        &DeviceBuildInfoStrSize);
    if (DeviceBuildInfoStrSize > 0) {
      std::vector<char> DeviceBuildInfo(DeviceBuildInfoStrSize);
      Plugin.call<PiApiKind::piProgramGetBuildInfo>(
          Program, Device, CL_PROGRAM_BUILD_LOG, DeviceBuildInfoStrSize,
          DeviceBuildInfo.data(), nullptr);
      DeviceBuildInfoString = std::string(DeviceBuildInfo.data());
    }

    std::string DeviceNameString;
    size_t DeviceNameStrSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME, 0,
                                            nullptr, &DeviceNameStrSize);
    if (DeviceNameStrSize > 0) {
      std::vector<char> DeviceName(DeviceNameStrSize);
      Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME,
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

static const char *getDeviceLibFilename(DeviceLibExt Extension) {
  switch (Extension) {
  case DeviceLibExt::cl_intel_devicelib_assert:
    return "libsycl-fallback-cassert.spv";
  case DeviceLibExt::cl_intel_devicelib_math:
    return "libsycl-fallback-cmath.spv";
  case DeviceLibExt::cl_intel_devicelib_math_fp64:
    return "libsycl-fallback-cmath-fp64.spv";
  case DeviceLibExt::cl_intel_devicelib_complex:
    return "libsycl-fallback-complex.spv";
  case DeviceLibExt::cl_intel_devicelib_complex_fp64:
    return "libsycl-fallback-complex-fp64.spv";
  case DeviceLibExt::cl_intel_devicelib_cstring:
    return "libsycl-fallback-cstring.spv";
  }
  throw compile_program_error("Unhandled (new?) device library extension",
                              PI_INVALID_OPERATION);
}

static const char *getDeviceLibExtensionStr(DeviceLibExt Extension) {
  switch (Extension) {
  case DeviceLibExt::cl_intel_devicelib_assert:
    return "cl_intel_devicelib_assert";
  case DeviceLibExt::cl_intel_devicelib_math:
    return "cl_intel_devicelib_math";
  case DeviceLibExt::cl_intel_devicelib_math_fp64:
    return "cl_intel_devicelib_math_fp64";
  case DeviceLibExt::cl_intel_devicelib_complex:
    return "cl_intel_devicelib_complex";
  case DeviceLibExt::cl_intel_devicelib_complex_fp64:
    return "cl_intel_devicelib_complex_fp64";
  case DeviceLibExt::cl_intel_devicelib_cstring:
    return "cl_intel_devicelib_cstring";
  }
  throw compile_program_error("Unhandled (new?) device library extension",
                              PI_INVALID_OPERATION);
}

static RT::PiProgram loadDeviceLibFallback(
    const ContextImplPtr Context, DeviceLibExt Extension,
    const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms) {

  const char *LibFileName = getDeviceLibFilename(Extension);
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
                                PI_INVALID_VALUE);
  }

  const detail::plugin &Plugin = Context->getPlugin();
  // TODO no spec constants are used in the std libraries, support in the future
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
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
                          PI_INVALID_VALUE);
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    std::unique_ptr<char[]> Data(new char[Size]);
    File.seekg(0);
    File.read(Data.get(), Size);
    File.close();
    if (!File.good())
      throw runtime_error(std::string("read from ") + SpvFile +
                              std::string(" failed"),
                          PI_INVALID_VALUE);
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
  std::vector<RTDeviceBinaryImageUPtr> &Imgs = *m_DeviceImages[KSId];
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  pi_uint32 ImgInd = 0;
  RTDeviceBinaryImage *Img = nullptr;

  // TODO: There may be cases with cl::sycl::program class usage in source code
  // that will result in a multi-device context. This case needs to be handled
  // here or at the program_impl class level

  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  std::vector<pi_device_binary> RawImgs(Imgs.size());
  for (unsigned I = 0; I < Imgs.size(); I++)
    RawImgs[I] = const_cast<pi_device_binary>(&Imgs[I]->getRawData());

  Ctx->getPlugin().call<PiApiKind::piextDeviceSelectBinary>(
      getSyclObjImpl(Device)->getHandleRef(), RawImgs.data(),
      (cl_uint)RawImgs.size(), &ImgInd);

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
                                  PI_INVALID_OPERATION);
    }
  }

  Img = Imgs[ImgInd].get();

  if (DbgProgMgr > 0) {
    std::cerr << "selected device image: " << &Img->getRawData() << "\n";
    Img->print();
  }

  if (std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile)
    dumpImage(*Img, KSId);
  return *Img;
}

static bool isDeviceLibRequired(DeviceLibExt Ext, uint32_t DeviceLibReqMask) {
  uint32_t Mask =
      0x1 << (static_cast<uint32_t>(Ext) -
              static_cast<uint32_t>(DeviceLibExt::cl_intel_devicelib_assert));
  return ((DeviceLibReqMask & Mask) == Mask);
}

static std::vector<RT::PiProgram> getDeviceLibPrograms(
    const ContextImplPtr Context, const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms,
    uint32_t DeviceLibReqMask) {
  std::vector<RT::PiProgram> Programs;

  std::pair<DeviceLibExt, bool> RequiredDeviceLibExt[] = {
      {DeviceLibExt::cl_intel_devicelib_assert,
       /* is fallback loaded? */ false},
      {DeviceLibExt::cl_intel_devicelib_math, false},
      {DeviceLibExt::cl_intel_devicelib_math_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_complex, false},
      {DeviceLibExt::cl_intel_devicelib_complex_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_cstring, false}};

  // Disable all devicelib extensions requiring fp64 support if at least
  // one underlying device doesn't support cl_khr_fp64.
  std::string DevExtList =
      get_device_info<std::string, info::device::extensions>::get(
          Device, Context->getPlugin());
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

    const char *ExtStr = getDeviceLibExtensionStr(Ext);

    bool InhibitNativeImpl = false;
    if (const char *Env = getenv("SYCL_DEVICELIB_INHIBIT_NATIVE")) {
      InhibitNativeImpl = strstr(Env, ExtStr) != nullptr;
    }

    bool DeviceSupports = DevExtList.npos != DevExtList.find(ExtStr);

    if (!DeviceSupports || InhibitNativeImpl) {
      Programs.push_back(
          loadDeviceLibFallback(Context, Ext, Device, CachedLibPrograms));
      FallbackIsLoaded = true;
    }
  }
  return Programs;
}

ProgramManager::ProgramPtr ProgramManager::build(
    ProgramPtr Program, const ContextImplPtr Context,
    const std::string &CompileOptions, const std::string &LinkOptions,
    const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms,
    uint32_t DeviceLibReqMask) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", ... " << Device
              << ")\n";
  }

  bool LinkDeviceLibs = (DeviceLibReqMask != 0);

  // TODO: Currently, online linking isn't implemented yet on Level Zero.
  // To enable device libraries and unify the behaviors on all backends,
  // online linking is disabled temporarily, all fallback device libraries
  // will be linked offline. When Level Zero supports online linking, we need
  // to remove the line of code below and switch back to online linking.
  LinkDeviceLibs = false;

  // TODO: this is a temporary workaround for GPU tests for ESIMD compiler.
  // We do not link with other device libraries, because it may fail
  // due to unrecognized SPIR-V format of those libraries.
  if (CompileOptions.find(std::string("-cmc")) != std::string::npos ||
      CompileOptions.find(std::string("-vc-codegen")) != std::string::npos)
    LinkDeviceLibs = false;

  std::vector<RT::PiProgram> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Device, CachedLibPrograms,
                                        DeviceLibReqMask);
  }

  static const char *ForceLinkEnv = std::getenv("SYCL_FORCE_LINK");
  static bool ForceLink = ForceLinkEnv && (*ForceLinkEnv == '1');

  const detail::plugin &Plugin = Context->getPlugin();
  if (LinkPrograms.empty() && !ForceLink) {
    const std::string &Options = LinkOptions.empty()
                                     ? CompileOptions
                                     : (CompileOptions + " " + LinkOptions);
    RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramBuild>(
        Program.get(), /*num devices =*/1, &Device, Options.c_str(), nullptr,
        nullptr);
    if (Error != PI_SUCCESS)
      throw compile_program_error(getProgramBuildLog(Program.get(), Context),
                                  Error);
    return Program;
  }

  // Include the main program and compile/link everything together
  Plugin.call<PiApiKind::piProgramCompile>(Program.get(), /*num devices =*/1,
                                           &Device, CompileOptions.c_str(), 0,
                                           nullptr, nullptr, nullptr, nullptr);
  LinkPrograms.push_back(Program.get());

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramLink>(
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
    Plugin.checkPiResult(Error);
  }
  return Program;
}

static ProgramManager::KernelArgMask
createKernelArgMask(const pi::ByteArray &Bytes) {
  const int NBytesForSize = 8;
  const int NBitsInElement = 8;
  std::uint64_t SizeInBits = 0;
  for (int I = 0; I < NBytesForSize; ++I)
    SizeInBits |= static_cast<std::uint64_t>(Bytes[I]) << I * NBitsInElement;

  ProgramManager::KernelArgMask Result;
  for (std::uint64_t I = 0; I < SizeInBits; ++I) {
    std::uint8_t Byte = Bytes[NBytesForSize + (I / NBitsInElement)];
    Result.push_back(Byte & (1 << (I % NBitsInElement)));
  }

  return Result;
}

void ProgramManager::cacheKernelUsesAssertInfo(OSModuleHandle M,
                                               RTDeviceBinaryImage &Img) {
  const pi::DeviceBinaryImage::PropertyRange &AssertUsedRange =
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

  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(RawImg);
    const _pi_offload_entry EntriesB = RawImg->EntriesBegin;
    const _pi_offload_entry EntriesE = RawImg->EntriesEnd;
    auto Img = make_unique_ptr<RTDeviceBinaryImage>(RawImg, M);

    // Fill the kernel argument mask map
    const pi::DeviceBinaryImage::PropertyRange &KPOIRange =
        Img->getKernelParamOptInfo();
    if (KPOIRange.isAvailable()) {
      KernelNameToArgMaskMap &ArgMaskMap =
          m_EliminatedKernelArgMasks[Img.get()];
      for (const auto &Info : KPOIRange)
        ArgMaskMap[Info->Name] =
            createKernelArgMask(pi::DeviceBinaryProperty(Info).asByteArray());
    }
    // Use the entry information if it's available
    if (EntriesB != EntriesE) {
      // The kernel sets for any pair of images are either disjoint or
      // identical, look up the kernel set using the first kernel name...
      StrToKSIdMap &KSIdMap = m_KernelSets[M];
      auto KSIdIt = KSIdMap.find(EntriesB->name);
      if (KSIdIt != KSIdMap.end()) {
        for (_pi_offload_entry EntriesIt = EntriesB + 1; EntriesIt != EntriesE;
             ++EntriesIt)
          assert(KSIdMap[EntriesIt->name] == KSIdIt->second &&
                 "Kernel sets are not disjoint");
        auto &Imgs = m_DeviceImages[KSIdIt->second];
        assert(Imgs && "Device image vector should have been already created");

        cacheKernelUsesAssertInfo(M, *Img);

        Imgs->push_back(std::move(Img));
        continue;
      }
      // ... or create the set first if it hasn't been
      KernelSetId KSId = getNextKernelSetId();
      {
        std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

        // Register all exported symbols
        auto ExportedSymbols = Img->getExportedSymbols();
        for (const pi_device_binary_property &ExportedSymbol : ExportedSymbols)
          m_ExportedSymbols.insert(ExportedSymbol->Name);

        for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
             ++EntriesIt) {
          auto Result = KSIdMap.insert(std::make_pair(EntriesIt->name, KSId));
          (void)Result;
          assert(Result.second && "Kernel sets are not disjoint");

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
          if (m_ExportedSymbols.find(EntriesIt->name) !=
              m_ExportedSymbols.end())
            continue;

          // ... and create a unique kernel ID for the entry
          std::shared_ptr<detail::kernel_id_impl> KernelIDImpl =
              std::make_shared<detail::kernel_id_impl>(EntriesIt->name);
          sycl::kernel_id KernelID =
              detail::createSyclObjFromImpl<sycl::kernel_id>(KernelIDImpl);
          m_KernelIDs.insert(
              std::make_pair(EntriesIt->name, std::move(KernelID)));
        }
      }
      // ... and initialize associated device_global information
      {
        std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);

        auto DeviceGlobals = Img->getDeviceGlobals();
        for (const pi_device_binary_property &DeviceGlobal : DeviceGlobals) {
          auto Entry = m_DeviceGlobals.find(DeviceGlobal->Name);
          assert(Entry != m_DeviceGlobals.end() &&
                 "Device global has not been registered.");

          pi::ByteArray DeviceGlobalInfo =
              pi::DeviceBinaryProperty(DeviceGlobal).asByteArray();

          // The supplied device_global info property is expected to contain:
          // * 8 bytes - Size of the property.
          // * 4 bytes - Size of the underlying type in the device_global.
          // * 1 byte  - 0 if device_global has device_image_scope and any value
          //             otherwise.
          // Note: Property may be padded.
          assert(DeviceGlobalInfo.size() >= 13 && "Unexpected property size");
          const std::uint32_t TypeSize =
              *reinterpret_cast<const std::uint32_t *>(&DeviceGlobalInfo[8]);
          const std::uint32_t DeviceImageScopeDecorated = DeviceGlobalInfo[12];
          Entry->second.initialize(TypeSize, DeviceImageScopeDecorated);
        }
      }
      m_DeviceImages[KSId].reset(new std::vector<RTDeviceBinaryImageUPtr>());

      cacheKernelUsesAssertInfo(M, *Img);

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
                      PI_INVALID_KERNEL_NAME);
}

void ProgramManager::dumpImage(const RTDeviceBinaryImage &Img,
                               KernelSetId KSId) const {
  std::string Fname("sycl_");
  const pi_device_binary_struct &RawImg = Img.getRawData();
  Fname += RawImg.DeviceTargetSpec;
  Fname += std::to_string(KSId);
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
            PI_INVALID_OPERATION);
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
  const pi::DeviceBinaryImage::PropertyRange &DLMRange =
      Img.getDeviceLibReqMask();
  if (DLMRange.isAvailable())
    return pi::DeviceBinaryProperty(*(DLMRange.begin())).asUint32();
  else
    return 0xFFFFFFFF;
}

// TODO consider another approach with storing the masks in the integration
// header instead.
ProgramManager::KernelArgMask ProgramManager::getEliminatedKernelArgMask(
    OSModuleHandle M, pi::PiProgram NativePrg, const std::string &KernelName) {
  // If instructed to use a spv file, assume no eliminated arguments.
  if (m_UseSpvFile && M == OSUtil::ExeModuleHandle)
    return {};

  // Bail out if there are no eliminated kernel arg masks in our images
  if (m_EliminatedKernelArgMasks.empty())
    return {};

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    auto ImgIt = NativePrograms.find(NativePrg);
    if (ImgIt != NativePrograms.end()) {
      auto MapIt = m_EliminatedKernelArgMasks.find(ImgIt->second);
      if (MapIt != m_EliminatedKernelArgMasks.end())
        return MapIt->second[KernelName];
      return {};
    }
  }

  // If the program was not cached iterate over all available images looking for
  // the requested kernel
  for (auto &Elem : m_EliminatedKernelArgMasks) {
    auto ArgMask = Elem.second.find(KernelName);
    if (ArgMask != Elem.second.end())
      return ArgMask->second;
  }

  // The kernel is not generated by DPCPP stack, so a mask doesn't exist for it
  return {};
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
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piextDeviceSelectBinary>(
      PIDeviceHandle, &DevBin,
      /*num bin images = */ (cl_uint)1, &SuitableImageID);
  if (Error != PI_SUCCESS && Error != PI_INVALID_BINARY)
    throw runtime_error("Invalid binary image or device", PI_INVALID_VALUE);

  return (0 == SuitableImageID);
}

kernel_id ProgramManager::getSYCLKernelID(const std::string &KernelName) {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  auto KernelID = m_KernelIDs.find(KernelName);
  if (KernelID == m_KernelIDs.end())
    throw runtime_error("No kernel found with the specified name",
                        PI_INVALID_KERNEL_NAME);

  return KernelID->second;
}

std::vector<kernel_id> ProgramManager::getAllSYCLKernelIDs() {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  std::vector<sycl::kernel_id> AllKernelIDs;
  AllKernelIDs.reserve(m_KernelIDs.size());
  for (std::pair<std::string, kernel_id> KernelID : m_KernelIDs) {
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

void ProgramManager::addDeviceGlobalEntry(void *DeviceGlobalPtr,
                                          const char *UniqueId) {
  std::lock_guard<std::mutex> DeviceGlobalsGuard(m_DeviceGlobalsMutex);

  assert(m_DeviceGlobals.find(UniqueId) == m_DeviceGlobals.end() &&
         "Device global has already been registered.");
  m_DeviceGlobals.insert({UniqueId, DeviceGlobalMapEntry(DeviceGlobalPtr)});
}

std::vector<device_image_plain>
ProgramManager::getSYCLDeviceImagesWithCompatibleState(
    const context &Ctx, const std::vector<device> &Devs,
    bundle_state TargetState) {

  // Collect raw device images
  std::vector<RTDeviceBinaryImage *> BinImages;
  {
    std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
    for (auto &ImagesSets : m_DeviceImages) {
      auto &ImagesUPtrs = *ImagesSets.second.get();
      for (auto &ImageUPtr : ImagesUPtrs) {
        const RTDeviceBinaryImage *BinImage = ImageUPtr.get();
        const bundle_state ImgState = getBinImageState(BinImage);

        // Ignore images with incompatible state. Image is considered compatible
        // with a target state if an image is already in the target state or can
        // be brought to target state by compiling/linking/building.
        //
        // Example: an image in "executable" state is not compatible with
        // "input" target state - there is no operation to convert the image it
        // to "input" state. An image in "input" state is compatible with
        // "executable" target state because it can be built to get into
        // "executable" state.
        if (ImgState > TargetState)
          continue;

        BinImages.push_back(ImageUPtr.get());
      }
    }
  }
  // TODO: Add a diagnostic on multiple device images with conflicting kernel
  // names, and remove OSModuleHandle usage, as conflicting kernel names will be
  // an error.

  // TODO: Cache device_image objects
  // Create SYCL device image from those that have compatible state and at least
  // one device
  std::vector<device_image_plain> SYCLDeviceImages;
  for (RTDeviceBinaryImage *BinImage : BinImages) {
    const bundle_state ImgState = getBinImageState(BinImage);

    for (const sycl::device &Dev : Devs) {
      if (!compatibleWithDevice(BinImage, Dev))
        continue;

      std::vector<sycl::kernel_id> KernelIDs;
      // Collect kernel names for the image
      pi_device_binary DevBin =
          const_cast<pi_device_binary>(&BinImage->getRawData());
      {
        std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
        for (_pi_offload_entry EntriesIt = DevBin->EntriesBegin;
             EntriesIt != DevBin->EntriesEnd; ++EntriesIt) {
          auto KernelID = m_KernelIDs.find(EntriesIt->name);

          if (KernelID == m_KernelIDs.end()) {
            // Service kernels and exported symbols do not have kernel IDs
            assert((m_ServiceKernels.find(EntriesIt->name) !=
                        m_ServiceKernels.end() ||
                    m_ExportedSymbols.find(EntriesIt->name) !=
                        m_ExportedSymbols.end()) &&
                   "Kernel ID in device binary missing from cache");
            continue;
          }

          KernelIDs.push_back(KernelID->second);
        }
      }

      // If the image does not contain any non-service kernels we can skip it.
      if (KernelIDs.empty())
        continue;

      // device_image_impl expects kernel ids to be sorted for fast search
      std::sort(KernelIDs.begin(), KernelIDs.end(), LessByNameComp{});

      DeviceImageImplPtr Impl = std::make_shared<detail::device_image_impl>(
          BinImage, Ctx, Devs, ImgState, KernelIDs, /*PIProgram=*/nullptr);

      SYCLDeviceImages.push_back(
          createSyclObjFromImpl<device_image_plain>(Impl));
      break;
    }
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
  // Brind device images with compatible state to desired state
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

    for (const kernel_id &ID : KernelIDs) {
      if (m_BuiltInKernelIDs.find(ID.get_name()) != m_BuiltInKernelIDs.end())
        throw sycl::exception(make_error_code(errc::kernel_argument),
                              "Attempting to use a built-in kernel. They are "
                              "not fully supported");
    }
  }

  // Collect device images with compatible state
  std::vector<device_image_plain> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState);

  // Filter out images that have no kernel_ids specified
  auto It = std::remove_if(DeviceImages.begin(), DeviceImages.end(),
                           [&KernelIDs](const device_image_plain &Image) {
                             return std::none_of(
                                 KernelIDs.begin(), KernelIDs.end(),
                                 [&Image](const sycl::kernel_id &KernelID) {
                                   return Image.has_kernel(KernelID);
                                 });
                           });

  DeviceImages.erase(It, DeviceImages.end());

  // Brind device images with compatible state to desired state
  bringSYCLDeviceImagesToState(DeviceImages, TargetState);
  return DeviceImages;
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

  const detail::plugin &Plugin =
      getSyclObjImpl(InputImpl->get_context())->getPlugin();

  // TODO: Add support for creating non-SPIRV programs from multiple devices.
  if (InputImpl->get_bin_image_ref()->getFormat() !=
          PI_DEVICE_BINARY_TYPE_SPIRV &&
      Devs.size() > 1)
    sycl::runtime_error(
        "Creating a program from AOT binary for multiple device is not "
        "supported",
        PI_INVALID_OPERATION);

  // Device is not used when creating program from SPIRV, so passing only one
  // device is OK.
  RT::PiProgram Prog = createPIProgram(*InputImpl->get_bin_image_ref(),
                                       InputImpl->get_context(), Devs[0]);

  if (InputImpl->get_bin_image_ref()->supportsSpecConstants())
    enableITTAnnotationsIfNeeded(Prog, Plugin);

  DeviceImageImplPtr ObjectImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), InputImpl->get_context(), Devs,
      bundle_state::object, InputImpl->get_kernel_ids_ref(), Prog,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  std::vector<pi_device> PIDevices;
  PIDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    PIDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

  // TODO: Set spec constatns here.

  // TODO: Handle zero sized Device list.
  std::string CompileOptions;
  appendCompileOptionsFromImage(CompileOptions,
                                *(InputImpl->get_bin_image_ref()));
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
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
ProgramManager::link(const std::vector<device_image_plain> &DeviceImages,
                     const std::vector<device> &Devs,
                     const property_list &PropList) {
  (void)PropList;

  std::vector<pi_program> PIPrograms;
  PIPrograms.reserve(DeviceImages.size());
  for (const device_image_plain &DeviceImage : DeviceImages)
    PIPrograms.push_back(getSyclObjImpl(DeviceImage)->get_program_ref());

  std::vector<pi_device> PIDevices;
  PIDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    PIDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

  std::string LinkOptionsStr;
  for (const device_image_plain &DeviceImage : DeviceImages) {
    const std::shared_ptr<device_image_impl> &InputImpl =
        getSyclObjImpl(DeviceImage);
    appendLinkOptionsFromImage(LinkOptionsStr,
                               *(InputImpl->get_bin_image_ref()));
  }
  const context &Context = getSyclObjImpl(DeviceImages[0])->get_context();
  const ContextImplPtr ContextImpl = getSyclObjImpl(Context);
  const detail::plugin &Plugin = ContextImpl->getPlugin();

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramLink>(
      ContextImpl->getHandleRef(), PIDevices.size(), PIDevices.data(),
      /*options=*/LinkOptionsStr.c_str(), PIPrograms.size(), PIPrograms.data(),
      /*pfn_notify=*/nullptr,
      /*user_data=*/nullptr, &LinkedProg);

  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      const std::string ErrorMsg = getProgramBuildLog(LinkedProg, ContextImpl);
      throw sycl::exception(make_error_code(errc::build), ErrorMsg);
    }
    Plugin.reportPiError(Error, "link()");
  }

  std::vector<kernel_id> KernelIDs;
  for (const device_image_plain &DeviceImage : DeviceImages) {
    // Duplicates are not expected here, otherwise piProgramLink should fail
    KernelIDs.insert(KernelIDs.end(),
                     getSyclObjImpl(DeviceImage)->get_kernel_ids().begin(),
                     getSyclObjImpl(DeviceImage)->get_kernel_ids().end());
  }
  // device_image_impl expects kernel ids to be sorted for fast search
  std::sort(KernelIDs.begin(), KernelIDs.end(), LessByNameComp{});

  DeviceImageImplPtr ExecutableImpl =
      std::make_shared<detail::device_image_impl>(
          /*BinImage=*/nullptr, Context, Devs, bundle_state::executable,
          std::move(KernelIDs), LinkedProg);

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

  using PiProgramT = KernelProgramCache::PiProgramT;
  using ProgramCacheT = KernelProgramCache::ProgramCacheT;

  KernelProgramCache &Cache = ContextImpl->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireCachedPrograms();
  };
  auto GetF = [](const Locked<ProgramCacheT> &LockedCache) -> ProgramCacheT & {
    return LockedCache.get();
  };

  std::string CompileOpts;
  std::string LinkOpts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);

  const RTDeviceBinaryImage *ImgPtr = InputImpl->get_bin_image_ref();
  const RTDeviceBinaryImage &Img = *ImgPtr;

  SerializedObj SpecConsts = InputImpl->get_spec_const_blob_ref();

  // TODO: Unify this code with getBuiltPIProgram
  auto BuildF = [this, &Context, &Img, &Devs, &CompileOpts, &LinkOpts,
                 &InputImpl, SpecConsts] {
    applyOptionsFromImage(CompileOpts, LinkOpts, Img);
    ContextImplPtr ContextImpl = getSyclObjImpl(Context);
    const detail::plugin &Plugin = ContextImpl->getPlugin();

    // TODO: Add support for creating non-SPIRV programs from multiple devices.
    if (InputImpl->get_bin_image_ref()->getFormat() !=
            PI_DEVICE_BINARY_TYPE_SPIRV &&
        Devs.size() > 1)
      sycl::runtime_error(
          "Creating a program from AOT binary for multiple device is not "
          "supported",
          PI_INVALID_OPERATION);

    // Device is not used when creating program from SPIRV, so passing only one
    // device is OK.
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreatePIProgram(
        Img, Context, Devs[0], CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache) {
      if (InputImpl->get_bin_image_ref()->supportsSpecConstants())
        enableITTAnnotationsIfNeeded(NativePrg, Plugin);

      {
        std::lock_guard<std::mutex> Lock{InputImpl->get_spec_const_data_lock()};
        const std::map<std::string,
                       std::vector<device_image_impl::SpecConstDescT>>
            &SpecConstData = InputImpl->get_spec_const_data_ref();

        for (const auto &DescPair : SpecConstData) {
          for (const device_image_impl::SpecConstDescT &SpecIDDesc :
               DescPair.second) {
            if (SpecIDDesc.IsSet) {
              Plugin.call<PiApiKind::piextProgramSetSpecializationConstant>(
                  NativePrg, SpecIDDesc.ID, SpecIDDesc.Size,
                  SpecConsts.data() + SpecIDDesc.BlobOffset);
            }
          }
        }
      }
    }

    ProgramPtr ProgramManaged(
        NativePrg, Plugin.getPiPlugin().PiFunctionTable.piProgramRelease);

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
              getRawSyclObjImpl(Devs[0])->getHandleRef(),
              ContextImpl->getCachedLibPrograms(), DeviceLibReqMask);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms[BuiltProgram.get()] = &Img;
    }

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache)
      PersistentDeviceCodeCache::putItemToDisc(
          Devs[0], Img, SpecConsts, CompileOpts + LinkOpts, BuiltProgram.get());

    return BuiltProgram.release();
  };

  const RT::PiDevice PiDevice = getRawSyclObjImpl(Devs[0])->getHandleRef();
  // TODO: Throw SYCL2020 style exception
  auto BuildResult = getOrBuild<PiProgramT, compile_program_error>(
      Cache,
      std::make_pair(std::make_pair(std::move(SpecConsts), (size_t)ImgPtr),
                     std::make_pair(PiDevice, CompileOpts + LinkOpts)),
      AcquireF, GetF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  RT::PiProgram ResProgram = BuildResult->Ptr.load();

  // Cache supports key with once device only, but here we have multiple
  // devices a program is built for, so add the program to the cache for all
  // other devices.
  const detail::plugin &Plugin = ContextImpl->getPlugin();
  auto CacheOtherDevices = [ResProgram, &Plugin]() {
    Plugin.call<PiApiKind::piProgramRetain>(ResProgram);
    return ResProgram;
  };

  // The program for device "0" is already added to the cache during the first
  // call to getOrBuild, so starting with "1"
  for (size_t Idx = 1; Idx < Devs.size(); ++Idx) {
    const RT::PiDevice PiDeviceAdd =
        getRawSyclObjImpl(Devs[Idx])->getHandleRef();

    getOrBuild<PiProgramT, compile_program_error>(
        Cache,
        std::make_pair(std::make_pair(std::move(SpecConsts), (size_t)ImgPtr),
                       std::make_pair(PiDeviceAdd, CompileOpts + LinkOpts)),
        AcquireF, GetF, CacheOtherDevices);
    // getOrBuild is not supposed to return nullptr
    assert(BuildResult != nullptr && "Invalid build result");
  }

  // devive_image_impl shares ownership of PIProgram with, at least, program
  // cache. The ref counter will be descremented in the destructor of
  // device_image_impl
  Plugin.call<PiApiKind::piProgramRetain>(ResProgram);

  DeviceImageImplPtr ExecImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), Context, Devs, bundle_state::executable,
      InputImpl->get_kernel_ids_ref(), ResProgram,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  return createSyclObjFromImpl<device_image_plain>(ExecImpl);
}

std::pair<RT::PiKernel, std::mutex *> ProgramManager::getOrCreateKernel(
    const context &Context, const std::string &KernelName,
    const property_list &PropList, RT::PiProgram Program) {

  (void)PropList;

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  using PiKernelT = KernelProgramCache::PiKernelT;
  using KernelCacheT = KernelProgramCache::KernelCacheT;
  using KernelByNameT = KernelProgramCache::KernelByNameT;

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireKernelsPerProgramCache();
  };
  auto GetF =
      [&Program](const Locked<KernelCacheT> &LockedCache) -> KernelByNameT & {
    return LockedCache.get()[Program];
  };
  auto BuildF = [&Program, &KernelName, &Ctx] {
    PiKernelT *Result = nullptr;

    const detail::plugin &Plugin = Ctx->getPlugin();
    Plugin.call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                           &Result);

    Plugin.call<PiApiKind::piKernelSetExecInfo>(Result, PI_USM_INDIRECT_ACCESS,
                                                sizeof(pi_bool), &PI_TRUE);

    return Result;
  };

  auto BuildResult = getOrBuild<PiKernelT, invalid_object_error>(
      Cache, KernelName, AcquireF, GetF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  return std::make_pair(BuildResult->Ptr.load(),
                        &(BuildResult->MBuildResultMutex));
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

extern "C" void __sycl_register_lib(pi_device_binaries desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(pi_device_binaries desc) {
  (void)desc;
  // TODO implement the function
}
