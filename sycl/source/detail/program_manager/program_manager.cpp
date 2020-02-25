//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;

static constexpr int DbgProgMgr = 0;

enum BuildState { BS_InProgress, BS_Done, BS_Failed };

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

ProgramManager &ProgramManager::getInstance() {
  // The singleton ProgramManager instance, uses the "magic static" idiom.
  static ProgramManager Instance;
  return Instance;
}

static RT::PiDevice getFirstDevice(const ContextImplPtr &Context) {
  pi_uint32 NumDevices = 0;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piContextGetInfo>(Context->getHandleRef(),
                                           PI_CONTEXT_INFO_NUM_DEVICES,
                                           sizeof(NumDevices), &NumDevices,
                                           /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 && "Context without devices?");

  vector_class<RT::PiDevice> Devices(NumDevices);
  size_t ParamValueSize = 0;
  Plugin.call<PiApiKind::piContextGetInfo>(
      Context->getHandleRef(), PI_CONTEXT_INFO_DEVICES,
      sizeof(cl_device_id) * NumDevices, &Devices[0], &ParamValueSize);
  assert(ParamValueSize == sizeof(cl_device_id) * NumDevices &&
         "Number of CL_CONTEXT_DEVICES should match CL_CONTEXT_NUM_DEVICES.");
  return Devices[0];
}

static RT::PiProgram createBinaryProgram(const ContextImplPtr Context,
                                         const unsigned char *Data,
                                         size_t DataLen) {
  // FIXME: we don't yet support multiple devices with a single binary.
  const detail::plugin &Plugin = Context->getPlugin();
#ifndef _NDEBUG
  cl_uint NumDevices = 0;
  Plugin.call<PiApiKind::piContextGetInfo>(Context->getHandleRef(),
                                           PI_CONTEXT_INFO_NUM_DEVICES,
                                           sizeof(NumDevices), &NumDevices,
                                           /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  RT::PiProgram Program;

  bool IsCUDA = false;

  // TODO: Implement `piProgramCreateWithBinary` to not require extra logic for
  //       the CUDA backend.
#if USE_PI_CUDA
  // All devices in a context are from the same platform.
  RT::PiDevice Device = getFirstDevice(Context);
  RT::PiPlatform Platform = nullptr;
  Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_PLATFORM, sizeof(Platform),
                           &Platform, nullptr);
  size_t PlatformNameSize = 0u;
  Plugin.call<PiApiKind::piPlatformGetInfo>(Platform, PI_PLATFORM_INFO_NAME, 0u, nullptr,
                             &PlatformNameSize);
  std::vector<char> PlatformName(PlatformNameSize, '\0');
  Plugin.call<PiApiKind::piPlatformGetInfo>(Platform, PI_PLATFORM_INFO_NAME,
                             PlatformName.size(), PlatformName.data(), nullptr);
  if (PlatformNameSize > 0u &&
      std::strncmp(PlatformName.data(), "NVIDIA CUDA", PlatformNameSize) == 0) {
    IsCUDA = true;
  }
#endif // USE_PI_CUDA

  if (IsCUDA) {
    // TODO: Reemplace CreateWithSource with CreateWithBinary in CUDA backend
    const char *SignedData = reinterpret_cast<const char *>(Data);
    Plugin.call<PiApiKind::piclProgramCreateWithSource>(Context->getHandleRef(), 1 /*one binary*/, &SignedData,
                                         &DataLen, &Program);
  } else {
    RT::PiDevice Device = getFirstDevice(Context);
    pi_int32 BinaryStatus = CL_SUCCESS;
    Plugin.call<PiApiKind::piclProgramCreateWithBinary>(Context->getHandleRef(), 1 /*one binary*/, &Device,
                                         &DataLen, &Data, &BinaryStatus,
                                         &Program);
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

DeviceImage &ProgramManager::getDeviceImage(OSModuleHandle M,
                                            const string_class &KernelName,
                                            const context &Context) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \""
              << KernelName << "\", " << getRawSyclObjImpl(Context) << ")\n";

  KernelSetId KSId = getKernelSetId(M, KernelName);
  return getDeviceImage(M, KSId, Context);
}

template <typename ExceptionT, typename RetT>
RetT *waitUntilBuilt(KernelProgramCache &Cache,
                     KernelProgramCache::BuildResult<RetT> *BuildResult) {
  // any thread which will find nullptr in cache will wait until the pointer
  // is not null anymore
  Cache.waitUntilBuilt([BuildResult]() {
    int State = BuildResult->State.load();

    return State == BS_Done || State == BS_Failed;
  });

  if (BuildResult->Error.FilledIn) {
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
template <typename RetT, typename ExceptionT, typename KeyT, typename AcquireFT,
          typename GetCacheFT, typename BuildFT>
RetT *getOrBuild(KernelProgramCache &KPCache, const KeyT &CacheKey,
                 AcquireFT &&Acquire, GetCacheFT &&GetCache, BuildFT &&Build) {
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
          return Result;

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

    BuildResult->State.store(BS_Done);

    KPCache.notifyAllBuild();

    return Desired;
  } catch (const exception &Ex) {
    BuildResult->Error.Msg = Ex.what();
    BuildResult->Error.Code = Ex.get_cl_code();
    BuildResult->Error.FilledIn = true;

    BuildResult->State.store(BS_Failed);

    KPCache.notifyAllBuild();

    std::rethrow_exception(std::current_exception());
  } catch (...) {
    BuildResult->State.store(BS_Failed);

    KPCache.notifyAllBuild();

    std::rethrow_exception(std::current_exception());
  }
}

static bool isDeviceBinaryTypeSupported(const context &C,
                                        RT::PiDeviceBinaryType Format) {
  // All formats except PI_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  vector_class<device> Devices = C.get_devices();

  // Program type is SPIR-V, so we need a device compiler to do JIT.
  for (const device &D : Devices) {
    if (!D.get_info<info::device::is_compiler_available>())
      return false;
  }

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (pi::useBackend(pi::SYCL_BE_PI_OPENCL) &&
      C.get_platform().get_info<info::platform::version>() >= "2.1")
    return true;

  for (const device &D : Devices) {
    // We need cl_khr_il_program extension to be present
    // and we can call clCreateProgramWithILKHR using the extension
    vector_class<string_class> Extensions =
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

RT::PiProgram ProgramManager::createPIProgram(const DeviceImage &Img,
                                              const context &Context) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::createPIProgram(" << &Img << ")\n";

  // perform minimal sanity checks on the device image and the descriptor
  if (Img.BinaryEnd < Img.BinaryStart) {
    throw runtime_error("Malformed device program image descriptor");
  }
  if (Img.BinaryEnd == Img.BinaryStart) {
    throw runtime_error("Invalid device program image: size is zero");
  }
  size_t ImgSize = static_cast<size_t>(Img.BinaryEnd - Img.BinaryStart);

  // TODO if the binary image is a part of the fat binary, the clang
  //   driver should have set proper format option to the
  //   clang-offload-wrapper. The fix depends on AOT compilation
  //   implementation, so will be implemented together with it.
  //   Img->Format can't be updated as it is inside of the in-memory
  //   OS module binary.
  RT::PiDeviceBinaryType Format = getFormat(Img);
  // RT::PiDeviceBinaryType Format = Img->Format;
  // assert(Format != PI_DEVICE_BINARY_TYPE_NONE && "Image format not set");

  if (!isDeviceBinaryTypeSupported(Context, Format))
    throw feature_not_supported(
        "Online compilation is not supported in this context");

  // Load the image
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  RT::PiProgram Res = Format == PI_DEVICE_BINARY_TYPE_SPIRV
                          ? createSpirvProgram(Ctx, Img.BinaryStart, ImgSize)
                          : createBinaryProgram(Ctx, Img.BinaryStart, ImgSize);

  if (DbgProgMgr > 1)
    std::cerr << "created program: " << Res
              << "; image format: " << getFormatStr(Format) << "\n";

  return Res;
}

RT::PiProgram
ProgramManager::getBuiltPIProgram(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName) {
  KernelSetId KSId = getKernelSetId(M, KernelName);

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  using PiProgramT = KernelProgramCache::PiProgramT;
  using ProgramCacheT = KernelProgramCache::ProgramCacheT;

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireCachedPrograms();
  };
  auto GetF = [](const Locked<ProgramCacheT> &LockedCache) -> ProgramCacheT& {
    return LockedCache.get();
  };
  auto BuildF = [this, &M, &KSId, &Context] {
    const DeviceImage &Img = getDeviceImage(M, KSId, Context);

    ContextImplPtr ContextImpl = getSyclObjImpl(Context);
    const detail::plugin &Plugin = ContextImpl->getPlugin();
    RT::PiProgram Prg = createPIProgram(Img, Context);
    ProgramPtr ProgramManaged(Prg,
                              Plugin.MPlugin.PiFunctionTable.piProgramRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs are supposed to be already linked.
    const bool LinkDeviceLibs = getFormat(Img) == PI_DEVICE_BINARY_TYPE_SPIRV;

    const std::vector<device> &Devices = ContextImpl->getDevices();
    std::vector<RT::PiDevice> PiDevices(Devices.size());
    std::transform(
        Devices.begin(), Devices.end(), PiDevices.begin(),
        [](const device Dev) { return getRawSyclObjImpl(Dev)->getHandleRef(); });

    ProgramPtr BuiltProgram =
        build(std::move(ProgramManaged), ContextImpl, Img.CompileOptions,
              Img.LinkOptions, PiDevices, ContextImpl->getCachedLibPrograms(),
              LinkDeviceLibs);

    return BuiltProgram.release();
  };

  return getOrBuild<PiProgramT, compile_program_error>(Cache, KSId, AcquireF,
                                                       GetF, BuildF);
}

RT::PiKernel ProgramManager::getOrCreateKernel(OSModuleHandle M,
                                               const context &Context,
                                               const string_class &KernelName) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << getRawSyclObjImpl(Context) << ", " << KernelName << ")\n";
  }

  RT::PiProgram Program = getBuiltPIProgram(M, Context, KernelName);
  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  using PiKernelT = KernelProgramCache::PiKernelT;
  using KernelCacheT = KernelProgramCache::KernelCacheT;
  using KernelByNameT = KernelProgramCache::KernelByNameT;

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto AcquireF = [] (KernelProgramCache &Cache) {
    return Cache.acquireKernelsPerProgramCache();
  };
  auto GetF = [&Program] (const Locked<KernelCacheT> &LockedCache) -> KernelByNameT& {
    return LockedCache.get()[Program];
  };
  auto BuildF = [this, &Program, &KernelName, &Ctx] {
    PiKernelT *Result = nullptr;

    // TODO need some user-friendly error/exception
    // instead of currently obscure one
    const detail::plugin &Plugin = Ctx->getPlugin();
    Plugin.call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                           &Result);

    return Result;
  };

  return getOrBuild<PiKernelT, invalid_object_error>(
        Cache, KernelName, AcquireF, GetF, BuildF);
}

RT::PiProgram
ProgramManager::getClProgramFromClKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context) {
  RT::PiProgram Program;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piKernelGetInfo>(
      Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(cl_program), &Program, nullptr);
  return Program;
}

string_class ProgramManager::getProgramBuildLog(const RT::PiProgram &Program,
                                                const ContextImplPtr Context) {
  size_t Size = 0;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, 0,
                                           nullptr, &Size);
  vector_class<RT::PiDevice> PIDevices(Size / sizeof(RT::PiDevice));
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, Size,
                                           PIDevices.data(), nullptr);
  string_class Log = "The program was built for " +
                     std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
    Plugin.call<PiApiKind::piProgramGetBuildInfo>(
        Program, Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &Size);
    vector_class<char> DeviceBuildInfo(Size);
    Plugin.call<PiApiKind::piProgramGetBuildInfo>(
        Program, Device, CL_PROGRAM_BUILD_LOG, Size, DeviceBuildInfo.data(),
        nullptr);
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME, 0,
                                            nullptr, &Size);
    vector_class<char> DeviceName(Size);
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME, Size,
                                            DeviceName.data(), nullptr);

    Log += "\nBuild program log for '" + string_class(DeviceName.data()) +
           "':\n" + string_class(DeviceBuildInfo.data());
  }
  return Log;
}

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

static const char* getDeviceLibFilename(DeviceLibExt Extension) {
  switch (Extension) {
  case cl_intel_devicelib_assert:
    return "libsycl-fallback-cassert.spv";
  case cl_intel_devicelib_math:
    return "libsycl-fallback-cmath.spv";
  case cl_intel_devicelib_math_fp64:
    return "libsycl-fallback-cmath-fp64.spv";
  case cl_intel_devicelib_complex:
    return "libsycl-fallback-complex.spv";
  case cl_intel_devicelib_complex_fp64:
    return "libsycl-fallback-complex-fp64.spv";
  }
  throw compile_program_error("Unhandled (new?) device library extension");
}

static const char* getDeviceLibExtensionStr(DeviceLibExt Extension) {
  switch (Extension) {
  case cl_intel_devicelib_assert:
    return "cl_intel_devicelib_assert";
  case cl_intel_devicelib_math:
    return "cl_intel_devicelib_math";
  case cl_intel_devicelib_math_fp64:
    return "cl_intel_devicelib_math_fp64";
  case cl_intel_devicelib_complex:
    return "cl_intel_devicelib_complex";
  case cl_intel_devicelib_complex_fp64:
    return "cl_intel_devicelib_complex_fp64";
  }
  throw compile_program_error("Unhandled (new?) device library extension");
}

static RT::PiProgram loadDeviceLibFallback(
    const ContextImplPtr Context, DeviceLibExt Extension,
    const std::vector<RT::PiDevice> &Devices,
    std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms) {

  const char *LibFileName = getDeviceLibFilename(Extension);
  std::map<DeviceLibExt, RT::PiProgram>::iterator LibProgIt;
  bool NotExists = false;
  std::tie(LibProgIt, NotExists) =
      CachedLibPrograms.insert({Extension, nullptr});
  RT::PiProgram &LibProg = LibProgIt->second;

  if (!NotExists) {
    return LibProg;
  }

  if (!loadDeviceLib(Context, LibFileName, LibProg)) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(std::string("Failed to load ") + LibFileName);
  }

  const detail::plugin &Plugin = Context->getPlugin();
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
      LibProg,
      // Assume that Devices contains all devices from Context.
      Devices.size(), Devices.data(),
      // Do not use compile options for library programs: it is not clear
      // if user options (image options) are supposed to be applied to
      // library program as well, and what actually happens to a SPIR-V
      // program if we apply them.
      "", 0, nullptr, nullptr, nullptr, nullptr);
  if (Error != PI_SUCCESS) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(
        ProgramManager::getProgramBuildLog(LibProg, Context));
  }

  return LibProg;
}

struct ImageDeleter {
  void operator()(DeviceImage *I) {
    delete[] I->BinaryStart;
    delete I;
  }
};

ProgramManager::ProgramManager() {
  const char *SpvFile = std::getenv(UseSpvEnv);
  // If a SPIRV file is specified with an environment variable,
  // register the corresponding image
  if (SpvFile) {
    m_UseSpvFile = true;
    // The env var requests that the program is loaded from a SPIRV file on disk
    std::ifstream File(SpvFile, std::ios::binary);

    if (!File.is_open())
      throw runtime_error(std::string("Can't open file specified via ") +
                          UseSpvEnv + ": " + SpvFile);
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    std::unique_ptr<unsigned char[]> Data(new unsigned char[Size]);
    File.seekg(0);
    File.read(reinterpret_cast<char *>(Data.get()), Size);
    File.close();
    if (!File.good())
      throw runtime_error(std::string("read from ") + SpvFile +
                          std::string(" failed"));

    std::unique_ptr<DeviceImage, ImageDeleter> ImgPtr(new DeviceImage(),
                                                      ImageDeleter());
    ImgPtr->Version = PI_DEVICE_BINARY_VERSION;
    ImgPtr->Kind = PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL;
    ImgPtr->DeviceTargetSpec = PI_DEVICE_BINARY_TARGET_UNKNOWN;
    ImgPtr->CompileOptions = "";
    ImgPtr->LinkOptions = "";
    ImgPtr->ManifestStart = nullptr;
    ImgPtr->ManifestEnd = nullptr;
    ImgPtr->BinaryStart = Data.release();
    ImgPtr->BinaryEnd = ImgPtr->BinaryStart + Size;
    ImgPtr->EntriesBegin = nullptr;
    ImgPtr->EntriesEnd = nullptr;
    // TODO the environment variable name implies that the only binary format
    // it accepts is SPIRV but that is not the case, should be aligned?
    ImgPtr->Format = getFormat(*ImgPtr);

    // No need for a mutex here since all access to these private fields is
    // blocked until the construction of the ProgramManager singleton is
    // finished.
    m_DeviceImages[SpvFileKSId].reset(
        new std::vector<DeviceImage *>({ImgPtr.get()}));

    m_OrphanDeviceImages.emplace_back(std::move(ImgPtr));

    if (DbgProgMgr > 0)
      std::cerr << "loaded device image from " << SpvFile << "\n";
  }
}

DeviceImage &ProgramManager::getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                                            const context &Context) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \"" << KSId
              << "\", " << getRawSyclObjImpl(Context) << ")\n";
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  std::vector<DeviceImage *> &Imgs = *m_DeviceImages[KSId];
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  DeviceImage *Img = nullptr;

  // TODO: There may be cases with cl::sycl::program class usage in source code
  // that will result in a multi-device context. This case needs to be handled
  // here or at the program_impl class level

  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  if (Imgs.size() > 1) {
    Ctx->getPlugin().call<PiApiKind::piextDeviceSelectBinary>(
        getFirstDevice(Ctx), Imgs.data(), (cl_uint)Imgs.size(), &Img);
  } else
    Img = Imgs[0];

  if (DbgProgMgr > 0) {
    std::cerr << "available device images:\n";
    debugDumpBinaryImages();
    std::cerr << "selected device image: " << Img << "\n";
    debugDumpBinaryImage(Img);
  }

  if (std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile)
    dumpImage(*Img, KSId);
  return *Img;
}

// TODO: getDeviceLibPrograms should also support Windows but
// current implementation doesn't work on Windows when multiple
// device libraries exist and this problem should be fixed.
static std::vector<RT::PiProgram>
getDeviceLibPrograms(const ContextImplPtr Context,
                     const std::vector<RT::PiDevice> &Devices,
                     std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms) {
  std::vector<RT::PiProgram> Programs;

  // TODO: SYCL compiler should generate a list of required extensions for a
  // particular program in order to allow us do a more fine-grained check here.
  // Require *all* possible devicelib extensions for now.
  std::pair<DeviceLibExt, bool> RequiredDeviceLibExt[] = {
      {cl_intel_devicelib_assert, /* is fallback loaded? */ false},
      {cl_intel_devicelib_math, false},
      {cl_intel_devicelib_math_fp64, false},
      {cl_intel_devicelib_complex, false},
      {cl_intel_devicelib_complex_fp64, false}
  };

  // Disable all devicelib extensions requiring fp64 support if at least
  // one underlying device doesn't support cl_khr_fp64.
  bool fp64Support = true;
  for (RT::PiDevice Dev : Devices) {
    std::string DevExtList =
	get_device_info<std::string, info::device::extensions>::get(
            Dev, Context->getPlugin());
    fp64Support = fp64Support &&
	          (DevExtList.npos != DevExtList.find("cl_khr_fp64"));
  }

  // Load a fallback library for an extension if at least one device does not
  // support it.
  for (RT::PiDevice Dev : Devices) {
    std::string DevExtList =
        get_device_info<std::string, info::device::extensions>::get(
            Dev, Context->getPlugin());
    for (auto &Pair : RequiredDeviceLibExt) {
      DeviceLibExt Ext = Pair.first;
      bool &FallbackIsLoaded = Pair.second;

      if (FallbackIsLoaded) {
        continue;
      }

      if ((Ext == cl_intel_devicelib_math_fp64 ||
	  Ext == cl_intel_devicelib_complex_fp64) && !fp64Support) {
        continue;
      }

      const char* ExtStr = getDeviceLibExtensionStr(Ext);

      bool InhibitNativeImpl = false;
      if (const char *Env = getenv("SYCL_DEVICELIB_INHIBIT_NATIVE")) {
        InhibitNativeImpl = strstr(Env, ExtStr) != nullptr;
      }

      bool DeviceSupports = DevExtList.npos != DevExtList.find(ExtStr);

      if (!DeviceSupports || InhibitNativeImpl) {
        Programs.push_back(
            loadDeviceLibFallback(Context, Ext, Devices, CachedLibPrograms));
        FallbackIsLoaded = true;
      }
    }
  }
  return Programs;
}

ProgramManager::ProgramPtr
ProgramManager::build(ProgramPtr Program, const ContextImplPtr Context,
                      const string_class &CompileOptions,
                      const string_class &LinkOptions,
                      const std::vector<RT::PiDevice> &Devices,
                      std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms,
                      bool LinkDeviceLibs) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", ... "
              << Devices.size() << ")\n";
  }

  const char *CompileOpts = std::getenv("SYCL_PROGRAM_COMPILE_OPTIONS");
  if (!CompileOpts) {
    CompileOpts = CompileOptions.c_str();
  }
  const char *LinkOpts = std::getenv("SYCL_PROGRAM_LINK_OPTIONS");
  if (!LinkOpts) {
    LinkOpts = LinkOptions.c_str();
  }

  std::vector<RT::PiProgram> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Devices, CachedLibPrograms);
  }

  const detail::plugin &Plugin = Context->getPlugin();
  if (LinkPrograms.empty()) {
    std::string Opts(CompileOpts);
    Opts += " ";
    Opts += LinkOpts;

    RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramBuild>(
        Program.get(), Devices.size(), Devices.data(), Opts.c_str(), nullptr,
        nullptr);
    if (Error != PI_SUCCESS)
      throw compile_program_error(getProgramBuildLog(Program.get(), Context));
    return Program;
  }

  // Include the main program and compile/link everything together
  Plugin.call<PiApiKind::piProgramCompile>(Program.get(), Devices.size(),
                                           Devices.data(), CompileOpts, 0,
                                           nullptr, nullptr, nullptr, nullptr);
  LinkPrograms.push_back(Program.get());

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramLink>(
      Context->getHandleRef(), Devices.size(), Devices.data(), LinkOpts,
      LinkPrograms.size(), LinkPrograms.data(), nullptr, nullptr, &LinkedProg);

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise. Release the original (user) program.
  Program.reset(LinkedProg);
  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg, and throw.
      throw compile_program_error(getProgramBuildLog(LinkedProg, Context));
    }
    Plugin.checkPiResult(Error);
  }
  return Program;
}

void ProgramManager::addImages(pi_device_binaries DeviceBinary) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary Img = &(DeviceBinary->DeviceBinaries[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(Img);
    const _pi_offload_entry EntriesB = Img->EntriesBegin;
    const _pi_offload_entry EntriesE = Img->EntriesEnd;
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
        Imgs->push_back(Img);
        continue;
      }
      // ... or create the set first if it hasn't been
      KernelSetId KSId = getNextKernelSetId();
      for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
           ++EntriesIt) {
        auto Result =
            KSIdMap.insert(std::make_pair(EntriesIt->name, KSId));
        (void)Result;
        assert(Result.second && "Kernel sets are not disjoint");
      }
      m_DeviceImages[KSId].reset(new std::vector<DeviceImage *>({Img}));
      continue;
    }
    // Otherwise assume that the image contains all kernels associated with the
    // module
    KernelSetId &KSId = m_OSModuleKernelSets[M];
    if (KSId == 0)
      KSId = getNextKernelSetId();

    auto &Imgs = m_DeviceImages[KSId];
    if (!Imgs)
      Imgs.reset(new std::vector<DeviceImage *>({Img}));
    else
      Imgs->push_back(Img);
  }
}

void ProgramManager::debugDumpBinaryImage(const DeviceImage *Img) const {
  std::cerr << "  --- Image " << Img << "\n";
  if (!Img)
    return;
  std::cerr << "    Version         : " << (int)Img->Version << "\n";
  std::cerr << "    Kind            : " << (int)Img->Kind << "\n";
  std::cerr << "    Format          : " << (int)Img->Format << "\n";
  std::cerr << "    Target          : " << Img->DeviceTargetSpec << "\n";
  std::cerr << "    Compile options : "
            << (Img->CompileOptions ? Img->CompileOptions : "NULL") << "\n";
  std::cerr << "    Link options    : "
            << (Img->LinkOptions ? Img->LinkOptions : "NULL") << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Img->BinaryEnd - (intptr_t)Img->BinaryStart) << "\n";
  std::cerr << "    Entries  : ";
  for (_pi_offload_entry EntriesIt = Img->EntriesBegin;
       EntriesIt != Img->EntriesEnd; ++EntriesIt)
    std::cerr << EntriesIt->name << " ";
  std::cerr << "\n";
}

void ProgramManager::debugDumpBinaryImages() const {
  for (const auto &ImgVecIt : m_DeviceImages) {
    std::cerr << "  ++++++ Kernel set: " << ImgVecIt.first << "\n";
    for (const auto &Img : *ImgVecIt.second)
      debugDumpBinaryImage(Img);
  }
}

KernelSetId ProgramManager::getNextKernelSetId() const {
  // No need for atomic, should be guarded by the caller
  static KernelSetId Result = LastKSId;
  return ++Result;
}

KernelSetId
ProgramManager::getKernelSetId(OSModuleHandle M,
                               const string_class &KernelName) const {
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

  throw runtime_error("No kernel named " + KernelName + " was found");
}

RT::PiDeviceBinaryType ProgramManager::getFormat(const DeviceImage &Img) const {
  if (Img.Format != PI_DEVICE_BINARY_TYPE_NONE)
    return Img.Format;

  struct {
    RT::PiDeviceBinaryType Fmt;
    const uint32_t Magic;
  } Fmts[] = {{PI_DEVICE_BINARY_TYPE_SPIRV, 0x07230203},
              {PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE, 0xDEC04342}};

  size_t ImgSize = static_cast<size_t>(Img.BinaryEnd - Img.BinaryStart);
  if (ImgSize >= sizeof(Fmts[0].Magic)) {
    std::remove_const<decltype(Fmts[0].Magic)>::type Hdr = 0;
    std::copy(Img.BinaryStart, Img.BinaryStart + sizeof(Hdr),
              reinterpret_cast<char *>(&Hdr));

    for (const auto &Fmt : Fmts) {
      if (Hdr == Fmt.Magic) {
        if (DbgProgMgr > 1)
          std::cerr << "determined image format: " << getFormatStr(Fmt.Fmt)
                    << "\n";
        return Fmt.Fmt;
      }
    }
  }

  return PI_DEVICE_BINARY_TYPE_NONE;
}

void ProgramManager::dumpImage(const DeviceImage &Img, KernelSetId KSId) const {
  std::string Fname("sycl_");
  Fname += Img.DeviceTargetSpec;
  Fname += std::to_string(KSId);
  std::string Ext;

  RT::PiDeviceBinaryType Format = getFormat(Img);
  if (Format == PI_DEVICE_BINARY_TYPE_SPIRV)
    Ext = ".spv";
  else if (Format == PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE)
    Ext = ".bc";
  else
    Ext = ".bin";
  Fname += Ext;

  std::ofstream F(Fname, std::ios::binary);

  if (!F.is_open()) {
    throw runtime_error(std::string("Can not write ") + Fname);
  }
  size_t ImgSize = static_cast<size_t>(Img.BinaryEnd - Img.BinaryStart);
  F.write(reinterpret_cast<const char *>(Img.BinaryStart), ImgSize);
  F.close();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

extern "C" void __sycl_register_lib(pi_device_binaries desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(pi_device_binaries desc) {
  // TODO implement the function
}
