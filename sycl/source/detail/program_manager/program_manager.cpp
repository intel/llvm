//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace cl {
namespace sycl {
namespace detail {

static constexpr int DbgProgMgr = 0;

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

ProgramManager &ProgramManager::getInstance() {
  // The singleton ProgramManager instance, uses the "magic static" idiom.
  static ProgramManager Instance;
  return Instance;
}

static RT::PiDevice getFirstDevice(RT::PiContext Context) {
  cl_uint NumDevices = 0;
  PI_CALL(piContextGetInfo)(Context, PI_CONTEXT_INFO_NUM_DEVICES,
                            sizeof(NumDevices), &NumDevices,
                            /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 && "Context without devices?");

  vector_class<RT::PiDevice> Devices(NumDevices);
  size_t ParamValueSize = 0;
  PI_CALL(piContextGetInfo)(Context, PI_CONTEXT_INFO_DEVICES,
                            sizeof(cl_device_id) * NumDevices, &Devices[0],
                            &ParamValueSize);
  assert(ParamValueSize == sizeof(cl_device_id) * NumDevices &&
         "Number of CL_CONTEXT_DEVICES should match CL_CONTEXT_NUM_DEVICES.");
  return Devices[0];
}

static RT::PiProgram createBinaryProgram(const RT::PiContext Context,
                                         const unsigned char *Data,
                                         size_t DataLen) {
  // FIXME: we don't yet support multiple devices with a single binary.
#ifndef _NDEBUG
  cl_uint NumDevices = 0;
  PI_CALL(piContextGetInfo)(Context, PI_CONTEXT_INFO_NUM_DEVICES,
                            sizeof(NumDevices), &NumDevices,
                            /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  RT::PiDevice Device = getFirstDevice(Context);
  pi_int32 BinaryStatus = CL_SUCCESS;
  RT::PiProgram Program;
  PI_CALL(piclProgramCreateWithBinary)(Context, 1 /*one binary*/, &Device,
                                       &DataLen, &Data, &BinaryStatus,
                                       &Program);
  return Program;
}

static RT::PiProgram createSpirvProgram(const RT::PiContext Context,
                                        const unsigned char *Data,
                                        size_t DataLen) {
  RT::PiProgram Program = nullptr;
  PI_CALL(piProgramCreate)(Context, Data, DataLen, &Program);
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

static bool isDeviceBinaryTypeSupported(const context &C,
                                        RT::PiDeviceBinaryType Format) {
  // All formats except PI_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (pi::useBackend(pi::SYCL_BE_PI_OPENCL) &&
      C.get_platform().get_info<info::platform::version>() >= "2.1")
    return true;

  // Otherwise we need cl_khr_il_program extension to be present
  // and we can call clCreateProgramWithILKHR using the extension
  for (const device &D : C.get_devices()) {
    vector_class<string_class> Extensions =
        D.get_info<info::device::extensions>();
    if (std::find(Extensions.begin(), Extensions.end(),
                  string_class("cl_khr_il_program")) != Extensions.end())
      return true;
  }

  // This device binary type is not supported.
  return false;
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
  const RT::PiContext &Ctx = getRawSyclObjImpl(Context)->getHandleRef();
  RT::PiProgram Res = Format == PI_DEVICE_BINARY_TYPE_SPIRV
                          ? createSpirvProgram(Ctx, Img.BinaryStart, ImgSize)
                          : createBinaryProgram(Ctx, Img.BinaryStart, ImgSize);

  if (DbgProgMgr > 1)
    std::cerr << "created native program: " << Res << "\n";

  return Res;
}

RT::PiProgram
ProgramManager::getBuiltPIProgram(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName) {
  KernelSetId KSId = getKernelSetId(M, KernelName);
  std::shared_ptr<context_impl> Ctx = getSyclObjImpl(Context);
  std::map<KernelSetId, RT::PiProgram> &CachedPrograms =
      Ctx->getCachedPrograms();
  auto It = CachedPrograms.find(KSId);
  if (It != CachedPrograms.end())
    return It->second;

  const DeviceImage &Img = getDeviceImage(M, KSId, Context);
  RT::PiProgram Prg = createPIProgram(Img, Context);
  ProgramPtr ProgramManaged(
      Prg, RT::PluginInformation.PiFunctionTable.piProgramRelease);

  // Link a fallback implementation of device libraries if they are not
  // supported by a device compiler.
  // Pre-compiled programs are supposed to be already linked.
  const bool LinkDeviceLibs = getFormat(Img) == PI_DEVICE_BINARY_TYPE_SPIRV;

  context_impl *ContextImpl = getRawSyclObjImpl(Context);
  RT::PiContext PiContext = ContextImpl->getHandleRef();
  const std::vector<device> Devices = ContextImpl->getDevices();
  std::vector<RT::PiDevice> PiDevices(Devices.size());
  std::transform(
      Devices.begin(), Devices.end(), PiDevices.begin(),
      [](const device Dev) { return getRawSyclObjImpl(Dev)->getHandleRef(); });

  ProgramPtr BuiltProgram =
      build(std::move(ProgramManaged), PiContext, Img.BuildOptions, PiDevices,
            ContextImpl->getCachedLibPrograms(), LinkDeviceLibs);
  CachedPrograms[KSId] = BuiltProgram.get();
  return BuiltProgram.release();
}

RT::PiKernel ProgramManager::getOrCreateKernel(OSModuleHandle M,
                                               const context &Context,
                                               const string_class &KernelName) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << getRawSyclObjImpl(Context) << ", " << KernelName << ")\n";
  }
  RT::PiProgram Program = getBuiltPIProgram(M, Context, KernelName);
  std::shared_ptr<context_impl> Ctx = getSyclObjImpl(Context);
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> &CachedKernels =
      Ctx->getCachedKernels();
  std::map<string_class, RT::PiKernel> &KernelsCache = CachedKernels[Program];
  RT::PiKernel &Kernel = KernelsCache[KernelName];
  if (!Kernel) {
    PI_CALL(piKernelCreate)(Program, KernelName.c_str(), &Kernel);
    // TODO need some user-friendly error/exception
    // instead of currently obscure one
  }
  return Kernel;
}

RT::PiProgram ProgramManager::getClProgramFromClKernel(RT::PiKernel Kernel) {
  RT::PiProgram Program;
  PI_CALL(piKernelGetInfo)(Kernel, CL_KERNEL_PROGRAM, sizeof(cl_program),
                           &Program, nullptr);
  return Program;
}

string_class ProgramManager::getProgramBuildLog(const RT::PiProgram &Program) {
  size_t Size = 0;
  PI_CALL(piProgramGetInfo)(Program, CL_PROGRAM_DEVICES, 0, nullptr, &Size);
  vector_class<RT::PiDevice> PIDevices(Size / sizeof(RT::PiDevice));
  PI_CALL(piProgramGetInfo)(Program, CL_PROGRAM_DEVICES, Size, PIDevices.data(),
                            nullptr);
  string_class Log = "The program was built for " +
                     std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
    PI_CALL(piProgramGetBuildInfo)(Program, Device, CL_PROGRAM_BUILD_LOG, 0,
                                   nullptr, &Size);
    vector_class<char> DeviceBuildInfo(Size);
    PI_CALL(piProgramGetBuildInfo)(Program, Device, CL_PROGRAM_BUILD_LOG, Size,
                                   DeviceBuildInfo.data(), nullptr);
    PI_CALL(piDeviceGetInfo)(Device, PI_DEVICE_INFO_NAME, 0, nullptr, &Size);
    vector_class<char> DeviceName(Size);
    PI_CALL(piDeviceGetInfo)(Device, PI_DEVICE_INFO_NAME, Size,
                             DeviceName.data(), nullptr);

    Log += "\nBuild program log for '" + string_class(DeviceName.data()) +
           "':\n" + string_class(DeviceBuildInfo.data());
  }
  return Log;
}

static bool loadDeviceLib(const RT::PiContext &Context, const char *Name,
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

static std::string getDeviceExtensions(const RT::PiDevice &Dev) {
  std::string DevExt;
  size_t DevExtSize = 0;
  PI_CALL(piDeviceGetInfo)(Dev, PI_DEVICE_INFO_EXTENSIONS,
                           /*param_value_size=*/0,
                           /*param_value=*/nullptr, &DevExtSize);
  DevExt.resize(DevExtSize);
  PI_CALL(piDeviceGetInfo)(Dev, PI_DEVICE_INFO_EXTENSIONS, DevExt.size(),
                           &DevExt[0],
                           /*param_value_size_ret=*/nullptr);
  return DevExt;
}

static const char* getDeviceLibFilename(DeviceLibExt Extension) {
  switch (Extension) {
  case cl_intel_devicelib_assert:
    return "libsycl-fallback-cassert.spv";
  }
  throw compile_program_error("Unhandled (new?) device library extension");
}

const char* getDeviceLibExtensionStr(DeviceLibExt Extension) {
  switch (Extension) {
  case cl_intel_devicelib_assert:
    return "cl_intel_devicelib_assert";
  }
  throw compile_program_error("Unhandled (new?) device library extension");
}

static RT::PiProgram
loadDeviceLibFallback(
    const RT::PiContext &Context,
    DeviceLibExt Extension,
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

  RT::PiResult Error = PI_CALL_NOCHECK(piProgramCompile)(
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
    throw compile_program_error(ProgramManager::getProgramBuildLog(LibProg));
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
    ImgPtr->BuildOptions = "";
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
  const RT::PiContext &Ctx = getRawSyclObjImpl(Context)->getHandleRef();
  DeviceImage *Img = nullptr;

  // TODO: There may be cases with cl::sycl::program class usage in source code
  // that will result in a multi-device context. This case needs to be handled
  // here or at the program_impl class level

  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  if (Imgs.size() > 1) {
    PI_CALL(piextDeviceSelectBinary)(getFirstDevice(Ctx), Imgs.data(),
                                     (cl_uint)Imgs.size(), &Img);
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

static std::vector<RT::PiProgram> getDeviceLibPrograms(
    const RT::PiContext Context,
    const std::vector<RT::PiDevice> &Devices,
    std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms) {

  std::vector<RT::PiProgram> Programs;

  // TODO: SYCL compiler should generate a list of required extensions for a
  // particular program in order to allow us do a more fine-grained check here.
  // Require *all* possible devicelib extensions for now.
  std::pair<DeviceLibExt, bool> RequiredDeviceLibExt[] = {
      {cl_intel_devicelib_assert, false}
  };

  // Load a fallback library for an extension if at least one device does not
  // support it.
  for (RT::PiDevice Dev : Devices) {
    std::string DevExtList = getDeviceExtensions(Dev);
    for (auto &Pair : RequiredDeviceLibExt) {
      DeviceLibExt Ext = Pair.first;
      bool &FallbackIsLoaded = Pair.second;

      const char* ExtStr = getDeviceLibExtensionStr(Ext);

      if (FallbackIsLoaded) {
        continue;
      }

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
ProgramManager::build(ProgramPtr Program, RT::PiContext Context,
                      const string_class &Options,
                      const std::vector<RT::PiDevice> &Devices,
                      std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms,
                      bool LinkDeviceLibs) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << Options << ", ... " << Devices.size() << ")\n";
  }
  const char *Opts = std::getenv("SYCL_PROGRAM_BUILD_OPTIONS");

  for (const auto &DeviceId : Devices) {
    if (!createSyclObjFromImpl<device>(std::make_shared<device_impl>(DeviceId))
             .get_info<info::device::is_compiler_available>()) {
      throw feature_not_supported(
          "Online compilation is not supported by this device");
    }
  }

  if (!Opts)
    Opts = Options.c_str();

  std::vector<RT::PiProgram> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Devices, CachedLibPrograms);
  }

  if (LinkPrograms.empty()) {
    RT::PiResult Error = PI_CALL_NOCHECK(piProgramBuild)(
        Program.get(), Devices.size(), Devices.data(), Opts, nullptr, nullptr);
    if (Error != PI_SUCCESS)
      throw compile_program_error(getProgramBuildLog(Program.get()));
    return Program;
  }

  // Include the main program and compile/link everything together
  PI_CALL(piProgramCompile)(Program.get(), Devices.size(), Devices.data(), Opts,
                            0, nullptr, nullptr, nullptr, nullptr);
  LinkPrograms.push_back(Program.get());

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = PI_CALL_NOCHECK(piProgramLink)(
      Context, Devices.size(), Devices.data(), Opts, LinkPrograms.size(),
      LinkPrograms.data(), nullptr, nullptr, &LinkedProg);

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise. Release the original (user) program.
  Program.reset(LinkedProg);
  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg, and throw.
      throw compile_program_error(getProgramBuildLog(LinkedProg));
    }
    pi::checkPiResult(Error);
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
  std::cerr << "    Version  : " << (int)Img->Version << "\n";
  std::cerr << "    Kind     : " << (int)Img->Kind << "\n";
  std::cerr << "    Format   : " << (int)Img->Format << "\n";
  std::cerr << "    Target   : " << Img->DeviceTargetSpec << "\n";
  std::cerr << "    Options  : "
            << (Img->BuildOptions ? Img->BuildOptions : "NULL") << "\n";
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
          std::cerr << "determined image format: " << (int)Fmt.Fmt << "\n";
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
} // namespace cl

extern "C" void __tgt_register_lib(pi_device_binaries desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __tgt_unregister_lib(pi_device_binaries desc) {
  // TODO implement the function
}
