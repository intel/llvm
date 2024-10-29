//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/compiler.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/spec_constant_impl.hpp>
#include <detail/split_string.hpp>
#include <detail/ur_info_code.hpp>
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
static void enableITTAnnotationsIfNeeded(const ur_program_handle_t &Prog,
                                         const AdapterPtr &Adapter) {
  if (SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::get() != nullptr) {
    constexpr char SpecValue = 1;
    ur_specialization_constant_info_t SpecConstInfo = {
        ITTSpecConstId, sizeof(char), &SpecValue};
    Adapter->call<UrApiKind::urProgramSetSpecializationConstants>(
        Prog, 1, &SpecConstInfo);
  }
}

ProgramManager &ProgramManager::getInstance() {
  return GlobalHandler::instance().getProgramManager();
}

static ur_program_handle_t
createBinaryProgram(const ContextImplPtr Context,
                    const std::vector<device> &Devices,
                    const uint8_t **Binaries, size_t *Lengths,
                    const std::vector<ur_program_metadata_t> Metadata) {
  const AdapterPtr &Adapter = Context->getAdapter();
  ur_program_handle_t Program;
  std::vector<ur_device_handle_t> DeviceHandles;
  std::transform(
      Devices.begin(), Devices.end(), std::back_inserter(DeviceHandles),
      [](const device &Dev) { return getSyclObjImpl(Dev)->getHandleRef(); });
  ur_result_t BinaryStatus = UR_RESULT_SUCCESS;
  ur_program_properties_t Properties = {};
  Properties.stype = UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES;
  Properties.pNext = nullptr;
  Properties.count = Metadata.size();
  Properties.pMetadatas = Metadata.data();

  assert(Devices.size() > 0 && "No devices provided for program creation");
  Adapter->call<UrApiKind::urProgramCreateWithBinary>(
      Context->getHandleRef(), DeviceHandles.size(), DeviceHandles.data(),
      Lengths, Binaries, &Properties, &Program);
  if (BinaryStatus != UR_RESULT_SUCCESS) {
    throw detail::set_ur_error(
        exception(make_error_code(errc::runtime),
                  "Creating program with binary failed."),
        BinaryStatus);
  }

  return Program;
}

static ur_program_handle_t createSpirvProgram(const ContextImplPtr Context,
                                              const unsigned char *Data,
                                              size_t DataLen) {
  ur_program_handle_t Program = nullptr;
  const AdapterPtr &Adapter = Context->getAdapter();
  Adapter->call<UrApiKind::urProgramCreateWithIL>(Context->getHandleRef(), Data,
                                                  DataLen, nullptr, &Program);
  return Program;
}

// TODO replace this with a new UR API function
static bool isDeviceBinaryTypeSupported(const context &C,
                                        ur::DeviceBinaryType Format) {
  // All formats except SYCL_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != SYCL_DEVICE_BINARY_TYPE_SPIRV)
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

// getFormatStr is used for debug-printing, so it may be unused.
[[maybe_unused]] static const char *getFormatStr(ur::DeviceBinaryType Format) {
  switch (Format) {
  case SYCL_DEVICE_BINARY_TYPE_NONE:
    return "none";
  case SYCL_DEVICE_BINARY_TYPE_NATIVE:
    return "native";
  case SYCL_DEVICE_BINARY_TYPE_SPIRV:
    return "SPIR-V";
  case SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE:
    return "LLVM IR";
  case SYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE:
    return "compressed none";
  }
  assert(false && "Unknown device image format");
  return "unknown";
}

[[maybe_unused]] auto VecToString = [](auto &Vec) -> std::string {
  std::ostringstream Out;
  Out << "{";
  for (auto Elem : Vec)
    Out << Elem << " ";
  Out << "}";
  return Out.str();
};

ur_program_handle_t
ProgramManager::createURProgram(const RTDeviceBinaryImage &Img,
                                const context &Context,
                                const std::vector<device> &Devices) {
  if constexpr (DbgProgMgr > 0) {
    std::vector<ur_device_handle_t> URDevices;
    std::transform(
        Devices.begin(), Devices.end(), std::back_inserter(URDevices),
        [](const device &Dev) { return getSyclObjImpl(Dev)->getHandleRef(); });
    std::cerr << ">>> ProgramManager::createPIProgram(" << &Img << ", "
              << getSyclObjImpl(Context).get() << ", " << VecToString(URDevices)
              << ")\n";
  }
  const sycl_device_binary_struct &RawImg = Img.getRawData();

  // perform minimal sanity checks on the device image and the descriptor
  if (RawImg.BinaryEnd < RawImg.BinaryStart) {
    throw exception(make_error_code(errc::runtime),
                    "Malformed device program image descriptor");
  }
  if (RawImg.BinaryEnd == RawImg.BinaryStart) {
    throw exception(make_error_code(errc::runtime),
                    "Invalid device program image: size is zero");
  }
  size_t ImgSize = Img.getSize();

  // TODO if the binary image is a part of the fat binary, the clang
  //   driver should have set proper format option to the
  //   clang-offload-wrapper. The fix depends on AOT compilation
  //   implementation, so will be implemented together with it.
  //   Img->Format can't be updated as it is inside of the in-memory
  //   OS module binary.
  ur::DeviceBinaryType Format = Img.getFormat();

  if (Format == SYCL_DEVICE_BINARY_TYPE_NONE)
    Format = ur::getBinaryImageFormat(RawImg.BinaryStart, ImgSize);
  // sycl::detail::pi::PiDeviceBinaryType Format = Img->Format;
  // assert(Format != SYCL_DEVICE_BINARY_TYPE_NONE && "Image format not set");

  if (!isDeviceBinaryTypeSupported(Context, Format))
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "SPIR-V online compilation is not supported in this context");

  // Get program metadata from properties
  auto ProgMetadata = Img.getProgramMetadataUR();

  // Load the image
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  std::vector<const uint8_t *> Binaries(
      Devices.size(), const_cast<uint8_t *>(RawImg.BinaryStart));
  std::vector<size_t> Lengths(Devices.size(), ImgSize);
  ur_program_handle_t Res =
      Format == SYCL_DEVICE_BINARY_TYPE_SPIRV
          ? createSpirvProgram(Ctx, RawImg.BinaryStart, ImgSize)
          : createBinaryProgram(Ctx, Devices, Binaries.data(), Lengths.data(),
                                ProgMetadata);

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    // associate the UR program with the image it was created for
    NativePrograms.insert({Res, &Img});
  }

  Ctx->addDeviceGlobalInitializer(Res, Devices, &Img);

  if constexpr (DbgProgMgr > 1)
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
  sycl_device_binary_property Prop = Img.getProperty(PropName);
  return Prop && (DeviceBinaryProperty(Prop).asUint32() != 0);
}

static std::string getUint32PropAsOptStr(const RTDeviceBinaryImage &Img,
                                         const char *PropName) {
  sycl_device_binary_property Prop = Img.getProperty(PropName);
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
  sycl_device_binary_property RegAllocModeProp =
      Img.getProperty("sycl-register-alloc-mode");
  sycl_device_binary_property GRFSizeProp = Img.getProperty("sycl-grf-size");

  if (!RegAllocModeProp && !GRFSizeProp)
    return;
  // The mutual exclusivity of these properties should have been checked in
  // sycl-post-link.
  assert(!RegAllocModeProp || !GRFSizeProp);
  bool Is256GRF = false;
  bool IsAutoGRF = false;
  if (RegAllocModeProp) {
    uint32_t RegAllocModePropVal =
        DeviceBinaryProperty(RegAllocModeProp).asUint32();
    Is256GRF = RegAllocModePropVal ==
               static_cast<uint32_t>(register_alloc_mode_enum::large);
    IsAutoGRF = RegAllocModePropVal ==
                static_cast<uint32_t>(register_alloc_mode_enum::automatic);
  } else {
    assert(GRFSizeProp);
    uint32_t GRFSizePropVal = DeviceBinaryProperty(GRFSizeProp).asUint32();
    Is256GRF = GRFSizePropVal == 256;
    IsAutoGRF = GRFSizePropVal == 0;
  }
  if (Is256GRF) {
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
                                          const AdapterPtr &) {
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

  const auto PlatformImpl = detail::getSyclObjImpl(Devs[0].get_platform());

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
                                  const AdapterPtr &Adapter) {
  appendCompileOptionsFromImage(CompileOpts, Img, Devices, Adapter);
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

std::pair<ur_program_handle_t, bool> ProgramManager::getOrCreateURProgram(
    const RTDeviceBinaryImage &MainImg,
    const std::vector<const RTDeviceBinaryImage *> &AllImages,
    const context &Context, const std::vector<device> &Devices,
    const std::string &CompileAndLinkOptions, SerializedObj SpecConsts) {
  ur_program_handle_t NativePrg; // TODO: Or native?

  auto BinProg = PersistentDeviceCodeCache::getItemFromDisc(
      Devices[0], AllImages, SpecConsts, CompileAndLinkOptions);
  if (BinProg.size()) {
    // Get program metadata from properties
    std::vector<ur_program_metadata_t> ProgMetadataVector;
    for (const RTDeviceBinaryImage *Img : AllImages) {
      auto &ImgProgMetadata = Img->getProgramMetadataUR();
      ProgMetadataVector.insert(ProgMetadataVector.end(),
                                ImgProgMetadata.begin(), ImgProgMetadata.end());
    }
    std::vector<const uint8_t *> Binaries(Devices.size(),
                                          (const uint8_t *)BinProg[0].data());
    std::vector<size_t> Lengths(Devices.size(), BinProg[0].size());
    NativePrg =
        createBinaryProgram(getSyclObjImpl(Context), Devices, Binaries.data(),
                            Lengths.data(), ProgMetadataVector);
  } else {
    NativePrg = createURProgram(MainImg, Context, Devices);
  }
  return {NativePrg, BinProg.size()};
}

/// Emits information about built programs if the appropriate contitions are
/// met, namely when SYCL_RT_WARNING_LEVEL is greater than or equal to 2.
static void emitBuiltProgramInfo(const ur_program_handle_t &Prog,
                                 const ContextImplPtr &Context) {
  if (SYCLConfig<SYCL_RT_WARNING_LEVEL>::get() >= 2) {
    std::string ProgramBuildLog =
        ProgramManager::getProgramBuildLog(Prog, Context);
    std::clog << ProgramBuildLog << std::endl;
  }
}

static const char *getUrDeviceTarget(const char *URDeviceTarget) {
  if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_UNKNOWN) == 0)
    return UR_DEVICE_BINARY_TARGET_UNKNOWN;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_SPIRV32) == 0)
    return UR_DEVICE_BINARY_TARGET_SPIRV32;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0)
    return UR_DEVICE_BINARY_TARGET_SPIRV64;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64) ==
           0)
    return UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0)
    return UR_DEVICE_BINARY_TARGET_SPIRV64_GEN;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA) ==
           0)
    return UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_NVPTX64) == 0)
    return UR_DEVICE_BINARY_TARGET_NVPTX64;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_AMDGCN) == 0)
    return UR_DEVICE_BINARY_TARGET_AMDGCN;
  else if (strcmp(URDeviceTarget, __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU) == 0)
    return "native_cpu"; // todo: define UR_DEVICE_BINARY_TARGET_NATIVE_CPU;

  return UR_DEVICE_BINARY_TARGET_UNKNOWN;
}

static bool compatibleWithDevice(RTDeviceBinaryImage *BinImage,
                                 const device &Dev) {
  const std::shared_ptr<detail::device_impl> &DeviceImpl =
      detail::getSyclObjImpl(Dev);
  auto &Adapter = DeviceImpl->getAdapter();

  const ur_device_handle_t &URDeviceHandle = DeviceImpl->getHandleRef();

  // Call urDeviceSelectBinary with only one image to check if an image is
  // compatible with implementation. The function returns invalid index if no
  // device images are compatible.
  uint32_t SuitableImageID = std::numeric_limits<uint32_t>::max();
  sycl_device_binary DevBin =
      const_cast<sycl_device_binary>(&BinImage->getRawData());

  ur_device_binary_t UrBinary{};
  UrBinary.pDeviceTargetSpec = getUrDeviceTarget(DevBin->DeviceTargetSpec);

  ur_result_t Error = Adapter->call_nocheck<UrApiKind::urDeviceSelectBinary>(
      URDeviceHandle, &UrBinary,
      /*num bin images = */ (uint32_t)1, &SuitableImageID);
  if (Error != UR_RESULT_SUCCESS && Error != UR_RESULT_ERROR_INVALID_BINARY)
    throw detail::set_ur_error(exception(make_error_code(errc::runtime),
                                         "Invalid binary image or device"),
                               Error);

  return (0 == SuitableImageID);
}

static bool checkLinkingSupport(device Dev, const RTDeviceBinaryImage &Img) {
  const char *Target = Img.getRawData().DeviceTargetSpec;
  // TODO replace with extension checks once implemented in UR.
  if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0) {
    return true;
  }
  if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) {
    return Dev.is_gpu() && Dev.get_backend() == backend::opencl;
  }
  return false;
}

std::set<RTDeviceBinaryImage *>
ProgramManager::collectDeviceImageDepsForImportedSymbols(
    const RTDeviceBinaryImage &MainImg, device Dev) {
  std::set<RTDeviceBinaryImage *> DeviceImagesToLink;
  std::set<std::string> HandledSymbols;
  std::queue<std::string> WorkList;
  for (const sycl_device_binary_property &ISProp :
       MainImg.getImportedSymbols()) {
    WorkList.push(ISProp->Name);
    HandledSymbols.insert(ISProp->Name);
  }
  ur::DeviceBinaryType Format = MainImg.getFormat();
  if (!WorkList.empty() && !checkLinkingSupport(Dev, MainImg))
    throw exception(make_error_code(errc::feature_not_supported),
                    "Cannot resolve external symbols, linking is unsupported "
                    "for the backend");
  while (!WorkList.empty()) {
    std::string Symbol = WorkList.front();
    WorkList.pop();

    auto Range = m_ExportedSymbolImages.equal_range(Symbol);
    bool Found = false;
    for (auto It = Range.first; It != Range.second; ++It) {
      RTDeviceBinaryImage *Img = It->second;
      if (Img->getFormat() != Format ||
          !doesDevSupportDeviceRequirements(Dev, *Img) ||
          !compatibleWithDevice(Img, Dev))
        continue;
      DeviceImagesToLink.insert(Img);
      Found = true;
      for (const sycl_device_binary_property &ISProp :
           Img->getImportedSymbols()) {
        if (HandledSymbols.insert(ISProp->Name).second)
          WorkList.push(ISProp->Name);
      }
      break;
    }
    if (!Found)
      throw sycl::exception(make_error_code(errc::build),
                            "No device image found for external symbol " +
                                Symbol);
  }
  DeviceImagesToLink.erase(const_cast<RTDeviceBinaryImage *>(&MainImg));
  return DeviceImagesToLink;
}

std::set<RTDeviceBinaryImage *>
ProgramManager::collectDependentDeviceImagesForVirtualFunctions(
    const RTDeviceBinaryImage &Img, device Dev) {
  // If virtual functions are used in a program, then we need to link several
  // device images together to make sure that vtable pointers stored in
  // objects are valid between different kernels (which could be in different
  // device images).
  std::set<RTDeviceBinaryImage *> DeviceImagesToLink;
  // KernelA may use some set-a, which is also used by KernelB that in turn
  // uses set-b, meaning that this search should be recursive. The set below
  // is used to stop that recursion, i.e. to avoid looking at sets we have
  // already seen.
  std::set<std::string> HandledSets;
  std::queue<std::string> WorkList;
  for (const sycl_device_binary_property &VFProp : Img.getVirtualFunctions()) {
    std::string StrValue = DeviceBinaryProperty(VFProp).asCString();
    // Device image passed to this function is expected to contain SYCL kernels
    // and therefore it may only use virtual function sets, but cannot provide
    // them. We expect to see just a single property here
    assert(std::string(VFProp->Name) == "uses-virtual-functions-set" &&
           "Unexpected virtual function property");
    for (const auto &SetName : detail::split_string(StrValue, ',')) {
      WorkList.push(SetName);
      HandledSets.insert(SetName);
    }
  }

  while (!WorkList.empty()) {
    std::string SetName = WorkList.front();
    WorkList.pop();

    // There could be more than one device image that uses the same set
    // of virtual functions, or provides virtual funtions from the same
    // set.
    for (RTDeviceBinaryImage *BinImage : m_VFSet2BinImage[SetName]) {
      // Here we can encounter both uses-virtual-functions-set and
      // virtual-functions-set properties, but their handling is the same: we
      // just grab all sets they reference and add them for consideration if
      // we haven't done so already.
      for (const sycl_device_binary_property &VFProp :
           BinImage->getVirtualFunctions()) {
        std::string StrValue = DeviceBinaryProperty(VFProp).asCString();
        for (const auto &SetName : detail::split_string(StrValue, ',')) {
          if (HandledSets.insert(SetName).second)
            WorkList.push(SetName);
        }
      }

      // TODO: Complete this part about handling of incompatible device images.
      // If device image uses the same virtual function set, then we only
      // link it if it is compatible.
      // However, if device image provides virtual function set and it is
      // incompatible, then we should link its "dummy" version to avoid link
      // errors about unresolved external symbols.
      if (doesDevSupportDeviceRequirements(Dev, *BinImage))
        DeviceImagesToLink.insert(BinImage);
    }
  }

  // We may have inserted the original image into the list as well, because it
  // is also a part of m_VFSet2BinImage map. No need to to return it to avoid
  // passing it twice to link call later.
  DeviceImagesToLink.erase(const_cast<RTDeviceBinaryImage *>(&Img));

  return DeviceImagesToLink;
}

static void
setSpecializationConstants(const std::shared_ptr<device_image_impl> &InputImpl,
                           ur_program_handle_t Prog,
                           const AdapterPtr &Adapter) {
  // Set ITT annotation specialization constant if needed.
  enableITTAnnotationsIfNeeded(Prog, Adapter);

  std::lock_guard<std::mutex> Lock{InputImpl->get_spec_const_data_lock()};
  const std::map<std::string, std::vector<device_image_impl::SpecConstDescT>>
      &SpecConstData = InputImpl->get_spec_const_data_ref();
  const SerializedObj &SpecConsts = InputImpl->get_spec_const_blob_ref();

  // Set all specialization IDs from descriptors in the input device image.
  for (const auto &[SpecConstNames, SpecConstDescs] : SpecConstData) {
    std::ignore = SpecConstNames;
    for (const device_image_impl::SpecConstDescT &SpecIDDesc : SpecConstDescs) {
      if (SpecIDDesc.IsSet) {
        ur_specialization_constant_info_t SpecConstInfo = {
            SpecIDDesc.ID, SpecIDDesc.Size,
            SpecConsts.data() + SpecIDDesc.BlobOffset};
        Adapter->call<UrApiKind::urProgramSetSpecializationConstants>(
            Prog, 1, &SpecConstInfo);
      }
    }
  }
}

static inline void CheckAndDecompressImage([[maybe_unused]] RTDeviceBinaryImage *Img) {
#ifndef SYCL_RT_ZSTD_NOT_AVAIABLE
  if (auto CompImg = dynamic_cast<CompressedRTDeviceBinaryImage *>(Img))
    if (CompImg->IsCompressed())
      CompImg->Decompress();
#endif
}

// When caching is enabled, the returned UrProgram will already have
// its ref count incremented.
ur_program_handle_t ProgramManager::getBuiltURProgram(
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

  ur_bool_t MustBuildOnSubdevice = true;
  ContextImpl->getAdapter()->call<UrApiKind::urDeviceGetInfo>(
      RootDevImpl->getHandleRef(), UR_DEVICE_INFO_BUILD_ON_SUBDEVICE,
      sizeof(ur_bool_t), &MustBuildOnSubdevice, nullptr);

  DeviceImplPtr Dev = (MustBuildOnSubdevice == true) ? DeviceImpl : RootDevImpl;
  auto Context = createSyclObjFromImpl<context>(ContextImpl);
  auto Device = createSyclObjFromImpl<device>(Dev);
  const RTDeviceBinaryImage &Img =
      getDeviceImage(KernelName, Context, Device, JITCompilationIsRequired);

  // Check that device supports all aspects used by the kernel
  if (auto exception = checkDevSupportDeviceRequirements(Device, Img, NDRDesc))
    throw *exception;

  // TODO collecting dependencies for virtual functions and imported symbols
  // should be combined since one can lead to new unresolved dependencies for
  // the other.
  std::set<RTDeviceBinaryImage *> DeviceImagesToLink =
      collectDependentDeviceImagesForVirtualFunctions(Img, Device);

  std::set<RTDeviceBinaryImage *> ImageDeps =
      collectDeviceImageDepsForImportedSymbols(Img, Device);
  DeviceImagesToLink.insert(ImageDeps.begin(), ImageDeps.end());

  // Decompress all DeviceImagesToLink
  for (RTDeviceBinaryImage *BinImg : DeviceImagesToLink)
    CheckAndDecompressImage(BinImg);

  std::vector<const RTDeviceBinaryImage *> AllImages;
  AllImages.reserve(ImageDeps.size() + 1);
  AllImages.push_back(&Img);
  std::copy(ImageDeps.begin(), ImageDeps.end(), std::back_inserter(AllImages));

  auto BuildF = [this, &Img, &Context, &ContextImpl, &Device, &CompileOpts,
                 &LinkOpts, SpecConsts, &DeviceImagesToLink, &AllImages] {
    const AdapterPtr &Adapter = ContextImpl->getAdapter();
    applyOptionsFromImage(CompileOpts, LinkOpts, Img, {Device}, Adapter);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOpts);
    appendLinkEnvironmentVariablesThatAppend(LinkOpts);
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreateURProgram(
        Img, AllImages, Context, {Device}, CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache) {
      if (Img.supportsSpecConstants())
        enableITTAnnotationsIfNeeded(NativePrg, Adapter);
    }

    UrFuncInfo<UrApiKind::urProgramRelease> programReleaseInfo;
    auto programRelease =
        programReleaseInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
    ProgramPtr ProgramManaged(NativePrg, programRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs (after AOT compilation or read from persitent
    // cache) are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    bool UseDeviceLibs = !DeviceCodeWasInCache &&
                         Img.getFormat() == SYCL_DEVICE_BINARY_TYPE_SPIRV &&
                         !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get();
    if (UseDeviceLibs)
      DeviceLibReqMask = getDeviceLibReqMask(Img);

    std::vector<ur_program_handle_t> ProgramsToLink;
    // If we had a program in cache, then it should have been the fully linked
    // program already.
    if (!DeviceCodeWasInCache) {
      for (RTDeviceBinaryImage *BinImg : DeviceImagesToLink) {
        if (UseDeviceLibs)
          DeviceLibReqMask |= getDeviceLibReqMask(*BinImg);
        device_image_plain DevImagePlain =
            getDeviceImageFromBinaryImage(BinImg, Context, Device);
        const std::shared_ptr<detail::device_image_impl> &DeviceImageImpl =
            detail::getSyclObjImpl(DevImagePlain);

        SerializedObj ImgSpecConsts =
            DeviceImageImpl->get_spec_const_blob_ref();

        ur_program_handle_t NativePrg =
            createURProgram(*BinImg, Context, {Device});

        if (BinImg->supportsSpecConstants())
          setSpecializationConstants(DeviceImageImpl, NativePrg, Adapter);

        ProgramsToLink.push_back(NativePrg);
      }
    }
    std::vector<ur_device_handle_t> Devs = {
        getSyclObjImpl(Device).get()->getHandleRef()};
    ;
    ProgramPtr BuiltProgram = build(
        std::move(ProgramManaged), ContextImpl, CompileOpts, LinkOpts, Devs,
        DeviceLibReqMask, ProgramsToLink,
        /*CreatedFromBinary*/ Img.getFormat() != SYCL_DEVICE_BINARY_TYPE_SPIRV);
    // Those extra programs won't be used anymore, just the final linked result
    for (ur_program_handle_t Prg : ProgramsToLink)
      Adapter->call<UrApiKind::urProgramRelease>(Prg);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms.insert({BuiltProgram.get(), &Img});
      for (RTDeviceBinaryImage *LinkedImg : DeviceImagesToLink) {
        NativePrograms.insert({BuiltProgram.get(), LinkedImg});
      }
    }

    ContextImpl->addDeviceGlobalInitializer(BuiltProgram.get(), {Device}, &Img);

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache) {
      PersistentDeviceCodeCache::putItemToDisc(Device, AllImages, SpecConsts,
                                               CompileOpts + LinkOpts,
                                               BuiltProgram.get());
    }
    return BuiltProgram.release();
  };

  uint32_t ImgId = Img.getImageID();
  const ur_device_handle_t UrDevice = Dev->getHandleRef();
  auto CacheKey = std::make_pair(std::make_pair(std::move(SpecConsts), ImgId),
                                 std::set{UrDevice});

  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get())
    return BuildF();

  auto BuildResult = Cache.getOrBuild<errc::build>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  ur_program_handle_t ResProgram = BuildResult->Val;
  auto Adapter = ContextImpl->getAdapter();

  // If we linked any extra device images, then we need to
  // cache them as well.
  for (const RTDeviceBinaryImage *BImg : DeviceImagesToLink) {
    // CacheKey is captured by reference by GetCachedBuildF, so we can simply
    // update it here and re-use that lambda.
    CacheKey.first.second = BImg->getImageID();
    bool DidInsert = Cache.insertBuiltProgram(CacheKey, ResProgram);
    if (DidInsert) {
      // For every cached copy of the program, we need to increment its refcount
      Adapter->call<UrApiKind::urProgramRetain>(ResProgram);
    }
  }

  // If caching is enabled, one copy of the program handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // program.
  ContextImpl->getAdapter()->call<UrApiKind::urProgramRetain>(ResProgram);
  return ResProgram;
}

// When caching is enabled, the returned UrProgram and UrKernel will
// already have their ref count incremented.
std::tuple<ur_kernel_handle_t, std::mutex *, const KernelArgMask *,
           ur_program_handle_t>
ProgramManager::getOrCreateKernel(const ContextImplPtr &ContextImpl,
                                  const DeviceImplPtr &DeviceImpl,
                                  const std::string &KernelName,
                                  const NDRDescT &NDRDesc) {
  if constexpr (DbgProgMgr > 0) {
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
  ur_device_handle_t UrDevice = DeviceImpl->getHandleRef();

  auto key = std::make_tuple(std::move(SpecConsts), UrDevice,
                             CompileOpts + LinkOpts, KernelName);
  if (SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    auto ret_tuple = Cache.tryToGetKernelFast(key);
    constexpr size_t Kernel = 0;  // see KernelFastCacheValT tuple
    constexpr size_t Program = 3; // see KernelFastCacheValT tuple
    if (std::get<Kernel>(ret_tuple)) {
      // Pulling a copy of a kernel and program from the cache,
      // so we need to retain those resources.
      ContextImpl->getAdapter()->call<UrApiKind::urKernelRetain>(
          std::get<Kernel>(ret_tuple));
      ContextImpl->getAdapter()->call<UrApiKind::urProgramRetain>(
          std::get<Program>(ret_tuple));
      return ret_tuple;
    }
  }

  ur_program_handle_t Program =
      getBuiltURProgram(ContextImpl, DeviceImpl, KernelName, NDRDesc);

  auto BuildF = [this, &Program, &KernelName, &ContextImpl] {
    ur_kernel_handle_t Kernel = nullptr;

    const AdapterPtr &Adapter = ContextImpl->getAdapter();
    Adapter->call<errc::kernel_not_supported, UrApiKind::urKernelCreate>(
        Program, KernelName.c_str(), &Kernel);

    // Only set UR_USM_INDIRECT_ACCESS if the platform can handle it.
    if (ContextImpl->getPlatformImpl()->supports_usm()) {
      // Some UR Adapters (like OpenCL) require this call to enable USM
      // For others, UR will turn this into a NOP.
      const ur_bool_t UrTrue = true;
      Adapter->call<UrApiKind::urKernelSetExecInfo>(
          Kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
          nullptr, &UrTrue);
    }

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

  auto BuildResult = Cache.getOrBuild<errc::invalid>(GetCachedBuildF, BuildF);
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
  ContextImpl->getAdapter()->call<UrApiKind::urKernelRetain>(
      KernelArgMaskPair.first);
  Cache.saveKernel(key, ret_val);
  return ret_val;
}

ur_program_handle_t
ProgramManager::getUrProgramFromUrKernel(ur_kernel_handle_t Kernel,
                                         const ContextImplPtr Context) {
  ur_program_handle_t Program;
  const AdapterPtr &Adapter = Context->getAdapter();
  Adapter->call<UrApiKind::urKernelGetInfo>(Kernel, UR_KERNEL_INFO_PROGRAM,
                                            sizeof(ur_program_handle_t),
                                            &Program, nullptr);
  return Program;
}

std::string
ProgramManager::getProgramBuildLog(const ur_program_handle_t &Program,
                                   const ContextImplPtr Context) {
  size_t URDevicesSize = 0;
  const AdapterPtr &Adapter = Context->getAdapter();
  Adapter->call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                             0, nullptr, &URDevicesSize);
  std::vector<ur_device_handle_t> URDevices(URDevicesSize /
                                            sizeof(ur_device_handle_t));
  Adapter->call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                             URDevicesSize, URDevices.data(),
                                             nullptr);
  std::string Log = "The program was built for " +
                    std::to_string(URDevices.size()) + " devices";
  for (ur_device_handle_t &Device : URDevices) {
    std::string DeviceBuildInfoString;
    size_t DeviceBuildInfoStrSize = 0;
    Adapter->call<UrApiKind::urProgramGetBuildInfo>(
        Program, Device, UR_PROGRAM_BUILD_INFO_LOG, 0, nullptr,
        &DeviceBuildInfoStrSize);
    if (DeviceBuildInfoStrSize > 0) {
      std::vector<char> DeviceBuildInfo(DeviceBuildInfoStrSize);
      Adapter->call<UrApiKind::urProgramGetBuildInfo>(
          Program, Device, UR_PROGRAM_BUILD_INFO_LOG, DeviceBuildInfoStrSize,
          DeviceBuildInfo.data(), nullptr);
      DeviceBuildInfoString = std::string(DeviceBuildInfo.data());
    }

    std::string DeviceNameString;
    size_t DeviceNameStrSize = 0;
    Adapter->call<UrApiKind::urDeviceGetInfo>(Device, UR_DEVICE_INFO_NAME, 0,
                                              nullptr, &DeviceNameStrSize);
    if (DeviceNameStrSize > 0) {
      std::vector<char> DeviceName(DeviceNameStrSize);
      Adapter->call<UrApiKind::urDeviceGetInfo>(Device, UR_DEVICE_INFO_NAME,
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
// sycl_device_binary_struct can be created for each of them.
static bool loadDeviceLib(const ContextImplPtr Context, const char *Name,
                          ur_program_handle_t &Prog) {
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
    throw exception(make_error_code(errc::build),
                    "Unhandled (new?) device library extension");
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
    throw exception(make_error_code(errc::build),
                    "Unhandled (new?) device library extension");
  return Ext->second;
}

static ur_result_t doCompile(const AdapterPtr &Adapter,
                             ur_program_handle_t Program, uint32_t NumDevs,
                             ur_device_handle_t *Devs, ur_context_handle_t Ctx,
                             const char *Opts) {
  // Try to compile with given devices, fall back to compiling with the program
  // context if unsupported by the adapter
  auto Result = Adapter->call_nocheck<UrApiKind::urProgramCompileExp>(
      Program, NumDevs, Devs, Opts);
  if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return Adapter->call_nocheck<UrApiKind::urProgramCompile>(Ctx, Program,
                                                              Opts);
  }
  return Result;
}

static ur_program_handle_t
loadDeviceLibFallback(const ContextImplPtr Context, DeviceLibExt Extension,
                      std::vector<ur_device_handle_t> &Devices,
                      bool UseNativeLib) {

  auto LibFileName = getDeviceLibFilename(Extension, UseNativeLib);
  auto LockedCache = Context->acquireCachedLibPrograms();
  auto &CachedLibPrograms = LockedCache.get();
  // Collect list of devices to compile the library for. Library was already
  // compiled for a device if there is a corresponding record in the per-context
  // cache.
  std::vector<ur_device_handle_t> DevicesToCompile;
  ur_program_handle_t URProgram = nullptr;
  assert(Devices.size() > 0 &&
         "At least one device is expected in the input vector");
  // Vector of devices that don't have the library cached.
  for (auto Dev : Devices) {
    auto CacheResult = CachedLibPrograms.emplace(
        std::make_pair(std::make_pair(Extension, Dev), nullptr));
    auto Cached = !CacheResult.second;
    if (!Cached) {
      DevicesToCompile.push_back(Dev);
    } else {
      auto CachedURProgram = CacheResult.first->second;
      assert(CachedURProgram && "If device lib UR program was cached then is "
                                "expected to be not a nullptr");
      assert(((URProgram && URProgram == CachedURProgram) || (!URProgram)) &&
             "All cached UR programs should be the same");
      if (!URProgram)
        URProgram = CachedURProgram;
    }
  }
  if (DevicesToCompile.empty())
    return URProgram;

  auto EraseProgramForDevices = [&]() {
    for (auto Dev : DevicesToCompile)
      CachedLibPrograms.erase(std::make_pair(Extension, Dev));
  };
  bool IsProgramCreated = !URProgram;

  // Create UR program for device lib if we don't have it yet.
  if (!URProgram && !loadDeviceLib(Context, LibFileName, URProgram)) {
    EraseProgramForDevices();
    throw exception(make_error_code(errc::build),
                    std::string("Failed to load ") + LibFileName);
  }

  // Insert URProgram into the cache for all devices that we compiled it for.
  // Retain UR program for each record in the cache.
  const AdapterPtr &Adapter = Context->getAdapter();

  // UR program handle is stored in the cache for each device that we compiled
  // it for. We have to retain UR program for each record in the cache. We need
  // to take into account that UR program creation makes its reference count to
  // be 1.
  size_t RetainCount =
      IsProgramCreated ? DevicesToCompile.size() - 1 : DevicesToCompile.size();
  for (size_t I = 0; I < RetainCount; ++I)
    Adapter->call<UrApiKind::urProgramRetain>(URProgram);

  for (auto Dev : DevicesToCompile)
    CachedLibPrograms[std::make_pair(Extension, Dev)] = URProgram;

  // TODO no spec constants are used in the std libraries, support in the future
  // Do not use compile options for library programs: it is not clear if user
  // options (image options) are supposed to be applied to library program as
  // well, and what actually happens to a SPIR-V program if we apply them.
  ur_result_t Error =
      doCompile(Adapter, URProgram, DevicesToCompile.size(),
                DevicesToCompile.data(), Context->getHandleRef(), "");
  if (Error != UR_RESULT_SUCCESS) {
    EraseProgramForDevices();
    throw detail::set_ur_error(
        exception(make_error_code(errc::build),
                  ProgramManager::getProgramBuildLog(URProgram, Context)),
        Error);
  }

  return URProgram;
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
      throw exception(make_error_code(errc::runtime),
                      std::string("Can't open file specified via ") +
                          UseSpvEnv + ": " + SpvFile);
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    std::unique_ptr<char[]> Data(new char[Size]);
    File.seekg(0);
    File.read(Data.get(), Size);
    File.close();
    if (!File.good())
      throw exception(make_error_code(errc::runtime),
                      std::string("read from ") + SpvFile +
                          std::string(" failed"));
    // No need for a mutex here since all access to these private fields is
    // blocked until the construction of the ProgramManager singleton is
    // finished.
    m_SpvFileImage =
        std::make_unique<DynRTDeviceBinaryImage>(std::move(Data), Size);

    if constexpr (DbgProgMgr > 0) {
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
  const sycl_device_binary_struct &RawImg = Image->getRawData();
  if ((strcmp(RawImg.DeviceTargetSpec,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
      (strcmp(RawImg.DeviceTargetSpec,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
      (strcmp(RawImg.DeviceTargetSpec,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0)) {
    throw sycl::exception(sycl::errc::feature_not_supported,
                          "Recompiling AOT image is not supported");
  }
}

const char *getArchName(const device &Device) {
  namespace syclex = sycl::ext::oneapi::experimental;
  auto Arch = Device.get_info<syclex::info::device::architecture>();
  switch (Arch) {
#define __SYCL_ARCHITECTURE(ARCH, VAL)                                         \
  case syclex::architecture::ARCH:                                             \
    return #ARCH;
#define __SYCL_ARCHITECTURE_ALIAS(ARCH, VAL)
#include <sycl/ext/oneapi/experimental/architectures.def>
#undef __SYCL_ARCHITECTURE
#undef __SYCL_ARCHITECTURE_ALIAS
  }
  return "unknown";
}

sycl_device_binary getRawImg(RTDeviceBinaryImage *Img) {
  return reinterpret_cast<sycl_device_binary>(
      const_cast<sycl_device_binary>(&Img->getRawData()));
}

template <typename StorageKey>
RTDeviceBinaryImage *getBinImageFromMultiMap(
    const std::unordered_multimap<StorageKey, RTDeviceBinaryImage *> &ImagesSet,
    const StorageKey &Key, const context &Context, const device &Device) {
  auto [ItBegin, ItEnd] = ImagesSet.equal_range(Key);
  if (ItBegin == ItEnd)
    return nullptr;

  // Here, we aim to select all the device images from the
  // [ItBegin, ItEnd) range that are AOT compiled for Device
  // (checked using info::device::architecture) or  JIT compiled.
  // This selection will then be passed to urDeviceSelectBinary
  // for final selection.
  std::string_view ArchName = getArchName(Device);
  std::vector<RTDeviceBinaryImage *> DeviceFilteredImgs;
  DeviceFilteredImgs.reserve(std::distance(ItBegin, ItEnd));
  for (auto It = ItBegin; It != ItEnd; ++It) {
    auto PropRange = It->second->getDeviceRequirements();
    auto PropIt =
        std::find_if(PropRange.begin(), PropRange.end(), [&](const auto &Prop) {
          return Prop->Name == std::string_view("compile_target");
        });
    auto AddImg = [&]() { DeviceFilteredImgs.push_back(It->second); };

    // Device image has no compile_target property, so it is JIT compiled.
    if (PropIt == PropRange.end()) {
      AddImg();
      continue;
    }

    // Device image has the compile_target property, so it is AOT compiled for
    // some device, check if that architecture is Device's architecture.
    auto CompileTargetByteArray = DeviceBinaryProperty(*PropIt).asByteArray();
    CompileTargetByteArray.dropBytes(8);
    std::string_view CompileTarget(
        reinterpret_cast<const char *>(&CompileTargetByteArray[0]),
        CompileTargetByteArray.size());
    // Note: there are no explicit targets for CPUs, so on x86_64,
    // intel_cpu_spr, and intel_cpu_gnr, we use a spir64_x86_64
    // compile target image.
    // TODO: When dedicated targets for CPU are added, (i.e.
    // -fsycl-targets=intel_cpu_spr etc.) remove this special
    // handling of CPU targets.
    if ((ArchName == CompileTarget) ||
        (CompileTarget == "spir64_x86_64" &&
         (ArchName == "x86_64" || ArchName == "intel_cpu_spr" ||
          ArchName == "intel_cpu_gnr"))) {
      AddImg();
    }
  }

  if (DeviceFilteredImgs.empty())
    return nullptr;

  std::vector<ur_device_binary_t> UrBinaries(DeviceFilteredImgs.size());
  for (uint32_t BinaryCount = 0; BinaryCount < DeviceFilteredImgs.size();
       BinaryCount++) {
    UrBinaries[BinaryCount].pDeviceTargetSpec = getUrDeviceTarget(
        getRawImg(DeviceFilteredImgs[BinaryCount])->DeviceTargetSpec);
  }

  uint32_t ImgInd = 0;
  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  getSyclObjImpl(Context)->getAdapter()->call<UrApiKind::urDeviceSelectBinary>(
      getSyclObjImpl(Device)->getHandleRef(), UrBinaries.data(),
      UrBinaries.size(), &ImgInd);
  return DeviceFilteredImgs[ImgInd];
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(const std::string &KernelName,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(\"" << KernelName << "\", "
              << getSyclObjImpl(Context).get() << ", "
              << getSyclObjImpl(Device).get() << ", "
              << JITCompilationIsRequired << ")\n";

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
      Img = getBinImageFromMultiMap(m_KernelIDs2BinImage, KernelId->second,
                                    Context, Device);
    } else {
      Img = getBinImageFromMultiMap(m_ServiceKernels, KernelName, Context,
                                    Device);
    }
  }

  // Decompress the image if it is compressed.
  CheckAndDecompressImage(Img);

  if (Img) {
    CheckJITCompilationForImage(Img, JITCompilationIsRequired);

    if constexpr (DbgProgMgr > 0) {
      std::cerr << "selected device image: " << &Img->getRawData() << "\n";
      Img->print();
    }
    return *Img;
  }

  throw exception(make_error_code(errc::runtime),
                  "No kernel named " + KernelName + " was found");
}

RTDeviceBinaryImage &ProgramManager::getDeviceImage(
    const std::unordered_set<RTDeviceBinaryImage *> &ImageSet,
    const context &Context, const device &Device,
    bool JITCompilationIsRequired) {
  assert(ImageSet.size() > 0);

  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(Custom SPV file "
              << getSyclObjImpl(Context).get() << ", "
              << getSyclObjImpl(Device).get() << ", "
              << JITCompilationIsRequired << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }

  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
  std::vector<sycl_device_binary> RawImgs(ImageSet.size());
  auto ImageIterator = ImageSet.begin();
  for (size_t i = 0; i < ImageSet.size(); i++, ImageIterator++)
    RawImgs[i] = reinterpret_cast<sycl_device_binary>(
        const_cast<sycl_device_binary>(&(*ImageIterator)->getRawData()));
  uint32_t ImgInd = 0;
  // Ask the native runtime under the given context to choose the device image
  // it prefers.

  std::vector<ur_device_binary_t> UrBinaries(RawImgs.size());
  for (uint32_t BinaryCount = 0; BinaryCount < RawImgs.size(); BinaryCount++) {
    UrBinaries[BinaryCount].pDeviceTargetSpec =
        getUrDeviceTarget(RawImgs[BinaryCount]->DeviceTargetSpec);
  }

  getSyclObjImpl(Context)->getAdapter()->call<UrApiKind::urDeviceSelectBinary>(
      getSyclObjImpl(Device)->getHandleRef(), UrBinaries.data(),
      UrBinaries.size(), &ImgInd);

  ImageIterator = ImageSet.begin();
  std::advance(ImageIterator, ImgInd);

  CheckJITCompilationForImage(*ImageIterator, JITCompilationIsRequired);

  if constexpr (DbgProgMgr > 0) {
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

static std::vector<ur_program_handle_t>
getDeviceLibPrograms(const ContextImplPtr Context,
                     std::vector<ur_device_handle_t> &Devices,
                     uint32_t DeviceLibReqMask) {
  std::vector<ur_program_handle_t> Programs;

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
  const bool fp64Support = std::all_of(
      Devices.begin(), Devices.end(), [&Context](ur_device_handle_t Device) {
        std::string DevExtList =
            Context->getPlatformImpl()
                ->getDeviceImpl(Device)
                ->get_device_info_string(
                    UrInfoCode<info::device::extensions>::value);
        return (DevExtList.npos != DevExtList.find("cl_khr_fp64"));
      });

  // Load a fallback library for an extension if the any device does not
  // support it.
  for (auto Device : Devices) {
    std::string DevExtList =
        Context->getPlatformImpl()
            ->getDeviceImpl(Device)
            ->get_device_info_string(
                UrInfoCode<info::device::extensions>::value);

    for (auto &Pair : RequiredDeviceLibExt) {
      DeviceLibExt Ext = Pair.first;
      bool &FallbackIsLoaded = Pair.second;

      if (FallbackIsLoaded) {
        continue;
      }

      if (!isDeviceLibRequired(Ext, DeviceLibReqMask)) {
        continue;
      }

      // Skip loading the fallback library that requires fp64 support if any
      // device in the list doesn't support fp64.
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
        Programs.push_back(loadDeviceLibFallback(Context, Ext, Devices,
                                                 /*UseNativeLib=*/false));
        FallbackIsLoaded = true;
      } else {
        // bfloat16 needs native library if device supports it
        if (Ext == DeviceLibExt::cl_intel_devicelib_bfloat16) {
          Programs.push_back(loadDeviceLibFallback(Context, Ext, Devices,
                                                   /*UseNativeLib=*/true));
          FallbackIsLoaded = true;
        }
      }
    }
  }
  return Programs;
}

// Check if device image is compressed.
static inline bool isDeviceImageCompressed(sycl_device_binary Bin) {

  auto currFormat = static_cast<ur::DeviceBinaryType>(Bin->Format);
  return currFormat == SYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE;
}

ProgramManager::ProgramPtr ProgramManager::build(
    ProgramPtr Program, const ContextImplPtr Context,
    const std::string &CompileOptions, const std::string &LinkOptions,
    std::vector<ur_device_handle_t> &Devices, uint32_t DeviceLibReqMask,
    const std::vector<ur_program_handle_t> &ExtraProgramsToLink,
    bool CreatedFromBinary) {

  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", "
              << VecToString(Devices) << ", " << std::hex << DeviceLibReqMask
              << std::dec << ", " << VecToString(ExtraProgramsToLink) << ", "
              << CreatedFromBinary << ")\n";
  }

  bool LinkDeviceLibs = (DeviceLibReqMask != 0);

  // TODO: this is a temporary workaround for GPU tests for ESIMD compiler.
  // We do not link with other device libraries, because it may fail
  // due to unrecognized SPIR-V format of those libraries.
  if (CompileOptions.find(std::string("-cmc")) != std::string::npos ||
      CompileOptions.find(std::string("-vc-codegen")) != std::string::npos)
    LinkDeviceLibs = false;

  std::vector<ur_program_handle_t> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Devices, DeviceLibReqMask);
  }

  static const char *ForceLinkEnv = std::getenv("SYCL_FORCE_LINK");
  static bool ForceLink = ForceLinkEnv && (*ForceLinkEnv == '1');

  const AdapterPtr &Adapter = Context->getAdapter();
  if (LinkPrograms.empty() && ExtraProgramsToLink.empty() && !ForceLink) {
    const std::string &Options = LinkOptions.empty()
                                     ? CompileOptions
                                     : (CompileOptions + " " + LinkOptions);
    ur_result_t Error = Adapter->call_nocheck<UrApiKind::urProgramBuildExp>(
        Program.get(), Devices.size(), Devices.data(), Options.c_str());
    if (Error == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Error = Adapter->call_nocheck<UrApiKind::urProgramBuild>(
          Context->getHandleRef(), Program.get(), Options.c_str());
    }

    if (Error != UR_RESULT_SUCCESS)
      throw detail::set_ur_error(
          exception(make_error_code(errc::build),
                    getProgramBuildLog(Program.get(), Context)),
          Error);

    return Program;
  }

  // Include the main program and compile/link everything together
  if (!CreatedFromBinary) {
    auto Res = doCompile(Adapter, Program.get(), Devices.size(), Devices.data(),
                         Context->getHandleRef(), CompileOptions.c_str());
    Adapter->checkUrResult<errc::build>(Res);
  }
  LinkPrograms.push_back(Program.get());

  for (ur_program_handle_t Prg : ExtraProgramsToLink) {
    if (!CreatedFromBinary) {
      auto Res = doCompile(Adapter, Prg, Devices.size(), Devices.data(),
                           Context->getHandleRef(), CompileOptions.c_str());
      Adapter->checkUrResult<errc::build>(Res);
    }
    LinkPrograms.push_back(Prg);
  }

  ur_program_handle_t LinkedProg = nullptr;
  auto doLink = [&] {
    auto Res = Adapter->call_nocheck<UrApiKind::urProgramLinkExp>(
        Context->getHandleRef(), Devices.size(), Devices.data(),
        LinkPrograms.size(), LinkPrograms.data(), LinkOptions.c_str(),
        &LinkedProg);
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter->call_nocheck<UrApiKind::urProgramLink>(
          Context->getHandleRef(), LinkPrograms.size(), LinkPrograms.data(),
          LinkOptions.c_str(), &LinkedProg);
    }
    return Res;
  };
  ur_result_t Error = doLink();
  if (Error == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
      Error == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
      Error == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    Context->getKernelProgramCache().reset();
    Error = doLink();
  }

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise. Release the original (user) program.
  Program.reset(LinkedProg);
  if (Error != UR_RESULT_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg, and throw.
      throw detail::set_ur_error(
          exception(make_error_code(errc::build),
                    getProgramBuildLog(LinkedProg, Context)),
          Error);
    }
    Adapter->checkUrResult(Error);
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

void ProgramManager::addImages(sycl_device_binaries DeviceBinary) {
  const bool DumpImages = std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile;
  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    sycl_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);
    const sycl_offload_entry EntriesB = RawImg->EntriesBegin;
    const sycl_offload_entry EntriesE = RawImg->EntriesEnd;
    // Treat the image as empty one
    if (EntriesB == EntriesE)
      continue;

    std::unique_ptr<RTDeviceBinaryImage> Img;
    if (isDeviceImageCompressed(RawImg))
#ifndef SYCL_RT_ZSTD_NOT_AVAIABLE
      Img = std::make_unique<CompressedRTDeviceBinaryImage>(RawImg);
#else
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Recieved a compressed device image, but "
                            "SYCL RT was built without ZSTD support."
                            "Aborting. ");
#endif
    else
      Img = std::make_unique<RTDeviceBinaryImage>(RawImg);

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
    for (const sycl_device_binary_property &ESProp :
         Img->getExportedSymbols()) {
      m_ExportedSymbolImages.insert({ESProp->Name, Img.get()});
    }

    // Record mapping between virtual function sets and device images
    for (const sycl_device_binary_property &VFProp :
         Img->getVirtualFunctions()) {
      std::string StrValue = DeviceBinaryProperty(VFProp).asCString();
      for (const auto &SetName : detail::split_string(StrValue, ','))
        m_VFSet2BinImage[SetName].insert(Img.get());
    }

    if (DumpImages) {
      const bool NeedsSequenceID = std::any_of(
          m_BinImg2KernelIDs.begin(), m_BinImg2KernelIDs.end(),
          [&](auto &CurrentImg) {
            return CurrentImg.first->getFormat() == Img->getFormat();
          });

      // Check if image is compressed, and decompress it before dumping.
      CheckAndDecompressImage(Img.get());

      dumpImage(*Img, NeedsSequenceID ? ++SequenceID : 0);
    }

    m_BinImg2KernelIDs[Img.get()].reset(new std::vector<kernel_id>);

    for (sycl_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
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
      if (m_ExportedSymbolImages.find(EntriesIt->name) !=
          m_ExportedSymbolImages.end())
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
      sycl_device_binary_property Prop = Img->getProperty("asanUsed");
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
      for (const sycl_device_binary_property &DeviceGlobal : DeviceGlobals) {
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
      for (const sycl_device_binary_property &HostPipe : HostPipes) {
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
  const sycl_device_binary_struct &RawImg = Img.getRawData();
  Fname += RawImg.DeviceTargetSpec;
  if (SequenceID)
    Fname += '_' + std::to_string(SequenceID);
  std::string Ext;

  ur::DeviceBinaryType Format = Img.getFormat();
  if (Format == SYCL_DEVICE_BINARY_TYPE_SPIRV)
    Ext = ".spv";
  else if (Format == SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE)
    Ext = ".bc";
  else
    Ext = ".bin";
  Fname += Ext;

  std::ofstream F(Fname, std::ios::binary);

  if (!F.is_open()) {
    throw exception(make_error_code(errc::runtime), "Can not write " + Fname);
  }
  Img.dump(F);
  F.close();
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
ProgramManager::getEliminatedKernelArgMask(ur_program_handle_t NativePrg,
                                           const std::string &KernelName) {
  // Bail out if there are no eliminated kernel arg masks in our images
  if (m_EliminatedKernelArgMasks.empty())
    return nullptr;

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    auto Range = NativePrograms.equal_range(NativePrg);
    for (auto ImgIt = Range.first; ImgIt != Range.second; ++ImgIt) {
      auto MapIt = m_EliminatedKernelArgMasks.find(ImgIt->second);
      if (MapIt == m_EliminatedKernelArgMasks.end())
        continue;
      auto ArgMaskMapIt = MapIt->second.find(KernelName);
      if (ArgMaskMapIt != MapIt->second.end())
        return &MapIt->second[KernelName];
    }
    if (Range.first != Range.second)
      return nullptr;
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
    return ((strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
            (strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
            (strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0));
  };

  // There are only two initial states so far - SPIRV which needs to be compiled
  // and linked and fully compiled(AOTed) binary

  const bool IsAOT = IsAOTBinary(BinImage->getRawData().DeviceTargetSpec);

  return IsAOT ? sycl::bundle_state::executable : sycl::bundle_state::input;
}

kernel_id ProgramManager::getSYCLKernelID(const std::string &KernelName) {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  auto KernelID = m_KernelName2KernelIDs.find(KernelName);
  if (KernelID == m_KernelName2KernelIDs.end())
    throw exception(make_error_code(errc::runtime),
                    "No kernel found with the specified name");

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

        // Check if device image is compressed and decompress it if needed
        CheckAndDecompressImage(BinImage);

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

device_image_plain
ProgramManager::compile(const device_image_plain &DeviceImage,
                        const std::vector<device> &Devs,
                        const property_list &) {

  // TODO: Extract compile options from property list once the Spec clarifies
  // how they can be passed.

  // TODO: Probably we could have cached compiled device images.
  const std::shared_ptr<device_image_impl> &InputImpl =
      getSyclObjImpl(DeviceImage);

  const AdapterPtr &Adapter =
      getSyclObjImpl(InputImpl->get_context())->getAdapter();

  // Device is not used when creating program from SPIRV, so passing only one
  // device is OK.
  ur_program_handle_t Prog = createURProgram(*InputImpl->get_bin_image_ref(),
                                             InputImpl->get_context(), Devs);

  if (InputImpl->get_bin_image_ref()->supportsSpecConstants())
    setSpecializationConstants(InputImpl, Prog, Adapter);

  DeviceImageImplPtr ObjectImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), InputImpl->get_context(), Devs,
      bundle_state::object, InputImpl->get_kernel_ids_ptr(), Prog,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  std::vector<ur_device_handle_t> URDevices;
  URDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    URDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

  // TODO: Handle zero sized Device list.
  std::string CompileOptions;
  applyCompileOptionsFromEnvironment(CompileOptions);
  appendCompileOptionsFromImage(
      CompileOptions, *(InputImpl->get_bin_image_ref()), Devs, Adapter);
  // Should always come last!
  appendCompileEnvironmentVariablesThatAppend(CompileOptions);
  ur_result_t Error = doCompile(
      Adapter, ObjectImpl->get_ur_program_ref(), Devs.size(), URDevices.data(),
      getSyclObjImpl(InputImpl->get_context()).get()->getHandleRef(),
      CompileOptions.c_str());
  if (Error != UR_RESULT_SUCCESS)
    throw sycl::exception(
        make_error_code(errc::build),
        getProgramBuildLog(ObjectImpl->get_ur_program_ref(),
                           getSyclObjImpl(ObjectImpl->get_context())));

  return createSyclObjFromImpl<device_image_plain>(ObjectImpl);
}

std::vector<device_image_plain>
ProgramManager::link(const device_image_plain &DeviceImage,
                     const std::vector<device> &Devs,
                     const property_list &PropList) {
  (void)PropList;

  std::vector<ur_program_handle_t> URPrograms;
  URPrograms.push_back(getSyclObjImpl(DeviceImage)->get_ur_program_ref());

  std::vector<ur_device_handle_t> URDevices;
  URDevices.reserve(Devs.size());
  for (const device &Dev : Devs)
    URDevices.push_back(getSyclObjImpl(Dev)->getHandleRef());

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
  const AdapterPtr &Adapter = ContextImpl->getAdapter();

  ur_program_handle_t LinkedProg = nullptr;
  auto doLink = [&] {
    auto Res = Adapter->call_nocheck<UrApiKind::urProgramLinkExp>(
        ContextImpl->getHandleRef(), URDevices.size(), URDevices.data(),
        URPrograms.size(), URPrograms.data(), LinkOptionsStr.c_str(),
        &LinkedProg);
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter->call_nocheck<UrApiKind::urProgramLink>(
          ContextImpl->getHandleRef(), URPrograms.size(), URPrograms.data(),
          LinkOptionsStr.c_str(), &LinkedProg);
    }
    return Res;
  };
  ur_result_t Error = doLink();
  if (Error == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
      Error == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
      Error == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    ContextImpl->getKernelProgramCache().reset();
    Error = doLink();
  }

  if (Error != UR_RESULT_SUCCESS) {
    if (LinkedProg) {
      const std::string ErrorMsg = getProgramBuildLog(LinkedProg, ContextImpl);
      throw sycl::exception(make_error_code(errc::build), ErrorMsg);
    }
    throw set_ur_error(exception(make_error_code(errc::build), "link() failed"),
                       Error);
  }

  std::shared_ptr<std::vector<kernel_id>> KernelIDs{new std::vector<kernel_id>};
  std::vector<unsigned char> NewSpecConstBlob;
  device_image_impl::SpecConstMapT NewSpecConstMap;

  std::shared_ptr<device_image_impl> DeviceImageImpl =
      getSyclObjImpl(DeviceImage);

  // Duplicates are not expected here, otherwise urProgramLink should fail
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
    const AdapterPtr &Adapter = ContextImpl->getAdapter();
    applyOptionsFromImage(CompileOpts, LinkOpts, Img, Devs, Adapter);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOpts);
    appendLinkEnvironmentVariablesThatAppend(LinkOpts);

    // Device is not used when creating program from SPIRV, so passing only one
    // device is OK.
    auto [NativePrg, DeviceCodeWasInCache] = getOrCreateURProgram(
        Img, {&Img}, Context, Devs, CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache &&
        InputImpl->get_bin_image_ref()->supportsSpecConstants())
      setSpecializationConstants(InputImpl, NativePrg, Adapter);

    UrFuncInfo<UrApiKind::urProgramRelease> programReleaseInfo;
    auto programRelease =
        programReleaseInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
    ProgramPtr ProgramManaged(NativePrg, programRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    if (Img.getFormat() == SYCL_DEVICE_BINARY_TYPE_SPIRV &&
        !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get())
      DeviceLibReqMask = getDeviceLibReqMask(Img);

    // TODO: Add support for dynamic linking with kernel bundles
    std::vector<ur_program_handle_t> ExtraProgramsToLink;
    std::vector<ur_device_handle_t> URDevices;
    for (auto Dev : Devs) {
      URDevices.push_back(getSyclObjImpl(Dev).get()->getHandleRef());
    }
    ProgramPtr BuiltProgram =
        build(std::move(ProgramManaged), ContextImpl, CompileOpts, LinkOpts,
              URDevices, DeviceLibReqMask, ExtraProgramsToLink);

    emitBuiltProgramInfo(BuiltProgram.get(), ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms.insert({BuiltProgram.get(), &Img});
    }

    ContextImpl->addDeviceGlobalInitializer(BuiltProgram.get(), Devs, &Img);

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache)
      PersistentDeviceCodeCache::putItemToDisc(Devs[0], {&Img}, SpecConsts,
                                               CompileOpts + LinkOpts,
                                               BuiltProgram.get());

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
  std::set<ur_device_handle_t> URDevicesSet;
  std::transform(Devs.begin(), Devs.end(),
                 std::inserter(URDevicesSet, URDevicesSet.begin()),
                 [](const device &Dev) {
                   return getSyclObjImpl(Dev).get()->getHandleRef();
                 });
  auto CacheKey = std::make_pair(std::make_pair(std::move(SpecConsts), ImgId),
                                 URDevicesSet);

  // CacheKey is captured by reference so when we overwrite it later we can
  // reuse this function.
  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  auto BuildResult = Cache.getOrBuild<errc::build>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");

  ur_program_handle_t ResProgram = BuildResult->Val;

  // Here we have multiple devices a program is built for, so add the program to
  // the cache for all subsets of provided list of devices.
  const AdapterPtr &Adapter = ContextImpl->getAdapter();
  auto CacheSubsets = [ResProgram, &Adapter]() {
    Adapter->call<UrApiKind::urProgramRetain>(ResProgram);
    return ResProgram;
  };

  if (URDevicesSet.size() > 1) {
    // emplace all subsets of the current set of devices into the cache.
    // Set of all devices is not included in the loop as it was already added
    // into the cache.
    for (int Mask = 1; Mask < (1 << URDevicesSet.size()) - 1; ++Mask) {
      std::set<ur_device_handle_t> Subset;
      int Index = 0;
      for (auto It = URDevicesSet.begin(); It != URDevicesSet.end();
           ++It, ++Index) {
        if (Mask & (1 << Index)) {
          Subset.insert(*It);
        }
      }
      // Change device in the cache key to reduce copying of spec const data.
      CacheKey.second = Subset;
      Cache.getOrBuild<errc::build>(GetCachedBuildF, CacheSubsets);
      // getOrBuild is not supposed to return nullptr
      assert(BuildResult != nullptr && "Invalid build result");
    }
  }

  // devive_image_impl shares ownership of PIProgram with, at least, program
  // cache. The ref counter will be descremented in the destructor of
  // device_image_impl
  Adapter->call<UrApiKind::urProgramRetain>(ResProgram);

  DeviceImageImplPtr ExecImpl = std::make_shared<detail::device_image_impl>(
      InputImpl->get_bin_image_ref(), Context, Devs, bundle_state::executable,
      InputImpl->get_kernel_ids_ptr(), ResProgram,
      InputImpl->get_spec_const_data_ref(),
      InputImpl->get_spec_const_blob_ref());

  return createSyclObjFromImpl<device_image_plain>(ExecImpl);
}

// When caching is enabled, the returned UrKernel will already have
// its ref count incremented.
std::tuple<ur_kernel_handle_t, std::mutex *, const KernelArgMask *>
ProgramManager::getOrCreateKernel(const context &Context,
                                  const std::string &KernelName,
                                  const property_list &PropList,
                                  ur_program_handle_t Program) {

  (void)PropList;

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto BuildF = [this, &Program, &KernelName, &Ctx] {
    ur_kernel_handle_t Kernel = nullptr;

    const AdapterPtr &Adapter = Ctx->getAdapter();
    Adapter->call<UrApiKind::urKernelCreate>(Program, KernelName.c_str(),
                                             &Kernel);

    // Only set UR_USM_INDIRECT_ACCESS if the platform can handle it.
    if (Ctx->getPlatformImpl()->supports_usm()) {
      bool EnableAccess = true;
      Adapter->call<UrApiKind::urKernelSetExecInfo>(
          Kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
          nullptr, &EnableAccess);
    }

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

  auto BuildResult = Cache.getOrBuild<errc::invalid>(GetCachedBuildF, BuildF);
  // getOrBuild is not supposed to return nullptr
  assert(BuildResult != nullptr && "Invalid build result");
  // If caching is enabled, one copy of the kernel handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // kernel.
  Ctx->getAdapter()->call<UrApiKind::urKernelRetain>(BuildResult->Val.first);
  return std::make_tuple(BuildResult->Val.first,
                         &(BuildResult->MBuildResultMutex),
                         BuildResult->Val.second);
}

ur_kernel_handle_t ProgramManager::getCachedMaterializedKernel(
    const std::string &KernelName,
    const std::vector<unsigned char> &SpecializationConsts) {
  if constexpr (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getCachedMaterializedKernel\n"
              << "KernelName: " << KernelName << "\n";

  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    if (auto KnownMaterializations = m_MaterializedKernels.find(KernelName);
        KnownMaterializations != m_MaterializedKernels.end()) {
      if constexpr (DbgProgMgr > 0)
        std::cerr << ">>> There are:" << KnownMaterializations->second.size()
                  << " materialized kernels.\n";
      if (auto Kernel =
              KnownMaterializations->second.find(SpecializationConsts);
          Kernel != KnownMaterializations->second.end()) {
        if constexpr (DbgProgMgr > 0)
          std::cerr << ">>> Kernel in the chache\n";
        return Kernel->second;
      }
    }
  }

  if constexpr (DbgProgMgr > 0)
    std::cerr << ">>> Kernel not in the chache\n";

  return nullptr;
}

ur_kernel_handle_t ProgramManager::getOrCreateMaterializedKernel(
    const RTDeviceBinaryImage &Img, const context &Context,
    const device &Device, const std::string &KernelName,
    const std::vector<unsigned char> &SpecializationConsts) {
  // Check if we already have the kernel in the cache.
  if constexpr (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getOrCreateMaterializedKernel\n"
              << "KernelName: " << KernelName << "\n";

  if (auto CachedKernel =
          getCachedMaterializedKernel(KernelName, SpecializationConsts))
    return CachedKernel;

  if constexpr (DbgProgMgr > 0)
    std::cerr << ">>> Adding the kernel to the cache.\n";
  auto Program = createURProgram(Img, Context, {Device});
  auto DeviceImpl = detail::getSyclObjImpl(Device);
  auto &Adapter = DeviceImpl->getAdapter();
  UrFuncInfo<UrApiKind::urProgramRelease> programReleaseInfo;
  auto programRelease =
      programReleaseInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  ProgramPtr ProgramManaged(Program, programRelease);

  std::string CompileOpts;
  std::string LinkOpts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);
  // No linking of extra programs reqruired.
  std::vector<ur_program_handle_t> ExtraProgramsToLink;
  std::vector<ur_device_handle_t> Devs = {DeviceImpl->getHandleRef()};
  auto BuildProgram =
      build(std::move(ProgramManaged), detail::getSyclObjImpl(Context),
            CompileOpts, LinkOpts, Devs,
            /*For non SPIR-V devices DeviceLibReqdMask is always 0*/ 0,
            ExtraProgramsToLink);
  ur_kernel_handle_t UrKernel{nullptr};
  Adapter->call<errc::kernel_not_supported, UrApiKind::urKernelCreate>(
      BuildProgram.get(), KernelName.c_str(), &UrKernel);
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    m_MaterializedKernels[KernelName][SpecializationConsts] = UrKernel;
  }

  return UrKernel;
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
    auto &PropRange = Img.getDeviceRequirements();
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
  auto WorkGroupNumDim = getPropIt("work_group_num_dim");

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

  if (JointMatrixPropIt) {
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

  if (JointMatrixMadPropIt) {
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

    size_t UserProvidedNumDims = 0;
    if (WorkGroupNumDim) {
      // We know the dimensions have been padded to 3, make sure that the pad
      // value is always set to 1 and record the number of dimensions specified
      // by the user.
      UserProvidedNumDims =
          DeviceBinaryProperty(*(WorkGroupNumDim.value())).asUint32();
#ifndef NDEBUG
      for (unsigned i = UserProvidedNumDims; i < 3; ++i)
        assert(ReqdWGSizeVec[i] == 1 &&
               "Incorrect padding in required work-group size metadata.");
#endif // NDEBUG
    } else {
      UserProvidedNumDims = Dims;
    }

    if (NDRDesc.Dims != 0 && NDRDesc.Dims != UserProvidedNumDims)
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

extern "C" void __sycl_register_lib(sycl_device_binaries desc) {
  sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(sycl_device_binaries desc) {
  (void)desc;
  // TODO implement the function
}
