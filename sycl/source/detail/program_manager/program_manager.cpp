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

static constexpr int DbgProgMgr = 0;

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

/// This function enables ITT annotations in SPIR-V module by setting
/// a specialization constant if INTEL_LIBITTNOTIFY64 env variable is set.
static void enableITTAnnotationsIfNeeded(const ur_program_handle_t &Prog,
                                         adapter_impl &Adapter) {
  if (SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::get() != nullptr) {
    constexpr char SpecValue = 1;
    ur_specialization_constant_info_t SpecConstInfo = {
        ITTSpecConstId, sizeof(char), &SpecValue};
    Adapter.call<UrApiKind::urProgramSetSpecializationConstants>(
        Prog, 1u, &SpecConstInfo);
  }
}

ProgramManager &ProgramManager::getInstance() {
  return GlobalHandler::instance().getProgramManager();
}

static Managed<ur_program_handle_t>
createBinaryProgram(context_impl &Context, devices_range Devices,
                    const uint8_t **Binaries, size_t *Lengths,
                    const std::vector<ur_program_metadata_t> &Metadata) {
  assert(!Devices.empty() && "No devices provided for program creation");

  adapter_impl &Adapter = Context.getAdapter();
  Managed<ur_program_handle_t> Program{Adapter};
  auto DeviceHandles = Devices.to<std::vector<ur_device_handle_t>>();
  ur_program_properties_t Properties = {};
  Properties.stype = UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES;
  Properties.pNext = nullptr;
  Properties.count = Metadata.size();
  Properties.pMetadatas = Metadata.data();

  Adapter.call<UrApiKind::urProgramCreateWithBinary>(
      Context.getHandleRef(), DeviceHandles.size(), DeviceHandles.data(),
      Lengths, Binaries, &Properties, &Program);

  return Program;
}

static Managed<ur_program_handle_t>
createSpirvProgram(context_impl &Context, const unsigned char *Data,
                   size_t DataLen) {
  adapter_impl &Adapter = Context.getAdapter();
  Managed<ur_program_handle_t> Program{Adapter};
  Adapter.call<UrApiKind::urProgramCreateWithIL>(Context.getHandleRef(), Data,
                                                 DataLen, nullptr, &Program);
  return Program;
}

// TODO replace this with a new UR API function
static bool isDeviceBinaryTypeSupported(context_impl &ContextImpl,
                                        ur::DeviceBinaryType Format) {
  // All formats except SYCL_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != SYCL_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  const backend ContextBackend = ContextImpl.getBackend();

  // The CUDA backend cannot use SPIR-V
  if (ContextBackend == backend::ext_oneapi_cuda)
    return false;

  devices_range Devices = ContextImpl.getDevices();

  // Program type is SPIR-V, so we need a device compiler to do JIT.
  if (!all_of(Devices, [](device_impl &D) {
        return D.get_info<info::device::is_compiler_available>();
      }))
    return false;

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (ContextBackend == backend::opencl) {
    std::string ver =
        ContextImpl.getPlatformImpl().get_info<info::platform::version>();
    if (ver.find("OpenCL 1.0") == std::string::npos &&
        ver.find("OpenCL 1.1") == std::string::npos &&
        ver.find("OpenCL 1.2") == std::string::npos &&
        ver.find("OpenCL 2.0") == std::string::npos)
      return true;
  }

  // We need cl_khr_il_program extension to be present
  // and we can call clCreateProgramWithILKHR using the extension
  return all_of(Devices, [](device_impl &D) {
    return D.has_extension("cl_khr_il_program");
  });
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

// The string produced by this function might be localized, with commas and
// periods inserted. Presently, it is used only for user facing error output.
[[maybe_unused]] auto VecToString = [](auto &Vec) -> std::string {
  std::ostringstream Out;
  Out << "{";
  for (auto Elem : Vec)
    Out << Elem << " ";
  Out << "}";
  return Out.str();
};

Managed<ur_program_handle_t>
ProgramManager::createURProgram(const RTDeviceBinaryImage &Img,
                                context_impl &ContextImpl,
                                devices_range Devices) {
  if constexpr (DbgProgMgr > 0) {
    auto URDevices = Devices.to<std::vector<ur_device_handle_t>>();
    std::cerr << ">>> ProgramManager::createPIProgram(" << &Img << ", "
              << ContextImpl.get() << ", " << VecToString(URDevices) << ")\n";
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

  if (!isDeviceBinaryTypeSupported(ContextImpl, Format))
    throw sycl::exception(
        sycl::errc::feature_not_supported,
        "SPIR-V online compilation is not supported in this context");

  // Get program metadata from properties
  const auto &ProgMetadata = Img.getProgramMetadataUR();

  // Load the image
  std::vector<const uint8_t *> Binaries(
      Devices.size(), const_cast<uint8_t *>(RawImg.BinaryStart));
  std::vector<size_t> Lengths(Devices.size(), ImgSize);
  Managed<ur_program_handle_t> Res =
      Format == SYCL_DEVICE_BINARY_TYPE_SPIRV
          ? createSpirvProgram(ContextImpl, RawImg.BinaryStart, ImgSize)
          : createBinaryProgram(ContextImpl, Devices, Binaries.data(),
                                Lengths.data(), ProgMetadata);

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    // associate the UR program with the image it was created for
    NativePrograms.insert({Res, {ContextImpl.shared_from_this(), &Img}});
  }

  ContextImpl.addDeviceGlobalInitializer(Res, Devices, &Img);

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
                                          devices_range Devs, adapter_impl &) {
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

  const platform_impl &PlatformImpl = Devs.front().getPlatformImpl();

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
           std::all_of(Devs.begin(), Devs.end(), [&](device_impl &Dev) {
             return &Dev.getPlatformImpl() == &PlatformImpl;
           }));
    const char *backend_option = nullptr;
    // Empty string is returned in backend_option when no appropriate backend
    // option is available for a given frontend option.
    PlatformImpl.getBackendOption(optLevelStr, &backend_option);
    if (backend_option && backend_option[0] != '\0') {
      if (!CompileOpts.empty())
        CompileOpts += " ";
      CompileOpts += std::string(backend_option);
    }
  }
  bool IsIntelGPU =
      (PlatformImpl.getBackend() == backend::ext_oneapi_level_zero ||
       PlatformImpl.getBackend() == backend::opencl) &&
      std::all_of(Devs.begin(), Devs.end(), [](device_impl &Dev) {
        return Dev.is_gpu() &&
               Dev.get_info<info::device::vendor_id>() == 0x8086;
      });
  if (!CompileOptsEnv) {
    static const char *TargetCompileFast = "-ftarget-compile-fast";
    if (auto Pos = CompileOpts.find(TargetCompileFast);
        Pos != std::string::npos) {
      const char *BackendOption = nullptr;
      if (IsIntelGPU)
        PlatformImpl.getBackendOption(TargetCompileFast, &BackendOption);
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
          std::all_of(Devs.begin(), Devs.end(), [&](device_impl &Dev) {
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
      CompileOpts = std::move(NewCompileOpts);
      OptPos = CompileOpts.find(TargetRegisterAllocMode);
    }
    constexpr std::string_view ReplaceOpts[] = {"-foffload-fp32-prec-div",
                                                "-foffload-fp32-prec-sqrt"};
    for (const std::string_view Opt : ReplaceOpts) {
      if (auto Pos = CompileOpts.find(Opt); Pos != std::string::npos) {
        const char *BackendOption = nullptr;
        PlatformImpl.getBackendOption(std::string(Opt).c_str(), &BackendOption);
        CompileOpts.replace(Pos, Opt.length(), BackendOption);
      }
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
                                  devices_range Devices,
                                  adapter_impl &Adapter) {
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

std::pair<Managed<ur_program_handle_t>, bool>
ProgramManager::getOrCreateURProgram(
    const RTDeviceBinaryImage &MainImg,
    const std::vector<const RTDeviceBinaryImage *> &AllImages,
    context_impl &ContextImpl, devices_range Devices,
    const std::string &CompileAndLinkOptions, SerializedObj SpecConsts) {
  Managed<ur_program_handle_t> NativePrg;

  // Get binaries for each device (1:1 correpsondence with input Devices).
  auto Binaries = PersistentDeviceCodeCache::getItemFromDisc(
      Devices, AllImages, SpecConsts, CompileAndLinkOptions);
  if (Binaries.empty())
    return {createURProgram(MainImg, ContextImpl, Devices), false};

  std::vector<const uint8_t *> BinPtrs;
  std::vector<size_t> Lengths;
  for (auto &Bin : Binaries) {
    Lengths.push_back(Bin.size());
    BinPtrs.push_back(reinterpret_cast<const uint8_t *>(Bin.data()));
  }

  // Get program metadata from properties
  std::vector<ur_program_metadata_t> ProgMetadataVector;
  for (const RTDeviceBinaryImage *Img : AllImages) {
    auto &ImgProgMetadata = Img->getProgramMetadataUR();
    ProgMetadataVector.insert(ProgMetadataVector.end(), ImgProgMetadata.begin(),
                              ImgProgMetadata.end());
  }
  return {createBinaryProgram(ContextImpl, Devices, BinPtrs.data(),
                              Lengths.data(), ProgMetadataVector),
          true};
}

/// Emits information about built programs if the appropriate contitions are
/// met, namely when SYCL_RT_WARNING_LEVEL is greater than or equal to 2.
static void emitBuiltProgramInfo(const ur_program_handle_t &Prog,
                                 context_impl &Context) {
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

static bool compatibleWithDevice(const RTDeviceBinaryImage *BinImage,
                                 const device_impl &DeviceImpl) {
  adapter_impl &Adapter = DeviceImpl.getAdapter();

  const ur_device_handle_t &URDeviceHandle = DeviceImpl.getHandleRef();

  // Call urDeviceSelectBinary with only one image to check if an image is
  // compatible with implementation. The function returns invalid index if no
  // device images are compatible.
  uint32_t SuitableImageID = std::numeric_limits<uint32_t>::max();
  const sycl_device_binary_struct &DevBin = BinImage->getRawData();

  ur_device_binary_t UrBinary{};
  UrBinary.pDeviceTargetSpec = getUrDeviceTarget(DevBin.DeviceTargetSpec);

  ur_result_t Error = Adapter.call_nocheck<UrApiKind::urDeviceSelectBinary>(
      URDeviceHandle, &UrBinary,
      /*num bin images = */ 1u, &SuitableImageID);
  if (Error != UR_RESULT_SUCCESS && Error != UR_RESULT_ERROR_INVALID_BINARY)
    throw detail::set_ur_error(exception(make_error_code(errc::runtime),
                                         "Invalid binary image or device"),
                               Error);

  return (0 == SuitableImageID);
}

// Check if the device image is a BF16 devicelib image.
bool ProgramManager::isBfloat16DeviceImage(
    const RTDeviceBinaryImage *BinImage) {
  // SYCL devicelib image.
  if ((m_Bfloat16DeviceLibImages[0].get() == BinImage) ||
      m_Bfloat16DeviceLibImages[1].get() == BinImage)
    return true;

  return false;
}

// Check if device natively support BF16 conversion and accordingly
// decide whether to use fallback or native BF16 devicelib image.
bool ProgramManager::shouldBF16DeviceImageBeUsed(
    const RTDeviceBinaryImage *BinImage, const device_impl &DeviceImpl) {
  // Decide whether a devicelib image should be used.
  int Bfloat16DeviceLibVersion = -1;
  if (m_Bfloat16DeviceLibImages[0].get() == BinImage)
    Bfloat16DeviceLibVersion = 0;
  else if (m_Bfloat16DeviceLibImages[1].get() == BinImage)
    Bfloat16DeviceLibVersion = 1;

  if (Bfloat16DeviceLibVersion != -1) {
    // Currently, only bfloat conversion devicelib are supported, so the prop
    // DeviceLibMeta are only used to represent fallback or native version.
    // For bfloat16 conversion devicelib, we have fallback and native version.
    // The native should be used on platform which supports native bfloat16
    // conversion capability and fallback version should be used on all other
    // platforms. The native bfloat16 capability can be queried via extension.
    // TODO: re-design the encode of the devicelib metadata if we must support
    // more devicelib images in this way.
    enum { DEVICELIB_FALLBACK = 0, DEVICELIB_NATIVE };
    ur_bool_t NativeBF16Supported = false;
    ur_result_t CallSuccessful =
        DeviceImpl.getAdapter().call_nocheck<UrApiKind::urDeviceGetInfo>(
            DeviceImpl.getHandleRef(),
            UR_DEVICE_INFO_BFLOAT16_CONVERSIONS_NATIVE, sizeof(ur_bool_t),
            &NativeBF16Supported, nullptr);
    if (CallSuccessful != UR_RESULT_SUCCESS) {
      // If backend query is not successful, we will use fallback bfloat16
      // device library for safety.
      return Bfloat16DeviceLibVersion == DEVICELIB_FALLBACK;
    } else
      return NativeBF16Supported ==
             (Bfloat16DeviceLibVersion == DEVICELIB_NATIVE);
  }

  return false;
}

static bool checkLinkingSupport(const device_impl &DeviceImpl,
                                const RTDeviceBinaryImage &Img) {
  const char *Target = Img.getRawData().DeviceTargetSpec;
  // TODO replace with extension checks once implemented in UR.
  if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0) {
    return true;
  }
  if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) {
    return DeviceImpl.is_gpu() && DeviceImpl.getBackend() == backend::opencl;
  }
  return false;
}

std::set<const RTDeviceBinaryImage *>
ProgramManager::collectDeviceImageDeps(const RTDeviceBinaryImage &Img,
                                       const device_impl &Dev,
                                       bool ErrorOnUnresolvableImport) {
  // TODO collecting dependencies for virtual functions and imported symbols
  // should be combined since one can lead to new unresolved dependencies for
  // the other.
  std::set<const RTDeviceBinaryImage *> DeviceImagesToLink =
      collectDependentDeviceImagesForVirtualFunctions(Img, Dev);

  std::set<const RTDeviceBinaryImage *> ImageDeps =
      collectDeviceImageDepsForImportedSymbols(Img, Dev,
                                               ErrorOnUnresolvableImport);
  DeviceImagesToLink.insert(ImageDeps.begin(), ImageDeps.end());
  return DeviceImagesToLink;
}

static inline void
CheckAndDecompressImage([[maybe_unused]] const RTDeviceBinaryImage *Img) {
#ifdef SYCL_RT_ZSTD_AVAILABLE
  if (auto CompImg = dynamic_cast<const CompressedRTDeviceBinaryImage *>(Img))
    if (CompImg->IsCompressed())
      const_cast<CompressedRTDeviceBinaryImage *>(CompImg)->Decompress();
#endif
}

std::set<const RTDeviceBinaryImage *>
ProgramManager::collectDeviceImageDepsForImportedSymbols(
    const RTDeviceBinaryImage &MainImg, const device_impl &Dev,
    bool ErrorOnUnresolvableImport) {
  std::set<const RTDeviceBinaryImage *> DeviceImagesToLink;
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
      const RTDeviceBinaryImage *Img = It->second;

      if (!doesDevSupportDeviceRequirements(Dev, *Img) ||
          !compatibleWithDevice(Img, Dev))
        continue;

      // If the image is a BF16 device image, we need to check if it
      // should be used for this device.
      if (isBfloat16DeviceImage(Img) && !shouldBF16DeviceImageBeUsed(Img, Dev))
        continue;

      // If any of the images is compressed, we need to decompress it
      // and then check if the format matches.
      if (Format == SYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE ||
          Img->getFormat() == SYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE) {
        CheckAndDecompressImage(&MainImg);
        CheckAndDecompressImage(Img);
        Format = MainImg.getFormat();
      }
      // Skip this image if its format differs from the main image.
      if (Img->getFormat() != Format)
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
    if (ErrorOnUnresolvableImport && !Found)
      throw sycl::exception(make_error_code(errc::build),
                            "No device image found for external symbol " +
                                Symbol);
  }
  DeviceImagesToLink.erase(&MainImg);
  return DeviceImagesToLink;
}

std::set<const RTDeviceBinaryImage *>
ProgramManager::collectDependentDeviceImagesForVirtualFunctions(
    const RTDeviceBinaryImage &Img, const device_impl &Dev) {
  // If virtual functions are used in a program, then we need to link several
  // device images together to make sure that vtable pointers stored in
  // objects are valid between different kernels (which could be in different
  // device images).
  std::set<const RTDeviceBinaryImage *> DeviceImagesToLink;
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

  if (!WorkList.empty()) {
    // Guard read access to m_VFSet2BinImage:
    // TODO: a better solution should be sought in the future, i.e. a different
    // mutex than m_KernelIDsMutex, check lock check pattern, etc.
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

    while (!WorkList.empty()) {
      std::string SetName = WorkList.front();
      WorkList.pop();

      // There could be more than one device image that uses the same set
      // of virtual functions, or provides virtual funtions from the same
      // set.
      for (const RTDeviceBinaryImage *BinImage : m_VFSet2BinImage.at(SetName)) {
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

        // TODO: Complete this part about handling of incompatible device
        // images. If device image uses the same virtual function set, then we
        // only link it if it is compatible. However, if device image provides
        // virtual function set and it is incompatible, then we should link its
        // "dummy" version to avoid link errors about unresolved external
        // symbols.
        if (doesDevSupportDeviceRequirements(Dev, *BinImage))
          DeviceImagesToLink.insert(BinImage);
      }
    }
  }

  // We may have inserted the original image into the list as well, because it
  // is also a part of m_VFSet2BinImage map. No need to to return it to avoid
  // passing it twice to link call later.
  DeviceImagesToLink.erase(&Img);

  return DeviceImagesToLink;
}

static void setSpecializationConstants(device_image_impl &InputImpl,
                                       ur_program_handle_t Prog,
                                       adapter_impl &Adapter) {
  std::lock_guard<std::mutex> Lock{InputImpl.get_spec_const_data_lock()};
  const std::map<std::string, std::vector<device_image_impl::SpecConstDescT>>
      &SpecConstData = InputImpl.get_spec_const_data_ref();
  const SerializedObj &SpecConsts = InputImpl.get_spec_const_blob_ref();

  // Set all specialization IDs from descriptors in the input device image.
  for (const auto &[SpecConstNames, SpecConstDescs] : SpecConstData) {
    std::ignore = SpecConstNames;
    for (const device_image_impl::SpecConstDescT &SpecIDDesc : SpecConstDescs) {
      if (SpecIDDesc.IsSet) {
        ur_specialization_constant_info_t SpecConstInfo = {
            SpecIDDesc.ID, SpecIDDesc.Size,
            SpecConsts.data() + SpecIDDesc.BlobOffset};
        Adapter.call<UrApiKind::urProgramSetSpecializationConstants>(
            Prog, 1u, &SpecConstInfo);
      }
    }
  }
}

// When caching is enabled, the returned UrProgram will already have
// its ref count incremented.
Managed<ur_program_handle_t> ProgramManager::getBuiltURProgram(
    context_impl &ContextImpl, device_impl &DeviceImpl,
    KernelNameStrRefT KernelName, const NDRDescT &NDRDesc) {
  device_impl *RootDevImpl;
  ur_bool_t MustBuildOnSubdevice = true;

  // Check if we can optimize program builds for sub-devices by using a program
  // built for the root device
  if (!DeviceImpl.isRootDevice()) {
    RootDevImpl = &DeviceImpl;
    while (!RootDevImpl->isRootDevice()) {
      device_impl &ParentDev = *detail::getSyclObjImpl(
          RootDevImpl->get_info<info::device::parent_device>());
      // Sharing is allowed within a single context only
      if (!ContextImpl.hasDevice(ParentDev))
        break;
      RootDevImpl = &ParentDev;
    }

    ContextImpl.getAdapter().call<UrApiKind::urDeviceGetInfo>(
        RootDevImpl->getHandleRef(), UR_DEVICE_INFO_BUILD_ON_SUBDEVICE,
        sizeof(ur_bool_t), &MustBuildOnSubdevice, nullptr);
  }

  device_impl &RootOrSubDevImpl =
      MustBuildOnSubdevice == true ? DeviceImpl : *RootDevImpl;

  const RTDeviceBinaryImage &Img =
      getDeviceImage(KernelName, ContextImpl, RootOrSubDevImpl);

  // Check that device supports all aspects used by the kernel
  if (auto exception =
          checkDevSupportDeviceRequirements(RootOrSubDevImpl, Img, NDRDesc))
    throw *exception;

  std::set<const RTDeviceBinaryImage *> DeviceImagesToLink =
      collectDeviceImageDeps(Img, {RootOrSubDevImpl});

  // Decompress all DeviceImagesToLink
  for (const RTDeviceBinaryImage *BinImg : DeviceImagesToLink)
    CheckAndDecompressImage(BinImg);

  std::vector<const RTDeviceBinaryImage *> AllImages;
  AllImages.reserve(DeviceImagesToLink.size() + 1);
  AllImages.push_back(&Img);
  std::copy(DeviceImagesToLink.begin(), DeviceImagesToLink.end(),
            std::back_inserter(AllImages));

  return getBuiltURProgram(std::move(AllImages), ContextImpl,
                           {RootOrSubDevImpl});
}

Managed<ur_program_handle_t>
ProgramManager::getBuiltURProgram(const BinImgWithDeps &ImgWithDeps,
                                  context_impl &ContextImpl, devices_range Devs,
                                  const DevImgPlainWithDeps *DevImgWithDeps,
                                  const SerializedObj &SpecConsts) {
  std::string CompileOpts;
  std::string LinkOpts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);
  auto BuildF = [this, &ImgWithDeps, &DevImgWithDeps, &ContextImpl, &Devs,
                 &CompileOpts, &LinkOpts, &SpecConsts] {
    adapter_impl &Adapter = ContextImpl.getAdapter();
    const RTDeviceBinaryImage &MainImg = *ImgWithDeps.getMain();
    applyOptionsFromImage(CompileOpts, LinkOpts, MainImg, Devs, Adapter);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOpts);
    appendLinkEnvironmentVariablesThatAppend(LinkOpts);

    auto [NativePrg, DeviceCodeWasInCache] =
        getOrCreateURProgram(MainImg, ImgWithDeps.getAll(), ContextImpl, Devs,
                             CompileOpts + LinkOpts, SpecConsts);

    if (!DeviceCodeWasInCache && MainImg.supportsSpecConstants()) {
      enableITTAnnotationsIfNeeded(NativePrg, Adapter);
      if (DevImgWithDeps)
        setSpecializationConstants(*getSyclObjImpl(DevImgWithDeps->getMain()),
                                   NativePrg, Adapter);
    }

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs (after AOT compilation or read from persitent
    // cache) are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    bool UseDeviceLibs = !DeviceCodeWasInCache &&
                         MainImg.getFormat() == SYCL_DEVICE_BINARY_TYPE_SPIRV &&
                         !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get();
    if (UseDeviceLibs)
      DeviceLibReqMask = getDeviceLibReqMask(MainImg);

    std::vector<Managed<ur_program_handle_t>> ProgramsToLink;
    // If we had a program in cache, then it should have been the fully linked
    // program already.
    if (!DeviceCodeWasInCache) {
      assert(!DevImgWithDeps ||
             DevImgWithDeps->getAll().size() == ImgWithDeps.getAll().size());
      // Oth image is the main one and has been handled, skip it.
      for (std::size_t I = 1; I < ImgWithDeps.getAll().size(); ++I) {
        const RTDeviceBinaryImage *BinImg = ImgWithDeps.getAll()[I];
        if (UseDeviceLibs)
          DeviceLibReqMask |= getDeviceLibReqMask(*BinImg);

        Managed<ur_program_handle_t> NativePrg =
            createURProgram(*BinImg, ContextImpl, Devs);

        if (BinImg->supportsSpecConstants()) {
          enableITTAnnotationsIfNeeded(NativePrg, Adapter);
          if (DevImgWithDeps)
            setSpecializationConstants(
                *getSyclObjImpl(DevImgWithDeps->getAll()[I]), NativePrg,
                Adapter);
        }
        ProgramsToLink.push_back(std::move(NativePrg));
      }
    }

    auto URDevices = Devs.to<std::vector<ur_device_handle_t>>();

    Managed<ur_program_handle_t> BuiltProgram =
        build(std::move(NativePrg), ContextImpl, CompileOpts, LinkOpts,
              URDevices, DeviceLibReqMask, ProgramsToLink,
              /*CreatedFromBinary*/ MainImg.getFormat() !=
                  SYCL_DEVICE_BINARY_TYPE_SPIRV);

    // Those extra programs won't be used anymore, just the final
    // linked result:
    ProgramsToLink.clear();
    emitBuiltProgramInfo(BuiltProgram, ContextImpl);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      // NativePrograms map does not intend to keep reference to program handle,
      // so keys in the map can be invalid (reference count went to zero and the
      // underlying program disposed of). Protecting from incorrect values by
      // removal of map entries with same handle (obviously invalid entries).
      std::ignore = NativePrograms.erase(BuiltProgram);
      for (const RTDeviceBinaryImage *Img : ImgWithDeps) {
        NativePrograms.insert(
            {BuiltProgram, {ContextImpl.shared_from_this(), Img}});
      }
    }

    ContextImpl.addDeviceGlobalInitializer(BuiltProgram, Devs, &MainImg);

    // Save program to persistent cache if it is not there
    if (!DeviceCodeWasInCache) {
      PersistentDeviceCodeCache::putItemToDisc(
          Devs, ImgWithDeps.getAll(), SpecConsts, CompileOpts + LinkOpts,
          BuiltProgram);
    }

    return BuiltProgram;
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get())
    return BuildF();

  uint32_t ImgId = ImgWithDeps.getMain()->getImageID();
  std::set<ur_device_handle_t> URDevicesSet;
  std::transform(Devs.begin(), Devs.end(),
                 std::inserter(URDevicesSet, URDevicesSet.begin()),
                 [](device_impl &Dev) { return Dev.getHandleRef(); });
  auto CacheKey =
      std::make_pair(std::make_pair(SpecConsts, ImgId), URDevicesSet);

  KernelProgramCache &Cache = ContextImpl.getKernelProgramCache();
  auto GetCachedBuildF = [&Cache, &CacheKey]() {
    return Cache.getOrInsertProgram(CacheKey);
  };

  auto EvictFunc = [&Cache, &CacheKey](ur_program_handle_t Program,
                                       bool isBuilt) -> void {
    Cache.registerProgramFetch(CacheKey, Program, isBuilt);
  };

  std::shared_ptr<KernelProgramCache::ProgramBuildResult> BuildResult =
      Cache.getOrBuild<errc::build>(GetCachedBuildF, BuildF, EvictFunc);
  assert(BuildResult && "getOrBuild isn't supposed to return nullptr!");

  Managed<ur_program_handle_t> &ResProgram = BuildResult->Val;

  // Here we have multiple devices a program is built for, so add the program to
  // the cache for all subsets of provided list of devices.

  // If we linked any extra device images, then we need to
  // cache them as well.
  auto CacheLinkedImages = [&Cache, &CacheKey, &ResProgram, &ImgWithDeps] {
    for (auto It = ImgWithDeps.depsBegin(); It != ImgWithDeps.depsEnd(); ++It) {
      const RTDeviceBinaryImage *BImg = *It;
      // CacheKey is captured by reference by GetCachedBuildF, so we can simply
      // update it here and re-use that lambda.
      CacheKey.first.second = BImg->getImageID();
      bool DidInsert = Cache.insertBuiltProgram(CacheKey, ResProgram);
      // Add to the eviction list.
      Cache.registerProgramFetch(CacheKey, ResProgram, DidInsert);
    }
  };
  CacheLinkedImages();

  if (URDevicesSet.size() > 1) {
    // emplace all subsets of the current set of devices into the cache.
    // Set of all devices is not included in the loop as it was already added
    // into the cache.
    int Mask = 1;
    if (URDevicesSet.size() > sizeof(Mask) * 8 - 1) {
      // Protection for the algorithm below. Although overflow is very unlikely
      // to be reached.
      throw sycl::exception(
          make_error_code(errc::runtime),
          "Unable to cache built program for more than 31 devices");
    }
    for (; Mask < (1 << URDevicesSet.size()) - 1; ++Mask) {
      std::set<ur_device_handle_t> Subset;
      int Index = 0;
      for (auto It = URDevicesSet.begin(); It != URDevicesSet.end();
           ++It, ++Index) {
        if (Mask & (1 << Index)) {
          Subset.insert(*It);
        }
      }
      // Change device in the cache key to reduce copying of spec const data.
      CacheKey.second = std::move(Subset);
      bool DidInsert = Cache.insertBuiltProgram(CacheKey, ResProgram);
      (void)DidInsert;
      CacheLinkedImages();
    }
  }

  // We don't know if `BuildResult` above is a single owner of this program (no
  // caching) or not (shared ownership with the record in the cache), so we
  // can't just `std::move(ResProgram)` that references
  // `Managed<ur_program_handle_t>` inside `BuildResult` and have to `retain`.
  //
  // If this a single owner indeed, then `BuildResult` will be automatically
  // destructed upon return and would cause automatic `urProgramRelease` which
  // might be unoptimal but still correct.
  return ResProgram.retain();
}

FastKernelCacheValPtr ProgramManager::getOrCreateKernel(
    context_impl &ContextImpl, device_impl &DeviceImpl,
    KernelNameStrRefT KernelName,
    KernelNameBasedCacheT *KernelNameBasedCachePtr, const NDRDescT &NDRDesc) {
  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << &ContextImpl
              << ", " << &DeviceImpl << ", " << KernelName << ")\n";
  }

  KernelProgramCache &Cache = ContextImpl.getKernelProgramCache();
  ur_device_handle_t UrDevice = DeviceImpl.getHandleRef();
  FastKernelSubcacheT *CacheHintPtr =
      KernelNameBasedCachePtr ? &KernelNameBasedCachePtr->FastKernelSubcache
                              : nullptr;
  if (SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    if (auto KernelCacheValPtr =
            Cache.tryToGetKernelFast(KernelName, UrDevice, CacheHintPtr)) {
      return KernelCacheValPtr;
    }
  }

  Managed<ur_program_handle_t> Program =
      getBuiltURProgram(ContextImpl, DeviceImpl, KernelName, NDRDesc);

  auto BuildF = [this, &Program, &KernelName, &ContextImpl] {
    adapter_impl &Adapter = ContextImpl.getAdapter();
    Managed<ur_kernel_handle_t> Kernel{Adapter};
    Adapter.call<errc::kernel_not_supported, UrApiKind::urKernelCreate>(
        Program, KernelName.data(), &Kernel);

    // Only set UR_USM_INDIRECT_ACCESS if the platform can handle it.
    if (ContextImpl.getPlatformImpl().supports_usm()) {
      // Some UR Adapters (like OpenCL) require this call to enable USM
      // For others, UR will turn this into a NOP.
      const ur_bool_t UrTrue = true;
      Adapter.call<UrApiKind::urKernelSetExecInfo>(
          Kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
          nullptr, &UrTrue);
    }

    const KernelArgMask *ArgMask = nullptr;
    if (!m_UseSpvFile)
      ArgMask = getEliminatedKernelArgMask(Program, KernelName);
    return std::make_pair(std::move(Kernel), ArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, &Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    // The built kernel cannot be shared between multiple
    // threads when caching is disabled, so we can return
    // nullptr for the mutex.
    auto [Kernel, ArgMask] = BuildF();
    return std::make_shared<FastKernelCacheVal>(std::move(Kernel), nullptr,
                                                ArgMask, std::move(Program),
                                                ContextImpl.getAdapter());
  }

  std::shared_ptr<KernelProgramCache::KernelBuildResult> BuildResult =
      Cache.getOrBuild<errc::invalid>(GetCachedBuildF, BuildF);
  assert(BuildResult && "getOrBuild isn't supposed to return nullptr!");
  std::pair<Managed<ur_kernel_handle_t>, const KernelArgMask *>
      &KernelArgMaskPair = BuildResult->Val;
  auto ret_val = std::make_shared<FastKernelCacheVal>(
      KernelArgMaskPair.first.retain(), &(BuildResult->MBuildResultMutex),
      KernelArgMaskPair.second, std::move(Program), ContextImpl.getAdapter());
  Cache.saveKernel(KernelName, UrDevice, ret_val, CacheHintPtr);
  return ret_val;
}

ur_program_handle_t
ProgramManager::getUrProgramFromUrKernel(ur_kernel_handle_t Kernel,
                                         context_impl &Context) {
  ur_program_handle_t Program;
  adapter_impl &Adapter = Context.getAdapter();
  Adapter.call<UrApiKind::urKernelGetInfo>(Kernel, UR_KERNEL_INFO_PROGRAM,
                                           sizeof(ur_program_handle_t),
                                           &Program, nullptr);
  return Program;
}

std::string
ProgramManager::getProgramBuildLog(const ur_program_handle_t &Program,
                                   context_impl &Context) {
  size_t URDevicesSize = 0;
  adapter_impl &Adapter = Context.getAdapter();
  Adapter.call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                            0u, nullptr, &URDevicesSize);
  std::vector<ur_device_handle_t> URDevices(URDevicesSize /
                                            sizeof(ur_device_handle_t));
  Adapter.call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                            URDevicesSize, URDevices.data(),
                                            nullptr);
  std::string Log = "The program was built for " +
                    std::to_string(URDevices.size()) + " devices";
  for (ur_device_handle_t &Device : URDevices) {
    std::string DeviceBuildInfoString;
    size_t DeviceBuildInfoStrSize = 0;
    Adapter.call<UrApiKind::urProgramGetBuildInfo>(
        Program, Device, UR_PROGRAM_BUILD_INFO_LOG, 0u, nullptr,
        &DeviceBuildInfoStrSize);
    if (DeviceBuildInfoStrSize > 0) {
      std::vector<char> DeviceBuildInfo(DeviceBuildInfoStrSize);
      Adapter.call<UrApiKind::urProgramGetBuildInfo>(
          Program, Device, UR_PROGRAM_BUILD_INFO_LOG, DeviceBuildInfoStrSize,
          DeviceBuildInfo.data(), nullptr);
      DeviceBuildInfoString = std::string(DeviceBuildInfo.data());
    }

    std::string DeviceNameString;
    size_t DeviceNameStrSize = 0;
    Adapter.call<UrApiKind::urDeviceGetInfo>(Device, UR_DEVICE_INFO_NAME, 0u,
                                             nullptr, &DeviceNameStrSize);
    if (DeviceNameStrSize > 0) {
      std::vector<char> DeviceName(DeviceNameStrSize);
      Adapter.call<UrApiKind::urDeviceGetInfo>(Device, UR_DEVICE_INFO_NAME,
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
static Managed<ur_program_handle_t> loadDeviceLib(context_impl &Context,
                                                  const char *Name) {
  std::string LibSyclDir = OSUtil::getCurrentDSODir();
  std::ifstream File(LibSyclDir + OSUtil::DirSep + Name,
                     std::ifstream::in | std::ifstream::binary);
  if (!File.good()) {
    return {};
  }

  File.seekg(0, std::ios::end);
  size_t FileSize = File.tellg();
  File.seekg(0, std::ios::beg);
  std::vector<char> FileContent(FileSize);
  File.read(&FileContent[0], FileSize);
  File.close();

  return createSpirvProgram(Context, (unsigned char *)&FileContent[0],
                            FileSize);
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

static ur_result_t doCompile(adapter_impl &Adapter, ur_program_handle_t Program,
                             uint32_t NumDevs, ur_device_handle_t *Devs,
                             ur_context_handle_t Ctx, const char *Opts) {
  // Try to compile with given devices, fall back to compiling with the program
  // context if unsupported by the adapter
  auto Result = Adapter.call_nocheck<UrApiKind::urProgramCompileExp>(
      Program, NumDevs, Devs, Opts);
  if (Result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return Adapter.call_nocheck<UrApiKind::urProgramCompile>(Ctx, Program,
                                                             Opts);
  }
  return Result;
}

static ur_program_handle_t
loadDeviceLibFallback(context_impl &Context, DeviceLibExt Extension,
                      std::vector<ur_device_handle_t> &Devices,
                      bool UseNativeLib) {

  auto LibFileName = getDeviceLibFilename(Extension, UseNativeLib);
  auto LockedCache = Context.acquireCachedLibPrograms();
  auto &CachedLibPrograms = LockedCache.get();
  // Collect list of devices to compile the library for. Library was already
  // compiled for a device if there is a corresponding record in the per-context
  // cache.
  std::vector<ur_device_handle_t> DevicesToCompile;
  Managed<ur_program_handle_t> *UrProgram = nullptr;
  assert(Devices.size() > 0 &&
         "At least one device is expected in the input vector");
  // Vector of devices that don't have the library cached.
  for (ur_device_handle_t Dev : Devices) {
    auto [It, Inserted] = CachedLibPrograms.emplace(
        std::make_pair(Extension, Dev), Managed<ur_program_handle_t>{});
    if (!Inserted) {
      Managed<ur_program_handle_t> &CachedUrProgram = It->second;
      assert(CachedUrProgram && "If device lib UR program was cached then is "
                                "expected to be not a nullptr");
      assert(!UrProgram || *UrProgram == CachedUrProgram);
      // Managed<ur_program_handle_t>::operator& is overloaded, use
      // `std::addressof`:
      UrProgram = std::addressof(CachedUrProgram);
    } else {
      DevicesToCompile.push_back(Dev);
    }
  }

  if (DevicesToCompile.empty())
    return *UrProgram;

  auto EraseProgramForDevices = [&]() {
    for (auto Dev : DevicesToCompile)
      CachedLibPrograms.erase(std::make_pair(Extension, Dev));
  };

  Managed<ur_program_handle_t> NewlyCreated;
  // Create UR program for device lib if we don't have it yet.
  if (!UrProgram) {
    NewlyCreated = loadDeviceLib(Context, LibFileName);
    if (NewlyCreated == nullptr) {
      EraseProgramForDevices();
      throw exception(make_error_code(errc::build),
                      std::string("Failed to load ") + LibFileName);
    }
  }

  // Insert UrProgram into the cache for all devices that we will compile for.
  for (auto Dev : DevicesToCompile) {
    Managed<ur_program_handle_t> &Cached =
        CachedLibPrograms[std::make_pair(Extension, Dev)];
    if (NewlyCreated) {
      Cached = std::move(NewlyCreated);
      UrProgram = std::addressof(Cached);
    } else {
      Cached = UrProgram->retain();
    }
  }

  adapter_impl &Adapter = Context.getAdapter();
  // TODO no spec constants are used in the std libraries, support in the future
  // Do not use compile options for library programs: it is not clear if user
  // options (image options) are supposed to be applied to library program as
  // well, and what actually happens to a SPIR-V program if we apply them.
  ur_result_t Error =
      doCompile(Adapter, *UrProgram, DevicesToCompile.size(),
                DevicesToCompile.data(), Context.getHandleRef(), "");
  if (Error != UR_RESULT_SUCCESS) {
    EraseProgramForDevices();
    throw detail::set_ur_error(
        exception(make_error_code(errc::build),
                  ProgramManager::getProgramBuildLog(*UrProgram, Context)),
        Error);
  }

  return *UrProgram;
}

ProgramManager::ProgramManager()
    : m_SanitizerFoundInImage(SanitizerType::None) {
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
    std::unique_ptr<char[], std::function<void(void *)>> Data(new char[Size],
                                                              std::free);
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

const char *getArchName(const device_impl &DeviceImpl) {
  namespace syclex = sycl::ext::oneapi::experimental;
  auto Arch = DeviceImpl.get_info<syclex::info::device::architecture>();
  switch (Arch) {
#define __SYCL_ARCHITECTURE(ARCH, VAL)                                         \
  case syclex::architecture::ARCH:                                             \
    return #ARCH;
#define __SYCL_ARCHITECTURE_ALIAS(ARCH, VAL)
#include <sycl/ext/oneapi/experimental/device_architecture.def>
#undef __SYCL_ARCHITECTURE
#undef __SYCL_ARCHITECTURE_ALIAS
  }
  return "unknown";
}

template <typename StorageKey>
const RTDeviceBinaryImage *getBinImageFromMultiMap(
    const std::unordered_multimap<StorageKey, const RTDeviceBinaryImage *>
        &ImagesSet,
    const StorageKey &Key, context_impl &ContextImpl,
    const device_impl &DeviceImpl) {
  auto [ItBegin, ItEnd] = ImagesSet.equal_range(Key);
  if (ItBegin == ItEnd)
    return nullptr;

  // Here, we aim to select all the device images from the
  // [ItBegin, ItEnd) range that are AOT compiled for Device
  // (checked using info::device::architecture) or JIT compiled.
  // This selection will then be passed to urDeviceSelectBinary
  // for final selection.
  std::vector<const RTDeviceBinaryImage *> DeviceFilteredImgs;
  DeviceFilteredImgs.reserve(std::distance(ItBegin, ItEnd));
  for (auto It = ItBegin; It != ItEnd; ++It) {
    if (doesImageTargetMatchDevice(*It->second, DeviceImpl))
      DeviceFilteredImgs.push_back(It->second);
  }

  if (DeviceFilteredImgs.empty())
    return nullptr;

  const size_t NumImgs = DeviceFilteredImgs.size();
  // Pass extra information to the HIP adapter to aid in binary selection. We
  // pass it the raw binary as a {ptr, length} pair.
  std::vector<std::pair<const unsigned char *, size_t>> UrBinariesStorage;
  if (DeviceImpl.getBackend() == backend::ext_oneapi_hip)
    UrBinariesStorage.reserve(NumImgs);

  std::vector<ur_device_binary_t> UrBinaries(NumImgs);
  for (uint32_t BinaryCount = 0; BinaryCount < NumImgs; BinaryCount++) {
    const sycl_device_binary_struct &RawImg =
        DeviceFilteredImgs[BinaryCount]->getRawData();
    UrBinaries[BinaryCount].pDeviceTargetSpec =
        getUrDeviceTarget(RawImg.DeviceTargetSpec);
    if (DeviceImpl.getBackend() == backend::ext_oneapi_hip) {
      UrBinariesStorage.emplace_back(
          RawImg.BinaryStart,
          std::distance(RawImg.BinaryStart, RawImg.BinaryEnd));
      UrBinaries[BinaryCount].pNext = &UrBinariesStorage[BinaryCount];
    }
  }

  uint32_t ImgInd = 0;
  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  ContextImpl.getAdapter().call<UrApiKind::urDeviceSelectBinary>(
      DeviceImpl.getHandleRef(), UrBinaries.data(), UrBinaries.size(), &ImgInd);
  return DeviceFilteredImgs[ImgInd];
}

const RTDeviceBinaryImage &
ProgramManager::getDeviceImage(KernelNameStrRefT KernelName,
                               context_impl &ContextImpl,
                               const device_impl &DeviceImpl) {
  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(\"" << KernelName << "\", "
              << ContextImpl.get() << ", " << &DeviceImpl << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }

  if (m_UseSpvFile) {
    assert(m_SpvFileImage);
    return getDeviceImage(
        std::unordered_set<const RTDeviceBinaryImage *>({m_SpvFileImage.get()}),
        ContextImpl, DeviceImpl);
  }

  const RTDeviceBinaryImage *Img = nullptr;
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    if (auto KernelId = m_KernelName2KernelIDs.find(KernelName);
        KernelId != m_KernelName2KernelIDs.end()) {
      Img = getBinImageFromMultiMap(m_KernelIDs2BinImage, KernelId->second,
                                    ContextImpl, DeviceImpl);
    } else {
      Img = getBinImageFromMultiMap(m_ServiceKernels, KernelName, ContextImpl,
                                    DeviceImpl);
    }
  }

  // Decompress the image if it is compressed.
  CheckAndDecompressImage(Img);

  if (Img) {
    if constexpr (DbgProgMgr > 0) {
      std::cerr << "selected device image: " << &Img->getRawData() << "\n";
      Img->print();
    }
    return *Img;
  }

  throw exception(make_error_code(errc::runtime),
                  "No kernel named " + std::string(KernelName) + " was found");
}

const RTDeviceBinaryImage &ProgramManager::getDeviceImage(
    const std::unordered_set<const RTDeviceBinaryImage *> &ImageSet,
    context_impl &ContextImpl, const device_impl &DeviceImpl) {
  assert(ImageSet.size() > 0);

  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(Custom SPV file "
              << ContextImpl.get() << ", " << &DeviceImpl << ")\n";

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

  ContextImpl.getAdapter().call<UrApiKind::urDeviceSelectBinary>(
      DeviceImpl.getHandleRef(), UrBinaries.data(), UrBinaries.size(), &ImgInd);

  ImageIterator = ImageSet.begin();
  std::advance(ImageIterator, ImgInd);

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
getDeviceLibPrograms(context_impl &Context,
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
        return Context.getPlatformImpl().getDeviceImpl(Device)->has_extension(
            "cl_khr_fp64");
      });

  // Load a fallback library for an extension if the any device does not
  // support it.
  for (auto Device : Devices) {
    // TODO: device_impl::has_extension should cache extension string, then we'd
    // be able to use that in the loop below directly.
    std::string DevExtList = urGetInfoString<UrApiKind::urDeviceGetInfo>(
        *Context.getPlatformImpl().getDeviceImpl(Device),
        UR_DEVICE_INFO_EXTENSIONS);

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

Managed<ur_program_handle_t> ProgramManager::build(
    Managed<ur_program_handle_t> Program, context_impl &Context,
    const std::string &CompileOptions, const std::string &LinkOptions,
    std::vector<ur_device_handle_t> &Devices, uint32_t DeviceLibReqMask,
    const std::vector<Managed<ur_program_handle_t>> &ExtraProgramsToLink,
    bool CreatedFromBinary) {

  if constexpr (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build("
              << static_cast<ur_program_handle_t>(Program) << ", "
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

  adapter_impl &Adapter = Context.getAdapter();
  if (LinkPrograms.empty() && ExtraProgramsToLink.empty() && !ForceLink) {
    const std::string &Options = LinkOptions.empty()
                                     ? CompileOptions
                                     : (CompileOptions + " " + LinkOptions);
    ur_result_t Error = Adapter.call_nocheck<UrApiKind::urProgramBuildExp>(
        Program, Devices.size(), Devices.data(), Options.c_str());
    if (Error == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Error = Adapter.call_nocheck<UrApiKind::urProgramBuild>(
          Context.getHandleRef(), Program, Options.c_str());
    }

    if (Error != UR_RESULT_SUCCESS)
      throw detail::set_ur_error(
          exception(make_error_code(errc::build),
                    getProgramBuildLog(Program, Context)),
          Error);

    return Program;
  }

  // Include the main program and compile/link everything together
  if (!CreatedFromBinary) {
    auto Res = doCompile(Adapter, Program, Devices.size(), Devices.data(),
                         Context.getHandleRef(), CompileOptions.c_str());
    Adapter.checkUrResult<errc::build>(Res);
  }
  LinkPrograms.push_back(Program);

  for (ur_program_handle_t Prg : ExtraProgramsToLink) {
    if (!CreatedFromBinary) {
      auto Res = doCompile(Adapter, Prg, Devices.size(), Devices.data(),
                           Context.getHandleRef(), CompileOptions.c_str());
      Adapter.checkUrResult<errc::build>(Res);
    }
    LinkPrograms.push_back(Prg);
  }

  Managed<ur_program_handle_t> LinkedProg{Adapter};
  auto doLink = [&] {
    auto Res = Adapter.call_nocheck<UrApiKind::urProgramLinkExp>(
        Context.getHandleRef(), Devices.size(), Devices.data(),
        LinkPrograms.size(), LinkPrograms.data(), LinkOptions.c_str(),
        &LinkedProg);
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter.call_nocheck<UrApiKind::urProgramLink>(
          Context.getHandleRef(), LinkPrograms.size(), LinkPrograms.data(),
          LinkOptions.c_str(), &LinkedProg);
    }
    return Res;
  };
  ur_result_t Error = doLink();
  if (Error == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
      Error == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
      Error == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    Context.getKernelProgramCache().reset();
    Error = doLink();
  }

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise.
  if (Error != UR_RESULT_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg (via implicit dtor call), and
      // throw.
      throw detail::set_ur_error(
          exception(make_error_code(errc::build),
                    getProgramBuildLog(LinkedProg, Context)),
          Error);
    }
    Adapter.checkUrResult(Error);
  }
  return LinkedProg;
}

void ProgramManager::cacheKernelUsesAssertInfo(const RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &AssertUsedRange =
      Img.getAssertUsed();
  if (AssertUsedRange.isAvailable())
    for (const auto &Prop : AssertUsedRange)
      m_KernelUsesAssert.insert(Prop->Name);
}

void ProgramManager::cacheKernelImplicitLocalArg(
    const RTDeviceBinaryImage &Img) {
  const RTDeviceBinaryImage::PropertyRange &ImplicitLocalArgRange =
      Img.getImplicitLocalArg();
  if (ImplicitLocalArgRange.isAvailable())
    for (auto Prop : ImplicitLocalArgRange) {
      m_KernelImplicitLocalArgPos[Prop->Name] =
          DeviceBinaryProperty(Prop).asUint32();
    }
}

std::optional<int> ProgramManager::kernelImplicitLocalArgPos(
    KernelNameStrRefT KernelName,
    KernelNameBasedCacheT *KernelNameBasedCachePtr) const {
  auto getLocalArgPos = [&]() -> std::optional<int> {
    auto it = m_KernelImplicitLocalArgPos.find(KernelName);
    if (it != m_KernelImplicitLocalArgPos.end())
      return it->second;
    return {};
  };

  if (!KernelNameBasedCachePtr)
    return getLocalArgPos();
  std::optional<std::optional<int>> &ImplicitLocalArgPos =
      KernelNameBasedCachePtr->ImplicitLocalArgPos;
  if (!ImplicitLocalArgPos.has_value()) {
    ImplicitLocalArgPos = getLocalArgPos();
  }
  return ImplicitLocalArgPos.value();
}

static bool isBfloat16DeviceLibImage(sycl_device_binary RawImg,
                                     uint32_t *LibVersion = nullptr) {
  sycl_device_binary_property_set ImgPS;
  for (ImgPS = RawImg->PropertySetsBegin; ImgPS != RawImg->PropertySetsEnd;
       ++ImgPS) {
    if (ImgPS->Name &&
        !strcmp(__SYCL_PROPERTY_SET_DEVICELIB_METADATA, ImgPS->Name)) {
      if (!LibVersion)
        return true;

      // Valid version for bfloat16 device library is 0(fallback), 1(native).
      *LibVersion = 2;
      sycl_device_binary_property ImgP;
      for (ImgP = ImgPS->PropertiesBegin; ImgP != ImgPS->PropertiesEnd;
           ++ImgP) {
        if (ImgP->Name && !strcmp("bfloat16", ImgP->Name) &&
            (ImgP->Type == SYCL_PROPERTY_TYPE_UINT32))
          break;
      }
      if (ImgP != ImgPS->PropertiesEnd)
        *LibVersion = DeviceBinaryProperty(ImgP).asUint32();
      return true;
    }
  }

  return false;
}

static sycl_device_binary_property_set
getExportedSymbolPS(sycl_device_binary RawImg) {
  sycl_device_binary_property_set ImgPS;
  for (ImgPS = RawImg->PropertySetsBegin; ImgPS != RawImg->PropertySetsEnd;
       ++ImgPS) {
    if (ImgPS->Name &&
        !strcmp(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS, ImgPS->Name))
      return ImgPS;
  }

  return nullptr;
}

static bool shouldSkipEmptyImage(sycl_device_binary RawImg) {
  // For bfloat16 device library image, we should keep it although it doesn't
  // include any kernel.
  if (isBfloat16DeviceLibImage(RawImg))
    return false;

  // We may extend the logic here other than bfloat16 device library image.
  return true;
}

void ProgramManager::addImage(sycl_device_binary RawImg,
                              bool RegisterImgExports,
                              RTDeviceBinaryImage **OutImage,
                              std::vector<kernel_id> *OutKernelIDs) {
  const bool DumpImages = std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile;
  const sycl_offload_entry EntriesB = RawImg->EntriesBegin;
  const sycl_offload_entry EntriesE = RawImg->EntriesEnd;
  // Treat the image as empty one
  if (EntriesB == EntriesE && shouldSkipEmptyImage(RawImg))
    return;

  uint32_t Bfloat16DeviceLibVersion = 0;
  const bool IsBfloat16DeviceLib =
      isBfloat16DeviceLibImage(RawImg, &Bfloat16DeviceLibVersion);
  const bool IsDeviceImageCompressed = isDeviceImageCompressed(RawImg);

  std::unique_ptr<RTDeviceBinaryImage> Img;
  if (IsDeviceImageCompressed) {
#ifdef SYCL_RT_ZSTD_AVAILABLE
    Img = std::make_unique<CompressedRTDeviceBinaryImage>(RawImg);
#else
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Recieved a compressed device image, but "
                          "SYCL RT was built without ZSTD support."
                          "Aborting. ");
#endif
  } else if (!IsBfloat16DeviceLib) {
    Img = std::make_unique<RTDeviceBinaryImage>(RawImg);
  }

  // If an output image is requested, set it to the newly allocated image.
  if (OutImage)
    *OutImage = Img.get();

  static uint32_t SequenceID = 0;

  // Fill the kernel argument mask map, no need to do this for bfloat16
  // device library image since it doesn't include any kernel.
  if (!IsBfloat16DeviceLib) {
    const RTDeviceBinaryImage::PropertyRange &KPOIRange =
        Img->getKernelParamOptInfo();
    if (KPOIRange.isAvailable()) {
      KernelNameToArgMaskMap &ArgMaskMap =
          m_EliminatedKernelArgMasks[Img.get()];
      for (const auto &Info : KPOIRange)
        ArgMaskMap[Info->Name] =
            createKernelArgMask(DeviceBinaryProperty(Info).asByteArray());
    }
  }

  // Fill maps for kernel bundles
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  // For bfloat16 device library image, it doesn't include any kernel, device
  // global, virtual function, so just skip adding it to any related maps.
  // The bfloat16 device library are provided by compiler and may be used by
  // different sycl device images, program manager will own single copy for
  // native and fallback version bfloat16 device library, these device
  // library images will not be erased unless program manager is destroyed.
  {
    if (IsBfloat16DeviceLib) {
      assert((Bfloat16DeviceLibVersion < 2) &&
             "Invalid Bfloat16 Device Library Index.");
      if (m_Bfloat16DeviceLibImages[Bfloat16DeviceLibVersion].get())
        return;

      std::unique_ptr<RTDeviceBinaryImage> DevImg;
      if (IsDeviceImageCompressed) {
        // Decompress the image.
        CheckAndDecompressImage(Img.get());
        DevImg = std::move(Img);
      } else {
        size_t ImgSize =
            static_cast<size_t>(RawImg->BinaryEnd - RawImg->BinaryStart);
        std::unique_ptr<char[], std::function<void(void *)>> Data(
            new char[ImgSize], std::free);
        std::memcpy(Data.get(), RawImg->BinaryStart, ImgSize);
        DevImg =
            std::make_unique<DynRTDeviceBinaryImage>(std::move(Data), ImgSize);
      }

      // Register export symbols for bfloat16 device library image.
      auto ESPropSet = getExportedSymbolPS(RawImg);
      for (auto ESProp = ESPropSet->PropertiesBegin;
           ESProp != ESPropSet->PropertiesEnd; ++ESProp) {
        m_ExportedSymbolImages.insert({ESProp->Name, DevImg.get()});
      }
      m_Bfloat16DeviceLibImages[Bfloat16DeviceLibVersion] = std::move(DevImg);

      return;
    }
  }

  // Register all exported symbols
  if (RegisterImgExports) {
    for (const sycl_device_binary_property &ESProp :
         Img->getExportedSymbols()) {
      m_ExportedSymbolImages.insert({ESProp->Name, Img.get()});
    }
  }

  // Record mapping between virtual function sets and device images
  for (const sycl_device_binary_property &VFProp : Img->getVirtualFunctions()) {
    std::string StrValue = DeviceBinaryProperty(VFProp).asCString();
    for (const auto &SetName : detail::split_string(StrValue, ','))
      m_VFSet2BinImage[SetName].insert(Img.get());
  }

  if (DumpImages) {
    const bool NeedsSequenceID =
        std::any_of(m_BinImg2KernelIDs.begin(), m_BinImg2KernelIDs.end(),
                    [&](auto &CurrentImg) {
                      return CurrentImg.first->getFormat() == Img->getFormat();
                    });

    // Check if image is compressed, and decompress it before dumping.
    CheckAndDecompressImage(Img.get());

    dumpImage(*Img, NeedsSequenceID ? ++SequenceID : 0);
  }

  std::shared_ptr<std::vector<kernel_id>> &KernelIDs =
      m_BinImg2KernelIDs[Img.get()];
  KernelIDs.reset(new std::vector<kernel_id>);

  for (sycl_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
       EntriesIt = EntriesIt->Increment()) {

    auto name = EntriesIt->GetName();

    // Skip creating unique kernel ID if it is a service kernel.
    // SYCL service kernels are identified by having
    // __sycl_service_kernel__ in the mangled name, primarily as part of
    // the namespace of the name type.
    if (std::strstr(name, "__sycl_service_kernel__")) {
      m_ServiceKernels.insert(std::make_pair(name, Img.get()));
      continue;
    }

    // Skip creating unique kernel ID if it is an exported device
    // function. Exported device functions appear in the offload entries
    // among kernels, but are identifiable by being listed in properties.
    if (m_ExportedSymbolImages.find(name) != m_ExportedSymbolImages.end())
      continue;

    // ... and create a unique kernel ID for the entry
    auto It = m_KernelName2KernelIDs.find(name);
    if (It == m_KernelName2KernelIDs.end()) {
      std::shared_ptr<detail::kernel_id_impl> KernelIDImpl =
          std::make_shared<detail::kernel_id_impl>(name);
      sycl::kernel_id KernelID =
          detail::createSyclObjFromImpl<sycl::kernel_id>(KernelIDImpl);

      It = m_KernelName2KernelIDs.emplace_hint(It, name, KernelID);
    }
    m_KernelIDs2BinImage.insert(std::make_pair(It->second, Img.get()));
    KernelIDs->push_back(It->second);
  }

  cacheKernelUsesAssertInfo(*Img);

  // check if kernel uses sanitizer
  {
    sycl_device_binary_property SanProp = Img->getProperty("sanUsed");
    if (SanProp) {
      std::string SanValue = detail::DeviceBinaryProperty(SanProp).asCString();

      if (SanValue.rfind("asan", 0) == 0) { // starts_with
        m_SanitizerFoundInImage = SanitizerType::AddressSanitizer;
      } else if (SanValue.rfind("msan", 0) == 0) {
        m_SanitizerFoundInImage = SanitizerType::MemorySanitizer;
      } else if (SanValue.rfind("tsan", 0) == 0) {
        m_SanitizerFoundInImage = SanitizerType::ThreadSanitizer;
      }
    }
  }

  cacheKernelImplicitLocalArg(*Img);

  // Sort kernel ids for faster search
  std::sort(KernelIDs->begin(), KernelIDs->end(), LessByHash<kernel_id>{});

  // If requested, copy the new (sorted) kernel IDs.
  if (OutKernelIDs)
    OutKernelIDs->insert(OutKernelIDs->end(), KernelIDs->begin(),
                         KernelIDs->end());

  // ... and initialize associated device_global information
  m_DeviceGlobals.initializeEntries(Img.get());
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

  m_DeviceImages.insert({RawImg, std::move(Img)});
}

void ProgramManager::addImages(sycl_device_binaries DeviceBinary) {
  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++)
    addImage(&(DeviceBinary->DeviceBinaries[I]));
}

void ProgramManager::removeImages(sycl_device_binaries DeviceBinary) {
  if (DeviceBinary->NumDeviceBinaries == 0)
    return;
  // Acquire lock to read and modify maps for kernel bundles
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    sycl_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);

    auto DevImgIt = m_DeviceImages.find(RawImg);
    if (DevImgIt == m_DeviceImages.end())
      continue;
    const sycl_offload_entry EntriesB = RawImg->EntriesBegin;
    const sycl_offload_entry EntriesE = RawImg->EntriesEnd;
    if (EntriesB == EntriesE)
      continue;

    RTDeviceBinaryImage *Img = DevImgIt->second.get();

    // Drop the kernel argument mask map
    m_EliminatedKernelArgMasks.erase(Img);

    // Unmap the unique kernel IDs for the offload entries
    for (sycl_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
         EntriesIt = EntriesIt->Increment()) {

      // Drop entry for service kernel
      if (std::strstr(EntriesIt->GetName(), "__sycl_service_kernel__")) {
        m_ServiceKernels.erase(EntriesIt->GetName());
        continue;
      }

      // Exported device functions won't have a kernel ID
      if (m_ExportedSymbolImages.find(EntriesIt->GetName()) !=
          m_ExportedSymbolImages.end()) {
        continue;
      }

      // remove everything associated with this KernelName
      m_KernelUsesAssert.erase(EntriesIt->GetName());
      m_KernelImplicitLocalArgPos.erase(EntriesIt->GetName());

      if (auto It = m_KernelName2KernelIDs.find(EntriesIt->GetName());
          It != m_KernelName2KernelIDs.end()) {
        m_KernelIDs2BinImage.erase(It->second);
        m_KernelName2KernelIDs.erase(It);
      }
    }

    // Drop reverse mapping
    m_BinImg2KernelIDs.erase(Img);

    // Unregister exported symbols (needs to happen after the ID unmap loop)
    for (const sycl_device_binary_property &ESProp :
         Img->getExportedSymbols()) {
      m_ExportedSymbolImages.erase(ESProp->Name);
    }

    for (const sycl_device_binary_property &VFProp :
         Img->getVirtualFunctions()) {
      std::string StrValue = DeviceBinaryProperty(VFProp).asCString();
      for (const auto &SetName : detail::split_string(StrValue, ','))
        m_VFSet2BinImage.erase(SetName);
    }

    m_DeviceGlobals.eraseEntries(Img);

    {
      std::lock_guard<std::mutex> HostPipesGuard(m_HostPipesMutex);
      auto HostPipes = Img->getHostPipes();
      for (const sycl_device_binary_property &HostPipe : HostPipes) {
        if (auto HostPipesIt = m_HostPipes.find(HostPipe->Name);
            HostPipesIt != m_HostPipes.end()) {
          auto findHostPipesByValue = std::find_if(
              m_Ptr2HostPipe.begin(), m_Ptr2HostPipe.end(),
              [&HostPipesIt](
                  const std::pair<const void *, HostPipeMapEntry *> &Entry) {
                return Entry.second == HostPipesIt->second.get();
              });
          if (findHostPipesByValue != m_Ptr2HostPipe.end())
            m_Ptr2HostPipe.erase(findHostPipesByValue);
          m_HostPipes.erase(HostPipesIt);
        }
      }
    }

    // Purge references to the image in native programs map
    {
      std::lock_guard<std::mutex> NativeProgramsGuard(MNativeProgramsMutex);

      // The map does not keep references to program handles; we can erase the
      // entry without calling UR release
      for (auto It = NativePrograms.begin(); It != NativePrograms.end();) {
        auto CurIt = It++;
        if (CurIt->second.second == Img) {
          if (auto ContextImpl = CurIt->second.first.lock()) {
            ContextImpl->getKernelProgramCache().removeAllRelatedEntries(
                Img->getImageID());
          }
          NativePrograms.erase(CurIt);
        }
      }
    }

    m_DeviceImages.erase(DevImgIt);
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
                                           KernelNameStrRefT KernelName) {
  // Bail out if there are no eliminated kernel arg masks in our images
  if (m_EliminatedKernelArgMasks.empty())
    return nullptr;

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    auto Range = NativePrograms.equal_range(NativePrg);
    for (auto ImgIt = Range.first; ImgIt != Range.second; ++ImgIt) {
      auto MapIt = m_EliminatedKernelArgMasks.find(ImgIt->second.second);
      if (MapIt == m_EliminatedKernelArgMasks.end())
        continue;
      auto ArgMaskMapIt = MapIt->second.find(KernelName);
      if (ArgMaskMapIt != MapIt->second.end())
        return &ArgMaskMapIt->second;
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

bundle_state
ProgramManager::getBinImageState(const RTDeviceBinaryImage *BinImage) {
  auto IsAOTBinary = [](const char *Format) {
    return ((strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
            (strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
            (strcmp(Format, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0));
  };

  // Three possible initial states:
  // - SPIRV that needs to be compiled and linked
  // - AOT compiled binary with dependnecies, needs linking.
  // - AOT compiled binary without dependencies.

  const bool IsAOT = IsAOTBinary(BinImage->getRawData().DeviceTargetSpec);

  if (!IsAOT)
    return sycl::bundle_state::input;
  return BinImage->getImportedSymbols().empty() ? sycl::bundle_state::executable
                                                : sycl::bundle_state::object;
}

std::optional<kernel_id>
ProgramManager::tryGetSYCLKernelID(KernelNameStrRefT KernelName) {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  auto KernelID = m_KernelName2KernelIDs.find(KernelName);
  if (KernelID == m_KernelName2KernelIDs.end())
    return std::nullopt;

  return KernelID->second;
}

kernel_id ProgramManager::getSYCLKernelID(KernelNameStrRefT KernelName) {
  if (std::optional<kernel_id> MaybeKernelID = tryGetSYCLKernelID(KernelName))
    return *MaybeKernelID;
  throw exception(make_error_code(errc::runtime),
                  "No kernel found with the specified name");
}

bool ProgramManager::hasCompatibleImage(const device_impl &DeviceImpl) {
  std::lock_guard<std::mutex> Guard(m_KernelIDsMutex);

  return std::any_of(
      m_BinImg2KernelIDs.cbegin(), m_BinImg2KernelIDs.cend(),
      [&](std::pair<const RTDeviceBinaryImage *,
                    std::shared_ptr<std::vector<kernel_id>>>
              Elem) { return compatibleWithDevice(Elem.first, DeviceImpl); });
}

std::vector<kernel_id> ProgramManager::getAllSYCLKernelIDs() {
  std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);

  std::vector<sycl::kernel_id> AllKernelIDs;
  AllKernelIDs.reserve(m_KernelName2KernelIDs.size());
  for (std::pair<KernelNameStrT, kernel_id> KernelID : m_KernelName2KernelIDs) {
    AllKernelIDs.push_back(KernelID.second);
  }
  return AllKernelIDs;
}

kernel_id ProgramManager::getBuiltInKernelID(KernelNameStrRefT KernelName) {
  std::lock_guard<std::mutex> BuiltInKernelIDsGuard(m_BuiltInKernelIDsMutex);

  auto KernelID = m_BuiltInKernelIDs.find(KernelName);
  if (KernelID == m_BuiltInKernelIDs.end()) {
    auto Impl = std::make_shared<kernel_id_impl>(KernelName);
    auto CachedID = createSyclObjFromImpl<kernel_id>(std::move(Impl));
    KernelID = m_BuiltInKernelIDs.insert({KernelName, CachedID}).first;
  }

  return KernelID->second;
}

void ProgramManager::addOrInitDeviceGlobalEntry(const void *DeviceGlobalPtr,
                                                const char *UniqueId) {
  m_DeviceGlobals.addOrInitialize(DeviceGlobalPtr, UniqueId);
}

void ProgramManager::registerKernelGlobalInfo(
    std::unordered_map<std::string_view, unsigned> &&GlobalInfoToCopy) {
  std::lock_guard<std::mutex> Guard(MNativeProgramsMutex);
  if (m_FreeFunctionKernelGlobalInfo.empty())
    m_FreeFunctionKernelGlobalInfo = std::move(GlobalInfoToCopy);
  else {
    for (auto &GlobalInfo : GlobalInfoToCopy) {
      m_FreeFunctionKernelGlobalInfo.insert(GlobalInfo);
    }
  }
}

std::optional<unsigned>
ProgramManager::getKernelGlobalInfoDesc(const char *UniqueId) {
  std::lock_guard<std::mutex> Guard(MNativeProgramsMutex);
  const auto It = m_FreeFunctionKernelGlobalInfo.find(UniqueId);
  if (It == m_FreeFunctionKernelGlobalInfo.end())
    return std::nullopt;
  return It->second;
}

std::set<const RTDeviceBinaryImage *>
ProgramManager::getRawDeviceImages(const std::vector<kernel_id> &KernelIDs) {
  std::set<const RTDeviceBinaryImage *> BinImages;
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
  return m_DeviceGlobals.getEntry(DeviceGlobalPtr);
}

DeviceGlobalMapEntry *
ProgramManager::tryGetDeviceGlobalEntry(const std::string &UniqueId,
                                        bool ExcludeDeviceImageScopeDecorated) {
  return m_DeviceGlobals.tryGetEntry(UniqueId,
                                     ExcludeDeviceImageScopeDecorated);
}

std::vector<DeviceGlobalMapEntry *> ProgramManager::getDeviceGlobalEntries(
    const std::vector<std::string> &UniqueIds,
    bool ExcludeDeviceImageScopeDecorated) {
  std::vector<DeviceGlobalMapEntry *> FoundEntries;
  FoundEntries.reserve(UniqueIds.size());
  m_DeviceGlobals.getEntries(UniqueIds, ExcludeDeviceImageScopeDecorated,
                             FoundEntries);
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
    const RTDeviceBinaryImage *BinImage, const context &Ctx,
    const device &Dev) {
  const bundle_state ImgState = getBinImageState(BinImage);

  assert(compatibleWithDevice(BinImage, *getSyclObjImpl(Dev).get()));

  std::shared_ptr<std::vector<sycl::kernel_id>> KernelIDs;
  // Collect kernel names for the image.
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    KernelIDs = m_BinImg2KernelIDs[BinImage];
  }

  return createSyclObjFromImpl<device_image_plain>(device_image_impl::create(
      BinImage, Ctx, Dev, ImgState, std::move(KernelIDs),
      Managed<ur_program_handle_t>{}, ImageOriginSYCLOffline));
}

std::vector<DevImgPlainWithDeps>
ProgramManager::getSYCLDeviceImagesWithCompatibleState(
    const context &Ctx, devices_range Devs, bundle_state TargetState,
    const std::vector<kernel_id> &KernelIDs) {

  // Collect unique raw device images taking into account kernel ids passed
  // TODO: Can we avoid repacking?
  std::set<const RTDeviceBinaryImage *> BinImages;
  if (!KernelIDs.empty()) {
    for (const auto &KID : KernelIDs) {
      bool isCompatibleWithAtLeastOneDev =
          std::any_of(Devs.begin(), Devs.end(), [&KID](device_impl &Dev) {
            return detail::is_compatible({KID}, Dev);
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

  // If a non-input state is requested, we can filter out some compatible
  // images and return only those with the highest compatible state for each
  // device-kernel pair. This map tracks how many kernel-device pairs need each
  // image, so that any unneeded ones are skipped.
  // TODO this has no effect if the requested state is input, consider having
  // a separate branch for that case to avoid unnecessary tracking work.
  struct DeviceBinaryImageInfo {
    std::shared_ptr<std::vector<sycl::kernel_id>> KernelIDs;
    std::set<const RTDeviceBinaryImage *> Deps;
    bundle_state State = bundle_state::input;
    int RequirementCounter = 0;
  };
  std::unordered_map<const RTDeviceBinaryImage *, DeviceBinaryImageInfo>
      ImageInfoMap;

  for (device_impl &Dev : Devs) {
    // Track the highest image state for each requested kernel.
    using StateImagesPairT =
        std::pair<bundle_state, std::vector<const RTDeviceBinaryImage *>>;
    using KernelImageMapT =
        std::map<kernel_id, StateImagesPairT, LessByNameComp>;
    KernelImageMapT KernelImageMap;
    if (!KernelIDs.empty())
      for (const kernel_id &KernelID : KernelIDs)
        KernelImageMap.insert({KernelID, {}});

    for (const RTDeviceBinaryImage *BinImage : BinImages) {
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
        ImgInfo.Deps = collectDeviceImageDeps(*BinImage, Dev);
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
          for (const RTDeviceBinaryImage *Img : KernelImages) {
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

  // Filter out main images that are represented as dependencies of other chosen
  // images to avoid unnecessary duplication.
  // TODO it might make sense to do something about shared dependencies as well.
  for (const auto &ImgInfoPair : ImageInfoMap) {
    if (ImgInfoPair.second.RequirementCounter == 0)
      continue;
    for (const RTDeviceBinaryImage *Dep : ImgInfoPair.second.Deps) {
      auto It = ImageInfoMap.find(Dep);
      if (It != ImageInfoMap.end())
        It->second.RequirementCounter = 0;
    }
  }

  std::vector<DevImgPlainWithDeps> SYCLDeviceImages;
  for (const auto &ImgInfoPair : ImageInfoMap) {
    if (ImgInfoPair.second.RequirementCounter == 0)
      continue;

    std::shared_ptr<device_image_impl> MainImpl = device_image_impl::create(
        ImgInfoPair.first, Ctx, Devs, ImgInfoPair.second.State,
        ImgInfoPair.second.KernelIDs, Managed<ur_program_handle_t>{},
        ImageOriginSYCLOffline);

    std::vector<device_image_plain> Images;
    const std::set<const RTDeviceBinaryImage *> &Deps = ImgInfoPair.second.Deps;
    Images.reserve(Deps.size() + 1);
    Images.push_back(
        createSyclObjFromImpl<device_image_plain>(std::move(MainImpl)));
    for (const RTDeviceBinaryImage *Dep : Deps)
      Images.push_back(
          createDependencyImage(Ctx, Devs, Dep, ImgInfoPair.second.State));
    SYCLDeviceImages.push_back(std::move(Images));
  }

  return SYCLDeviceImages;
}

device_image_plain
ProgramManager::createDependencyImage(const context &Ctx, devices_range Devs,
                                      const RTDeviceBinaryImage *DepImage,
                                      bundle_state DepState) {
  std::shared_ptr<std::vector<sycl::kernel_id>> DepKernelIDs;
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    // For device library images, they are not in m_BinImg2KernelIDs since
    // no kernel is included.
    auto DepIt = m_BinImg2KernelIDs.find(DepImage);
    if (DepIt != m_BinImg2KernelIDs.end())
      DepKernelIDs = DepIt->second;
  }

  assert(DepState == getBinImageState(DepImage) &&
         "State mismatch between main image and its dependency");

  return createSyclObjFromImpl<device_image_plain>(device_image_impl::create(
      DepImage, Ctx, Devs, DepState, std::move(DepKernelIDs),
      Managed<ur_program_handle_t>{}, ImageOriginSYCLOffline));
}

void ProgramManager::bringSYCLDeviceImageToState(
    DevImgPlainWithDeps &DeviceImage, bundle_state TargetState) {
  device_image_plain &MainImg = DeviceImage.getMain();
  device_image_impl &MainImgImpl = *getSyclObjImpl(MainImg);
  const bundle_state DevImageState = getSyclObjImpl(MainImg)->get_state();
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
      DeviceImage = compile(DeviceImage, MainImgImpl.get_devices(),
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
      DeviceImage = build(DeviceImage, MainImgImpl.get_devices(),
                          /*PropList=*/{});
      break;
    case bundle_state::object: {
      std::vector<device_image_plain> LinkedDevImages =
          link(DeviceImage.getAll(), MainImgImpl.get_devices(),
               /*PropList=*/{});
      // Since only one device image is passed here one output device image is
      // expected
      assert(LinkedDevImages.size() == 1 && "Expected one linked image here");
      DeviceImage = LinkedDevImages[0];
      break;
    }
    case bundle_state::executable:
      DeviceImage = build(DeviceImage, MainImgImpl.get_devices(),
                          /*PropList=*/{});
      break;
    }
    break;
  }
  }
}

void ProgramManager::bringSYCLDeviceImagesToState(
    std::vector<DevImgPlainWithDeps> &DeviceImages, bundle_state TargetState) {
  for (DevImgPlainWithDeps &ImgWithDeps : DeviceImages)
    bringSYCLDeviceImageToState(ImgWithDeps, TargetState);
}

std::vector<DevImgPlainWithDeps>
ProgramManager::getSYCLDeviceImages(const context &Ctx, devices_range Devs,
                                    bundle_state TargetState) {
  // Collect device images with compatible state
  std::vector<DevImgPlainWithDeps> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState);
  // Bring device images with compatible state to desired state.
  bringSYCLDeviceImagesToState(DeviceImages, TargetState);
  return DeviceImages;
}

std::vector<DevImgPlainWithDeps>
ProgramManager::getSYCLDeviceImages(const context &Ctx, devices_range Devs,
                                    const DevImgSelectorImpl &Selector,
                                    bundle_state TargetState) {
  // Collect device images with compatible state
  std::vector<DevImgPlainWithDeps> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState);

  // Filter out images that are rejected by Selector
  // TODO Clarify spec, should the selector be able to affect dependent images
  // here?
  auto It = std::remove_if(
      DeviceImages.begin(), DeviceImages.end(),
      [&Selector](const DevImgPlainWithDeps &ImageWithDeps) {
        return !Selector(getSyclObjImpl(ImageWithDeps.getMain()));
      });
  DeviceImages.erase(It, DeviceImages.end());

  // The spec says that the function should not call online compiler or linker
  // to translate device images into target state
  return DeviceImages;
}

std::vector<DevImgPlainWithDeps>
ProgramManager::getSYCLDeviceImages(const context &Ctx, devices_range Devs,
                                    const std::vector<kernel_id> &KernelIDs,
                                    bundle_state TargetState) {
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
  std::vector<DevImgPlainWithDeps> DeviceImages =
      getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, TargetState, KernelIDs);

  // Bring device images with compatible state to desired state.
  bringSYCLDeviceImagesToState(DeviceImages, TargetState);
  return DeviceImages;
}

DevImgPlainWithDeps
ProgramManager::compile(const DevImgPlainWithDeps &ImgWithDeps,
                        devices_range Devs, const property_list &PropList) {
  {
    auto NoAllowedPropertiesCheck = [](int) { return false; };
    detail::PropertyValidator::checkPropsAndThrow(
        PropList, NoAllowedPropertiesCheck, NoAllowedPropertiesCheck);
  }

  // TODO: Extract compile options from property list once the Spec clarifies
  // how they can be passed.

  // TODO: Probably we could have cached compiled device images.

  // TODO: Handle zero sized Device list.

  auto URDevices = Devs.to<std::vector<ur_device_handle_t>>();

  std::vector<device_image_plain> CompiledImages;
  CompiledImages.reserve(ImgWithDeps.size());
  for (const device_image_plain &DeviceImage : ImgWithDeps.getAll()) {
    device_image_impl &InputImpl = *getSyclObjImpl(DeviceImage);

    adapter_impl &Adapter =
        getSyclObjImpl(InputImpl.get_context())->getAdapter();

    Managed<ur_program_handle_t> Prog =
        createURProgram(*InputImpl.get_bin_image_ref(),
                        *getSyclObjImpl(InputImpl.get_context()), Devs);

    if (InputImpl.get_bin_image_ref()->supportsSpecConstants())
      setSpecializationConstants(InputImpl, Prog, Adapter);

    KernelNameSetT KernelNames = InputImpl.getKernelNames();
    std::map<std::string, KernelArgMask, std::less<>> EliminatedKernelArgMasks =
        InputImpl.getEliminatedKernelArgMasks();

    std::optional<detail::KernelCompilerBinaryInfo> RTCInfo =
        InputImpl.getRTCInfo();
    std::shared_ptr<device_image_impl> ObjectImpl = device_image_impl::create(
        InputImpl.get_bin_image_ref(), InputImpl.get_context(), Devs,
        bundle_state::object, InputImpl.get_kernel_ids_ptr(), std::move(Prog),
        InputImpl.get_spec_const_data_ref(),
        InputImpl.get_spec_const_blob_ref(), InputImpl.getOriginMask(),
        std::move(RTCInfo), std::move(KernelNames),
        std::move(EliminatedKernelArgMasks), nullptr);

    std::string CompileOptions;
    applyCompileOptionsFromEnvironment(CompileOptions);
    appendCompileOptionsFromImage(
        CompileOptions, *(InputImpl.get_bin_image_ref()), Devs, Adapter);
    // Should always come last!
    appendCompileEnvironmentVariablesThatAppend(CompileOptions);
    ur_result_t Error = doCompile(
        Adapter, ObjectImpl->get_ur_program(), Devs.size(), URDevices.data(),
        getSyclObjImpl(InputImpl.get_context()).get()->getHandleRef(),
        CompileOptions.c_str());
    if (Error != UR_RESULT_SUCCESS)
      throw sycl::exception(
          make_error_code(errc::build),
          getProgramBuildLog(ObjectImpl->get_ur_program(),
                             *getSyclObjImpl(ObjectImpl->get_context())));

    CompiledImages.push_back(
        createSyclObjFromImpl<device_image_plain>(std::move(ObjectImpl)));
  }
  return CompiledImages;
}

// Returns a merged device binary image, new set of kernel IDs and new
// specialization constant data.
static const RTDeviceBinaryImage *
mergeImageData(const std::vector<device_image_plain> &Imgs,
               std::vector<kernel_id> &KernelIDs,
               std::vector<unsigned char> &NewSpecConstBlob,
               device_image_impl::SpecConstMapT &NewSpecConstMap,
               std::unique_ptr<DynRTDeviceBinaryImage> &MergedImageStorage) {
  for (const device_image_plain &Img : Imgs) {
    device_image_impl &DeviceImageImpl = *getSyclObjImpl(Img);
    // Duplicates are not expected here, otherwise urProgramLink should fail
    KernelIDs.insert(KernelIDs.end(), DeviceImageImpl.get_kernel_ids().begin(),
                     DeviceImageImpl.get_kernel_ids().end());
    // To be able to answer queries about specialziation constants, the new
    // device image should have the specialization constants from all the linked
    // images.
    const std::lock_guard<std::mutex> SpecConstLock(
        DeviceImageImpl.get_spec_const_data_lock());
    // Copy all map entries to the new map. Since the blob will be copied to
    // the end of the new blob we need to move the blob offset of each entry.
    for (const auto &SpecConstIt : DeviceImageImpl.get_spec_const_data_ref()) {
      std::vector<device_image_impl::SpecConstDescT> &NewDescEntries =
          NewSpecConstMap[SpecConstIt.first];

      if (NewDescEntries.empty()) {
        NewDescEntries.reserve(SpecConstIt.second.size());
        for (const device_image_impl::SpecConstDescT &SpecConstDesc :
             SpecConstIt.second) {
          device_image_impl::SpecConstDescT NewSpecConstDesc = SpecConstDesc;
          NewSpecConstDesc.BlobOffset += NewSpecConstBlob.size();
          NewDescEntries.push_back(std::move(NewSpecConstDesc));
        }
      }
    }

    // Copy the blob from the device image into the new blob. This moves the
    // offsets of the following blobs.
    NewSpecConstBlob.insert(NewSpecConstBlob.end(),
                            DeviceImageImpl.get_spec_const_blob_ref().begin(),
                            DeviceImageImpl.get_spec_const_blob_ref().end());
  }
  // device_image_impl expects kernel ids to be sorted for fast search
  std::sort(KernelIDs.begin(), KernelIDs.end(), LessByHash<kernel_id>{});

  // If there is only a single image, use it as the result.
  if (Imgs.size() == 1)
    return getSyclObjImpl(Imgs[0])->get_bin_image_ref();

  // Otherwise we create a dynamic image with the merged information.
  std::vector<const RTDeviceBinaryImage *> BinImgs;
  BinImgs.reserve(Imgs.size());
  for (const device_image_plain &Img : Imgs) {
    auto ImgBinRef = getSyclObjImpl(Img)->get_bin_image_ref();
    // For some cases, like SYCL kernel compiler binaries, we don't have
    // binaries. For these we assume no properties associated, so they can be
    // safely ignored.
    if (ImgBinRef)
      BinImgs.push_back(ImgBinRef);
  }
  MergedImageStorage = std::make_unique<DynRTDeviceBinaryImage>(BinImgs);
  return MergedImageStorage.get();
}

std::vector<device_image_plain>
ProgramManager::link(const std::vector<device_image_plain> &Imgs,
                     devices_range Devs, const property_list &PropList) {
  {
    auto NoAllowedPropertiesCheck = [](int) { return false; };
    detail::PropertyValidator::checkPropsAndThrow(
        PropList, NoAllowedPropertiesCheck, NoAllowedPropertiesCheck);
  }

  std::vector<ur_program_handle_t> URPrograms;
  URPrograms.reserve(Imgs.size());
  for (const device_image_plain &Img : Imgs)
    URPrograms.push_back(getSyclObjImpl(Img)->get_ur_program());

  auto URDevices = Devs.to<std::vector<ur_device_handle_t>>();

  // FIXME: Linker options are picked from the first object, but is that safe?
  std::string LinkOptionsStr;
  applyLinkOptionsFromEnvironment(LinkOptionsStr);
  device_image_impl &FirstImgImpl = *getSyclObjImpl(Imgs[0]);
  if (LinkOptionsStr.empty() && FirstImgImpl.get_bin_image_ref())
    appendLinkOptionsFromImage(LinkOptionsStr,
                               *(FirstImgImpl.get_bin_image_ref()));
  // Should always come last!
  appendLinkEnvironmentVariablesThatAppend(LinkOptionsStr);
  const context &Context = FirstImgImpl.get_context();
  context_impl &ContextImpl = *getSyclObjImpl(Context);
  adapter_impl &Adapter = ContextImpl.getAdapter();

  Managed<ur_program_handle_t> LinkedProg{Adapter};
  auto doLink = [&] {
    auto Res = Adapter.call_nocheck<UrApiKind::urProgramLinkExp>(
        ContextImpl.getHandleRef(), URDevices.size(), URDevices.data(),
        URPrograms.size(), URPrograms.data(), LinkOptionsStr.c_str(),
        &LinkedProg);
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter.call_nocheck<UrApiKind::urProgramLink>(
          ContextImpl.getHandleRef(), URPrograms.size(), URPrograms.data(),
          LinkOptionsStr.c_str(), &LinkedProg);
    }
    return Res;
  };
  ur_result_t Error = doLink();
  if (Error == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
      Error == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
      Error == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    ContextImpl.getKernelProgramCache().reset();
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
  std::unique_ptr<DynRTDeviceBinaryImage> MergedImageStorage;
  const RTDeviceBinaryImage *NewBinImg = mergeImageData(
      Imgs, *KernelIDs, NewSpecConstBlob, NewSpecConstMap, MergedImageStorage);

  // With both the new program and the merged image data, initialize associated
  // device_global variables.
  ContextImpl.addDeviceGlobalInitializer(LinkedProg, Devs, NewBinImg);

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    // NativePrograms map does not intend to keep reference to program handle,
    // so keys in the map can be invalid (reference count went to zero and the
    // underlying program disposed of). Protecting from incorrect values by
    // removal of map entries with same handle (obviously invalid entries).
    std::ignore = NativePrograms.erase(LinkedProg);
    for (const device_image_plain &Img : Imgs) {
      if (auto BinImageRef = getSyclObjImpl(Img)->get_bin_image_ref())
        NativePrograms.insert(
            {LinkedProg, {ContextImpl.shared_from_this(), BinImageRef}});
    }
  }

  // The origin becomes the combination of all the origins.
  uint8_t CombinedOrigins = 0;
  // For the kernel compiler binary info, we collect pointers to all of the
  // input ones and then merge them afterwards.
  std::vector<const std::optional<detail::KernelCompilerBinaryInfo> *>
      RTCInfoPtrs;
  RTCInfoPtrs.reserve(Imgs.size());
  KernelNameSetT MergedKernelNames;
  std::map<std::string, KernelArgMask, std::less<>>
      MergedEliminatedKernelArgMasks;
  for (const device_image_plain &DevImg : Imgs) {
    device_image_impl &DevImgImpl = *getSyclObjImpl(DevImg);
    CombinedOrigins |= DevImgImpl.getOriginMask();
    RTCInfoPtrs.emplace_back(&(DevImgImpl.getRTCInfo()));
    MergedKernelNames.insert(DevImgImpl.getKernelNames().begin(),
                             DevImgImpl.getKernelNames().end());
    MergedEliminatedKernelArgMasks.insert(
        DevImgImpl.getEliminatedKernelArgMasks().begin(),
        DevImgImpl.getEliminatedKernelArgMasks().end());
  }
  auto MergedRTCInfo = detail::KernelCompilerBinaryInfo::Merge(RTCInfoPtrs);

  // TODO: Make multiple sets of device images organized by devices they are
  // compiled for.
  return {createSyclObjFromImpl<device_image_plain>(device_image_impl::create(
      NewBinImg, Context, Devs, bundle_state::executable, std::move(KernelIDs),
      std::move(LinkedProg), std::move(NewSpecConstMap),
      std::move(NewSpecConstBlob), CombinedOrigins, std::move(MergedRTCInfo),
      std::move(MergedKernelNames), std::move(MergedEliminatedKernelArgMasks),
      std::move(MergedImageStorage)))};
}

// The function duplicates most of the code from existing getBuiltPIProgram.
// The differences are:
// Different API - uses different objects to extract required info
// Supports caching of a program built for multiple devices
device_image_plain
ProgramManager::build(const DevImgPlainWithDeps &DevImgWithDeps,
                      devices_range Devs, const property_list &PropList) {
  {
    auto NoAllowedPropertiesCheck = [](int) { return false; };
    detail::PropertyValidator::checkPropsAndThrow(
        PropList, NoAllowedPropertiesCheck, NoAllowedPropertiesCheck);
  }

  device_image_impl &MainInputImpl = *getSyclObjImpl(DevImgWithDeps.getMain());

  const context &Context = MainInputImpl.get_context();
  context_impl &ContextImpl = *detail::getSyclObjImpl(Context);

  std::vector<const RTDeviceBinaryImage *> BinImgs;
  BinImgs.reserve(DevImgWithDeps.size());
  for (const device_image_plain &DevImg : DevImgWithDeps)
    BinImgs.push_back(getSyclObjImpl(DevImg)->get_bin_image_ref());

  std::shared_ptr<std::vector<kernel_id>> KernelIDs;
  std::vector<unsigned char> SpecConstBlob;
  device_image_impl::SpecConstMapT SpecConstMap;

  std::unique_ptr<DynRTDeviceBinaryImage> MergedImageStorage;
  const RTDeviceBinaryImage *ResultBinImg = MainInputImpl.get_bin_image_ref();
  if (DevImgWithDeps.hasDeps()) {
    KernelIDs = std::make_shared<std::vector<kernel_id>>();
    // Sort the images to make the order of spec constant values used for
    // caching consistent.
    std::vector<device_image_plain> SortedImgs = DevImgWithDeps.getAll();
    std::sort(SortedImgs.begin(), SortedImgs.end(),
              [](const auto &A, const auto &B) {
                return getSyclObjImpl(A)->get_bin_image_ref()->getImageID() <
                       getSyclObjImpl(B)->get_bin_image_ref()->getImageID();
              });
    ResultBinImg = mergeImageData(SortedImgs, *KernelIDs, SpecConstBlob,
                                  SpecConstMap, MergedImageStorage);
  } else {
    KernelIDs = MainInputImpl.get_kernel_ids_ptr();
    SpecConstBlob = MainInputImpl.get_spec_const_blob_ref();
    SpecConstMap = MainInputImpl.get_spec_const_data_ref();
  }

  Managed<ur_program_handle_t> ResProgram = getBuiltURProgram(
      std::move(BinImgs), ContextImpl, Devs, &DevImgWithDeps, SpecConstBlob);

  // The origin becomes the combination of all the origins.
  uint8_t CombinedOrigins = 0;
  for (const device_image_plain &DevImg : DevImgWithDeps)
    CombinedOrigins |= getSyclObjImpl(DevImg)->getOriginMask();

  std::vector<const std::optional<detail::KernelCompilerBinaryInfo> *>
      RTCInfoPtrs;
  RTCInfoPtrs.reserve(DevImgWithDeps.size());
  KernelNameSetT MergedKernelNames;
  std::map<std::string, KernelArgMask, std::less<>>
      MergedEliminatedKernelArgMasks;
  for (const device_image_plain &DevImg : DevImgWithDeps) {
    device_image_impl &DevImgImpl = *getSyclObjImpl(DevImg);
    RTCInfoPtrs.emplace_back(&(DevImgImpl.getRTCInfo()));
    MergedKernelNames.insert(DevImgImpl.getKernelNames().begin(),
                             DevImgImpl.getKernelNames().end());
    MergedEliminatedKernelArgMasks.insert(
        DevImgImpl.getEliminatedKernelArgMasks().begin(),
        DevImgImpl.getEliminatedKernelArgMasks().end());
  }
  auto MergedRTCInfo = detail::KernelCompilerBinaryInfo::Merge(RTCInfoPtrs);

  return createSyclObjFromImpl<device_image_plain>(device_image_impl::create(
      ResultBinImg, Context, Devs, bundle_state::executable,
      std::move(KernelIDs), std::move(ResProgram), std::move(SpecConstMap),
      std::move(SpecConstBlob), CombinedOrigins, std::move(MergedRTCInfo),
      std::move(MergedKernelNames), std::move(MergedEliminatedKernelArgMasks),
      std::move(MergedImageStorage)));
}

// When caching is enabled, the returned UrKernel will already have
// its ref count incremented.
std::tuple<Managed<ur_kernel_handle_t>, std::mutex *, const KernelArgMask *>
ProgramManager::getOrCreateKernel(const context &Context,
                                  KernelNameStrRefT KernelName,
                                  const property_list &PropList,
                                  ur_program_handle_t Program) {

  {
    auto NoAllowedPropertiesCheck = [](int) { return false; };
    detail::PropertyValidator::checkPropsAndThrow(
        PropList, NoAllowedPropertiesCheck, NoAllowedPropertiesCheck);
  }

  context_impl &Ctx = *getSyclObjImpl(Context);

  KernelProgramCache &Cache = Ctx.getKernelProgramCache();

  auto BuildF = [this, &Program, &KernelName, &Ctx] {
    adapter_impl &Adapter = Ctx.getAdapter();
    Managed<ur_kernel_handle_t> Kernel{Adapter};

    Adapter.call<UrApiKind::urKernelCreate>(Program, KernelName.data(),
                                            &Kernel);

    // Only set UR_USM_INDIRECT_ACCESS if the platform can handle it.
    if (Ctx.getPlatformImpl().supports_usm()) {
      bool EnableAccess = true;
      Adapter.call<UrApiKind::urKernelSetExecInfo>(
          Kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, sizeof(ur_bool_t),
          nullptr, &EnableAccess);
    }

    // Ignore possible m_UseSpvFile for now.
    // TODO consider making m_UseSpvFile interact with kernel bundles as well.
    const KernelArgMask *KernelArgMask =
        getEliminatedKernelArgMask(Program, KernelName);

    return std::make_pair(std::move(Kernel), KernelArgMask);
  };

  auto GetCachedBuildF = [&Cache, &KernelName, Program]() {
    return Cache.getOrInsertKernel(Program, KernelName);
  };

  if (!SYCLConfig<SYCL_CACHE_IN_MEM>::get()) {
    // The built kernel cannot be shared between multiple
    // threads when caching is disabled, so we can return
    // nullptr for the mutex.
    auto [Kernel, ArgMask] = BuildF();
    return make_tuple(std::move(Kernel), nullptr, ArgMask);
  }

  std::shared_ptr<KernelProgramCache::KernelBuildResult> BuildResult =
      Cache.getOrBuild<errc::invalid>(GetCachedBuildF, BuildF);
  assert(BuildResult && "getOrBuild isn't supposed to return nullptr!");
  // If caching is enabled, one copy of the kernel handle will be
  // stored in the cache, and one handle is returned to the
  // caller. In that case, we need to increase the ref count of the
  // kernel.
  return std::make_tuple(BuildResult->Val.first.retain(),
                         &(BuildResult->MBuildResultMutex),
                         BuildResult->Val.second);
}

ur_kernel_handle_t ProgramManager::getCachedMaterializedKernel(
    KernelNameStrRefT KernelName,
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
    const device &Device, KernelNameStrRefT KernelName,
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
  context_impl &ContextImpl = *detail::getSyclObjImpl(Context);
  detail::device_impl &DeviceImpl = *detail::getSyclObjImpl(Device);
  adapter_impl &Adapter = DeviceImpl.getAdapter();

  Managed<ur_program_handle_t> ProgramManaged =
      createURProgram(Img, ContextImpl, {Device});

  std::string CompileOpts;
  std::string LinkOpts;
  applyOptionsFromEnvironment(CompileOpts, LinkOpts);
  // No linking of extra programs reqruired.
  std::vector<Managed<ur_program_handle_t>> ExtraProgramsToLink;
  std::vector<ur_device_handle_t> Devs = {DeviceImpl.getHandleRef()};
  auto BuildProgram =
      build(std::move(ProgramManaged), ContextImpl, CompileOpts, LinkOpts, Devs,
            /*For non SPIR-V devices DeviceLibReqdMask is always 0*/ 0,
            ExtraProgramsToLink);
  Managed<ur_kernel_handle_t> UrKernel{Adapter};
  Adapter.call<errc::kernel_not_supported, UrApiKind::urKernelCreate>(
      BuildProgram, KernelName.data(), &UrKernel);
  ur_kernel_handle_t RawUrKernel = UrKernel;
  {
    std::lock_guard<std::mutex> KernelIDsGuard(m_KernelIDsMutex);
    m_MaterializedKernels[KernelName][SpecializationConsts] =
        std::move(UrKernel);
  }

  return RawUrKernel;
}

bool doesDevSupportDeviceRequirements(const device_impl &Dev,
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
checkDevSupportDeviceRequirements(const device_impl &Dev,
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

bool doesImageTargetMatchDevice(const RTDeviceBinaryImage &Img,
                                const device_impl &DevImpl) {
  auto PropRange = Img.getDeviceRequirements();
  auto PropIt =
      std::find_if(PropRange.begin(), PropRange.end(), [&](const auto &Prop) {
        return Prop->Name == std::string_view("compile_target");
      });
  // Device image has no compile_target property, check target.
  if (PropIt == PropRange.end()) {
    sycl::backend BE = DevImpl.getBackend();
    const char *Target = Img.getRawData().DeviceTargetSpec;
    // On Offload, the image format depends on the platform. As with the UR CTS,
    // the easiest way to check this is the platform name which corresponds with
    // the Offload plugin name. In the future the true backend type will be
    // transparently passed through instead.
    if (BE == sycl::backend::ext_oneapi_offload) {
      std::string PlatformName =
          DevImpl.getPlatformImpl().get_info<info::platform::name>();
      if (PlatformName == "CUDA") {
        return (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_NVPTX64) == 0 ||
                strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_LLVM_NVPTX64) == 0);
      }
      if (PlatformName == "AMDGPU") {
        return (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_AMDGCN) == 0 ||
                strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_LLVM_AMDGCN) == 0);
      }
      assert(false && "Unhandled liboffload platform");
      return false;
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0) {
      return (BE == sycl::backend::opencl ||
              BE == sycl::backend::ext_oneapi_level_zero);
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) {
      return DevImpl.is_cpu();
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) {
      return DevImpl.is_gpu() && (BE == sycl::backend::opencl ||
                                  BE == sycl::backend::ext_oneapi_level_zero);
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0) {
      return DevImpl.is_accelerator();
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_NVPTX64) == 0 ||
        strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_LLVM_NVPTX64) == 0) {
      return BE == sycl::backend::ext_oneapi_cuda;
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_AMDGCN) == 0 ||
        strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_LLVM_AMDGCN) == 0) {
      return BE == sycl::backend::ext_oneapi_hip;
    }
    if (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU) == 0) {
      return BE == sycl::backend::ext_oneapi_native_cpu;
    }
    assert(false && "Unexpected image target");
    return false;
  }

  // Device image has the compile_target property, so it is AOT compiled for
  // some device, check if that architecture is Device's architecture.
  auto CompileTargetByteArray = DeviceBinaryProperty(*PropIt).asByteArray();
  // Drop 8 bytes describing the size of the byte array.
  CompileTargetByteArray.dropBytes(8);
  std::string_view CompileTarget(
      reinterpret_cast<const char *>(&CompileTargetByteArray[0]),
      CompileTargetByteArray.size());
  std::string_view ArchName = getArchName(DevImpl);
  // Note: there are no explicit targets for CPUs, so on x86_64,
  // intel_cpu_spr, and intel_cpu_gnr, we use a spir64_x86_64
  // compile target image.
  // TODO: When dedicated targets for CPU are added, (i.e.
  // -fsycl-targets=intel_cpu_spr etc.) remove this special
  // handling of CPU targets.
  return ((ArchName == CompileTarget) ||
          (CompileTarget == "spir64_x86_64" &&
           (ArchName == "x86_64" || ArchName == "intel_cpu_spr" ||
            ArchName == "intel_cpu_gnr")));
}

} // namespace detail
} // namespace _V1
} // namespace sycl

extern "C" void __sycl_register_lib(sycl_device_binaries desc) {
  sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(sycl_device_binaries desc) {
  // Partial cleanup is not necessary at shutdown
#ifndef _WIN32
  if (!sycl::detail::GlobalHandler::instance().isOkToDefer())
    return;
  sycl::detail::ProgramManager::getInstance().removeImages(desc);
#endif
}
