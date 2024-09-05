//==---------- ur.cpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// Implementation of C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#include "context_impl.hpp"
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/plugin.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/version.hpp>

#include <bitset>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stddef.h>
#include <string>
#include <tuple>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace pi {
void contextSetExtendedDeleter(const sycl::context &context,
                               pi_context_extended_deleter func,
                               void *user_data) {
  auto impl = getSyclObjImpl(context);
  const auto &Plugin = impl->getPlugin();
  Plugin->call(urContextSetExtendedDeleter, impl->getHandleRef(),
               reinterpret_cast<ur_context_extended_deleter_t>(func),
               user_data);
}
} // namespace pi

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Global (to the SYCL runtime) graph handle that all command groups are a
// child of
/// Event to be used by graph related activities
xpti_td *GSYCLGraphEvent = nullptr;
#endif // XPTI_ENABLE_INSTRUMENTATION

template <sycl::backend BE>
void *getPluginOpaqueData([[maybe_unused]] void *OpaqueDataParam) {
  // This was formerly a call to piextPluginGetOpaqueData, a deprecated PI entry
  // point introduced for the now deleted ESIMD plugin. All calls to this entry
  // point returned a similar error code to INVALID_OPERATION and would have
  // resulted in a similar throw to this one
  throw exception(
      make_error_code(errc::feature_not_supported),
      "This operation is not supported by any existing backends.");
  return nullptr;
}

namespace ur {
bool trace(TraceLevel Level) {
  auto TraceLevelMask = SYCLConfig<SYCL_UR_TRACE>::get();
  return (TraceLevelMask & Level) == Level;
}

static void initializePlugins(std::vector<PluginPtr> &Plugins,
                              ur_loader_config_handle_t LoaderConfig);

bool XPTIInitDone = false;

// Initializes all available Plugins.
std::vector<PluginPtr> &initializeUr(ur_loader_config_handle_t LoaderConfig) {
  static std::once_flag PluginsInitDone;
  // std::call_once is blocking all other threads if a thread is already
  // creating a vector of plugins. So, no additional lock is needed.
  std::call_once(PluginsInitDone, [&]() {
    // TODO: Remove this SYCL_PI_TRACE notification in the first patch release
    // after the next ABI breaking window.
    if (std::getenv("SYCL_PI_TRACE")) {
      std::cerr << "SYCL_PI_TRACE has been removed use SYCL_UR_TRACE instead\n";
      std::exit(1);
    }

    initializePlugins(GlobalHandler::instance().getPlugins(), LoaderConfig);
  });
  return GlobalHandler::instance().getPlugins();
}

static void initializePlugins(std::vector<PluginPtr> &Plugins,
                              ur_loader_config_handle_t LoaderConfig) {
#define CHECK_UR_SUCCESS(Call)                                                 \
  __SYCL_CHECK_OCL_CODE_NO_EXC(Call)

  bool OwnLoaderConfig = false;
  // If we weren't provided with a custom config handle create our own.
  if(!LoaderConfig) {
    CHECK_UR_SUCCESS(urLoaderConfigCreate(&LoaderConfig))
    OwnLoaderConfig = true;
  }

  const char *LogOptions = "level:info;output:stdout;flush:info";
  if (trace(TraceLevel::TRACE_CALLS)) {
#ifdef _WIN32
    _putenv_s("UR_LOG_TRACING", LogOptions);
#else
    setenv("UR_LOG_TRACING", LogOptions, 1);
#endif
    CHECK_UR_SUCCESS(
        urLoaderConfigEnableLayer(LoaderConfig, "UR_LAYER_TRACING"));
  }

  if (trace(TraceLevel::TRACE_BASIC)) {
#ifdef _WIN32
    _putenv_s("UR_LOG_LOADER", LogOptions);
#else
    setenv("UR_LOG_LOADER", LogOptions, 1);
#endif
  }

  CHECK_UR_SUCCESS(urLoaderConfigSetCodeLocationCallback(
      LoaderConfig, codeLocationCallback, nullptr));

  if (ProgramManager::getInstance().kernelUsesAsan()) {
    if (urLoaderConfigEnableLayer(LoaderConfig, "UR_LAYER_ASAN")) {
      urLoaderConfigRelease(LoaderConfig);
      std::cerr << "Failed to enable ASAN layer\n";
      return;
    }
  }

  urLoaderConfigSetCodeLocationCallback(LoaderConfig, codeLocationCallback,
                                        nullptr);

  if (ProgramManager::getInstance().kernelUsesAsan()) {
    if (urLoaderConfigEnableLayer(LoaderConfig, "UR_LAYER_ASAN")) {
      urLoaderConfigRelease(LoaderConfig);
      std::cerr << "Failed to enable ASAN layer\n";
      return;
    }
  }

  ur_device_init_flags_t device_flags = 0;
  CHECK_UR_SUCCESS(urLoaderInit(device_flags, LoaderConfig));

  if (OwnLoaderConfig) {
    CHECK_UR_SUCCESS(urLoaderConfigRelease(LoaderConfig));
  }

  uint32_t adapterCount = 0;
  CHECK_UR_SUCCESS(urAdapterGet(0, nullptr, &adapterCount));
  std::vector<ur_adapter_handle_t> adapters(adapterCount);
  CHECK_UR_SUCCESS(urAdapterGet(adapterCount, adapters.data(), nullptr));

  auto UrToSyclBackend = [](ur_adapter_backend_t backend) -> sycl::backend {
    switch (backend) {
    case UR_ADAPTER_BACKEND_LEVEL_ZERO:
      return backend::ext_oneapi_level_zero;
    case UR_ADAPTER_BACKEND_OPENCL:
      return backend::opencl;
    case UR_ADAPTER_BACKEND_CUDA:
      return backend::ext_oneapi_cuda;
    case UR_ADAPTER_BACKEND_HIP:
      return backend::ext_oneapi_hip;
    case UR_ADAPTER_BACKEND_NATIVE_CPU:
      return backend::ext_oneapi_native_cpu;
    default:
      // Throw an exception, this should be unreachable.
      CHECK_UR_SUCCESS(UR_RESULT_ERROR_INVALID_ENUMERATION)
      return backend::all;
    }
  };

  for (const auto &adapter : adapters) {
    ur_adapter_backend_t adapterBackend = UR_ADAPTER_BACKEND_UNKNOWN;
    CHECK_UR_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                      sizeof(adapterBackend), &adapterBackend,
                                      nullptr));
    auto syclBackend = UrToSyclBackend(adapterBackend);
    Plugins.emplace_back(std::make_shared<plugin>(adapter, syclBackend));
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();

  if (!(xptiTraceEnabled() && !XPTIInitDone))
    return;
  // Not sure this is the best place to initialize the framework; SYCL runtime
  // team needs to advise on the right place, until then we piggy-back on the
  // initialization of the UR layer.

  // Initialize the global events just once, in the case ur::initialize() is
  // called multiple times
  XPTIInitDone = true;
  // Registers a new stream for 'sycl' and any plugin that wants to listen to
  // this stream will register itself using this string or stream ID for this
  // string.
  uint8_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  //  Let all tool plugins know that a stream by the name of 'sycl' has been
  //  initialized and will be generating the trace stream.
  GlobalHandler::instance().getXPTIRegistry().initializeStream(
      SYCL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  // Create a tracepoint to indicate the graph creation
  xpti::payload_t GraphPayload("application_graph");
  uint64_t GraphInstanceNo;
  GSYCLGraphEvent =
      xptiMakeEvent("application_graph", &GraphPayload, xpti::trace_graph_event,
                    xpti_at::active, &GraphInstanceNo);
  if (GSYCLGraphEvent) {
    // The graph event is a global event and will be used as the parent for
    // all nodes (command groups)
    xptiNotifySubscribers(StreamID, xpti::trace_graph_create, nullptr,
                          GSYCLGraphEvent, GraphInstanceNo, nullptr);
  }
#endif
#undef CHECK_UR_SUCCESS
}

// Get the plugin serving given backend.
template <backend BE> const PluginPtr &getPlugin() {
  static PluginPtr *Plugin = nullptr;
  if (Plugin)
    return *Plugin;

  std::vector<PluginPtr> &Plugins = ur::initializeUr();
  for (auto &P : Plugins)
    if (P->hasBackend(BE)) {
      Plugin = &P;
      return *Plugin;
    }

  throw exception(errc::runtime, "ur::getPlugin couldn't find plugin");
}

template const PluginPtr &getPlugin<backend::opencl>();
template const PluginPtr &getPlugin<backend::ext_oneapi_level_zero>();
template const PluginPtr &getPlugin<backend::ext_oneapi_cuda>();
template const PluginPtr &getPlugin<backend::ext_oneapi_hip>();

// Reads an integer value from ELF data.
template <typename ResT>
static ResT readELFValue(const unsigned char *Data, size_t NumBytes,
                         bool IsBigEndian) {
  assert(NumBytes <= sizeof(ResT));
  ResT Result = 0;
  if (IsBigEndian) {
    for (size_t I = 0; I < NumBytes; ++I) {
      Result = (Result << 8) | static_cast<ResT>(Data[I]);
    }
  } else {
    std::copy(Data, Data + NumBytes, reinterpret_cast<char *>(&Result));
  }
  return Result;
}

// Checks if an ELF image contains a section with a specified name.
static bool checkELFSectionPresent(const std::string &ExpectedSectionName,
                                   const unsigned char *ImgData,
                                   size_t ImgSize) {
  // Check for 64bit and big-endian.
  bool Is64bit = ImgData[4] == 2;
  bool IsBigEndian = ImgData[5] == 2;

  // Make offsets based on whether the ELF file is 64bit or not.
  size_t SectionHeaderOffsetInfoOffset = Is64bit ? 0x28 : 0x20;
  size_t SectionHeaderSizeInfoOffset = Is64bit ? 0x3A : 0x2E;
  size_t SectionHeaderNumInfoOffset = Is64bit ? 0x3C : 0x30;
  size_t SectionStringsHeaderIndexInfoOffset = Is64bit ? 0x3E : 0x32;

  // if the image doesn't contain enough data for the header values, end early.
  if (ImgSize < SectionStringsHeaderIndexInfoOffset + 2)
    return false;

  // Read the e_shoff, e_shentsize, e_shnum, and e_shstrndx entries in the
  // header.
  uint64_t SectionHeaderOffset = readELFValue<uint64_t>(
      ImgData + SectionHeaderOffsetInfoOffset, Is64bit ? 8 : 4, IsBigEndian);
  uint16_t SectionHeaderSize = readELFValue<uint16_t>(
      ImgData + SectionHeaderSizeInfoOffset, 2, IsBigEndian);
  uint16_t SectionHeaderNum = readELFValue<uint16_t>(
      ImgData + SectionHeaderNumInfoOffset, 2, IsBigEndian);
  uint16_t SectionStringsHeaderIndex = readELFValue<uint16_t>(
      ImgData + SectionStringsHeaderIndexInfoOffset, 2, IsBigEndian);

  // End early if we do not have the expected number of section headers or
  // if the read section string header index is out-of-range.
  if (ImgSize < SectionHeaderOffset + static_cast<uint64_t>(SectionHeaderNum) *
                                          SectionHeaderSize ||
      SectionStringsHeaderIndex >= SectionHeaderNum)
    return false;

  // Get the location of the section string data.
  size_t SectionStringsInfoOffset = Is64bit ? 0x18 : 0x10;
  const unsigned char *SectionStringsHeaderData =
      ImgData + SectionHeaderOffset +
      SectionStringsHeaderIndex * SectionHeaderSize;
  uint64_t SectionStrings = readELFValue<uint64_t>(
      SectionStringsHeaderData + SectionStringsInfoOffset, Is64bit ? 8 : 4,
      IsBigEndian);
  const unsigned char *SectionStringsData = ImgData + SectionStrings;

  // For each section, check the name against the expected section and return
  // true if we find it.
  for (size_t I = 0; I < SectionHeaderNum; ++I) {
    // Get the offset into the section string data of this sections name.
    const unsigned char *HeaderData =
        ImgData + SectionHeaderOffset + I * SectionHeaderSize;
    uint32_t SectionNameOffset =
        readELFValue<uint32_t>(HeaderData, 4, IsBigEndian);

    // Read the section name and check if it is the same as the name we are
    // looking for.
    const char *SectionName =
        reinterpret_cast<const char *>(SectionStringsData + SectionNameOffset);
    if (SectionName == ExpectedSectionName)
      return true;
  }
  return false;
}

// Returns the e_type field from an ELF image.
static uint16_t getELFHeaderType(const unsigned char *ImgData, size_t ImgSize) {
  (void)ImgSize;
  assert(ImgSize >= 18 && "Not enough bytes to have an ELF header type.");

  bool IsBigEndian = ImgData[5] == 2;
  return readELFValue<uint16_t>(ImgData + 16, 2, IsBigEndian);
}

sycl_device_binary_type getBinaryImageFormat(const unsigned char *ImgData,
                                             size_t ImgSize) {
  // Top-level magic numbers for the recognized binary image formats.
  auto MatchMagicNumber = [&](auto Number) {
    return ImgSize >= sizeof(Number) &&
           std::memcmp(ImgData, &Number, sizeof(Number)) == 0;
  };

  if (MatchMagicNumber(uint32_t{0x07230203}))
    return SYCL_DEVICE_BINARY_TYPE_SPIRV;

  if (MatchMagicNumber(uint32_t{0xDEC04342}))
    return SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE;

  if (MatchMagicNumber(uint32_t{0x43544E49}))
    // 'I', 'N', 'T', 'C' ; Intel native
    return SYCL_DEVICE_BINARY_TYPE_LLVMIR_BITCODE;

  // Check for ELF format, size requirements include data we'll read in case of
  // succesful match.
  if (ImgSize >= 18 && MatchMagicNumber(uint32_t{0x464c457F})) {
    uint16_t ELFHdrType = getELFHeaderType(ImgData, ImgSize);
    if (ELFHdrType == 0xFF04)
      // OpenCL executable.
      return SYCL_DEVICE_BINARY_TYPE_NATIVE;

    if (ELFHdrType == 0xFF12)
      // ZEBIN executable.
      return SYCL_DEVICE_BINARY_TYPE_NATIVE;

    // Newer ZEBIN format does not have a special header type, but can instead
    // be identified by having a required .ze_info section.
    if (checkELFSectionPresent(".ze_info", ImgData, ImgSize))
      return SYCL_DEVICE_BINARY_TYPE_NATIVE;
  }

  if (MatchMagicNumber(std::array{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}))
    // "ar" format is used to pack binaries for multiple devices, e.g. via
    //
    //   -Xsycl-target-backend=spir64_gen "-device acm-g10,acm-g11"
    //
    // option.
    return SYCL_DEVICE_BINARY_TYPE_NATIVE;

  return SYCL_DEVICE_BINARY_TYPE_NONE;
}

ur_program_metadata_t mapDeviceBinaryPropertyToProgramMetadata(
    const sycl_device_binary_property &DeviceBinaryProperty) {
  ur_program_metadata_t URMetadata{};
  URMetadata.pName = DeviceBinaryProperty->Name;
  URMetadata.size = DeviceBinaryProperty->ValSize;
  switch (DeviceBinaryProperty->Type) {
  case SYCL_PROPERTY_TYPE_UINT32:
    URMetadata.type = UR_PROGRAM_METADATA_TYPE_UINT32;
    URMetadata.value.data32 = DeviceBinaryProperty->ValSize;
    break;
  case SYCL_PROPERTY_TYPE_BYTE_ARRAY:
    URMetadata.type = UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY;
    URMetadata.value.pData = DeviceBinaryProperty->ValAddr;
    break;
  case SYCL_PROPERTY_TYPE_STRING:
    URMetadata.type = UR_PROGRAM_METADATA_TYPE_STRING;
    URMetadata.value.pString =
        reinterpret_cast<char *>(DeviceBinaryProperty->ValAddr);
    break;
  default:
    break;
  }
  return URMetadata;
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
