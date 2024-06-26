//===-- pi.cpp - PI utilities implementation -------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.cpp
/// Implementation of C++ wrappers for PI interface.
///
/// \ingroup sycl_pi

#include "context_impl.hpp"
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/plugin.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/stl_type_traits.hpp>
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
#ifdef XPTI_ENABLE_INSTRUMENTATION
// Global (to the SYCL runtime) graph handle that all command groups are a
// child of
/// Event to be used by graph related activities
xpti_td *GSYCLGraphEvent = nullptr;
/// Event to be used by PI layer related activities
xpti_td *GPICallEvent = nullptr;
/// Event to be used by PI layer calls with arguments
xpti_td *GPIArgCallEvent = nullptr;
xpti_td *GPIArgCallActiveEvent = nullptr;

uint8_t PiCallStreamID = 0;
uint8_t PiDebugCallStreamID = 0;

#endif // XPTI_ENABLE_INSTRUMENTATION

template <sycl::backend BE> void *getPluginOpaqueData(void *OpaqueDataParam) {
  void *ReturnOpaqueData = nullptr;
  const PluginPtr &Plugin = pi::getPlugin<BE>();

  Plugin->call<sycl::detail::PiApiKind::piextPluginGetOpaqueData>(
      OpaqueDataParam, &ReturnOpaqueData);

  return ReturnOpaqueData;
}

namespace pi {

static void initializePlugins(std::vector<PluginPtr> &Plugins);

bool XPTIInitDone = false;

// Implementation of the SYCL PI API call tracing methods that use XPTI
// framework to emit these traces that will be used by tools.
uint64_t emitFunctionBeginTrace(const char *FName) {
  uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // The function_begin and function_end trace point types are defined to
  // trace library API calls and they are currently enabled here for support
  // tools that need the API scope. The methods emitFunctionBeginTrace() and
  // emitFunctionEndTrace() can be extended to also trace the arguments of the
  // PI API call using a trace point type the extends the predefined trace
  // point types.
  //
  // You can use the sample collector in llvm/xptifw/samples/syclpi_collector
  // to print the API traces and also extend them to support  arguments that
  // may be traced later.
  //
  /// Example Usage:
  /// \code{cpp}
  /// // Two diagnostic trace types defined for function begin and function end
  /// // with different semantics than the one in the default trace type list.
  /// typedef enum {
  ///   diagnostic_func_begin = XPTI_TRACE_POINT_BEGIN(0),
  ///   diagnostic_func_end = XPTI_TRACE_POINT_END(0),
  /// }syclpi_extension_t;
  /// ...
  /// uint16_t pi_func_begin =
  ///     xptiRegisterUserDefinedTracePoint("sycl.pi", func_begin);
  /// uint16_t pi_func_end =
  ///     xptiRegisterUserDefinedTracePoint("sycl.pi", func_end);
  /// ...
  /// // Setup argument data for the function being traced
  /// ...
  /// xptiNotifySubscribers(stream_id, pi_func_begin, parent, event, instance,
  ///                       (void *)argument_data);
  /// \endcode
  constexpr uint16_t NotificationTraceType =
      (uint16_t)xpti::trace_point_type_t::function_begin;
  if (xptiCheckTraceEnabled(PiCallStreamID, NotificationTraceType)) {
    CorrelationID = xptiGetUniqueId();
    xptiNotifySubscribers(PiCallStreamID, NotificationTraceType, GPICallEvent,
                          nullptr, CorrelationID,
                          static_cast<const void *>(FName));
  }
#endif // XPTI_ENABLE_INSTRUMENTATION
  return CorrelationID;
}

void emitFunctionEndTrace(uint64_t CorrelationID, const char *FName) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      (uint16_t)xpti::trace_point_type_t::function_end;
  if (xptiCheckTraceEnabled(PiCallStreamID, NotificationTraceType)) {
    // CorrelationID is the unique ID that ties together a function_begin and
    // function_end pair of trace calls. The splitting of a scoped_notify into
    // two function calls incurs an additional overhead as the StreamID must
    // be looked up twice.
    xptiNotifySubscribers(PiCallStreamID, NotificationTraceType, GPICallEvent,
                          nullptr, CorrelationID,
                          static_cast<const void *>(FName));
  }
#endif // XPTI_ENABLE_INSTRUMENTATION
}

uint64_t emitFunctionWithArgsBeginTrace(uint32_t FuncID, const char *FuncName,
                                        unsigned char *ArgsData,
                                        pi_plugin Plugin) {
  uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      (uint16_t)xpti::trace_point_type_t::function_with_args_begin;
  if (xptiCheckTraceEnabled(PiDebugCallStreamID, NotificationTraceType)) {
    xpti::function_with_args_t Payload{FuncID, FuncName, ArgsData, nullptr,
                                       &Plugin};
    {
      detail::tls_code_loc_t Tls;
      auto CodeLoc = Tls.query();
      xpti::payload_t PL = xpti::payload_t(
          CodeLoc.functionName(), CodeLoc.fileName(), CodeLoc.lineNumber(),
          CodeLoc.columnNumber(), nullptr);
      uint64_t InstanceNumber{};
      assert(GPIArgCallActiveEvent == nullptr);
      GPIArgCallActiveEvent =
          xptiMakeEvent("Plugin interface call", &PL, xpti::trace_graph_event,
                        xpti_at::active, &InstanceNumber);
    }

    CorrelationID = xptiGetUniqueId();
    xptiNotifySubscribers(PiDebugCallStreamID, NotificationTraceType,
                          GPIArgCallEvent, GPIArgCallActiveEvent, CorrelationID,
                          &Payload);
  }
#endif
  return CorrelationID;
}

void emitFunctionWithArgsEndTrace(uint64_t CorrelationID, uint32_t FuncID,
                                  const char *FuncName, unsigned char *ArgsData,
                                  pi_result Result, pi_plugin Plugin) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      (uint16_t)xpti::trace_point_type_t::function_with_args_end;
  if (xptiCheckTraceEnabled(PiDebugCallStreamID, NotificationTraceType)) {
    xpti::function_with_args_t Payload{FuncID, FuncName, ArgsData, &Result,
                                       &Plugin};

    xptiNotifySubscribers(PiDebugCallStreamID, NotificationTraceType,
                          GPIArgCallEvent, GPIArgCallActiveEvent, CorrelationID,
                          &Payload);
    GPIArgCallActiveEvent = nullptr;
  }
#endif
}

void contextSetExtendedDeleter(const sycl::context &context,
                               pi_context_extended_deleter func,
                               void *user_data) {
  auto impl = getSyclObjImpl(context);
  auto contextHandle = reinterpret_cast<pi_context>(impl->getHandleRef());
  const auto &Plugin = impl->getPlugin();
  Plugin->call<PiApiKind::piextContextSetExtendedDeleter>(contextHandle, func,
                                                          user_data);
}

std::string platformInfoToString(pi_platform_info info) {
  switch (info) {
  case PI_PLATFORM_INFO_PROFILE:
    return "PI_PLATFORM_INFO_PROFILE";
  case PI_PLATFORM_INFO_VERSION:
    return "PI_PLATFORM_INFO_VERSION";
  case PI_PLATFORM_INFO_NAME:
    return "PI_PLATFORM_INFO_NAME";
  case PI_PLATFORM_INFO_VENDOR:
    return "PI_PLATFORM_INFO_VENDOR";
  case PI_PLATFORM_INFO_EXTENSIONS:
    return "PI_PLATFORM_INFO_EXTENSIONS";
  case PI_EXT_PLATFORM_INFO_BACKEND:
    return "PI_EXT_PLATFORM_INFO_BACKEND";
  }
  die("Unknown pi_platform_info value passed to "
      "sycl::detail::pi::platformInfoToString");
}

std::string memFlagToString(pi_mem_flags Flag) {
  assertion(((Flag == 0u) || ((Flag & (Flag - 1)) == 0)) &&
            "More than one bit set");

  std::stringstream Sstream;

  switch (Flag) {
  case pi_mem_flags{0}:
    Sstream << "pi_mem_flags(0)";
    break;
  case PI_MEM_FLAGS_ACCESS_RW:
    Sstream << "PI_MEM_FLAGS_ACCESS_RW";
    break;
  case PI_MEM_FLAGS_HOST_PTR_USE:
    Sstream << "PI_MEM_FLAGS_HOST_PTR_USE";
    break;
  case PI_MEM_FLAGS_HOST_PTR_COPY:
    Sstream << "PI_MEM_FLAGS_HOST_PTR_COPY";
    break;
  default:
    Sstream << "unknown pi_mem_flags bit == " << Flag;
  }

  return Sstream.str();
}

std::string memFlagsToString(pi_mem_flags Flags) {
  std::stringstream Sstream;
  bool FoundFlag = false;

  auto FlagSeparator = [](bool FoundFlag) { return FoundFlag ? "|" : ""; };

  pi_mem_flags ValidFlags[] = {PI_MEM_FLAGS_ACCESS_RW,
                               PI_MEM_FLAGS_HOST_PTR_USE,
                               PI_MEM_FLAGS_HOST_PTR_COPY};

  if (Flags == 0u) {
    Sstream << "pi_mem_flags(0)";
  } else {
    for (const auto Flag : ValidFlags) {
      if (Flag & Flags) {
        Sstream << FlagSeparator(FoundFlag) << memFlagToString(Flag);
        FoundFlag = true;
      }
    }

    std::bitset<64> UnkownBits(Flags & ~(PI_MEM_FLAGS_ACCESS_RW |
                                         PI_MEM_FLAGS_HOST_PTR_USE |
                                         PI_MEM_FLAGS_HOST_PTR_COPY));
    if (UnkownBits.any()) {
      Sstream << FlagSeparator(FoundFlag)
              << "unknown pi_mem_flags bits == " << UnkownBits;
    }
  }

  return Sstream.str();
}

// GlobalPlugin is a global Plugin used with Interoperability constructors that
// use OpenCL objects to construct SYCL class objects.
// TODO: GlobalPlugin does not seem to be needed anymore. Consider removing it!
std::shared_ptr<plugin> GlobalPlugin;

// Find the plugin at the appropriate location and return the location.
std::vector<std::pair<std::string, backend>> findPlugins() {
  std::vector<std::pair<std::string, backend>> PluginNames;

  // TODO: Based on final design discussions, change the location where the
  // plugin must be searched; how to identify the plugins etc. Currently the
  // search is done for libpi_opencl.so/pi_opencl.dll file in LD_LIBRARY_PATH
  // env only.
  //
  ods_target_list *OdsTargetList = SYCLConfig<ONEAPI_DEVICE_SELECTOR>::get();
  if (!OdsTargetList) {
    PluginNames.emplace_back(__SYCL_OPENCL_PLUGIN_NAME, backend::opencl);
    PluginNames.emplace_back(__SYCL_LEVEL_ZERO_PLUGIN_NAME,
                             backend::ext_oneapi_level_zero);
    PluginNames.emplace_back(__SYCL_CUDA_PLUGIN_NAME, backend::ext_oneapi_cuda);
    PluginNames.emplace_back(__SYCL_HIP_PLUGIN_NAME, backend::ext_oneapi_hip);
    PluginNames.emplace_back(__SYCL_UR_PLUGIN_NAME, backend::all);
    PluginNames.emplace_back(__SYCL_NATIVE_CPU_PLUGIN_NAME,
                             backend::ext_oneapi_native_cpu);

  } else {
    ods_target_list &list = *OdsTargetList;
    if (list.backendCompatible(backend::opencl)) {
      PluginNames.emplace_back(__SYCL_OPENCL_PLUGIN_NAME, backend::opencl);
    }
    if (list.backendCompatible(backend::ext_oneapi_level_zero)) {
      PluginNames.emplace_back(__SYCL_LEVEL_ZERO_PLUGIN_NAME,
                               backend::ext_oneapi_level_zero);
    }
    if (list.backendCompatible(backend::ext_oneapi_cuda)) {
      PluginNames.emplace_back(__SYCL_CUDA_PLUGIN_NAME,
                               backend::ext_oneapi_cuda);
    }
    if (list.backendCompatible(backend::ext_oneapi_hip)) {
      PluginNames.emplace_back(__SYCL_HIP_PLUGIN_NAME, backend::ext_oneapi_hip);
    }
    if (list.backendCompatible(backend::ext_oneapi_native_cpu)) {
      PluginNames.emplace_back(__SYCL_NATIVE_CPU_PLUGIN_NAME,
                               backend::ext_oneapi_native_cpu);
    }
    PluginNames.emplace_back(__SYCL_UR_PLUGIN_NAME, backend::all);
  }
  return PluginNames;
}

// Load the Plugin by calling the OS dependent library loading call.
// Return the handle to the Library.
void *loadPlugin(const std::string &PluginPath) {
  return loadOsPluginLibrary(PluginPath);
}

// Unload the given plugin by calling teh OS-specific library unloading call.
// \param Library OS-specific library handle created when loading.
int unloadPlugin(void *Library) { return unloadOsPluginLibrary(Library); }

// Binds all the PI Interface APIs to Plugin Library Function Addresses.
// TODO: Remove the 'OclPtr' extension to PI_API.
// TODO: Change the functionality such that a single getOsLibraryFuncAddress
// call is done to get all Interface API mapping. The plugin interface also
// needs to setup infrastructure to route PI_CALLs to the appropriate plugins.
// Currently, we bind to a singe plugin.
bool bindPlugin(void *Library,
                const std::shared_ptr<PiPlugin> &PluginInformation) {

  decltype(::piPluginInit) *PluginInitializeFunction =
      (decltype(&::piPluginInit))(getOsLibraryFuncAddress(Library,
                                                          "piPluginInit"));
  if (PluginInitializeFunction == nullptr)
    return false;

  int Err = PluginInitializeFunction(PluginInformation.get());

  // TODO: Compare Supported versions and check for backward compatibility.
  // Make sure err is PI_SUCCESS.
  assert((Err == PI_SUCCESS) && "Unexpected error when binding to Plugin.");
  (void)Err;

  // TODO: Return a more meaningful value/enum.
  return true;
}

bool trace(TraceLevel Level) {
  auto TraceLevelMask = SYCLConfig<SYCL_PI_TRACE>::get();
  return (TraceLevelMask & Level) == Level;
}

// Initializes all available Plugins.
std::vector<PluginPtr> &initialize() {
  // This uses static variable initialization to work around a gcc bug with
  // std::call_once and exceptions.
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66146
  auto initializeHelper = []() {
    initializePlugins(GlobalHandler::instance().getPlugins());
    return true;
  };
  static bool Initialized = initializeHelper();
  std::ignore = Initialized;

  return GlobalHandler::instance().getPlugins();
}

// Implementation of this function is OS specific. Please see windows_pi.cpp and
// posix_pi.cpp.
// TODO: refactor code when support matrix for DPCPP changes and <filesystem> is
// available on all supported systems.
std::vector<std::tuple<std::string, backend, void *>>
loadPlugins(const std::vector<std::pair<std::string, backend>> &&PluginNames);

static void initializePlugins(std::vector<PluginPtr> &Plugins) {
  const std::vector<std::pair<std::string, backend>> PluginNames =
      findPlugins();

  if (PluginNames.empty() && trace(PI_TRACE_ALL))
    std::cerr << "SYCL_PI_TRACE[all]: "
              << "No Plugins Found." << std::endl;

  // Get library handles for the list of plugins.
  std::vector<std::tuple<std::string, backend, void *>> LoadedPlugins =
      loadPlugins(std::move(PluginNames));

  bool IsAsanUsed = ProgramManager::getInstance().kernelUsesAsan();

  for (auto &[Name, Backend, Library] : LoadedPlugins) {
    std::shared_ptr<PiPlugin> PluginInformation =
        std::make_shared<PiPlugin>(PiPlugin{
            _PI_H_VERSION_STRING, _PI_H_VERSION_STRING,
            /*Targets=*/nullptr, /*FunctionPointers=*/{},
            /*IsAsanUsed*/
            IsAsanUsed ? _PI_SANITIZE_TYPE_ADDRESS : _PI_SANITIZE_TYPE_NONE});

    if (!Library) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[all]: "
                  << "Check if plugin is present. "
                  << "Failed to load plugin: " << Name << std::endl;
      }
      continue;
    }

    if (!bindPlugin(Library, PluginInformation)) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[all]: "
                  << "Failed to bind PI APIs to the plugin: " << Name
                  << std::endl;
      }
      continue;
    }
    PluginPtr &NewPlugin = Plugins.emplace_back(
        std::make_shared<plugin>(PluginInformation, Backend, Library));
    if (trace(TraceLevel::PI_TRACE_BASIC))
      std::cerr << "SYCL_PI_TRACE[basic]: "
                << "Plugin found and successfully loaded: " << Name
                << " [ PluginVersion: "
                << NewPlugin->getPiPlugin().PluginVersion << " ]" << std::endl;
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();

  if (!(xptiTraceEnabled() && !XPTIInitDone))
    return;
  // Not sure this is the best place to initialize the framework; SYCL runtime
  // team needs to advise on the right place, until then we piggy-back on the
  // initialization of the PI layer.

  // Initialize the global events just once, in the case pi::initialize() is
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

  // Let subscribers know a new stream is being initialized
  GlobalHandler::instance().getXPTIRegistry().initializeStream(
      SYCL_PICALL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  xpti::payload_t PIPayload("Plugin Interface Layer");
  uint64_t PiInstanceNo;
  GPICallEvent =
      xptiMakeEvent("PI Layer", &PIPayload, xpti::trace_algorithm_event,
                    xpti_at::active, &PiInstanceNo);

  GlobalHandler::instance().getXPTIRegistry().initializeStream(
      SYCL_PIDEBUGCALL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  xpti::payload_t PIArgPayload(
      "Plugin Interface Layer (with function arguments)");
  uint64_t PiArgInstanceNo;
  GPIArgCallEvent = xptiMakeEvent("PI Layer with arguments", &PIArgPayload,
                                  xpti::trace_algorithm_event, xpti_at::active,
                                  &PiArgInstanceNo);

  PiCallStreamID = xptiRegisterStream(SYCL_PICALL_STREAM_NAME);
  PiDebugCallStreamID = xptiRegisterStream(SYCL_PIDEBUGCALL_STREAM_NAME);
#endif
}

// Get the plugin serving given backend.
template <backend BE> const PluginPtr &getPlugin() {
  static PluginPtr *Plugin = nullptr;
  if (Plugin)
    return *Plugin;

  std::vector<PluginPtr> &Plugins = pi::initialize();
  for (auto &P : Plugins)
    if (P->hasBackend(BE)) {
      Plugin = &P;
      return *Plugin;
    }

  throw runtime_error("pi::getPlugin couldn't find plugin",
                      PI_ERROR_INVALID_OPERATION);
}

template __SYCL_EXPORT const PluginPtr &getPlugin<backend::opencl>();
template __SYCL_EXPORT const PluginPtr &
getPlugin<backend::ext_oneapi_level_zero>();
template __SYCL_EXPORT const PluginPtr &getPlugin<backend::ext_oneapi_cuda>();
template __SYCL_EXPORT const PluginPtr &getPlugin<backend::ext_oneapi_hip>();

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message) {
  std::cerr << "pi_die: " << Message << std::endl;
  std::terminate();
}

void assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

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
  if (ImgSize < SectionHeaderOffset + SectionHeaderNum * SectionHeaderSize ||
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

sycl::detail::pi::PiDeviceBinaryType
getBinaryImageFormat(const unsigned char *ImgData, size_t ImgSize) {
  // Top-level magic numbers for the recognized binary image formats.
  auto MatchMagicNumber = [&](auto Number) {
    return ImgSize >= sizeof(Number) &&
           std::memcmp(ImgData, &Number, sizeof(Number)) == 0;
  };

  if (MatchMagicNumber(uint32_t{0x07230203}))
    return PI_DEVICE_BINARY_TYPE_SPIRV;

  if (MatchMagicNumber(uint32_t{0xDEC04342}))
    return PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE;

  if (MatchMagicNumber(uint32_t{0x43544E49}))
    // 'I', 'N', 'T', 'C' ; Intel native
    return PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE;

  // Check for ELF format, size requirements include data we'll read in case of
  // succesful match.
  if (ImgSize >= 18 && MatchMagicNumber(uint32_t{0x464c457F})) {
    uint16_t ELFHdrType = getELFHeaderType(ImgData, ImgSize);
    if (ELFHdrType == 0xFF04)
      // OpenCL executable.
      return PI_DEVICE_BINARY_TYPE_NATIVE;

    if (ELFHdrType == 0xFF12)
      // ZEBIN executable.
      return PI_DEVICE_BINARY_TYPE_NATIVE;

    // Newer ZEBIN format does not have a special header type, but can instead
    // be identified by having a required .ze_info section.
    if (checkELFSectionPresent(".ze_info", ImgData, ImgSize))
      return PI_DEVICE_BINARY_TYPE_NATIVE;
  }

  if (MatchMagicNumber(std::array{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}))
    // "ar" format is used to pack binaries for multiple devices, e.g. via
    //
    //   -Xsycl-target-backend=spir64_gen "-device acm-g10,acm-g11"
    //
    // option.
    return PI_DEVICE_BINARY_TYPE_NATIVE;

  return PI_DEVICE_BINARY_TYPE_NONE;
}

} // namespace pi
} // namespace detail
} // namespace _V1
} // namespace sycl
