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
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp>
#include <CL/sycl/version.hpp>
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/plugin.hpp>
#include <detail/xpti_registry.hpp>

#include <bitset>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stddef.h>
#include <string>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

#define STR(x) #x
#define SYCL_VERSION_STR                                                       \
  "sycl " STR(__LIBSYCL_MAJOR_VERSION) "." STR(__LIBSYCL_MINOR_VERSION)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
/// Constants being used as placeholder until one is able to reliably get the
/// version of the SYCL runtime
constexpr uint32_t GMajVer = __LIBSYCL_MAJOR_VERSION;
constexpr uint32_t GMinVer = __LIBSYCL_MINOR_VERSION;
constexpr const char *GVerStr = SYCL_VERSION_STR;
#endif // XPTI_ENABLE_INSTRUMENTATION

template <cl::sycl::backend BE>
void *getPluginOpaqueData(void *OpaqueDataParam) {
  void *ReturnOpaqueData = nullptr;
  const cl::sycl::detail::plugin &Plugin =
      cl::sycl::detail::pi::getPlugin<BE>();

  Plugin.call<cl::sycl::detail::PiApiKind::piextPluginGetOpaqueData>(
      OpaqueDataParam, &ReturnOpaqueData);

  return ReturnOpaqueData;
}

template __SYCL_EXPORT void *
getPluginOpaqueData<cl::sycl::backend::ext_intel_esimd_emulator>(void *);

namespace pi {

static void initializePlugins(std::vector<plugin> &Plugins);

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
  if (xptiTraceEnabled()) {
    uint8_t StreamID = xptiRegisterStream(SYCL_PICALL_STREAM_NAME);
    CorrelationID = xptiGetUniqueId();
    xptiNotifySubscribers(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_begin,
        GPICallEvent, nullptr, CorrelationID, static_cast<const void *>(FName));
  }
#endif // XPTI_ENABLE_INSTRUMENTATION
  return CorrelationID;
}

void emitFunctionEndTrace(uint64_t CorrelationID, const char *FName) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    // CorrelationID is the unique ID that ties together a function_begin and
    // function_end pair of trace calls. The splitting of a scoped_notify into
    // two function calls incurs an additional overhead as the StreamID must
    // be looked up twice.
    uint8_t StreamID = xptiRegisterStream(SYCL_PICALL_STREAM_NAME);
    xptiNotifySubscribers(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_end,
        GPICallEvent, nullptr, CorrelationID, static_cast<const void *>(FName));
  }
#endif // XPTI_ENABLE_INSTRUMENTATION
}

uint64_t emitFunctionWithArgsBeginTrace(uint32_t FuncID, const char *FuncName,
                                        unsigned char *ArgsData,
                                        pi_plugin Plugin) {
  uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    uint8_t StreamID = xptiRegisterStream(SYCL_PIDEBUGCALL_STREAM_NAME);
    CorrelationID = xptiGetUniqueId();

    xpti::function_with_args_t Payload{FuncID, FuncName, ArgsData, nullptr,
                                       &Plugin};

    xptiNotifySubscribers(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_begin,
        GPIArgCallEvent, nullptr, CorrelationID, &Payload);
  }
#endif
  return CorrelationID;
}

void emitFunctionWithArgsEndTrace(uint64_t CorrelationID, uint32_t FuncID,
                                  const char *FuncName, unsigned char *ArgsData,
                                  pi_result Result, pi_plugin Plugin) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    uint8_t StreamID = xptiRegisterStream(SYCL_PIDEBUGCALL_STREAM_NAME);

    xpti::function_with_args_t Payload{FuncID, FuncName, ArgsData, &Result,
                                       &Plugin};

    xptiNotifySubscribers(
        StreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_end,
        GPIArgCallEvent, nullptr, CorrelationID, &Payload);
  }
#endif
}

void contextSetExtendedDeleter(const cl::sycl::context &context,
                               pi_context_extended_deleter func,
                               void *user_data) {
  auto impl = getSyclObjImpl(context);
  auto contextHandle = reinterpret_cast<pi_context>(impl->getHandleRef());
  auto plugin = impl->getPlugin();
  plugin.call<PiApiKind::piextContextSetExtendedDeleter>(contextHandle, func,
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
  }
  die("Unknown pi_platform_info value passed to "
      "cl::sycl::detail::pi::platformInfoToString");
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
std::shared_ptr<plugin> GlobalPlugin;

// Find the plugin at the appropriate location and return the location.
std::vector<std::pair<std::string, backend>> findPlugins() {
  std::vector<std::pair<std::string, backend>> PluginNames;

  // TODO: Based on final design discussions, change the location where the
  // plugin must be searched; how to identify the plugins etc. Currently the
  // search is done for libpi_opencl.so/pi_opencl.dll file in LD_LIBRARY_PATH
  // env only.
  //
  device_filter_list *FilterList = SYCLConfig<SYCL_DEVICE_FILTER>::get();
  if (!FilterList) {
    PluginNames.emplace_back(__SYCL_OPENCL_PLUGIN_NAME, backend::opencl);
    PluginNames.emplace_back(__SYCL_LEVEL_ZERO_PLUGIN_NAME,
                             backend::ext_oneapi_level_zero);
    PluginNames.emplace_back(__SYCL_CUDA_PLUGIN_NAME, backend::ext_oneapi_cuda);
    PluginNames.emplace_back(__SYCL_HIP_PLUGIN_NAME, backend::ext_oneapi_hip);
    PluginNames.emplace_back(__SYCL_ESIMD_EMULATOR_PLUGIN_NAME,
                             backend::ext_intel_esimd_emulator);
  } else {
    std::vector<device_filter> Filters = FilterList->get();
    bool OpenCLFound = false;
    bool LevelZeroFound = false;
    bool CudaFound = false;
    bool EsimdCpuFound = false;
    bool HIPFound = false;
    for (const device_filter &Filter : Filters) {
      backend Backend = Filter.Backend;
      if (!OpenCLFound &&
          (Backend == backend::opencl || Backend == backend::all)) {
        PluginNames.emplace_back(__SYCL_OPENCL_PLUGIN_NAME, backend::opencl);
        OpenCLFound = true;
      }
      if (!LevelZeroFound && (Backend == backend::ext_oneapi_level_zero ||
                              Backend == backend::all)) {
        PluginNames.emplace_back(__SYCL_LEVEL_ZERO_PLUGIN_NAME,
                                 backend::ext_oneapi_level_zero);
        LevelZeroFound = true;
      }
      if (!CudaFound &&
          (Backend == backend::ext_oneapi_cuda || Backend == backend::all)) {
        PluginNames.emplace_back(__SYCL_CUDA_PLUGIN_NAME,
                                 backend::ext_oneapi_cuda);
        CudaFound = true;
      }
      if (!EsimdCpuFound && (Backend == backend::ext_intel_esimd_emulator ||
                             Backend == backend::all)) {
        PluginNames.emplace_back(__SYCL_ESIMD_EMULATOR_PLUGIN_NAME,
                                 backend::ext_intel_esimd_emulator);
        EsimdCpuFound = true;
      }
      if (!HIPFound &&
          (Backend == backend::ext_oneapi_hip || Backend == backend::all)) {
        PluginNames.emplace_back(__SYCL_HIP_PLUGIN_NAME,
                                 backend::ext_oneapi_hip);
        HIPFound = true;
      }
    }
  }
  return PluginNames;
}

// Load the Plugin by calling the OS dependent library loading call.
// Return the handle to the Library.
void *loadPlugin(const std::string &PluginPath) {
  return loadOsLibrary(PluginPath);
}

// Unload the given plugin by calling teh OS-specific library unloading call.
// \param Library OS-specific library handle created when loading.
int unloadPlugin(void *Library) { return unloadOsLibrary(Library); }

// Binds all the PI Interface APIs to Plugin Library Function Addresses.
// TODO: Remove the 'OclPtr' extension to PI_API.
// TODO: Change the functionality such that a single getOsLibraryFuncAddress
// call is done to get all Interface API mapping. The plugin interface also
// needs to setup infrastructure to route PI_CALLs to the appropriate plugins.
// Currently, we bind to a singe plugin.
bool bindPlugin(void *Library,
                const std::shared_ptr<PiPlugin> &PluginInformation) {

  decltype(::piPluginInit) *PluginInitializeFunction = (decltype(
      &::piPluginInit))(getOsLibraryFuncAddress(Library, "piPluginInit"));
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
std::vector<plugin> &initialize() {
  static std::once_flag PluginsInitDone;
  // std::call_once is blocking all other threads if a thread is already
  // creating a vector of plugins. So, no additional lock is needed.
  std::call_once(PluginsInitDone, [&]() {
    initializePlugins(GlobalHandler::instance().getPlugins());
  });
  return GlobalHandler::instance().getPlugins();
}

static void initializePlugins(std::vector<plugin> &Plugins) {
  std::vector<std::pair<std::string, backend>> PluginNames = findPlugins();

  if (PluginNames.empty() && trace(PI_TRACE_ALL))
    std::cerr << "SYCL_PI_TRACE[all]: "
              << "No Plugins Found." << std::endl;

  for (unsigned int I = 0; I < PluginNames.size(); I++) {
    std::shared_ptr<PiPlugin> PluginInformation = std::make_shared<PiPlugin>(
        PiPlugin{_PI_H_VERSION_STRING, _PI_H_VERSION_STRING,
                 /*Targets=*/nullptr, /*FunctionPointers=*/{}});

    void *Library = loadPlugin(PluginNames[I].first);

    if (!Library) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[all]: "
                  << "Check if plugin is present. "
                  << "Failed to load plugin: " << PluginNames[I].first
                  << std::endl;
      }
      continue;
    }

    if (!bindPlugin(Library, PluginInformation)) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[all]: "
                  << "Failed to bind PI APIs to the plugin: "
                  << PluginNames[I].first << std::endl;
      }
      continue;
    }
    backend *BE = SYCLConfig<SYCL_BE>::get();
    // Use OpenCL as the default interoperability plugin.
    // This will go away when we make backend interoperability selection
    // explicit in SYCL-2020.
    backend InteropBE = BE ? *BE : backend::opencl;

    if (InteropBE == backend::opencl &&
        PluginNames[I].first.find("opencl") != std::string::npos) {
      // Use the OpenCL plugin as the GlobalPlugin
      GlobalPlugin =
          std::make_shared<plugin>(PluginInformation, backend::opencl, Library);
    } else if (InteropBE == backend::ext_oneapi_cuda &&
               PluginNames[I].first.find("cuda") != std::string::npos) {
      // Use the CUDA plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(
          PluginInformation, backend::ext_oneapi_cuda, Library);
    } else if (InteropBE == backend::ext_oneapi_hip &&
               PluginNames[I].first.find("hip") != std::string::npos) {
      // Use the HIP plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(PluginInformation,
                                              backend::ext_oneapi_hip, Library);
    } else if (InteropBE == backend::ext_oneapi_level_zero &&
               PluginNames[I].first.find("level_zero") != std::string::npos) {
      // Use the LEVEL_ZERO plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(
          PluginInformation, backend::ext_oneapi_level_zero, Library);
    } else if (InteropBE == backend::ext_intel_esimd_emulator &&
               PluginNames[I].first.find("esimd_emulator") !=
                   std::string::npos) {
      // Use the ESIMD_EMULATOR plugin as the GlobalPlugin
      GlobalPlugin = std::make_shared<plugin>(
          PluginInformation, backend::ext_intel_esimd_emulator, Library);
    }
    Plugins.emplace_back(
        plugin(PluginInformation, PluginNames[I].second, Library));
    if (trace(TraceLevel::PI_TRACE_BASIC))
      std::cerr << "SYCL_PI_TRACE[basic]: "
                << "Plugin found and successfully loaded: "
                << PluginNames[I].first << std::endl;
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
#endif
}

// Get the plugin serving given backend.
template <backend BE> const plugin &getPlugin() {
  static const plugin *Plugin = nullptr;
  if (Plugin)
    return *Plugin;

  const std::vector<plugin> &Plugins = pi::initialize();
  for (const auto &P : Plugins)
    if (P.getBackend() == BE) {
      Plugin = &P;
      return *Plugin;
    }

  throw runtime_error("pi::getPlugin couldn't find plugin",
                      PI_INVALID_OPERATION);
}

template __SYCL_EXPORT const plugin &getPlugin<backend::opencl>();
template __SYCL_EXPORT const plugin &
getPlugin<backend::ext_oneapi_level_zero>();
template __SYCL_EXPORT const plugin &
getPlugin<backend::ext_intel_esimd_emulator>();
template __SYCL_EXPORT const plugin &getPlugin<backend::ext_oneapi_cuda>();

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

std::ostream &operator<<(std::ostream &Out, const DeviceBinaryProperty &P) {
  switch (P.Prop->Type) {
  case PI_PROPERTY_TYPE_UINT32:
    Out << "[UINT32] ";
    break;
  case PI_PROPERTY_TYPE_BYTE_ARRAY:
    Out << "[Byte array] ";
    break;
  case PI_PROPERTY_TYPE_STRING:
    Out << "[String] ";
    break;
  default:
    assert(false && "unsupported property");
    return Out;
  }
  Out << P.Prop->Name << "=";

  switch (P.Prop->Type) {
  case PI_PROPERTY_TYPE_UINT32:
    Out << P.asUint32();
    break;
  case PI_PROPERTY_TYPE_BYTE_ARRAY: {
    ByteArray BA = P.asByteArray();
    std::ios_base::fmtflags FlagsBackup = Out.flags();
    Out << std::hex;
    for (const auto &Byte : BA) {
      Out << "0x" << static_cast<unsigned>(Byte) << " ";
    }
    Out.flags(FlagsBackup);
    break;
  }
  case PI_PROPERTY_TYPE_STRING:
    Out << P.asCString();
    break;
  default:
    assert(false && "Unsupported property");
    return Out;
  }
  return Out;
}

void DeviceBinaryImage::print() const {
  std::cerr << "  --- Image " << Bin << "\n";
  if (!Bin)
    return;
  std::cerr << "    Version  : " << (int)Bin->Version << "\n";
  std::cerr << "    Kind     : " << (int)Bin->Kind << "\n";
  std::cerr << "    Format   : " << (int)Bin->Format << "\n";
  std::cerr << "    Target   : " << Bin->DeviceTargetSpec << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Bin->BinaryEnd - (intptr_t)Bin->BinaryStart) << "\n";
  std::cerr << "    Compile options : "
            << (Bin->CompileOptions ? Bin->CompileOptions : "NULL") << "\n";
  std::cerr << "    Link options    : "
            << (Bin->LinkOptions ? Bin->LinkOptions : "NULL") << "\n";
  std::cerr << "    Entries  : ";
  for (_pi_offload_entry EntriesIt = Bin->EntriesBegin;
       EntriesIt != Bin->EntriesEnd; ++EntriesIt)
    std::cerr << EntriesIt->name << " ";
  std::cerr << "\n";
  std::cerr << "    Properties [" << Bin->PropertySetsBegin << "-"
            << Bin->PropertySetsEnd << "]:\n";

  for (pi_device_binary_property_set PS = Bin->PropertySetsBegin;
       PS != Bin->PropertySetsEnd; ++PS) {
    std::cerr << "      Category " << PS->Name << " [" << PS->PropertiesBegin
              << "-" << PS->PropertiesEnd << "]:\n";

    for (pi_device_binary_property P = PS->PropertiesBegin;
         P != PS->PropertiesEnd; ++P) {
      std::cerr << "        " << DeviceBinaryProperty(P) << "\n";
    }
  }
}

void DeviceBinaryImage::dump(std::ostream &Out) const {
  size_t ImgSize = getSize();
  Out.write(reinterpret_cast<const char *>(Bin->BinaryStart), ImgSize);
}

static pi_uint32 asUint32(const void *Addr) {
  assert(Addr && "Addr is NULL");
  const auto *P = reinterpret_cast<const unsigned char *>(Addr);
  return (*P) | (*(P + 1) << 8) | (*(P + 2) << 16) | (*(P + 3) << 24);
}

pi_uint32 DeviceBinaryProperty::asUint32() const {
  assert(Prop->Type == PI_PROPERTY_TYPE_UINT32 && "property type mismatch");
  // if type fits into the ValSize - it is used to store the property value
  assert(Prop->ValAddr == nullptr && "primitive types must be stored inline");
  return sycl::detail::pi::asUint32(&Prop->ValSize);
}

ByteArray DeviceBinaryProperty::asByteArray() const {
  assert(Prop->Type == PI_PROPERTY_TYPE_BYTE_ARRAY && "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  const auto *Data = pi::cast<const std::uint8_t *>(Prop->ValAddr);
  return {Data, Prop->ValSize};
}

const char *DeviceBinaryProperty::asCString() const {
  assert(Prop->Type == PI_PROPERTY_TYPE_STRING && "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  return pi::cast<const char *>(Prop->ValAddr);
}

void DeviceBinaryImage::PropertyRange::init(pi_device_binary Bin,
                                            const char *PropSetName) {
  assert(!this->Begin && !this->End && "already initialized");
  pi_device_binary_property_set PS = nullptr;

  for (PS = Bin->PropertySetsBegin; PS != Bin->PropertySetsEnd; ++PS) {
    assert(PS->Name && "nameless property set - bug in the offload wrapper?");
    if (!strcmp(PropSetName, PS->Name))
      break;
  }
  if (PS == Bin->PropertySetsEnd) {
    Begin = End = nullptr;
    return;
  }
  Begin = PS->PropertiesBegin;
  End = Begin ? PS->PropertiesEnd : nullptr;
}

pi_device_binary_property
DeviceBinaryImage::getProperty(const char *PropName) const {
  DeviceBinaryImage::PropertyRange BoolProp;
  BoolProp.init(Bin, __SYCL_PI_PROPERTY_SET_SYCL_MISC_PROP);
  if (!BoolProp.isAvailable())
    return nullptr;
  auto It = std::find_if(BoolProp.begin(), BoolProp.end(),
                         [=](pi_device_binary_property Prop) {
                           return !strcmp(PropName, Prop->Name);
                         });
  if (It == BoolProp.end())
    return nullptr;

  return *It;
}

RT::PiDeviceBinaryType getBinaryImageFormat(const unsigned char *ImgData,
                                            size_t ImgSize) {
  struct {
    RT::PiDeviceBinaryType Fmt;
    const uint32_t Magic;
  } Fmts[] = {{PI_DEVICE_BINARY_TYPE_SPIRV, 0x07230203},
              {PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE, 0xDEC04342}};

  if (ImgSize >= sizeof(Fmts[0].Magic)) {
    detail::remove_const_t<decltype(Fmts[0].Magic)> Hdr = 0;
    std::copy(ImgData, ImgData + sizeof(Hdr), reinterpret_cast<char *>(&Hdr));

    for (const auto &Fmt : Fmts) {
      if (Hdr == Fmt.Magic)
        return Fmt.Fmt;
    }
  }
  return PI_DEVICE_BINARY_TYPE_NONE;
}

void DeviceBinaryImage::init(pi_device_binary Bin) {
  this->Bin = Bin;
  // If device binary image format wasn't set by its producer, then can't change
  // now, because 'Bin' data is part of the executable image loaded into memory
  // which can't be modified (easily).
  // TODO clang driver + ClangOffloadWrapper can figure out the format and set
  // it when invoking the offload wrapper job
  Format = static_cast<pi::PiDeviceBinaryType>(Bin->Format);

  if (Format == PI_DEVICE_BINARY_TYPE_NONE)
    // try to determine the format; may remain "NONE"
    Format = getBinaryImageFormat(Bin->BinaryStart, getSize());

  SpecConstIDMap.init(Bin, __SYCL_PI_PROPERTY_SET_SPEC_CONST_MAP);
  DeviceLibReqMask.init(Bin, __SYCL_PI_PROPERTY_SET_DEVICELIB_REQ_MASK);
  KernelParamOptInfo.init(Bin, __SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO);
  ProgramMetadata.init(Bin, __SYCL_PI_PROPERTY_SET_PROGRAM_METADATA);
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
