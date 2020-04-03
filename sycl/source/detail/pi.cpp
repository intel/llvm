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
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>

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
#include "xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
// Stream name being used for traces generated from the SYCL runtime
constexpr const char *PICALL_STREAM_NAME = "sycl.pi";
// Global (to the SYCL runtime) graph handle that all command groups are a
// child of
///< Event to be used by graph related activities
xpti_td *GSYCLGraphEvent = nullptr;
///< Event to be used by PI layer related activities
xpti_td *GPICallEvent = nullptr;
///< Constansts being used as placeholder until one is able to reliably get the
///< version of the SYCL runtime
constexpr uint32_t GMajVer = 1;
constexpr uint32_t GMinVer = 0;
constexpr const char *GVerStr = "sycl 1.0";
#endif

namespace pi {

bool XPTIInitDone = false;

void contextSetExtendedDeleter(const cl::sycl::context &context,
                               pi_context_extended_deleter func,
                               void *user_data) {
  auto impl = getSyclObjImpl(context);
  auto contextHandle = reinterpret_cast<pi_context>(impl->getHandleRef());
  auto plugin = impl->getPlugin();
  plugin.call_nocheck<PiApiKind::piextContextSetExtendedDeleter>(
      contextHandle, func, user_data);
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
  default:
    die("Unknown pi_platform_info value passed to "
        "cl::sycl::detail::pi::platformInfoToString");
  }
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

// A singleton class to aid that PI configuration parameters
// are processed only once, like reading a string from environment
// and converting it into a typed object.
//
template <typename T, const char *E> class Config {
  static Config *m_Instance;
  T m_Data;
  Config();

public:
  static T get() {
    if (!m_Instance) {
      m_Instance = new Config();
    }
    return m_Instance->m_Data;
  }
};

template <typename T, const char *E>
Config<T, E> *Config<T, E>::m_Instance = nullptr;

// Lists valid configuration environment variables.
static constexpr char SYCL_BE[] = "SYCL_BE";
static constexpr char SYCL_INTEROP_BE[] = "SYCL_INTEROP_BE";
static constexpr char SYCL_PI_TRACE[] = "SYCL_PI_TRACE";

// SYCL_PI_TRACE gives the mask of enabled tracing components (0 default)
template <> Config<int, SYCL_PI_TRACE>::Config() {
  const char *Env = std::getenv(SYCL_PI_TRACE);
  m_Data = (Env ? std::atoi(Env) : 0);
}

static Backend getBE(const char *EnvVar) {
  const char *BE = std::getenv(EnvVar);
  const std::map<std::string, Backend> SyclBeMap{
      {"PI_OTHER", SYCL_BE_PI_OTHER},
      {"PI_CUDA", SYCL_BE_PI_CUDA},
      {"PI_OPENCL", SYCL_BE_PI_OPENCL}};
  if (BE) {
    auto It = SyclBeMap.find(BE);
    if (It == SyclBeMap.end())
      pi::die("Invalid backend. "
              "Valid values are PI_OPENCL/PI_CUDA");
    return It->second;
  }
  // Default backend
  return SYCL_BE_PI_OPENCL;
}

template <> Config<Backend, SYCL_BE>::Config() { m_Data = getBE(SYCL_BE); }

// SYCL_INTEROP_BE is a way to specify the interoperability plugin.
template <> Config<Backend, SYCL_INTEROP_BE>::Config() {
  m_Data = getBE(SYCL_INTEROP_BE);
}

// Helper interface to not expose "pi::Config" outside of pi.cpp
Backend getPreferredBE() { return Config<Backend, SYCL_BE>::get(); }

// GlobalPlugin is a global Plugin used with Interoperability constructors that
// use OpenCL objects to construct SYCL class objects.
std::shared_ptr<plugin> GlobalPlugin;

// Find the plugin at the appropriate location and return the location.
bool findPlugins(vector_class<std::pair<std::string, Backend>> &PluginNames) {
  // TODO: Based on final design discussions, change the location where the
  // plugin must be searched; how to identify the plugins etc. Currently the
  // search is done for libpi_opencl.so/pi_opencl.dll file in LD_LIBRARY_PATH
  // env only.
  //
  PluginNames.push_back(std::make_pair<std::string, Backend>(
      OPENCL_PLUGIN_NAME, SYCL_BE_PI_OPENCL));
  PluginNames.push_back(
      std::make_pair<std::string, Backend>(CUDA_PLUGIN_NAME, SYCL_BE_PI_CUDA));
  return true;
}

// Load the Plugin by calling the OS dependent library loading call.
// Return the handle to the Library.
void *loadPlugin(const std::string &PluginPath) {
  return loadOsLibrary(PluginPath);
}

// Binds all the PI Interface APIs to Plugin Library Function Addresses.
// TODO: Remove the 'OclPtr' extension to PI_API.
// TODO: Change the functionality such that a single getOsLibraryFuncAddress
// call is done to get all Interface API mapping. The plugin interface also
// needs to setup infrastructure to route PI_CALLs to the appropriate plugins.
// Currently, we bind to a singe plugin.
bool bindPlugin(void *Library, PiPlugin *PluginInformation) {

  decltype(::piPluginInit) *PluginInitializeFunction = (decltype(
      &::piPluginInit))(getOsLibraryFuncAddress(Library, "piPluginInit"));
  if (PluginInitializeFunction == nullptr)
    return false;

  int Err = PluginInitializeFunction(PluginInformation);

  // TODO: Compare Supported versions and check for backward compatibility.
  // Make sure err is PI_SUCCESS.
  assert((Err == PI_SUCCESS) && "Unexpected error when binding to Plugin.");
  (void)Err;

  // TODO: Return a more meaningful value/enum.
  return true;
}

bool trace(TraceLevel Level) {
  auto TraceLevelMask = Config<int, SYCL_PI_TRACE>::get();
  return (TraceLevelMask & Level) == Level;
}

// Initializes all available Plugins.
vector_class<plugin> initialize() {
  vector_class<plugin> Plugins;
  vector_class<std::pair<std::string, Backend>> PluginNames;
  findPlugins(PluginNames);

  if (PluginNames.empty() && trace(PI_TRACE_ALL))
    std::cerr << "SYCL_PI_TRACE[-1]: No Plugins Found." << std::endl;

  PiPlugin PluginInformation;
  for (unsigned int I = 0; I < PluginNames.size(); I++) {
    void *Library = loadPlugin(PluginNames[I].first);

    if (!Library) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[-1]: Check if plugin is present. "
                  << "Failed to load plugin: " << PluginNames[I].first
                  << std::endl;
      }
      continue;
    }

    if (!bindPlugin(Library, &PluginInformation)) {
      if (trace(PI_TRACE_ALL)) {
        std::cerr << "SYCL_PI_TRACE[-1]: Failed to bind PI APIs to the plugin: "
                  << PluginNames[I].first << std::endl;
      }
      continue;
    }
    // Set the Global Plugin based on SYCL_INTEROP_BE.
    // Rework this when it will be explicit in the code which BE is used in the
    // interoperability methods.
    if (Config<Backend, SYCL_INTEROP_BE>::get() == PluginNames[I].second) {
      GlobalPlugin =
          std::make_shared<plugin>(PluginInformation, PluginNames[I].second);
    }
    Plugins.emplace_back(plugin(PluginInformation, PluginNames[I].second));
    if (trace(TraceLevel::PI_TRACE_BASIC))
      std::cerr << "SYCL_PI_TRACE[1]: Plugin found and successfully loaded: "
                << PluginNames[I].first << std::endl;
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && !XPTIInitDone))
    return Plugins;
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
  //
  //                                           +--- Minor version #
  //            Major version # ------+        |   Version string
  //                                  |        |       |
  //                                  v        v       v
  xptiInitialize(SYCL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
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

  xpti::payload_t PIPayload("Plugin Interface Layer");
  uint64_t PiInstanceNo;
  GPICallEvent =
      xptiMakeEvent("PI Layer", &PIPayload, xpti::trace_algorithm_event,
                    xpti_at::active, &PiInstanceNo);
#endif

  return Plugins;
}

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
  case PI_PROPERTY_TYPE_STRING:
    Out << "[String] ";
    break;
  default:
    assert("unsupported property");
    return Out;
  }
  Out << P.Prop->Name << "=";

  switch (P.Prop->Type) {
  case PI_PROPERTY_TYPE_UINT32:
    Out << P.asUint32();
    break;
  case PI_PROPERTY_TYPE_STRING:
    Out << P.asCString();
    break;
  default:
    assert("unsupported property");
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

RT::PiDeviceBinaryType getBinaryImageFormat(const unsigned char *ImgData,
                                            size_t ImgSize) {
  struct {
    RT::PiDeviceBinaryType Fmt;
    const uint32_t Magic;
  } Fmts[] = {{PI_DEVICE_BINARY_TYPE_SPIRV, 0x07230203},
              {PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE, 0xDEC04342}};

  if (ImgSize >= sizeof(Fmts[0].Magic)) {
    std::remove_const<decltype(Fmts[0].Magic)>::type Hdr = 0;
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

  SpecConstIDMap.init(Bin, PI_PROPERTY_SET_SPEC_CONST_MAP);
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
