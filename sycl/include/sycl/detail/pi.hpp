//==---------- pi.hpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.hpp
/// C++ wrapper of extern "C" PI interfaces
///
/// \ingroup sycl_pi

#pragma once

#include <sycl/backend_types.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class context;

namespace detail {

enum class PiApiKind {
#define _PI_API(api) api,
#include <sycl/detail/pi.def>
};
class plugin;

template <sycl::backend BE>
__SYCL_EXPORT void *getPluginOpaqueData(void *opaquedata_arg);

namespace pi {

// The SYCL_PI_TRACE sets what we will trace.
// This is a bit-mask of various things we'd want to trace.
enum TraceLevel {
  PI_TRACE_BASIC = 0x1,
  PI_TRACE_CALLS = 0x2,
  PI_TRACE_ALL = -1
};

// Return true if we want to trace PI related activities.
bool trace(TraceLevel level);

#ifdef __SYCL_RT_OS_WINDOWS
#define __SYCL_OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "pi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "pi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "pi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll"
#elif defined(__SYCL_RT_OS_LINUX)
#define __SYCL_OPENCL_PLUGIN_NAME "libpi_opencl.so"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "libpi_level_zero.so"
#define __SYCL_CUDA_PLUGIN_NAME "libpi_cuda.so"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "libpi_esimd_emulator.so"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.so"
#elif defined(__SYCL_RT_OS_DARWIN)
#define __SYCL_OPENCL_PLUGIN_NAME "libpi_opencl.dylib"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "libpi_level_zero.dylib"
#define __SYCL_CUDA_PLUGIN_NAME "libpi_cuda.dylib"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "libpi_esimd_emulator.dylib"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dylib"
#else
#error "Unsupported OS"
#endif

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] __SYCL_EXPORT void die(const char *Message);

__SYCL_EXPORT void assertion(bool Condition, const char *Message = nullptr);

template <typename T>
void handleUnknownParamName(const char *functionName, T parameter) {
  std::stringstream stream;
  stream << "Unknown parameter " << parameter << " passed to " << functionName
         << "\n";
  auto str = stream.str();
  auto msg = str.c_str();
  die(msg);
}

// This macro is used to report invalid enumerators being passed to PI API
// GetInfo functions. It will print the name of the function that invoked it
// and the value of the unknown enumerator.
#define __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(parameter)                         \
  { sycl::detail::pi::handleUnknownParamName(__func__, parameter); }

using PiPlugin = ::pi_plugin;
using PiResult = ::pi_result;
using PiPlatform = ::pi_platform;
using PiDevice = ::pi_device;
using PiDeviceType = ::pi_device_type;
using PiDeviceInfo = ::pi_device_info;
using PiDeviceBinaryType = ::pi_device_binary_type;
using PiContext = ::pi_context;
using PiProgram = ::pi_program;
using PiKernel = ::pi_kernel;
using PiQueue = ::pi_queue;
using PiQueueProperties = ::pi_queue_properties;
using PiMem = ::pi_mem;
using PiMemFlags = ::pi_mem_flags;
using PiEvent = ::pi_event;
using PiSampler = ::pi_sampler;
using PiSamplerInfo = ::pi_sampler_info;
using PiSamplerProperties = ::pi_sampler_properties;
using PiSamplerAddressingMode = ::pi_sampler_addressing_mode;
using PiSamplerFilterMode = ::pi_sampler_filter_mode;
using PiMemImageFormat = ::pi_image_format;
using PiMemImageDesc = ::pi_image_desc;
using PiMemImageInfo = ::pi_image_info;
using PiMemObjectType = ::pi_mem_type;
using PiMemImageChannelOrder = ::pi_image_channel_order;
using PiMemImageChannelType = ::pi_image_channel_type;

__SYCL_EXPORT void contextSetExtendedDeleter(const sycl::context &constext,
                                             pi_context_extended_deleter func,
                                             void *user_data);

// Function to load the shared library
// Implementation is OS dependent.
void *loadOsLibrary(const std::string &Library);

// Function to unload the shared library
// Implementation is OS dependent (see posix-pi.cpp and windows-pi.cpp)
int unloadOsLibrary(void *Library);

// OS agnostic function to unload the shared library
int unloadPlugin(void *Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// Get a string representing a _pi_platform_info enum
std::string platformInfoToString(pi_platform_info info);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Holds the PluginInformation for the plugin that is bound.
// Currently a global variable is used to store OpenCL plugin information to be
// used with SYCL Interoperability Constructors.
extern std::shared_ptr<plugin> GlobalPlugin;

// Performs PI one-time initialization.
std::vector<plugin> &initialize();

// Get the plugin serving given backend.
template <backend BE> __SYCL_EXPORT const plugin &getPlugin();

// Utility Functions to get Function Name for a PI Api.
template <PiApiKind PiApiOffset> struct PiFuncInfo {};

#define _PI_API(api)                                                           \
  template <> struct PiFuncInfo<PiApiKind::api> {                              \
    using FuncPtrT = decltype(&::api);                                         \
    inline const char *getFuncName() { return #api; }                          \
    inline FuncPtrT getFuncPtr(PiPlugin MPlugin) {                             \
      return MPlugin.PiFunctionTable.api;                                      \
    }                                                                          \
  };
#include <sycl/detail/pi.def>

/// Emits an XPTI trace before a PI API call is made
/// \param FName The name of the PI API call
/// \return The correlation ID for the API call that is to be used by the
/// emitFunctionEndTrace() call
uint64_t emitFunctionBeginTrace(const char *FName);

/// Emits an XPTI trace after the PI API call has been made
/// \param CorrelationID The correlation ID for the API call generated by the
/// emitFunctionBeginTrace() call.
/// \param FName The name of the PI API call
void emitFunctionEndTrace(uint64_t CorrelationID, const char *FName);

/// Notifies XPTI subscribers about PI function calls and packs call arguments.
///
/// \param FuncID is the API hash ID from PiApiID type trait.
/// \param FName The name of the PI API call.
/// \param ArgsData is a pointer to packed function call arguments.
/// \param Plugin is the plugin, which is used to make call.
uint64_t emitFunctionWithArgsBeginTrace(uint32_t FuncID, const char *FName,
                                        unsigned char *ArgsData,
                                        pi_plugin Plugin);

/// Notifies XPTI subscribers about PI function call result.
///
/// \param CorrelationID The correlation ID for the API call generated by the
/// emitFunctionWithArgsBeginTrace() call.
/// \param FuncID is the API hash ID from PiApiID type trait.
/// \param FName The name of the PI API call.
/// \param ArgsData is a pointer to packed function call arguments.
/// \param Result is function call result value.
/// \param Plugin is the plugin, which is used to make call.
void emitFunctionWithArgsEndTrace(uint64_t CorrelationID, uint32_t FuncID,
                                  const char *FName, unsigned char *ArgsData,
                                  pi_result Result, pi_plugin Plugin);

/// Tries to determine the device binary image foramat. Returns
/// PI_DEVICE_BINARY_TYPE_NONE if unsuccessful.
PiDeviceBinaryType getBinaryImageFormat(const unsigned char *ImgData,
                                        size_t ImgSize);

} // namespace pi

namespace RT = sycl::detail::pi;

// Workaround for build with GCC 5.x
// An explicit specialization shall be declared in the namespace block.
// Having namespace as part of template name is not supported by GCC
// older than 7.x.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=56480
namespace pi {
// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> inline To cast(From value) {
  // TODO: see if more sanity checks are possible.
  RT::assertion((sizeof(From) == sizeof(To)), "assert: cast failed size check");
  return (To)(value);
}

// Helper traits for identifying std::vector with arbitrary element type.
template <typename T> struct IsStdVector : std::false_type {};
template <typename T> struct IsStdVector<std::vector<T>> : std::true_type {};

// Overload for vectors that applies the cast to all elements. This
// creates a new vector.
template <class To, class FromE> To cast(std::vector<FromE> Values) {
  static_assert(IsStdVector<To>::value, "Return type must be a vector.");
  To ResultVec;
  ResultVec.reserve(Values.size());
  for (FromE &Val : Values)
    ResultVec.push_back(cast<typename To::value_type>(Val));
  return ResultVec;
}

} // namespace pi
} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = sycl::detail::pi;

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef _PI_API
