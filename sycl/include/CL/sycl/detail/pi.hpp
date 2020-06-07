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

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.h>

#include <cassert>
#include <sstream>
#include <string>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class context;

namespace detail {

enum class PiApiKind {
#define _PI_API(api) api,
#include <CL/sycl/detail/pi.def>
};
class plugin;
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

#ifdef SYCL_RT_OS_WINDOWS
#define OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define CUDA_PLUGIN_NAME "pi_cuda.dll"
#else
#define OPENCL_PLUGIN_NAME "libpi_opencl.so"
#define CUDA_PLUGIN_NAME "libpi_cuda.so"
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
#define PI_HANDLE_UNKNOWN_PARAM_NAME(parameter)                                \
  { cl::sycl::detail::pi::handleUnknownParamName(__func__, parameter); }

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

__SYCL_EXPORT void contextSetExtendedDeleter(const cl::sycl::context &constext,
                                             pi_context_extended_deleter func,
                                             void *user_data);

// Function to load the shared library
// Implementation is OS dependent.
void *loadOsLibrary(const std::string &Library);

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
vector_class<plugin> initialize();

// Utility Functions to get Function Name for a PI Api.
template <PiApiKind PiApiOffset> struct PiFuncInfo {};

#define _PI_API(api)                                                           \
  template <> struct PiFuncInfo<PiApiKind::api> {                              \
    using FuncPtrT = decltype(&::api);                                         \
    inline std::string getFuncName() { return #api; }                          \
    inline FuncPtrT getFuncPtr(PiPlugin MPlugin) {                             \
      return MPlugin.PiFunctionTable.api;                                      \
    }                                                                          \
  };
#include <CL/sycl/detail/pi.def>

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

// Helper utilities for PI Tracing
// The run-time tracing of PI calls.
// Print functions used by Trace class.
template <typename T> inline void print(T val) {
  std::cout << "<unknown> : " << val << std::endl;
}

template <> inline void print<>(PiPlatform val) {
  std::cout << "pi_platform : " << val << std::endl;
}

template <> inline void print<>(PiResult val) {
  std::cout << "pi_result : ";
  if (val == PI_SUCCESS)
    std::cout << "PI_SUCCESS" << std::endl;
  else
    std::cout << val << std::endl;
}

// cout does not resolve a nullptr.
template <> inline void print<>(std::nullptr_t val) { print<void *>(val); }

inline void printArgs(void) {}
template <typename Arg0, typename... Args>
void printArgs(Arg0 arg0, Args... args) {
  std::cout << "       ";
  print(arg0);
  pi::printArgs(std::forward<Args>(args)...);
}

// C++ wrapper over the _pi_device_binary_property_struct structure.
class DeviceBinaryProperty {
public:
  DeviceBinaryProperty(const _pi_device_binary_property_struct *Prop)
      : Prop(Prop) {}

  pi_uint32 asUint32() const;
  const char *asCString() const;

protected:
  friend std::ostream &operator<<(std::ostream &Out,
                                  const DeviceBinaryProperty &P);
  const _pi_device_binary_property_struct *Prop;
};

std::ostream &operator<<(std::ostream &Out, const DeviceBinaryProperty &P);

// C++ convenience wrapper over the pi_device_binary_struct structure.
class DeviceBinaryImage {
public:
  // Represents a range of properties to enable iteration over them.
  // Implements the standard C++ STL input iterator interface.
  class PropertyRange {
  public:
    using ValTy = std::remove_pointer<pi_device_binary_property>::type;

    class ConstIterator
        : public std::iterator<std::input_iterator_tag, // iterator_category
                               ValTy,                   // value_type
                               ptrdiff_t,               // difference_type
                               const pi_device_binary_property, // pointer
                               pi_device_binary_property>       // reference
    {
      pi_device_binary_property Cur;

    public:
      ConstIterator(pi_device_binary_property Cur = nullptr) : Cur(Cur) {}
      ConstIterator &operator++() {
        Cur++;
        return *this;
      }
      ConstIterator operator++(int) {
        ConstIterator Ret = *this;
        ++(*this);
        return Ret;
      }
      bool operator==(ConstIterator Other) const { return Cur == Other.Cur; }
      bool operator!=(ConstIterator Other) const { return !(*this == Other); }
      reference operator*() const { return Cur; }
    };
    ConstIterator begin() const { return ConstIterator(Begin); }
    ConstIterator end() const { return ConstIterator(End); }
    friend class DeviceBinaryImage;

  private:
    PropertyRange() : Begin(nullptr), End(nullptr) {}
    // Searches for a property set with given name and constructs a
    // PropertyRange spanning all its elements. If property set is not found,
    // the range will span zero elements.
    PropertyRange(pi_device_binary Bin, const char *PropSetName)
        : PropertyRange() {
      init(Bin, PropSetName);
    };
    void init(pi_device_binary Bin, const char *PropSetName);
    pi_device_binary_property Begin;
    pi_device_binary_property End;
  };

public:
  DeviceBinaryImage(pi_device_binary Bin) { init(Bin); }
  DeviceBinaryImage() : Bin(nullptr){};

  virtual void print() const;
  virtual void dump(std::ostream &Out) const;

  size_t getSize() const {
    assert(Bin && "binary image data not set");
    return static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart);
  }

  const char *getCompileOptions() const {
    assert(Bin && "binary image data not set");
    return Bin->CompileOptions;
  }

  const char *getLinkOptions() const {
    assert(Bin && "binary image data not set");
    return Bin->LinkOptions;
  }

  /// Returns the format of the binary image
  pi::PiDeviceBinaryType getFormat() const {
    assert(Bin && "binary image data not set");
    return Format;
  }

  /// Gets the iterator range over specialization constants in this this binary
  /// image. For each property pointed to by an iterator within the range, the
  /// name of the property is the specializaion constant symbolic ID and the
  /// value is 32-bit unsigned integer ID.
  const PropertyRange &getSpecConstants() const { return SpecConstIDMap; }
  virtual ~DeviceBinaryImage() {}

protected:
  void init(pi_device_binary Bin);
  pi_device_binary get() const { return Bin; }

  pi_device_binary Bin;
  pi::PiDeviceBinaryType Format = PI_DEVICE_BINARY_TYPE_NONE;
  DeviceBinaryImage::PropertyRange SpecConstIDMap;
};

/// Tries to determine the device binary image foramat. Returns
/// PI_DEVICE_BINARY_TYPE_NONE if unsuccessful.
PiDeviceBinaryType getBinaryImageFormat(const unsigned char *ImgData,
                                        size_t ImgSize);

} // namespace pi

namespace RT = cl::sycl::detail::pi;

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

// These conversions should use PI interop API.
template <> inline pi::PiProgram cast(cl_program) {
  RT::assertion(false, "pi::cast -> use piextProgramFromNative");
  return {};
}

template <> inline pi::PiDevice cast(cl_device_id) {
  RT::assertion(false, "pi::cast -> use piextDeviceFromNative");
  return {};
}
} // namespace pi
} // namespace detail

// For shortness of using PI from the top-level sycl files.
namespace RT = cl::sycl::detail::pi;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
