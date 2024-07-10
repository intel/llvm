//==---------- ur.hpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#pragma once

#include <sycl/backend_types.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/ur_device_binary.h>
#include <ur_api.h>

#include <memory>
#include <type_traits>
#include <vector>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

namespace sycl {
inline namespace _V1 {

class context;

namespace detail {

class plugin;
using PluginPtr = std::shared_ptr<plugin>;

template <sycl::backend BE>
__SYCL_EXPORT void *getPluginOpaqueData(void *opaquedata_arg);

namespace ur {

// The SYCL_PI_TRACE sets what we will trace.
// This is a bit-mask of various things we'd want to trace.
enum TraceLevel {
  PI_TRACE_BASIC = 0x1,
  PI_TRACE_CALLS = 0x2,
  PI_TRACE_ALL = -1
};

// Return true if we want to trace UR related activities.
bool trace(TraceLevel level);

__SYCL_EXPORT void contextSetExtendedDeleter(const sycl::context &constext,
                                             ur_context_extended_deleter_t func,
                                             void *user_data);

// Function to load a shared library
// Implementation is OS dependent
void *loadOsLibrary(const std::string &Library);

// Function to unload a shared library
// Implementation is OS dependent (see posix-pi.cpp and windows-pi.cpp)
int unloadOsLibrary(void *Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// Performs UR one-time initialization.
std::vector<PluginPtr> &initializeUr();

// Get the plugin serving given backend.
template <backend BE> __SYCL_EXPORT const PluginPtr &getPlugin();

/// Tries to determine the device binary image foramat. Returns
/// PI_DEVICE_BINARY_TYPE_NONE if unsuccessful.
pi_device_binary_type getBinaryImageFormat(const unsigned char *ImgData,
                                           size_t ImgSize);

// Report error and no return (keeps compiler happy about no return statements).
[[noreturn]] __SYCL_EXPORT void die(const char *Message);

__SYCL_EXPORT void assertion(bool Condition, const char *Message = nullptr);

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> inline To cast(From value) {
  // TODO: see if more sanity checks are possible.
  assertion(sizeof(From) == sizeof(To), "assert: cast failed size check");
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
  for (FromE &Val : Values) {
    ResultVec.push_back(cast<typename To::value_type>(Val));
  }
  return ResultVec;
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
