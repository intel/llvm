//==---------- pi.hpp - Plugin Interface for SYCL RT -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi.hpp
/// C++ wrapper of extern "C" UR interfaces
///
/// \ingroup sycl_pi

#pragma once

#include <ur_api.h>

#include <sycl/backend_types.hpp>  // for backend
#include <sycl/detail/export.hpp>  // for __SYCL_EXPORT
#include <sycl/detail/os_util.hpp> // for __SYCL_RT_OS_LINUX
#include <sycl/detail/pi.h>        // for pi binary stuff
                                   //
#include <memory>                  // for shared_ptr
#include <stddef.h>                // for size_t
#include <string>                  // for char_traits, string
#include <vector>                  // for vector

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

namespace pi {

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

} // namespace pi
} // namespace detail
} // namespace _V1
} // namespace sycl

#undef _PI_API
