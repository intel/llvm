//==---------------- posix_pi.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/defines.hpp>
#include <sycl/detail/pi.hpp>

#include <dlfcn.h>
#include <string>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail::pi {

void *loadOsLibrary(const std::string &PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  void *so = dlopen(PluginPath.c_str(), RTLD_NOW);
  if (!so && trace(TraceLevel::PI_TRACE_ALL)) {
    char *Error = dlerror();
    std::cerr << "SYCL_PI_TRACE[-1]: dlopen(" << PluginPath << ") failed with <"
              << (Error ? Error : "unknown error") << ">" << std::endl;
  }
  return so;
}

int unloadOsLibrary(void *Library) {
  // The mock plugin does not have an associated library, so we allow nullptr
  // here to avoid it trying to free a non-existent library.
  if (!Library)
    return 0;
  return dlclose(Library);
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}

} // namespace detail::pi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
