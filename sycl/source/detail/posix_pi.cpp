//==---------------- posix_pi.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <dlfcn.h>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace pi {

void *loadOsLibrary(const std::string &PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  void *so = dlopen(PluginPath.c_str(), RTLD_NOW);
  if (!so && trace(TraceLevel::PI_TRACE_ALL)) {
    std::cerr << "SYCL_PI_TRACE[-1]: dlopen(" << PluginPath << ") failed with <"
              << dlerror() << ">" << std::endl;
  }
  return so;
}

int unloadOsLibrary(void *Library) { return dlclose(Library); }

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
