//==---------------- posix_pi.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/pi.hpp>

#include <dlfcn.h>
#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail::pi {

void *loadOsLibrary(const std::string &LibraryPath) {
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  void *so = dlopen(LibraryPath.c_str(), RTLD_NOW);
  if (!so && trace(TraceLevel::PI_TRACE_ALL)) {
    char *Error = dlerror();
    std::cerr << "SYCL_PI_TRACE[-1]: dlopen(" << LibraryPath
              << ") failed with <" << (Error ? Error : "unknown error") << ">"
              << std::endl;
  }
  return so;
}

int unloadOsLibrary(void *Library) { return dlclose(Library); }

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}

} // namespace detail::pi
} // namespace _V1
} // namespace sycl
