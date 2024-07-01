// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <detail/plugin.hpp>
#include <string>

namespace pi {
inline std::string GetBackendString(const sycl::detail::PluginPtr &Plugin) {
  std::stringstream Str;
  for (sycl::backend Backend :
       {sycl::backend::opencl, sycl::backend::ext_oneapi_level_zero,
        sycl::backend::ext_oneapi_cuda, sycl::backend::ext_oneapi_hip}) {
    if (Plugin->hasBackend(Backend)) {
      Str << Backend;
    }
  }
  return Str.str();
}
} // namespace pi
