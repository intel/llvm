// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <detail/plugin.hpp>

namespace pi {
inline const char *GetBackendString(cl::sycl::backend backend) {
  switch (backend) {
#define PI_BACKEND_STR(backend_name)                                           \
  case cl::sycl::backend::backend_name:                                        \
    return #backend_name
    PI_BACKEND_STR(rocm);
    PI_BACKEND_STR(cuda);
    PI_BACKEND_STR(host);
    PI_BACKEND_STR(opencl);
    PI_BACKEND_STR(level_zero);
#undef PI_BACKEND_STR
  default:
    return "Unknown Plugin";
  }
}
} // namespace pi
