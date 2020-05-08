#pragma once

#include <detail/plugin.hpp>

using namespace cl::sycl;

namespace pi {
inline const char *GetBackendString(backend backend) {
  switch (backend) {
#define PI_BACKEND_STR(backend_name)                                           \
  case cl::sycl::backend::backend_name:                                        \
    return #backend_name
    PI_BACKEND_STR(cuda);
    PI_BACKEND_STR(host);
    PI_BACKEND_STR(opencl);
#undef PI_BACKEND_STR
  default:
    return "Unknown Plugin";
  }
}
} // namespace pi