#include <string_view>
#include <sycl/backend_types.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

inline std::string_view get_backend_name_no_vendor(backend Backend) {
  switch (Backend) {
  case backend::host:
    return "host";
  case backend::opencl:
    return "opencl";
  case backend::ext_oneapi_level_zero:
    return "level_zero";
  case backend::ext_oneapi_cuda:
    return "cuda";
  case backend::ext_oneapi_hip:
    return "hip";
  case backend::ext_oneapi_native_cpu:
    return "native_cpu";
  case backend::ext_oneapi_offload:
    return "offload";
  case backend::all:
    return "all";
  }

  return "";
}

} // namespace detail
} // namespace _V1
} // namespace sycl