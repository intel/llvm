#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
__SYCL_EXPORT void throw_invalid_parameter(const char *Msg) {
  throw sycl::exception(make_error_code(errc::kernel_argument), Msg);
}
} // namespace detail
} // namespace _V1
} // namespace sycl
