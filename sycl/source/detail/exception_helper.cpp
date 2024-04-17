#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
__SYCL_EXPORT void throw_invalid_parameter(const char *Msg,
                                           const pi_int32 PIErr) {
  throw sycl::invalid_parameter_error(Msg, PIErr);
}
} // namespace detail
} // namespace _V1
} // namespace sycl
