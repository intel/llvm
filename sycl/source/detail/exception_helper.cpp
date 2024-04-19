#include <sycl/errc.hpp>
#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
__SYCL_EXPORT void throw_exception(errc Ec, const char *Msg) {
  throw sycl::exception(make_error_code(Ec), Msg);
}
} // namespace detail
} // namespace _V1
} // namespace sycl
