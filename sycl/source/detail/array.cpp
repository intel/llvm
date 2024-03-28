#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>                 // for invalid_parameter_error
#include <sycl/detail/array.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
  template <int dimensions>
  void array<dimensions>::check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension >= dimensions || dimension < 0) {
      throw sycl::invalid_parameter_error("Index out of range",
                                          PI_ERROR_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }
} // namespace detail
} // namespace _V1
} // namespace sycl
