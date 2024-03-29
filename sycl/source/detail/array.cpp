#include <sycl/detail/array.hpp>
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>                 // for invalid_parameter_error

namespace sycl {
inline namespace _V1 {
namespace detail {
  template <int dimensions>
__SYCL_ALWAYS_INLINE  void array<dimensions>::check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension >= dimensions || dimension < 0) {
      throw sycl::invalid_parameter_error("Index out of range",
                                          PI_ERROR_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }

  template class array<1>;
  template class array<2>;
  template class array<3>;
} // namespace detail
} // namespace _V1
} // namespace sycl
