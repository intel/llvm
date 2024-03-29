
#include <sycl/property_list.hpp>
#include <sycl/exception.hpp>                  // for invalid_object_error

namespace sycl {
inline namespace _V1 {
  template <typename PropT> PropT property_list::get_property() const {
    if (!has_property<PropT>())
      throw sycl::invalid_object_error("The property is not found",
                                       PI_ERROR_INVALID_VALUE);

    return get_property_helper<PropT>();
  }
} // namespace _V1
} // namespace sycl
