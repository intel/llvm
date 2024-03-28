
#include <sycl/detail/property_list_base.hpp>
#include <sycl/detail/pi.h>                // for PI_ERROR_INVALID_VALUE
#include <sycl/exception.hpp>              // for invalid_object_error

namespace sycl {
inline namespace _V1 {
namespace detail {
  template <typename PropT>
  typename std::enable_if_t<std::is_base_of_v<PropertyWithDataBase, PropT>,
                            PropT>
  PropertyListBase::get_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    if (PropKind >= PropWithDataKind::PropWithDataKindSize)
      throw sycl::invalid_object_error("The property is not found",
                                       PI_ERROR_INVALID_VALUE);

    for (const std::shared_ptr<PropertyWithDataBase> &Prop : MPropsWithData)
      if (Prop->isSame(PropKind))
        return *static_cast<PropT *>(Prop.get());

    throw sycl::invalid_object_error("The property is not found",
                                     PI_ERROR_INVALID_VALUE);
  }
} // namespace detail
} // namespace _V1
} // namespace sycl
