#pragma once
#include <sycl/ext/oneapi/properties/properties.hpp>
namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
template <typename DataT, typename PropertiesT = empty_properties_t>
class work_group_memory;
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
