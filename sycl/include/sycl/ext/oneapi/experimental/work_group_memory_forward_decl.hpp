#pragma once
namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename> class properties;

template <typename DataT, typename PropertiesT = properties<std::tuple<>>>
class work_group_memory;
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
