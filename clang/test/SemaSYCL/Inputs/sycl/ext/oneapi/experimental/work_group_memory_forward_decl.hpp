#pragma once
// Dummy header file to avoid integration-header #include errors.
// It shadows the file 
// sycl/include/sycl/ext/oneapi/experimental/work_group_memory_forward_decl.hpp
namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
template <typename DataT, typename PropertiesT = int>
class work_group_memory;
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
