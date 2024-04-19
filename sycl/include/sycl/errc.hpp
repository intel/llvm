#pragma once

namespace sycl {
inline namespace _V1 {

enum class errc : unsigned int {
  success = 0,
  runtime = 1,
  kernel = 2,
  accessor = 3,
  nd_range = 4,
  event = 5,
  kernel_argument = 6,
  build = 7,
  invalid = 8,
  memory_allocation = 9,
  platform = 10,
  profiling = 11,
  feature_not_supported = 12,
  kernel_not_supported = 13,
  backend_mismatch = 14,
};

} // namespace _V1
} // namespace sycl
