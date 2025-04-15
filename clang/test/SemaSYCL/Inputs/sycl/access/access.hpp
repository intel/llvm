// Dummy header field needed by SemaSYCL tests.

#pragma once

namespace sycl {
inline namespace _V1 {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer = 2015,
  locaL = 2016,
  image = 2017,
  host_buffer = 2018,
  host_image = 2019,
  image_array = 2020,
  host_task = 2021,
  device = global_buffer,
};

enum class mode {
  read = 1024,
  write = 1025,
  read_write = 1026,
  discard_write = 1027,
  discard_read_write = 1028,
  atomic = 1029
};

enum class fence_space {
  local_space = 0,
  global_space = 1,
  global_and_local = 2
};

enum class placeholder { false_t = 0, true_t = 1 };

enum class address_space : int {
  private_space = 0,
  global_space = 1,
  constant_space = 2,
  local_space = 3,
  ext_intel_global_device_space = 4,
  ext_intel_global_host_space = 5,
  generic_space = 6,
};

enum class decorated : int { no = 0, yes = 1, legacy = 2 };
} // namespace access
} // namespace _V1
} // namespace sycl
