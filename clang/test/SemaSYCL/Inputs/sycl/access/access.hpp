// Dummy header field needed by SemaSYCL tests.

#pragma once

namespace sycl {
inline namespace _V1 {
namespace access {

enum class target {
  global_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2014,
  constant_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2015,
  local __SYCL2020_DEPRECATED("use `local_accessor` instead") = 2016,
  image = 2017,
  host_buffer __SYCL2020_DEPRECATED("use 'host_accessor' instead") = 2018,
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
  constant_space __SYCL2020_DEPRECATED("sycl::access::address_space::constant_"
                                       "space is deprecated since SYCL 2020") =
      2,
  local_space = 3,
  ext_intel_global_device_space = 4,
  ext_intel_global_host_space = 5,
  generic_space = 6, // TODO generic_space address space is not supported yet
};

enum class decorated : int { no = 0, yes = 1, legacy = 2 };
} // namespace access
