// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -D_GLIBCXX_USE_CXX11_ABI=0 -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: level_zero && gpu

// This test case tests if compiling works with or without
// _GLIBCXX_USE_CXX11_ABI=0.

#include <sycl/sycl.hpp>

int main() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  std::cout << "is_cxx11_abi: " << (_GLIBCXX_USE_CXX11_ABI != 0) << std::endl;
  ;
#else
  std::cout << "is_cxx11_abi: 1" << std::endl;
#endif
  std::vector<sycl::device> root_devices;
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from GPU platform firstly.
  for (const auto &platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto &device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }
  assert(root_devices.size() > 0);
  std::cout << "gpu number: " << root_devices.size() << std::endl;
  constexpr sycl::info::partition_property partition_by_affinity =
      sycl::info::partition_property::partition_by_affinity_domain;
  constexpr sycl::info::partition_affinity_domain next_partitionable =
      sycl::info::partition_affinity_domain::next_partitionable;
  for (const auto &root_device : root_devices) {
    try {
      auto sub_devices = root_device.create_sub_devices<partition_by_affinity>(
          next_partitionable);
      std::cout << "tile partition is supported!" << std::endl;
    } catch (sycl::exception &e) {
      if (e.code() != sycl::errc::feature_not_supported &&
          e.code() != sycl::errc::invalid) {
        throw std::runtime_error(
            std::string("Failed to apply tile partition: ") + e.what());
      } else {
        std::cout << "tile partition is UNSUPPORTED!" << std::endl;
      }
    }
  }
  std::cout << "pass!" << std::endl;
}
