// RUN: %{build} -o %t.out
// RUN: %if any-device-is-cpu %{ env SYCL_DEVICE_FILTER=cpu %{run-unfiltered-devices} %t.out %}
// RUN: %if any-device-is-gpu %{ env SYCL_DEVICE_FILTER=gpu %{run-unfiltered-devices} %t.out %}
// RUN: %if any-device-is-acc %{ env SYCL_DEVICE_FILTER=acc %{run-unfiltered-devices} %t.out %}
// TODO: Remove this test once SYCL_DEVICE_FILTER is removed.

#include <sycl/sycl.hpp>

int main() {
  namespace dev_info = sycl::info::device;

  auto device_type = [](sycl::device d) -> std::string_view {
    switch (d.get_info<dev_info::device_type>()) {
    case sycl::info::device_type::cpu:
      return "cpu";
    case sycl::info::device_type::gpu:
      return "gpu";
    case sycl::info::device_type::accelerator:
      return "acc";
    default:
      return "unknown";
    }
  };

  auto devices = sycl::device::get_devices();
  for (sycl::device d : devices) {
    std::cout << device_type(d) << " " << d.get_info<dev_info::name>()
              << std::endl;
  }
  assert(!devices.empty());
  auto expected_type = std::getenv("SYCL_DEVICE_FILTER");
  assert(std::all_of(devices.begin(), devices.end(),
                     [=](auto d) { return expected_type == device_type(d); }));
  return 0;
}
