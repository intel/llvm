// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=":" %{run-unfiltered-devices} %t.out
#include <sycl/detail/core.hpp>
#include <vector>

// Check that the application does not hang when we attempt
// to initialize plugins multiple times with invalid values
// of ONEAPI_DEVICE_SELECTOR.
int main() {
  for (int I = 0; I < 3; ++I) {
    try {
      std::vector<sycl::platform> pl = sycl::platform::get_platforms();
    } catch (std::exception const &e) {
    }
  }

  return 0;
}
